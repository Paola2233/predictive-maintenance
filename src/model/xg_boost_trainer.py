"""
train.py — XGBoost trainer that reads features directly from BigQuery
and saves model artifacts natively to GCS + Vertex AI Model Registry.

No local files, no CSV middleman. BigQuery is the single source of truth.

Flow
────
  BigQuery (features_engineered_v2)
      ↓  _load_data_from_bigquery()
  pd.DataFrame (in memory)
      ↓  _prepare_features()
  X, y (in memory)
      ↓  TimeSeriesSplit CV + final fit
  XGBClassifier + StandardScaler
      ↓  _save_artifacts_to_gcs()
  gs://<bucket>/models/<name>/<timestamp>/
      ↓  _register_model()
  Vertex AI Model Registry

GCS layout written by this script
───────────────────────────────────
gs://<bucket>/models/<model_name>/<timestamp>/
    model.bst           ← XGBoost native binary
    scaler.pkl          ← fitted StandardScaler
    feature_names.json  ← ordered feature column list
"""

import json
import logging
import os
import pickle
import tempfile
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from google.cloud import bigquery, storage
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import google.cloud.aiplatform as aiplatform

# ---------------------------------------------------------------------------
# Configuration — all overridable via env vars for container deployments
# ---------------------------------------------------------------------------

PROJECT_ID  = os.getenv("GCP_PROJECT_ID")
REGION      = os.getenv("GCP_REGION", "us-central1")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")       # no gs:// prefix
MODEL_NAME  = os.getenv("MODEL_NAME", "XGBoostPredictiveMaintenance")
EXPERIMENT  = os.getenv("VERTEX_EXPERIMENT", "predictive-maintenance-xgboost")
BQ_DATASET  = os.getenv("BQ_DATASET", "preprocessed_data")
BQ_TABLE    = os.getenv("BQ_TABLE", "features_engineered_v2")
# BG_TABLE_URI  = os.getenv("BQ_DATASET", f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}")

# Columns excluded to prevent data leakage
# 'date'      → encodes temporal position; leaks time-ordering into features
# 'machineID' → row identifier; model must generalise across machines
LEAKAGE_COLS = ["date", "machineID"]
TARGET_COL   = "will_fail_30_days"


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class XGBoostTrainer:
    """
    Trains an XGBoost binary classifier for predictive maintenance.

    Design decisions
    ────────────────
    • Reads features directly from BigQuery — no local files, no CSV.
    • TimeSeriesSplit instead of random split → no future data leaks into folds.
    • 'date' and 'machineID' dropped explicitly → no identifier leakage.
    • scale_pos_weight auto-computed → handles heavy class imbalance.
    • Model artifacts saved to GCS; registered in Vertex AI Model Registry.
    • Metrics logged to Vertex AI Experiments (visible in Cloud Console).
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        random_state: int = 42,
        n_cv_splits: int = 5,
        bq_dataset: str = BQ_DATASET,
        bq_table: str = BQ_TABLE,
    ):
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.learning_rate    = learning_rate
        self.subsample        = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.random_state     = random_state
        self.n_cv_splits      = n_cv_splits
        self.bq_dataset       = bq_dataset
        self.bq_table         = bq_table

        self._validate_env()
        self._setup_logging()
        self._init_vertex()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _validate_env(self):
        missing = [v for v in ("GCP_PROJECT_ID", "GCS_BUCKET_NAME") if not os.getenv(v)]
        if missing:
            raise EnvironmentError(f"Missing required env vars: {missing}")

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _init_vertex(self):
        """
        Initialise Vertex AI SDK and create (or reuse) an Experiment.
        An Experiment groups all runs and their metrics in the Cloud Console UI
        under Vertex AI → Experiments.
        """
        aiplatform.init(
            project=PROJECT_ID,
            location=REGION,
            experiment=EXPERIMENT,
            experiment_tensorboard=False,
        )
        self.logger.info(f"Vertex AI Experiment: '{EXPERIMENT}'")

    # ------------------------------------------------------------------
    # Data — read directly from BigQuery, nothing written locally
    # ------------------------------------------------------------------

    def _load_data_from_bigquery(self) -> pd.DataFrame:
        """
        Pull the full feature table from BigQuery into a pandas DataFrame.

        Rows are sorted by machineID + date immediately so that
        TimeSeriesSplit respects the temporal ordering of the data.

        The query requests all columns; leakage columns (date, machineID)
        are kept here and dropped later in _prepare_features() so the
        sort can use them.
        """
        full_table = f"{PROJECT_ID}.{self.bq_dataset}.{self.bq_table}"
        query = f"""
            SELECT *
            FROM `{full_table}`
            ORDER BY machineID, date
        """
        self.logger.info(f"Reading from BigQuery: {full_table}")

        client = bigquery.Client(project=PROJECT_ID)
        data   = client.query(query).to_dataframe()

        # Ensure date column is parsed as datetime (BQ returns it as object
        # when the column type is DATE rather than TIMESTAMP)
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"])

        self.logger.info(
            f"Loaded {len(data):,} rows × {data.shape[1]} cols from BigQuery"
        )
        return data.reset_index(drop=True)

    def _prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Drop leakage columns and the target; return (X, y).

        Why these columns are dropped:
        • 'date'      – encodes temporal position; leaks ordering into features.
        • 'machineID' – identifier; model must generalise across machines.
        Both are still used for sorting before this method is called.
        """
        drop_cols = [c for c in LEAKAGE_COLS + [TARGET_COL] if c in data.columns]
        X = data.drop(columns=drop_cols)
        y = data[TARGET_COL]
        self.logger.info(f"Features ({len(X.columns)}): {list(X.columns)}")
        return X, y

    def _compute_scale_pos_weight(self, y: pd.Series) -> float:
        """count(negatives) / count(positives) to compensate class imbalance."""
        neg, pos = (y == 0).sum(), (y == 1).sum()
        ratio = neg / max(pos, 1)
        self.logger.info(
            f"Class distribution  0:{neg:,}  1:{pos:,}  scale_pos_weight={ratio:.2f}"
        )
        return ratio

    # ------------------------------------------------------------------
    # GCS artifact storage
    # ------------------------------------------------------------------

    def _gcs_upload(self, local_path: str, gcs_blob_path: str):
        """Upload a local file to the project GCS bucket."""
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)
        blob   = bucket.blob(gcs_blob_path)
        blob.upload_from_filename(local_path)
        self.logger.info(f"Uploaded → gs://{BUCKET_NAME}/{gcs_blob_path}")

    def _save_artifacts_to_gcs(
        self,
        model: XGBClassifier,
        scaler: StandardScaler,
        feature_names: List[str],
        run_ts: str,
    ) -> str:
        """
        Persist model + scaler + feature list to GCS.

        Uses a tempdir so nothing lingers on the container's local disk.
        Returns the GCS directory URI that Vertex AI Model Registry points at.

        Layout:
            gs://<bucket>/models/<MODEL_NAME>/<run_ts>/model.bst
            gs://<bucket>/models/<MODEL_NAME>/<run_ts>/scaler.pkl
            gs://<bucket>/models/<MODEL_NAME>/<run_ts>/feature_names.json
        """
        gcs_dir = f"models/{MODEL_NAME}/{run_ts}"

        with tempfile.TemporaryDirectory() as tmp:
            # XGBoost native binary (preferred over pickle for portability)
            model_path = os.path.join(tmp, "model.bst")
            model.save_model(model_path)
            self._gcs_upload(model_path, f"{gcs_dir}/model.bst")

            # Scaler (needed at serving time to transform incoming features)
            scaler_path = os.path.join(tmp, "scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            self._gcs_upload(scaler_path, f"{gcs_dir}/scaler.pkl")

            # Feature name list (validates input schema at serving time)
            fn_path = os.path.join(tmp, "feature_names.json")
            with open(fn_path, "w") as f:
                json.dump(feature_names, f)
            self._gcs_upload(fn_path, f"{gcs_dir}/feature_names.json")

        return f"gs://{BUCKET_NAME}/{gcs_dir}"

    # ------------------------------------------------------------------
    # Vertex AI Model Registry
    # ------------------------------------------------------------------

    def _register_model(self, artifact_uri: str, cv_metrics: Dict) -> aiplatform.Model:
        """
        Upload the GCS artifact directory to Vertex AI Model Registry.

        Uses Google's pre-built XGBoost serving container — no custom
        Docker image needed. The model appears in:
            Cloud Console → Vertex AI → Model Registry
        """
        self.logger.info("Registering model in Vertex AI Model Registry...")

        vertex_model = aiplatform.Model.upload(
            display_name=MODEL_NAME,
            artifact_uri=artifact_uri,
            # Google-managed pre-built XGBoost serving container
            serving_container_image_uri=(
                "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest"
            ),
            # These labels appear as metadata in the Cloud Console
            labels={
                "framework":  "xgboost",
                "pipeline":   "predictive-maintenance",
                "experiment": EXPERIMENT,
            },
            description=(
                f"XGBoost binary classifier — TimeSeriesSplit CV. "
                f"cv_roc_auc={cv_metrics.get('cv_roc_auc', 0):.4f}. "
                f"Source: {PROJECT_ID}.{self.bq_dataset}.{self.bq_table}"
            ),
        )

        self.logger.info(f"Model registered: {vertex_model.resource_name}")
        return vertex_model

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def train(self) -> Dict:
        """
        Full pipeline — no local file paths involved:
          1. Read features from BigQuery
          2. TimeSeriesSplit cross-validation
          3. Train final model on full in-memory dataset
          4. Save artifacts to GCS (via tempdir, nothing persisted locally)
          5. Log metrics + params to Vertex AI Experiment
          6. Register model in Vertex AI Model Registry

        Returns a dict with keys:
            vertex_model   → aiplatform.Model object
            resource_name  → full Vertex AI resource name string
            artifact_uri   → gs:// path to model artifacts
            cv_metrics     → dict of CV metric values
        This avoids callers having to re-fetch metadata from the SDK object,
        since aiplatform.Model does not expose artifact_uri after upload().
        """
        run_ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

        # ── 1. Load from BigQuery ──────────────────────────────────────
        data             = self._load_data_from_bigquery()
        X, y             = self._prepare_features(data)
        scale_pos_weight = self._compute_scale_pos_weight(y)

        # ── 2. Cross-validation (time-aware) ───────────────────────────
        tscv     = TimeSeriesSplit(n_splits=self.n_cv_splits)
        fold_buf = {
            "accuracy": [], "f1_macro": [],
            "recall_1": [], "recall_0": [], "roc_auc": []
        }

        for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X), start=1):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            scaler   = StandardScaler()
            X_tr_sc  = scaler.fit_transform(X_tr)
            X_val_sc = scaler.transform(X_val)

            mdl = XGBClassifier(
                n_estimators     = self.n_estimators,
                max_depth        = self.max_depth,
                learning_rate    = self.learning_rate,
                subsample        = self.subsample,
                colsample_bytree = self.colsample_bytree,
                min_child_weight = self.min_child_weight,
                scale_pos_weight = scale_pos_weight,
                random_state     = self.random_state,
                eval_metric      = "logloss",
                use_label_encoder= False,
            )
            mdl.fit(X_tr_sc, y_tr, eval_set=[(X_val_sc, y_val)], verbose=False)

            y_pred = mdl.predict(X_val_sc)
            y_prob = mdl.predict_proba(X_val_sc)[:, 1]
            rpt    = classification_report(y_val, y_pred, output_dict=True, zero_division=0)

            fold_buf["accuracy"].append(rpt["accuracy"])
            fold_buf["f1_macro"].append(rpt["macro avg"]["f1-score"])
            fold_buf["recall_1"].append(rpt.get("1", rpt.get("1.0", {})).get("recall", 0.0))
            fold_buf["recall_0"].append(rpt.get("0", rpt.get("0.0", {})).get("recall", 0.0))
            fold_buf["roc_auc"].append(
                roc_auc_score(y_val, y_prob) if y_val.nunique() > 1 else 0.0
            )
            self.logger.info(
                f"Fold {fold_idx}/{self.n_cv_splits} | "
                f"acc={fold_buf['accuracy'][-1]:.4f}  "
                f"f1={fold_buf['f1_macro'][-1]:.4f}  "
                f"auc={fold_buf['roc_auc'][-1]:.4f}"
            )

        mn = lambda lst: float(np.mean(lst))
        cv_metrics = {
            "cv_accuracy": mn(fold_buf["accuracy"]),
            "cv_f1_macro": mn(fold_buf["f1_macro"]),
            "cv_recall_1": mn(fold_buf["recall_1"]),
            "cv_recall_0": mn(fold_buf["recall_0"]),
            "cv_roc_auc":  mn(fold_buf["roc_auc"]),
        }
        self.logger.info(f"CV results: {cv_metrics}")

        # ── 3. Final model on full in-memory dataset ───────────────────
        self.logger.info("Training final model on full dataset...")
        scaler_final = StandardScaler()
        X_scaled     = scaler_final.fit_transform(X)

        final_model = XGBClassifier(
            n_estimators     = self.n_estimators,
            max_depth        = self.max_depth,
            learning_rate    = self.learning_rate,
            subsample        = self.subsample,
            colsample_bytree = self.colsample_bytree,
            min_child_weight = self.min_child_weight,
            scale_pos_weight = scale_pos_weight,
            random_state     = self.random_state,
            eval_metric      = "logloss",
            use_label_encoder= False,
        )
        final_model.fit(X_scaled, y, verbose=False)

        # ── 4. Save artifacts to GCS (no local persistence) ───────────
        artifact_uri = self._save_artifacts_to_gcs(
            model         = final_model,
            scaler        = scaler_final,
            feature_names = list(X.columns),
            run_ts        = run_ts,
        )

        # ── 5. Log to Vertex AI Experiment ────────────────────────────
        with aiplatform.start_run(run=f"run-{run_ts}"):
            aiplatform.log_params({
                "n_estimators":          self.n_estimators,
                "max_depth":             self.max_depth,
                "learning_rate":         self.learning_rate,
                "subsample":             self.subsample,
                "colsample_bytree":      self.colsample_bytree,
                "min_child_weight":      self.min_child_weight,
                "scale_pos_weight":      round(scale_pos_weight, 4),
                "n_cv_splits":           self.n_cv_splits,
                "leakage_cols_excluded": str(LEAKAGE_COLS),
                "bq_source":             f"{PROJECT_ID}.{self.bq_dataset}.{self.bq_table}",
                "artifact_uri":          artifact_uri,
            })
            aiplatform.log_metrics(cv_metrics)

        self.logger.info(f"Experiment run logged → '{EXPERIMENT}/run-{run_ts}'")

        # ── 6. Register in Vertex AI Model Registry ────────────────────
        vertex_model = self._register_model(artifact_uri, cv_metrics)

        # Return everything the pipeline needs explicitly.
        # aiplatform.Model does not expose artifact_uri after upload(),
        # so we carry it forward ourselves from _save_artifacts_to_gcs().
        return {
            "vertex_model":  vertex_model,
            "resource_name": vertex_model.resource_name,
            "artifact_uri":  artifact_uri,
            "cv_metrics":    cv_metrics,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    trainer = XGBoostTrainer()
    result  = trainer.train()
    print(f"Done. Model resource: {result['resource_name']}")
    print(f"Artifact URI:         {result['artifact_uri']}")
    print(f"CV metrics:           {result['cv_metrics']}")