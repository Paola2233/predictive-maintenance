"""train.py — XGBoost trainer that reads features directly from BigQuery
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
    cv_metrics.json     ← full per-fold + aggregate metrics

Observability improvements (v2)
────────────────────────────────
• Per-fold metrics logged individually (fold_1…fold_N) so you can inspect
  variance across time windows, not just averages.
• Std-deviation of each metric logged alongside the mean.
• XGBoost early-stopping eval log (logloss per boosting round) captured and
  saved to GCS as training_curves.json for custom dashboards.
• Vertex AI Experiments run includes ALL params (not just 4) + all metrics.
• Model Registry description and labels carry the full metric snapshot.
• _build_cv_summary() centralises metric aggregation → single source of
  truth consumed by both the trainer return value and the registry call.
"""

import json
import logging
import os
import pickle
import tempfile
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import optuna
import numpy as np
import pandas as pd
from google.cloud import bigquery, storage
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import google.cloud.aiplatform as aiplatform

# ---------------------------------------------------------------------------
# Configuration — all overridable via env vars for container deployments
# ---------------------------------------------------------------------------

PROJECT_ID   = os.getenv("GCP_PROJECT_ID")
REGION       = os.getenv("GCP_REGION", "us-central1")
BUCKET_NAME  = os.getenv("GCS_BUCKET_NAME")
MODEL_NAME   = os.getenv("MODEL_NAME", "XGBoostPredictiveMaintenance")
EXPERIMENT   = os.getenv("VERTEX_EXPERIMENT", "predictive-maintenance-xgboost")
BQ_DATASET   = os.getenv("BQ_DATASET", "preprocessed_data")
BQ_TABLE     = os.getenv("BQ_TABLE", "features_engineered_v2")

LEAKAGE_COLS = ["date", "machineID", "days_until_failure"]
TARGET_COL   = "will_fail_7_days"

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
    • ALL metrics (per-fold + aggregate + std) logged to Vertex AI Experiments.

    Observability contract
    ──────────────────────
    The trainer logs three levels of granularity to Vertex AI Experiments:

    1. Hyper-parameters  (aiplatform.log_params)
       All XGBoost constructor args + derived scale_pos_weight.

    2. Per-fold metrics  (aiplatform.log_metrics, prefixed fold_N_*)
       Lets you spot overfitting or instability across time windows.

    3. Aggregate metrics (aiplatform.log_metrics, prefixed cv_*)
       cv_<metric>_mean  – mean across folds (used for gating)
       cv_<metric>_std   – std across folds (flags unstable models)

    Additionally, the XGBoost eval log (logloss per round) for every fold
    is saved as training_curves.json in GCS for custom BigQuery/Looker
    dashboards.
    """

    def __init__(
        self,
        n_estimators: int     = 300,
        max_depth: int        = 6,
        learning_rate: float  = 0.05,
        subsample: float      = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        random_state: int     = 42,
        n_cv_splits: int      = 5,
        early_stopping_rounds: int = 20,
        bq_dataset: str       = BQ_DATASET,
        bq_table: str         = BQ_TABLE,
    ):
        self.n_estimators         = n_estimators
        self.max_depth            = max_depth
        self.learning_rate        = learning_rate
        self.subsample            = subsample
        self.colsample_bytree     = colsample_bytree
        self.min_child_weight     = min_child_weight
        self.random_state         = random_state
        self.n_cv_splits          = n_cv_splits
        self.early_stopping_rounds = early_stopping_rounds
        self.bq_dataset           = bq_dataset
        self.bq_table             = bq_table

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("XGBoostTrainer")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------

    def _load_data_from_bigquery(self) -> pd.DataFrame:
        self.logger.info(
            f"Reading from {PROJECT_ID}.{self.bq_dataset}.{self.bq_table}"
        )
        client = bigquery.Client(project=PROJECT_ID)
        query = (
            f"SELECT * FROM `{PROJECT_ID}.{self.bq_dataset}.{self.bq_table}` "
            f"ORDER BY date ASC"
        )
        df = client.query(query).to_dataframe()
        self.logger.info(f"Loaded {df.shape[0]:,} rows.")
        return df

    def _prepare_features(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Drop leakage columns and the target; return (X, y).

        Why these columns are dropped:
        • 'date'      – encodes temporal position; leaks ordering into features.
        • 'machineID' – identifier; model must generalise across machines.
        Both are still used for sorting before this method is called.
        """
        # Checking data leakage columns dynamically in case of future additions (e.g. rolling stats)
        rolling_cols = [c for c in data.columns if '_mean_' in c or '_std_' in c]
        drop_cols = [c for c in LEAKAGE_COLS + [TARGET_COL] + rolling_cols if c in data.columns]
        # drop_cols = [c for c in LEAKAGE_COLS + [TARGET_COL] if c in data.columns]
        X = data.drop(columns=drop_cols)
        y = data[TARGET_COL]
        self.logger.info(f"Features ({len(X.columns)}): {list(X.columns)}")
        return X, y

    def _compute_scale_pos_weight(self, y: pd.Series) -> float:
        neg = (y == 0).sum()
        pos = (y == 1).sum()
        spw = neg / pos if pos > 0 else 1.0
        self.logger.info(
            f"Class distribution: Neg={neg}, Pos={pos} → scale_pos_weight={spw:.2f}"
        )
        return spw

    # -----------------------------------------------------------------------
    # GCS helpers
    # -----------------------------------------------------------------------

    def _gcs_upload(self, local_path: str, gcs_path: str):
        client  = storage.Client(project=PROJECT_ID)
        bucket  = client.bucket(BUCKET_NAME)
        blob    = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        self.logger.info(f"Uploaded → gs://{BUCKET_NAME}/{gcs_path}")

    def _save_artifacts_to_gcs(
        self,
        model: XGBClassifier,
        scaler: StandardScaler,
        feature_names: List[str],
        cv_metrics: Dict,
        training_curves: List[Dict],
        run_ts: str,
    ) -> str:
        """
        Save model artifacts + observability assets to GCS.

        Files written
        ─────────────
        model.bst             ← XGBoost native binary
        scaler.pkl            ← fitted StandardScaler
        feature_names.json    ← ordered feature column list
        cv_metrics.json       ← full aggregate + per-fold metrics
        training_curves.json  ← logloss per boosting round, per fold
        """
        gcs_dir = f"models/{MODEL_NAME}/{run_ts}"
        tmp = tempfile.mkdtemp()

        # Model binary
        model_path = os.path.join(tmp, "model.bst")
        model.save_model(model_path)
        self._gcs_upload(model_path, f"{gcs_dir}/model.bst")

        # Scaler
        scaler_path = os.path.join(tmp, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        self._gcs_upload(scaler_path, f"{gcs_dir}/scaler.pkl")

        # Feature list
        fn_path = os.path.join(tmp, "feature_names.json")
        with open(fn_path, "w") as f:
            json.dump(feature_names, f)
        self._gcs_upload(fn_path, f"{gcs_dir}/feature_names.json")

        # Full CV metrics snapshot (aggregate + per-fold)
        metrics_path = os.path.join(tmp, "cv_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(cv_metrics, f, indent=2)
        self._gcs_upload(metrics_path, f"{gcs_dir}/cv_metrics.json")

        # Training curves (logloss per boosting round, per fold)
        curves_path = os.path.join(tmp, "training_curves.json")
        with open(curves_path, "w") as f:
            json.dump(training_curves, f)
        self._gcs_upload(curves_path, f"{gcs_dir}/training_curves.json")

        return f"gs://{BUCKET_NAME}/{gcs_dir}"

    # -----------------------------------------------------------------------
    # Metric aggregation
    # -----------------------------------------------------------------------

    def _build_cv_summary(
        self,
        fold_buf: Dict[str, List[float]],
        per_fold_rows: List[Dict],
    ) -> Dict:
        """
        Centralised metric aggregation consumed by Vertex Experiments,
        the trainer return value, and the Model Registry label/description.

        Returns a flat dict with keys:
          cv_<metric>_mean   – mean across folds
          cv_<metric>_std    – std  across folds
          fold_<N>_<metric>  – individual fold values (for Vertex Experiments)
        """
        summary: Dict = {}

        # Aggregate stats
        for metric, values in fold_buf.items():
            arr = np.array(values)
            summary[f"cv_{metric}_mean"] = float(arr.mean())
            summary[f"cv_{metric}_std"]  = float(arr.std())

        # Per-fold flat keys (fold_1_roc_auc, fold_2_roc_auc, …)
        for row in per_fold_rows:
            fold_n = row["fold"]
            for metric, value in row.items():
                if metric == "fold":
                    continue
                summary[f"fold_{fold_n}_{metric}"] = float(value)

        return summary

    # -----------------------------------------------------------------------
    # Model Registry
    # -----------------------------------------------------------------------

    def _register_model(
        self, artifact_uri: str, cv_summary: Dict, scale_pos_weight: float
    ) -> aiplatform.Model:
        self.logger.info("Registering model in Vertex AI Model Registry...")

        # Vertex AI labels must be strings ≤63 chars, no periods allowed.
        def _label_val(v: float) -> str:
            return str(round(v, 4)).replace(".", "_")[:63]

        roc_auc  = cv_summary.get("cv_roc_auc_mean", 0)
        pr_auc   = cv_summary.get("cv_pr_auc_mean", 0)
        f2       = cv_summary.get("cv_f2_score_mean", 0)
        f1       = cv_summary.get("cv_f1_macro_mean", 0)
        rec_1    = cv_summary.get("cv_recall_1_mean", 0)
        rec_0    = cv_summary.get("cv_recall_0_mean", 0)

        vertex_model = aiplatform.Model.upload(
            display_name          = MODEL_NAME,
            artifact_uri          = artifact_uri,
            serving_container_image_uri=(
                "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest"
            ),
            labels={
                "framework":       "xgboost",
                "pipeline":        "predictive-maintenance",
                "experiment":      EXPERIMENT,
                # Snapshot of key metrics baked into the registry label
                # so you can filter/compare model versions without opening
                # an experiment run.
                "cv-roc-auc":      _label_val(roc_auc),
                "cv-pr-auc":       _label_val(pr_auc),
                "cv-f2-score":     _label_val(f2),
                "cv-f1-macro":     _label_val(f1),
                "cv-recall-pos":   _label_val(rec_1),
                "cv-recall-neg":   _label_val(rec_0),
            },
            description=(
                f"XGBoost binary classifier — TimeSeriesSplit CV "
                f"({self.n_cv_splits} folds). "
                f"cv_roc_auc={roc_auc:.4f}±{cv_summary.get('cv_roc_auc_std', 0):.4f} | "
                f"cv_pr_auc={pr_auc:.4f}±{cv_summary.get('cv_pr_auc_std', 0):.4f} | "
                f"cv_f2={f2:.4f}±{cv_summary.get('cv_f2_score_std', 0):.4f} | "
                f"cv_recall_pos={rec_1:.4f} | cv_recall_neg={rec_0:.4f}. "
                f"Source: {PROJECT_ID}.{self.bq_dataset}.{self.bq_table}"
            ),
        )

        self.logger.info(f"Model registered: {vertex_model.resource_name}")
        return vertex_model

    # -----------------------------------------------------------------------
    # Vertex AI Experiments
    # -----------------------------------------------------------------------

    def _log_to_vertex_experiments(
        self,
        run_ts: str,
        scale_pos_weight: float,
        cv_summary: Dict,
    ):
        """
        Log ALL hyper-parameters and ALL CV metrics (mean + std + per-fold)
        to a Vertex AI Experiment run.

        Why this matters:
        • Compare runs side-by-side in Cloud Console → Vertex AI → Experiments.
        • Plot metric trends over pipeline runs.
        • std metrics reveal unstable models even when the mean looks good.
        """
        aiplatform.init(project=PROJECT_ID, location=REGION, experiment=EXPERIMENT)

        with aiplatform.start_run(run=f"xgb-{run_ts}"):
            # All hyper-params — not just the 4 logged previously
            aiplatform.log_params({
                "n_estimators":          self.n_estimators,
                "max_depth":             self.max_depth,
                "learning_rate":         self.learning_rate,
                "subsample":             self.subsample,
                "colsample_bytree":      self.colsample_bytree,
                "min_child_weight":      self.min_child_weight,
                "random_state":          self.random_state,
                "n_cv_splits":           self.n_cv_splits,
                "early_stopping_rounds": self.early_stopping_rounds,
                "scale_pos_weight":      round(scale_pos_weight, 4),
                "bq_dataset":            self.bq_dataset,
                "bq_table":              self.bq_table,
            })

            # All aggregate + per-fold metrics in one call.
            # Vertex AI Experiments accepts any flat float dict.
            aiplatform.log_metrics(cv_summary)

        self.logger.info(f"Experiment run logged: xgb-{run_ts}")

    # -----------------------------------------------------------------------
    # Main training entry-point
    # -----------------------------------------------------------------------

    def train(self) -> Dict:
        run_ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

        # ── 1. Load & Prepare ────────────────────────────────────────────
        data             = self._load_data_from_bigquery()
        X, y             = self._prepare_features(data)
        scale_pos_weight = self._compute_scale_pos_weight(y)

        # ── 2. Cross-validation ──────────────────────────────────────────
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)

        fold_buf: Dict[str, List[float]] = {
            "accuracy":  [],
            "f1_macro":  [],
            "f2_score":  [],
            "recall_1":  [],
            "recall_0":  [],
            "precision_1": [],
            "roc_auc":   [],
            "pr_auc":    [],
        }

        # Per-fold rows (used to build per-fold flat keys in cv_summary)
        per_fold_rows: List[Dict] = []

        # XGBoost eval log (logloss per round) keyed by fold — saved to GCS
        training_curves: List[Dict] = []

        # ── 3. Training Loop ─────────────────────────────────────────────
        for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X), start=1):
            X_tr,  X_val  = X.iloc[tr_idx],  X.iloc[val_idx]
            y_tr,  y_val  = y.iloc[tr_idx],  y.iloc[val_idx]
            print(f"Fold {fold_idx}: pos_rate={y_val.mean():.3f} n={len(y_val)}")

            scaler    = StandardScaler()
            X_tr_sc   = scaler.fit_transform(X_tr)
            X_val_sc  = scaler.transform(X_val)

            # Capture eval results dict for training curve export
            # evals_result: Dict = {}

            mdl = XGBClassifier(
                n_estimators          = self.n_estimators,
                max_depth             = self.max_depth,
                learning_rate         = self.learning_rate,
                subsample             = self.subsample,
                colsample_bytree      = self.colsample_bytree,
                min_child_weight      = self.min_child_weight,
                scale_pos_weight      = scale_pos_weight,
                random_state          = self.random_state,
                eval_metric           = "logloss",
                early_stopping_rounds = self.early_stopping_rounds,
                use_label_encoder     = False,
            )
            mdl.fit(
                X_tr_sc, y_tr,
                eval_set        = [(X_tr_sc, y_tr), (X_val_sc, y_val)],
                verbose         = False,
                # callbacks       = [],
                # xgboost stores curves in evals_result after fit
            )
            # evals_result is populated after fit when passed as kwarg
            # (XGBoost ≥1.6 populates mdl.evals_result() instead)
            curves = mdl.evals_result()  # {"validation_0": {"logloss": [...]}, ...}
            training_curves.append({
                "fold":        fold_idx,
                "train_logloss": curves.get("validation_0", {}).get("logloss", []),
                "val_logloss":   curves.get("validation_1", {}).get("logloss", []),
                "best_iteration": int(mdl.best_iteration)
                    if hasattr(mdl, "best_iteration") else self.n_estimators,
            })

            y_pred = mdl.predict(X_val_sc)
            y_prob = mdl.predict_proba(X_val_sc)[:, 1]

            rep = classification_report(
                y_val, y_pred, output_dict=True, zero_division=0
            )
            cm  = confusion_matrix(y_val, y_pred)

            fold_metrics = {
                "accuracy":    rep["accuracy"],
                "f1_macro":    rep["macro avg"]["f1-score"],
                "recall_1":    rep.get("1", {}).get("recall", 0),
                "recall_0":    rep.get("0", {}).get("recall", 0),
                "precision_1": rep.get("1", {}).get("precision", 0),
                "roc_auc":     roc_auc_score(y_val, y_prob),
                "pr_auc":      average_precision_score(y_val, y_prob),
                "f2_score":    fbeta_score(
                    y_val, y_pred, beta=2, average="binary", zero_division=0
                ),
                # Confusion matrix cells — useful for per-fold TP/FP inspection
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            }

            for metric in fold_buf:
                fold_buf[metric].append(fold_metrics[metric])

            per_fold_rows.append({"fold": fold_idx, **fold_metrics})

            self.logger.info(
                f"Fold {fold_idx}/{self.n_cv_splits} — "
                f"ROC-AUC={fold_metrics['roc_auc']:.4f} | "
                f"PR-AUC={fold_metrics['pr_auc']:.4f} | "
                f"F2={fold_metrics['f2_score']:.4f} | "
                f"Recall(+)={fold_metrics['recall_1']:.4f} | "
                f"Recall(-)={fold_metrics['recall_0']:.4f} | "
                f"best_iter={training_curves[-1]['best_iteration']}"
            )

        # ── 4. Aggregate Metrics ─────────────────────────────────────────
        cv_summary = self._build_cv_summary(fold_buf, per_fold_rows)
        self.logger.info(
            f"CV Summary:\n{json.dumps(cv_summary, indent=2)}"
        )

        # ── 5. Final Model Fit (on full dataset) ─────────────────────────
        scaler  = StandardScaler()
        X_sc    = scaler.fit_transform(X)

        final_model = XGBClassifier(
            n_estimators          = self.n_estimators,
            max_depth             = self.max_depth,
            learning_rate         = self.learning_rate,
            subsample             = self.subsample,
            colsample_bytree      = self.colsample_bytree,
            min_child_weight      = self.min_child_weight,
            scale_pos_weight      = scale_pos_weight,
            random_state          = self.random_state,
            use_label_encoder     = False,
        )
        final_model.fit(X_sc, y)

        # ── 6. Save Artifacts to GCS ─────────────────────────────────────
        artifact_uri = self._save_artifacts_to_gcs(
            model          = final_model,
            scaler         = scaler,
            feature_names  = list(X.columns),
            cv_metrics     = cv_summary,
            training_curves= training_curves,
            run_ts         = run_ts,
        )

        # ── 7. Log to Vertex AI Experiments ─────────────────────────────
        self._log_to_vertex_experiments(run_ts, scale_pos_weight, cv_summary)

        # ── 8. Register in Model Registry ───────────────────────────────
        vertex_model = self._register_model(artifact_uri, cv_summary, scale_pos_weight)

        return {
            "vertex_model":  vertex_model,
            "resource_name": vertex_model.resource_name,
            "artifact_uri":  artifact_uri,
            "cv_summary":    cv_summary,
            # Back-compat aliases for evaluate_component
            "cv_metrics": {
                "cv_roc_auc":   cv_summary["cv_roc_auc_mean"],
                "cv_pr_auc":    cv_summary["cv_pr_auc_mean"],
                "cv_f2_score":  cv_summary["cv_f2_score_mean"],
                "cv_f1_macro":  cv_summary["cv_f1_macro_mean"],
                "cv_recall_1":  cv_summary["cv_recall_1_mean"],
                "cv_recall_0":  cv_summary["cv_recall_0_mean"],
            },
        }
    
    def _optuna_objective(self, trial, X, y, scale_pos_weight):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
        f2_scores = []
        
        for tr_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_val_sc = scaler.transform(X_val)
            
            mdl = XGBClassifier(
                **params,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                use_label_encoder=False,
            )
            mdl.fit(X_tr_sc, y_tr, eval_set=[(X_val_sc, y_val)], verbose=False)
            
            y_pred = mdl.predict(X_val_sc)
            f2 = fbeta_score(y_val, y_pred, beta=2, average="binary", zero_division=0)
            f2_scores.append(f2)
        
        return np.mean(f2_scores)  # ← Optuna maximizes this

    def tune(self, n_trials: int = 30) -> Dict:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        data = self._load_data_from_bigquery()
        X, y = self._prepare_features(data)
        scale_pos_weight = self._compute_scale_pos_weight(y)
        
        study = optuna.create_study(direction="maximize")  # maximize F2
        study.optimize(
            lambda trial: self._optuna_objective(trial, X, y, scale_pos_weight),
            n_trials=n_trials,
        )
        
        self.logger.info(f"Best F2: {study.best_value:.4f}")
        self.logger.info(f"Best params: {study.best_params}")
        
        # Update trainer with best params before calling train()
        for k, v in study.best_params.items():
            setattr(self, k, v)
        
        return self.train()  # runs full training with best params