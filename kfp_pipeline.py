"""
pipeline.py — KFP v2 pipeline for Predictive Maintenance on Vertex AI.

All experiment tracking, model storage, and model registration are done
natively through the Vertex AI SDK.  No MLflow dependency.

Pipeline stages
───────────────
  1. preprocess   – feature engineering → BigQuery + GCS CSV artefact
  2. train        – XGBoost training → GCS artifacts → Vertex AI Experiment run
  3. evaluate     – quality gate (ROC-AUC threshold) → Vertex AI Model Registry
  4. deploy       – conditional: only runs when gate passes → Vertex AI Endpoint

Observability improvements (v2)
────────────────────────────────
• train_component:
    – Logs ALL cv metrics (mean + std for every metric) to Output[Metrics].
    – Logs a ClassificationMetrics artifact (confusion matrix + ROC curve) so
      the Vertex AI Pipelines UI renders the visual panels natively.
    – Writes the full cv_summary (including per-fold values) into the JSON
      handoff file so downstream components can inspect it.

• evaluate_component:
    – Added Output[Metrics] → evaluation gate results appear in Pipelines UI.
    – Added Output[ClassificationMetrics] for the gate decision record.
    – Multi-metric gate: ROC-AUC AND PR-AUC AND F2 must pass their thresholds
      (configurable). The gate decision and margin for each metric are logged.

GCP resources this pipeline expects to already exist (see README section below):
  • GCS bucket         : gs://<BUCKET_NAME>/          (pipeline root + model artifacts)
  • BigQuery dataset   : <PROJECT>.raw_data            (source tables)
  • BigQuery dataset   : <PROJECT>.preprocessed_data   (output table)
  • Vertex AI Experiment: created automatically by the SDK if missing
  • Service Account   : with roles listed in README

Run
───
  # 1. Compile only
  python pipeline.py --project my-proj --bucket my-bucket --compile-only

  # 2. Compile + submit
  python pipeline.py \\
      --project       my-proj \\
      --region        us-central1 \\
      --pipeline-root gs://my-bucket/pipeline-root \\
      --service-account  sa@my-proj.iam.gserviceaccount.com \\
      --bucket-name   my-bucket
"""

import argparse
from typing import NamedTuple

import google.cloud.aiplatform as aip
from kfp import compiler, dsl
from kfp.dsl import ClassificationMetrics, Metrics, Model, Output, component, Input

# ---------------------------------------------------------------------------
# Shared base image for all components
# ---------------------------------------------------------------------------
BASE_IMAGE = "us-central1-docker.pkg.dev/abiding-kingdom-491823-b0/predictive-maintenance-repo/predictive-maintenance:v1"

# ===========================================================================
# Component 1 – Feature Engineering / Preprocessing
# ===========================================================================

@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "polars>=0.20",
        "google-cloud-bigquery>=3.0",
        "db-dtypes",
        "pyarrow",
        "google-cloud-storage",
    ],
)
def preprocess_component(
    project_id: str,
    bucket_name: str,
    target_window_size: int,
    bq_dataset: str,
    bq_table: str,
):
    """
    Runs DataPreprocessor:
      • Loads raw tables from BigQuery (raw_data dataset)
      • Engineers rolling-window telemetry features + lag counts + target label
      • Saves the result to BigQuery (preprocessed_data dataset)

    No local file or KFP artifact is produced — BigQuery is the handoff
    between this component and the train component.
    """
    import os
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("preprocess")

    os.environ["GCP_PROJECT_ID"] = project_id
    os.environ["GCS_BUCKET_NAME"] = bucket_name

    from src.model.preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor(target_window_size=target_window_size)
    df = preprocessor.create_features(save_df=True)

    logger.info(
        f"Preprocessing complete: {df.shape[0]:,} rows × {df.shape[1]} cols "
        f"→ BigQuery {project_id}.{bq_dataset}.{bq_table}"
    )


# ===========================================================================
# Component 2 – XGBoost Training
# ===========================================================================

@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "xgboost>=2.0",
        "scikit-learn>=1.3",
        "pandas>=2.0",
        "numpy",
        "google-cloud-bigquery>=3.0",
        "db-dtypes",
        "google-cloud-storage",
        "google-cloud-aiplatform>=1.40",
    ],
)
def train_component(
    project_id: str,
    region: str,
    bucket_name: str,
    model_name: str,
    vertex_experiment: str,
    bq_dataset: str,
    bq_table: str,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    n_cv_splits: int,
    output_model: Output[Model],
    output_metrics: Output[Metrics],
    output_clf_metrics: Output[ClassificationMetrics],
) -> NamedTuple(
    "TrainOutputs",
    [
        ("roc_auc", float),
        ("pr_auc", float),
        ("f2_score", float),
        ("recall_pos", float),
        ("recall_neg", float),
        ("artifact_uri", str),
    ],
):
    """
    Trains XGBoost with TimeSeriesSplit CV.

    KFP UI artefacts produced
    ─────────────────────────
    output_metrics       → scalar cards in the Vertex AI Pipelines run view.
                           Includes mean ± std for every CV metric.
    output_clf_metrics   → Confusion matrix + ROC curve panels rendered
                           natively in the Pipelines UI (no external tool).
    output_model (JSON)  → handoff payload for evaluate_component.
    """
    import os
    import json
    import numpy as np
    from collections import namedtuple

    os.environ["GCP_PROJECT_ID"]   = project_id
    os.environ["GCP_REGION"]       = region
    os.environ["GCS_BUCKET_NAME"]  = bucket_name
    os.environ["MODEL_NAME"]       = model_name
    os.environ["VERTEX_EXPERIMENT"]= vertex_experiment
    os.environ["BQ_DATASET"]       = bq_dataset
    os.environ["BQ_TABLE"]         = bq_table

    from src.model.xg_boost_trainer import XGBoostTrainer

    trainer = XGBoostTrainer(
        n_estimators     = n_estimators,
        max_depth        = max_depth,
        learning_rate    = learning_rate,
        subsample        = subsample,
        colsample_bytree = colsample_bytree,
        n_cv_splits      = n_cv_splits,
        bq_dataset       = bq_dataset,
        bq_table         = bq_table,
    )

    # result keys: vertex_model, resource_name, artifact_uri, cv_summary, cv_metrics
    result      = trainer.tune()
    resource_name = result["resource_name"]
    artifact_uri  = result["artifact_uri"]
    cv_summary    = result["cv_summary"]   # full flat dict (mean + std + per-fold)
    cv_metrics    = result["cv_metrics"]   # back-compat aliases (plain mean values)

    # ── Extract headline metrics ─────────────────────────────────────────
    roc_auc    = float(cv_metrics.get("cv_roc_auc",  0.0))
    pr_auc     = float(cv_metrics.get("cv_pr_auc",   0.0))
    f2_score   = float(cv_metrics.get("cv_f2_score", 0.0))
    f1_macro   = float(cv_metrics.get("cv_f1_macro", 0.0))
    recall_pos = float(cv_metrics.get("cv_recall_1", 0.0))
    recall_neg = float(cv_metrics.get("cv_recall_0", 0.0))
    precision_pos = float(cv_summary.get("cv_precision_1_mean", 0.0))

    roc_auc_std  = float(cv_summary.get("cv_roc_auc_std",  0.0))
    pr_auc_std   = float(cv_summary.get("cv_pr_auc_std",   0.0))
    f2_std       = float(cv_summary.get("cv_f2_score_std", 0.0))
    f1_std       = float(cv_summary.get("cv_f1_macro_std", 0.0))
    recall_pos_std = float(cv_summary.get("cv_recall_1_std", 0.0))
    recall_neg_std = float(cv_summary.get("cv_recall_0_std", 0.0))

    # ── Output[Metrics] — scalar cards in Pipelines UI ───────────────────
    # Aggregate means
    output_metrics.log_metric("cv_roc_auc_mean",      roc_auc)
    output_metrics.log_metric("cv_pr_auc_mean",        pr_auc)
    output_metrics.log_metric("cv_f2_score_mean",      f2_score)
    output_metrics.log_metric("cv_f1_macro_mean",      f1_macro)
    output_metrics.log_metric("cv_recall_pos_mean",    recall_pos)
    output_metrics.log_metric("cv_recall_neg_mean",    recall_neg)
    output_metrics.log_metric("cv_precision_pos_mean", precision_pos)

    # Std deviations — a high std on recall_pos flags instability
    output_metrics.log_metric("cv_roc_auc_std",      roc_auc_std)
    output_metrics.log_metric("cv_pr_auc_std",        pr_auc_std)
    output_metrics.log_metric("cv_f2_score_std",      f2_std)
    output_metrics.log_metric("cv_f1_macro_std",      f1_std)
    output_metrics.log_metric("cv_recall_pos_std",    recall_pos_std)
    output_metrics.log_metric("cv_recall_neg_std",    recall_neg_std)

    # Per-fold ROC-AUC (quick variance inspection without opening the run)
    for fold in range(1, n_cv_splits + 1):
        fold_roc = float(cv_summary.get(f"fold_{fold}_roc_auc", 0.0))
        fold_pr  = float(cv_summary.get(f"fold_{fold}_pr_auc",  0.0))
        fold_f2  = float(cv_summary.get(f"fold_{fold}_f2_score",0.0))
        output_metrics.log_metric(f"fold_{fold}_roc_auc", fold_roc)
        output_metrics.log_metric(f"fold_{fold}_pr_auc",  fold_pr)
        output_metrics.log_metric(f"fold_{fold}_f2_score",fold_f2)

    # ── Output[ClassificationMetrics] — visual panels in Pipelines UI ────
    # Aggregate confusion matrix from fold data
    agg_tn = int(sum(cv_summary.get(f"fold_{f}_tn", 0) for f in range(1, n_cv_splits + 1)))
    agg_fp = int(sum(cv_summary.get(f"fold_{f}_fp", 0) for f in range(1, n_cv_splits + 1)))
    agg_fn = int(sum(cv_summary.get(f"fold_{f}_fn", 0) for f in range(1, n_cv_splits + 1)))
    agg_tp = int(sum(cv_summary.get(f"fold_{f}_tp", 0) for f in range(1, n_cv_splits + 1)))

    output_clf_metrics.log_confusion_matrix(
        categories=["No Failure (0)", "Will Fail (1)"],
        matrix=[[agg_tn, agg_fp], [agg_fn, agg_tp]],
    )

    # ROC curve — reconstruct from aggregate TPR/FPR approximation.
    # A production pipeline would pass the full threshold sweep from the
    # trainer; here we log the mean operating point + convex hull endpoints
    # which is sufficient for the Pipelines UI panel.
    tpr = recall_pos
    fpr = 1.0 - recall_neg
    output_clf_metrics.log_roc_curve(
        fpr=[0.0, round(fpr, 4), 1.0],
        tpr=[0.0, round(tpr, 4), 1.0],
        threshold=[1.0, 0.5, 0.0],
    )

    # ── JSON handoff for evaluate_component ──────────────────────────────
    summary = {
        "resource_name": resource_name,
        "artifact_uri":  artifact_uri,
        # headline metrics (evaluate_component reads these)
        "roc_auc":       roc_auc,
        "pr_auc":        pr_auc,
        "f2_score":      f2_score,
        "f1_macro":      f1_macro,
        "recall_pos":    recall_pos,
        "recall_neg":    recall_neg,
        # full summary for any downstream component that wants it
        "cv_summary":    cv_summary,
    }
    with open(output_model.path, "w") as fh:
        json.dump(summary, fh, indent=2)

    output_model.metadata["resource_name"] = resource_name
    output_model.metadata["artifact_uri"]  = artifact_uri
    output_model.metadata["cv_roc_auc"]    = roc_auc
    output_model.metadata["cv_pr_auc"]     = pr_auc
    output_model.metadata["cv_f2_score"]   = f2_score

    TrainOutputs = namedtuple(
        "TrainOutputs",
        ["roc_auc", "pr_auc", "f2_score", "recall_pos", "recall_neg", "artifact_uri"],
    )
    return TrainOutputs(
        roc_auc      = roc_auc,
        pr_auc       = pr_auc,
        f2_score     = f2_score,
        recall_pos   = recall_pos,
        recall_neg   = recall_neg,
        artifact_uri = artifact_uri,
    )


# ===========================================================================
# Component 3 – Evaluation Gate
# ===========================================================================

@component(
    base_image=BASE_IMAGE,
    packages_to_install=["google-cloud-aiplatform>=1.40"],
)
def evaluate_component(
    input_model: Input[Model],
    project_id: str,
    region: str,
    roc_auc_threshold: float,
    pr_auc_threshold: float,
    f2_score_threshold: float,
    recall_pos_threshold: float,
    approved_model: Output[Model],
    eval_metrics: Output[Metrics],
    eval_clf_metrics: Output[ClassificationMetrics],
) -> NamedTuple("EvalOutputs", [("approved", bool)]):
    """
    Multi-metric quality gate.

    Gate logic
    ──────────
    The model is APPROVED only when ALL four thresholds are met:
      • ROC-AUC  ≥ roc_auc_threshold    (overall discrimination)
      • PR-AUC   ≥ pr_auc_threshold     (precision-recall trade-off on minority class)
      • F2       ≥ f2_score_threshold   (recall-weighted F-score; penalises missed failures)
      • Recall(+)≥ recall_pos_threshold (minimum detection rate for actual failures)

    Why multi-metric?
    ─────────────────
    A model can pass a single ROC-AUC gate while having unacceptably low
    recall on the positive (failure) class — exactly the scenario we must
    avoid in predictive maintenance. Gating on F2 and recall_pos directly
    closes that loophole.

    KFP UI artefacts produced
    ─────────────────────────
    eval_metrics     → per-metric cards + pass/fail margin.
    eval_clf_metrics → confusion matrix forwarded from train summary.
    """
    import json
    import logging
    from collections import namedtuple

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("evaluate")

    with open(input_model.path) as fh:
        summary = json.load(fh)

    roc_auc    = float(summary.get("roc_auc",    0))
    pr_auc     = float(summary.get("pr_auc",     0))
    f2_score   = float(summary.get("f2_score",   0))
    recall_pos = float(summary.get("recall_pos", 0))
    recall_neg = float(summary.get("recall_neg", 0))
    resource_name = summary.get("resource_name", "")
    cv_summary    = summary.get("cv_summary", {})

    # ── Gate decisions (all must pass) ───────────────────────────────────
    checks = {
        "roc_auc":    (roc_auc,    roc_auc_threshold),
        "pr_auc":     (pr_auc,     pr_auc_threshold),
        "f2_score":   (f2_score,   f2_score_threshold),
        "recall_pos": (recall_pos, recall_pos_threshold),
    }

    approved = all(value >= threshold for value, threshold in checks.values())

    for metric, (value, threshold) in checks.items():
        passed = value >= threshold
        margin = value - threshold
        logger.info(
            f"  {metric}: {value:.4f} {'≥' if passed else '<'} "
            f"{threshold:.4f} (margin={margin:+.4f}) → {'PASS ✓' if passed else 'FAIL ✗'}"
        )

    logger.info(
        f"Gate result: {'APPROVED ✓' if approved else 'REJECTED ✗'}"
    )

    # ── Output[Metrics] — evaluation gate cards in Pipelines UI ──────────
    eval_metrics.log_metric("roc_auc",              roc_auc)
    eval_metrics.log_metric("roc_auc_threshold",    roc_auc_threshold)
    eval_metrics.log_metric("roc_auc_margin",       round(roc_auc    - roc_auc_threshold,    4))

    eval_metrics.log_metric("pr_auc",               pr_auc)
    eval_metrics.log_metric("pr_auc_threshold",     pr_auc_threshold)
    eval_metrics.log_metric("pr_auc_margin",        round(pr_auc     - pr_auc_threshold,     4))

    eval_metrics.log_metric("f2_score",             f2_score)
    eval_metrics.log_metric("f2_score_threshold",   f2_score_threshold)
    eval_metrics.log_metric("f2_score_margin",      round(f2_score   - f2_score_threshold,   4))

    eval_metrics.log_metric("recall_pos",           recall_pos)
    eval_metrics.log_metric("recall_pos_threshold", recall_pos_threshold)
    eval_metrics.log_metric("recall_pos_margin",    round(recall_pos - recall_pos_threshold, 4))

    eval_metrics.log_metric("recall_neg",           recall_neg)
    eval_metrics.log_metric("approved",             float(approved))

    # ── Output[ClassificationMetrics] — visual panel in Pipelines UI ─────
    n_folds = int(cv_summary.get("cv_roc_auc_mean", 0) and
                  sum(1 for k in cv_summary if k.startswith("fold_") and k.endswith("_tp")))
    # Fall back: reconstruct from aggregated confusion matrix if available
    agg_tn = int(cv_summary.get("fold_1_tn", 0))  # just fold 1 if summary missing
    agg_fp = int(cv_summary.get("fold_1_fp", 0))
    agg_fn = int(cv_summary.get("fold_1_fn", 0))
    agg_tp = int(cv_summary.get("fold_1_tp", 0))
    # Sum all folds for the aggregate view
    folds_found = [k.split("_")[1] for k in cv_summary if k.startswith("fold_") and k.endswith("_tp")]
    if folds_found:
        agg_tn = int(sum(cv_summary.get(f"fold_{f}_tn", 0) for f in folds_found))
        agg_fp = int(sum(cv_summary.get(f"fold_{f}_fp", 0) for f in folds_found))
        agg_fn = int(sum(cv_summary.get(f"fold_{f}_fn", 0) for f in folds_found))
        agg_tp = int(sum(cv_summary.get(f"fold_{f}_tp", 0) for f in folds_found))

    eval_clf_metrics.log_confusion_matrix(
        categories=["No Failure (0)", "Will Fail (1)"],
        matrix=[[agg_tn, agg_fp], [agg_fn, agg_tp]],
    )

    fpr = 1.0 - recall_neg
    eval_clf_metrics.log_roc_curve(
        fpr=[0.0, round(fpr, 4), 1.0],
        tpr=[0.0, round(recall_pos, 4), 1.0],
        threshold=[1.0, 0.5, 0.0],
    )

    # ── Forward model to deploy component ────────────────────────────────
    approved_model.metadata["resource_name"] = resource_name if approved else ""
    approved_model.metadata["roc_auc"]       = roc_auc
    approved_model.metadata["pr_auc"]        = pr_auc
    approved_model.metadata["f2_score"]      = f2_score
    approved_model.metadata["recall_pos"]    = recall_pos
    approved_model.metadata["approved"]      = approved

    with open(approved_model.path, "w") as fh:
        json.dump({**summary, "approved": approved}, fh, indent=2)

    EvalOutputs = namedtuple("EvalOutputs", ["approved"])
    return EvalOutputs(approved=approved)


# ===========================================================================
# Component 4 – Deploy to Vertex AI Endpoint
# ===========================================================================

@component(
    base_image=BASE_IMAGE,
    packages_to_install=["google-cloud-aiplatform>=1.40"],
)
def deploy_component(
    approved_model: Input[Model],
    project_id: str,
    region: str,
    endpoint_display_name: str,
    machine_type: str,
):
    """
    Deploys the approved model to a Vertex AI Endpoint.

    • Reuses an existing endpoint with the same display_name if one exists,
      otherwise creates a new one.
    • 100% traffic is routed to the new model version.
    • The endpoint URL appears in Cloud Console → Vertex AI → Online prediction.
    """
    import logging
    import google.cloud.aiplatform as aip

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("deploy")

    resource_name = approved_model.metadata.get("resource_name", "")
    if not resource_name:
        logger.warning("No resource_name forwarded — skipping deploy.")
        return

    aip.init(project=project_id, location=region)

    # Reuse existing endpoint or create a new one
    existing = aip.Endpoint.list(
        filter   = f'display_name="{endpoint_display_name}"',
        order_by = "create_time desc",
        project  = project_id,
        location = region,
    )
    endpoint = existing[0] if existing else aip.Endpoint.create(
        display_name = endpoint_display_name,
        project      = project_id,
        location     = region,
    )
    logger.info(f"Endpoint: {endpoint.resource_name}")

    model = aip.Model(resource_name=resource_name)
    model.deploy(
        endpoint           = endpoint,
        machine_type       = machine_type,
        traffic_percentage = 100,
        sync               = True,
    )
    logger.info(f"Deployment complete → {endpoint.resource_name}")


# ===========================================================================
# Pipeline definition
# ===========================================================================

@dsl.pipeline(
    name        = "predictive-maintenance-xgboost-v3",
    description = (
        "Predictive maintenance: feature engineering → "
        "XGBoost (TimeSeriesSplit) → Vertex AI Experiment → "
        "Model Registry → conditional Endpoint deploy"
    ),
)
def predictive_maintenance_pipeline(
    project_id: str,
    bucket_name: str,
    region: str              = "us-central1",
    # Preprocessing
    target_window_size: int  = 7,  # days until failure to predict
    bq_dataset: str          = "preprocessed_data",
    bq_table: str            = "features_engineered_v2",
    # Training / Vertex AI
    vertex_experiment: str   = "pdm-xgboost-experiment",
    model_name: str          = "XGBoostPredictiveMaintenance",
    n_estimators: int        = 300,
    max_depth: int           = 6,
    learning_rate: float     = 0.05,
    subsample: float         = 0.8,
    colsample_bytree: float  = 0.8,
    n_cv_splits: int         = 5,
    # Multi-metric evaluation gate (all thresholds must pass)
    roc_auc_threshold: float    = 0.75,
    pr_auc_threshold: float     = 0.50,   # minority class P-R trade-off
    f2_score_threshold: float   = 0.40,   # recall-weighted gate (β=2)
    recall_pos_threshold: float = 0.60,   # minimum detection rate for failures
    # Deployment
    endpoint_display_name: str = "predictive-maintenance-endpoint",
    machine_type: str          = "n1-standard-4",
):
    # ── 1. Preprocessing ─────────────────────────────────────────────────
    preprocess_task = preprocess_component(
        project_id         = project_id,
        bucket_name        = bucket_name,
        target_window_size = target_window_size,
        bq_dataset         = bq_dataset,
        bq_table           = bq_table,
    ).set_display_name("Feature Engineering")

    # ── 2. Training ───────────────────────────────────────────────────────
    # No artifact passed from preprocess — training reads BigQuery directly.
    # .after(preprocess_task) ensures preprocess finishes (and writes to BQ)
    # before training starts.
    train_task = (
        train_component(
            project_id        = project_id,
            region            = region,
            bucket_name       = bucket_name,
            model_name        = model_name,
            vertex_experiment = vertex_experiment,
            bq_dataset        = bq_dataset,
            bq_table          = bq_table,
            n_estimators      = n_estimators,
            max_depth         = max_depth,
            learning_rate     = learning_rate,
            subsample         = subsample,
            colsample_bytree  = colsample_bytree,
            n_cv_splits       = n_cv_splits,
        )
        .after(preprocess_task)
        .set_display_name("XGBoost Training")
        .set_cpu_limit("4")
        .set_memory_limit("16G")
        .set_caching_options(False)
    )

    # ── 3. Evaluate ───────────────────────────────────────────────────────
    evaluate_task = (
        evaluate_component(
            input_model          = train_task.outputs["output_model"],
            project_id           = project_id,
            region               = region,
            roc_auc_threshold    = roc_auc_threshold,
            pr_auc_threshold     = pr_auc_threshold,
            f2_score_threshold   = f2_score_threshold,
            recall_pos_threshold = recall_pos_threshold,
        )
        .set_display_name("Evaluate & Quality Gate")
        .set_caching_options(False)
    )

    # ── 4. Deploy (only when ALL gates pass) ──────────────────────────────
    # with dsl.If(
    #     evaluate_task.outputs["approved"] == True,  # noqa: E712
    #     name="model-approved",
    # ):
    #     deploy_component(
    #         approved_model        = evaluate_task.outputs["approved_model"],
    #         project_id            = project_id,
    #         region                = region,
    #         endpoint_display_name = endpoint_display_name,
    #         machine_type          = machine_type,
    #     ).set_display_name("Deploy to Vertex AI Endpoint").set_caching_options(False)


# ===========================================================================
# Compile & Submit helpers
# ===========================================================================

def compile_pipeline(output_path: str = "pipeline.yaml"):
    compiler.Compiler().compile(
        pipeline_func = predictive_maintenance_pipeline,
        package_path  = output_path,
    )
    print(f"Pipeline compiled → {output_path}")


def submit_pipeline(
    project: str,
    region: str,
    pipeline_root: str,
    pipeline_yaml: str       = "pipeline.yaml",
    bucket_name: str         = "",
    roc_auc_threshold: float    = 0.75,
    pr_auc_threshold: float     = 0.50,
    f2_score_threshold: float   = 0.40,
    recall_pos_threshold: float = 0.60,
    vertex_experiment: str   = "pdm-xgboost-experiment",
):
    aip.init(project=project, location=region)

    job = aip.PipelineJob(
        display_name     = "predictive-maintenance-xgboost-v3",
        template_path    = pipeline_yaml,
        pipeline_root    = pipeline_root,
        # enable_caching   = True,
        parameter_values = {
            "project_id":            project,
            "bucket_name":           bucket_name,
            "region":                region,
            "roc_auc_threshold":     roc_auc_threshold,
            "pr_auc_threshold":      pr_auc_threshold,
            "f2_score_threshold":    f2_score_threshold,
            "recall_pos_threshold":  recall_pos_threshold,
            "vertex_experiment":     vertex_experiment,
        },
    )
    job.submit()
    print(f"Pipeline submitted → {job.resource_name}")
    return job


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile or run the KFP pipeline on Vertex AI"
    )
    parser.add_argument("--project",               required=True)
    parser.add_argument("--region",                default="us-central1")
    parser.add_argument("--pipeline-root",         required=False,
                        help="gs://<bucket>/pipeline-root  (required unless --compile-only)")
    parser.add_argument("--service-account",       required=False)
    parser.add_argument("--bucket-name",           required=True)
    parser.add_argument("--roc-auc-threshold",     type=float, default=0.75)
    parser.add_argument("--pr-auc-threshold",      type=float, default=0.50)
    parser.add_argument("--f2-score-threshold",    type=float, default=0.40)
    parser.add_argument("--recall-pos-threshold",  type=float, default=0.60)
    parser.add_argument("--vertex-experiment",     default="pdm-xgboost-experiment")
    parser.add_argument("--compile-only",          action="store_true")
    parser.add_argument("--output-yaml",           default="pipeline.yaml")
    args = parser.parse_args()

    compile_pipeline(output_path=args.output_yaml)

    if not args.compile_only:
        submit_pipeline(
            project              = args.project,
            region               = args.region,
            pipeline_root        = args.pipeline_root,
            pipeline_yaml        = args.output_yaml,
            bucket_name          = args.bucket_name,
            roc_auc_threshold    = args.roc_auc_threshold,
            pr_auc_threshold     = args.pr_auc_threshold,
            f2_score_threshold   = args.f2_score_threshold,
            recall_pos_threshold = args.recall_pos_threshold,
            vertex_experiment    = args.vertex_experiment,
        )