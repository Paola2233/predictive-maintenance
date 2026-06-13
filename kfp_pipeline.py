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
from kfp.dsl import Metrics, Model, Output, component, Input

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
) -> NamedTuple("TrainOutputs", [("roc_auc", float), ("f1_macro", float), ("artifact_uri", str)]):
    """
    Calls XGBoostTrainer.train():
      • Reads features directly from BigQuery (no local CSV)
      • TimeSeriesSplit cross-validation
      • Saves model.bst + scaler.pkl + feature_names.json to GCS
      • Logs params + metrics to a Vertex AI Experiment run
      • Registers the model in Vertex AI Model Registry
      • Returns CV metrics for the evaluation gate

    trainer.train() returns a dict — artifact_uri and resource_name are
    taken from that dict directly, since aiplatform.Model does not expose
    artifact_uri as an attribute after Model.upload().
    """
    import os
    import json
    from collections import namedtuple

    os.environ["GCP_PROJECT_ID"]    = project_id
    os.environ["GCP_REGION"]        = region
    os.environ["GCS_BUCKET_NAME"]   = bucket_name
    os.environ["MODEL_NAME"]        = model_name
    os.environ["VERTEX_EXPERIMENT"] = vertex_experiment
    os.environ["BQ_DATASET"]        = bq_dataset
    os.environ["BQ_TABLE"]          = bq_table

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

    # train() returns a dict: {vertex_model, resource_name, artifact_uri, cv_metrics}
    result = trainer.train()

    resource_name = result["resource_name"]
    artifact_uri  = result["artifact_uri"]
    cv_metrics    = result["cv_metrics"]
    roc_auc       = float(cv_metrics.get("cv_roc_auc", 0.0))
    f1_macro      = float(cv_metrics.get("cv_f1_macro", 0.0))

    # Surface metrics in the KFP / Vertex AI Pipelines UI
    output_metrics.log_metric("cv_roc_auc", roc_auc)
    output_metrics.log_metric("cv_f1_macro", f1_macro)
    output_metrics.log_metric("cv_recall_1", float(cv_metrics.get("cv_recall_1", 0.0)))

    # Write JSON summary so evaluate_component can read it
    summary = {
        "resource_name": resource_name,
        "artifact_uri":  artifact_uri,
        "roc_auc":       roc_auc,
        "f1_macro":      f1_macro,
    }
    with open(output_model.path, "w") as fh:
        json.dump(summary, fh)

    output_model.metadata["resource_name"] = resource_name
    output_model.metadata["artifact_uri"]  = artifact_uri

    TrainOutputs = namedtuple("TrainOutputs", ["roc_auc", "f1_macro", "artifact_uri"])
    return TrainOutputs(roc_auc=roc_auc, f1_macro=f1_macro, artifact_uri=artifact_uri)


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
    approved_model: Output[Model],
) -> NamedTuple("EvalOutputs", [("approved", bool)]):
    """
    Quality gate: reads the CV ROC-AUC from the train component's JSON summary.
    If it meets the threshold, the model resource name is forwarded to the
    deploy component.  Otherwise the deploy step is skipped via dsl.If().

    The model is already registered in Vertex AI Model Registry at this point
    (the train component did it). This gate controls whether it gets *deployed*.
    """
    import json
    import logging
    from collections import namedtuple
    import google.cloud.aiplatform as aip

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("evaluate")

    with open(input_model.path) as fh:
        summary = json.load(fh)

    roc_auc       = float(summary.get("roc_auc", 0))
    resource_name = summary.get("resource_name", "")
    approved      = roc_auc >= roc_auc_threshold

    logger.info(
        f"ROC-AUC={roc_auc:.4f}  threshold={roc_auc_threshold}  "
        f"→ {'APPROVED ✓' if approved else 'REJECTED ✗'}"
    )

    # Forward metadata so deploy_component knows which model to deploy
    approved_model.metadata["resource_name"] = resource_name if approved else ""
    approved_model.metadata["roc_auc"]       = roc_auc
    approved_model.metadata["approved"]      = approved

    # Write the same summary so the artefact path is not empty
    with open(approved_model.path, "w") as fh:
        json.dump({**summary, "approved": approved}, fh)

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
        filter        = f'display_name="{endpoint_display_name}"',
        order_by      = "create_time desc",
        project       = project_id,
        location      = region,
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
    region: str             = "us-central1",
    # Preprocessing
    target_window_size: int = 30,
    bq_dataset: str         = "preprocessed_data",
    bq_table: str           = "features_engineered_v2",
    # Training / Vertex AI
    vertex_experiment: str  = "pdm-xgboost-experiment",
    model_name: str         = "XGBoostPredictiveMaintenance",
    n_estimators: int       = 300,
    max_depth: int          = 6,
    learning_rate: float    = 0.05,
    subsample: float        = 0.8,
    colsample_bytree: float = 0.8,
    n_cv_splits: int        = 5,
    # Evaluation gate
    roc_auc_threshold: float = 0.75,
    # Deployment
    endpoint_display_name: str = "predictive-maintenance-endpoint",
    machine_type: str          = "n1-standard-4",
):
    # ── 1. Preprocessing ────────────────────────────────────────────────
    preprocess_task = preprocess_component(
        project_id         = project_id,
        bucket_name        = bucket_name,
        target_window_size = target_window_size,
        bq_dataset         = bq_dataset,
        bq_table           = bq_table,
    ).set_display_name("Feature Engineering")

    # ── 2. Training ─────────────────────────────────────────────────────
    # No artifact passed from preprocess — training reads BigQuery directly.
    # .after(preprocess_task) ensures preprocess finishes (and writes to BQ)
    # before training starts.
    train_task = (
        train_component(
            project_id       = project_id,
            region           = region,
            bucket_name      = bucket_name,
            model_name       = model_name,
            vertex_experiment= vertex_experiment,
            bq_dataset       = bq_dataset,
            bq_table         = bq_table,
            n_estimators     = n_estimators,
            max_depth        = max_depth,
            learning_rate    = learning_rate,
            subsample        = subsample,
            colsample_bytree = colsample_bytree,
            n_cv_splits      = n_cv_splits,
        )
        .after(preprocess_task)          # explicit ordering, no artifact wire
        .set_display_name("XGBoost Training")
        .set_cpu_limit("4")
        .set_memory_limit("16G")
    )

    # ── 3. Evaluate ──────────────────────────────────────────────────────
    evaluate_task = evaluate_component(
        input_model       = train_task.outputs["output_model"],
        project_id        = project_id,
        region            = region,
        roc_auc_threshold = roc_auc_threshold,
    ).set_display_name("Evaluate & Quality Gate")

    # # ── 4. Deploy (only when approved) ───────────────────────────────────
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
    #     ).set_display_name("Deploy to Vertex AI Endpoint")


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
    # service_account: str,
    pipeline_yaml: str   = "pipeline.yaml",
    bucket_name: str     = "",
    roc_auc_threshold: float = 0.75,
    vertex_experiment: str   = "pdm-xgboost-experiment",
):
    aip.init(project=project, location=region)

    job = aip.PipelineJob(
        display_name     = "predictive-maintenance-xgboost-v3",
        template_path    = pipeline_yaml,
        pipeline_root    = pipeline_root,
        enable_caching   = True,
        parameter_values = {
            "project_id":        project,
            "bucket_name":       bucket_name,
            "region":            region,
            "roc_auc_threshold": roc_auc_threshold,
            "vertex_experiment": vertex_experiment,
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
    parser.add_argument("--project",           required=True)
    parser.add_argument("--region",            default="us-central1")
    parser.add_argument("--pipeline-root",     required=False,
                        help="gs://<bucket>/pipeline-root  (required unless --compile-only)")
    parser.add_argument("--service-account",   required=False)
    parser.add_argument("--bucket-name",       required=True)
    parser.add_argument("--roc-auc-threshold", type=float, default=0.75)
    parser.add_argument("--vertex-experiment", default="pdm-xgboost-experiment")
    parser.add_argument("--compile-only",      action="store_true")
    parser.add_argument("--output-yaml",       default="pipeline.yaml")
    args = parser.parse_args()

    compile_pipeline(output_path=args.output_yaml)

    if not args.compile_only:
        # if not args.pipeline_root or not args.service_account:
        #     parser.error("--pipeline-root and --service-account are required to submit.")
        submit_pipeline(
            project           = args.project,
            region            = args.region,
            pipeline_root     = args.pipeline_root,
            # service_account   = args.service_account,
            pipeline_yaml     = args.output_yaml,
            bucket_name       = args.bucket_name,
            roc_auc_threshold = args.roc_auc_threshold,
            vertex_experiment = args.vertex_experiment,
        )