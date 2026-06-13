## Migrating the Predictive Maintenance model to GCP + Vertex AI

This document describes an end-to-end infrastructure and migration plan to move the predictive maintenance pipeline from local / MLflow-based deployment to Google Cloud Platform (GCP) using Vertex AI and managed GCP services. It covers data extraction, processing, modeling, serving, monitoring, security and CI/CD, plus concise next steps.

### Requirements checklist
- Provide a cloud-native data ingestion and storage solution (batch and streaming). (Done: design below)
- Provide managed preprocessing and feature pipelines. (Done)
- Provide model training, experiment tracking and model registry on GCP/Vertex. (Done)
- Provide an online prediction endpoint and a method for batch predictions. (Done)
- Provide CI/CD for model builds, tests, and deployments. (Done)
- Provide monitoring, logging, explainability, and retraining plan. (Done)

### High-level architecture

- Data sources: existing raw CSVs (on-prem or local) and future telemetry streams from devices.
- Ingest: Cloud Storage (GCS) for batch; Pub/Sub for streaming telemetry/events.
- Processing & feature engineering: Dataflow (Apache Beam) or Cloud Run jobs for lightweight tasks. Optionally use Dataproc/Spark for heavy transformations.
- Feature store: Vertex Feature Store (recommended) or BigQuery as canonical feature table for training and serving.
- Training: Vertex AI Training (custom container) or Vertex Pipelines (Vertex SDK + Kubeflow Pipelines) to run training jobs and create repeatable pipelines.
- Model registry & experiments: Migrate MLflow artifacts into Vertex Model Registry and use Vertex Experiments for tracking (or continue MLflow on GCS/Cloud SQL if preferred). Store artifacts in Artifact Registry or GCS.
- Serving (online): Vertex AI Endpoint with deployed model (autoscaling, private network if required).
- Batch predictions: Vertex AI Batch Prediction jobs reading from GCS or BigQuery and writing outputs to GCS/BigQuery.
- Orchestration & CI/CD: Cloud Build + Cloud Deploy / GitHub Actions + Cloud Build; Vertex Pipelines for ML workflows.
- Observability: Cloud Logging, Cloud Monitoring, and Vertex AI Model Monitoring (drift, skew). Use Cloud Storage / BigQuery for prediction logs and feedback.

### Component details and choices

- Storage
  - Use GCS buckets for raw and processed artifacts: `gs://<project>-pdm-raw`, `gs://<project>-pdm-processed`, `gs://<project>-pdm-models`.
  - Use BigQuery as the analytics store and for large-scale joins/aggregations.

- Ingest
  - Batch: Upload historical CSVs to GCS and load into BigQuery using load jobs.
  - Streaming: Devices -> Pub/Sub -> Dataflow -> Feature Store/BigQuery.

- Processing & Feature Engineering
  - Implement Beam pipelines (Dataflow runner) for deterministic transformations used in training and serving.
  - Persist engineered features to Vertex Feature Store (online for low latency) and to BigQuery (offline store) for training.

- Feature Store vs BigQuery
  - Vertex Feature Store gives low-latency lookups for online prediction and consistent feature versioning.
  - Use BigQuery for large-scale model training and batch predictions.

- Model Training & Experiment Tracking
  - Create a reproducible training container (Docker) that uses the same code from `src/model/train.py` adapted to read from GCS/BigQuery.
  - Run training on Vertex AI Training (custom container) or Vertex Pipelines. Store model artifacts in GCS/Artifact Registry.
  - Migrate experiment metadata: either continue MLflow (self-hosted on Cloud SQL + GCS storage) or adopt Vertex experiments and model registry. Short term: keep MLflow and push artifacts to GCS; mid-term: migrate to Vertex Model Registry.

- Serving
  - Deploy the trained model to a Vertex AI Endpoint (supports autoscaling, versions, and A/B traffic splitting).
  - Use container inference if the model uses custom preprocessing; otherwise use prebuilt TensorFlow/Sklearn containers.
  - Secure endpoint with IAM and (optionally) VPC service controls and private endpoints.

- Batch Predictions
  - Use Vertex AI Batch Prediction reading from BigQuery or GCS and writing results to GCS/BigQuery.

- CI/CD and Automation
  - Build images with Cloud Build and store in Artifact Registry.
  - Use GitHub Actions or Cloud Build triggers to run unit tests, linters, and build training/serving images on push to `main`.
  - Use Vertex Pipelines to orchestrate data preprocessing → training → validation → deployment.

- Monitoring, Logging & Model Quality
  - Ship application logs to Cloud Logging; metrics to Cloud Monitoring.
  - Enable Vertex AI Model Monitoring: prediction drift, feature skew, label drift. Store monitoring data in BigQuery.
  - Set alerts for data/skew/drift and endpoint errors.

- Security & Compliance
  - Use IAM roles and least-privilege service accounts for Dataflow, Vertex, Cloud Build.
  - Use Customer-Managed Encryption Keys (CMEK) for sensitive buckets if required.
  - Enable VPC Service Controls and private IPs for Vertex endpoints if you need to restrict public access.

- Cost control
  - Use preemptible training workers for non-critical experiments.
  - Use autoscaling endpoints and set minimum/maximum nodes.
  - Regularly clean up old artifacts and model versions.

### Contract: inputs, outputs, error modes, success criteria

- Inputs: telemetry/error/failure/maintenance records in BigQuery/GCS; streaming events via Pub/Sub.
- Outputs: a Vertex AI Endpoint for online predictions, batch prediction outputs in BigQuery/GCS, model artifacts in GCS/Artifact Registry.
- Error modes: schema drift, missing features, training failures, deployment failures.
- Success criteria: reproducible pipeline (Data -> Model -> Deploy) with automated tests and successful online endpoint returning predictions with latency < target (e.g., 200ms) and monitored model quality (no major drift for N days).

### Edge cases and mitigations
- Missing data: implement default imputation at serving and robust feature checks in pipelines.
- Large spikes in traffic: autoscale endpoint and add rate-limiting/queueing via Cloud Run or API Gateway.
- Data schema changes: use schema validation in Dataflow; alert on schema changes and pause pipelines.

### Minimal migration plan (phased)

1) Preparation and inventory (1–2 weeks)
   - Inventory all scripts and dependencies in `src/model/*.py` and notebooks.
   - Move training data to GCS and BigQuery. Create GCS buckets.
   - Containerize training & inference (Dockerfiles). Create `Dockerfile.train` and `Dockerfile.serving` if needed.

2) Reproducible training on Vertex (1–2 weeks)
   - Build a training container, adapt `train.py` to read from GCS/BigQuery and write model to GCS.
   - Run a Vertex AI training job that produces a model artifact.
   - Validate results vs local baseline.

3) Feature store and preprocessing (2–3 weeks)
   - Implement Beam/Dataflow pipelines for preprocessing.
   - Populate Vertex Feature Store or BigQuery offline features.

4) Serving & endpoints (1 week)
   - Create an endpoint in Vertex AI, deploy the model.
   - Update `src/app.py` to call Vertex AI endpoint for production predictions, or deploy an API on Cloud Run that calls Vertex.

5) CI/CD and monitoring (2–3 weeks)
   - Add Cloud Build triggers, Artifact Registry, and Vertex Pipelines for full automation.
   - Enable Vertex Model Monitoring and Cloud alerts.

6) Production rollout and validation (ongoing)
   - Canary deployments, A/B testing, rollback strategies, automating periodic retraining.

### Concrete next steps (practical tasks)
1. Create a GCP project and enable APIs: Vertex AI, BigQuery, Dataflow, Pub/Sub, Cloud Build, Artifact Registry, IAM.
2. Create GCS buckets for raw/processed/models.
3. Add Dockerfiles for training and serving (use base python:3.12 image matching local dev).
4. Modify `src/model/train.py` to accept GCS/BigQuery URIs and save artifacts to GCS.
5. Containerize and run a test training job locally, then submit a Vertex AI training job.
6. Implement a minimal Vertex AI deployment: upload a trained model to GCS and create a Vertex Endpoint to serve it.
7. Update `src/app.py` to call Vertex AI Endpoint (use google-cloud-aiplatform client) and add feature lookup logic.

### Helpful references & commands

- Enable APIs:
```
gcloud services enable aiplatform.googleapis.com bigquery.googleapis.com dataflow.googleapis.com pubsub.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

- Build image to Artifact Registry (example):
```
gcloud builds submit --tag <LOCATION>-docker.pkg.dev/<PROJECT>/<REPOSITORY>/pdm-train:latest
```

- Submit Vertex training (example):
```
gcloud ai custom-jobs create --region=us-central1 --display-name=pdm-train --worker-pool-spec=machine-type=n1-standard-8,replica-count=1,container-image-uri=<IMAGE_URI>
```

### Quality gates and validation
- Unit tests: add tests for preprocessing, model training inputs/outputs. Run on Cloud Build.
- Smoke test: call Vertex endpoint with a small example payload and assert schema/latency.
- Monitoring: enable Vertex Model Monitoring (skew/drift) after first week of production traffic.

### Follow-ups and optional improvements
- Use Vertex Explainable AI to provide per-prediction attributions.
- Use Vertex Continuous Evaluation for ongoing model assessment.
- Implement cost-monitoring dashboards in Cloud Monitoring.
- Migrate MLflow experiments into Cloud SQL + GCS or adopt Vertex experiment tooling.

---

Completion summary
- This document provides an actionable migration blueprint from local/MLflow to GCP/Vertex AI covering ingestion, processing, training, serving, CI/CD, security, monitoring and phased next steps. Follow the concrete next steps above to begin the migration.
