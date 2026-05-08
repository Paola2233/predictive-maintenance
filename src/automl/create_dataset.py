from google.cloud import aiplatform


aiplatform.init(
    project="abiding-kingdom-491823-b0",
    location="us-east1",
)

# Or use BigQuery directly
dataset = aiplatform.TabularDataset.create(
    display_name="predictive_maintenance",
    bq_source="bq://abiding-kingdom-491823-b0.preprocessed_data.features_engineered",
)
