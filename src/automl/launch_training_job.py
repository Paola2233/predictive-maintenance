from google.cloud import aiplatform


aiplatform.init(
    project="abiding-kingdom-491823-b0",
    location="us-east1",
)

# Load the dataset by display name
datasets = aiplatform.TabularDataset.list(
    filter='display_name="predictive_maintenance"'
)

if not datasets:
    raise ValueError(
        "Dataset 'predictive_maintenance' not found. "
        "Please run src/automl/create_dataset.py first."
    )

dataset = datasets[0]

# Launch AutoML tabular classification training
job = aiplatform.AutoMLTabularTrainingJob(
    display_name="failure_prediction_training_job",
    optimization_prediction_type="classification",
    optimization_objective="minimize-log-loss",
)

model = job.run(
    dataset=dataset,
    target_column="will_fail_30_days",
    training_fraction_split=0.8,
    validation_fraction_split=0.1,
    test_fraction_split=0.1,
    budget_milli_node_hours=1000,  # 1 node-hour
    model_display_name="failure_prediction_model",
    disable_early_stopping=False,
)