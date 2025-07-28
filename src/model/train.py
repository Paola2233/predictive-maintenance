import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import logging
import os


MODEL_NAME = os.getenv('MODEL_NAME', 'DecisionTreeModel')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlruns.db')

class DecisionTreeTrainer:
    def __init__(self, criterion:str='gini', max_depth:int=3, min_samples_leaf:int=1, min_samples_split:int=5, random_state:int=42):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        # Setup MLflow tracking and logging
        self.setup_mlflow()
        self.setup_logging()

    def setup_mlflow(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("Machine Failure Prediction")

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _load_data(self, data_path:str):
        """Load dataset from a CSV file."""
        try:
            data = pd.read_csv(data_path, parse_dates=['date']).sort_values(['machineID', 'date'])
            self.logger.info(f"Data loaded successfully from {data_path}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise e

    def _split_data(self, data:pd.DataFrame, target:str='will_fail_30_days', test_size:float=0.2):
        """Split data into training and testing sets."""
        try:
            X = data.drop(columns=['date', target])
            y = data[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state)
            self.logger.info("Data split into training and testing sets")

            # Normalize the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            return X_train_scaled, X_test_scaled, y_train, y_test
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            raise e

    def train_decision_tree(self):
        """Train a Decision Tree model and log it with MLflow."""

        # Load data
        data = self._load_data('data/processed/telemetry.csv')

        # Split data
        X_train, X_test, y_train, y_test = self._split_data(data)

        # Start MLflow run
        mlflow.sklearn.autolog()
        with mlflow.start_run() as run:
            model_tree = DecisionTreeClassifier(
                criterion = self.criterion,
                max_depth = self.max_depth,
                min_samples_leaf = self.min_samples_leaf,
                min_samples_split = self.min_samples_split,
                random_state = self.random_state
            )
            model_tree.fit(X_train, y_train)

            # Make predictions
            y_pred = model_tree.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            self.logger.info(classification_report(y_test, y_pred))

            # Log model with MLflow
            mlflow.log_param("criterion", self.criterion)
            mlflow.log_param("max_depth", self.max_depth)
            mlflow.log_param("min_samples_leaf", self.min_samples_leaf)
            mlflow.log_param("min_samples_split", self.min_samples_split)
            mlflow.log_metric('accuracy', report['accuracy'])
            mlflow.log_metric('recall_class_1', report['1.0']['recall'])
            mlflow.log_metric('recall_class_0', report['0.0']['recall'])
            mlflow.log_metric('f1_score_macro', report['macro avg']['f1-score'])
            mlflow.sklearn.log_model(model_tree,
                                     name="model",
                                     input_example=X_train,
                                     registered_model_name=MODEL_NAME)
            mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)
            self.logger.info(f"Model trained with f1_score_macro: {report['macro avg']['f1-score']:.4f}")
            self.logger.info(f"Model info: {run.info}")
            self.logger.info(f"Run ID: {run.info.run_id}")
            self.logger.info(f"Model URI: runs:/{run.info.run_id}/model")

            return run.info.run_id

if __name__ == "__main__":
    trainer = DecisionTreeTrainer()
    run_id = trainer.train_decision_tree()