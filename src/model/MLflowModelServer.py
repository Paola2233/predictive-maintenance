import warnings

import mlflow.tracking
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
import pandas as pd
import mlflow
import mlflow.pyfunc
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlruns.db')

class MLflowModelServer:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.model_version = None
        self.client = mlflow.MlflowClient()

        # Setup logging
        self.setup_logging()
        self.setup_mlflow()

    def setup_logging(self):
        # logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def setup_mlflow(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_registry_uri(MLFLOW_TRACKING_URI)

    def load_model(self, model_name=None):
        """Load the model from MLflow."""
        try:
            # Get all model versions
            all_versions = self.client.search_model_versions(f"name='{model_name}'")

            # Sort by version (as string, so convert to int)
            latest_version = max(all_versions, key=lambda x: int(x.version))
            model_version = latest_version.version
            model_uri = f"models:/{model_name}/{model_version}"

            # Load the model
            self.model = mlflow.pyfunc.load_model(model_uri)

            # Store model metadata
            self.model_name = model_name
            self.model_version = model_version
            self.logger.info(f"Model loaded: {model_name}, version: {model_version}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise e

    def predict(self, data):
        if self.model is None:
            raise Exception("Model not loaded")

        try:
            df = pd.DataFrame([data])
            # Model name preprocessing
            df['model'] = df['model'].replace({'model1': 0, 'model2': 1, 'model3': 2, 'model4': 3})

            # Normalize the features
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df)
            predictions = self.model.predict(df_scaled)

            if isinstance(predictions, np.ndarray):
                predictions = predictions.tolist()

            self.logger.info(f"Predictions made successfully: {predictions}")
            return predictions

        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            raise e