from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
import mlflow
import uvicorn
from model.MLflowModelServer import MLflowModelServer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlruns.db')
MODEL_NAME = os.getenv('MODEL_NAME', 'DecisionTreeModel')

model_server = MLflowModelServer()

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str = None
    model_version: int = None

class PredictionRequest(BaseModel):
    data: dict

class PredictionResponse(BaseModel):
    predictions: list
    status: str

@app.get("/health", response_model=HealthResponse)
def health():
    return {
        'status': 'healthy',
        'model_loaded': model_server.model is not None,
        'model_name': model_server.model_name,
        'model_version': model_server.model_version
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        if model_server.model is None:
            logger.error("Model not loaded")
            raise HTTPException(status_code=500, detail='Model not loaded')

        predictions = model_server.predict(request.data)

        return {
            'predictions': predictions,
            'status': 'success'
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
    try:
        model_server.load_model(
            model_name=MODEL_NAME
        )
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.warning(f"Could not load model on startup: {str(e)}")

    logger.info(f"Starting ML model server on localhost:8080...")
    uvicorn.run(app, host='0.0.0.0', port=8080)