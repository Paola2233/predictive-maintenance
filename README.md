# Microsoft Azure Predictive Maintenance

## Overview
This project implements a predictive maintenance system using machine learning, with integrated model management through MLflow. It provides a FastAPI-based REST API for real-time predictions and is designed to help anticipate industrial equipment failures. The deployed model predicts whether a machine is likely to fail within the next 30 days.

## Features
- Data ingestion and preprocessing
- Feature engineering and exploratory data analysis (EDA)
- Model training and evaluation
- MLflow integration for experiment tracking and model registry
- REST API (FastAPI) for real-time predictions
- Dockerized deployment

## Project Structure
```
├── data/                # Raw and processed datasets
├── notebooks/           # Jupyter notebooks for EDA, feature engineering, modeling
├── src/                 # Source code (API, model server, preprocessing)
├── requirements.txt     # Python dependencies
├── docker-compose.yml   # Docker orchestration
├── mlServerDockerfile   # Dockerfile for model server
├── mlFlowDockerfile     # Dockerfile for MLflow server
```

## Getting Started

### Prerequisites

* Python 3.12
* Docker & Docker Compose
* Jupyter Notebook (for running notebooks)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Paola2233/predictive-maintenance.git
   cd predictive-maintenance
   ```
2. (Recommended) Create and activate a virtual environment for local development:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start with Docker Compose:
   ```bash
   docker-compose up
   ```

## *(Optional)* Data Processing and Training Pipelines

The project includes scripts for processing data and training machine learning models. These pipelines can be executed from the command line.

### Run Data Processing Pipeline
To process the raw data and generate the processed dataset, run:
```bash
python src/model/preprocessing.py
```
This will read the raw data from the `data/raw/` directory and output processed CSV file to `data/processed/`.

### Run Model Training Pipeline
To train a new model and log it to MLflow, run:
```bash
python src/model/train.py
```
This will use the processed data, train the model, and register it in the MLflow tracking server.

## *(Optional)* Notebooks

The `notebooks/` directory contains Jupyter notebooks for exploratory data analysis (EDA), feature engineering, and model training. These notebooks provide step-by-step guidance and code for understanding the data, creating features, and building predictive models.

**Description:**
The notebooks are organized to guide you through the full predictive maintenance workflow:
- `01_EDA.ipynb`: Exploratory data analysis and initial insights.
- `02_feature_engineering.ipynb`: Feature creation and preprocessing.
- `03_classification_models.ipynb`: Model training, evaluation, and selection.

## Running the API
You can run the FastAPI server locally:
```bash
python src/app.py
```
The API will be available at `http://localhost:8080/docs`.

### API Endpoints
- `GET /health` — Check server and model status
- `POST /predict` — Get predictions (send JSON with input data)

#### Example: Predict with cURL

You can test the prediction endpoint with the following cURL command:

```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "machineID": 1,
      "volt": 167,
      "rotate": 440,
      "pressure": 98,
      "vibration": 40,
      "error_last_7_days": 0,
      "error_last_14_days": 0,
      "error_last_30_days": 0,
      "failure_last_7_days": 0,
      "failure_last_14_days": 0,
      "failure_last_30_days": 0,
      "maint_last_7_days": 0,
      "maint_last_14_days": 0,
      "maint_last_30_days": 1,
      "model": "model2",
      "age": 18
    }
  }'
```

## MLflow Tracking UI
To access the MLflow UI, run:
```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```
Then open [http://localhost:5000](http://localhost:5000) in your browser.

## License
This project is licensed under the MIT License.

