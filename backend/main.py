import logging
import joblib
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from model_pipeline import prepare_data, train_gbm, evaluate_model, save_model, load_model
import mlflow
import mlflow.sklearn

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI()

# Define paths
DATA_DIR = "data"
MODEL_DIR = "models"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data")
PREPARED_DATA_PATH = os.path.join(DATA_DIR, "prepared_data.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "GBM_model.joblib")
MLFLOW_TRACKING_URI = "http://mlflow:5000"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# MLflow setup
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("GBM_Experiment")

@app.get("/")
def home():
    return {"message": "Backend API is running"}

@app.post("/prepare_data")
async def prepare_data_endpoint(train_file: UploadFile = File(...), test_file: UploadFile = File(...)):
    try:
        train_path = os.path.join(RAW_DATA_DIR, "train.csv")
        test_path = os.path.join(RAW_DATA_DIR, "test.csv")
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        with open(train_path, "wb") as f:
            f.write(await train_file.read())
        with open(test_path, "wb") as f:
            f.write(await test_file.read())
        
        with mlflow.start_run(run_name="Data Preparation"):
            logging.info("Starting data preparation...")
            X_train, X_test, y_train, y_test = prepare_data(train_path, test_path)
            joblib.dump((X_train, X_test, y_train, y_test), PREPARED_DATA_PATH)
            mlflow.log_param("train_file", train_path)
            mlflow.log_param("test_file", test_path)
            mlflow.log_artifact(PREPARED_DATA_PATH)
            logging.info("Data preparation completed and saved.")
        return {"status": "Data prepared successfully"}
    except Exception as e:
        logging.error(f"Error in data preparation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
def train():
    try:
        with mlflow.start_run(run_name="Model Training"):
            logging.info("Training the Gradient Boosting Model...")
            X_train, X_test, y_train, y_test = joblib.load(PREPARED_DATA_PATH)
            gbm_model = train_gbm(X_train, y_train)
            save_model(gbm_model, MODEL_DIR)
            mlflow.sklearn.log_model(gbm_model, "model")
            mlflow.log_param("n_estimators", 120)
            mlflow.log_param("learning_rate", 0.08)
            logging.info("Model training completed and saved.")
        return {"status": "Model trained successfully"}
    except Exception as e:
        logging.error(f"Error in training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluate")
def evaluate():
    try:
        with mlflow.start_run(run_name="Model Evaluation"):
            logging.info("Evaluating the model...")
            model = load_model(MODEL_PATH)
            X_train, X_test, y_train, y_test = joblib.load(PREPARED_DATA_PATH)
            metrics = evaluate_model(model, X_test, y_test)
            mlflow.log_metrics({"accuracy": metrics["accuracy"], "roc_auc": metrics["roc_auc"]})
            mlflow.log_artifact(MODEL_PATH)
            logging.info("Model evaluation completed.")
        return metrics
    except Exception as e:
        logging.error(f"Evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        with mlflow.start_run(run_name="Prediction"):
            model = load_model(MODEL_PATH)
            df = pd.read_csv(file.file)
            X = df.drop(columns=["Churn"]) if "Churn" in df.columns else df
            predictions = model.predict(X)
            mlflow.log_param("prediction_file", file.filename)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
