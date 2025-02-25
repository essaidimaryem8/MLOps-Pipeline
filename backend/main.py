import logging
import joblib
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from model_pipeline.preprocessing import preprocess_data
from model_pipeline.training import train_gbm
from model_pipeline.evaluation import evaluate_model
from model_pipeline.io import save_model, load_model
import mlflow
import mlflow.sklearn
import pandas as pd
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI()

DATA_DIR = "data"
MODEL_DIR = "models"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data")
train_path = os.path.join(RAW_DATA_DIR, "churn-bigml-80.csv")
test_path = os.path.join(RAW_DATA_DIR, "churn-bigml-20.csv")
PREPARED_DATA_PATH = os.path.join(DATA_DIR, "prepared_data.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "GBM_model.joblib")
MLFLOW_TRACKING_URI = "http://mlflow:5000"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("GBM_Experiment")

@app.get("/")
def home():
    return {"message": "Backend API is running"}

@app.post("/prepare_data")
async def prepare_data_endpoint(train_file: UploadFile = File(...), test_file: UploadFile = File(...)):
    try:
        train_path_api = os.path.join(RAW_DATA_DIR, "churn-bigml-80.csv")
        test_path_api = os.path.join(RAW_DATA_DIR, "churn-bigml-20.csv")
        with open(train_path_api, "wb") as f:
            f.write(await train_file.read())
        with open(test_path_api, "wb") as f:
            f.write(await test_file.read())
        
        with mlflow.start_run(run_name="Data Preparation"):
            logging.info("Starting data preparation...")
            df_train = pd.read_csv(train_path_api)
            df_test = pd.read_csv(test_path_api)
            X_train, y_train, train_info = preprocess_data(df_train, is_train=True)
            X_test, y_test, _ = preprocess_data(df_test, is_train=False)
            joblib.dump((X_train, X_test, y_train, y_test, train_info), PREPARED_DATA_PATH)
            mlflow.log_param("train_file", train_path_api)
            mlflow.log_param("test_file", test_path_api)
            mlflow.log_artifact(PREPARED_DATA_PATH)
            logging.info("Data preparation completed and saved.")
        return {"status": "Data prepared successfully"}
    except Exception as e:
        logging.error(f"Error in data preparation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
def train():
    try:
        if not os.path.exists(PREPARED_DATA_PATH):
            raise HTTPException(status_code=400, detail="Prepared data not found. Run /prepare_data first.")
        
        with mlflow.start_run(run_name="Model Training"):
            logging.info("Training the Gradient Boosting Model...")
            X_train, X_test, y_train, y_test, _ = joblib.load(PREPARED_DATA_PATH)
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
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=400, detail="Model not found. Run /train first.")
        if not os.path.exists(PREPARED_DATA_PATH):
            raise HTTPException(status_code=400, detail="Prepared data not found. Run /prepare_data first.")
        
        with mlflow.start_run(run_name="Model Evaluation"):
            logging.info("Evaluating the model...")
            model = load_model(MODEL_PATH)
            X_train, X_test, y_train, y_test, _ = joblib.load(PREPARED_DATA_PATH)
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
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=400, detail="Model not found. Run /train first.")
        
        with mlflow.start_run(run_name="Prediction"):
            model = load_model(MODEL_PATH)
            df = pd.read_csv(file.file)
            X, _, _ = preprocess_data(df, is_train=False)
            predictions = model.predict(X)
            mlflow.log_param("prediction_file", file.filename)
            # Store predictions in a database (e.g., PostgreSQL) - optional for excellence
            # For now, return predictions
        return {"predictions": predictions.tolist()}
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")  # New endpoint for excellence
def retrain():
    try:
        if not os.path.exists(PREPARED_DATA_PATH):
            raise HTTPException(status_code=400, detail="Prepared data not found. Run /prepare_data first.")
        
        with mlflow.start_run(run_name="Model Retraining"):
            logging.info("Retraining the Gradient Boosting Model...")
            X_train, X_test, y_train, y_test, _ = joblib.load(PREPARED_DATA_PATH)
            gbm_model = train_gbm(X_train, y_train)
            save_model(gbm_model, MODEL_DIR)
            mlflow.sklearn.log_model(gbm_model, "retrained_model")
            mlflow.log_param("n_estimators", 120)
            mlflow.log_param("learning_rate", 0.08)
            logging.info("Model retraining completed and saved.")
        return {"status": "Model retrained successfully"}
    except Exception as e:
        logging.error(f"Error in retraining: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/login")
def login(data: dict):
    username = data.get("username")
    password = data.get("password")
    if username == "admin" and password == "password123":
        return {"status": "success", "message": "Login successful"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

def run_pipeline(args):
    try:
        if args.prepare_data or not any([args.prepare_data, args.train, args.evaluate, args.retrain]):
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                logging.error("Training or test data file not found. Please upload via the frontend first.")
                raise FileNotFoundError("Training or test data file not found.")
            with mlflow.start_run(run_name="Data Preparation"):
                logging.info("Starting data preparation...")
                X_train, X_test, y_train, y_test = prepare_data(train_path, test_path)
                joblib.dump((X_train, X_test, y_train, y_test), PREPARED_DATA_PATH)
                mlflow.log_param("train_file", train_path)
                mlflow.log_param("test_file", test_path)
                mlflow.log_artifact(PREPARED_DATA_PATH)
                logging.info("Data preparation completed and saved.")

        if args.train:
            if not os.path.exists(PREPARED_DATA_PATH):
                logging.error("Prepared data not found. Run --prepare_data first.")
                raise FileNotFoundError("Prepared data not found.")
            with mlflow.start_run(run_name="Model Training"):
                logging.info("Training the Gradient Boosting Model...")
                X_train, X_test, y_train, y_test = joblib.load(PREPARED_DATA_PATH)
                gbm_model = train_gbm(X_train, y_train)
                save_model(gbm_model, MODEL_DIR)
                mlflow.sklearn.log_model(gbm_model, "model")
                mlflow.log_param("n_estimators", 120)
                mlflow.log_param("learning_rate", 0.08)
                logging.info("Model training completed and saved.")

        if args.evaluate:
            if not os.path.exists(MODEL_PATH):
                logging.error("Model not found. Run --train first.")
                raise FileNotFoundError("Model not found.")
            if not os.path.exists(PREPARED_DATA_PATH):
                logging.error("Prepared data not found. Run --prepare_data first.")
                raise FileNotFoundError("Prepared data not found.")
            with mlflow.start_run(run_name="Model Evaluation"):
                logging.info("Evaluating the model...")
                model = load_model(MODEL_PATH)
                X_train, X_test, y_train, y_test = joblib.load(PREPARED_DATA_PATH)
                metrics = evaluate_model(model, X_test, y_test)
                print("\nModel Performance:")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"ROC AUC Score: {metrics['roc_auc']:.4f}")
                print("\nClassification Report:\n")
                print(metrics["classification_report"])
                mlflow.log_metrics({"accuracy": metrics["accuracy"], "roc_auc": metrics["roc_auc"]})
                mlflow.log_artifact(MODEL_PATH)
                logging.info("Model evaluation completed.")

        if args.retrain:
            if not os.path.exists(PREPARED_DATA_PATH):
                logging.error("Prepared data not found. Run --prepare_data first.")
                raise FileNotFoundError("Prepared data not found.")
            with mlflow.start_run(run_name="Model Retraining"):
                logging.info("Retraining the Gradient Boosting Model...")
                X_train, X_test, y_train, y_test = joblib.load(PREPARED_DATA_PATH)
                gbm_model = train_gbm(X_train, y_train)
                save_model(gbm_model, MODEL_DIR)
                mlflow.sklearn.log_model(gbm_model, "retrained_model")
                mlflow.log_param("n_estimators", 120)
                mlflow.log_param("learning_rate", 0.08)
                logging.info("Model retraining completed and saved.")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parts of the ML pipeline or start FastAPI server.")
    parser.add_argument("--prepare_data", action="store_true", help="Run data preparation.")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model.")
    parser.add_argument("--retrain", action="store_true", help="Retrain the model.")
    args = parser.parse_args()
    if any([args.prepare_data, args.train, args.evaluate, args.retrain]):
        run_pipeline(args)
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
