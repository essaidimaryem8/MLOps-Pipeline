import logging
import joblib
import os
import smtplib
from email.mime.text import MIMEText
from fastapi import FastAPI, File, UploadFile, HTTPException
from model_pipeline.preprocessing import preprocess_data
from model_pipeline.training import train_gbm
from model_pipeline.evaluation import evaluate_model
from model_pipeline.io import save_model, load_model
import mlflow
import mlflow.sklearn
import pandas as pd
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI()

# Define paths (absolute paths to match Docker volume mounts and test_main.py)
DATA_DIR = "/app/data"
MODEL_DIR = "/app/models"
RAW_DATA_DIR = "/app/data/raw_data"
TRAIN_PATH = "/app/data/raw_data/churn-bigml-80.csv"
TEST_PATH = "/app/data/raw_data/churn-bigml-20.csv"
PREPARED_DATA_PATH = "/app/data/prepared_data.joblib"
MODEL_PATH = "/app/models/GBM_model.joblib"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Selected features for the model
SELECTED_FEATURES = [
    "Total day minutes",
    "International plan",
    "Customer service calls",
    "Total intl minutes",
    "Voice mail plan",
    "Number vmail messages"
]

# Check if running in test mode (to disable MLflow)
IS_TESTING = os.getenv("TESTING", "false") == "true"

# Use MLFLOW_TRACKING_URI environment variable if set, otherwise determine based on environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if not MLFLOW_TRACKING_URI:
    if os.getenv("JENKINS", "false") == "true":
        MLFLOW_TRACKING_URI = "http://localhost:5000"  # MLflow server in Jenkins CI
    elif os.getenv("DOCKER", "false") == "true":
        MLFLOW_TRACKING_URI = "http://mlflow:5000"  # Docker Compose network
    else:
        MLFLOW_TRACKING_URI = "http://localhost:5000"  # Local development in WSL 2

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Only set up MLflow experiment if running the script directly and not in test mode
if __name__ == "__main__" and not IS_TESTING:
    mlflow.set_experiment("GBM_Experiment")


@app.get("/")
def home():
    return {"message": "Backend API is running"}


def send_email(subject, body, to_email):
    """Send an email notification with the given subject and body."""
    try:
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = os.getenv("SENDER_EMAIL", "maryemessaidi8@gmail.com")
        sender_password = os.getenv("EMAIL_PASSWORD")  # Ensure this is a Gmail App Password
        recipient_email = to_email

        if not sender_password:
            logging.error("EMAIL_PASSWORD environment variable not set.")
            return

        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = recipient_email

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        logging.info(f"Email sent to {recipient_email} with subject: {subject}")
    except Exception as e:
        logging.error(f"Failed to send email: {str(e)}")


@app.post("/prepare_data")
async def prepare_data_endpoint(train_file: UploadFile = File(...), test_file: UploadFile = File(...)):
    """
    Prepare training and test data for model training by preprocessing and saving it.
    """
    try:
        # Save uploaded files temporarily
        train_path_api = os.path.join(RAW_DATA_DIR, "churn-bigml-80.csv")
        test_path_api = os.path.join(RAW_DATA_DIR, "churn-bigml-20.csv")
        with open(train_path_api, "wb") as f:
            f.write(await train_file.read())
        with open(test_path_api, "wb") as f:
            f.write(await test_file.read())

        # Read and preprocess data
        train_data = pd.read_csv(train_path_api)
        test_data = pd.read_csv(test_path_api)
        # Filter to selected features (plus Churn for train_data)
        train_columns = SELECTED_FEATURES + ["Churn"] if "Churn" in train_data.columns else SELECTED_FEATURES
        test_columns = SELECTED_FEATURES
        train_data = train_data[train_columns]
        test_data = test_data[test_columns]
        # Preprocess the data
        X_train, X_test = preprocess_data(train_data, test_data)
        # Save the preprocessed data
        joblib.dump((X_train, X_test), PREPARED_DATA_PATH)

        # Skip MLflow logging during tests
        if not IS_TESTING:
            with mlflow.start_run(run_name="Data Preparation"):
                logging.info("Starting data preparation...")
                mlflow.log_param("train_file", train_file.filename)
                mlflow.log_param("test_file", test_file.filename)
                mlflow.log_artifact(PREPARED_DATA_PATH)
                mlflow.log_artifact(__file__, "code_artifact")
                mlflow.log_artifact("model_pipeline/preprocessing.py", "code_artifact")
                logging.info("Data preparation completed and saved.")
                run_id = mlflow.active_run().info.run_id
            send_email("Pipeline Step Completed: Prepare Data", f"Data preparation completed. Run ID: {run_id}", os.getenv("RECIPIENT_EMAIL", "maryem.essaidi@esprit.tn"))
        else:
            logging.info("Skipping MLflow logging during tests.")

        return {"status": "Data prepared successfully"}
    except Exception as e:
        logging.error(f"Error in data preparation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in data preparation: {str(e)}")


@app.post("/train")
def train():
    """
    Train the Gradient Boosting Model using the preprocessed data.
    """
    try:
        if not os.path.exists(PREPARED_DATA_PATH):
            raise HTTPException(status_code=400, detail="Prepared data not found. Run /prepare_data first.")

        # Skip MLflow logging during tests
        if not IS_TESTING:
            with mlflow.start_run(run_name="Model Training"):
                logging.info("Training the Gradient Boosting Model...")
                X_train, X_test = joblib.load(PREPARED_DATA_PATH)
                # Extract features and target
                y_train = X_train['Churn']
                X_train = X_train.drop('Churn', axis=1)
                gbm_model = train_gbm(X_train, y_train)
                save_model(gbm_model, MODEL_PATH)
                mlflow.sklearn.log_model(gbm_model, "model")
                mlflow.log_param("n_estimators", 120)
                mlflow.log_param("learning_rate", 0.08)
                mlflow.log_artifact(__file__, "code_artifact")
                mlflow.log_artifact("model_pipeline/training.py", "code_artifact")
                logging.info("Model training completed and saved.")
                run_id = mlflow.active_run().info.run_id
            send_email("Pipeline Step Completed: Train Model", f"Model training completed. Run ID: {run_id}", os.getenv("RECIPIENT_EMAIL", "maryem.essaidi@esprit.tn"))
        else:
            logging.info("Skipping MLflow logging during tests.")
            X_train, X_test = joblib.load(PREPARED_DATA_PATH)
            y_train = X_train['Churn']
            X_train = X_train.drop('Churn', axis=1)
            gbm_model = train_gbm(X_train, y_train)
            save_model(gbm_model, MODEL_PATH)

        return {"status": "Model trained successfully"}
    except Exception as e:
        logging.error(f"Error in training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in training: {str(e)}")


@app.get("/evaluate")
def evaluate():
    """
    Evaluate the trained model on the test set.
    """
    try:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=400, detail="Model not found. Run /train first.")
        if not os.path.exists(PREPARED_DATA_PATH):
            raise HTTPException(status_code=400, detail="Prepared data not found. Run /prepare_data first.")

        model = load_model(MODEL_PATH)
        X_train, X_test = joblib.load(PREPARED_DATA_PATH)
        # Since X_test doesn't have Churn (preprocessed), we need the original test data for evaluation
        test_data = pd.read_csv(TEST_PATH)
        test_data = test_data[SELECTED_FEATURES + ["Churn"]]
        # Pass test_data.drop('Churn', axis=1) as the first argument, and None as the second to avoid SMOTE
        _, X_test = preprocess_data(test_data.drop('Churn', axis=1), None)
        y_test = test_data['Churn'].astype(int)
        metrics = evaluate_model(model, X_test, y_test)

        # Skip MLflow logging during tests
        if not IS_TESTING:
            with mlflow.start_run(run_name="Model Evaluation"):
                logging.info("Evaluating the model...")
                mlflow.log_metrics({"accuracy": metrics["accuracy"], "roc_auc": metrics["roc_auc"]})
                mlflow.log_artifact(MODEL_PATH)
                mlflow.log_artifact(__file__, "code_artifact")
                mlflow.log_artifact("model_pipeline/evaluation.py", "code_artifact")
                logging.info("Model evaluation completed.")
                run_id = mlflow.active_run().info.run_id
            message = f"Model evaluation completed. Metrics: Accuracy={metrics['accuracy']:.4f}, ROC AUC={metrics['roc_auc']:.4f}. Run ID: {run_id}"
            send_email("Pipeline Step Completed: Evaluate Model", message, os.getenv("RECIPIENT_EMAIL", "maryem.essaidi@esprit.tn"))
        else:
            logging.info("Skipping MLflow logging during tests.")

        return metrics
    except Exception as e:
        logging.error(f"Evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """
    Predict churn for a new dataset using the trained model.
    """
    try:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=400, detail="Model not found. Run /train first.")

        model = load_model(MODEL_PATH)
        df = pd.read_csv(file.file)
        # Filter to only the selected features
        df = df[SELECTED_FEATURES]
        X, _ = preprocess_data(df)
        predictions = model.predict(X)
        df['Churn'] = predictions
        result = df.to_dict(orient="records")

        # Skip MLflow logging during tests
        if not IS_TESTING:
            with mlflow.start_run(run_name="Prediction"):
                logging.info("Predicting churn...")
                mlflow.log_param("prediction_file", file.filename)
                mlflow.log_artifact(__file__, "code_artifact")
                mlflow.log_artifact("model_pipeline/preprocessing.py", "code_artifact")
                run_id = mlflow.active_run().info.run_id
            send_email("Pipeline Step Completed: Predict", f"Predictions completed. Run ID: {run_id}", os.getenv("RECIPIENT_EMAIL", "maryem.essaidi@esprit.tn"))
        else:
            logging.info("Skipping MLflow logging during tests.")

        return {"predictions": result}
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/retrain")
def retrain():
    """
    Retrain the model with the preprocessed data.
    """
    try:
        if not os.path.exists(PREPARED_DATA_PATH):
            raise HTTPException(status_code=400, detail="Prepared data not found. Run /prepare_data first.")

        # Skip MLflow logging during tests
        if not IS_TESTING:
            with mlflow.start_run(run_name="Model Retraining"):
                logging.info("Retraining the Gradient Boosting Model...")
                X_train, X_test = joblib.load(PREPARED_DATA_PATH)
                # Extract features and target
                y_train = X_train['Churn']
                X_train = X_train.drop('Churn', axis=1)
                gbm_model = train_gbm(X_train, y_train)
                save_model(gbm_model, MODEL_PATH)
                mlflow.sklearn.log_model(gbm_model, "retrained_model")
                mlflow.log_param("n_estimators", 120)
                mlflow.log_param("learning_rate", 0.08)
                mlflow.log_artifact(__file__, "code_artifact")
                mlflow.log_artifact("model_pipeline/training.py", "code_artifact")
                logging.info("Model retraining completed and saved.")
                run_id = mlflow.active_run().info.run_id
            send_email("Pipeline Step Completed: Retrain Model", f"Model retraining completed. Run ID: {run_id}", os.getenv("RECIPIENT_EMAIL", "maryem.essaidi@esprit.tn"))
        else:
            logging.info("Skipping MLflow logging during tests.")
            X_train, X_test = joblib.load(PREPARED_DATA_PATH)
            y_train = X_train['Churn']
            X_train = X_train.drop('Churn', axis=1)
            gbm_model = train_gbm(X_train, y_train)
            save_model(gbm_model, MODEL_PATH)

        return {"status": "Model retrained successfully"}
    except Exception as e:
        logging.error(f"Error in retraining: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in retraining: {str(e)}")


@app.post("/login")
def login(data: dict):
    """
    Simple login endpoint for authentication.
    """
    username = data.get("username")
    password = data.get("password")
    if username == "admin" and password == "password123":
        return {"status": "success", "message": "Login successful"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")


def run_pipeline(args):
    """
    Run the ML pipeline based on command-line arguments.
    """
    try:
        if args.prepare_data or not any([args.prepare_data, args.train, args.evaluate, args.retrain]):
            if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
                logging.error("Training or test data file not found. Please upload via the frontend first.")
                raise FileNotFoundError("Training or test data file not found.")
            train_data = pd.read_csv(TRAIN_PATH)
            test_data = pd.read_csv(TEST_PATH)
            # Filter to selected features (plus Churn for train_data)
            train_columns = SELECTED_FEATURES + ["Churn"] if "Churn" in train_data.columns else SELECTED_FEATURES
            test_columns = SELECTED_FEATURES
            train_data = train_data[train_columns]
            test_data = test_data[test_columns]
            X_train, X_test = preprocess_data(train_data, test_data)
            joblib.dump((X_train, X_test), PREPARED_DATA_PATH)

            if not IS_TESTING:
                with mlflow.start_run(run_name="Data Preparation"):
                    logging.info("Starting data preparation...")
                    mlflow.log_param("train_file", TRAIN_PATH)
                    mlflow.log_param("test_file", TEST_PATH)
                    mlflow.log_artifact(PREPARED_DATA_PATH)
                    mlflow.log_artifact(__file__, "code_artifact")
                    mlflow.log_artifact("model_pipeline/preprocessing.py", "code_artifact")
                    logging.info("Data preparation completed and saved.")
                    run_id = mlflow.active_run().info.run_id
                send_email("Pipeline Step Completed: Prepare Data", f"Data preparation completed. Run ID: {run_id}", os.getenv("RECIPIENT_EMAIL", "maryem.essaidi@esprit.tn"))
            else:
                logging.info("Skipping MLflow logging during tests.")

        if args.train:
            if not os.path.exists(PREPARED_DATA_PATH):
                logging.error("Prepared data not found. Run --prepare_data first.")
                raise FileNotFoundError("Prepared data not found.")
            if not IS_TESTING:
                with mlflow.start_run(run_name="Model Training"):
                    logging.info("Training the Gradient Boosting Model...")
                    X_train, X_test = joblib.load(PREPARED_DATA_PATH)
                    y_train = X_train['Churn']
                    X_train = X_train.drop('Churn', axis=1)
                    gbm_model = train_gbm(X_train, y_train)
                    save_model(gbm_model, MODEL_PATH)
                    mlflow.sklearn.log_model(gbm_model, "model")
                    mlflow.log_param("n_estimators", 120)
                    mlflow.log_param("learning_rate", 0.08)
                    mlflow.log_artifact(__file__, "code_artifact")
                    mlflow.log_artifact("model_pipeline/training.py", "code_artifact")
                    logging.info("Model training completed and saved.")
                    run_id = mlflow.active_run().info.run_id
                send_email("Pipeline Step Completed: Train Model", f"Model training completed. Run ID: {run_id}", os.getenv("RECIPIENT_EMAIL", "maryem.essaidi@esprit.tn"))
            else:
                logging.info("Skipping MLflow logging during tests.")
                X_train, X_test = joblib.load(PREPARED_DATA_PATH)
                y_train = X_train['Churn']
                X_train = X_train.drop('Churn', axis=1)
                gbm_model = train_gbm(X_train, y_train)
                save_model(gbm_model, MODEL_PATH)

        if args.evaluate:
            if not os.path.exists(MODEL_PATH):
                logging.error("Model not found. Run --train first.")
                raise FileNotFoundError("Model not found.")
            if not os.path.exists(PREPARED_DATA_PATH):
                logging.error("Prepared data not found. Run --prepare_data first.")
                raise FileNotFoundError("Prepared data not found.")
            model = load_model(MODEL_PATH)
            X_train, X_test = joblib.load(PREPARED_DATA_PATH)
            test_data = pd.read_csv(TEST_PATH)
            test_data = test_data[SELECTED_FEATURES + ["Churn"]]
            _, X_test = preprocess_data(test_data, test_data.drop('Churn', axis=1))
            y_test = test_data['Churn'].astype(int)
            metrics = evaluate_model(model, X_test, y_test)
            if not IS_TESTING:
                with mlflow.start_run(run_name="Model Evaluation"):
                    logging.info("Evaluating the model...")
                    mlflow.log_metrics({"accuracy": metrics["accuracy"], "roc_auc": metrics["roc_auc"]})
                    mlflow.log_artifact(MODEL_PATH)
                    mlflow.log_artifact(__file__, "code_artifact")
                    mlflow.log_artifact("model_pipeline/evaluation.py", "code_artifact")
                    print("\nModel Performance:")
                    print(f"Accuracy: {metrics['accuracy']:.4f}")
                    print(f"ROC AUC Score: {metrics['roc_auc']:.4f}")
                    print("\nClassification Report:\n")
                    print(metrics["classification_report"])
                    logging.info("Model evaluation completed.")
                    run_id = mlflow.active_run().info.run_id
                message = f"Model evaluation completed. Metrics: Accuracy={metrics['accuracy']:.4f}, ROC AUC={metrics['roc_auc']:.4f}. Run ID: {run_id}"
                send_email("Pipeline Step Completed: Evaluate Model", message, os.getenv("RECIPIENT_EMAIL", "maryem.essaidi@esprit.tn"))
            else:
                logging.info("Skipping MLflow logging during tests.")

        if args.retrain:
            if not os.path.exists(PREPARED_DATA_PATH):
                logging.error("Prepared data not found. Run --prepare_data first.")
                raise FileNotFoundError("Prepared data not found.")
            if not IS_TESTING:
                with mlflow.start_run(run_name="Model Retraining"):
                    logging.info("Retraining the Gradient Boosting Model...")
                    X_train, X_test = joblib.load(PREPARED_DATA_PATH)
                    y_train = X_train['Churn']
                    X_train = X_train.drop('Churn', axis=1)
                    gbm_model = train_gbm(X_train, y_train)
                    save_model(gbm_model, MODEL_PATH)
                    mlflow.sklearn.log_model(gbm_model, "retrained_model")
                    mlflow.log_param("n_estimators", 120)
                    mlflow.log_param("learning_rate", 0.08)
                    mlflow.log_artifact(__file__, "code_artifact")
                    mlflow.log_artifact("model_pipeline/training.py", "code_artifact")
                    logging.info("Model retraining completed and saved.")
                    run_id = mlflow.active_run().info.run_id
                send_email("Pipeline Step Completed: Retrain Model", f"Model retraining completed. Run ID: {run_id}", os.getenv("RECIPIENT_EMAIL", "maryem.essaidi@esprit.tn"))
            else:
                logging.info("Skipping MLflow logging during tests.")
                X_train, X_test = joblib.load(PREPARED_DATA_PATH)
                y_train = X_train['Churn']
                X_train = X_train.drop('Churn', axis=1)
                gbm_model = train_gbm(X_train, y_train)
                save_model(gbm_model, MODEL_PATH)

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
