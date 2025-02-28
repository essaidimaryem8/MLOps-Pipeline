import joblib
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def save_model(model, model_dir):
    try:
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "GBM_model.joblib")
        joblib.dump(model, model_path)
        logging.info(f"Model saved at {model_path}")
    except Exception as e:
        logging.error(f"Error saving model: {str(e)}")
        raise


def load_model(model_path):
    try:
        model = joblib.load(model_path)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise
