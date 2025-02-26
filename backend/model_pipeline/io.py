import joblib
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def save_model(model, save_dir="models"):
    """Save trained model to disk."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "GBM_model.joblib")
        joblib.dump(model, save_path)
        logging.info(f"Model saved at {save_path}")
        return save_path
    except Exception as e:
        logging.error(f"Error in saving model: {str(e)}")
        raise


def load_model(model_path):
    """Load a saved GBM model from disk."""
    try:
        model = joblib.load(model_path)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error in loading model: {str(e)}")
        raise
