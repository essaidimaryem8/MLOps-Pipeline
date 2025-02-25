from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "classification_report": classification_report(y_test, y_pred),
        }
        logging.info("Model evaluation completed successfully")
        return metrics
    except Exception as e:
        logging.error(f"Error in model evaluation: {str(e)}")
        raise
