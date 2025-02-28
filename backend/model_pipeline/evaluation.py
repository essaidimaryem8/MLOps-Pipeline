from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        report = classification_report(y_test, y_pred)
        metrics = {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "classification_report": report
        }
        logging.info("Model evaluation completed.")
        return metrics
    except Exception as e:
        logging.error(f"Error in model evaluation: {str(e)}")
        raise
