from sklearn.ensemble import GradientBoostingClassifier
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def train_gbm(X_train, y_train):
    try:
        gbm_model = GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, random_state=42)
        gbm_model.fit(X_train, y_train)
        logging.info("Gradient Boosting Model training completed.")
        return gbm_model
    except Exception as e:
        logging.error(f"Error in training GBM: {str(e)}")
        raise
