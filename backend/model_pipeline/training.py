from sklearn.ensemble import GradientBoostingClassifier
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_gbm(X_train, y_train):
    """Train the Gradient Boosting Model (GBM)."""
    try:
        model = GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.08, max_depth=4,
            min_samples_leaf=1, min_samples_split=4, subsample=0.9, random_state=42
        )
        model.fit(X_train, y_train)
        logging.info("Gradient Boosting Model trained successfully")
        return model
    except Exception as e:
        logging.error(f"Error in GBM training: {str(e)}")
        raise
