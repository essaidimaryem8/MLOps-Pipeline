import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from imblearn.under_sampling import NearMiss
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def prepare_data(train_path: str, test_path: str) -> tuple:
    """
    Load and preprocess the data.
    
    Args:
        train_path (str): Path to training data CSV
        test_path (str): Path to test data CSV
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    try:
        # Load datasets
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        logging.info("Data loaded successfully")

        # Separate features and target
        target_col = "Churn"
        X_train, y_train = df_train.drop(columns=[target_col]), df_train[target_col]
        X_test, y_test = df_test.drop(columns=[target_col]), df_test[target_col]

        # Handle categorical variables
        label_encoders = {}
        for col in X_train.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col])
            X_test[col] = le.transform(X_test[col])
            label_encoders[col] = le

        # Handle missing values using KNN imputer
        imputer = KNNImputer(n_neighbors=5)
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

        # Standardize numerical features
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        # Handle class imbalance using NearMiss
        nearmiss = NearMiss(version=3)
        X_train_balanced, y_train_balanced = nearmiss.fit_resample(X_train, y_train)

        logging.info("Data preparation completed successfully")
        return X_train_balanced, X_test, y_train_balanced, y_test

    except Exception as e:
        logging.error(f"Error in data preparation: {str(e)}")
        raise

def train_gbm(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingClassifier:
    """
    Train the Gradient Boosting Model (GBM).
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        
    Returns:
        GradientBoostingClassifier: Trained GBM model
    """
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

def evaluate_model(model: GradientBoostingClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate model performance.
    
    Args:
        model (GradientBoostingClassifier): Trained GBM model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        
    Returns:
        dict: Dictionary containing performance metrics
    """
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

def save_model(model: GradientBoostingClassifier, save_dir: str = "models") -> str:
    """
    Save trained model to disk.
    
    Args:
        model (GradientBoostingClassifier): Trained GBM model
        save_dir (str): Directory to save the model
        
    Returns:
        str: Path where model was saved
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "GBM_model.joblib")
        joblib.dump(model, save_path)
        logging.info(f"Model saved at {save_path}")
        return save_path

    except Exception as e:
        logging.error(f"Error in saving model: {str(e)}")
        raise

def load_model(model_path: str) -> GradientBoostingClassifier:
    """
    Load a saved GBM model from disk.
    
    Args:
        model_path (str): Path to saved model
        
    Returns:
        GradientBoostingClassifier: Loaded GBM model
    """
    try:
        model = joblib.load(model_path)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error in loading model: {str(e)}")
        raise
