import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from imblearn.under_sampling import NearMiss
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def preprocess_data(df, target_col="Churn", is_train=True):
    """Preprocess data including encoding, imputation, scaling, and balancing."""
    try:
        X, y = df.drop(columns=[target_col]), df[target_col]

        # Handle categorical variables
        label_encoders = {}
        for col in X.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col]) if is_train else le.transform(X[col])
            label_encoders[col] = le

        # Handle missing values
        imputer = KNNImputer(n_neighbors=5)
        X = pd.DataFrame(imputer.fit_transform(X) if is_train else imputer.transform(X), columns=X.columns)

        # Standardize numerical features
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X) if is_train else scaler.transform(X), columns=X.columns)

        if is_train:
            # Handle class imbalance using NearMiss
            nearmiss = NearMiss(version=3)
            X_balanced, y_balanced = nearmiss.fit_resample(X, y)
            logging.info("Data preprocessing completed successfully with balancing")
            return X_balanced, y_balanced, (X, y, label_encoders, scaler)
        else:
            logging.info("Data preprocessing completed successfully")
            return X, y, None

    except Exception as e:
        logging.error(f"Error in preprocessing: {str(e)}")
        raise
