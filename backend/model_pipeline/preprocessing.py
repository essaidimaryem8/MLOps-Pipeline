import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from imblearn.under_sampling import NearMiss
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def preprocess_data(train_path, test_path=None, target_col="Churn", is_train=True):
    """Preprocess data including encoding, imputation, scaling, and balancing."""
    try:
        # Handle various input types
        if is_train:
            # Load data from files if paths provided
            if isinstance(train_path, str) and isinstance(test_path, str):
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)
                
                # Process training data
                X_train, y_train, preprocessors = _process_data(train_df, target_col, is_train=True)
                
                # Process test data using same preprocessors
                X_test, y_test, _ = _process_data(test_df, target_col, is_train=False, 
                                                 label_encoders=preprocessors[2], 
                                                 scaler=preprocessors[3])
                
                return X_train, X_test, y_train, y_test
            else:
                logging.error("When is_train=True, both train_path and test_path must be provided as strings")
                raise ValueError("Invalid input types for training mode")
        else:
            # Handle prediction case (single input)
            if isinstance(train_path, pd.DataFrame):
                df = train_path
            elif isinstance(train_path, str):
                df = pd.read_csv(train_path)
            else:
                raise ValueError("Input must be a DataFrame or a file path")
                
            X, y, _ = _process_data(df, target_col, is_train=False)
            return X, y, None
            
    except Exception as e:
        logging.error(f"Error in preprocessing: {str(e)}")
        raise


def _process_data(df, target_col="Churn", is_train=True, label_encoders=None, scaler=None):
    """Internal function to process a single dataframe"""
    X, y = df.drop(columns=[target_col]), df[target_col]
    
    # Handle categorical variables
    if is_train:
        label_encoders = {}
        for col in X.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
    else:
        # Use provided encoders
        for col in X.select_dtypes(include=["object"]).columns:
            if col in label_encoders:
                X[col] = label_encoders[col].transform(X[col])
    
    # Handle missing values
    imputer = KNNImputer(n_neighbors=5)
    X = pd.DataFrame(imputer.fit_transform(X) if is_train else imputer.transform(X), columns=X.columns)

    # Standardize numerical features
    if is_train:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    else:
        # Use provided scaler
        X = pd.DataFrame(scaler.transform(X), columns=X.columns)

    if is_train:
        # Handle class imbalance using NearMiss
        nearmiss = NearMiss(version=3)
        X_balanced, y_balanced = nearmiss.fit_resample(X, y)
        logging.info("Data preprocessing completed successfully with balancing")
        return X_balanced, y_balanced, (X, y, label_encoders, scaler)
    else:
        logging.info("Data preprocessing completed successfully")
        return X, y, None
