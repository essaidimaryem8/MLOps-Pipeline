import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def preprocess_data(data, test_data=None, is_train=True):
    try:
        if isinstance(data, pd.DataFrame):
            train_df = data
        else:
            train_df = pd.read_csv(data)
        test_df = pd.read_csv(test_data) if test_data else None

        # Define categorical and numerical columns
        categorical_cols = train_df.select_dtypes(include=['object', 'bool']).columns.tolist()
        numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Remove target column from features
        if 'Churn' in categorical_cols:
            categorical_cols.remove('Churn')
        if 'Churn' in numerical_cols:
            numerical_cols.remove('Churn')

        # Preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
            ])

        # Apply preprocessing
        if is_train:
            X_train = train_df.drop(columns=['Churn'])
            y_train = train_df['Churn'].astype(int)

            # Apply SMOTE only to training data
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42))
            ])
            X_train_processed, y_train = pipeline.named_steps['smote'].fit_resample(
                pipeline.named_steps['preprocessor'].fit_transform(X_train), y_train
            )
            logging.info("Data preprocessing completed successfully with balancing")
            return X_train_processed, None, y_train, None
        else:
            X = preprocessor.fit_transform(data)
            logging.info("Data preprocessing completed successfully")
            return X, None, None

    except Exception as e:
        logging.error(f"Error in data preprocessing: {str(e)}")
        raise
