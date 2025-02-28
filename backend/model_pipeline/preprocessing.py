import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Selected features for the model
SELECTED_FEATURES = [
    "Total day minutes",
    "International plan",
    "Customer service calls",
    "Total intl minutes",
    "Voice mail plan",
    "Number vmail messages"
]

def preprocess_data(train_data: pd.DataFrame, test_data: pd.DataFrame = None):
    """
    Preprocess the data by selecting features, encoding categorical variables, and applying SMOTE.
    
    Args:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame, optional): Test dataset. If None, only preprocess train_data.
    
    Returns:
        tuple: (preprocessed train_data, preprocessed test_data) or (preprocessed train_data, None)
    """
    try:
        # Select only the specified features (and Churn for the target)
        features = SELECTED_FEATURES.copy()
        if 'Churn' in train_data.columns:
            features.append('Churn')
        train_data = train_data[features]
        if test_data is not None:
            test_data = test_data[SELECTED_FEATURES]

        # Encode categorical variables: 'International plan' and 'Voice mail plan'
        le = LabelEncoder()
        categorical_cols = ['International plan', 'Voice mail plan']
        
        for col in categorical_cols:
            if col in train_data.columns:
                train_data[col] = le.fit_transform(train_data[col])
                if test_data is not None and col in test_data.columns:
                    test_data[col] = le.transform(test_data[col])

        if 'Churn' in train_data.columns:
            # Encode the target variable 'Churn'
            train_data['Churn'] = train_data['Churn'].astype(int)

            # Separate features and target
            X_train = train_data.drop('Churn', axis=1)
            y_train = train_data['Churn']

            # Apply SMOTE to balance the dataset
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

            # Reconstruct the preprocessed training DataFrame
            train_data = pd.DataFrame(X_train, columns=SELECTED_FEATURES)
            train_data['Churn'] = y_train

        # Ensure test_data has the same columns as train_data (excluding Churn)
        if test_data is not None:
            test_data = test_data[SELECTED_FEATURES]

        logging.info("Data preprocessing completed successfully.")
        return train_data, test_data

    except Exception as e:
        logging.error(f"Error in preprocessing data: {str(e)}")
        raise
