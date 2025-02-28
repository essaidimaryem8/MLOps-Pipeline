import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from backend.main import app, PREPARED_DATA_PATH, MODEL_PATH, TEST_PATH
from model_pipeline.preprocessing import preprocess_data
from model_pipeline.training import train_gbm
from model_pipeline.evaluation import evaluate_model
import pandas as pd

client = TestClient(app)

# Static mock data for testing (small lists of dictionaries with pre-encoded categorical columns)
train_data = [
    {"Total day minutes": 100, "International plan": 0, "Customer service calls": 1,
     "Total intl minutes": 5, "Voice mail plan": 0, "Number vmail messages": 0, "Churn": False},
    {"Total day minutes": 150, "International plan": 1, "Customer service calls": 2,
     "Total intl minutes": 7, "Voice mail plan": 1, "Number vmail messages": 10, "Churn": True}
]

# Updated test_data to have two samples with different Churn values to satisfy SMOTE
test_data = [
    {"Total day minutes": 80, "International plan": 0, "Customer service calls": 0,
     "Total intl minutes": 4, "Voice mail plan": 0, "Number vmail messages": 0, "Churn": False},
    {"Total day minutes": 85, "International plan": 0, "Customer service calls": 1,
     "Total intl minutes": 5, "Voice mail plan": 0, "Number vmail messages": 0, "Churn": True}
]

predict_data = [
    {"Total day minutes": 90, "International plan": 0, "Customer service calls": 1,
     "Total intl minutes": 6, "Voice mail plan": 0, "Number vmail messages": 0}
]


# Helper function to convert list of dictionaries to DataFrame
def to_dataframe(data):
    return pd.DataFrame(data)


# Create a mock model with a predict method
mock_model = MagicMock()
mock_model.predict.return_value = [True]  # Default return value for other tests


# Helper function to determine what joblib.load should return based on the path
def joblib_load_side_effect(path):
    if path == PREPARED_DATA_PATH:
        return (to_dataframe(train_data), to_dataframe(test_data))
    elif path == MODEL_PATH:
        return mock_model
    else:
        raise ValueError(f"Unexpected path in joblib.load: {path}")


# Test /prepare_data endpoint
def test_prepare_data():
    # Convert static data to DataFrames
    train_df = to_dataframe(train_data)
    test_df = to_dataframe(test_data)

    # Mock pd.read_csv and joblib.dump
    with patch("pandas.read_csv", side_effect=[train_df, test_df]), \
         patch("model_pipeline.preprocessing.preprocess_data", return_value=(train_df, test_df)) as mock_preprocess, \
         patch("joblib.dump") as mock_dump:
        response = client.post("/prepare_data", files={
            "train_file": ("churn-bigml-80.csv", b"mock_train_data", "text/csv"),
            "test_file": ("churn-bigml-20.csv", b"mock_test_data", "text/csv")
        })
    assert response.status_code == 200
    assert response.json()["status"] == "Data prepared successfully"


# Test /train endpoint
def test_train():
    # Prepare data first
    train_df = to_dataframe(train_data)
    test_df = to_dataframe(test_data)

    with patch("pandas.read_csv", side_effect=[train_df, test_df]), \
         patch("model_pipeline.preprocessing.preprocess_data", return_value=(train_df, test_df)), \
         patch("model_pipeline.training.train_gbm", return_value=mock_model) as mock_train, \
         patch("model_pipeline.io.save_model"), \
         patch("joblib.dump"), \
         patch("joblib.load", side_effect=joblib_load_side_effect):
        client.post("/prepare_data", files={
            "train_file": ("churn-bigml-80.csv", b"mock_train_data", "text/csv"),
            "test_file": ("churn-bigml-20.csv", b"mock_test_data", "text/csv")
        })
        response = client.post("/train")
    assert response.status_code == 200
    assert response.json()["status"] == "Model trained successfully"


# Test /evaluate endpoint
def test_evaluate():
    train_df = to_dataframe(train_data)
    test_df = to_dataframe(test_data)

    # Ensure preprocess_data returns consistent sample sizes for X_test and y_test
    eval_test_df = to_dataframe(test_data)  # Same as test_df, used for evaluation

    # Set mock_model.predict to return predictions matching the number of samples in X_test
    mock_model.predict.return_value = [True, False]  # Matches the 2 samples in test_df

    with patch("pandas.read_csv", side_effect=[train_df, test_df, test_df]), \
         patch("model_pipeline.preprocessing.preprocess_data", side_effect=[(train_df, test_df), (None, eval_test_df)]), \
         patch("model_pipeline.training.train_gbm", return_value=mock_model) as mock_train, \
         patch("model_pipeline.io.save_model"), \
         patch("model_pipeline.io.load_model", return_value=mock_model), \
         patch("joblib.dump"), \
         patch("joblib.load", side_effect=joblib_load_side_effect):
        client.post("/prepare_data", files={
            "train_file": ("churn-bigml-80.csv", b"mock_train_data", "text/csv"),
            "test_file": ("churn-bigml-20.csv", b"mock_test_data", "text/csv")
        })
        client.post("/train")
        response = client.get("/evaluate")
    assert response.status_code == 200
    assert "accuracy" in response.json()
    assert response.json()["accuracy"] == 0.95


# Test /predict endpoint
def test_predict():
    train_df = to_dataframe(train_data)
    test_df = to_dataframe(test_data)
    predict_df = to_dataframe(predict_data)

    # Update predict return value to match predict_data size (1 sample)
    mock_model.predict.return_value = [True]

    with patch("pandas.read_csv", side_effect=[train_df, test_df, predict_df]), \
         patch("model_pipeline.preprocessing.preprocess_data", side_effect=[(train_df, test_df), predict_df]), \
         patch("model_pipeline.training.train_gbm", return_value=mock_model) as mock_train, \
         patch("model_pipeline.io.save_model"), \
         patch("model_pipeline.io.load_model", return_value=mock_model), \
         patch("pandas.DataFrame.to_dict", return_value=[{"Churn": True}]) as mock_to_dict, \
         patch("joblib.dump"), \
         patch("joblib.load", side_effect=joblib_load_side_effect):
        client.post("/prepare_data", files={
            "train_file": ("churn-bigml-80.csv", b"mock_train_data", "text/csv"),
            "test_file": ("churn-bigml-20.csv", b"mock_test_data", "text/csv")
        })
        client.post("/train")
        response = client.post("/predict", files={
            "file": ("prediction.csv", b"mock_predict_data", "text/csv")
        })
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert response.json()["predictions"] == [{"Churn": True}]


# Test /retrain endpoint
def test_retrain():
    train_df = to_dataframe(train_data)
    test_df = to_dataframe(test_data)

    with patch("pandas.read_csv", side_effect=[train_df, test_df]), \
         patch("model_pipeline.preprocessing.preprocess_data", return_value=(train_df, test_df)), \
         patch("model_pipeline.training.train_gbm", return_value=mock_model) as mock_train, \
         patch("model_pipeline.io.save_model"), \
         patch("joblib.dump"), \
         patch("joblib.load", side_effect=joblib_load_side_effect):
        client.post("/prepare_data", files={
            "train_file": ("churn-bigml-80.csv", b"mock_train_data", "text/csv"),
            "test_file": ("churn-bigml-20.csv", b"mock_test_data", "text/csv")
        })
        client.post("/train")
        response = client.post("/retrain")
    assert response.status_code == 200
    assert response.json()["status"] == "Model retrained successfully"


# Test /login endpoint
def test_login():
    response = client.post("/login", json={"username": "admin", "password": "password123"})
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    response = client.post("/login", json={"username": "wrong", "password": "wrong"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"
