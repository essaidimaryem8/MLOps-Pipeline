import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from backend.main import app
from model_pipeline.preprocessing import preprocess_data
from model_pipeline.training import train_gbm
from model_pipeline.evaluation import evaluate_model
import pandas as pd

client = TestClient(app)

# Static mock data for testing (small lists of dictionaries with correct column names)
train_data = [
    {"Total day minutes": 100, "International plan": "No", "Customer service calls": 1,
     "Total intl minutes": 5, "Voice mail plan": "No", "Number vmail messages": 0, "Churn": False},
    {"Total day minutes": 150, "International plan": "Yes", "Customer service calls": 2,
     "Total intl minutes": 7, "Voice mail plan": "Yes", "Number vmail messages": 10, "Churn": True}
]

test_data = [
    {"Total day minutes": 80, "International plan": "No", "Customer service calls": 0,
     "Total intl minutes": 4, "Voice mail plan": "No", "Number vmail messages": 0}
]

predict_data = [
    {"Total day minutes": 90, "International plan": "No", "Customer service calls": 1,
     "Total intl minutes": 6, "Voice mail plan": "No", "Number vmail messages": 0}
]


# Helper function to convert list of dictionaries to DataFrame
def to_dataframe(data):
    return pd.DataFrame(data)


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
         patch("model_pipeline.training.train_gbm", return_value=MagicMock()) as mock_train, \
         patch("model_pipeline.io.save_model"), \
         patch("joblib.dump"), \
         patch("joblib.load", return_value=(train_df, test_df)):
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

    with patch("pandas.read_csv", side_effect=[train_df, test_df, train_df, test_df]), \
         patch("model_pipeline.preprocessing.preprocess_data", return_value=(train_df, test_df)), \
         patch("model_pipeline.training.train_gbm", return_value=MagicMock()) as mock_train, \
         patch("model_pipeline.io.save_model"), \
         patch("model_pipeline.io.load_model", return_value=MagicMock()), \
         patch("model_pipeline.evaluation.evaluate_model", return_value={"accuracy": 0.95}) as mock_evaluate, \
         patch("joblib.dump"), \
         patch("joblib.load", return_value=(train_df, test_df)):
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

    with patch("pandas.read_csv", side_effect=[train_df, test_df, predict_df]), \
         patch("model_pipeline.preprocessing.preprocess_data", side_effect=[(train_df, test_df), predict_df]), \
         patch("model_pipeline.training.train_gbm", return_value=MagicMock()) as mock_train, \
         patch("model_pipeline.io.save_model"), \
         patch("model_pipeline.io.load_model", return_value=MagicMock()) as mock_load, \
         patch("pandas.DataFrame.to_dict", return_value=[{"Churn": True}]) as mock_to_dict, \
         patch("joblib.dump"), \
         patch("joblib.load", return_value=(train_df, test_df)):
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
         patch("model_pipeline.training.train_gbm", return_value=MagicMock()) as mock_train, \
         patch("model_pipeline.io.save_model"), \
         patch("joblib.dump"), \
         patch("joblib.load", return_value=(train_df, test_df)):
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
