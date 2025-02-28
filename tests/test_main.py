import pytest
import os
from fastapi.testclient import TestClient
from backend.main import app, RAW_DATA_DIR, PREPARED_DATA_PATH, MODEL_PATH

client = TestClient(app)


# Minimal mock data for testing (2 rows for training, 1 for testing)
def create_mock_data():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    # Simplified training data
    with open(os.path.join(RAW_DATA_DIR, "churn-bigml-80.csv"), "w") as f:
        f.write("State,Account length,Area code,International plan,Voice mail plan,Number vmail messages,Total day minutes,Total day calls,Total day charge,Total eve minutes,Total eve calls,Total eve charge,Total night minutes,Total night calls,Total night charge,Total intl minutes,Total intl calls,Total intl charge,Customer service calls,Churn\n")
        f.write("KS,10,415,No,No,0,100,50,17,100,50,8.5,100,50,4.5,5,2,1.35,1,False\n")
        f.write("OH,20,510,Yes,Yes,10,150,60,25.5,120,60,10.2,120,60,5.4,7,3,1.89,2,True\n")
    # Simplified test data
    with open(os.path.join(RAW_DATA_DIR, "churn-bigml-20.csv"), "w") as f:
        f.write("State,Account length,Area code,International plan,Voice mail plan,Number vmail messages,Total day minutes,Total day calls,Total day charge,Total eve minutes,Total eve calls,Total eve charge,Total night minutes,Total night calls,Total night charge,Total intl minutes,Total intl calls,Total intl charge,Customer service calls,Churn\n")
        f.write("MO,30,408,No,No,0,80,40,13.6,80,40,6.8,80,40,3.6,4,1,1.08,0,False\n")


def cleanup():
    # Simple cleanup to remove generated files
    for file in [PREPARED_DATA_PATH, MODEL_PATH, os.path.join(RAW_DATA_DIR, "churn-bigml-80.csv"), os.path.join(RAW_DATA_DIR, "churn-bigml-20.csv")]:
        if os.path.exists(file):
            os.remove(file)
    if os.path.exists(RAW_DATA_DIR):
        os.rmdir(RAW_DATA_DIR)


# Test /prepare_data endpoint
def test_prepare_data():
    create_mock_data()
    with open(os.path.join(RAW_DATA_DIR, "churn-bigml-80.csv"), "rb") as train_file, \
         open(os.path.join(RAW_DATA_DIR, "churn-bigml-20.csv"), "rb") as test_file:
        response = client.post("/prepare_data", files={"train_file": train_file, "test_file": test_file})
    assert response.status_code == 200
    assert response.json()["status"] == "Data prepared successfully"
    cleanup()


# Test /train endpoint
def test_train():
    create_mock_data()
    # Prepare data first
    with open(os.path.join(RAW_DATA_DIR, "churn-bigml-80.csv"), "rb") as train_file, \
         open(os.path.join(RAW_DATA_DIR, "churn-bigml-20.csv"), "rb") as test_file:
        client.post("/prepare_data", files={"train_file": train_file, "test_file": test_file})
    # Train
    response = client.post("/train")
    assert response.status_code == 200
    assert response.json()["status"] == "Model trained successfully"
    cleanup()


# Test /evaluate endpoint
def test_evaluate():
    create_mock_data()
    # Prepare and train first
    with open(os.path.join(RAW_DATA_DIR, "churn-bigml-80.csv"), "rb") as train_file, \
         open(os.path.join(RAW_DATA_DIR, "churn-bigml-20.csv"), "rb") as test_file:
        client.post("/prepare_data", files={"train_file": train_file, "test_file": test_file})
    client.post("/train")
    # Evaluate
    response = client.get("/evaluate")
    assert response.status_code == 200
    assert "accuracy" in response.json()
    cleanup()


# Test /predict endpoint
def test_predict():
    create_mock_data()
    # Prepare and train first
    with open(os.path.join(RAW_DATA_DIR, "churn-bigml-80.csv"), "rb") as train_file, \
         open(os.path.join(RAW_DATA_DIR, "churn-bigml-20.csv"), "rb") as test_file:
        client.post("/prepare_data", files={"train_file": train_file, "test_file": test_file})
    client.post("/train")
    # Create a minimal prediction file
    with open(os.path.join(RAW_DATA_DIR, "prediction.csv"), "w") as f:
        f.write("State,Account length,Area code,International plan,Voice mail plan,Number vmail messages,Total day minutes,Total day calls,Total day charge,Total eve minutes,Total eve calls,Total eve charge,Total night minutes,Total night calls,Total night charge,Total intl minutes,Total intl calls,Total intl charge,Customer service calls\n")
        f.write("TX,40,510,No,No,0,90,45,15.3,90,45,7.65,90,45,4.05,6,2,1.62,1\n")
    with open(os.path.join(RAW_DATA_DIR, "prediction.csv"), "rb") as pred_file:
        response = client.post("/predict", files={"file": pred_file})
    assert response.status_code == 200
    assert "predictions" in response.json()
    cleanup()


# Test /retrain endpoint
def test_retrain():
    create_mock_data()
    # Prepare and train first
    with open(os.path.join(RAW_DATA_DIR, "churn-bigml-80.csv"), "rb") as train_file, \
         open(os.path.join(RAW_DATA_DIR, "churn-bigml-20.csv"), "rb") as test_file:
        client.post("/prepare_data", files={"train_file": train_file, "test_file": test_file})
    client.post("/train")
    # Retrain
    response = client.post("/retrain")
    assert response.status_code == 200
    assert response.json()["status"] == "Model retrained successfully"
    cleanup()


# Test /login endpoint
def test_login():
    response = client.post("/login", json={"username": "admin", "password": "password123"})
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    response = client.post("/login", json={"username": "wrong", "password": "wrong"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"
