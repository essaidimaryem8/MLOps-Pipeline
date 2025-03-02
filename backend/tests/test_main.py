import pytest
from fastapi.testclient import TestClient
from backend.main import app

# Setup the FastAPI test client
client = TestClient(app)

def test_login():
    # Test successful login
    response = client.post("/login", json={"username": "admin", "password": "password123"})
    assert response.status_code == 200
    assert response.json() == {"status": "success", "message": "Login successful"}

    # Test failed login
    response = client.post("/login", json={"username": "wrong", "password": "wrong"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"
