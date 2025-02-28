import streamlit as st
import requests
import os

st.title("ML Pipeline Dashboard")

# Connect to FastAPI backend
backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")

# Upload files and trigger pipeline steps
st.header("Upload Training and Test Data")
train_file = st.file_uploader("Upload Training CSV", type="csv")
test_file = st.file_uploader("Upload Test CSV", type="csv")

if st.button("Prepare Data"):
    if train_file and test_file:
        files = {
            "train_file": train_file,
            "test_file": test_file
        }
        try:
            response = requests.post(f"{backend_url}/prepare_data", files=files)
            st.write(response.json())
        except Exception as e:
            st.error(f"Error preparing data: {str(e)}")
    else:
        st.error("Please upload both training and test files.")

# Buttons for pipeline steps
if st.button("Train Model"):
    try:
        response = requests.post(f"{backend_url}/train")
        st.write(response.json())
    except Exception as e:
        st.error(f"Error training model: {str(e)}")

if st.button("Evaluate Model"):
    try:
        response = requests.get(f"{backend_url}/evaluate")
        st.write(response.json())
    except Exception as e:
        st.error(f"Error evaluating model: {str(e)}")

if st.button("Predict"):
    prediction_file = st.file_uploader("Upload Prediction CSV", type="csv")
    if prediction_file:
        files = {"file": prediction_file}
        try:
            response = requests.post(f"{backend_url}/predict", files=files)
            st.write(response.json())
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")

if st.button("Retrain Model"):
    try:
        response = requests.post(f"{backend_url}/retrain")
        st.write(response.json())
    except Exception as e:
        st.error(f"Error retraining model: {str(e)}")

if st.button("Login"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if username and password:
        data = {"username": username, "password": password}
        try:
            response = requests.post(f"{backend_url}/login", json=data)
            st.write(response.json())
        except Exception as e:
            st.error(f"Error logging in: {str(e)}")
