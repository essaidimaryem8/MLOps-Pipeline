import streamlit as st
import requests
import json

st.set_page_config(page_title="ML App", layout="wide", initial_sidebar_state="expanded")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prepare Data", "Train Model", "Evaluate Model", "Predict", "Profile", "Login"])

backend_url = "http://backend:8000"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

def display_response(response, success_msg="Operation successful!", error_msg="Operation failed!"):
    if response.status_code == 200:
        st.success(success_msg)
        return response.json()
    else:
        st.error(f"{error_msg} (Error {response.status_code}: {response.text})")
        return None

if page == "Prepare Data":
    st.title("Prepare Data")
    st.markdown("Upload your training and test datasets here.")
    with st.form(key="prepare_data_form"):
        train_file = st.file_uploader("Upload Training Data", type=["csv"], key="train")
        test_file = st.file_uploader("Upload Test Data", type=["csv"], key="test")
        submit_button = st.form_submit_button(label="Prepare")
    if submit_button and train_file and test_file:
        files = {"train_file": train_file, "test_file": test_file}
        response = requests.post(f"{backend_url}/prepare_data", files=files)
        display_response(response, "Data prepared successfully!", "Data preparation failed.")

elif page == "Train Model":
    st.title("Train Model")
    st.markdown("Click below to train the Gradient Boosting Model.")
    if st.button("Train"):
        response = requests.post(f"{backend_url}/train")
        display_response(response, "Model trained successfully!", "Training failed.")

elif page == "Evaluate Model":
    st.title("Model Evaluation")
    st.markdown("Evaluate the performance of the trained model.")
    if st.button("Evaluate"):
        response = requests.get(f"{backend_url}/evaluate")
        if response.status_code == 200:
            metrics = response.json()
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            with col2:
                st.metric("ROC AUC Score", f"{metrics['roc_auc']:.4f}")
            st.subheader("Classification Report")
            st.text(metrics["classification_report"])
        else:
            st.error(f"Evaluation failed (Error {response.status_code}: {response.text})")

elif page == "Predict":
    st.title("Make Predictions")
    st.markdown("Upload a CSV file to get predictions from the trained model.")
    with st.form(key="predict_form"):
        input_data = st.file_uploader("Upload Data for Prediction", type=["csv"])
        predict_button = st.form_submit_button(label="Predict")
    if predict_button and input_data:
        response = requests.post(f"{backend_url}/predict", files={"file": input_data})
        if response.status_code == 200:
            predictions = response.json()["predictions"]
            st.subheader("Predictions")
            st.write(predictions)
        else:
            st.error(f"Prediction failed (Error {response.status_code}: {response.text})")

elif page == "Profile":
    st.title("User Profile")
    if st.session_state.logged_in:
        st.markdown(f"Welcome back, **{st.session_state.username}**!")
        st.subheader("Profile Details")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Username:", st.session_state.username)
        with col2:
            st.write("Status:", "Logged In")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.success("Logged out successfully!")
            st.experimental_rerun()
    else:
        st.warning("Please log in to view your profile.")
        st.markdown("Go to the **Login** page to sign in.")

elif page == "Login":
    st.title("Login Page")
    if st.session_state.logged_in:
        st.success(f"Already logged in as {st.session_state.username}.")
        if st.button("Go to Profile"):
            st.session_state.page = "Profile"
            st.experimental_rerun()
    else:
        st.markdown("Enter your credentials to log in.")
        with st.form(key="login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button(label="Login")
        if login_button:
            response = requests.post(f"{backend_url}/login", json={"username": username, "password": password})
            if response.status_code == 200 and response.json().get("status") == "success":
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful!")
                st.experimental_rerun()
            else:
                st.error("Login failed. Please check your credentials.")

st.sidebar.markdown("---")
st.sidebar.info("ML App | Powered by Streamlit")
