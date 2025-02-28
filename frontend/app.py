import streamlit as st
import requests
import pandas as pd
import os

# Backend API URL (adjust based on your environment)
BACKEND_URL = "http://backend:8000" if st._is_running_with_streamlit else "http://localhost:8000"

# Selected features for prediction
SELECTED_FEATURES = [
    "Total day minutes",
    "International plan",
    "Customer service calls",
    "Total intl minutes",
    "Voice mail plan",
    "Number vmail messages"
]

# Initialize session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# Custom CSS for professional styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #2c3e50;
        color: white;
    }
    .css-1d391kg .css-1v0mbdj {
        color: white !important;
        font-weight: bold;
    }
    .css-1d391kg .css-1v0mbdj:hover {
        background-color: #34495e;
        border-radius: 5px;
    }
    /* Title styling */
    h1 {
        color: #2c3e50;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    /* Subheader styling */
    h2 {
        color: #34495e;
        font-family: 'Arial', sans-serif;
    }
    /* Button styling */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-family: 'Arial', sans-serif;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    /* Tabs styling */
    .stTabs [data-baseweb="tab"] {
        background-color: #ecf0f1;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        font-family: 'Arial', sans-serif;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #3498db;
        color: white;
    }
    /* Input fields styling */
    .stNumberInput, .stSelectbox, .stTextInput, .stFileUploader {
        background-color: #ffffff;
        border: 1px solid #bdc3c7;
        border-radius: 5px;
        padding: 5px;
    }
    /* Success and error messages */
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border-radius: 5px;
        padding: 10px;
        font-family: 'Arial', sans-serif;
    }
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        border-radius: 5px;
        padding: 10px;
        font-family: 'Arial', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)


# Login page
def login_page():
    st.markdown("<h1>Admin Login</h1>", unsafe_allow_html=True)
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        login_button = st.form_submit_button("Login")

        if login_button:
            if not username or not password:
                st.error("Please enter both username and password.")
                return
            data = {"username": username, "password": password}
            try:
                response = requests.post(f"{BACKEND_URL}/login", json=data)
                if response.status_code == 200:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("Login successful! Redirecting to dashboard...")
                    st.experimental_rerun()
                else:
                    st.error(f"Login failed: {response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Error logging in: {str(e)}")


# Dashboard page
def dashboard_page():
    st.markdown("<h1>Churn Prediction Dashboard</h1>", unsafe_allow_html=True)

    # Sidebar menu
    with st.sidebar:
        st.markdown(f"**Welcome, {st.session_state.username}!**", unsafe_allow_html=True)
        st.markdown("### Navigation Menu", unsafe_allow_html=True)
        menu = st.selectbox("Select an Action", [
            "Prepare Data",
            "Train Model",
            "Evaluate Model",
            "Predict",
            "Retrain Model",
            "Logout"
        ])

    # Main content based on menu selection
    if menu == "Logout":
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.success("Logged out successfully! Redirecting to login page...")
        st.experimental_rerun()

    elif menu == "Prepare Data":
        st.markdown("<h2>Prepare Data</h2>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Upload Train/Test Sets", "Manual Input"])
        
        with tab1:
            st.markdown("**Upload Train and Test Datasets**")
            train_file = st.file_uploader("Upload Train Dataset (CSV)", type="csv", key="train_file")
            test_file = st.file_uploader("Upload Test Dataset (CSV)", type="csv", key="test_file")

            if train_file and test_file:
                train_data = pd.read_csv(train_file)
                test_data = pd.read_csv(test_file)

                # Filter to selected features (plus Churn for train_data)
                train_columns = SELECTED_FEATURES + ["Churn"] if "Churn" in train_data.columns else SELECTED_FEATURES
                test_columns = SELECTED_FEATURES
                train_data = train_data[train_columns]
                test_data = test_data[test_columns]

                # Save filtered data to temporary files
                train_data.to_csv("temp_train.csv", index=False)
                test_data.to_csv("temp_test.csv", index=False)

                if st.button("Prepare Data (Upload)"):
                    with open("temp_train.csv", "rb") as train_f, open("temp_test.csv", "rb") as test_f:
                        files = {"train_file": train_f, "test_file": test_f}
                        try:
                            response = requests.post(f"{BACKEND_URL}/prepare_data", files=files)
                            if response.status_code == 200:
                                st.success("Data prepared successfully!")
                            else:
                                st.error(f"Error preparing data: {response.json().get('detail', 'Unknown error')}")
                        except Exception as e:
                            st.error(f"Error preparing data: {str(e)}")

        with tab2:
            st.markdown("**Enter Train and Test Data Manually**")
            st.warning("Manual input for preparing datasets is not supported in this version. Please use the upload option.")

    elif menu == "Train Model":
        st.markdown("<h2>Train Model</h2>", unsafe_allow_html=True)
        if st.button("Train Model"):
            try:
                response = requests.post(f"{BACKEND_URL}/train")
                if response.status_code == 200:
                    st.success("Model trained successfully!")
                else:
                    st.error(f"Error training model: {response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Error training model: {str(e)}")

    elif menu == "Evaluate Model":
        st.markdown("<h2>Evaluate Model</h2>", unsafe_allow_html=True)
        if st.button("Evaluate Model"):
            try:
                response = requests.get(f"{BACKEND_URL}/evaluate")
                if response.status_code == 200:
                    st.markdown("**Evaluation Metrics:**")
                    st.json(response.json())
                else:
                    st.error(f"Error evaluating model: {response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Error evaluating model: {str(e)}")

    elif menu == "Predict":
        st.markdown("<h2>Predict Churn</h2>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Upload Prediction Dataset", "Manual Input"])

        with tab1:
            st.markdown("**Upload Prediction Dataset**")
            predict_file = st.file_uploader("Upload Prediction Dataset (CSV)", type="csv", key="predict_file_upload")
            if predict_file:
                predict_data = pd.read_csv(predict_file)
                predict_data = predict_data[SELECTED_FEATURES]  # Filter to selected features
                predict_data.to_csv("temp_predict.csv", index=False)

                if st.button("Predict (Upload)"):
                    with open("temp_predict.csv", "rb") as pred_f:
                        files = {"file": pred_f}
                        try:
                            response = requests.post(f"{BACKEND_URL}/predict", files=files)
                            if response.status_code == 200:
                                predictions = response.json().get("predictions", [])
                                st.markdown("**Predictions:**")
                                st.json(predictions)
                            else:
                                st.error(f"Error predicting: {response.json().get('detail', 'Unknown error')}")
                        except Exception as e:
                            st.error(f"Error predicting: {str(e)}")

        with tab2:
            st.markdown("**Enter Values Manually for Prediction**")
            with st.form("manual_input_form"):
                total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0, value=90.0, help="Enter total daily call minutes")
                international_plan = st.selectbox("International Plan", ["No", "Yes"], help="Does the customer have an international plan?")
                customer_service_calls = st.number_input("Customer Service Calls", min_value=0, value=1, help="Number of calls to customer service")
                total_intl_minutes = st.number_input("Total International Minutes", min_value=0.0, value=6.0, help="Enter total international call minutes")
                voice_mail_plan = st.selectbox("Voice Mail Plan", ["No", "Yes"], help="Does the customer have a voicemail plan?")
                number_vmail_messages = st.number_input("Number of Voicemail Messages", min_value=0, value=0, help="Number of voicemail messages")

                submitted = st.form_submit_button("Predict")

                if submitted:
                    # Create a DataFrame with the input data
                    input_data = pd.DataFrame([{
                        "Total day minutes": total_day_minutes,
                        "International plan": international_plan,
                        "Customer service calls": customer_service_calls,
                        "Total intl minutes": total_intl_minutes,
                        "Voice mail plan": voice_mail_plan,
                        "Number vmail messages": number_vmail_messages
                    }])
                    input_data.to_csv("temp_manual_predict.csv", index=False)

                    with open("temp_manual_predict.csv", "rb") as pred_f:
                        files = {"file": pred_f}
                        try:
                            response = requests.post(f"{BACKEND_URL}/predict", files=files)
                            if response.status_code == 200:
                                predictions = response.json().get("predictions", [])
                                st.markdown("**Prediction Result:**")
                                st.json(predictions)
                            else:
                                st.error(f"Error predicting: {response.json().get('detail', 'Unknown error')}")
                        except Exception as e:
                            st.error(f"Error predicting: {str(e)}")

    elif menu == "Retrain Model":
        st.markdown("<h2>Retrain Model</h2>", unsafe_allow_html=True)
        if st.button("Retrain Model"):
            try:
                response = requests.post(f"{BACKEND_URL}/retrain")
                if response.status_code == 200:
                    st.success("Model retrained successfully!")
                else:
                    st.error(f"Error retraining model: {response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Error retraining model: {str(e)}")


# Display the appropriate page based on login status
if not st.session_state.logged_in:
    login_page()
else:
    dashboard_page()
