# Customer Churn Prediction MLOps Pipeline

A complete machine learning pipeline for predicting customer churn in the telecommunications industry, built with FastAPI, Streamlit, MLflow, Docker, Jenkins, Elasticsearch, and Kibana.

## Project Overview

This project implements an end-to-end machine learning pipeline for predicting whether customers will leave a telecom service. It includes data preparation, model training, evaluation, and prediction capabilities through both a web interface and API.

## Key Features

- Data preparation and preprocessing
- Gradient Boosting Model for churn prediction
- Model evaluation with accuracy and ROC AUC metrics
- User-friendly Streamlit frontend
- FastAPI backend API
- MLflow for experiment tracking
- Jenkins CI/CD pipeline
- Docker containerization
- Elasticsearch and Kibana monitoring

## Quick Start

### Prerequisites

- WSL 2 or Linux environment
- Docker and Docker Compose
- Jenkins (for CI/CD)
- Git
- Gmail account with App Password for email notifications

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/essaidimaryem/ML-Pipeline.git
   cd ML-Pipeline
   ```

2. **Set Environment Variables**:
   Create a `.env` file with:
   ```
   SENDER_EMAIL=your-email@gmail.com
   EMAIL_PASSWORD=your-app-password
   RECIPIENT_EMAIL=notification-recipient@example.com
   ```

3. **Build and Start Services**:
   ```bash
   docker-compose build
   docker-compose up -d
   ```

4. **Access the Applications**:
   - Frontend: http://localhost:8501
   - MLflow: http://localhost:5001
   - Kibana: http://localhost:5601

## Usage

1. **Login** to the Streamlit frontend (default: admin/password123)
2. **Upload data** or enter it manually
3. **Train the model** using the prepared data
4. **Evaluate** model performance
5. **Make predictions** on new customer data

## API Endpoints

- `/login` - Authenticate user
- `/prepare_data` - Process training and test data
- `/train` - Train the model
- `/evaluate` - Get model performance metrics
- `/predict` - Predict churn for new data
- `/retrain` - Update the model with new data

## Project Structure

```
ML-Pipeline/
├── backend/                # FastAPI backend service
│   ├── model_pipeline/     # ML model components
│   └── tests/              # Backend tests
├── frontend/               # Streamlit frontend
├── data/                   # Data storage
├── docker-compose.yml      # Service orchestration
├── Dockerfile.dependencies # Base dependencies image
├── Dockerfile.mlflow       # MLflow server image
└── Jenkinsfile             # CI/CD pipeline definition
```

## CI/CD Pipeline

The Jenkins pipeline automatically:
- Builds and updates Docker images
- Runs code quality checks
- Executes tests
- Deploys all services
