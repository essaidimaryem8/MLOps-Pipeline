import pytest
import os
import sys
import pandas as pd
import numpy as np

# Adjust sys.path to include the backend directory relative to tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

try:
    from model_pipeline import train_gbm
except ModuleNotFoundError:
    pytest.exit("Error: model_pipeline.py not found or not importable.")

@pytest.fixture
def simple_training_data():
    X_train = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10]
    })
    y_train = pd.Series([0, 1, 0, 1, 0])
    return X_train, y_train

def test_train_model(simple_training_data):
    X_train, y_train = simple_training_data
    model = train_gbm(X_train, y_train)
    assert model is not None, "Model training failed"
