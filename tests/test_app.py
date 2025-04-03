import pickle
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import app


# Create a test client
client = TestClient(app)


class DummyModel:
    def __init__(self, predictions, probabilities):
        self.predictions = predictions
        self.probabilities = probabilities
    
    def predict(self, x):
        return self.predictions
    
    def predict_proba(self, x):
        return self.probabilities


@pytest.fixture
def mock_model():
    """Fixture to create a dummy model"""
    return DummyModel(
        predictions=np.array([0]),
        probabilities=np.array([[0.8, 0.2]])
    )


def test_root_endpoint():
    """Test the root endpoint returns welcome message"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Breast Cancer Detection API"}


@patch("app.model")
def test_predict_benign(mock_model):
    """Test prediction endpoint with features that should predict benign"""
    # Configure mock model
    mock_model.predict.return_value = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
    
    # Mock input features
    input_data = {
        "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001]
    }
    
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    
    result = response.json()
    assert result["prediction"] == 0
    assert result["message"] == "Benign"
    assert isinstance(result["probability"], float)
    assert 0 <= result["probability"] <= 1


@patch("app.model")
def test_predict_malignant(mock_model):
    """Test prediction endpoint with features that should predict malignant"""
    # Configure mock model
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])
    
    # Mock input features
    input_data = {
        "features": [27.32, 15.62, 172.5, 2288.0, 0.1634, 0.1947, 0.2378]
    }
    
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    
    result = response.json()
    assert result["prediction"] == 1
    assert result["message"] == "Malignant"
    assert isinstance(result["probability"], float)
    assert 0 <= result["probability"] <= 1


def test_invalid_input_length():
    """Test prediction endpoint with invalid input length"""
    # Test with empty features
    input_data = {"features": []}
    response = client.post("/predict", json=input_data)
    assert response.status_code == 400
    
    # Test with too many features
    input_data = {"features": [1.0] * 100}
    response = client.post("/predict", json=input_data)
    assert response.status_code == 400


def test_invalid_input_type():
    """Test prediction endpoint with invalid input types"""
    # Test with string instead of float
    input_data = {"features": ["not_a_number", 10.38, 122.8]}
    response = client.post("/predict", json=input_data)
    assert response.status_code == 422
    
    # Test with None values
    input_data = {"features": [None, 10.38, 122.8]}
    response = client.post("/predict", json=input_data)
    assert response.status_code == 422


@patch("app.model", None)
def test_model_not_loaded():
    """Test prediction when model is not loaded"""
    input_data = {
        "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001]
    }
    
    response = client.post("/predict", json=input_data)
    assert response.status_code == 500
    assert response.json()["detail"] == "Model not loaded"
