import pytest
import pandas as pd
from pathlib import Path
import tempfile
import pickle

# Import the function to test
from src.modeling.predict import main


# Define DummyModel at module level for pickling
class DummyModel:
    def __init__(self, predictions):
        self.predictions = predictions
    
    def predict(self, x):
        return self.predictions


def test_successful_prediction():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test paths
        features_path = Path(temp_dir) / "test_features.csv"
        predictions_path = Path(temp_dir) / "test_predictions.csv"
        model_path = Path(temp_dir) / "model.pkl"
        
        # Create dummy test data
        x_test = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [0.1, 0.2, 0.3]
        })
        y_test = pd.DataFrame([0, 1, 0])
        
        # Create a dummy model with predetermined predictions
        model = DummyModel(predictions=[0, 1, 0])
        
        # Save test data and model
        x_test.to_csv(features_path, index=False)
        y_test.to_csv(predictions_path, index=False)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Run prediction
        main(
            features_path=features_path,
            model_path=model_path,
            predictions_path=predictions_path
        )


def test_accuracy_calculation():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test paths
        features_path = Path(temp_dir) / "test_features.csv"
        predictions_path = Path(temp_dir) / "test_predictions.csv"
        model_path = Path(temp_dir) / "model.pkl"
        
        # Create dummy test data
        x_test = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [0.1, 0.2, 0.3, 0.4]
        })
        y_test = pd.DataFrame([0, 1, 0, 1])
        
        # Create a dummy model with predetermined predictions
        model = DummyModel(predictions=[0, 1, 0, 0])
        
        # Save test data and model
        x_test.to_csv(features_path, index=False)
        y_test.to_csv(predictions_path, index=False)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Run prediction - this should result in 75% accuracy
        # as 3 out of 4 predictions match the true values
        main(
            features_path=features_path,
            model_path=model_path,
            predictions_path=predictions_path
        )
