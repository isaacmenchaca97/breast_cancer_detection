import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock
import pickle

# Import the function to test
from src.modeling.predict import main


def test_model_loading_error():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test paths with valid CSV files but invalid model file
        features_path = Path(temp_dir) / "test_features.csv"
        predictions_path = Path(temp_dir) / "test_predictions.csv"
        model_path = Path(temp_dir) / "model.pkl"
        
        # Create dummy data files
        pd.DataFrame([[1, 2], [3, 4]]).to_csv(features_path, index=False)
        pd.DataFrame([0, 1]).to_csv(predictions_path, index=False)
        
        # Create invalid model file
        with open(model_path, "w") as f:
            f.write("This is not a valid pickle file")
        
        # The function should raise an exception when loading the model
        with pytest.raises(Exception):
            main(
                features_path=features_path,
                model_path=model_path,
                predictions_path=predictions_path
            )

def test_file_not_found():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set paths to files that don't exist
        features_path = Path(temp_dir) / "nonexistent_features.csv"
        predictions_path = Path(temp_dir) / "nonexistent_predictions.csv"
        model_path = Path(temp_dir) / "nonexistent_model.pkl"
        
        # The function should raise an exception
        with pytest.raises(FileNotFoundError):
            main(
                features_path=features_path,
                model_path=model_path,
                predictions_path=predictions_path
            )