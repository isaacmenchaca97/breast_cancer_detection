import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock
import pickle

# Import the function to test
from src.modeling.train import main

@pytest.fixture
def mock_mlflow():
    with patch("src.modeling.train.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__.return_value = None
        mock_mlflow.start_run.return_value.__exit__.return_value = None
        yield mock_mlflow

@pytest.fixture
def mock_dagshub():
    with patch("src.modeling.train.dagshub") as mock_dagshub:
        yield mock_dagshub

def test_successful_model_training(mock_mlflow, mock_dagshub):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test input files
        features_path = Path(temp_dir) / "features.csv"
        labels_path = Path(temp_dir) / "labels.csv"
        model_path = Path(temp_dir) / "model.pkl"
        test_features_path = Path(temp_dir) / "test_features.csv"
        test_predictions_path = Path(temp_dir) / "test_predictions.csv"
        
        # Create synthetic data
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        # Features with 5 columns
        features = np.random.rand(n_samples, n_features)
        # Binary classification labels
        labels = np.random.randint(0, 2, n_samples)
        
        pd.DataFrame(features, columns=[f"feature{i}" for i in range(n_features)]).to_csv(features_path, index=False)
        pd.DataFrame(labels, columns=["target"]).to_csv(labels_path, index=False)
        
        # Run the main function
        main(
            features_path=features_path,
            labels_path=labels_path,
            model_path=model_path,
            test_features_path=test_features_path,
            test_predictions_path=test_predictions_path
        )
        
        # Verify outputs
        assert model_path.exists()
        assert test_features_path.exists()
        assert test_predictions_path.exists()
        
        # Load and check the model
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # Check that the model is a pipeline with the expected components
        assert "minmaxscaler" in str(model).lower()
        assert "pca" in str(model).lower()
        assert "logisticregression" in str(model).lower()
        
        # Check the test features and predictions
        test_features = pd.read_csv(test_features_path)
        test_predictions = pd.read_csv(test_predictions_path)
        
        # Test split should be 20% of data
        assert len(test_features) == 20
        assert len(test_predictions) == 20
        
        # Verify MLflow integration
        mock_dagshub.init.assert_called_once()
        mock_mlflow.sklearn.autolog.assert_called_once()
        mock_mlflow.set_experiment.assert_called_once()
        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.log_metric.assert_called_once()

def test_handles_exception(mock_mlflow, mock_dagshub):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Non-existent input paths
        features_path = Path(temp_dir) / "nonexistent_features.csv"
        labels_path = Path(temp_dir) / "nonexistent_labels.csv"
        model_path = Path(temp_dir) / "model.pkl"
        test_features_path = Path(temp_dir) / "test_features.csv"
        test_predictions_path = Path(temp_dir) / "test_predictions.csv"
        
        # Use patch to verify logger.exception was called
        with patch("src.modeling.train.logger.exception") as mock_logger:
            main(
                features_path=features_path,
                labels_path=labels_path,
                model_path=model_path,
                test_features_path=test_features_path,
                test_predictions_path=test_predictions_path
            )
            mock_logger.assert_called_once()

def test_class_weights_calculation(mock_mlflow, mock_dagshub):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test input files with imbalanced classes
        features_path = Path(temp_dir) / "features.csv"
        labels_path = Path(temp_dir) / "labels.csv"
        model_path = Path(temp_dir) / "model.pkl"
        test_features_path = Path(temp_dir) / "test_features.csv"
        test_predictions_path = Path(temp_dir) / "test_predictions.csv"
        
        # Create imbalanced dataset (75% class 0, 25% class 1)
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        features = np.random.rand(n_samples, n_features)
        labels = np.zeros(n_samples)
        labels[:25] = 1  # 25% of samples are class 1
        
        pd.DataFrame(features, columns=[f"feature{i}" for i in range(n_features)]).to_csv(features_path, index=False)
        pd.DataFrame(labels, columns=["target"]).to_csv(labels_path, index=False)
        
        # Mock LogisticRegression to capture class_weight parameter
        mock_lr = MagicMock()
        
        with patch("src.modeling.train.LogisticRegression", return_value=mock_lr) as mock_lr_class:
            main(
                features_path=features_path,
                labels_path=labels_path,
                model_path=model_path,
                test_features_path=test_features_path,
                test_predictions_path=test_predictions_path
            )
            
            # Get the class_weight parameter
            _, kwargs = mock_lr_class.call_args
            class_weights = kwargs.get('class_weight')
            
            # Expected class weights: {0: 1.0, 1: 3.0} (75/25 = 3) with some rounding
            assert abs(class_weights[0] - 1.0) < 0.1
            assert abs(class_weights[1] - 3.0) < 0.1