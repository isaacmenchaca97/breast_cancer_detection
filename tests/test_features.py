import pytest
import pandas as pd
from pathlib import Path
import tempfile
import os
from unittest.mock import patch, MagicMock
import numpy as np

# Import the function to test
from src.features import main

def test_main_successful_execution():
    # Create temporary input and output files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test input file with some features correlated to target and some not
        input_path = Path(temp_dir) / "test_dataset.csv"
        
        # Create synthetic data where feature1 is related to target but feature2 is not
        np.random.seed(42)
        n_samples = 100
        
        # Target: 0 or 1
        target = np.random.randint(0, 2, n_samples)
        
        # feature1: related to target (different means for each target class)
        feature1 = np.where(target == 1, np.random.normal(10, 2, n_samples), np.random.normal(5, 2, n_samples))
        
        # feature2: not related to target (same distribution regardless of target)
        feature2 = np.random.normal(7, 2, n_samples)
        
        test_df = pd.DataFrame({
            "feature1": feature1,
            "feature2": feature2,
            "target": target
        })
        test_df.to_csv(input_path, index=False)
        
        # Set output paths
        feature_output_path = Path(temp_dir) / "test_features.csv"
        label_output_path = Path(temp_dir) / "test_labels.csv"
        
        # Call the main function
        main(
            input_path=input_path, 
            feature_output_path=feature_output_path, 
            label_output_path=label_output_path
        )
        
        # Verify output files exist
        assert feature_output_path.exists()
        assert label_output_path.exists()
        
        # Verify content was processed correctly
        features_df = pd.read_csv(feature_output_path)
        labels_df = pd.read_csv(label_output_path)
        
        # feature1 should be kept, feature2 should be dropped
        assert "feature1" in features_df.columns
        assert "feature2" not in features_df.columns
        assert "target" not in features_df.columns
        
        # Check labels file
        assert "target" in labels_df.columns
        assert len(labels_df) == n_samples

def test_main_handles_exception():
    # Test with non-existent input file
    with tempfile.TemporaryDirectory() as temp_dir:
        non_existent_path = Path(temp_dir) / "does_not_exist.csv"
        feature_output_path = Path(temp_dir) / "test_features.csv"
        label_output_path = Path(temp_dir) / "test_labels.csv"
        
        # Use patch to verify logger.exception was called
        with patch("src.features.logger.exception") as mock_logger:
            main(
                input_path=non_existent_path, 
                feature_output_path=feature_output_path, 
                label_output_path=label_output_path
            )
            mock_logger.assert_called_once()
