from src.preprocess import preprocess
from unittest.mock import patch
import pytest
import tempfile
import shutil
import os
import numpy as np
import pandas as pd
from io import StringIO


@pytest.fixture
def temp_dir():
    """Fixture to create a temporary directory for test outputs."""
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir)


@pytest.fixture
def mock_dataset():
    """Fixture to mock the load_breast_cancer function."""
    # Create a mock dataset similar to breast cancer dataset structure
    class MockDataset:
        def __init__(self):
            self.data = np.array([
                [17.99, 10.38, 122.8, 1001.0, 0.1184],
                [20.57, 17.77, 132.9, 1326.0, 0.08474],
                [19.69, 21.25, 130.0, 1203.0, 0.1096],
                [11.42, 20.38, 77.58, 386.1, 0.1425],
            ])
            self.feature_names = [
                'mean radius',
                'mean texture',
                'mean perimeter',
                'mean area',
                'mean smoothness'
            ]
            self.target = np.array([0, 1, 1, 0])

    return MockDataset()


def test_data_loading_and_transformation(mock_dataset, temp_dir):
    """Test that data is loaded and transformed correctly."""
    output_path = os.path.join(temp_dir, "processed/train.csv")

    # Mock the breast cancer dataset loading
    with patch('src.preprocess.load_breast_cancer', return_value=mock_dataset):
        # Mock the yaml loading to use our test output path
        mock_params = {"output": output_path}
        with patch('src.preprocess.params', mock_params):
            # Capture stdout to prevent printing during tests
            with patch('sys.stdout', new=StringIO()):
                # Run the preprocessing function
                preprocess(output_path)

    # Check that the output file was created
    assert os.path.exists(output_path)

    # Load the processed data and check structure
    processed_df = pd.read_csv(output_path)

    # Check column names are transformed correctly
    for col in processed_df.columns:
        assert col.islower()
        assert " " not in col

    # Check target column exists
    assert "target" in processed_df.columns
