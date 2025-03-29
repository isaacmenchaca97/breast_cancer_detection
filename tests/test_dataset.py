import pytest
import pandas as pd
from pathlib import Path
import tempfile
import os
from unittest.mock import patch

# Import the function to test
from src.dataset import main


def test_main_successful_execution():
    # Create temporary input and output files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test input file
        input_path = Path(temp_dir) / "test_input.csv"
        test_df = pd.DataFrame({
            "Column Name": [1, 2, 3],
            "Another Column": ["a", "b", "c"]
        })
        test_df.to_csv(input_path, index=False)
        
        # Set output path
        output_path = Path(temp_dir) / "test_output.csv"
        
        # Call the main function
        main(input_path=input_path, output_path=output_path)
        
        # Verify output file exists
        assert output_path.exists()
        
        # Verify content was processed correctly
        result_df = pd.read_csv(output_path)
        assert list(result_df.columns) == ["column_name", "another_column"]
        assert len(result_df) == 3


def test_main_handles_exception():
    # Test with non-existent input file
    with tempfile.TemporaryDirectory() as temp_dir:
        non_existent_path = Path(temp_dir) / "does_not_exist.csv"
        output_path = Path(temp_dir) / "test_output.csv"
        
        # Use patch to verify logger.exception was called
        with patch("src.dataset.logger.exception") as mock_logger:
            main(input_path=non_existent_path, output_path=output_path)
            mock_logger.assert_called_once()