"""Tests for the G25 unpaid assistance processing module."""

import pytest
import pandas as pd
from pathlib import Path
from etl_logic.tables.g25_unpaid_assistance import process_g25_file

def test_process_g25_file_runs(mocker):
    """Test that the process_g25_file function runs without errors with mocked dependencies."""
    # Mock _process_single_census_csv
    mock_process_csv = mocker.patch(
        'etl_logic.tables.g25_unpaid_assistance._process_single_census_csv',
        return_value=pd.DataFrame()
    )

    # Mock create_demographic_dimension
    mock_create_demographic_dimension = mocker.patch(
        'etl_logic.tables.g25_unpaid_assistance.create_demographic_dimension',
        side_effect=lambda df, *args, **kwargs: df # Passthrough mock
    )

    # Define dummy input arguments
    dummy_file = Path("dummy.csv")

    # Call the function under test
    result = process_g25_file(dummy_file)

    # Assertions
    mock_process_csv.assert_called_once()
    mock_create_demographic_dimension.assert_called_once()
    assert isinstance(result, pd.DataFrame)

def test_process_g25_file_transformation_and_output(mocker):
    """Test the complete process_g25_file flow with sample data and mocked dependencies."""
    # Define sample raw data
    sample_raw_data = pd.DataFrame({
        'Geography': ['Area1', 'Area2', 'Area3', 'Area1'],
        'Sex': ['Male', 'Female', 'Male', 'Female'],
        'Age': ['15-24 years', '25-34 years', '15-24 years', '25-34 years'],
        'Assistance Provided - Yes': [10, 20, 15, 25],
        'Assistance Provided - No': [5, 10, 7, 12],
        'Total': [15, 30, 22, 37],
        '..': ['..', '..', '..', '..'] # Example of a column to be dropped
    })

    # Define expected output data after processing
    # Includes renaming, melting, handling '..', and adding dummy demographic_key
    expected_output_data = pd.DataFrame({
        'geo_area': ['Area1', 'Area1', 'Area2', 'Area2', 'Area3', 'Area3'],
        'sex': ['Male', 'Male', 'Female', 'Female', 'Male', 'Male'],
        'age_group': ['15-24 years', '15-24 years', '25-34 years', '25-34 years', '15-24 years', '15-24 years'],
        'assistance_provided': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
        'count': [10, 5, 20, 10, 15, 7],
        'demographic_key': [1, 1, 1, 1, 1, 1] # Dummy key added by mocked create_demographic_dimension
    })
    # Ensure correct data types for comparison
    expected_output_data['count'] = expected_output_data['count'].astype(int)
    expected_output_data['demographic_key'] = expected_output_data['demographic_key'].astype(int)

    # Mock _process_single_census_csv
    mock_process_csv = mocker.patch(
        'etl_logic.tables.g25_unpaid_assistance._process_single_census_csv',
        return_value=sample_raw_data
    )

    # Mock create_demographic_dimension to add a dummy key
    def mock_create_demographic_dimension_with_key(df, *args, **kwargs):
        df['demographic_key'] = 1
        return df

    mock_create_demographic_dimension = mocker.patch(
        'etl_logic.tables.g25_unpaid_assistance.create_demographic_dimension',
        side_effect=mock_create_demographic_dimension_with_key
    )

    # Mock validation functions
    mocker.patch('etl_logic.tables.g25_unpaid_assistance.validate_dataframe', return_value=None)
    mocker.patch('etl_logic.tables.g25_unpaid_assistance.check_expected_columns', return_value=True)
    mocker.patch('etl_logic.tables.g25_unpaid_assistance.check_nulls', return_value=True)
    mocker.patch('etl_logic.tables.g25_unpaid_assistance.check_categorical_values', return_value=True)

    # Define dummy input arguments
    dummy_file = Path("dummy.csv")

    # Call the function under test
    result = process_g25_file(dummy_file)

    # Assert output DataFrame matches expected
    pd.testing.assert_frame_equal(result, expected_output_data, check_dtype=True)

