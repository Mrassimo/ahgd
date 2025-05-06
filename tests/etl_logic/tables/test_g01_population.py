import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import polars as pl
from polars.testing import assert_frame_equal
import logging

# Assuming the project structure allows this import
from etl_logic.tables import g01_population
from etl_logic import config, census

# Define expected configurations for G01 (mirroring config or fallback)
MOCK_G01_MEASURE_MAP = {
    'total_persons': ['Tot_P_P', 'Total_Persons_P', 'Persons_Total_P', 'P_Tot'],
    'total_male': ['Tot_P_M', 'Total_Persons_M', 'Males_Total_M', 'M_Tot'],
    'total_female': ['Tot_P_F', 'Total_Persons_F', 'Females_Total_F', 'F_Tot'],
    'total_indigenous': ['Indigenous_P_Tot_P', 'Tot_Indigenous_P', 'Indigenous_Tot_P', 'Indigenous_P']
}
MOCK_G01_REQUIRED_COLS = ['total_persons', 'total_male', 'total_female']
MOCK_G01_CONFIG = {
    "G01": {
        "measure_column_map": MOCK_G01_MEASURE_MAP,
        "required_target_columns": MOCK_G01_REQUIRED_COLS,
        # Add other potential config keys if needed by the processor
    }
}

@pytest.fixture
def mock_csv_path():
    """Provides a dummy Path object for the CSV file."""
    return Path("dummy/path/to/G01_TEST.csv")

@patch('etl_logic.tables.g01_population.census._process_single_census_csv')
@patch('etl_logic.tables.g01_population.config.CENSUS_COLUMN_MAPPINGS', MOCK_G01_CONFIG)
def test_process_g01_file_success(mock_process_csv, mock_csv_path):
    """Test process_g01_file successfully calls the generic processor with correct args."""
    # Arrange
    mock_output_df = pl.DataFrame({
        "region_id": ["123", "456"],
        "total_persons": [100, 200],
        "total_male": [50, 90],
        "total_female": [50, 110]
    })
    mock_process_csv.return_value = mock_output_df
    expected_geo_options = ['region_id', 'SA1_CODE21', 'SA2_CODE21', 'SA1_CODE_2021', 'SA2_CODE_2021']

    # Act
    result_df = g01_population.process_g01_file(mock_csv_path)

    # Assert
    mock_process_csv.assert_called_once_with(
        csv_file=mock_csv_path,
        geo_column_options=expected_geo_options,
        measure_column_map=MOCK_G01_MEASURE_MAP,
        required_target_columns=MOCK_G01_REQUIRED_COLS,
        table_code="G01"
    )
    assert result_df is not None
    assert_frame_equal(result_df, mock_output_df)

@patch('etl_logic.tables.g01_population.census._process_single_census_csv')
@patch('etl_logic.tables.g01_population.config.CENSUS_COLUMN_MAPPINGS', {}) # Simulate config missing G01
def test_process_g01_file_fallback_config(mock_process_csv, mock_csv_path, caplog):
    """Test process_g01_file uses fallback config when G01 is missing from config."""
    # Arrange
    mock_output_df = pl.DataFrame({"region_id": ["123"], "total_persons": [100]}) # Dummy output
    mock_process_csv.return_value = mock_output_df
    expected_geo_options = ['region_id', 'SA1_CODE21', 'SA2_CODE21', 'SA1_CODE_2021', 'SA2_CODE_2021']
    # Fallback map used in g01_population.py
    fallback_measure_map = {
        'total_persons': ['Tot_P_P', 'Total_Persons_P', 'Persons_Total_P', 'P_Tot'],
        'total_male': ['Tot_P_M', 'Total_Persons_M', 'Males_Total_M', 'M_Tot'],
        'total_female': ['Tot_P_F', 'Total_Persons_F', 'Females_Total_F', 'F_Tot'],
        'total_indigenous': ['Indigenous_P_Tot_P', 'Tot_Indigenous_P', 'Indigenous_Tot_P', 'Indigenous_P']
    }
    fallback_required_cols = ['total_persons', 'total_male', 'total_female']


    # Act
    with caplog.at_level(logging.WARNING):
        result_df = g01_population.process_g01_file(mock_csv_path)

    # Assert
    assert "[G01] Mappings not found in config, using fallback (limited)." in caplog.text
    mock_process_csv.assert_called_once_with(
        csv_file=mock_csv_path,
        geo_column_options=expected_geo_options,
        measure_column_map=fallback_measure_map,
        required_target_columns=fallback_required_cols,
        table_code="G01"
    )
    assert result_df is not None
    assert_frame_equal(result_df, mock_output_df)

@patch('etl_logic.tables.g01_population.census._process_single_census_csv')
@patch('etl_logic.tables.g01_population.config.CENSUS_COLUMN_MAPPINGS', MOCK_G01_CONFIG)
def test_process_g01_file_processor_returns_none(mock_process_csv, mock_csv_path, caplog):
    """Test process_g01_file handles None return from the generic processor."""
    # Arrange
    mock_process_csv.return_value = None

    # Act
    with caplog.at_level(logging.ERROR):
        result_df = g01_population.process_g01_file(mock_csv_path)

    # Assert
    assert result_df is None
    assert "[G01] _process_single_census_csv returned None" in caplog.text
    mock_process_csv.assert_called_once() # Verify it was called

@patch('etl_logic.tables.g01_population.census._process_single_census_csv')
@patch('etl_logic.tables.g01_population.config.CENSUS_COLUMN_MAPPINGS', {"G01": {}}) # Config exists but empty
def test_process_g01_file_no_measure_map(mock_process_csv, mock_csv_path, caplog):
    """Test process_g01_file handles missing measure map in config."""
    # Arrange
    # Act
    with caplog.at_level(logging.ERROR):
        result_df = g01_population.process_g01_file(mock_csv_path)

    # Assert
    assert result_df is None
    assert "[G01] No measure columns defined" in caplog.text
    mock_process_csv.assert_not_called() # Should not call processor if no map 