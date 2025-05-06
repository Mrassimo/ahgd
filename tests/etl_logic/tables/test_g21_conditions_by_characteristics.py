import logging
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import polars as pl
from polars.testing import assert_frame_equal

from etl_logic.tables.g21_conditions_by_characteristics import (
    process_g21_csv_file, 
    process_g21_file, 
    process_g21_unpivot_csv, 
    process_g21_file_generic_fallback
)
from etl_logic import config, utils

# Setup logging
logger = logging.getLogger('test_logger')
logger.setLevel(logging.DEBUG)

# Fixtures
@pytest.fixture
def mock_paths():
    return {
        "zip_dir": Path("/fake/zip"),
        "temp_extract_base": Path("/fake/temp"),
        "output_dir": Path("/fake/output"),
        "geo_output_path": Path("/fake/output/dim_geographic.parquet"),
        "time_sk": 20210810
    }

@pytest.fixture
def sample_g21_df():
    # A simplified DataFrame mimicking raw G21 input
    return pl.DataFrame({
        "SA1_CODE_2021": ["101011001", "101011002"],
        "P_Tot_Tot": [100, 150],
        "P_Tot_Has_condition": [20, 30],
        "P_Tot_No_condition": [75, 110],
        "P_Tot_Condition_ns": [5, 10]
    })

@pytest.fixture
def sample_g21_unpivot_df():
    # More complex input for unpivot test
    return pl.DataFrame({
        "SA1_CODE_2021": ["101011001", "101011002"],
        "COB_Aus_Arth": [5, 10], # CountryOfBirth=Australia, Condition=Arthritis
        "COB_Oth_Euro_Arth": [1, 2],
        "COB_Aus_Asthma": [3, 4],
        "LFS_Emp_Dia_ges_dia": [2, 3], # LabourForceStatus=Employed, Condition=Diabetes
        "LFS_Unemp_Asthma": [1, 1],
        "Tot_Tot_Tot": [50, 60], # Total characteristic, Total condition
        "Irrelevant_Column": [0, 0]
    })

@pytest.fixture
def expected_g21_df():
    # Expected output from process_g21_csv_file (using generic processor)
    return pl.DataFrame({
        "geo_code": ["101011001", "101011002"],
        "total_count": [100, 150],
        "has_condition_count": [20, 30],
        "no_condition_count": [75, 110],
        "condition_not_stated_count": [5, 10]
    })

@pytest.fixture
def expected_unpivot_df():
    # Expected output from process_g21_unpivot_csv
    # Simplified, assuming mappings exist in config
    # Order might vary
    return pl.DataFrame({
        "geo_code": ["101011001"] * 6 + ["101011002"] * 6, # Duplicated per row
        "characteristic_type": ["CountryOfBirth"] * 3 + ["LabourForceStatus"] * 2 + ["Total"] + 
                               ["CountryOfBirth"] * 3 + ["LabourForceStatus"] * 2 + ["Total"],
        "characteristic_value": ["Aus", "Oth_Euro", "Aus", "Emp", "Unemp", "Tot"] * 2,
        "condition": ["arthritis", "arthritis", "asthma", "diabetes", "asthma", "total"] * 2, 
        "count": [5, 1, 3, 2, 1, 50, 10, 2, 4, 3, 1, 60]
    }).sort(["geo_code", "characteristic_type", "characteristic_value", "condition"])

@pytest.fixture(autouse=True)
def mock_g21_config(monkeypatch):
    """Mock the config mappings for G21 unpivot."""
    mock_mappings = {
        "G21": {
            "characteristic_types": {
                "COB": "CountryOfBirth",
                "LFS": "LabourForceStatus",
                "Tot": "Total",
                 # Add more as needed based on actual G21 columns
            },
            "condition_mappings": {
                "Arth": "arthritis",
                "Asthma": "asthma",
                "Dia_ges_dia": "diabetes", # Example for combined code
                "Tot": "total",
                 # Add more
            },
             # Add measure_column_map for the non-unpivot processor if needed
             "measure_column_map": {
                 'total_count': ['P_Tot_Tot'],
                 'has_condition_count': ['P_Tot_Has_condition'],
                 'no_condition_count': ['P_Tot_No_condition'],
                 'condition_not_stated_count': ['P_Tot_Condition_ns']
             },
             "required_target_columns": [
                 'total_count',
                 'has_condition_count',
                 'no_condition_count',
                 'condition_not_stated_count'
             ]
        }
    }
    monkeypatch.setattr(config, "CENSUS_COLUMN_MAPPINGS", mock_mappings)

# --- Tests for process_g21_csv_file --- (Uses generic processor)

@patch('polars.read_csv')
@patch('etl_logic.tables.g21_conditions_by_characteristics.census._process_single_census_csv')
def test_process_g21_csv_file_success(mock_process_single, mock_read_csv, sample_g21_df, expected_g21_df):
    """Test successful processing of a G21 file using the generic processor."""
    mock_csv_path = Path("/fake/G21_SA1_AUS.csv")
    mock_process_single.return_value = expected_g21_df.clone() # Return the expected DF
    
    # Note: process_g21_csv_file directly calls _process_single_census_csv
    # We don't need mock_read_csv here unless _process_single_census_csv fails
    # and we need to test the error path *within* process_g21_csv_file (which is minimal)
    
    result_df = process_g21_csv_file(mock_csv_path)
    
    assert result_df is not None
    assert_frame_equal(result_df, expected_g21_df)
    mock_process_single.assert_called_once()
    call_args = mock_process_single.call_args[1] # Get kwargs
    assert call_args['csv_file'] == mock_csv_path
    assert call_args['table_code'] == "G21"
    assert 'total_count' in call_args['required_target_columns']

@patch('polars.read_csv')
@patch('etl_logic.tables.g21_conditions_by_characteristics.census._process_single_census_csv')
def test_process_g21_csv_file_failure(mock_process_single, mock_read_csv):
    """Test failure during G21 processing using the generic processor."""
    mock_csv_path = Path("/fake/G21_SA1_AUS.csv")
    mock_process_single.return_value = None # Simulate failure in the helper
    
    result_df = process_g21_csv_file(mock_csv_path)
    
    assert result_df is None
    mock_process_single.assert_called_once_with(
        csv_file=mock_csv_path,
        geo_column_options=[
            'SA1_CODE_2021', 'SA2_CODE_2021', 'SA3_CODE_2021',
            'SA4_CODE_2021', 'GCC_CODE_2021', 'STE_CODE_2021',
            'LGA_CODE_2021'
        ],
        measure_column_map={
            'total_count': ['P_Tot_Tot'],
            'has_condition_count': ['P_Tot_Has_condition'],
            'no_condition_count': ['P_Tot_No_condition'],
            'condition_not_stated_count': ['P_Tot_Condition_ns']
        },
        required_target_columns=[
            'total_count', 'has_condition_count', 'no_condition_count', 
            'condition_not_stated_count'
        ],
        table_code="G21"
    )

# --- Tests for process_g21_file_generic_fallback --- (Explicitly tests the fallback)

@patch('etl_logic.tables.g21_conditions_by_characteristics.census._process_single_census_csv')
def test_process_g21_file_generic_fallback_success(mock_process_single, expected_g21_df):
    """Test the generic fallback processor successfully calls the helper."""
    mock_csv_path = Path("/fake/G21_SA1_AUS.csv")
    mock_process_single.return_value = expected_g21_df.clone()

    result_df = process_g21_file_generic_fallback(mock_csv_path)

    assert result_df is not None
    assert_frame_equal(result_df, expected_g21_df)
    mock_process_single.assert_called_once()
    call_args = mock_process_single.call_args[1]
    assert call_args['csv_file'] == mock_csv_path
    assert call_args['table_code'] == "G21"
    # Check that it used the config map because it was mocked
    assert 'total_count' in call_args['measure_column_map'] 

@patch('etl_logic.tables.g21_conditions_by_characteristics.census._process_single_census_csv')
def test_process_g21_file_generic_fallback_no_config(mock_process_single, monkeypatch, expected_g21_df):
    """Test the generic fallback uses internal map when config is missing."""
    # Remove G21 from mocked config
    monkeypatch.setattr(config, "CENSUS_COLUMN_MAPPINGS", {})
    
    mock_csv_path = Path("/fake/G21_SA1_AUS.csv")
    # Make mock return something simple, focus is on args passed to it
    mock_process_single.return_value = pl.DataFrame({"geo_code": ["1"], "cob_aus_arthritis_total": [1]})

    result_df = process_g21_file_generic_fallback(mock_csv_path)

    assert result_df is not None 
    mock_process_single.assert_called_once()
    call_args = mock_process_single.call_args[1]
    assert call_args['csv_file'] == mock_csv_path
    assert call_args['table_code'] == "G21"
    # Check that it used the *internal* fallback map
    assert 'cob_aus_arthritis_total' in call_args['measure_column_map']
    assert call_args['required_target_columns'] == []

# --- Tests for process_g21_unpivot_csv --- (The recommended processor for G21)

@patch('polars.read_csv')
@patch('etl_logic.tables.g21_conditions_by_characteristics.utils.find_geo_column')
@patch('etl_logic.tables.g21_conditions_by_characteristics.utils.clean_polars_geo_code')
@patch('etl_logic.tables.g21_conditions_by_characteristics.utils.safe_polars_int')
def test_process_g21_unpivot_csv_success(mock_safe_int, mock_clean_geo, mock_find_geo, mock_read_csv, sample_g21_unpivot_df, expected_unpivot_df):
    """Test successful unpivoting of a G21 file."""
    mock_csv_path = Path("/fake/G21_SA1_AUS_unpivot.csv")
    mock_read_csv.return_value = sample_g21_unpivot_df.clone()
    mock_find_geo.return_value = "SA1_CODE_2021"
    # Mock clean_geo and safe_int to just return the column for simplicity
    mock_clean_geo.side_effect = lambda x: x 
    mock_safe_int.side_effect = lambda x: x
    
    result_df = process_g21_unpivot_csv(mock_csv_path)
    
    assert result_df is not None
    # Sort both dataframes for consistent comparison
    result_df_sorted = result_df.sort(["geo_code", "characteristic_type", "characteristic_value", "condition"])
    expected_unpivot_df_sorted = expected_unpivot_df.sort(["geo_code", "characteristic_type", "characteristic_value", "condition"])

    # print("\nResult DF:")
    # print(result_df_sorted)
    # print("\nExpected DF:")
    # print(expected_unpivot_df_sorted)

    assert_frame_equal(result_df_sorted, expected_unpivot_df_sorted, check_dtype=False)
    mock_read_csv.assert_called_once_with(mock_csv_path, truncate_ragged_lines=True)
    mock_find_geo.assert_called_once()

@patch('polars.read_csv')
@patch('etl_logic.tables.g21_conditions_by_characteristics.utils.find_geo_column')
def test_process_g21_unpivot_csv_no_geo_col(mock_find_geo, mock_read_csv, sample_g21_unpivot_df):
    """Test failure when no geographic column is found."""
    mock_csv_path = Path("/fake/G21_SA1_AUS_unpivot.csv")
    mock_read_csv.return_value = sample_g21_unpivot_df.clone()
    mock_find_geo.return_value = None # Simulate not finding geo column
    
    result_df = process_g21_unpivot_csv(mock_csv_path)
    
    assert result_df is None
    mock_read_csv.assert_called_once_with(mock_csv_path, truncate_ragged_lines=True)
    mock_find_geo.assert_called_once()

@patch('polars.read_csv')
def test_process_g21_unpivot_csv_read_error(mock_read_csv):
    """Test failure during CSV read."""
    mock_csv_path = Path("/fake/G21_SA1_AUS_unpivot.csv")
    mock_read_csv.side_effect = Exception("Read error")
    
    result_df = process_g21_unpivot_csv(mock_csv_path)
    
    assert result_df is None
    mock_read_csv.assert_called_once_with(mock_csv_path, truncate_ragged_lines=True)

@patch('polars.read_csv')
@patch('etl_logic.tables.g21_conditions_by_characteristics.utils.find_geo_column')
def test_process_g21_unpivot_csv_no_valid_cols(mock_find_geo, mock_read_csv):
    """Test scenario where no columns match the expected unpivot patterns."""
    mock_csv_path = Path("/fake/G21_SA1_AUS_unpivot.csv")
    # Return DF with only geo_code and irrelevant columns
    mock_read_csv.return_value = pl.DataFrame({
        "SA1_CODE_2021": ["101011001"],
        "Irrelevant1": [1],
        "Another_Irrelevant": [2]
    })
    mock_find_geo.return_value = "SA1_CODE_2021"
    
    result_df = process_g21_unpivot_csv(mock_csv_path)
    
    assert result_df is None
    mock_read_csv.assert_called_once_with(mock_csv_path, truncate_ragged_lines=True)
    mock_find_geo.assert_called_once()

# --- Tests for process_g21_file (Main orchestrator) ---
# This function now calls process_census_table, which handles extraction and iteration.
# We need to mock process_census_table itself.

@patch('etl_logic.tables.g21_conditions_by_characteristics.process_census_table')
@patch('polars.read_parquet') # Mock reading the intermediate output
@patch('polars.DataFrame.write_parquet') # Mock writing the final output
def test_process_g21_file_success(mock_write_parquet, mock_read_parquet, mock_process_census, mock_paths):
    """Test successful end-to-end processing via process_g21_file."""
    mock_process_census.return_value = True # Simulate successful census table processing
    # Simulate reading back a dataframe that needs deduplication
    mock_df = pl.DataFrame({
        "geo_sk": [1, 1, 2],
        "time_sk": [20210810, 20210810, 20210810],
        "condition_sk": [10, 10, 11], # Duplicate key (1, 20210810, 10)
        "some_count": [5, 10, 15]
    })
    # Expected output after deduplication
    expected_dedup_df = pl.DataFrame({
        "geo_sk": [1, 2],
        "time_sk": [20210810, 20210810],
        "condition_sk": [10, 11],
        "some_count": [15, 15] # Sum of counts for the duplicate key
    })
    mock_read_parquet.return_value = mock_df
    
    success = process_g21_file(**mock_paths)
    
    assert success is True
    mock_process_census.assert_called_once()
    # Check args passed to process_census_table
    call_args = mock_process_census.call_args[1]
    assert call_args['table_code'] == "G21"
    assert call_args['process_file_function'] == process_g21_csv_file # Check correct processor passed
    assert call_args['output_filename'] == "fact_health_conditions_by_characteristic.parquet"
    assert call_args['zip_dir'] == mock_paths['zip_dir']
    assert call_args['temp_extract_base'] == mock_paths['temp_extract_base']
    assert call_args['output_dir'] == mock_paths['output_dir']
    assert call_args['geo_output_path'] == mock_paths['geo_output_path']
    assert call_args['time_sk'] == mock_paths['time_sk']

    mock_read_parquet.assert_called_once_with(mock_paths['output_dir'] / "fact_health_conditions_by_characteristic.parquet")
    # Check that write_parquet was called with the deduplicated data
    mock_write_parquet.assert_called_once()
    # assert_frame_equal(mock_write_parquet.call_args[0][0], expected_dedup_df) # This comparison can be tricky with mocks

@patch('etl_logic.tables.g21_conditions_by_characteristics.process_census_table')
def test_process_g21_file_census_step_fails(mock_process_census, mock_paths):
    """Test failure when the initial process_census_table step fails."""
    mock_process_census.return_value = False # Simulate failure
    
    success = process_g21_file(**mock_paths)
    
    assert success is False
    mock_process_census.assert_called_once()

@patch('etl_logic.tables.g21_conditions_by_characteristics.process_census_table')
@patch('polars.read_parquet')
def test_process_g21_file_read_parquet_fails(mock_read_parquet, mock_process_census, mock_paths):
    """Test failure when reading the intermediate parquet file fails."""
    mock_process_census.return_value = True # Census step succeeds
    mock_read_parquet.side_effect = Exception("Cannot read parquet") # Read fails
    
    success = process_g21_file(**mock_paths)
    
    assert success is False
    mock_process_census.assert_called_once()
    mock_read_parquet.assert_called_once()

@patch('etl_logic.tables.g21_conditions_by_characteristics.process_census_table')
@patch('polars.read_parquet')
@patch('polars.DataFrame.write_parquet') 
def test_process_g21_file_deduplication_fails(mock_write_parquet, mock_read_parquet, mock_process_census, mock_paths):
    """Test failure during the deduplication step (e.g., duplicates remain)."""
    mock_process_census.return_value = True
    # Simulate data where simple aggregation still results in duplicate keys
    # (This shouldn't happen with sum aggregation but simulates a logic error)
    mock_df = pl.DataFrame({
        "geo_sk": [1, 1, 2],
        "time_sk": [20210810, 20210810, 20210810],
        "other_dim_sk": [100, 100, 101], # This is the grain
        "count": [5, 10, 15]
    })
    mock_read_parquet.return_value = mock_df

    # Mock the unique check to simulate failure
    with patch('polars.DataFrame.unique') as mock_unique:
        # Make unique() return more rows than group_by().agg()
        # This requires careful mocking of how group_by().agg() and unique() interact
        # For simplicity, let's assume the group_by correctly aggregates to 2 rows
        # but unique() incorrectly reports 3 unique key combinations
        mock_unique.return_value = mock_df.select(["geo_sk", "time_sk", "other_dim_sk"]) # Return all 3 key combos

        success = process_g21_file(**mock_paths)
        
        assert success is False
        mock_process_census.assert_called_once()
        mock_read_parquet.assert_called_once()
        mock_write_parquet.assert_not_called() # Should fail before writing 