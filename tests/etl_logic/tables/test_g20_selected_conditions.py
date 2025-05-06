import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import polars as pl
from polars.testing import assert_frame_equal
import logging

from etl_logic.tables import g20_selected_conditions
from etl_logic import utils, config, census

@pytest.fixture
def mock_csv_path():
    """Provides a dummy Path object for the CSV file."""
    return Path("dummy/path/to/G20_TEST.csv")

# --- Fixtures for process_g20_unpivot_csv ---

@pytest.fixture
def sample_g20_data_basic():
    return pl.DataFrame({
        "SA1_CODE_2021": ["101011001", "101011002"],
        "P_No_cond_0_4": [50, 60],
        "F_One_cond_15_24": [10, 12],
        "M_Two_cond_65_74": [5, 8],
        "P_Three_more_85ov": [1, 0],
        "F_Not_stated_Tot": [2, 3],
        "P_Tot_Tot": [1000, 1200] # Total persons in demographic
    })

@pytest.fixture
def sample_g20_data_malformed():
    return pl.DataFrame({
        "SA2_CODE_2021": ["10101"],
        "P_No_cond_0_4": [40],
        "INVALID_COL": [1],
        "F_Four_cond_15_24": [2] # Invalid condition count category
    })

@pytest.fixture
def expected_g20_output_basic():
    return pl.DataFrame({
        "geo_code": ["101011001", "101011002", "101011001", "101011002", "101011001", "101011002", "101011001", "101011001", "101011002", "101011001", "101011002"],
        "sex": ["P", "P", "F", "F", "M", "M", "P", "F", "F", "P", "P"],
        "age_range": ["0-4", "0-4", "15-24", "15-24", "65-74", "65-74", "85+", "total", "total", "total", "total"],
        "condition_count_category": [
            "no_condition", "no_condition",
            "one_condition", "one_condition",
            "two_conditions", "two_conditions",
            "three_or_more",
            "not_stated", "not_stated",
            "total_persons", "total_persons"
        ],
        "count": pl.Series([50, 60, 10, 12, 5, 8, 1, 2, 3, 1000, 1200], dtype=pl.Int64)
    }).sort(["geo_code", "sex", "age_range", "condition_count_category"])

@pytest.fixture
def expected_g20_output_malformed():
    return pl.DataFrame({
        "geo_code": ["10101"],
        "sex": ["P"],
        "age_range": ["0-4"],
        "condition_count_category": ["no_condition"],
        "count": pl.Series([40], dtype=pl.Int64)
    }).sort(["geo_code", "sex", "age_range", "condition_count_category"])

# --- Tests for process_g20_unpivot_csv ---

@patch('etl_logic.tables.g20_selected_conditions.pl.read_csv')
def test_process_g20_unpivot_csv_success(mock_read_csv, mock_csv_path, sample_g20_data_basic, expected_g20_output_basic):
    """Test successful processing using process_g20_unpivot_csv."""
    mock_read_csv.return_value = sample_g20_data_basic
    result_df = g20_selected_conditions.process_g20_unpivot_csv(mock_csv_path)
    assert result_df is not None
    assert_frame_equal(result_df.sort(["geo_code", "sex", "age_range", "condition_count_category"]), expected_g20_output_basic)

@patch('etl_logic.tables.g20_selected_conditions.pl.read_csv')
def test_process_g20_unpivot_csv_malformed(mock_read_csv, mock_csv_path, sample_g20_data_malformed, expected_g20_output_malformed, caplog):
    """Test process_g20_unpivot_csv handles malformed columns."""
    mock_read_csv.return_value = sample_g20_data_malformed
    with caplog.at_level(logging.DEBUG):
        result_df = g20_selected_conditions.process_g20_unpivot_csv(mock_csv_path)
    assert result_df is not None
    assert_frame_equal(result_df.sort(["geo_code", "sex", "age_range", "condition_count_category"]), expected_g20_output_malformed)
    assert "Failed to fully parse column" in caplog.text or "Failed to find condition count category" in caplog.text

@patch('etl_logic.tables.g20_selected_conditions.pl.read_csv')
def test_process_g20_unpivot_csv_no_geo(mock_read_csv, mock_csv_path, caplog):
    """Test process_g20_unpivot_csv failure when no geographic column is found."""
    mock_read_csv.return_value = pl.DataFrame({"Some_Data": [1]})
    with caplog.at_level(logging.ERROR):
        result_df = g20_selected_conditions.process_g20_unpivot_csv(mock_csv_path)
    assert result_df is None
    assert "Could not find geographic code column" in caplog.text

@patch('etl_logic.tables.g20_selected_conditions.pl.read_csv')
def test_process_g20_unpivot_csv_no_parsable(mock_read_csv, mock_csv_path, caplog):
    """Test process_g20_unpivot_csv failure when no columns can be parsed."""
    mock_read_csv.return_value = pl.DataFrame({"SA1_CODE_2021": ["GEO1"], "ColA": [1]})
    with caplog.at_level(logging.ERROR):
        result_df = g20_selected_conditions.process_g20_unpivot_csv(mock_csv_path)
    assert result_df is None
    assert "No valid columns found to process" in caplog.text

@patch('etl_logic.tables.g20_selected_conditions.pl.read_csv')
def test_process_g20_unpivot_csv_read_error(mock_read_csv, mock_csv_path, caplog):
    """Test process_g20_unpivot_csv failure when read_csv raises an exception."""
    mock_read_csv.side_effect = Exception("Read Error")
    with caplog.at_level(logging.ERROR):
        result_df = g20_selected_conditions.process_g20_unpivot_csv(mock_csv_path)
    assert result_df is None
    assert "Error processing file" in caplog.text
    assert "Read Error" in caplog.text

# --- Tests for process_g20_file (Fallback Generic Processor) ---

# Minimal config for the fallback test
MOCK_G20_FLAT_CONFIG = {
    "G20": {
        "measure_column_map": {
            'no_condition_total': ['P_Tot_No_cond'],
            'one_condition_total': ['P_Tot_One_cond']
        },
        "required_target_columns": ['no_condition_total']
    }
}

@patch('etl_logic.tables.g20_selected_conditions.census._process_single_census_csv')
@patch('etl_logic.tables.g20_selected_conditions.config.CENSUS_COLUMN_MAPPINGS', MOCK_G20_FLAT_CONFIG)
def test_process_g20_file_calls_generic(mock_process_csv, mock_csv_path):
    """Test process_g20_file calls the generic processor correctly when configured."""
    # Arrange
    mock_output_df = pl.DataFrame({"region_id": ["123"], "no_condition_total": [100]})
    mock_process_csv.return_value = mock_output_df
    expected_geo_options = ['region_id', 'SA1_CODE21', 'SA2_CODE21', 'SA1_CODE_2021', 'SA2_CODE_2021']

    # Act
    result_df = g20_selected_conditions.process_g20_file(mock_csv_path)

    # Assert
    mock_process_csv.assert_called_once_with(
        csv_file=mock_csv_path,
        geo_column_options=expected_geo_options,
        measure_column_map=MOCK_G20_FLAT_CONFIG["G20"]["measure_column_map"],
        required_target_columns=MOCK_G20_FLAT_CONFIG["G20"]["required_target_columns"],
        table_code="G20"
    )
    assert result_df is not None
    assert_frame_equal(result_df, mock_output_df)

@patch('etl_logic.tables.g20_selected_conditions.census._process_single_census_csv')
@patch('etl_logic.tables.g20_selected_conditions.config.CENSUS_COLUMN_MAPPINGS', {}) # No config
def test_process_g20_file_uses_fallback_map(mock_process_csv, mock_csv_path, caplog):
    """Test process_g20_file uses its internal fallback map if config is missing."""
    # Arrange
    mock_output_df = pl.DataFrame({"region_id": ["123"], "no_condition_total": [100]})
    mock_process_csv.return_value = mock_output_df
    expected_geo_options = ['region_id', 'SA1_CODE21', 'SA2_CODE21', 'SA1_CODE_2021', 'SA2_CODE_2021']
    # Fallback map from the function itself
    fallback_map = {
            'no_condition_total': ['P_Tot_No_cond'],
            'one_condition_total': ['P_Tot_One_cond'],
            'two_conditions_total': ['P_Tot_Two_cond'],
            'three_or_more_total': ['P_Tot_Three_more'],
            'not_stated_total': ['P_Tot_Not_stated'],
        }
    fallback_required = list(fallback_map.keys())

    # Act
    with caplog.at_level(logging.WARNING):
        result_df = g20_selected_conditions.process_g20_file(mock_csv_path)

    # Assert
    assert "Mappings not found in config, using limited fallback" in caplog.text
    assert "Using generic _process_single_census_csv for G20" in caplog.text
    mock_process_csv.assert_called_once_with(
        csv_file=mock_csv_path,
        geo_column_options=expected_geo_options,
        measure_column_map=fallback_map,
        required_target_columns=fallback_required,
        table_code="G20"
    )
    assert result_df is not None
    assert_frame_equal(result_df, mock_output_df) 