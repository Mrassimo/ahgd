import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import polars as pl
from polars.testing import assert_frame_equal

from etl_logic.tables import g19_health_conditions
from etl_logic import utils, config

@pytest.fixture
def mock_csv_path():
    """Provides a dummy Path object for the CSV file."""
    return Path("dummy/path/to/G19_TEST.csv")

# --- Fixtures for process_g19_file ---

@pytest.fixture
def sample_g19_data_basic():
    return pl.DataFrame({
        "SA1_CODE_2021": ["101011001", "101011002"],
        "P_Arthritis_0_14": [2, 3],
        "F_Asthma_15_24": [5, 8],
        "M_NS_85_over": [1, 0], # Not Stated
        "P_No_condition_Tot": [100, 150]
    })

@pytest.fixture
def sample_g19_data_variations():
    return pl.DataFrame({
        "region_id": ["GEO1"],
        "P_Arth_0_14": [4], # Abbreviated condition
        "M_LTHC_NS_85_over": [2], # Variation of Not Stated
        "F_None_Tot": [90] # Variation for No Condition
    })

@pytest.fixture
def sample_g19_data_malformed():
    return pl.DataFrame({
        "SA2_CODE_2021": ["10101"],
        "P_Arthritis_0_14": [3],
        "INVALID_COLUMN": [9],
        "M_INVALID_CONDITION_65_74": [2]
    })

@pytest.fixture
def expected_g19_output_basic():
    return pl.DataFrame({
        "geo_code": ["101011001", "101011002", "101011001", "101011002", "101011001", "101011001", "101011002"],
        "sex": ["P", "P", "F", "F", "M", "P", "P"],
        "condition": ["arthritis", "arthritis", "asthma", "asthma", "not_stated", "no_condition", "no_condition"],
        "age_range": ["0-14", "0-14", "15-24", "15-24", "85+", "total", "total"],
        "count": pl.Series([2, 3, 5, 8, 1, 100, 150], dtype=pl.Int64)
    }).sort(["geo_code", "sex", "condition", "age_range"])

@pytest.fixture
def expected_g19_output_variations():
    return pl.DataFrame({
        "geo_code": ["GEO1", "GEO1", "GEO1"],
        "sex": ["P", "M", "F"],
        "condition": ["arthritis", "not_stated", "no_condition"],
        "age_range": ["0-14", "85+", "total"],
        "count": pl.Series([4, 2, 90], dtype=pl.Int64)
    }).sort(["geo_code", "sex", "condition", "age_range"])

@pytest.fixture
def expected_g19_output_malformed():
    return pl.DataFrame({
        "geo_code": ["10101"],
        "sex": ["P"],
        "condition": ["arthritis"],
        "age_range": ["0-14"],
        "count": pl.Series([3], dtype=pl.Int64)
    }).sort(["geo_code", "sex", "condition", "age_range"])

# --- Tests for process_g19_file ---

@patch('etl_logic.tables.g19_health_conditions.pl.read_csv')
@patch('etl_logic.tables.g19_health_conditions.config.G19_UNPIVOT', {
    "sex_prefixes": ["M", "F", "P"],  # Male, Female, Person
    "health_conditions_map": {
        "Arthritis": "arthritis",
        "Asthma": "asthma",
        "NS": "not_stated",
        "No_condition": "no_condition"
    },
    "age_range_patterns": {
        "0_14": "0-14",
        "15_24": "15-24",
        "85_over": "85+",
        "Tot": "total"
    }
})
@patch('etl_logic.tables.g19_health_conditions.config.GEO_COLUMN_OPTIONS', ['SA1_CODE_2021', 'SA2_CODE_2021', 'region_id'])
def test_process_g19_file_success_basic(mock_read_csv, mock_csv_path, sample_g19_data_basic, expected_g19_output_basic):
    """Test successful processing of a basic G19 file while verifying config usage."""
    mock_read_csv.return_value = sample_g19_data_basic
    result_df = g19_health_conditions.process_g19_file(mock_csv_path)
    
    # Verify config was accessed
    assert hasattr(g19_health_conditions, 'config')
    assert g19_health_conditions.config.G19_UNPIVOT["sex_prefixes"] == ["M", "F", "P"]
    
    # Verify processing results
    assert result_df is not None
    assert_frame_equal(result_df.sort(["geo_code", "sex", "condition", "age_range"]), expected_g19_output_basic)

@patch('etl_logic.tables.g19_health_conditions.pl.read_csv')
@patch('etl_logic.tables.g19_health_conditions.config.G19_UNPIVOT', {
    "sex_prefixes": ["M", "F", "P"],  # Male, Female, Person
    "health_conditions_map": {
        "Arth": "arthritis",
        "LTHC_NS": "not_stated",
        "None": "no_condition"
    },
    "age_range_patterns": {
        "0_14": "0-14",
        "85_over": "85+",
        "Tot": "total"
    }
})
@patch('etl_logic.tables.g19_health_conditions.config.GEO_COLUMN_OPTIONS', ['region_id'])
def test_process_g19_file_success_variations(mock_read_csv, mock_csv_path, sample_g19_data_variations, expected_g19_output_variations):
    """Test successful processing with variations while verifying config usage."""
    mock_read_csv.return_value = sample_g19_data_variations
    result_df = g19_health_conditions.process_g19_file(mock_csv_path)
    
    # Verify config was accessed with expected values
    assert g19_health_conditions.config.G19_UNPIVOT["health_conditions_map"]["Arth"] == "arthritis"
    
    # Verify processing results
    assert result_df is not None
    assert_frame_equal(result_df.sort(["geo_code", "sex", "condition", "age_range"]), expected_g19_output_variations)

@patch('etl_logic.tables.g19_health_conditions.pl.read_csv')
def test_process_g19_file_malformed_cols(mock_read_csv, mock_csv_path, sample_g19_data_malformed, expected_g19_output_malformed, caplog):
    """Test processing handles and ignores malformed columns."""
    mock_read_csv.return_value = sample_g19_data_malformed
    with caplog.at_level(logging.DEBUG):
        result_df = g19_health_conditions.process_g19_file(mock_csv_path)
    assert result_df is not None
    assert_frame_equal(result_df.sort(["geo_code", "sex", "condition", "age_range"]), expected_g19_output_malformed)
    assert "Could not parse condition" in caplog.text or "Could not parse age range" in caplog.text # Check log for ignored cols

@patch('etl_logic.tables.g19_health_conditions.pl.read_csv')
def test_process_g19_file_no_geo_column(mock_read_csv, mock_csv_path, caplog):
    """Test failure when no geographic column is found."""
    mock_read_csv.return_value = pl.DataFrame({"Some_Data": [1]})
    with caplog.at_level(logging.ERROR):
        result_df = g19_health_conditions.process_g19_file(mock_csv_path)
    assert result_df is None
    assert "Could not find geographic code column" in caplog.text

@patch('etl_logic.tables.g19_health_conditions.pl.read_csv')
def test_process_g19_file_no_parsable_columns(mock_read_csv, mock_csv_path, caplog):
    """Test failure when no columns can be parsed."""
    mock_read_csv.return_value = pl.DataFrame({"SA1_CODE_2021": ["GEO1"], "ColA": [1]})
    with caplog.at_level(logging.ERROR):
        result_df = g19_health_conditions.process_g19_file(mock_csv_path)
    assert result_df is None
    assert "No valid columns found to unpivot" in caplog.text

@patch('etl_logic.tables.g19_health_conditions.pl.read_csv')
def test_process_g19_file_read_error(mock_read_csv, mock_csv_path, caplog):
    """Test failure when read_csv raises an exception."""
    mock_read_csv.side_effect = Exception("Read Error")
    with caplog.at_level(logging.ERROR):
        result_df = g19_health_conditions.process_g19_file(mock_csv_path)
    assert result_df is None
    assert "Error processing G19 file" in caplog.text
    assert "Read Error" in caplog.text

# --- Fixtures for process_g19_detailed_csv (can reuse some logic if applicable) ---
# Assuming detailed has similar structure but maybe different cols/mappings expected
# Add specific fixtures for detailed if needed, otherwise reuse basic/variations

# --- Tests for process_g19_detailed_csv ---

# Example: Reuse basic data if structure is similar
@patch('etl_logic.tables.g19_health_conditions.pl.read_csv')
@patch('etl_logic.tables.g19_health_conditions.config.G19_UNPIVOT', {
    "sex_prefixes": ["M", "F", "P"],  # Male, Female, Person
    "health_conditions_map": {
        "Arthritis": "arthritis",
        "Asthma": "asthma",
        "NS": "not_stated",
        "No_condition": "no_condition"
    },
    "age_range_patterns": {
        "0_14": "0-14",
        "15_24": "15-24",
        "85_over": "85+",
        "Tot": "total"
    }
})
@patch('etl_logic.tables.g19_health_conditions.config.GEO_COLUMN_OPTIONS', ['SA1_CODE_2021', 'SA2_CODE_2021'])
def test_process_g19_detailed_csv_success(mock_read_csv, mock_csv_path, sample_g19_data_basic, expected_g19_output_basic):
    """Test successful processing via process_g19_detailed_csv while verifying config usage."""
    # Arrange
    mock_read_csv.return_value = sample_g19_data_basic

    # Act
    result_df = g19_health_conditions.process_g19_detailed_csv(mock_csv_path)

    # Verify config was accessed with expected values
    assert g19_health_conditions.config.G19_UNPIVOT["age_range_patterns"]["0_14"] == "0-14"
    
    # Assert
    mock_read_csv.assert_called_once_with(mock_csv_path, truncate_ragged_lines=True)
    assert result_df is not None
    # Compare against the *same* expected output as process_g19_file for this example
    # If detailed logic differs, create a new expected fixture
    assert_frame_equal(result_df.sort(["geo_code", "sex", "condition", "age_range"]), expected_g19_output_basic)

@patch('etl_logic.tables.g19_health_conditions.pl.read_csv')
def test_process_g19_detailed_csv_no_geo(mock_read_csv, mock_csv_path, caplog):
    """Test process_g19_detailed_csv handles missing geo column."""
    # Arrange
    mock_read_csv.return_value = pl.DataFrame({"Some_Data": [1]})

    # Act
    with caplog.at_level(logging.ERROR):
        result_df = g19_health_conditions.process_g19_detailed_csv(mock_csv_path)

    # Assert
    assert result_df is None
    assert "Could not find geographic code column" in caplog.text

@patch('etl_logic.tables.g19_health_conditions.pl.read_csv')
def test_process_g19_detailed_csv_read_error(mock_read_csv, mock_csv_path, caplog):
    """Test process_g19_detailed_csv handles read errors."""
    # Arrange
    mock_read_csv.side_effect = Exception("Detailed Read Error")

    # Act
    with caplog.at_level(logging.ERROR):
        result_df = g19_health_conditions.process_g19_detailed_csv(mock_csv_path)

    # Assert
    assert result_df is None
    assert "Error processing G19_detailed file" in caplog.text # Note the different table_code
    assert "Detailed Read Error" in caplog.text

# TODO: Add more tests for process_g19_detailed_csv if its logic
# significantly differs from process_g19_file (e.g., different column patterns,
# different mappings, different required columns). 