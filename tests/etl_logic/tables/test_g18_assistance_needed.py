import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import polars as pl
from polars.testing import assert_frame_equal

from etl_logic.tables import g18_assistance_needed
from etl_logic import utils

@pytest.fixture
def mock_csv_path():
    """Provides a dummy Path object for the CSV file."""
    return Path("dummy/path/to/G18_TEST.csv")

# --- Test Data Fixtures ---

@pytest.fixture
def sample_g18_data_basic():
    """Basic valid G18 data."""
    return pl.DataFrame({
        "SA1_CODE_2021": ["101011001", "101011002"],
        "P_0_4_Need_for_assistance": [5, 10],
        "F_15_19_No_need_for_assistance": [50, 60],
        "M_85_over_Not_stated": [1, 0], # Includes zero value
        "P_Tot_Tot": [200, 250] # Total column (should be parsed if possible, or ignored)
    })

@pytest.fixture
def sample_g18_data_variations():
    """G18 data with variations in column names."""
    return pl.DataFrame({
        "region_id": ["GEO1"],
        "P_0_4_Need_for_assistance": [8],
        "M_85ov_Need_for_assistance_ns": [2], # variation of age and assistance status
        "F_5_14_No_need_for_assistance": [30]
    })

@pytest.fixture
def sample_g18_data_malformed():
    """G18 data with unparsable columns."""
    return pl.DataFrame({
        "SA2_CODE_2021": ["10101"],
        "P_0_4_Need_for_assistance": [3],
        "INVALID_COLUMN": [9],
        "M_65_74_INVALID_STATUS": [2] # Invalid assistance status
    })

@pytest.fixture
def expected_g18_output_basic():
    """Expected output for sample_g18_data_basic."""
    return pl.DataFrame({
        "geo_code": ["101011001", "101011002", "101011001", "101011002", "101011001", "101011001", "101011002"],
        "sex": ["P", "P", "F", "F", "M", "P", "P"],
        "assistance_category": [
            "needs_assistance", "needs_assistance",
            "no_need_for_assistance", "no_need_for_assistance",
            "assistance_not_stated",
            "total", "total" # Assuming P_Tot_Tot parses as total assistance/age
        ],
        "age_range": ["0-4", "0-4", "15-19", "15-19", "85+", "total", "total"],
        "count": pl.Series([5, 10, 50, 60, 1, 200, 250], dtype=pl.Int64)
    }).sort(["geo_code", "sex", "assistance_category", "age_range"])

@pytest.fixture
def expected_g18_output_variations():
    """Expected output for sample_g18_data_variations."""
    return pl.DataFrame({
        "geo_code": ["GEO1", "GEO1", "GEO1"],
        "sex": ["P", "M", "F"],
        "assistance_category": ["needs_assistance", "assistance_not_stated", "no_need_for_assistance"],
        "age_range": ["0-4", "85+", "5-14"],
        "count": pl.Series([8, 2, 30], dtype=pl.Int64)
    }).sort(["geo_code", "sex", "assistance_category", "age_range"])

@pytest.fixture
def expected_g18_output_malformed():
    """Expected output for sample_g18_data_malformed (only valid cols)."""
    return pl.DataFrame({
        "geo_code": ["10101"],
        "sex": ["P"],
        "assistance_category": ["needs_assistance"],
        "age_range": ["0-4"],
        "count": pl.Series([3], dtype=pl.Int64)
    }).sort(["geo_code", "sex", "assistance_category", "age_range"])

# --- Test Cases ---

@patch('etl_logic.tables.g18_assistance_needed.pl.read_csv')
@patch('etl_logic.tables.g18_assistance_needed.G18_UNPIVOT')
def test_process_g18_file_success_basic(mock_g18_unpivot, mock_read_csv, mock_csv_path,
                                      sample_g18_data_basic, expected_g18_output_basic):
    """Test successful processing of a basic G18 file."""
    # Arrange mock config data that matches test expectations
    mock_g18_unpivot.return_value = {
        "sex_prefixes": ["M", "F", "P"],
        "assistance_categories": {
            "Need_for_assistance": "needs_assistance",
            "No_need_for_assistance": "no_need_for_assistance",
            "Need_for_assistance_ns": "assistance_not_stated",
            "Not_stated": "assistance_not_stated"
        },
        "age_range_patterns": {
            "0_4": "0-4",
            "15_19": "15-19",
            "85_over": "85+",
            "Tot": "total"
        }
    }
    mock_read_csv.return_value = sample_g18_data_basic

    # Act
    result_df = g18_assistance_needed.process_g18_file(mock_csv_path)

    # Assert
    mock_read_csv.assert_called_once_with(mock_csv_path, truncate_ragged_lines=True)
    mock_g18_unpivot.assert_called_once()  # Verify config was accessed
    assert result_df is not None
    assert_frame_equal(result_df.sort(["geo_code", "sex", "assistance_category", "age_range"]), expected_g18_output_basic)

@patch('etl_logic.tables.g18_assistance_needed.pl.read_csv')
@patch('etl_logic.tables.g18_assistance_needed.G18_UNPIVOT')
def test_process_g18_file_success_variations(mock_g18_unpivot, mock_read_csv, mock_csv_path,
                                           sample_g18_data_variations, expected_g18_output_variations):
    """Test successful processing with variations in column names."""
    # Arrange mock config data that matches test expectations
    mock_g18_unpivot.return_value = {
        "sex_prefixes": ["M", "F", "P"],
        "assistance_categories": {
            "Need_for_assistance": "needs_assistance",
            "No_need_for_assistance": "no_need_for_assistance",
            "Need_for_assistance_ns": "assistance_not_stated",
            "Not_stated": "assistance_not_stated"
        },
        "age_range_patterns": {
            "0_4": "0-4",
            "5_14": "5-14",
            "85ov": "85+",
            "Tot": "total"
        }
    }
    mock_read_csv.return_value = sample_g18_data_variations

    # Act
    result_df = g18_assistance_needed.process_g18_file(mock_csv_path)

    # Assert
    mock_read_csv.assert_called_once_with(mock_csv_path, truncate_ragged_lines=True)
    mock_g18_unpivot.assert_called_once()  # Verify config was accessed
    assert result_df is not None
    assert_frame_equal(result_df.sort(["geo_code", "sex", "assistance_category", "age_range"]), expected_g18_output_variations)

@patch('etl_logic.tables.g18_assistance_needed.pl.read_csv')
def test_process_g18_file_malformed_cols(mock_read_csv, mock_csv_path, sample_g18_data_malformed, expected_g18_output_malformed, caplog):
    """Test processing handles and ignores malformed columns."""
    # Arrange
    mock_read_csv.return_value = sample_g18_data_malformed

    # Act
    with caplog.at_level(logging.DEBUG):
      result_df = g18_assistance_needed.process_g18_file(mock_csv_path)

    # Assert
    mock_read_csv.assert_called_once_with(mock_csv_path, truncate_ragged_lines=True)
    assert result_df is not None
    assert_frame_equal(result_df.sort(["geo_code", "sex", "assistance_category", "age_range"]), expected_g18_output_malformed)
    # Check logs for ignored columns
    assert "Could not parse assistance category" in caplog.text or "Could not parse age range" in caplog.text

@patch('etl_logic.tables.g18_assistance_needed.pl.read_csv')
def test_process_g18_file_no_geo_column(mock_read_csv, mock_csv_path, caplog):
    """Test failure when no geographic column is found."""
    # Arrange
    df_no_geo = pl.DataFrame({"Some_Data": [1, 2]})
    mock_read_csv.return_value = df_no_geo

    # Act
    with caplog.at_level(logging.ERROR):
      result_df = g18_assistance_needed.process_g18_file(mock_csv_path)

    # Assert
    assert result_df is None
    assert "Could not find geographic code column" in caplog.text

@patch('etl_logic.tables.g18_assistance_needed.pl.read_csv')
def test_process_g18_file_no_parsable_columns(mock_read_csv, mock_csv_path, caplog):
    """Test failure when no columns can be parsed into sex/age/assistance."""
    # Arrange
    df_no_parsable = pl.DataFrame({"SA1_CODE_2021": ["GEO1"], "ColA": [1], "ColB": [2]})
    mock_read_csv.return_value = df_no_parsable

    # Act
    with caplog.at_level(logging.ERROR):
      result_df = g18_assistance_needed.process_g18_file(mock_csv_path)

    # Assert
    assert result_df is None
    assert "No valid columns found to unpivot" in caplog.text

@patch('etl_logic.tables.g18_assistance_needed.pl.read_csv')
def test_process_g18_file_read_error(mock_read_csv, mock_csv_path, caplog):
    """Test failure when polars.read_csv raises an exception."""
    # Arrange
    mock_read_csv.side_effect = Exception("Failed to read file")

    # Act
    with caplog.at_level(logging.ERROR):
      result_df = g18_assistance_needed.process_g18_file(mock_csv_path)

    # Assert
    assert result_df is None
    assert "Error processing G18 file" in caplog.text
    assert "Failed to read file" in caplog.text 