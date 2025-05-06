import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from etl_logic import config
import polars as pl
from polars.testing import assert_frame_equal

from etl_logic.tables import g17_income
from etl_logic import utils # Assuming utils contains helper functions

@pytest.fixture
def mock_g17_config():
    """Provides mocked G17_UNPIVOT config for testing."""
    return {
        "income_categories": {
            "Neg_Nil_income": "negative_nil_income",
            "150_299": "income_150_299",
            "3000_3499": "income_3000_3499",
            "Tot": "total"
        },
        "age_range_patterns": {
            "15_19_yrs": "15-19",
            "20_24_yrs": "20-24",
            "85_yrs_ovr": "85+",
            "Tot": "total"
        }
    }

@pytest.fixture
def mock_csv_path():
    """Provides a dummy Path object for the CSV file."""
    return Path("dummy/path/to/G17_TEST.csv")

# --- Test Data Fixtures ---

@pytest.fixture
def sample_g17_data_basic():
    """Basic valid G17 data."""
    return pl.DataFrame({
        "SA1_CODE_2021": ["101011001", "101011002"],
        "P_Neg_Nil_income_15_19_yrs": [10, 20],
        "F_150_299_20_24_yrs": [5, 8],
        "M_3000_3499_85_yrs_ovr": [1, 0], # Includes a zero value
        "P_Tot_Tot": [100, 150] # Total column often exists
    })

@pytest.fixture
def sample_g17_data_alt_geo():
    """G17 data with a different geo column name."""
    return pl.DataFrame({
        "region_id": ["GEO1", "GEO2"],
        "P_Neg_Nil_income_15_19_yrs": [15, 25],
        "M_1_149_35_44_yrs": [7, 9],
    })

@pytest.fixture
def sample_g17_data_malformed_cols():
    """G17 data with some columns that won't parse."""
    return pl.DataFrame({
        "SA2_CODE_2021": ["10101", "10102"],
        "P_Neg_Nil_income_15_19_yrs": [10, 20],
        "INVALID_COLUMN": [1, 2], # Should be ignored
        "F_150_299": [3, 4],      # Missing age part
        "M_3000_3499_INVALID_AGE": [5, 6] # Invalid age part
    })

@pytest.fixture
def expected_g17_output_basic():
    """Expected output for sample_g17_data_basic."""
    return pl.DataFrame({
        "geo_code": ["101011001", "101011002", "101011001", "101011002", "101011001", "101011001", "101011002"],
        "sex": ["P", "P", "F", "F", "M", "P", "P"],
        "income_category": [
            "negative_nil_income", "negative_nil_income",
            "income_150_299", "income_150_299",
            "income_3000_3499",
            "total", "total"
        ],
        "age_range": ["15-19", "15-19", "20-24", "20-24", "85+", "total", "total"],
        "count": pl.Series([10, 20, 5, 8, 1, 100, 150], dtype=pl.Int64)
    }).sort(["geo_code", "sex", "income_category", "age_range"])

@pytest.fixture
def expected_g17_output_malformed():
    """Expected output for sample_g17_data_malformed_cols (only valid cols)."""
    return pl.DataFrame({
        "geo_code": ["10101", "10102"],
        "sex": ["P", "P"],
        "income_category": ["negative_nil_income", "negative_nil_income"],
        "age_range": ["15-19", "15-19"],
        "count": pl.Series([10, 20], dtype=pl.Int64)
    }).sort(["geo_code", "sex", "income_category", "age_range"])


# --- Test Cases ---

@patch('etl_logic.tables.g17_income.pl.read_csv')
@patch('etl_logic.tables.g17_income.config.G17_UNPIVOT', new_callable=lambda: {
        "income_categories": {
            "Neg_Nil_income": "negative_nil_income",
            "150_299": "income_150_299",
            "3000_3499": "income_3000_3499",
            "Tot": "total"
        },
        "age_range_patterns": {
            "15_19_yrs": "15-19",
            "20_24_yrs": "20-24",
            "85_yrs_ovr": "85+",
            "Tot": "total"
        }
    })
def test_process_g17_file_success_basic(mock_read_csv, mock_csv_path, sample_g17_data_basic, expected_g17_output_basic):
    """Test successful processing of a basic G17 file."""
    # Arrange
    mock_read_csv.return_value = sample_g17_data_basic

    # Act
    result_df = g17_income.process_g17_file(mock_csv_path)

    # Assert
    mock_read_csv.assert_called_once_with(mock_csv_path, truncate_ragged_lines=True)
    assert result_df is not None
    # Sort both dataframes for consistent comparison
    assert_frame_equal(result_df.sort(["geo_code", "sex", "income_category", "age_range"]), expected_g17_output_basic)

@patch('etl_logic.tables.g17_income.pl.read_csv')
@patch('etl_logic.tables.g17_income.config.G17_UNPIVOT', new_callable=lambda: {
        "income_categories": {
            "Neg_Nil_income": "negative_nil_income",
            "150_299": "income_150_299",
            "3000_3499": "income_3000_3499",
            "Tot": "total"
        },
        "age_range_patterns": {
            "15_19_yrs": "15-19",
            "20_24_yrs": "20-24",
            "85_yrs_ovr": "85+",
            "Tot": "total"
        }
    })
def test_process_g17_file_success_alt_geo(mock_read_csv, mock_csv_path, sample_g17_data_alt_geo):
    """Test successful processing with an alternative geo column name."""
    # Arrange
    mock_read_csv.return_value = sample_g17_data_alt_geo
    expected_output = pl.DataFrame({
        "geo_code": ["GEO1", "GEO2", "GEO1", "GEO2"],
        "sex": ["P", "P", "M", "M"],
        "income_category": ["negative_nil_income", "negative_nil_income", "income_1_149", "income_1_149"],
        "age_range": ["15-19", "15-19", "35-44", "35-44"],
        "count": pl.Series([15, 25, 7, 9], dtype=pl.Int64)
    }).sort(["geo_code", "sex", "income_category", "age_range"])


    # Act
    result_df = g17_income.process_g17_file(mock_csv_path)

    # Assert
    mock_read_csv.assert_called_once_with(mock_csv_path, truncate_ragged_lines=True)
    assert result_df is not None
    assert_frame_equal(result_df.sort(["geo_code", "sex", "income_category", "age_range"]), expected_output)

@patch('etl_logic.tables.g17_income.pl.read_csv')
@patch('etl_logic.tables.g17_income.config.G17_UNPIVOT', new_callable=lambda: {
        "income_categories": {
            "Neg_Nil_income": "negative_nil_income",
            "150_299": "income_150_299",
            "3000_3499": "income_3000_3499",
            "Tot": "total"
        },
        "age_range_patterns": {
            "15_19_yrs": "15-19",
            "20_24_yrs": "20-24",
            "85_yrs_ovr": "85+",
            "Tot": "total"
        }
    })
def test_process_g17_file_malformed_cols(mock_read_csv, mock_csv_path, sample_g17_data_malformed_cols, expected_g17_output_malformed, caplog):
    """Test processing handles and ignores malformed columns."""
    # Arrange
    mock_read_csv.return_value = sample_g17_data_malformed_cols

    # Act
    with caplog.at_level(logging.DEBUG): # Capture debug messages about parsing failures
      result_df = g17_income.process_g17_file(mock_csv_path)

    # Assert
    mock_read_csv.assert_called_once_with(mock_csv_path, truncate_ragged_lines=True)
    assert result_df is not None
    assert_frame_equal(result_df.sort(["geo_code", "sex", "income_category", "age_range"]), expected_g17_output_malformed)
    # Check logs for ignored columns (adjust based on actual logging messages)
    assert "Could not parse age range 'INVALID_AGE' in column M_3000_3499_INVALID_AGE" in caplog.text
    # Add checks for other expected logging messages if applicable

@patch('etl_logic.tables.g17_income.pl.read_csv')
@patch('etl_logic.tables.g17_income.config.G17_UNPIVOT', new_callable=lambda: {
        "income_categories": {
            "Neg_Nil_income": "negative_nil_income",
            "150_299": "income_150_299",
            "3000_3499": "income_3000_3499",
            "Tot": "total"
        },
        "age_range_patterns": {
            "15_19_yrs": "15-19",
            "20_24_yrs": "20-24",
            "85_yrs_ovr": "85+",
            "Tot": "total"
        }
    })
def test_process_g17_file_no_geo_column(mock_read_csv, mock_csv_path, caplog):
    """Test failure when no geographic column is found."""
    # Arrange
    df_no_geo = pl.DataFrame({"Some_Data": [1, 2]})
    mock_read_csv.return_value = df_no_geo

    # Act
    with caplog.at_level(logging.ERROR):
      result_df = g17_income.process_g17_file(mock_csv_path)

    # Assert
    assert result_df is None
    assert "Could not find geographic code column" in caplog.text

@patch('etl_logic.tables.g17_income.pl.read_csv')
@patch('etl_logic.tables.g17_income.config.G17_UNPIVOT', new_callable=lambda: {
        "income_categories": {
            "Neg_Nil_income": "negative_nil_income",
            "150_299": "income_150_299",
            "3000_3499": "income_3000_3499",
            "Tot": "total"
        },
        "age_range_patterns": {
            "15_19_yrs": "15-19",
            "20_24_yrs": "20-24",
            "85_yrs_ovr": "85+",
            "Tot": "total"
        }
    })
def test_process_g17_file_no_parsable_columns(mock_read_csv, mock_csv_path, caplog):
    """Test failure when no columns can be parsed into sex/income/age."""
    # Arrange
    df_no_parsable = pl.DataFrame({"region_id": ["GEO1"], "ColA": [1], "ColB": [2]})
    mock_read_csv.return_value = df_no_parsable

    # Act
    with caplog.at_level(logging.ERROR):
      result_df = g17_income.process_g17_file(mock_csv_path)

    # Assert
    assert result_df is None
    assert "No valid columns found to unpivot" in caplog.text

@patch('etl_logic.tables.g17_income.pl.read_csv')
@patch('etl_logic.tables.g17_income.config.G17_UNPIVOT', new_callable=lambda: {
        "income_categories": {
            "Neg_Nil_income": "negative_nil_income",
            "150_299": "income_150_299",
            "3000_3499": "income_3000_3499",
            "Tot": "total"
        },
        "age_range_patterns": {
            "15_19_yrs": "15-19",
            "20_24_yrs": "20-24",
            "85_yrs_ovr": "85+",
            "Tot": "total"
        }
    })
def test_process_g17_file_read_error(mock_read_csv, mock_csv_path, caplog):
    """Test failure when polars.read_csv raises an exception."""
    # Arrange
    mock_read_csv.side_effect = Exception("Failed to read file")

    # Act
    with caplog.at_level(logging.ERROR):
      result_df = g17_income.process_g17_file(mock_csv_path)

    # Assert
    assert result_df is None
    assert "Error processing G17 file" in caplog.text
    assert "Failed to read file" in caplog.text 
def test_config_access_in_process_g17_file(mocker, mock_csv_path):
    """Test that process_g17_file accesses and uses the centralized config variables."""
    # Mock G17_UNPIVOT config
    mock_config = {
        "income_categories": {"Neg_Nil_income": "negative_nil_income"},
        "age_range_patterns": {"15_19_yrs": "15-19"}
    }
    mocker.patch('etl_logic.tables.g17_income.config.G17_UNPIVOT', mock_config)
    
    # Mock the CSV reading and geo column detection
    mock_df = pl.DataFrame({
        "SA1_CODE_2021": ["101011001"],
        "P_Neg_Nil_income_15_19_yrs": [10],
        "P_Tot_Tot": [100]
    })
    mocker.patch('etl_logic.tables.g17_income.pl.read_csv', return_value=mock_df)
    mocker.patch('etl_logic.utils.find_geo_column', return_value="SA1_CODE_2021")

    # Call the function
    result = g17_income.process_g17_file(mock_csv_path)
    
    # Verify config was used by checking the output matches our mock values
    assert result is not None
    expected_df = pl.DataFrame({
        "geo_code": ["101011001", "101011001"],
        "sex": ["P", "P"],
        "income_category": ["negative_nil_income", "total"],
        "age_range": ["15-19", "total"],
        "count": pl.Series([10, 100], dtype=pl.Int64)
    })
    assert_frame_equal(result.sort(["geo_code", "sex", "income_category", "age_range"]), 
                      expected_df.sort(["geo_code", "sex", "income_category", "age_range"]))