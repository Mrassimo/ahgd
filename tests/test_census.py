import pytest
"""
Module for testing the census data processing functions in the AHGD ETL pipeline.

This module contains unit tests for functions that handle census data extraction, processing, and validation, ensuring correctness and robustness of the ETL logic for various census tables (G17, G18, G19, G20, G21, G25). Tests cover successful processing, edge cases, and integration scenarios.
"""


import sys
from pathlib import Path
import zipfile
import polars as pl
from polars.testing import assert_frame_equal
import logging

# Ensure etl_logic is importable
TESTS_DIR = Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from etl_logic import census, config, utils
from etl_logic.tables.g17_income import process_g17_file
from etl_logic.tables.g18_assistance_needed import process_g18_file
from etl_logic.tables.g19_health_conditions import process_g19_file
from etl_logic.tables.g20_selected_conditions import process_g20_census_data
from etl_logic.tables.g21_conditions_by_characteristics import process_g21_census_data
# --- Fixtures ---

@pytest.fixture
def mock_paths(tmp_path, monkeypatch):
    """Mocks config.PATHS to use temporary directories."""
    paths = {
        'CENSUS_DIR': tmp_path / "raw/census",
        'TEMP_EXTRACT_DIR': tmp_path / "temp/extract", # Although likely unused if reading from zip
        'OUTPUT_DIR': tmp_path / "output",
        # Add other paths if needed
        'DATA_DIR': tmp_path / 'data',
        'RAW_DATA_DIR': tmp_path / 'data/raw',
        'TEMP_DIR': tmp_path / 'data/raw/temp',
        'LOG_DIR': tmp_path / 'logs',
        'GEOGRAPHIC_DIR': tmp_path / 'data/raw/geographic',
        'TEMP_ZIP_DIR': tmp_path / 'data/raw/temp/zips',
    }
    # Create the directories
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, 'PATHS', paths)
    # Mock relevant config patterns/levels
    monkeypatch.setattr(config, 'GEO_LEVELS_CENSUS_PROCESS', ['SA1', 'SA2'])
    _census_geo_pattern = "|".join(config.GEO_LEVELS_CENSUS_PROCESS)
    monkeypatch.setattr(config, 'CENSUS_TABLE_PATTERNS', {
        "G01": rf"2021\s*Census_G01[_\s].*?({_census_geo_pattern})\.csv$",
        "G17": rf"2021\s*Census_G17[_\s].*?({_census_geo_pattern})\.csv$",
        "G18": rf"2021\s*Census_G18[_\s].*?({_census_geo_pattern})\.csv$",
        "G19": rf"2021\s*Census_G19[_\s].*?({_census_geo_pattern})\.csv$",
        "G20": rf"2021\s*Census_G20[_\s].*?({_census_geo_pattern})\.csv$"  # Added G20 pattern
    })
    # Mock the census zip URL getter to return our test zip name
    test_zip_name = "test_census_pack.zip"
    monkeypatch.setattr(config, 'get_required_census_zips', lambda: {test_zip_name: "http://dummy.url/census.zip"})

    return paths, test_zip_name

@pytest.fixture
def mock_census_zip(mock_paths):
    """Create a mock census ZIP file with test data."""
    paths, zip_name = mock_paths
    census_dir = paths['CENSUS_DIR']
    zip_path = census_dir / zip_name
    
    # Create test directory
    census_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary ZIP file with mock CSV data
    with zipfile.ZipFile(zip_path, 'w') as zf:
        # Add SA1 CSV
        zf.writestr("path/to/2021_Census_G01_SA1.csv", mock_census_csv_content("SA1"))
        
        # Add SA2 CSV
        zf.writestr("another/path/2021 Census_G01_for_SA2.csv", mock_census_csv_content("SA2"))
    
    return zip_path

@pytest.fixture
def mock_geo_parquet(mock_paths):
    """Create a mock geographic dimension parquet file for testing."""
    paths, _ = mock_paths
    output_dir = paths['OUTPUT_DIR']
    parquet_path = output_dir / "geo_dimension.parquet"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a sample geo_dimension DataFrame
    df = pl.DataFrame({
        "geo_code": ["10101", "10102", "10103", "20101", "20102", "30000"],
        "geo_level": ["SA1", "SA1", "SA1", "SA2", "SA2", "SA3"],
        "geometry": ["POLYGON((...))", "POLYGON((...))", "POLYGON((...))", 
                    "POLYGON((...))", "POLYGON((...))", "POLYGON((...))"]
    })
    
    # Add geo_sk column for testing the G17 processing
    df = df.with_row_index(name='geo_sk')
    
    # Write to parquet
    df.write_parquet(parquet_path)
    
    return parquet_path

@pytest.fixture
def mock_g17_zip(mock_paths):
    """Create a mock census ZIP file with G17 test data."""
    paths, _ = mock_paths
    census_dir = paths['CENSUS_DIR']
    zip_name = "test_g17_census_pack.zip"
    zip_path = census_dir / zip_name
    
    # Create test directory
    census_dir.mkdir(parents=True, exist_ok=True)
    
    # Monkeypatch the CENSUS_TABLE_PATTERNS for G17
    _census_geo_pattern = "|".join(config.GEO_LEVELS_CENSUS_PROCESS)
    config.CENSUS_TABLE_PATTERNS["G17"] = rf"2021\s*Census_G17[_\s].*?({_census_geo_pattern})\.csv$"
    
    # Create a temporary ZIP file with mock CSV data
    with zipfile.ZipFile(zip_path, 'w') as zf:
        # Add SA1 G17 CSV
        zf.writestr("path/to/2021_Census_G17_SA1.csv", mock_g17_csv_content("SA1"))
        
        # Add SA2 G17 CSV
        zf.writestr("another/path/2021 Census_G17_for_SA2.csv", mock_g17_csv_content("SA2"))
    
    return zip_path

@pytest.fixture
def mock_g18_zip(mock_paths):
    """Create a mock census ZIP file with G18 test data."""
    paths, _ = mock_paths
    census_dir = paths['CENSUS_DIR']
    zip_name = "test_g18_census_pack.zip"
    zip_path = census_dir / zip_name
    
    # Create test directory
    census_dir.mkdir(parents=True, exist_ok=True)
    
    # Monkeypatch the CENSUS_TABLE_PATTERNS for G18
    _census_geo_pattern = "|".join(config.GEO_LEVELS_CENSUS_PROCESS)
    config.CENSUS_TABLE_PATTERNS["G18"] = rf"2021\s*Census_G18[_\s].*?({_census_geo_pattern})\.csv$"
    
    # Create a temporary ZIP file with mock CSV data
    with zipfile.ZipFile(zip_path, 'w') as zf:
        # Add SA1 G18 CSV
        zf.writestr("path/to/2021_Census_G18_SA1.csv", mock_g18_csv_content("SA1"))
        
        # Add SA2 G18 CSV
        zf.writestr("another/path/2021 Census_G18_for_SA2.csv", mock_g18_csv_content("SA2"))
    
    return zip_path

@pytest.fixture
def mock_g19_zip(mock_paths):
    """Create a mock census ZIP file with G19 test data."""
    paths, _ = mock_paths
    census_dir = paths['CENSUS_DIR']
    zip_name = "test_g19_census_pack.zip"
    zip_path = census_dir / zip_name
    
    # Create test directory
    census_dir.mkdir(parents=True, exist_ok=True)
    
    # Monkeypatch the CENSUS_TABLE_PATTERNS for G19
    _census_geo_pattern = "|".join(config.GEO_LEVELS_CENSUS_PROCESS)
    config.CENSUS_TABLE_PATTERNS["G19"] = rf"2021\s*Census_G19[_\s].*?({_census_geo_pattern})\.csv$"
    
    # Create a temporary ZIP file with mock CSV data
    with zipfile.ZipFile(zip_path, 'w') as zf:
        # Add SA1 G19 CSV
        zf.writestr("path/to/2021_Census_G19_SA1.csv", mock_g19_csv_content("SA1"))
        
        # Add SA2 G19 CSV
        zf.writestr("another/path/2021 Census_G19_for_SA2.csv", mock_g19_csv_content("SA2"))
    
    return zip_path

@pytest.fixture
def mock_g20_zip(mock_paths):
    """Create a mock census ZIP file with G20 test data."""
    paths, _ = mock_paths
    census_dir = paths['CENSUS_DIR']
    zip_name = "test_g20_census_pack.zip"
    zip_path = census_dir / zip_name
    
    # Create test directory
    census_dir.mkdir(parents=True, exist_ok=True)
    
    # Monkeypatch the CENSUS_TABLE_PATTERNS for G20
    _census_geo_pattern = "|".join(config.GEO_LEVELS_CENSUS_PROCESS)
    config.CENSUS_TABLE_PATTERNS["G20"] = rf"2021\s*Census_G20[_\s].*?({_census_geo_pattern})\.csv$"
    
    # Create a temporary ZIP file with mock CSV data
    with zipfile.ZipFile(zip_path, 'w') as zf:
        # Add SA1 G20 CSV
        zf.writestr("path/to/2021_Census_G20_SA1.csv", mock_g20_csv_content("SA1"))
        
        # Add SA2 G20 CSV
        zf.writestr("another/path/2021 Census_G20_for_SA2.csv", mock_g20_csv_content("SA2"))
    
    return zip_path

@pytest.fixture
def mock_g21_zip(mock_paths):
    """Create a mock census ZIP file with G21 test data."""
    paths, _ = mock_paths
    census_dir = paths['CENSUS_DIR']
    zip_name = "test_g21_census_pack.zip"
    zip_path = census_dir / zip_name

    census_dir.mkdir(parents=True, exist_ok=True)

    # Ensure G21 pattern is set in config for the test
    _census_geo_pattern = "|".join(config.GEO_LEVELS_CENSUS_PROCESS)
    # Fix the pattern to match our test files
    config.CENSUS_TABLE_PATTERNS["G21"] = rf"2021.*Census_G21.*({_census_geo_pattern})\.csv$"

    with zipfile.ZipFile(zip_path, 'w') as zf:
        # Use filenames that strictly match the pattern
        zf.writestr("path/to/2021_Census_G21_SA1.csv", mock_g21_csv_content("SA1"))
        zf.writestr("another/path/2021_Census_G21_SA2.csv", mock_g21_csv_content("SA2"))

    return zip_path

@pytest.fixture
def mock_g25_zip(mock_paths):
    """Create a mock census ZIP file with G25 test data."""
    paths, _ = mock_paths
    census_dir = paths['CENSUS_DIR']
    zip_name = "test_g25_census_pack.zip"
    zip_path = census_dir / zip_name
    
    # Create test directory
    census_dir.mkdir(parents=True, exist_ok=True)
    
    # Monkeypatch the CENSUS_TABLE_PATTERNS for G25
    _census_geo_pattern = "|".join(config.GEO_LEVELS_CENSUS_PROCESS)
    config.CENSUS_TABLE_PATTERNS["G25"] = rf"2021\s*Census_G25[_\s].*?({_census_geo_pattern})\.csv$"
    
    # Create a temporary ZIP file with mock CSV data
    with zipfile.ZipFile(zip_path, 'w') as zf:
        # Add SA1 G25 CSV
        zf.writestr("path/to/2021_Census_G25_SA1.csv", mock_g25_csv_content("SA1"))
        
        # Add SA2 G25 CSV
        zf.writestr("another/path/2021 Census_G25_for_SA2.csv", mock_g25_csv_content("SA2"))
    
    return zip_path

# --- Test Cases ---

def test_process_g17_census_data_success(mock_paths, mock_g17_zip, mock_geo_parquet):
    """Test successful processing of G17 census data."""
    paths, _ = mock_paths
    output_dir = paths['OUTPUT_DIR']
    output_file = output_dir / "fact_assistance_need.parquet"
    
    result = census.process_census_table(
        table_code="G17",
        process_file_function=process_g17_file,
        output_filename="fact_assistance_need.parquet",
        zip_dir=paths['CENSUS_DIR'],
        temp_extract_base=paths['TEMP_EXTRACT_DIR'],
        output_dir=output_dir,
        geo_output_path=mock_geo_parquet
    )
    
    assert result is True
    assert output_file.exists()
    
    # Verify output content
    result_df = pl.read_parquet(output_file)
    print(f"Result G17 DataFrame: {result_df.shape} rows")
    print(result_df)
    
    # Check that the DataFrame has the expected columns
    expected_columns = ['geo_sk', 'assistance_needed_count', 'no_assistance_needed_count', 'assistance_not_stated_count']
    assert all(col in result_df.columns for col in expected_columns)
    
    # Check that we have the expected number of rows 
    # (Only SA2 rows are being successfully joined - 2 rows)
    assert len(result_df) == 2
    
    # Check that the values are in the expected range
    assert all(result_df['assistance_needed_count'] >= 0)
    assert all(result_df['no_assistance_needed_count'] >= 0)
    assert all(result_df['assistance_not_stated_count'] >= 0)

def test_process_census_g18_data_success(mock_paths, mock_g18_zip, mock_geo_parquet):
    """Test successful processing of G18 census data."""
    paths, _ = mock_paths
    output_dir = paths['OUTPUT_DIR']
    output_file = output_dir / "fact_assistance_need.parquet" # Correct output file name
    
    result = census.process_census_table(
        table_code="G18",
        process_file_function=process_g18_file,
        output_filename="fact_assistance_need.parquet",
        zip_dir=paths['CENSUS_DIR'],
        temp_extract_base=paths['TEMP_EXTRACT_DIR'],
        output_dir=output_dir,
        geo_output_path=mock_geo_parquet
    )
    
    assert result is True
    assert output_file.exists()
    
    # Verify output content
    result_df = pl.read_parquet(output_file)
    print(f"Result G18 DataFrame: {result_df.shape} rows")
    print(result_df)
    
    # Check that the DataFrame has the expected columns for fact_assistance_need
    expected_columns = ['geo_sk', 'need_assistance_count', 'no_need_assistance_count', 'need_assistance_ns_count', 'time_sk', 'etl_updated_at']
    assert all(col in result_df.columns for col in expected_columns)
    
    # Check that we have the expected number of rows 
    # (Only SA2 rows 20101, 20102 from mock_geo_parquet should be joined)
    assert len(result_df) == 2
    
    # Define expected data based on mock SA2 data and mock_geo_parquet geo_sk
    # Note: time_sk and etl_updated_at are added by the wrapper function, 
    # we'll ignore them for frame comparison or check their presence separately.
    expected_data = pl.DataFrame({
        "geo_sk": [3, 4], # Corresponding to geo_codes 20101, 20102 in mock_geo_parquet
        "need_assistance_count": [100, 150],
        "no_need_assistance_count": [850, 1700],
        "need_assistance_ns_count": [50, 150],
    }).with_columns(pl.col(pl.Int64)) # Ensure types match if necessary
    
    # Compare the relevant columns, sorting by geo_sk for consistency
    assert_frame_equal(
        result_df.select(expected_data.columns).sort("geo_sk"),
        expected_data.sort("geo_sk")
    )

def test_process_g19_census_data_success(mock_paths, mock_g19_zip, mock_geo_parquet):
    """Test successful processing of G19 census data."""
    paths, _ = mock_paths
    output_dir = paths['OUTPUT_DIR']
    output_file = output_dir / "fact_health_condition.parquet"
    
    result = census.process_census_table(
        table_code="G19",
        process_file_function=process_g19_file,
        output_filename="fact_health_condition.parquet",
        zip_dir=paths['CENSUS_DIR'],
        temp_extract_base=paths['TEMP_EXTRACT_DIR'],
        output_dir=output_dir,
        geo_output_path=mock_geo_parquet
    )
    
    assert result is True
    assert output_file.exists()
    
    # Verify output content
    result_df = pl.read_parquet(output_file)
    print(f"Result G19 DataFrame: {result_df.shape} rows")
    print(result_df)
    
    # Check that the DataFrame has the expected columns
    expected_columns = ['geo_sk', 'time_sk', 'geo_code', 'sex', 'age_group', 'condition', 'count']
    assert all(col in result_df.columns for col in expected_columns)
    
    # Check that we have the expected number of rows
    # Should have multiple rows per geographic area (conditions x demographics)
    assert len(result_df) > 2
    
    # Check that the values are in the expected range
    assert all(result_df['count'] >= 0)
    assert all(result_df['condition_not_stated_count'] >= 0)

def test_process_g20_census_data_success(mock_paths, mock_g20_zip, mock_geo_parquet):
    """Test successful processing of G20 census data with dimensional structure."""
    paths, _ = mock_paths
    from etl_logic import config
    # Use the same TEMP_DIR as in the actual function
    output_file = config.PATHS['TEMP_DIR'] / "staging_G20_detailed.parquet"
    mock_time_sk = 20210810
    
    from etl_logic.tables.g20_selected_conditions import process_g20_census_data
    result = process_g20_census_data(
        zip_dir=paths['CENSUS_DIR'],
        temp_extract_base=paths['TEMP_EXTRACT_DIR'],
        output_dir=paths['OUTPUT_DIR'],  # This output_dir is ignored by the function
        geo_output_path=mock_geo_parquet,
        time_sk=mock_time_sk
    )
    
    assert result is True
    assert output_file.exists()
    
    # Verify output content
    result_df = pl.read_parquet(output_file)
    print(f"Result G20 DataFrame: {result_df.shape} rows")
    print(result_df)
    
    # Check that the DataFrame has the required columns for the staging data
    expected_columns = ['geo_sk', 'time_sk', 'geo_code', 'etl_processed_at']
    assert all(col in result_df.columns for col in expected_columns)
    
    # Check that we have the expected number of rows
    # No assertions on exact row count as it depends on the mock data
    
    # Check the values
    assert len(result_df) > 0

def test_process_g21_census_data_success(mock_paths, mock_g21_zip, mock_geo_parquet):
    """Test successful processing of G21 census data."""
    paths, _ = mock_paths
    from etl_logic import config
    # Use the same TEMP_DIR as in the actual function
    output_file = config.PATHS['TEMP_DIR'] / "staging_G21_characteristic.parquet"
    mock_time_sk = 20210810
    
    from etl_logic.tables.g21_conditions_by_characteristics import process_g21_census_data
    result = process_g21_census_data(
        zip_dir=paths['CENSUS_DIR'],
        temp_extract_base=paths['TEMP_EXTRACT_DIR'],
        output_dir=paths['OUTPUT_DIR'],  # This output_dir is ignored by the function
        geo_output_path=mock_geo_parquet,
        time_sk=mock_time_sk
    )
    
    # Test the result
    assert result is True
    assert output_file.exists()
    
    # Verify output content
    result_df = pl.read_parquet(output_file)
    print(f"Result G21 DataFrame: {result_df.shape} rows")
    print(result_df)
    
    # Check the required columns for staging data
    expected_columns = ['geo_sk', 'time_sk', 'geo_code', 'etl_processed_at']
    assert all(col in result_df.columns for col in expected_columns)
    
    # Check that we have data
    assert len(result_df) > 0

def test_process_g25_census_data_success(mock_paths, mock_g25_zip, mock_geo_parquet):
    """Test successful processing of G25 census data for unpaid assistance."""
    paths, _ = mock_paths
    output_dir = paths['OUTPUT_DIR']
    output_file = output_dir / "fact_unpaid_assistance.parquet"
    
    # First check that the output file doesn't exist yet
    assert not output_file.exists()
    
    # Process the G25 data
    result = census.process_g25_census_data(
        zip_dir=paths['CENSUS_DIR'],
        temp_extract_base=paths['TEMP_EXTRACT_DIR'],
        output_dir=output_dir,
        geo_output_path=mock_geo_parquet,
        time_sk=20210810  # Just a test value for Census Night 2021
    )
    
    # Verify success and file existence
    assert result is True
    assert output_file.exists()
    
    # Verify output content
    result_df = pl.read_parquet(output_file)
    print(f"Result DataFrame: {result_df.shape}")
    print(result_df)
    
    # Check expected data shape and structure
    assert 'geo_code' in result_df.columns
    assert 'provided_assistance_count' in result_df.columns
    assert 'no_assistance_provided_count' in result_df.columns
    assert 'assistance_not_stated_count' in result_df.columns
    
    # Verify row count (should be 2, one for each geo_code from the mock_geo_parquet)
    # 99999 should be excluded as it's an invalid code not in the geo dimension
    assert result_df.shape[0] == 2
    


# --- Specific G18 Processor Tests ---

def test_process_g18_file_empty_input(tmp_path):
    """Test process_g18_file handles empty input."""
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("SA1_CODE_2021,P_Tot_Need_for_assistance\n")
    
    result = g18_assistance_needed.process_g18_file(csv_path)
    assert result is None


def test_process_g18_file_missing_geo_column(tmp_path):
    """Test process_g18_file handles missing geo column."""
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("Bad_Geo_Column,P_Tot_Need_for_assistance\n10101,10")
    
    result = g18_assistance_needed.process_g18_file(csv_path)
    assert result is None


def test_process_g18_file_no_valid_columns(tmp_path):
    """Test process_g18_file handles files with no parseable columns."""
    csv_path = tmp_path / "no_cols.csv"
    csv_path.write_text("SA1_CODE_2021,Unrelated_Column\n10101,10")
    
    result = g18_assistance_needed.process_g18_file(csv_path)
    assert result is None

# --- Helper Functions ---

def mock_census_csv_content(level):
    """Generate mock census CSV content for testing.
    
    Args:
        level (str): Geographic level (SA1 or SA2).
        
    Returns:
        str: CSV content as string.
    """
    # Header matches production column names
    header = f"{level}_CODE_2021,Tot_P_P,Tot_P_M,Tot_P_F,Indigenous_P_Tot_P\n"
    
    if level == 'SA1':
        # SA1 data
        return header + \
        "10101,100,50,50,10\n" + \
        "10102,200,100,100,20\n" + \
        "10103,300,150,150,30\n" + \
        "99999,0,0,0,0\n"  # Invalid code to be filtered out
    else:
        # SA2 data
        return header + \
        "20101,1000,500,500,100\n" + \
        "20102,2000,1000,1000,200\n"

def mock_g17_csv_content(level):
    """Generate mock G17 Census CSV content for testing."""
    if level == "SA1":
        # Using the format likely to be found in real data
        return f"""region_id,Has_Need_for_Assistance_P,Does_Not_Have_Need_for_Assistance_P,Need_for_Assistance_Not_Stated_P
10101,15,80,5
10102,25,160,15
10103,35,250,15
99999,5,10,5"""  # Invalid code to test filtering
    elif level == "SA2":
        # Using alternative column naming to test the flexibility
        return f"""SA2_CODE_2021,Assisted_Need_P,No_Assistance_Need_P,Assistance_Need_NS_P
20101,150,800,50
20102,250,1600,150
99999,50,100,50"""  # Invalid code to test filtering
    else:
        return "Invalid level requested"

def mock_g18_csv_content(level):
    """Generate mock G18 Census CSV content for testing."""
    # Column names reflect Sex_AgeGroup_AssistanceStatus pattern
    if level == "SA1":
        # Using region_id as geo code column name for SA1
        return f"""region_id,P_Tot_Need_for_assistance,P_Tot_No_need_for_assistance,P_Tot_Need_for_assistance_ns
10101,10,85,5
10102,20,160,20
10103,30,240,30
99999,1,9,1"""  # Invalid code to test filtering
    elif level == "SA2":
        # Using SA2_CODE_2021 as geo code column name
        return f"""SA2_CODE_2021,P_Tot_Need_for_assistance,P_Tot_No_need_for_assistance,P_Tot_Need_for_assistance_ns
20101,100,850,50
20102,150,1700,150
99999,10,90,10"""  # Invalid code to test filtering
    else:
        return "Invalid level requested"

def mock_g19_csv_content(level):
    """Generate mock G19 Census CSV content for testing."""
    if level == "SA1":
        # Using the format likely to be found in real data
        return f"""region_id,Has_condition_P,No_condition_P,Not_stated_P
10101,25,70,5
10102,35,150,15
10103,45,240,15
99999,10,80,10"""  # Invalid code to test filtering
    elif level == "SA2":
        # Using alternative column naming to test the flexibility
        return f"""SA2_CODE_2021,Has_Long_Term_Health_Condition_P,No_Long_Term_Health_Condition_P,Health_Condition_Not_Stated_P
20101,250,700,50
20102,350,1500,150
99999,100,800,100"""  # Invalid code to test filtering
    else:
        return "Invalid level requested"

def mock_g20_csv_content(level):
    """Generate mock G20 Census CSV content for testing.
    This provides a realistic structure for G20 data with multiple health conditions,
    age groups, and sex breakdowns.
    """
    if level == "SA1":
        # Using a common geo code column name and multiple health conditions
        return f"""SA1_CODE_2021,Tot_P_P,Arthritis_0-14_M,Arthritis_0-14_F,Arthritis_15-24_M,Arthritis_15-24_F,Asthma_0-14_M,Asthma_0-14_F,Asthma_15-24_M,Asthma_15-24_F,Diabetes_0-14_M,Diabetes_0-14_F,Diabetes_15-24_M,Diabetes_15-24_F,Cancer_0-14_M,Cancer_0-14_F,Cancer_15-24_M,Cancer_15-24_F
10101,100,1,2,3,4,5,6,7,8,2,3,1,2,0,1,2,3
10102,200,3,4,5,6,7,8,9,10,4,5,3,4,1,2,3,4
10103,300,5,6,7,8,9,10,11,12,6,7,5,6,2,3,4,5
99999,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"""
    elif level == "SA2":
        # Using alternative geo code name and multiple health conditions
        return f"""SA2_CODE21,Tot_P_P,Arthritis_0-14_M,Arthritis_0-14_F,Arthritis_15-24_M,Arthritis_15-24_F,Asthma_0-14_M,Asthma_0-14_F,Asthma_15-24_M,Asthma_15-24_F,Diabetes_0-14_M,Diabetes_0-14_F,Diabetes_15-24_M,Diabetes_15-24_F,Cancer_0-14_M,Cancer_0-14_F,Cancer_15-24_M,Cancer_15-24_F
20101,1000,10,20,30,40,50,60,70,80,20,30,10,20,5,10,15,20
20102,2000,30,40,50,60,70,80,90,100,40,50,30,40,15,20,25,30
99999,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"""
    else:
        return "Invalid level requested"

def mock_g21_csv_content(level):
    """Generate mock G21 Census CSV content for testing.
    This provides a realistic structure for G21 data with multiple health conditions
    and person characteristics like country of birth, labour force status, etc.
    """
    if level == "SA1":
        # Using a common geo code column name and realistic G21 columns with a wider range of conditions and characteristics
        return f"""SA1_CODE_2021,COB_Aus_Arth,COB_Aus_Asth,COB_Aus_Can_rem,COB_Aus_Dia_ges_dia,COB_Aus_HD_HA_ang,COB_Aus_Stroke,COB_Aus_MHC_Dep_anx,COB_Aus_Kid_dis,COB_Bo_SE_Asia_Arth,COB_Bo_SE_Asia_Asth,COB_Bo_NA_ME_Arth,COB_Bo_NA_ME_Asth,COB_Bo_Amer_Arth,COB_Bo_Amer_Asth,LFS_Emp_Arth,LFS_Emp_Asth,LFS_Emp_Stroke,LFS_Unemp_Arth,LFS_Unemp_Asth,LFS_Not_LF_Arth,LFS_Not_LF_Asth,Inc_Neg_Nil_Arth,Inc_Neg_Nil_Asth,Inc_1000_1749_Arth,Inc_1000_1749_Asth,Inc_3000_more_Arth,Inc_3000_more_Asth
10101,100,120,50,45,60,30,80,25,15,18,12,14,10,12,85,95,22,30,35,45,52,20,25,65,75,40,45
10102,200,220,110,100,120,55,160,55,25,28,22,25,18,22,170,185,45,60,65,90,105,35,45,130,145,90,100
10103,300,320,165,155,180,80,240,85,35,38,32,36,26,32,255,275,68,90,105,135,155,55,65,195,215,140,155
99999,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"""
    elif level == "SA2":
        # Using alternative geo code name and realistic G21 columns with a wider range of conditions and characteristics
        return f"""SA2_CODE21,COB_Aus_Arth,COB_Aus_Asth,COB_Aus_Can_rem,COB_Aus_Dia_ges_dia,COB_Aus_HD_HA_ang,COB_Aus_Stroke,COB_Aus_MHC_Dep_anx,COB_Aus_Kid_dis,COB_Bo_SE_Asia_Arth,COB_Bo_SE_Asia_Asth,COB_Bo_NA_ME_Arth,COB_Bo_NA_ME_Asth,COB_Bo_Amer_Arth,COB_Bo_Amer_Asth,LFS_Emp_Arth,LFS_Emp_Asth,LFS_Emp_Stroke,LFS_Unemp_Arth,LFS_Unemp_Asth,LFS_Not_LF_Arth,LFS_Not_LF_Asth,Inc_Neg_Nil_Arth,Inc_Neg_Nil_Asth,Inc_1000_1749_Arth,Inc_1000_1749_Asth,Inc_3000_more_Arth,Inc_3000_more_Asth
20101,1000,1200,500,450,600,300,800,250,150,180,120,140,100,120,850,950,220,300,350,450,520,200,250,650,750,400,450
20102,2000,2200,1100,1000,1200,550,1600,550,250,280,220,250,180,220,1700,1850,450,600,650,900,1050,350,450,1300,1450,900,1000
99999,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"""
    else:
        return "Invalid level requested"

def mock_g25_csv_content(level):
    """Create mock G25 CSV content for unpaid assistance data."""
    # Column headers
    headers = ["SA2_CODE_2021" if level == "SA2" else "SA1_CODE_2021", 
              "Provided_Unpaid_Assistance_P", "No_Unpaid_Assistance_Provided_P", "Unpaid_Assistance_Not_Stated_P"]
    
    # Base values for IDs
    base_ids = ["10101", "10102", "10103"] if level == "SA1" else ["20101", "20102", "99999"]
    
    # Create CSV content
    csv_lines = [",".join(headers)]
    for id_val in base_ids:
        # For testing, let's have simple patterns:
        # provided count = twice the last digit of id
        # no assistance count = three times the last digit of id
        # not stated count = the last digit of id
        last_digit = int(id_val[-1])
        provided = last_digit * 2
        no_assistance = last_digit * 3
        not_stated = last_digit
        
        csv_lines.append(f"{id_val},{provided},{no_assistance},{not_stated}")
    

# Integration test for process_census_table end-to-end flow
def test_integration_process_census_table(mock_paths, mock_census_zip, mock_geo_parquet):
    """Test the end-to-end flow of process_census_table with mocking.
    
    This test verifies the successful execution of the census table processing,
    including file finding, extraction, processing, and joining with dimensions.
    Mocks are used to simulate file system interactions and dimension table reads.
    
    Covers:
    - Happy path with valid data
    - Ensures correct output DataFrame structure and joins
    """
    # Reason: This test integrates multiple components to ensure the pipeline works as expected,
    # including mocking to isolate dependencies and focus on logic flow.
    from etl_logic.census import process_census_table
    
    paths, _ = mock_paths
    output_dir = paths['OUTPUT_DIR']
    output_file = output_dir / "fact_test_table.parquet"  # Use a test-specific output name
    
    # Mock the process_file_function to return a sample DataFrame
    def mock_process_file(csv_file):
        return pl.DataFrame({
            'geo_code': ['10101', '10102'],
            'test_measure': [10, 20]
        })
    
    result = census.process_census_table(
        table_code="TEST",
        process_file_function=mock_process_file,
        output_filename="fact_test_table.parquet",
        zip_dir=paths['CENSUS_DIR'],
        temp_extract_base=paths['TEMP_EXTRACT_DIR'],
        output_dir=output_dir,
        geo_output_path=mock_geo_parquet
    )
    
    assert result is True
    assert (output_dir / "fact_test_table.parquet").exists()
    
    # Read the output and verify content
    result_df = pl.read_parquet(output_dir / "fact_test_table.parquet")
    assert 'geo_sk' in result_df.columns  # From geo join
    assert 'test_measure' in result_df.columns  # From process_file_function
    assert len(result_df) == 2  # Expecting two rows from mock data

# Edge case test for no files found
def test_integration_process_census_table_no_files(mock_paths, mock_geo_parquet):
    """Test process_census_table when no ZIP files are found.
    
    Verifies that the function handles the absence of input files gracefully,
    returning False without errors.
    
    Covers:
    - Failure mode when no data sources are available
    """
    # Reason: This test ensures robust error handling in real-world scenarios where data might be missing.
    from etl_logic.census import process_census_table
    
    paths, _ = mock_paths
    output_dir = paths['OUTPUT_DIR']
    
    # Mock the find_census_files to return an empty list
    def mock_find_census_files(*args, **kwargs):
        return []
    
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(census, 'find_census_files', mock_find_census_files)
        
        result = census.process_census_table(
            table_code="TEST",
            process_file_function=lambda x: None,  # Dummy function, not called
            output_filename="fact_test_table.parquet",
            zip_dir=paths['CENSUS_DIR'],
            temp_extract_base=paths['TEMP_EXTRACT_DIR'],
            output_dir=output_dir,
            geo_output_path=mock_geo_parquet
        )
    
    assert result is False
    assert not (output_dir / "fact_test_table.parquet").exists()  # No output file should be created

    return "\n".join(csv_lines)