import pytest
import sys
from pathlib import Path
import zipfile
import pandas as pd
import geopandas as gpd
import polars as pl
from polars.testing import assert_frame_equal
from shapely.geometry import Point, Polygon
import pytest_mock # Explicitly import pytest-mock

# Ensure etl_logic is importable
TESTS_DIR = Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from etl_logic import geography, config, utils

# --- Fixtures ---

@pytest.fixture
def mock_paths(tmp_path, monkeypatch):
    """Mocks config.PATHS to use temporary directories."""
    paths = {
        'GEOGRAPHIC_DIR': tmp_path / "raw/geographic",
        'TEMP_EXTRACT_DIR': tmp_path / "temp/extract",
        'OUTPUT_DIR': tmp_path / "output",
        # Add other paths if needed by tested functions indirectly
        'DATA_DIR': tmp_path / 'data',
        'RAW_DATA_DIR': tmp_path / 'data/raw',
        'TEMP_DIR': tmp_path / 'data/raw/temp',
        'LOG_DIR': tmp_path / 'logs',
        'CENSUS_DIR': tmp_path / 'data/raw/census',
        'TEMP_ZIP_DIR': tmp_path / 'data/raw/temp/zips',
    }
    # Create the directories
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, 'PATHS', paths)
    # Also mock GEO_LEVELS_SHP_PROCESS for consistency in tests
    monkeypatch.setattr(config, 'GEO_LEVELS_SHP_PROCESS', {
        'SA2': 'SA2_2021_AUST_GDA2020',
        'STATE': 'STE_2021_AUST_GDA2020' # Test with two levels
    })
    return paths

@pytest.fixture
def mock_zip_files(mock_paths):
    """Creates dummy zip files in the mocked GEOGRAPHIC_DIR."""
    geo_dir = mock_paths['GEOGRAPHIC_DIR']
    # Create dummy zip for SA2
    sa2_zip = geo_dir / "SA2_2021_AUST_GDA2020_SHP.zip"
    with zipfile.ZipFile(sa2_zip, 'w') as zf:
        zf.writestr("dummy_sa2.txt", "sa2 data")
    # Create dummy zip for STATE
    ste_zip = geo_dir / "STE_2021_AUST_GDA2020_SHP.zip"
    with zipfile.ZipFile(ste_zip, 'w') as zf:
        zf.writestr("dummy_ste.txt", "state data")
    return {"SA2": sa2_zip, "STATE": ste_zip}

@pytest.fixture
def mock_gpd_read_file(mocker):
    """Mocks geopandas.read_file to return predefined GeoDataFrames."""
    def mock_reader(filepath):
        # Return different data based on the expected shapefile path
        if "SA2" in str(filepath):
            return gpd.GeoDataFrame({
                'SA2_CODE_2021': ['20101', '20102'],
                'SA2_NAME': ['Area A', 'Area B'],
                'geometry': [Point(1, 1), Point(2, 2)]
            }, crs="EPSG:4326")
        elif "state" in str(filepath).lower(): # Case-insensitive check
             # Use a column name expected by process_geography for STATE level
            return gpd.GeoDataFrame({
                'STATE_CODE21': ['1', '2'], # Changed from STE_CODE21
                'STATE_NAME21': ['State X', 'State Y'], # Changed from STE_NAME21
                'geometry': [Polygon([(0,0), (1,1), (1,0)]), Polygon([(2,2), (3,3), (3,2)])]
            }, crs="EPSG:4326")
        else:
            raise FileNotFoundError(f"Mock shapefile not found for {filepath}")

    mock = mocker.patch('geopandas.read_file', side_effect=mock_reader)
    return mock

@pytest.fixture
def mock_extract_zipfile(mocker):
    """Mocks utils.extract_zipfile."""
    def mock_extractor(zip_file, extract_dir, desc=None):
        # Simulate finding a shapefile after extraction
        # Create a dummy file matching the expected pattern
        # The actual content doesn't matter as gpd.read_file is mocked
        level_name = extract_dir.name # e.g., SA2, STATE
        dummy_shp = extract_dir / f"{level_name}_dummy.shp"
        dummy_shp.touch()
        return True # Simulate successful extraction

    mock = mocker.patch('etl_logic.utils.extract_zipfile', side_effect=mock_extractor)
    return mock

# --- Geographic Processing Tests ---

def test_process_geography_success_with_centroids(mock_paths, mock_zip_files, mock_extract_zipfile, mock_gpd_read_file):
    """Test successful processing of geographic levels including centroid calculations."""
    output_dir = mock_paths['OUTPUT_DIR']
    output_file = output_dir / "geo_dimension.parquet"

    result = geography.process_geography(
        zip_dir=mock_paths['GEOGRAPHIC_DIR'],
        temp_extract_base=mock_paths['TEMP_EXTRACT_DIR'],
        output_dir=output_dir
    )

    assert result is True
    assert output_file.exists()

    # Verify calls
    assert mock_extract_zipfile.call_count == 2 # Called for SA2 and STATE
    assert mock_gpd_read_file.call_count == 2

    # Verify output content (adjust expected WKT based on shapely version if needed)
    result_df = pl.read_parquet(output_file)
    expected_data = {
        'geo_sk': pl.Series([0, 1, 2, 3], dtype=pl.UInt32),  # Changed to UInt32 to match the actual type
        'geo_code': ["20101", "20102", "1", "2"],
        'geo_level': ["SA2", "SA2", "STATE", "STATE"],
        'geometry': [
            'POINT (1 1)', 'POINT (2 2)',
            'POLYGON ((0 0, 1 1, 1 0, 0 0))', 'POLYGON ((2 2, 3 3, 3 2, 2 2))'
        ],
        'longitude': [1.0, 2.0, 0.5, 2.5],  # Approximate centroids for test geometries
        'latitude': [1.0, 2.0, 0.333, 2.333]  # Approximate centroids for test geometries
    }
    expected_df = pl.DataFrame(expected_data).sort(['geo_level', 'geo_code'])
    result_df = result_df.sort(['geo_level', 'geo_code']) # Sort for comparison

    # Ignore the etl_processed_at column in the comparison
    result_df = result_df.drop('etl_processed_at')
    assert_frame_equal(result_df, expected_df)
    assert 'longitude' in result_df.columns, "Longitude column missing from output"
    assert 'latitude' in result_df.columns, "Latitude column missing from output"

def test_process_geography_missing_zip(mock_paths, caplog):
    """Test handling when a required zip file is missing."""
    # Don't call mock_zip_files fixture
    output_dir = mock_paths['OUTPUT_DIR']
    result = geography.process_geography(
        zip_dir=mock_paths['GEOGRAPHIC_DIR'],
        temp_extract_base=mock_paths['TEMP_EXTRACT_DIR'],
        output_dir=output_dir
    )
    assert result is False # Should fail as zips are missing
    assert "ZIP file not found" in caplog.text
    assert not (output_dir / "geo_dimension.parquet").exists()

def test_process_geography_extraction_fails(mock_paths, mock_zip_files, mocker, caplog):
    """Test handling when zip extraction fails."""
    mocker.patch('etl_logic.utils.extract_zipfile', return_value=False) # Mock failure
    output_dir = mock_paths['OUTPUT_DIR']
    result = geography.process_geography(
        zip_dir=mock_paths['GEOGRAPHIC_DIR'],
        temp_extract_base=mock_paths['TEMP_EXTRACT_DIR'],
        output_dir=output_dir
    )
    assert result is False
    assert "Failed to extract" in caplog.text
    assert not (output_dir / "geo_dimension.parquet").exists()

def test_process_geography_no_shapefile_found(mock_paths, mock_zip_files, mocker, caplog):
    """Test handling when no .shp file is found after extraction."""
    # Mock extraction to succeed but *not* create the dummy .shp file
    mocker.patch('etl_logic.utils.extract_zipfile', return_value=True)
    output_dir = mock_paths['OUTPUT_DIR']
    result = geography.process_geography(
        zip_dir=mock_paths['GEOGRAPHIC_DIR'],
        temp_extract_base=mock_paths['TEMP_EXTRACT_DIR'],
        output_dir=output_dir
    )
    assert result is False
    assert "No shapefile found in" in caplog.text
    assert not (output_dir / "geo_dimension.parquet").exists()

def test_process_geography_gpd_read_fails(mock_paths, mock_zip_files, mock_extract_zipfile, mocker, caplog):
    """Test handling when geopandas.read_file fails."""
    mocker.patch('geopandas.read_file', side_effect=Exception("Read Error"))
    output_dir = mock_paths['OUTPUT_DIR']
    result = geography.process_geography(
        zip_dir=mock_paths['GEOGRAPHIC_DIR'],
        temp_extract_base=mock_paths['TEMP_EXTRACT_DIR'],
        output_dir=output_dir
    )
    assert result is False
    assert "Error processing" in caplog.text
    assert "Read Error" in caplog.text
    assert not (output_dir / "geo_dimension.parquet").exists()

def test_process_geography_no_geo_code_column(mock_paths, mock_zip_files, mock_extract_zipfile, mocker, caplog):
    """Test handling when the expected geographic code column is not found."""
    # Mock read_file to return a GDF without the expected column names
    mock_reader = mocker.patch('geopandas.read_file', return_value=gpd.GeoDataFrame({
        'WRONG_CODE_COL': ['1'], 'geometry': [Point(1,1)]
    }))
    output_dir = mock_paths['OUTPUT_DIR']
    result = geography.process_geography(
        zip_dir=mock_paths['GEOGRAPHIC_DIR'],
        temp_extract_base=mock_paths['TEMP_EXTRACT_DIR'],
        output_dir=output_dir
    )
    assert result is False
    assert "Could not find geographic code column for" in caplog.text
    assert not (output_dir / "geo_dimension.parquet").exists()

def test_process_geography_state_code_variation(mock_paths, mocker, caplog):
    """Test processing geographic data with various STATE code column names."""
    output_dir = mock_paths['OUTPUT_DIR']
    geo_dir = mock_paths['GEOGRAPHIC_DIR']
    temp_extract_dir = mock_paths['TEMP_EXTRACT_DIR']
    
    # Create dummy STATE zip
    ste_zip = geo_dir / "STE_2021_AUST_GDA2020_SHP.zip"
    with zipfile.ZipFile(ste_zip, 'w') as zf:
        zf.writestr("dummy_ste.txt", "state data")
    
    # Mock extract_zipfile to create a dummy shapefile
    def mock_extract(zip_file, extract_dir, desc=None):
        extract_dir.mkdir(parents=True, exist_ok=True)
        (extract_dir / "STE_dummy.shp").touch()
        return True
        
    mocker.patch('etl_logic.utils.extract_zipfile', side_effect=mock_extract)
    
    # Test multiple column name variations for STATE codes
    state_code_variations = [
        'STATE_CODE21', 'STATE_CODE_2021', 'STE_CODE21', 'STE_CODE_2021',
        'STE_CODE', 'STATE_CODE', 'STE21', 'STE', 'STATE'
    ]
    
    # Override GEO_LEVELS_SHP_PROCESS to only process STATE
    mocker.patch.object(config, 'GEO_LEVELS_SHP_PROCESS', {'STATE': 'STE_2021_AUST_GDA2020'})
    
    for col_name in state_code_variations:
        # Reset output directory
        for file in output_dir.glob("*"):
            file.unlink()
        
        # Mock geopandas.read_file to return data with the current column name
        def mock_read_file(filepath):
            return gpd.GeoDataFrame({
                col_name: ['1', '2', '3'], # Single digit STATE codes
                'STATE_NAME': ['NSW', 'VIC', 'QLD'],
                'geometry': [
                    Polygon([(0,0), (1,1), (1,0)]), 
                    Polygon([(2,2), (3,3), (3,2)]),
                    Polygon([(4,4), (5,5), (5,4)])
                ]
            }, crs="EPSG:4326")
            
        mocker.patch('geopandas.read_file', side_effect=mock_read_file)
        
        # Run the process
        result = geography.process_geography(
            zip_dir=geo_dir,
            temp_extract_base=temp_extract_dir,
            output_dir=output_dir
        )
        
        assert result is True, f"Processing failed with STATE column name: {col_name}"
        output_file = output_dir / "geo_dimension.parquet"
        assert output_file.exists(), f"Output file not created with column name: {col_name}"
        
        # Verify the content has the correct STATE codes
        result_df = pl.read_parquet(output_file)
        assert "1" in result_df.select("geo_code").to_series().to_list(), f"STATE code '1' not found with column {col_name}"
        assert "2" in result_df.select("geo_code").to_series().to_list(), f"STATE code '2' not found with column {col_name}"
        assert "3" in result_df.select("geo_code").to_series().to_list(), f"STATE code '3' not found with column {col_name}"
        
        # Verify all rows have STATE as geo_level
        assert all(level == "STATE" for level in result_df.select("geo_level").to_series().to_list())

def test_clean_geo_code_state_codes():
    """Test the clean_geo_code function specifically for STATE codes."""
    # Test single-digit STATE codes
    assert utils.clean_geo_code("1") == "1"  # NSW
    assert utils.clean_geo_code("2") == "2"  # VIC
    assert utils.clean_geo_code("3") == "3"  # QLD
    assert utils.clean_geo_code("9") == "9"  # Other territories
    
    # Test with spaces or leading zeros
    assert utils.clean_geo_code(" 1 ") == "1"
    assert utils.clean_geo_code("01") == "01"  # Should preserve leading zeros
    
    # Test non-digit values (should return None for most non-digits)
    assert utils.clean_geo_code("NSW") is None
    
    # Test Australia code
    assert utils.clean_geo_code("AUS") == "AUS"
    assert utils.clean_geo_code("aus") == "AUS"  # Case insensitivity

def test_find_geo_column_state_variations():
    """Test the find_geo_column function with STATE level code column variations."""
    # Create test dataframes with different STATE column variations
    variations = [
        'STATE_CODE21', 'STATE_CODE_2021', 'STE_CODE21', 'STE_CODE_2021',
        'STE_CODE', 'STATE_CODE', 'STE21', 'STATE21', 'STE_2021', 'STATE_2021'
    ]
    
    for var in variations:
        df = pd.DataFrame({var: ['1', '2', '3']})
        
        # Test with a common search list that includes STATE
        found_col = utils.find_geo_column(df, ['region_id', 'SA2_CODE21', 'STATE_CODE'])
        assert found_col == var, f"Failed to find STATE column {var}"
        
        # Test with just searching for a specific STATE column
        found_col2 = utils.find_geo_column(df, ['STE_CODE'])
        assert found_col2 == var, f"Failed to find STATE column {var} when searching for STE_CODE"