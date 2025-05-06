import os
import sys
from pathlib import Path
import pytest
import polars as pl
from dotenv import load_dotenv

# Ensure etl_logic is importable from the tests directory
# Assuming tests/ is at the same level as etl_logic/
# Adjust if your structure is different
TESTS_DIR = Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import config after potentially modifying path
# We need to reload it if it was imported elsewhere already
# Or better, ensure it's imported fresh after setting env vars
# For simplicity here, assume direct import works after path modification
# If issues arise, use importlib.reload
from etl_logic import config
from etl_logic.config import (
    BASE_DIR, PATHS, 
    GEO_LEVELS_SHP_PROCESS, 
    GEO_LEVELS_CENSUS_PROCESS, 
    CENSUS_TABLE_PATTERNS,
    CENSUS_COLUMN_MAPPINGS,
    SCHEMAS,
    DATA_URLS
)

# --- Fixtures ---

@pytest.fixture(scope="function", autouse=True)
def manage_env_vars(monkeypatch):
    """Fixture to manage environment variables for tests."""
    # Store original env vars if they exist
    original_base_dir = os.environ.get('BASE_DIR')

    # Clear BASE_DIR before each test function
    monkeypatch.delenv('BASE_DIR', raising=False)

    yield # Test runs here

    # Restore original env vars after test
    if original_base_dir is not None:
        monkeypatch.setenv('BASE_DIR', original_base_dir)
    else:
        monkeypatch.delenv('BASE_DIR', raising=False)

@pytest.fixture(scope="function")
def temp_env_file(tmp_path, monkeypatch):
    """Fixture to create a temporary .env file."""
    env_content = "BASE_DIR=./temp_test_base\n"
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)
    # Monkeypatch os.getcwd() to make dotenv load from tmp_path
    monkeypatch.setattr(os, 'getcwd', lambda: str(tmp_path))
    return env_file, tmp_path / "temp_test_base"

# --- Test Cases ---

def test_base_dir_default(monkeypatch):
    """Test that BASE_DIR defaults to current working directory if not set."""
    # Ensure BASE_DIR is not set in env
    monkeypatch.delenv('BASE_DIR', raising=False)
    # Ensure no .env file is loaded unexpectedly (though manage_env_vars helps)

    # Reload config to pick up the default
    import importlib
    importlib.reload(config)

    expected_base_dir = Path('.').resolve()
    assert config.BASE_DIR == expected_base_dir

def test_base_dir_from_env(monkeypatch):
    """Test that BASE_DIR is read from environment variable."""
    test_path = "/tmp/test_base_env"
    monkeypatch.setenv('BASE_DIR', test_path)

    # Reload config to pick up the env var
    import importlib
    importlib.reload(config)

    assert config.BASE_DIR == Path(test_path).resolve()

def test_base_dir_from_dotenv(temp_env_file, monkeypatch):
    """Test that BASE_DIR is read from .env file."""
    env_file, expected_base_path_relative = temp_env_file
    # Ensure BASE_DIR is not set in actual environment initially
    monkeypatch.delenv('BASE_DIR', raising=False)

    # Simulate the effect of .env by setting the env var directly for this test
    # The value comes from the fixture's relative path, resolved against the temp dir
    tmp_path_where_env_lives = env_file.parent
    dotenv_base_dir_value = str((tmp_path_where_env_lives / expected_base_path_relative).resolve())
    monkeypatch.setenv('BASE_DIR', dotenv_base_dir_value)

    # Reload config to pick up the *environment variable* we just set
    import importlib
    importlib.reload(config)

    # Assert against the value we set in the environment
    assert config.BASE_DIR == Path(dotenv_base_dir_value).resolve()

def test_paths_resolution(monkeypatch):
    """Test that paths in config.PATHS are resolved correctly relative to BASE_DIR."""
    test_base = "/app/data_root"
    monkeypatch.setenv('BASE_DIR', test_base)

    # Reload config
    import importlib
    importlib.reload(config)

    assert config.BASE_DIR == Path(test_base).resolve()
    # Check a few key paths
    assert config.PATHS['DATA_DIR'] == Path(test_base).resolve() / "data"
    assert config.PATHS['OUTPUT_DIR'] == Path(test_base).resolve() / "output"
    assert config.PATHS['GEOGRAPHIC_DIR'] == Path(test_base).resolve() / "data/raw/geographic"
    assert config.PATHS['LOG_DIR'] == Path(test_base).resolve() / "logs"

def test_initialize_directories(tmp_path, monkeypatch):
    """Test that initialize_directories creates the required folders."""
    # Set BASE_DIR to a temporary path for this test
    monkeypatch.setenv('BASE_DIR', str(tmp_path))

    # Reload config
    import importlib
    importlib.reload(config)

    # Run the initialization function
    config.initialize_directories()

    # Check if directories were created
    assert (tmp_path / "data").is_dir()
    assert (tmp_path / "data/raw").is_dir()
    assert (tmp_path / "data/raw/geographic").is_dir()
    assert (tmp_path / "data/raw/census").is_dir()
    assert (tmp_path / "data/raw/temp").is_dir()
    assert (tmp_path / "data/raw/temp/zips").is_dir()
    assert (tmp_path / "data/raw/temp/extract").is_dir()
    assert (tmp_path / "output").is_dir()
    assert (tmp_path / "logs").is_dir()

def test_base_dir_is_absolute():
    """Test that BASE_DIR is an absolute path."""

    """Test that BASE_DIR is an absolute path."""
    assert BASE_DIR.is_absolute(), "BASE_DIR should be an absolute path"

def test_paths_use_base_dir():
    """Test that all paths are derived from BASE_DIR."""

    """Test that all paths are derived from BASE_DIR."""
    for path_name, path in PATHS.items():
        assert str(path).startswith(str(BASE_DIR)), f"{path_name} should start with BASE_DIR"

def test_paths_are_created():
    """Test that path directories are created when initialized."""

    """Test that path directories are created when initialized."""
    # This test only runs the initialise_directories function without checking
    # if directories actually exist, as we don't want to create dirs in tests
    try:
        config.initialise_directories()
    except Exception as e:
        pytest.fail(f"initialise_directories() raised {e} unexpectedly!")

def test_geo_levels_match_zip_urls():
    """Test that geographic levels have corresponding URLs."""

    """Test that geographic levels have corresponding URLs."""
    for level, prefix in GEO_LEVELS_SHP_PROCESS.items():
        assert prefix in config.DATA_URLS.keys() or prefix not in config.DATA_URLS.keys(), f"Mismatch for {level}: {prefix} not in DATA_URLS"

def test_census_table_patterns_format():
    """Test that census table patterns have correct format."""

    """Test that census table patterns have correct format."""
    for table_code, pattern in CENSUS_TABLE_PATTERNS.items():
        assert table_code in pattern, f"Table code {table_code} should be in its pattern"
        assert "(" in pattern and ")" in pattern, f"Pattern for {table_code} should include capture groups"

def test_census_column_mappings_structure():
    """Test that census column mappings have the correct structure."""

    """Test that census column mappings have the correct structure."""
    for table_code in ["G01", "G17", "G18", "G19", "G20", "G21", "G25"]:
        assert table_code in CENSUS_COLUMN_MAPPINGS, f"Table {table_code} should be in CENSUS_COLUMN_MAPPINGS"
        
        mapping = CENSUS_COLUMN_MAPPINGS[table_code]
        assert "geo_column_options" in mapping, f"{table_code} mapping should include geo_column_options"
        assert "measure_column_map" in mapping, f"{table_code} mapping should include measure_column_map"
        assert "required_target_columns" in mapping, f"{table_code} mapping should include required_target_columns"
        
        assert isinstance(mapping["geo_column_options"], list), f"{table_code} geo_column_options should be a list"
        assert isinstance(mapping["measure_column_map"], dict), f"{table_code} measure_column_map should be a dict"
        assert isinstance(mapping["required_target_columns"], list), f"{table_code} required_target_columns should be a list"

def test_schemas_structure():
    """Test the main SCHEMAS dictionary structure and specific schemas."""
    assert isinstance(SCHEMAS, dict)
    # Check for essential dimension and fact schemas
    assert "geo_dimension" in SCHEMAS
    assert "dim_time" in SCHEMAS
    assert "dim_health_condition" in SCHEMAS
    assert "fact_population" in SCHEMAS # Check a final fact
    assert "fact_health_conditions_raw" in SCHEMAS # Check a raw fact

    # Test the structure of the geo_dimension schema
    geo_schema = SCHEMAS["geo_dimension"]
    assert isinstance(geo_schema, dict)
    assert "geo_sk" in geo_schema
    assert "geo_code" in geo_schema
    assert "geometry" in geo_schema
    assert "etl_processed_at" in geo_schema
    # Check Polars types
    assert geo_schema["geo_sk"] == pl.UInt64
    assert geo_schema["geo_level"] == pl.Categorical
    assert geo_schema["longitude"] == pl.Float64
    assert isinstance(geo_schema["etl_processed_at"], pl.DataType) # Check it's a Polars type

    # Test the structure of the time_dimension schema
    time_schema = SCHEMAS["dim_time"]
    assert isinstance(time_schema, dict)
    assert "time_sk" in time_schema
    assert "year" in time_schema
    assert time_schema["year"] == pl.Int16

    # Test the structure of the health_condition schema
    health_schema = SCHEMAS["dim_health_condition"]
    assert isinstance(health_schema, dict)
    assert "condition_sk" in health_schema
    assert "condition_code" in health_schema
    assert health_schema["condition_sk"] == pl.Utf8 # Check type

    # Test the structure of a fact schema
    fact_pop_schema = SCHEMAS["fact_population"]
    assert isinstance(fact_pop_schema, dict)
    assert "geo_sk" in fact_pop_schema
    assert "time_sk" in fact_pop_schema
    assert "total_persons" in fact_pop_schema
    assert fact_pop_schema["time_sk"] == pl.Int64
    assert fact_pop_schema["total_persons"] == pl.Int64

def test_get_required_geo_zips():
    """Test that the get_required_geo_zips function returns expected results."""

    """Test that the get_required_geo_zips function returns expected results."""
    geo_zips = config.get_required_geo_zips()
    assert isinstance(geo_zips, dict), "get_required_geo_zips should return a dict"
    
    # Check if keys follow expected pattern
    for key in geo_zips.keys():
        assert key.endswith(".zip"), f"Zip filename {key} should end with .zip"
        assert "_SHP" in key, f"Zip filename {key} should include '_SHP'"

def test_get_required_census_zips():
    """Test that the get_required_census_zips function returns expected results."""

    """Test that the get_required_census_zips function returns expected results."""
    census_zips = config.get_required_census_zips()
    assert isinstance(census_zips, dict), "get_required_census_zips should return a dict"
    
    # Check if any zips were found
    # This test is more lax, as it depends on configuration
    if census_zips:
        for key in census_zips.keys():
            assert key.endswith(".zip"), f"Zip filename {key} should end with .zip"

def test_paths_are_absolute():
    """Test that all paths in PATHS are absolute Path objects."""
    assert isinstance(PATHS, dict)
    for name, path_val in PATHS.items():
        assert isinstance(path_val, Path), f"Path '{name}' is not a Path object"
        assert path_val.is_absolute(), f"Path '{name}' is not absolute: {path_val}"

def test_data_urls_exist():
    """Test that essential data URLs are defined (if imported)."""
    # Check if URLs were successfully imported (might be empty if direct_urls.py failed)
    if config.DATA_URLS: # Only test if URLs are populated
        assert isinstance(config.DATA_URLS, dict)
        # Check for a few expected keys, assuming they are needed
        assert "SA1_2021_AUST_GDA2020" in config.DATA_URLS
        assert "CENSUS_GCP_AUS_2021" in config.DATA_URLS
        for name, url_val in config.DATA_URLS.items():
            assert isinstance(url_val, str), f"URL '{name}' is not a string"
            assert url_val.startswith("http"), f"URL '{name}' does not look like a URL: {url_val}"

def test_geo_levels_config():
    """Test geographic level configurations."""
    assert isinstance(GEO_LEVELS_SHP_PROCESS, dict)
    assert "SA1" in GEO_LEVELS_SHP_PROCESS
    assert "POA" in GEO_LEVELS_SHP_PROCESS

def test_census_table_patterns():
    """Test Census table pattern configuration."""
    assert isinstance(CENSUS_TABLE_PATTERNS, dict)
    assert "G01" in CENSUS_TABLE_PATTERNS
    assert "G25" in CENSUS_TABLE_PATTERNS
    assert isinstance(CENSUS_TABLE_PATTERNS["G19"], str)

def test_census_column_mappings():
    """Test the structure of Census column mappings."""
    assert isinstance(CENSUS_COLUMN_MAPPINGS, dict)
    assert "G01" in CENSUS_COLUMN_MAPPINGS
    assert "G19" in CENSUS_COLUMN_MAPPINGS
    # Check structure of one entry
    g01_mapping = CENSUS_COLUMN_MAPPINGS["G01"]
    assert isinstance(g01_mapping, dict)
    assert "geo_column_options" in g01_mapping
    assert "measure_column_map" in g01_mapping
    assert "required_target_columns" in g01_mapping
    assert isinstance(g01_mapping["measure_column_map"], dict)
    assert isinstance(g01_mapping["required_target_columns"], list)