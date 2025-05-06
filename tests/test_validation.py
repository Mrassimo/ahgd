"""
Tests for the ETL data quality validation module.
"""
def test_verify_fact_health_conditions_joins(temp_dir):
    """Verify joins between fact_health_conditions_refined and all dimension tables.
    
    Tests that all surrogate keys in fact_health_conditions_refined have matching
    records in the corresponding dimension tables.
    """
    # Create test data for fact table
    fact_data = pl.DataFrame({
        "geo_sk": [1, 2, 3],
        "time_sk": [1, 2, 3],
        "health_condition_sk": [1, 2, 3],
        "demographic_sk": [1, 2, 3],
        "count": [10, 20, 30]
    })
    fact_file = temp_dir / "fact_health_conditions_refined.parquet"
    fact_data.write_parquet(fact_file)

    # Create test dimension tables
    geo_dim = pl.DataFrame({"geo_sk": [1, 2, 3], "name": ["A", "B", "C"]})
    time_dim = pl.DataFrame({"time_sk": [1, 2, 3], "year": [2020, 2021, 2022]})
    health_dim = pl.DataFrame({"health_condition_sk": [1, 2, 3], "condition": ["X", "Y", "Z"]})
    demo_dim = pl.DataFrame({"demographic_sk": [1, 2, 3], "group": ["G1", "G2", "G3"]})

    # Test joins with each dimension table
    for dim_df, dim_name, join_col in [
        (geo_dim, "geo_dimension", "geo_sk"),
        (time_dim, "dim_time", "time_sk"),
        (health_dim, "dim_health_condition", "health_condition_sk"),
        (demo_dim, "dim_demographic", "demographic_sk")
    ]:
        dim_file = temp_dir / f"{dim_name}.parquet"
        dim_df.write_parquet(dim_file)
        
        fact_df = pl.read_parquet(fact_file)
        dim_df = pl.read_parquet(dim_file)
        
        # Perform join and verify count remains unchanged
        joined = fact_df.join(dim_df, on=join_col, how="inner")
        assert len(joined) == len(fact_df), f"Join with {dim_name} failed"

def test_verify_fact_conditions_by_char_joins(temp_dir):
    """Verify joins between fact_health_conditions_by_characteristic_refined and dimensions.
    
    Tests that all surrogate keys in fact_health_conditions_by_characteristic_refined
    have matching records in the corresponding dimension tables.
    """
    # Create test data for fact table
    fact_data = pl.DataFrame({
        "geo_sk": [1, 2, 3],
        "time_sk": [1, 2, 3],
        "health_condition_sk": [1, 2, 3],
        "person_characteristic_sk": [1, 2, 3],
        "count": [10, 20, 30]
    })
    fact_file = temp_dir / "fact_health_conditions_by_characteristic_refined.parquet"
    fact_data.write_parquet(fact_file)

    # Create test dimension tables
    geo_dim = pl.DataFrame({"geo_sk": [1, 2, 3], "name": ["A", "B", "C"]})
    time_dim = pl.DataFrame({"time_sk": [1, 2, 3], "year": [2020, 2021, 2022]})
    health_dim = pl.DataFrame({"health_condition_sk": [1, 2, 3], "condition": ["X", "Y", "Z"]})
    char_dim = pl.DataFrame({"person_characteristic_sk": [1, 2, 3], "characteristic": ["C1", "C2", "C3"]})

    # Test joins with each dimension table
    for dim_df, dim_name, join_col in [
        (geo_dim, "geo_dimension", "geo_sk"),
        (time_dim, "dim_time", "time_sk"),
        (health_dim, "dim_health_condition", "health_condition_sk"),
        (char_dim, "dim_person_characteristic", "person_characteristic_sk")
    ]:
        dim_file = temp_dir / f"{dim_name}.parquet"
        dim_df.write_parquet(dim_file)
        
        fact_df = pl.read_parquet(fact_file)
        dim_df = pl.read_parquet(dim_file)
        
        # Perform join and verify count remains unchanged
        joined = fact_df.join(dim_df, on=join_col, how="inner")
        assert len(joined) == len(fact_df), f"Join with {dim_name} failed"

import pytest
from pathlib import Path
import polars as pl
import tempfile
import shutil
from etl_logic import validation

@pytest.fixture
def temp_dir():
    """Create and cleanup a temporary directory for test files."""
    test_dir = Path(tempfile.mkdtemp())
    yield test_dir
    shutil.rmtree(test_dir)

def test_validate_table_record_count(temp_dir):
    """Test record count validation."""
    # Create test data
    df = pl.DataFrame({
        "id": [1, 2, 3],
        "value": ["a", "b", "c"]
    })
    test_file = temp_dir / "test.parquet"
    df.write_parquet(test_file)

    # Test with records
    assert validation.validate_table_record_count(df, "test_table") is True

    # Test empty dataframe
    empty_df = pl.DataFrame()
    assert validation.validate_table_record_count(empty_df, "empty_table") is False

def test_validate_null_values(temp_dir):
    """Test null value validation."""
    # Create test data with and without nulls
    good_df = pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["a", "b", "c"]
    })
    
    bad_df = pl.DataFrame({
        "id": [1, None, 3],
        "name": ["a", "b", None]
    })

    # Test with no nulls
    assert validation.validate_null_values(good_df, "good_table", ["id", "name"]) is True

    # Test with nulls
    assert validation.validate_null_values(bad_df, "bad_table", ["id", "name"]) is False

    # Test with missing column
    assert validation.validate_null_values(good_df, "test_table", ["missing"]) is False

def test_validate_range_values(temp_dir):
    """Test range validation for numeric columns."""
    # Create test data
    good_df = pl.DataFrame({
        "count": [0, 1, 2],
        "value": [10, 20, 30]
    })
    
    bad_df = pl.DataFrame({
        "count": [-1, 0, 1],
        "value": [-5, 0, 5]
    })

    # Test valid ranges
    assert validation.validate_range_values(good_df, "good_table", ["count"]) is True

    # Test invalid ranges
    assert validation.validate_range_values(bad_df, "bad_table", ["count"]) is False

    # Test with missing column
    assert validation.validate_range_values(good_df, "test_table", ["missing"]) is False

def test_validate_dimension_table(temp_dir):
    """Test dimension table validation."""
    # Create test dimension data
    good_dim = pl.DataFrame({
        "dim_sk": [1, 2, 3],
        "name": ["a", "b", "c"]
    })
    good_file = temp_dir / "good_dim.parquet"
    good_dim.write_parquet(good_file)

    bad_dim = pl.DataFrame({
        "dim_sk": [1, None, 3],
        "name": ["a", "b", "c"]
    })
    bad_file = temp_dir / "bad_dim.parquet"
    bad_dim.write_parquet(bad_file)

    # Test valid dimension
    assert validation.validate_dimension_table(good_file, "good_dim", ["dim_sk"]) is True

    # Test invalid dimension
    assert validation.validate_dimension_table(bad_file, "bad_dim", ["dim_sk"]) is False

    # Test missing file
    assert validation.validate_dimension_table(
        temp_dir / "missing.parquet", "missing", ["dim_sk"]) is False

def test_validate_fact_table(temp_dir):
    """Test fact table validation."""
    # Create test fact data
    good_fact = pl.DataFrame({
        "fk_sk": [1, 2, 3],
        "count": [0, 1, 2]
    })
    good_file = temp_dir / "good_fact.parquet"
    good_fact.write_parquet(good_file)

    bad_fact = pl.DataFrame({
        "fk_sk": [1, None, 3],
        "count": [-1, 1, 2]
    })
    bad_file = temp_dir / "bad_fact.parquet"
    bad_fact.write_parquet(bad_file)

    # Test valid fact
    assert validation.validate_fact_table(
        good_file, "good_fact", ["fk_sk"], ["count"]) is True

    # Test invalid fact (nulls and negative values)
    assert validation.validate_fact_table(
        bad_file, "bad_fact", ["fk_sk"], ["count"]) is False

    # Test missing file
    assert validation.validate_fact_table(
        temp_dir / "missing.parquet", "missing", ["fk_sk"], ["count"]) is False

import os
import pytest
from pathlib import Path
import polars as pl
import tempfile

from etl_logic import validation

# --- Fixtures ---

@pytest.fixture
def sample_dimension_df():
    """Create a sample dimension table DataFrame."""
    return pl.DataFrame({
        'dim_sk': [1, 2, 3, 4, 5],
        'code': ['A', 'B', 'C', 'D', 'E'],
        'name': ['Alpha', 'Beta', 'Charlie', 'Delta', 'Echo'],
        'category': ['Cat1', 'Cat2', 'Cat1', 'Cat3', 'Cat2'],
        'etl_processed_at': [None, None, None, None, None]
    })

@pytest.fixture
def sample_fact_df():
    """Create a sample fact table DataFrame."""
    return pl.DataFrame({
        'geo_sk': [1, 1, 2, 2, 3],
        'time_sk': [20210101, 20210101, 20210101, 20220101, 20220101],
        'dim_sk': [1, 2, 3, 4, 5],
        'count': [10, 20, 15, 25, 30],
        'etl_processed_at': [None, None, None, None, None]
    })

@pytest.fixture
def sample_fact_df_with_nulls():
    """Create a sample fact table DataFrame with nulls."""
    return pl.DataFrame({
        'geo_sk': [1, None, 2],
        'time_sk': [20210101, 20210101, None],
        'dim_sk': [1, 2, None],
        'count': [-5, 20, 15] # Also include negative count for range check test
    })

@pytest.fixture
def sample_fact_df_with_duplicates():
    """Create a sample fact table DataFrame with duplicate keys."""
    return pl.DataFrame({
        'geo_sk': [1, 1, 2, 3, 3], # Duplicate keys (1, 2021) and (3, 2022)
        'time_sk': [2021, 2021, 2022, 2022, 2022],
        'dim_sk': [1, 2, 3, 4, 4], # Duplicate (3, 2022, 4)
        'count': [10, 20, 15, 25, 30]
    })

@pytest.fixture
def sample_fact_df_with_missing_fk():
    """Create a sample fact table DataFrame with missing foreign keys."""
    return pl.DataFrame({
        'geo_sk': [1, 2, 3],
        'time_sk': [2021, 2022, 2022],
        'dim_sk': [1, 3, 99], # 99 does not exist in sample_dimension_df
        'count': [10, 15, 30]
    })

@pytest.fixture
def sample_fact_df_with_null_keys():
    """Create a sample fact table with nulls in the columns used for uniqueness check."""
    return pl.DataFrame({
        'geo_sk': [1, 1, None, 2, 2],
        'time_sk': [2021, 2021, 2021, 2022, None],
        'dim_sk': [1, 2, 3, 4, 5],
        'count': [10, 20, 15, 25, 30]
    })

# --- Tests ---

def test_validate_table_record_count_with_records(sample_fact_df):
    """Test validation of record count with records present."""
    result = validation.validate_table_record_count(sample_fact_df, "test_fact")
    assert result is True

def test_validate_table_record_count_empty():
    """Test validation of record count with empty table."""
    empty_df = pl.DataFrame({'col1': [], 'col2': []})
    result = validation.validate_table_record_count(empty_df, "empty_table")
    assert result is False

def test_validate_null_values_without_nulls(sample_fact_df):
    """Test validation of nulls when none are present."""
    result = validation.validate_null_values(
        sample_fact_df, "test_fact", ["geo_sk", "time_sk", "dim_sk"]
    )
    assert result is True

def test_validate_null_values_with_nulls(sample_fact_df_with_nulls):
    """Test validation of nulls when nulls are present."""
    result = validation.validate_null_values(
        sample_fact_df_with_nulls, "test_fact", ["geo_sk", "time_sk", "dim_sk"]
    )
    assert result is False

def test_validate_range_values_valid(sample_fact_df):
    """Test validation of range values with valid values."""
    result = validation.validate_range_values(
        sample_fact_df, "test_fact", ["count"]
    )
    assert result is True

def test_validate_range_values_invalid(sample_fact_df_with_nulls):
    """Test validation of range values with invalid values."""
    result = validation.validate_range_values(
        sample_fact_df_with_nulls, "test_fact", ["count"]
    )
    assert result is False

def test_validate_key_uniqueness_unique(sample_fact_df):
    """Test validation of key uniqueness when keys are unique."""
    result = validation.validate_key_uniqueness(
        sample_fact_df, "test_fact", ["geo_sk", "time_sk", "dim_sk"]
    )
    assert result is True

def test_validate_key_uniqueness_duplicates(sample_fact_df_with_duplicates):
    """Test validation of key uniqueness when duplicates exist."""
    result = validation.validate_key_uniqueness(
        sample_fact_df_with_duplicates, "test_fact", ["geo_sk", "time_sk", "dim_sk"]
    )
    assert result is False

def test_validate_key_uniqueness_with_null_keys(sample_fact_df_with_null_keys, caplog):
    """Test key uniqueness validation when key columns contain nulls."""
    # Arrange
    key_cols = ["geo_sk", "time_sk"]

    # Act
    result = validation.validate_key_uniqueness(sample_fact_df_with_null_keys, "fact_table_null_keys", key_cols)

    # Assert
    assert result is False # Should fail because of nulls in keys
    assert f"FAIL: Found 3 null values in keys {key_cols}" in caplog.text
    # Check if duplicates are also reported (if any exist besides nulls)
    # In this specific data, (1, 2021) is duplicated even excluding nulls
    assert f"FAIL: Found 1 duplicate key(s)" in caplog.text

def test_validate_referential_integrity_valid(sample_fact_df, sample_dimension_df):
    """Test validation of referential integrity when all keys exist."""
    result = validation.validate_referential_integrity(
        sample_fact_df, sample_dimension_df, "dim_sk", "dim_sk",
        "test_fact", "test_dimension"
    )
    assert result is True

def test_validate_key_uniqueness_logging(caplog):
    """Test detailed logging of duplicate keys."""
    df = pl.DataFrame({
        'geo_sk': [1, 1, 2],
        'time_sk': [20210101, 20210101, 20210101],
        'condition_sk': [1, 1, 2],
        'count': [10, 20, 30]
    })
    
    validation.validate_key_uniqueness(df, "test_fact", ["geo_sk", "time_sk", "condition_sk"])
    
    # Verify error logs contain duplicate details
    assert "FAIL: Found 1 duplicate key(s)" in caplog.text
    assert "Duplicate keys (first 10 shown)" in caplog.text

def test_validate_referential_integrity_invalid(sample_fact_df_with_missing_fk, sample_dimension_df):
    """Test validation of referential integrity when keys are missing."""
    result = validation.validate_referential_integrity(
        sample_fact_df_with_missing_fk, sample_dimension_df, "dim_sk", "dim_sk", 
        "test_fact", "test_dimension"
    )
    assert result is False

def test_validate_dimension_table(tmp_path, sample_dimension_df):
    """Test validation of dimension table using file path."""
    # Create a temporary parquet file with the sample dimension DataFrame
    dim_path = tmp_path / "test_dimension.parquet"
    sample_dimension_df.write_parquet(dim_path)
    
    result = validation.validate_dimension_table(
        dim_path, "test_dimension", ["dim_sk"]
    )
    assert result is True

def test_validate_fact_table(tmp_path, sample_fact_df):
    """Test validation of fact table using file path."""
    # Create a temporary parquet file with the sample fact DataFrame
    fact_path = tmp_path / "test_fact.parquet"
    sample_fact_df.write_parquet(fact_path)
    
    result = validation.validate_fact_table(
        fact_path, "test_fact", ["geo_sk", "time_sk", "dim_sk"], ["count"]
    )
    assert result is True

def test_run_all_data_quality_checks(tmp_path, sample_fact_df, sample_dimension_df):
    """Test running all data quality checks."""
    # Create temporary parquet files
    output_dir = tmp_path
    
    # Dimension files
    geo_dim_path = output_dir / "geo_dimension.parquet"
    time_dim_path = output_dir / "dim_time.parquet"
    health_dim_path = output_dir / "dim_health_condition.parquet"
    
    # Create a similar structure to expected dimension tables
    geo_dim = pl.DataFrame({
        'geo_sk': [1, 2, 3],
        'geo_code': ['10001', '10002', '10003'],
        'geo_level': ['SA1', 'SA1', 'SA1']
    })
    
    time_dim = pl.DataFrame({
        'time_sk': [20210101, 20220101],
        'full_date': ['2021-01-01', '2022-01-01']
    })
    
    health_dim = sample_dimension_df.rename({'dim_sk': 'condition_sk'})
    
    # Fact file
    fact_path = output_dir / "fact_health_conditions_refined.parquet"
    
    # Write all files
    geo_dim.write_parquet(geo_dim_path)
    time_dim.write_parquet(time_dim_path)
    health_dim.write_parquet(health_dim_path)
    
    # Create a fact that references these dimensions correctly
    fact_df = pl.DataFrame({
        'geo_sk': [1, 2, 3],
        'time_sk': [20210101, 20210101, 20220101],
        'condition_sk': [1, 2, 3],
        'demographic_sk': [1, 2, 3],
        'count': [10, 20, 30]
    })
    fact_df.write_parquet(fact_path)
    
    # Mock the logger to avoid actual logging during tests
    class MockLogger:
        def info(self, msg): pass
        def error(self, msg): pass
        def warning(self, msg): pass
    
    # Test with a limited scope (just checking the function runs)
    # In a real test, we would create all required tables and data
    try:
        result = validation.run_all_data_quality_checks(output_dir, MockLogger())
        # Expect False because we don't have all the required tables
        assert result is False
    except Exception as e:
        # We expect the function to run without exceptions, even if it fails validation
        pytest.fail(f"run_all_data_quality_checks raised {e}")

def test_validate_dim_health_condition_schema_and_joins(temp_dir):
    """Test schema and joins for dim_health_condition after ETL corrections."""
    # Set up sample dimension table
    health_dim = pl.DataFrame({
        "condition_sk": ["sk1", "sk2", "sk3"],
        "condition_code": ["code1", "code2", "code3"],
        "condition_description": ["desc1", "desc2", "desc3"]
    })
    health_file = temp_dir / "dim_health_condition.parquet"
    health_dim.write_parquet(health_file)
    
    # Set up sample fact table
    fact_data = pl.DataFrame({
        "geo_sk": [1, 2],
        "time_sk": [20230101, 20230101],
        "condition_sk": ["sk1", "sk2"],
        "count": [10, 20]
    })
    fact_file = temp_dir / "fact_health_conditions_refined.parquet"
    fact_data.write_parquet(fact_file)
    
    # Validate schema of dim_health_condition
    schema_valid = validation.validate_dimension_table(health_file, "dim_health_condition", ["condition_sk"])
    assert schema_valid is True
    
    # Validate referential integrity with fact table
    ri_valid = validation.validate_referential_integrity(fact_data, health_dim, "condition_sk", "condition_sk", "fact_health_conditions_refined", "dim_health_condition")
    assert ri_valid is True
