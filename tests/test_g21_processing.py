"""Unit tests for G21 (Health Conditions by Characteristics) processing."""
import pytest
from unittest.mock import patch
from pathlib import Path
import polars as pl

from etl_logic import dimensions
from etl_logic.tables.g21_conditions_by_characteristics import process_g21_file

# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "test_data"
G21_SAMPLE_FILE = TEST_DATA_DIR / "g21_sample.csv"

# Mock dimension lookups
def mock_get_health_condition_sk(condition):
    """Mock health condition SK lookup."""
    condition_map = {
        "Arthritis": 1,
        "Asthma": 2
    }
    return pl.lit(condition_map.get(condition, -1))

def mock_get_person_characteristic_sk(char_type, char_code):
    """Mock person characteristic SK lookup."""
    char_map = {
        ("age_group", "25_34"): 101,
        ("age_group", "35_44"): 102,
        ("sex", "M"): 201
    }
    return pl.lit(char_map.get((char_type, char_code), -1))


def test_process_g21_file():
    """Test processing G21 file into long format."""
    result = process_g21_file(G21_SAMPLE_FILE)
    
    assert result is not None
    assert isinstance(result, pl.DataFrame)
    
    # Check expected columns
    expected_cols = [
        'geo_code',
        'characteristic_type', 
        'characteristic_code',
        'condition',
        'count'
    ]
    assert all(col in result.columns for col in expected_cols)
    
    # Check some sample values
    df = result.to_pandas()
    
    # Verify age group characteristic
    age_rows = df[df['characteristic_type'] == 'age_group']
    assert len(age_rows) > 0
    
    # Verify characteristic_sk was properly set
    assert all(age_rows['characteristic_sk'].isin([101, 102]))
    
    assert set(age_rows['characteristic_code']) == {'25_34', '35_44'}
    
    # Verify sex characteristic
    sex_rows = df[df['characteristic_type'] == 'sex'] 
    assert len(sex_rows) > 0
    assert set(sex_rows['characteristic_code']) == {'M'}
    
    # Verify conditions
    assert set(df['condition']) == {'Arthritis', 'Asthma'}

@patch.object(dimensions, 'get_health_condition_sk', mock_get_health_condition_sk)
def test_process_g21_unpivot_file_missing_geo():
    """Test handling of missing geographic column."""
    # Create test data without geo column
    test_data = """P_Age_25_34_Arthritis,P_Age_25_34_Asthma
12,8
15,10"""
    
    test_file = TEST_DATA_DIR / "g21_missing_geo.csv"
    test_file.write_text(test_data)
    
    result = process_g21_file(test_file)
    assert result is None

@patch.object(dimensions, 'get_health_condition_sk', mock_get_health_condition_sk)
def test_process_g21_unpivot_file_empty():
    """Test handling of empty input file."""
    test_file = TEST_DATA_DIR / "g21_empty.csv"
    test_file.write_text("")
    
    result = process_g21_file(test_file)
    assert result is None

@patch.object(dimensions, 'get_health_condition_sk', mock_get_health_condition_sk)
@patch.object(dimensions, 'get_person_characteristic_sk', mock_get_person_characteristic_sk)
@pytest.mark.skip(reason="Additional G21 processing tests to be implemented")
def test_g21_additional_processing():
    """Reserved for future G21 processing tests"""
    pass