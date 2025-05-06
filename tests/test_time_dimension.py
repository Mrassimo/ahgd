import pytest
import sys
from pathlib import Path
import polars as pl
from datetime import date, datetime
from polars.testing import assert_frame_equal

# Ensure etl_logic is importable
TESTS_DIR = Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from etl_logic import time_dimension

def test_create_time_dimension():
    """Test that create_time_dimension generates correct output for a small date range."""
    # Test with a small range to keep tests fast
    start_year = 2021
    end_year = 2021
    
    # Create time dimension
    time_df = time_dimension.create_time_dimension(start_year, end_year)
    
    # Verify structure
    assert isinstance(time_df, pl.DataFrame)
    
    # Check expected number of rows
    # For 2021: Jan 1 to Dec 31 = 365 days
    expected_rows = 365
    assert len(time_df) == expected_rows, f"Expected {expected_rows} rows, got {len(time_df)}"
    
    # Check all required columns are present
    expected_columns = [
        'time_sk', 'full_date', 'year', 'quarter', 'month', 'month_name',
        'day_of_month', 'day_of_week', 'day_name', 'financial_year',
        'is_weekday', 'is_census_year', 'etl_processed_at'
    ]
    assert set(expected_columns).issubset(set(time_df.columns)), f"Missing columns: {set(expected_columns) - set(time_df.columns)}"
    
    # Verify data types
    assert time_df.schema['time_sk'] == pl.Int64
    assert time_df.schema['full_date'] == pl.Date
    assert time_df.schema['year'] == pl.Int32
    assert time_df.schema['quarter'] == pl.Int32
    assert time_df.schema['month'] == pl.Int32
    assert time_df.schema['day_of_month'] == pl.Int32
    assert time_df.schema['day_of_week'] == pl.Int32
    assert time_df.schema['is_weekday'] == pl.Boolean
    assert time_df.schema['is_census_year'] == pl.Boolean
    
    # Verify sample values
    # Test first day of year
    first_day = time_df.filter(pl.col("full_date") == date(2021, 1, 1))
    assert len(first_day) == 1
    assert first_day[0, "time_sk"] == 20210101
    assert first_day[0, "year"] == 2021
    assert first_day[0, "quarter"] == 1
    assert first_day[0, "month"] == 1
    assert first_day[0, "month_name"] == "January"
    assert first_day[0, "day_of_month"] == 1
    assert first_day[0, "day_name"] == "Friday"
    assert first_day[0, "financial_year"] == "2020/21"
    assert first_day[0, "is_weekday"] == True
    assert first_day[0, "is_census_year"] == True  # 2021 is a census year
    
    # Test July 1st (financial year transition)
    july_first = time_df.filter(pl.col("full_date") == date(2021, 7, 1))
    assert len(july_first) == 1
    assert july_first[0, "time_sk"] == 20210701
    assert july_first[0, "financial_year"] == "2021/22"

def test_generate_time_dimension(tmp_path):
    """Test that generate_time_dimension creates and saves a Parquet file."""
    # Create temporary output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Generate time dimension for a small date range
    result = time_dimension.generate_time_dimension(output_dir, 2021, 2021)
    
    # Verify function success
    assert result is True
    
    # Verify file was created
    output_file = output_dir / "dim_time.parquet"
    assert output_file.exists()
    
    # Try reading the file to verify it's valid
    try:
        df = pl.read_parquet(output_file)
        assert len(df) > 0
        assert 'time_sk' in df.columns
    except Exception as e:
        pytest.fail(f"Failed to read generated Parquet file: {e}")

def test_create_time_dimension_multiyear():
    """Test that create_time_dimension works correctly for multiple years."""
    # Test with multiple years
    start_year = 2020
    end_year = 2022
    
    # Create time dimension
    time_df = time_dimension.create_time_dimension(start_year, end_year)
    
    # Verify correct number of rows (3 years, including leap year 2020)
    # 2020: 366 days (leap year)
    # 2021: 365 days
    # 2022: 365 days
    expected_rows = 366 + 365 + 365
    assert len(time_df) == expected_rows
    
    # Check year distribution
    year_counts = time_df.group_by("year").agg(pl.len()).sort("year")
    assert year_counts.shape == (3, 2)
    
    # Check leap year handling (2020 should have 366 days)
    leap_year_days = time_df.filter(pl.col("year") == 2020).shape[0]
    assert leap_year_days == 366 