"""
Unit tests for data processing functionality.

This module tests data loading, validation, and transformation functions
used throughout the Australian Health Geography Data Analytics project.
"""

import pytest
import pandas as pd
import polars as pl
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import duckdb

# Import modules under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


class TestDataLoading:
    """Test class for data loading functions."""
    
    def test_load_csv_data_success(self, temp_dir):
        """Test successful CSV data loading."""
        # Create test CSV file
        test_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'value': [10.5, 20.3, 30.1]
        })
        
        csv_path = temp_dir / "test_data.csv"
        test_data.to_csv(csv_path, index=False)
        
        # Test loading
        loaded_data = pd.read_csv(csv_path)
        
        assert len(loaded_data) == 3
        assert list(loaded_data.columns) == ['id', 'name', 'value']
        assert loaded_data['value'].dtype == float

    def test_load_excel_data_success(self, temp_dir):
        """Test successful Excel data loading."""
        # Create test Excel file
        test_data = pd.DataFrame({
            'sa2_code': ['101021007', '101021008'],
            'sa2_name': ['Sydney CBD', 'Melbourne CBD'],
            'population': [15000, 25000]
        })
        
        excel_path = temp_dir / "test_data.xlsx"
        test_data.to_excel(excel_path, index=False)
        
        # Test loading
        loaded_data = pd.read_excel(excel_path)
        
        assert len(loaded_data) == 2
        assert 'sa2_code' in loaded_data.columns
        assert loaded_data['population'].sum() == 40000

    def test_load_parquet_data_success(self, temp_dir):
        """Test successful Parquet data loading."""
        test_data = pd.DataFrame({
            'postcode': ['2000', '3000', '4000'],
            'median_income': [65000, 72000, 62000],
            'disadvantage_score': [950, 1100, 980]
        })
        
        parquet_path = temp_dir / "test_data.parquet"
        test_data.to_parquet(parquet_path)
        
        # Test loading
        loaded_data = pd.read_parquet(parquet_path)
        
        assert len(loaded_data) == 3
        assert loaded_data['median_income'].mean() == 66333.33333333333

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading non-existent file raises appropriate error."""
        non_existent_path = temp_dir / "does_not_exist.csv"
        
        with pytest.raises(FileNotFoundError):
            pd.read_csv(non_existent_path)

    @patch('pandas.read_csv')
    def test_load_csv_with_encoding_issues(self, mock_read_csv):
        """Test handling of encoding issues in CSV loading."""
        mock_read_csv.side_effect = UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid start byte')
        
        # Should raise appropriate error
        with pytest.raises(UnicodeDecodeError):
            pd.read_csv('dummy_path.csv')

    def test_load_empty_csv(self, temp_dir):
        """Test loading empty CSV file."""
        empty_csv = temp_dir / "empty.csv"
        empty_csv.write_text("")
        
        with pytest.raises(pd.errors.EmptyDataError):
            pd.read_csv(empty_csv)


class TestDataValidation:
    """Test class for data validation functions."""
    
    def test_validate_required_columns_success(self):
        """Test successful column validation."""
        df = pd.DataFrame({
            'sa2_code': ['123', '456'],
            'population': [1000, 2000],
            'income': [50000, 60000]
        })
        
        required_columns = ['sa2_code', 'population']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        assert len(missing_columns) == 0

    def test_validate_required_columns_missing(self):
        """Test column validation with missing columns."""
        df = pd.DataFrame({
            'sa2_code': ['123', '456'],
            'population': [1000, 2000]
        })
        
        required_columns = ['sa2_code', 'population', 'income']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        assert 'income' in missing_columns

    def test_validate_data_types_correct(self):
        """Test data type validation with correct types."""
        df = pd.DataFrame({
            'sa2_code': ['123', '456'],
            'population': [1000, 2000],
            'rate': [5.5, 6.2]
        })
        
        assert df['sa2_code'].dtype == object
        assert pd.api.types.is_numeric_dtype(df['population'])
        assert pd.api.types.is_numeric_dtype(df['rate'])

    def test_validate_no_null_values_success(self):
        """Test null value validation with clean data."""
        df = pd.DataFrame({
            'sa2_code': ['123', '456'],
            'population': [1000, 2000]
        })
        
        null_counts = df.isnull().sum()
        assert null_counts.sum() == 0

    def test_validate_null_values_detected(self):
        """Test null value validation with missing data."""
        df = pd.DataFrame({
            'sa2_code': ['123', None],
            'population': [1000, np.nan]
        })
        
        null_counts = df.isnull().sum()
        assert null_counts['sa2_code'] == 1
        assert null_counts['population'] == 1

    def test_validate_value_ranges_valid(self):
        """Test value range validation with valid data."""
        df = pd.DataFrame({
            'percentage': [0.1, 0.5, 0.95],
            'count': [10, 50, 100]
        })
        
        # Percentages should be between 0 and 1
        assert (df['percentage'] >= 0).all()
        assert (df['percentage'] <= 1).all()
        
        # Counts should be positive
        assert (df['count'] > 0).all()

    def test_validate_value_ranges_invalid(self):
        """Test value range validation with invalid data."""
        df = pd.DataFrame({
            'percentage': [-0.1, 1.5, 0.5],
            'count': [-10, 0, 100]
        })
        
        # Check for invalid percentages
        invalid_percentages = (df['percentage'] < 0) | (df['percentage'] > 1)
        assert invalid_percentages.sum() == 2
        
        # Check for invalid counts
        invalid_counts = df['count'] <= 0
        assert invalid_counts.sum() == 2

    def test_duplicate_detection(self):
        """Test duplicate record detection."""
        df = pd.DataFrame({
            'sa2_code': ['123', '456', '123'],  # Duplicate SA2 code
            'year': [2021, 2021, 2021],
            'value': [100, 200, 100]
        })
        
        duplicates = df.duplicated(subset=['sa2_code', 'year'])
        assert duplicates.sum() == 1


class TestDataTransformation:
    """Test class for data transformation functions."""
    
    def test_standardise_column_names(self):
        """Test column name standardisation."""
        df = pd.DataFrame({
            'SA2 Code 2021': ['123', '456'],
            'SA2-Name': ['Area A', 'Area B'],
            'Population (Total)': [1000, 2000]
        })
        
        # Standardise column names
        df.columns = [col.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '') for col in df.columns]
        
        expected_columns = ['sa2_code_2021', 'sa2_name', 'population_total']
        assert list(df.columns) == expected_columns

    def test_convert_data_types(self):
        """Test data type conversion."""
        df = pd.DataFrame({
            'sa2_code': [123, 456],  # Should be string
            'population': ['1000', '2000'],  # Should be numeric
            'rate': ['5.5', '6.2']  # Should be float
        })
        
        # Convert types
        df['sa2_code'] = df['sa2_code'].astype(str)
        df['population'] = pd.to_numeric(df['population'])
        df['rate'] = pd.to_numeric(df['rate'])
        
        assert df['sa2_code'].dtype == object
        assert pd.api.types.is_integer_dtype(df['population'])
        assert pd.api.types.is_float_dtype(df['rate'])

    def test_handle_missing_values_fillna(self):
        """Test missing value handling with fillna."""
        df = pd.DataFrame({
            'value': [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        
        # Fill with mean
        df_filled = df.fillna(df['value'].mean())
        assert df_filled['value'].isnull().sum() == 0
        assert df_filled['value'].iloc[1] == 3.0  # Mean of 1, 3, 5

    def test_handle_missing_values_dropna(self):
        """Test missing value handling with dropna."""
        df = pd.DataFrame({
            'sa2_code': ['123', None, '456'],
            'value': [1.0, 2.0, np.nan]
        })
        
        # Drop rows with any null values
        df_clean = df.dropna()
        assert len(df_clean) == 1
        assert df_clean.iloc[0]['sa2_code'] == '123'

    def test_calculate_derived_columns(self):
        """Test calculation of derived columns."""
        df = pd.DataFrame({
            'population': [1000, 2000, 3000],
            'deaths': [10, 25, 45]
        })
        
        # Calculate death rate per 1000
        df['death_rate'] = (df['deaths'] / df['population']) * 1000
        
        expected_rates = [10.0, 12.5, 15.0]
        assert list(df['death_rate']) == expected_rates

    def test_group_and_aggregate(self):
        """Test grouping and aggregation operations."""
        df = pd.DataFrame({
            'state': ['NSW', 'NSW', 'VIC', 'VIC'],
            'population': [1000, 2000, 1500, 2500],
            'income': [50000, 60000, 55000, 65000]
        })
        
        # Group by state and calculate totals/means
        grouped = df.groupby('state').agg({
            'population': 'sum',
            'income': 'mean'
        }).reset_index()
        
        assert len(grouped) == 2
        assert grouped[grouped['state'] == 'NSW']['population'].iloc[0] == 3000
        assert grouped[grouped['state'] == 'VIC']['income'].iloc[0] == 60000


class TestDataIntegrity:
    """Test class for data integrity checks."""
    
    def test_referential_integrity_success(self):
        """Test referential integrity between datasets."""
        seifa_data = pd.DataFrame({
            'sa2_code': ['123', '456', '789']
        })
        
        health_data = pd.DataFrame({
            'sa2_code': ['123', '456', '789'],
            'mortality_rate': [5.1, 6.2, 4.8]
        })
        
        # Check all health data SA2 codes exist in SEIFA data
        missing_codes = set(health_data['sa2_code']) - set(seifa_data['sa2_code'])
        assert len(missing_codes) == 0

    def test_referential_integrity_failure(self):
        """Test referential integrity with missing references."""
        seifa_data = pd.DataFrame({
            'sa2_code': ['123', '456']
        })
        
        health_data = pd.DataFrame({
            'sa2_code': ['123', '456', '789'],  # '789' missing from SEIFA
            'mortality_rate': [5.1, 6.2, 4.8]
        })
        
        missing_codes = set(health_data['sa2_code']) - set(seifa_data['sa2_code'])
        assert '789' in missing_codes

    def test_temporal_consistency(self):
        """Test temporal consistency in time series data."""
        df = pd.DataFrame({
            'year': [2019, 2020, 2021, 2020],  # Out of order
            'value': [100, 110, 120, 115]
        })
        
        # Sort by year
        df_sorted = df.sort_values('year')
        expected_years = [2019, 2020, 2020, 2021]
        assert list(df_sorted['year']) == expected_years

    def test_logical_consistency(self):
        """Test logical consistency between related fields."""
        df = pd.DataFrame({
            'total_population': [1000, 2000, 3000],
            'male_population': [500, 1100, 1500],  # Male > Total for row 1
            'female_population': [500, 900, 1500]
        })
        
        # Check that male + female = total
        df['calculated_total'] = df['male_population'] + df['female_population']
        inconsistent = df['total_population'] != df['calculated_total']
        
        # Row 1 should be inconsistent (1100 + 900 != 2000)
        assert inconsistent.iloc[1] == True


@pytest.mark.integration
class TestDataProcessingPipeline:
    """Integration tests for complete data processing pipeline."""
    
    def test_end_to_end_csv_processing(self, temp_dir):
        """Test complete CSV processing pipeline."""
        # Create raw data file
        raw_data = pd.DataFrame({
            'SA2_CODE_2021': [123, 456, 789],
            'SA2_NAME_2021': ['Area A', 'Area B', 'Area C'],
            'Population_2021': ['1000', '2000', '3000'],
            'Median_Income': ['50000', '60000', '55000']
        })
        
        raw_file = temp_dir / "raw_data.csv"
        raw_data.to_csv(raw_file, index=False)
        
        # Load and process
        df = pd.read_csv(raw_file)
        
        # Standardise column names
        df.columns = [col.lower().replace('_2021', '') for col in df.columns]
        
        # Convert data types
        df['sa2_code'] = df['sa2_code'].astype(str)
        df['population'] = pd.to_numeric(df['population'])
        df['median_income'] = pd.to_numeric(df['median_income'])
        
        # Validate
        assert len(df) == 3
        assert df['population'].sum() == 6000
        assert df['sa2_code'].dtype == object

    @pytest.mark.slow
    def test_large_dataset_processing_performance(self):
        """Test processing performance with large dataset."""
        import time
        
        # Generate large dataset
        large_df = pd.DataFrame({
            'id': range(10000),
            'value': np.random.rand(10000),
            'category': np.random.choice(['A', 'B', 'C'], 10000)
        })
        
        start_time = time.time()
        
        # Perform typical processing operations
        result = large_df.groupby('category')['value'].agg(['mean', 'sum', 'count'])
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process reasonably quickly
        assert processing_time < 5.0  # seconds
        assert len(result) == 3