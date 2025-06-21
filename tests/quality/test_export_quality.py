"""Quality tests for exported data integrity and completeness.

Validates that exported data maintains integrity, completeness, and quality
across all supported formats and configurations.

British English spelling is used throughout (optimise, standardise, etc.).
"""

import pytest
import tempfile
import hashlib
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import pyarrow.parquet as pq
import json

from src.pipelines.export_pipeline import ExportPipeline
from src.loaders.format_exporters import (
    ParquetExporter, CSVExporter, JSONExporter, GeoJSONExporter
)
from src.validators.quality_checker import QualityChecker


class TestExportQuality:
    """Quality tests for export pipeline data integrity."""
    
    @pytest.fixture
    def quality_test_data(self):
        """High-quality test dataset with known characteristics."""
        np.random.seed(12345)  # Fixed seed for reproducible quality tests
        
        n_records = 5000
        
        # Create dataset with specific quality characteristics
        data = pd.DataFrame({
            # Unique identifier - should have no duplicates
            'record_id': range(n_records),
            
            # Geographic codes - specific format requirements
            'sa2_code': [f"{11000 + i:05d}" for i in range(n_records)],
            'state_code': np.random.choice(['NSW', 'VIC', 'QLD', 'WA', 'SA', 'TAS', 'ACT', 'NT'], n_records),
            'postcode': [f"{2000 + (i % 8000):04d}" for i in range(n_records)],
            
            # Coordinates - precise Australian locations
            'latitude': np.random.uniform(-43.6, -10.7, n_records),
            'longitude': np.random.uniform(113.3, 153.6, n_records),
            
            # Numeric data with controlled quality
            'population': np.random.randint(50, 100000, n_records),
            'life_expectancy': np.random.normal(82.5, 2.0, n_records),
            'income_median': np.random.normal(65000, 15000, n_records),
            
            # Percentage data (0-100)
            'obesity_rate': np.random.beta(2, 5, n_records) * 100,
            'education_completion': np.random.beta(5, 2, n_records) * 100,
            
            # Categorical data
            'remoteness': np.random.choice(
                ['Major Cities', 'Inner Regional', 'Outer Regional', 'Remote', 'Very Remote'],
                n_records,
                p=[0.7, 0.18, 0.09, 0.02, 0.01]
            ),
            
            # Temporal data
            'financial_year': np.random.choice(['2020-21', '2021-22', '2022-23'], n_records),
            'last_updated': pd.date_range('2023-01-01', periods=n_records, freq='6H')[:n_records],
            
            # Text data
            'region_name': [f"Health Region {i % 50}" for i in range(n_records)],
            'notes': [f"Quality test record {i} with standard content" for i in range(n_records)]
        })
        
        # Introduce controlled quality issues for testing
        # 2% missing values in non-critical fields
        missing_indices = np.random.choice(n_records, size=int(n_records * 0.02), replace=False)
        data.loc[missing_indices, 'notes'] = None
        
        # 1% missing values in numeric fields
        missing_numeric = np.random.choice(n_records, size=int(n_records * 0.01), replace=False)
        data.loc[missing_numeric, 'income_median'] = None
        
        return data
    
    @pytest.fixture
    def problematic_test_data(self):
        """Dataset with various quality issues for testing validation."""
        data = pd.DataFrame({
            'id': [1, 2, 2, 4, 5],  # Duplicate ID
            'value': [10.5, None, 25.0, -999, 100.0],  # Missing and sentinel values
            'category': ['A', 'B', '', 'Invalid', 'C'],  # Empty string
            'percentage': [50.0, 150.0, -10.0, 75.0, None],  # Out of range values
            'postcode': ['2000', '999', '12345', 'ABCD', '3000'],  # Invalid formats
            'email': ['valid@test.com', 'invalid-email', '', None, 'another@test.com']
        })
        return data
    
    @pytest.fixture
    def export_pipeline(self):
        """Export pipeline configured for quality testing."""
        config = {
            'export_pipeline': {
                'quality_checks': 'comprehensive',
                'validation_level': 'strict',
                'data_integrity_checks': True
            }
        }
        return ExportPipeline(config)
    
    @pytest.fixture
    def temp_output_dir(self):
        """Temporary directory for quality test outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of DataFrame for integrity checking."""
        # Create deterministic string representation
        data_str = data.sort_index().to_string()
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _load_exported_data(self, file_path: Path, format_type: str) -> pd.DataFrame:
        """Load data from exported file based on format."""
        if format_type == 'parquet':
            return pd.read_parquet(file_path)
        elif format_type == 'csv':
            return pd.read_csv(file_path)
        elif format_type == 'json':
            # Handle potential metadata wrapper
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            if isinstance(json_data, dict) and 'data' in json_data:
                # Has metadata wrapper
                return pd.DataFrame(json_data['data'])
            else:
                return pd.DataFrame(json_data)
        else:
            raise ValueError(f"Unsupported format for loading: {format_type}")
    
    def test_data_integrity_preservation(self, export_pipeline, quality_test_data, temp_output_dir):
        """Test that data integrity is preserved across export formats."""
        original_hash = self._calculate_data_hash(quality_test_data)
        
        # Export in multiple formats
        results = export_pipeline.export_data(
            data=quality_test_data,
            output_path=temp_output_dir,
            formats=['parquet', 'csv'],
            export_options={'compress': False, 'partition': False}  # Disable for integrity testing
        )
        
        # Verify export completed successfully
        assert results['pipeline_status'] == 'completed'
        
        # Check data integrity for each format
        export_results = results['export_results']
        
        for format_type in ['parquet', 'csv']:
            format_info = export_results['formats'][format_type]
            file_info = format_info['files'][0]  # Single file (no partitioning)
            file_path = Path(file_info['path'])
            
            # Load exported data
            exported_data = self._load_exported_data(file_path, format_type)
            
            # Basic integrity checks
            assert len(exported_data) == len(quality_test_data), f"{format_type}: Row count mismatch"
            assert len(exported_data.columns) == len(quality_test_data.columns), f"{format_type}: Column count mismatch"
            
            # Column names should match
            assert set(exported_data.columns) == set(quality_test_data.columns), f"{format_type}: Column names mismatch"
            
            # For formats that preserve types, check key columns
            if format_type == 'parquet':
                # Check that unique values are preserved
                assert exported_data['record_id'].nunique() == quality_test_data['record_id'].nunique()
                assert exported_data['sa2_code'].nunique() == quality_test_data['sa2_code'].nunique()
    
    def test_numeric_precision_preservation(self, export_pipeline, quality_test_data, temp_output_dir):
        """Test that numeric precision is appropriately preserved."""
        results = export_pipeline.export_data(
            data=quality_test_data,
            output_path=temp_output_dir,
            formats=['parquet', 'csv', 'json'],
            export_options={'web_optimise': False}  # Disable optimisation to test precision
        )
        
        export_results = results['export_results']
        
        # Check precision preservation in each format
        precision_tests = {
            'latitude': 6,  # Should preserve 6 decimal places
            'longitude': 6,
            'life_expectancy': 2,  # 2 decimal places should be sufficient
            'income_median': 0,  # Whole numbers
        }
        
        for format_type in ['parquet', 'csv']:
            format_info = export_results['formats'][format_type]
            file_path = Path(format_info['files'][0]['path'])
            exported_data = self._load_exported_data(file_path, format_type)
            
            for column, expected_precision in precision_tests.items():
                original_values = quality_test_data[column].dropna()
                exported_values = exported_data[column].dropna()
                
                # Check that values are within acceptable precision
                max_diff = 10 ** (-expected_precision)
                
                # Compare first 100 values to avoid performance issues
                comparison_size = min(100, len(original_values))
                orig_sample = original_values.iloc[:comparison_size]
                exp_sample = exported_values.iloc[:comparison_size]
                
                differences = abs(orig_sample - exp_sample)
                max_observed_diff = differences.max()
                
                assert max_observed_diff <= max_diff, (
                    f"{format_type}: {column} precision loss {max_observed_diff} > {max_diff}"
                )
    
    def test_categorical_data_integrity(self, export_pipeline, quality_test_data, temp_output_dir):
        """Test that categorical data maintains integrity."""
        results = export_pipeline.export_data(
            data=quality_test_data,
            output_path=temp_output_dir,
            formats=['parquet', 'csv', 'json']
        )
        
        export_results = results['export_results']
        
        # Test categorical fields
        categorical_fields = ['state_code', 'remoteness', 'financial_year']
        
        for format_type in ['parquet', 'csv']:
            format_info = export_results['formats'][format_type]
            file_path = Path(format_info['files'][0]['path'])
            exported_data = self._load_exported_data(file_path, format_type)
            
            for field in categorical_fields:
                original_values = set(quality_test_data[field].dropna().unique())
                exported_values = set(exported_data[field].dropna().unique())
                
                assert original_values == exported_values, (
                    f"{format_type}: {field} categorical values changed"
                )
    
    def test_null_value_handling(self, export_pipeline, quality_test_data, temp_output_dir):
        """Test proper handling of null/missing values."""
        results = export_pipeline.export_data(
            data=quality_test_data,
            output_path=temp_output_dir,
            formats=['parquet', 'csv', 'json']
        )
        
        export_results = results['export_results']
        
        # Check null handling in each format
        for format_type in ['parquet', 'csv']:
            format_info = export_results['formats'][format_type]
            file_path = Path(format_info['files'][0]['path'])
            exported_data = self._load_exported_data(file_path, format_type)
            
            # Check null counts for fields known to have nulls
            fields_with_nulls = ['notes', 'income_median']
            
            for field in fields_with_nulls:
                original_null_count = quality_test_data[field].isnull().sum()
                exported_null_count = exported_data[field].isnull().sum()
                
                # Allow for format-specific null representation differences
                # but the counts should be similar
                null_count_diff = abs(original_null_count - exported_null_count)
                tolerance = max(1, original_null_count * 0.1)  # 10% tolerance or 1 record
                
                assert null_count_diff <= tolerance, (
                    f"{format_type}: {field} null count diff {null_count_diff} > tolerance {tolerance}"
                )
    
    def test_geographic_data_quality(self, export_pipeline, temp_output_dir):
        """Test quality of geographic data export."""
        # Create geographic test data
        n_points = 1000
        
        # Australian coordinate bounds
        latitudes = np.random.uniform(-43.6, -10.7, n_points)
        longitudes = np.random.uniform(113.3, 153.6, n_points)
        
        geo_data = gpd.GeoDataFrame({
            'location_id': range(n_points),
            'state': np.random.choice(['NSW', 'VIC', 'QLD'], n_points),
            'population': np.random.randint(100, 50000, n_points)
        }, geometry=[Point(xy) for xy in zip(longitudes, latitudes)], crs='EPSG:4326')
        
        results = export_pipeline.export_data(
            data=geo_data,
            output_path=temp_output_dir,
            formats=['geojson', 'parquet']
        )
        
        export_results = results['export_results']
        
        # Test GeoJSON quality
        if 'geojson' in export_results['formats']:
            geojson_info = export_results['formats']['geojson']
            geojson_path = Path(geojson_info['files'][0]['path'])
            
            # Load and validate GeoJSON
            with open(geojson_path, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)
            
            # Basic GeoJSON structure validation
            assert geojson_data['type'] == 'FeatureCollection'
            assert 'features' in geojson_data
            assert len(geojson_data['features']) == n_points
            
            # Check feature structure
            first_feature = geojson_data['features'][0]
            assert 'type' in first_feature
            assert 'geometry' in first_feature
            assert 'properties' in first_feature
            
            # Check coordinate precision
            geometry = first_feature['geometry']
            coordinates = geometry['coordinates']
            
            # Coordinates should be numbers with reasonable precision
            assert isinstance(coordinates[0], (int, float))
            assert isinstance(coordinates[1], (int, float))
            
            # Should be within Australian bounds
            lon, lat = coordinates
            assert 113.3 <= lon <= 153.6
            assert -43.6 <= lat <= -10.7
    
    def test_quality_validation_detection(self, export_pipeline, problematic_test_data, temp_output_dir):
        """Test that quality validation detects known issues."""
        results = export_pipeline.export_data(
            data=problematic_test_data,
            output_path=temp_output_dir,
            formats=['csv'],
            export_options={'validation_level': 'comprehensive'}
        )
        
        # Export should complete but validation should detect issues
        assert results['pipeline_status'] == 'completed'
        
        validation_results = results['validation_results']
        quality_results = results['quality_results']
        
        # Should detect quality issues
        assert validation_results['overall_status'] in ['warning', 'fail'] or \
               quality_results['overall_status'] in ['warning', 'fail']
        
        # Check metadata for quality metrics
        metadata = results['metadata']
        quality_metrics = metadata['quality_metrics']
        
        # Completeness should be less than 100% due to missing values
        completeness = quality_metrics['completeness']['percentage']
        assert completeness < 100, "Should detect missing values"
        
        # Consistency should flag duplicate IDs
        consistency = quality_metrics['consistency']
        assert consistency['duplicate_records'] > 0, "Should detect duplicate records"
    
    def test_data_type_consistency(self, export_pipeline, quality_test_data, temp_output_dir):
        """Test that data types are handled consistently across formats."""
        results = export_pipeline.export_data(
            data=quality_test_data,
            output_path=temp_output_dir,
            formats=['parquet', 'csv']
        )
        
        export_results = results['export_results']
        
        # Define expected data type categories
        expected_types = {
            'record_id': 'integer',
            'sa2_code': 'string',
            'state_code': 'string',
            'postcode': 'string',
            'latitude': 'float',
            'longitude': 'float',
            'population': 'integer',
            'life_expectancy': 'float',
            'income_median': 'float',
            'obesity_rate': 'float',
            'last_updated': 'datetime'
        }
        
        # Check Parquet (should preserve types well)
        parquet_info = export_results['formats']['parquet']
        parquet_path = Path(parquet_info['files'][0]['path'])
        parquet_data = pd.read_parquet(parquet_path)
        
        for column, expected_type in expected_types.items():
            actual_dtype = str(parquet_data[column].dtype)
            
            if expected_type == 'integer':
                assert 'int' in actual_dtype.lower(), f"Parquet: {column} not integer type: {actual_dtype}"
            elif expected_type == 'float':
                assert 'float' in actual_dtype.lower(), f"Parquet: {column} not float type: {actual_dtype}"
            elif expected_type == 'string':
                assert 'object' in actual_dtype.lower() or 'string' in actual_dtype.lower(), \
                    f"Parquet: {column} not string type: {actual_dtype}"
            elif expected_type == 'datetime':
                assert 'datetime' in actual_dtype.lower(), f"Parquet: {column} not datetime type: {actual_dtype}"
    
    def test_compression_quality_impact(self, export_pipeline, quality_test_data, temp_output_dir):
        """Test that compression doesn't negatively impact data quality."""
        # Export without compression
        results_uncompressed = export_pipeline.export_data(
            data=quality_test_data,
            output_path=temp_output_dir / 'uncompressed',
            formats=['csv'],
            export_options={'compress': False, 'partition': False}
        )
        
        # Export with compression
        results_compressed = export_pipeline.export_data(
            data=quality_test_data,
            output_path=temp_output_dir / 'compressed',
            formats=['csv'],
            export_options={'compress': True, 'compression_algorithm': 'gzip', 'partition': False}
        )
        
        # Both should complete successfully
        assert results_uncompressed['pipeline_status'] == 'completed'
        assert results_compressed['pipeline_status'] == 'completed'
        
        # Load both datasets
        uncompressed_path = Path(results_uncompressed['export_results']['formats']['csv']['files'][0]['path'])
        compressed_files = results_compressed['export_results']['formats']['csv']['files']
        
        # Find the compressed file (might have .gz extension)
        compressed_path = None
        for file_info in compressed_files:
            file_path = Path(file_info['path'])
            if file_path.exists():
                compressed_path = file_path
                break
        
        assert compressed_path is not None, "Compressed file not found"
        
        # Load uncompressed data
        uncompressed_data = pd.read_csv(uncompressed_path)
        
        # Load compressed data (handle .gz files)
        if compressed_path.suffix == '.gz':
            import gzip
            with gzip.open(compressed_path, 'rt', encoding='utf-8') as f:
                compressed_data = pd.read_csv(f)
        else:
            compressed_data = pd.read_csv(compressed_path)
        
        # Data should be identical
        assert len(uncompressed_data) == len(compressed_data), "Compression changed row count"
        assert len(uncompressed_data.columns) == len(compressed_data.columns), "Compression changed column count"
        
        # Check key numeric columns for equality
        numeric_columns = ['latitude', 'longitude', 'population', 'life_expectancy']
        for column in numeric_columns:
            if column in uncompressed_data.columns:
                uncompressed_values = uncompressed_data[column].fillna(0)
                compressed_values = compressed_data[column].fillna(0)
                
                # Should be essentially identical (allowing for tiny floating point differences)
                max_diff = abs(uncompressed_values - compressed_values).max()
                assert max_diff < 1e-10, f"Compression affected {column} values: max diff {max_diff}"
    
    def test_partitioning_completeness(self, export_pipeline, quality_test_data, temp_output_dir):
        """Test that partitioning doesn't lose data."""
        # Create larger dataset to justify partitioning
        large_data = pd.concat([quality_test_data] * 3, ignore_index=True)
        large_data['record_id'] = range(len(large_data))  # Ensure unique IDs
        
        results = export_pipeline.export_data(
            data=large_data,
            output_path=temp_output_dir,
            formats=['parquet'],
            export_options={'partition': True, 'partition_strategy': 'state_based'}
        )
        
        assert results['pipeline_status'] == 'completed'
        
        # Collect all partition files
        parquet_info = results['export_results']['formats']['parquet']
        partition_files = [Path(file_info['path']) for file_info in parquet_info['files']]
        
        # Load all partitions
        all_partitioned_data = []
        for partition_file in partition_files:
            partition_data = pd.read_parquet(partition_file)
            all_partitioned_data.append(partition_data)
        
        # Combine all partitions
        combined_data = pd.concat(all_partitioned_data, ignore_index=True)
        
        # Check completeness
        assert len(combined_data) == len(large_data), "Partitioning lost records"
        assert len(combined_data.columns) == len(large_data.columns), "Partitioning lost columns"
        
        # Check that all states are represented
        original_states = set(large_data['state_code'].unique())
        partitioned_states = set(combined_data['state_code'].unique())
        assert original_states == partitioned_states, "Partitioning lost states"
        
        # Check that record IDs are unique and complete
        original_ids = set(large_data['record_id'])
        partitioned_ids = set(combined_data['record_id'])
        assert original_ids == partitioned_ids, "Partitioning affected record IDs"
    
    def test_metadata_quality_reporting(self, export_pipeline, quality_test_data, temp_output_dir):
        """Test that metadata accurately reports data quality metrics."""
        results = export_pipeline.export_data(
            data=quality_test_data,
            output_path=temp_output_dir,
            formats=['parquet']
        )
        
        metadata = results['metadata']
        
        # Check data schema accuracy
        data_schema = metadata['data_schema']
        assert data_schema['summary_statistics']['total_fields'] == len(quality_test_data.columns)
        
        # Check field-level metadata
        fields = data_schema['fields']
        for column in quality_test_data.columns:
            assert column in fields, f"Column {column} missing from metadata"
            
            field_info = fields[column]
            
            # Check null count accuracy
            reported_nulls = field_info['null_count']
            actual_nulls = quality_test_data[column].isnull().sum()
            assert reported_nulls == actual_nulls, f"{column}: null count mismatch"
            
            # Check unique count
            reported_unique = field_info['unique_count']
            actual_unique = quality_test_data[column].nunique()
            assert reported_unique == actual_unique, f"{column}: unique count mismatch"
        
        # Check quality metrics
        quality_metrics = metadata['quality_metrics']
        
        # Completeness calculation
        completeness = quality_metrics['completeness']
        total_cells = len(quality_test_data) * len(quality_test_data.columns)
        null_cells = quality_test_data.isnull().sum().sum()
        expected_completeness = (total_cells - null_cells) / total_cells * 100
        
        reported_completeness = completeness['percentage']
        assert abs(reported_completeness - expected_completeness) < 0.1, \
            f"Completeness calculation error: reported {reported_completeness}, expected {expected_completeness}"
    
    @pytest.mark.parametrize('format_type', ['parquet', 'csv', 'json'])
    def test_format_specific_quality_preservation(self, export_pipeline, quality_test_data, temp_output_dir, format_type):
        """Test quality preservation for specific formats."""
        results = export_pipeline.export_data(
            data=quality_test_data,
            output_path=temp_output_dir,
            formats=[format_type]
        )
        
        assert results['pipeline_status'] == 'completed'
        
        # Verify validation passed or has acceptable warnings
        validation_results = results['validation_results']
        assert validation_results['overall_status'] in ['pass', 'warning']
        
        # Check format-specific validation
        format_validations = validation_results.get('format_validations', {})
        if format_type in format_validations:
            format_validation = format_validations[format_type]
            assert format_validation['status'] in ['pass', 'warning']
        
        # Verify file quality
        export_results = results['export_results']
        format_info = export_results['formats'][format_type]
        
        for file_info in format_info['files']:
            file_path = Path(file_info['path'])
            assert file_path.exists(), f"{format_type} file not created"
            assert file_path.stat().st_size > 0, f"{format_type} file is empty"
            
            # Basic format-specific checks
            if format_type == 'json':
                # JSON should be valid
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)  # Will raise if invalid JSON
                assert isinstance(json_data, (list, dict)), "JSON should be list or dict"
    
    def test_export_reproducibility(self, export_pipeline, quality_test_data, temp_output_dir):
        """Test that exports are reproducible with same inputs."""
        # Export data twice with identical settings
        export_options = {
            'compress': True,
            'partition': False,
            'web_optimise': False,
            'random_seed': 42  # If supported
        }
        
        results1 = export_pipeline.export_data(
            data=quality_test_data,
            output_path=temp_output_dir / 'export1',
            formats=['parquet'],
            export_options=export_options
        )
        
        results2 = export_pipeline.export_data(
            data=quality_test_data,
            output_path=temp_output_dir / 'export2',
            formats=['parquet'],
            export_options=export_options
        )
        
        # Both should complete successfully
        assert results1['pipeline_status'] == 'completed'
        assert results2['pipeline_status'] == 'completed'
        
        # Load both exports
        file1_path = Path(results1['export_results']['formats']['parquet']['files'][0]['path'])
        file2_path = Path(results2['export_results']['formats']['parquet']['files'][0]['path'])
        
        data1 = pd.read_parquet(file1_path)
        data2 = pd.read_parquet(file2_path)
        
        # Data should be identical
        assert len(data1) == len(data2), "Reproducibility: row count differs"
        assert len(data1.columns) == len(data2.columns), "Reproducibility: column count differs"
        assert list(data1.columns) == list(data2.columns), "Reproducibility: column order differs"
        
        # Check a few key columns for exact equality
        key_columns = ['record_id', 'sa2_code', 'state_code', 'population']
        for column in key_columns:
            if column in data1.columns:
                assert data1[column].equals(data2[column]), f"Reproducibility: {column} values differ"
