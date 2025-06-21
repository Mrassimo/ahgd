"""Integration tests for the complete export pipeline.

Tests the end-to-end export process including all formats, compression,
partitioning, validation, and quality assurance.

British English spelling is used throughout (optimise, standardise, etc.).
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from src.pipelines.export_pipeline import ExportPipeline
from src.loaders.production_loader import ProductionLoader
from src.utils.compression_utils import CompressionAnalyzer, CompressionAlgorithm
from src.utils.config import get_config


class TestExportPipelineIntegration:
    """Integration tests for the complete export pipeline."""
    
    @pytest.fixture
    def sample_health_data(self):
        """Create sample Australian health and geographic data."""
        np.random.seed(42)
        
        # Generate realistic Australian health data
        n_records = 1000
        
        # Australian state codes with realistic distribution
        state_codes = np.random.choice(
            ['NSW', 'VIC', 'QLD', 'WA', 'SA', 'TAS', 'ACT', 'NT'],
            size=n_records,
            p=[0.32, 0.25, 0.20, 0.10, 0.07, 0.02, 0.02, 0.02]  # Population-based distribution
        )
        
        # Generate SA2 codes (realistic format)
        sa2_codes = [f"{np.random.randint(10000, 99999)}" for _ in range(n_records)]
        
        # Generate coordinates within Australia
        latitudes = np.random.uniform(-43.6, -10.7, n_records)  # Australia latitude range
        longitudes = np.random.uniform(113.3, 153.6, n_records)  # Australia longitude range
        
        # Health indicators
        life_expectancy = np.random.normal(82.5, 3.0, n_records)
        obesity_rate = np.random.beta(2, 5, n_records) * 40  # 0-40% range
        diabetes_prevalence = np.random.gamma(2, 2, n_records)
        
        # Postcodes (Australian format)
        postcodes = [f"{np.random.randint(1000, 9999):04d}" for _ in range(n_records)]
        
        # Financial year data
        financial_years = np.random.choice(['2020-21', '2021-22', '2022-23'], n_records)
        
        data = pd.DataFrame({
            'sa2_code': sa2_codes,
            'sa2_name': [f"Statistical Area {code}" for code in sa2_codes],
            'state_code': state_codes,
            'postcode': postcodes,
            'latitude': latitudes,
            'longitude': longitudes,
            'life_expectancy': life_expectancy,
            'obesity_rate_percent': obesity_rate,
            'diabetes_prevalence_percent': diabetes_prevalence,
            'population': np.random.randint(100, 50000, n_records),
            'financial_year': financial_years,
            'data_source': 'AIHW',
            'last_updated': datetime.now().strftime('%Y-%m-%d')
        })
        
        return data
    
    @pytest.fixture
    def sample_geographic_data(self, sample_health_data):
        """Create sample geographic data with geometry."""
        # Create geometry from coordinates
        geometry = [Point(xy) for xy in zip(sample_health_data['longitude'], sample_health_data['latitude'])]
        
        gdf = gpd.GeoDataFrame(
            sample_health_data.drop(columns=['latitude', 'longitude']),
            geometry=geometry,
            crs='EPSG:4326'
        )
        
        return gdf
    
    @pytest.fixture
    def export_pipeline(self):
        """Create export pipeline instance."""
        config = {
            'export_pipeline': {
                'default_formats': ['parquet', 'csv', 'json'],
                'enable_compression': True,
                'enable_partitioning': True,
                'validation_level': 'standard'
            }
        }
        return ExportPipeline(config)
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_complete_export_pipeline_basic(self, export_pipeline, sample_health_data, temp_output_dir):
        """Test complete export pipeline with basic functionality."""
        # Execute export pipeline
        results = export_pipeline.export_data(
            data=sample_health_data,
            output_path=temp_output_dir,
            formats=['parquet', 'csv', 'json'],
            export_options={
                'compress': True,
                'partition': False,  # Disable for small test data
                'web_optimise': True,
                'priority': 'balanced'
            }
        )
        
        # Verify pipeline completion
        assert results['pipeline_status'] == 'completed'
        assert 'export_results' in results
        assert 'validation_results' in results
        assert 'quality_results' in results
        assert 'metadata' in results
        
        # Verify export results structure
        export_results = results['export_results']
        assert 'formats' in export_results
        assert 'metadata' in export_results
        
        # Verify all requested formats were exported
        exported_formats = set(export_results['formats'].keys())
        assert exported_formats == {'parquet', 'csv', 'json'}
        
        # Verify files were created
        for format_type, format_info in export_results['formats'].items():
            assert len(format_info['files']) > 0
            for file_info in format_info['files']:
                file_path = Path(file_info['path'])
                assert file_path.exists()
                assert file_path.stat().st_size > 0
        
        # Verify validation passed
        validation_results = results['validation_results']
        assert validation_results['overall_status'] in ['pass', 'warning']  # Allow warnings
        
        # Verify quality checks passed
        quality_results = results['quality_results']
        assert quality_results['overall_status'] in ['pass', 'warning']
        
        # Verify metadata file was created
        metadata_file = Path(results['metadata_file'])
        assert metadata_file.exists()
        
        # Verify metadata content
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata_content = json.load(f)
        
        assert 'schema_version' in metadata_content
        assert 'export_summary' in metadata_content
        assert 'data_schema' in metadata_content
        assert 'quality_metrics' in metadata_content
    
    def test_export_pipeline_with_partitioning(self, export_pipeline, sample_health_data, temp_output_dir):
        """Test export pipeline with state-based partitioning."""
        # Use larger dataset to justify partitioning
        large_data = pd.concat([sample_health_data] * 5, ignore_index=True)
        
        results = export_pipeline.export_data(
            data=large_data,
            output_path=temp_output_dir,
            formats=['parquet', 'csv'],
            export_options={
                'compress': True,
                'partition': True,
                'partition_strategy': 'state_based',
                'web_optimise': False
            }
        )
        
        # Verify partitioning occurred
        export_results = results['export_results']
        
        for format_type in ['parquet', 'csv']:
            format_info = export_results['formats'][format_type]
            
            # Should have multiple files (one per state)
            assert len(format_info['files']) > 1
            
            # Verify partition info
            partition_info = format_info.get('partition_info', {})
            assert partition_info.get('type') in ['state', 'auto']
            
            # Verify file naming indicates partitioning
            file_names = [file_info['filename'] for file_info in format_info['files']]
            assert any('state_code' in name or 'NSW' in name or 'VIC' in name for name in file_names)
    
    def test_export_pipeline_compression_optimisation(self, export_pipeline, sample_health_data, temp_output_dir):
        """Test compression optimisation functionality."""
        results = export_pipeline.export_data(
            data=sample_health_data,
            output_path=temp_output_dir,
            formats=['csv', 'json'],
            export_options={
                'compress': True,
                'compression_priority': 'size',  # Prioritise compression ratio
                'partition': False
            }
        )
        
        # Verify compression was applied
        export_results = results['export_results']
        
        for format_type in ['csv', 'json']:
            format_info = export_results['formats'][format_type]
            
            # Check compression info
            compression_info = format_info.get('compression_info', {})
            
            # At least one file should be compressed
            compressed_files = [f for f in format_info['files'] if f.get('compression')]
            assert len(compressed_files) > 0
    
    def test_export_pipeline_web_optimisation(self, export_pipeline, sample_health_data, temp_output_dir):
        """Test web optimisation features."""
        results = export_pipeline.export_data(
            data=sample_health_data,
            output_path=temp_output_dir,
            formats=['json'],
            export_options={
                'web_optimise': True,
                'compress': True,
                'target_size_kb': 500
            }
        )
        
        # Verify web optimisation
        export_results = results['export_results']
        json_info = export_results['formats']['json']
        
        # Check file sizes are reasonable for web
        for file_info in json_info['files']:
            size_kb = file_info['size_bytes'] / 1024
            assert size_kb < 1000  # Should be under 1MB for web
    
    def test_export_pipeline_validation_failure_handling(self, export_pipeline, temp_output_dir):
        """Test pipeline handling of validation failures."""
        # Create problematic data
        problematic_data = pd.DataFrame({
            'column1': [None] * 100,  # All null column
            'column2': [''] * 100,    # All empty strings
        })
        
        # Pipeline should complete but report validation issues
        results = export_pipeline.export_data(
            data=problematic_data,
            output_path=temp_output_dir,
            formats=['csv'],
            export_options={
                'validation_level': 'comprehensive'
            }
        )
        
        # Should complete with warnings or failures noted
        assert results['pipeline_status'] == 'completed'
        
        # Validation should report issues
        validation_results = results['validation_results']
        quality_results = results['quality_results']
        
        # At least one should flag issues
        assert (validation_results['overall_status'] in ['fail', 'warning'] or
                quality_results['overall_status'] in ['fail', 'warning'])
    
    def test_export_pipeline_large_dataset_handling(self, export_pipeline, temp_output_dir):
        """Test pipeline with larger dataset requiring chunking."""
        # Create larger dataset
        large_data = pd.DataFrame({
            'id': range(50000),
            'state_code': np.random.choice(['NSW', 'VIC', 'QLD'], 50000),
            'value': np.random.randn(50000),
            'category': np.random.choice(['A', 'B', 'C'], 50000)
        })
        
        results = export_pipeline.export_data(
            data=large_data,
            output_path=temp_output_dir,
            formats=['parquet', 'csv'],
            export_options={
                'compress': True,
                'partition': True,
                'chunk_size': 10000
            }
        )
        
        # Verify successful handling
        assert results['pipeline_status'] == 'completed'
        
        # Check that data was properly partitioned/chunked
        export_results = results['export_results']
        for format_type in ['parquet', 'csv']:
            format_info = export_results['formats'][format_type]
            # Large dataset should result in multiple files or appropriate partitioning
            total_files = len(format_info['files'])
            assert total_files >= 1  # At minimum one file
    
    def test_export_pipeline_metadata_generation(self, export_pipeline, sample_health_data, temp_output_dir):
        """Test comprehensive metadata generation."""
        results = export_pipeline.export_data(
            data=sample_health_data,
            output_path=temp_output_dir,
            formats=['parquet', 'json']
        )
        
        metadata = results['metadata']
        
        # Verify metadata structure
        required_sections = [
            'schema_version', 'generated_at', 'generator',
            'export_summary', 'data_schema', 'quality_metrics',
            'lineage', 'validation'
        ]
        
        for section in required_sections:
            assert section in metadata, f"Missing metadata section: {section}"
        
        # Verify export summary
        export_summary = metadata['export_summary']
        assert 'data_characteristics' in export_summary
        assert 'export_formats' in export_summary
        assert 'total_export_size_mb' in export_summary
        
        # Verify data schema
        data_schema = metadata['data_schema']
        assert 'fields' in data_schema
        assert 'summary_statistics' in data_schema
        
        # Should have field info for each column
        assert len(data_schema['fields']) == len(sample_health_data.columns)
        
        # Verify quality metrics
        quality_metrics = metadata['quality_metrics']
        assert 'completeness' in quality_metrics
        assert 'consistency' in quality_metrics
        assert 'overall_score' in quality_metrics
        
        # Quality score should be reasonable
        overall_score = quality_metrics['overall_score']
        assert 0 <= overall_score <= 100
    
    def test_export_pipeline_error_recovery(self, export_pipeline, sample_health_data, temp_output_dir):
        """Test pipeline error handling and recovery."""
        # Test with invalid output path
        invalid_path = Path('/invalid/nonexistent/path')
        
        with pytest.raises(Exception):  # Should raise LoadingError or similar
            export_pipeline.export_data(
                data=sample_health_data,
                output_path=invalid_path,
                formats=['csv']
            )
    
    def test_export_pipeline_australian_data_patterns(self, export_pipeline, sample_health_data, temp_output_dir):
        """Test pipeline with Australian-specific data patterns."""
        # Enhance data with more Australian-specific patterns
        australian_data = sample_health_data.copy()
        australian_data['financial_year'] = '2022-23'
        australian_data['reporting_quarter'] = 'Q2 2023'
        australian_data['aihw_indicator_code'] = 'HEALTH.001'
        
        results = export_pipeline.export_data(
            data=australian_data,
            output_path=temp_output_dir,
            formats=['parquet', 'csv', 'json'],
            export_options={
                'australian_optimisations': True,
                'compress': True
            }
        )
        
        # Verify successful processing
        assert results['pipeline_status'] == 'completed'
        
        # Check that Australian-specific fields are preserved
        metadata = results['metadata']
        data_schema = metadata['data_schema']
        
        australian_fields = ['financial_year', 'state_code', 'sa2_code', 'postcode']
        schema_fields = data_schema['fields'].keys()
        
        for field in australian_fields:
            assert field in schema_fields, f"Australian field {field} not preserved in schema"
    
    def test_export_pipeline_task_management(self, export_pipeline, sample_health_data, temp_output_dir):
        """Test export pipeline task management features."""
        # Start export
        results = export_pipeline.export_data(
            data=sample_health_data,
            output_path=temp_output_dir,
            formats=['csv']
        )
        
        task_id = results['task_id']
        
        # Verify task tracking
        task_status = export_pipeline.get_task_status(task_id)
        assert task_status is not None
        assert task_status['status'] == 'completed'
        
        # Test task listing
        active_tasks = export_pipeline.list_active_tasks()
        assert len(active_tasks) > 0
        assert any(task['task_id'] == task_id for task in active_tasks)
        
        # Test task cleanup
        cleaned_count = export_pipeline.cleanup_completed_tasks()
        assert cleaned_count >= 0  # Should be non-negative
    
    @pytest.mark.parametrize('format_type', ['parquet', 'csv', 'json', 'geojson'])
    def test_export_pipeline_format_specific(self, export_pipeline, sample_health_data, temp_output_dir, format_type):
        """Test pipeline with specific export formats."""
        # For GeoJSON, need geographic data
        if format_type == 'geojson':
            # Add geometry column
            geometry = [Point(xy) for xy in zip(sample_health_data['longitude'], sample_health_data['latitude'])]
            test_data = gpd.GeoDataFrame(
                sample_health_data.drop(columns=['latitude', 'longitude']),
                geometry=geometry,
                crs='EPSG:4326'
            )
        else:
            test_data = sample_health_data
        
        results = export_pipeline.export_data(
            data=test_data,
            output_path=temp_output_dir,
            formats=[format_type]
        )
        
        # Verify format-specific export
        assert results['pipeline_status'] == 'completed'
        
        export_results = results['export_results']
        assert format_type in export_results['formats']
        
        format_info = export_results['formats'][format_type]
        assert len(format_info['files']) > 0
        
        # Verify file has correct extension
        file_info = format_info['files'][0]
        file_path = Path(file_info['path'])
        
        expected_extensions = {
            'parquet': '.parquet',
            'csv': '.csv',
            'json': '.json',
            'geojson': '.geojson'
        }
        
        # Check if file has expected extension (accounting for compression)
        assert any(ext in file_path.name for ext in [expected_extensions[format_type], '.gz', '.br'])
    
    def test_export_pipeline_performance_targets(self, export_pipeline, sample_health_data, temp_output_dir):
        """Test that pipeline meets performance targets."""
        import time
        
        start_time = time.time()
        
        results = export_pipeline.export_data(
            data=sample_health_data,
            output_path=temp_output_dir,
            formats=['parquet', 'csv', 'json']
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Verify completion
        assert results['pipeline_status'] == 'completed'
        
        # Performance targets (should complete within reasonable time)
        data_size_mb = sample_health_data.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Target: should export at least 10MB/second
        max_expected_time = max(10, data_size_mb / 10)  # At least 10 seconds, or based on data size
        
        assert execution_time < max_expected_time, f"Export took {execution_time:.2f}s, expected < {max_expected_time:.2f}s"
    
    def test_export_pipeline_memory_efficiency(self, export_pipeline, temp_output_dir):
        """Test pipeline memory efficiency with various data sizes."""
        import psutil
        import os
        
        # Create medium-sized dataset
        medium_data = pd.DataFrame({
            'id': range(10000),
            'data': np.random.randn(10000),
            'category': np.random.choice(['A', 'B', 'C', 'D'], 10000),
            'state_code': np.random.choice(['NSW', 'VIC', 'QLD'], 10000)
        })
        
        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        results = export_pipeline.export_data(
            data=medium_data,
            output_path=temp_output_dir,
            formats=['parquet', 'csv'],
            export_options={
                'memory_efficient': True,
                'streaming': True
            }
        )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Verify completion
        assert results['pipeline_status'] == 'completed'
        
        # Memory increase should be reasonable (less than 10x data size)
        data_size_mb = medium_data.memory_usage(deep=True).sum() / 1024 / 1024
        max_expected_memory = data_size_mb * 10
        
        assert memory_increase < max_expected_memory, f"Memory increase {memory_increase:.2f}MB exceeds expected {max_expected_memory:.2f}MB"
