"""Enhanced comprehensive tests for export pipeline and output validation.

This module provides comprehensive testing for the AHGD export pipeline including:
- Production export pipeline testing with all formats
- Compression algorithm validation and performance
- Data partitioning strategies and validation
- Target output compliance testing
- Web-optimised export validation
- Data integrity preservation across formats

British English spelling is used throughout (optimise, standardise, etc.).
"""

import pytest
import tempfile
import time
import hashlib
import json
import gzip
import brotli
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import Point, Polygon
import psutil
import os

from src.loaders.production_loader import ProductionLoader
from src.loaders.format_exporters import (
    ParquetExporter, CSVExporter, JSONExporter, 
    GeoJSONExporter, WebExporter
)
from src.utils.compression_utils import CompressionAnalyzer, SizeCalculator
from src.validators.quality_checker import QualityChecker
from schemas.target_outputs import (
    ExportSpecification, DataQualityReport, WebPlatformDataStructure,
    ExportFormat, CompressionType
)


class TestComprehensiveExportPipeline:
    """Comprehensive export pipeline tests covering all formats and configurations."""
    
    @pytest.fixture(scope="class")
    def export_test_config(self):
        """Configuration for comprehensive export testing."""
        return {
            'target_performance': {
                'throughput_mb_per_second': 30,
                'max_memory_multiplier': 3.0,
                'max_export_time_seconds': 300
            },
            'compression_targets': {
                'gzip': {'ratio': 0.6, 'speed_mb_s': 20},
                'brotli': {'ratio': 0.5, 'speed_mb_s': 15}, 
                'snappy': {'ratio': 0.7, 'speed_mb_s': 100},
                'lz4': {'ratio': 0.65, 'speed_mb_s': 80}
            },
            'format_requirements': {
                'parquet': {'max_size_mb': 100, 'schema_preservation': True},
                'csv': {'max_size_mb': 50, 'encoding': 'utf-8'},
                'json': {'max_size_mb': 75, 'structure_validation': True},
                'geojson': {'max_size_mb': 200, 'geometry_validation': True},
                'xlsx': {'max_size_mb': 25, 'sheet_validation': True}
            }
        }
    
    @pytest.fixture
    def large_australian_dataset(self):
        """Create large, realistic Australian health and geographic dataset."""
        np.random.seed(42)
        
        # Create comprehensive Australian data
        n_records = 50000  # Large enough to test performance
        
        # Australian geographic structure
        states = ['NSW', 'VIC', 'QLD', 'WA', 'SA', 'TAS', 'ACT', 'NT']
        state_weights = [0.32, 0.25, 0.20, 0.10, 0.07, 0.02, 0.02, 0.02]
        
        # Generate realistic SA2 codes
        sa2_codes = []
        state_codes = []
        for i in range(n_records):
            state = np.random.choice(states, p=state_weights)
            state_codes.append(state)
            # Generate realistic SA2 code for each state
            if state == 'NSW':
                sa2_codes.append(f"1{np.random.randint(10000, 99999)}")
            elif state == 'VIC':
                sa2_codes.append(f"2{np.random.randint(10000, 99999)}")
            elif state == 'QLD':
                sa2_codes.append(f"3{np.random.randint(10000, 99999)}")
            else:
                sa2_codes.append(f"{np.random.randint(40000, 99999)}")
        
        # Generate realistic coordinates within Australia
        coords = self._generate_australian_coordinates(n_records, state_codes)
        
        data = pd.DataFrame({
            # Geographic identifiers
            'sa2_code': sa2_codes,
            'sa2_name': [f"Statistical Area Level 2 {code}" for code in sa2_codes],
            'sa3_code': [code[:4] + '0' for code in sa2_codes],
            'sa4_code': [code[:3] + '00' for code in sa2_codes],
            'state_code': state_codes,
            'state_name': [self._get_state_name(code) for code in state_codes],
            'postcode': [f"{np.random.randint(1000, 9999):04d}" for _ in range(n_records)],
            
            # Coordinates
            'latitude': coords['lat'],
            'longitude': coords['lng'],
            'centroid_lat': coords['lat'] + np.random.normal(0, 0.001, n_records),
            'centroid_lon': coords['lng'] + np.random.normal(0, 0.001, n_records),
            
            # Population and demographics
            'total_population': np.random.randint(100, 50000, n_records),
            'population_density_per_sqkm': np.random.exponential(500, n_records),
            'median_age': np.random.normal(38, 8, n_records),
            'indigenous_population_percent': np.random.beta(1, 20, n_records) * 100,
            
            # Health indicators
            'life_expectancy': np.random.normal(82.5, 3.0, n_records),
            'infant_mortality_rate': np.random.gamma(2, 1.5, n_records),
            'obesity_rate_percent': np.random.beta(2, 5, n_records) * 40,
            'diabetes_prevalence_percent': np.random.gamma(2, 2, n_records),
            'mental_health_score': np.random.normal(75, 15, n_records),
            'smoking_rate_percent': np.random.beta(2, 8, n_records) * 30,
            
            # Socioeconomic indicators
            'seifa_irsad_score': np.random.randint(500, 1200, n_records),
            'seifa_irsad_decile': np.random.randint(1, 11, n_records),
            'median_household_income': np.random.normal(65000, 20000, n_records),
            'unemployment_rate_percent': np.random.beta(2, 8, n_records) * 15,
            'education_bachelor_percent': np.random.beta(3, 5, n_records) * 60,
            
            # Environmental factors
            'air_quality_index': np.random.normal(50, 15, n_records),
            'green_space_percent': np.random.beta(2, 3, n_records) * 100,
            'noise_pollution_db': np.random.normal(45, 10, n_records),
            'water_quality_score': np.random.normal(85, 10, n_records),
            
            # Healthcare access
            'gp_per_1000_population': np.random.gamma(2, 0.5, n_records),
            'hospital_beds_per_1000': np.random.gamma(3, 1, n_records),
            'pharmacy_access_score': np.random.normal(7.5, 2, n_records),
            'specialist_wait_time_days': np.random.exponential(30, n_records),
            
            # Geographic characteristics
            'area_sqkm': np.random.exponential(50, n_records),
            'remoteness_category': np.random.choice(
                ['Major Cities', 'Inner Regional', 'Outer Regional', 'Remote', 'Very Remote'],
                n_records, p=[0.7, 0.18, 0.09, 0.02, 0.01]
            ),
            'coastal_proximity_km': np.random.exponential(100, n_records),
            
            # Data management
            'data_quality_flag': np.random.choice(
                ['High', 'Medium', 'Low'], n_records, p=[0.8, 0.15, 0.05]
            ),
            'financial_year': np.random.choice(['2020-21', '2021-22', '2022-23'], n_records),
            'data_source': np.random.choice(['ABS', 'AIHW', 'DoH', 'State_Gov'], n_records),
            'last_updated': pd.date_range(
                start='2023-01-01', end='2023-12-31', periods=n_records
            ),
            'created_timestamp': datetime.now(),
            'data_version': '1.2.3',
            
            # Text fields for testing
            'description': [f"Health profile for SA2 {code} containing demographic and health data" 
                          for code in sa2_codes],
            'notes': [f"Record {i} - comprehensive health and geographic data" 
                     if i % 10 != 0 else None for i in range(n_records)]  # Some nulls
        })
        
        # Add some controlled quality issues for testing
        # Missing values in non-critical fields
        missing_indices = np.random.choice(n_records, size=int(n_records * 0.02), replace=False)
        data.loc[missing_indices, 'notes'] = None
        
        # A few missing values in numeric fields
        missing_numeric = np.random.choice(n_records, size=int(n_records * 0.005), replace=False)
        data.loc[missing_numeric, 'median_household_income'] = None
        
        # Ensure data types are appropriate
        data = self._ensure_appropriate_dtypes(data)
        
        return data
    
    def _generate_australian_coordinates(self, n_records: int, state_codes: List[str]) -> pd.DataFrame:
        """Generate realistic coordinates within Australian state boundaries."""
        coords = []
        
        # Approximate state bounding boxes
        state_bounds = {
            'NSW': {'lat': (-37.5, -28.2), 'lng': (140.9, 153.6)},
            'VIC': {'lat': (-39.2, -33.9), 'lng': (140.9, 149.9)},
            'QLD': {'lat': (-29.0, -10.7), 'lng': (138.0, 153.6)},
            'WA': {'lat': (-35.1, -13.8), 'lng': (113.2, 129.0)},
            'SA': {'lat': (-38.1, -25.0), 'lng': (129.0, 141.0)},
            'TAS': {'lat': (-43.6, -39.6), 'lng': (143.8, 148.5)},
            'ACT': {'lat': (-35.9, -35.1), 'lng': (148.7, 149.4)},
            'NT': {'lat': (-26.0, -10.9), 'lng': (129.0, 138.0)}
        }
        
        for state in state_codes:
            bounds = state_bounds[state]
            lat = np.random.uniform(bounds['lat'][0], bounds['lat'][1])
            lng = np.random.uniform(bounds['lng'][0], bounds['lng'][1])
            coords.append({'lat': lat, 'lng': lng})
            
        return pd.DataFrame(coords)
    
    def _get_state_name(self, state_code: str) -> str:
        """Convert state code to full name."""
        state_names = {
            'NSW': 'New South Wales',
            'VIC': 'Victoria', 
            'QLD': 'Queensland',
            'WA': 'Western Australia',
            'SA': 'South Australia',
            'TAS': 'Tasmania',
            'ACT': 'Australian Capital Territory',
            'NT': 'Northern Territory'
        }
        return state_names.get(state_code, state_code)
    
    def _ensure_appropriate_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure appropriate data types for testing."""
        # Integer columns
        int_columns = [
            'total_population', 'seifa_irsad_score', 'seifa_irsad_decile'
        ]
        for col in int_columns:
            if col in data.columns:
                data[col] = data[col].fillna(0).astype('int64')
        
        # Float columns with appropriate precision
        float_columns = [
            'latitude', 'longitude', 'centroid_lat', 'centroid_lon',
            'life_expectancy', 'obesity_rate_percent', 'median_household_income'
        ]
        for col in float_columns:
            if col in data.columns:
                data[col] = data[col].astype('float64')
        
        # Categorical columns
        categorical_columns = [
            'state_code', 'remoteness_category', 'data_quality_flag', 
            'financial_year', 'data_source'
        ]
        for col in categorical_columns:
            if col in data.columns:
                data[col] = data[col].astype('category')
                
        return data
    
    @pytest.fixture
    def temp_export_directory(self):
        """Create temporary directory for export testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def production_loader(self):
        """Production loader instance for testing."""
        config = {
            'export_pipeline': {
                'performance_mode': True,
                'validation_level': 'comprehensive',
                'enable_optimisation': True
            },
            'compression': {
                'enable_auto_selection': True,
                'performance_priority': 'balanced'
            }
        }
        return ProductionLoader(config)
    
    def test_complete_multi_format_export_pipeline(
        self, production_loader, large_australian_dataset, temp_export_directory, export_test_config
    ):
        """Test complete export pipeline with all supported formats."""
        start_time = time.time()
        
        # Export all supported formats
        formats = ['parquet', 'csv', 'json', 'geojson', 'xlsx']
        
        results = production_loader.load(
            data=large_australian_dataset,
            output_path=temp_export_directory,
            formats=formats,
            compress=True,
            partition=True,
            optimise_for_web=True,
            quality_validation=True
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Basic completion validation
        assert results['pipeline_status'] == 'completed'
        assert 'export_results' in results
        assert 'validation_results' in results
        assert 'quality_results' in results
        
        # Performance validation
        data_size_mb = large_australian_dataset.memory_usage(deep=True).sum() / 1024 / 1024
        throughput = data_size_mb / total_time
        
        target_throughput = export_test_config['target_performance']['throughput_mb_per_second']
        assert throughput > target_throughput * 0.7, (
            f"Throughput {throughput:.2f} MB/s below acceptable threshold "
            f"({target_throughput * 0.7:.2f} MB/s)"
        )
        
        # Format-specific validation
        export_results = results['export_results']
        for format_type in formats:
            assert format_type in export_results['formats'], f"Format {format_type} missing from results"
            
            format_info = export_results['formats'][format_type]
            assert len(format_info['files']) > 0, f"No files exported for format {format_type}"
            
            # Validate file existence and size
            for file_info in format_info['files']:
                file_path = Path(file_info['path'])
                assert file_path.exists(), f"Export file missing: {file_path}"
                assert file_path.stat().st_size > 0, f"Export file empty: {file_path}"
                
                # Check size limits
                file_size_mb = file_path.stat().st_size / 1024 / 1024
                max_size = export_test_config['format_requirements'][format_type]['max_size_mb']
                assert file_size_mb <= max_size, (
                    f"{format_type} file {file_size_mb:.2f}MB exceeds limit {max_size}MB"
                )
        
        # Validation results check
        validation_results = results['validation_results']
        assert validation_results['overall_status'] in ['pass', 'warning']
        
        # Quality results check
        quality_results = results['quality_results']
        assert quality_results['overall_status'] in ['pass', 'warning']
        assert quality_results['overall_score'] >= 70  # Minimum quality threshold
    
    def test_compression_algorithm_performance(
        self, production_loader, large_australian_dataset, temp_export_directory, export_test_config
    ):
        """Test performance and effectiveness of different compression algorithms."""
        compression_algorithms = ['gzip', 'brotli', 'snappy', 'lz4']
        compression_results = {}
        
        for algorithm in compression_algorithms:
            if algorithm not in export_test_config['compression_targets']:
                continue
                
            target = export_test_config['compression_targets'][algorithm]
            
            # Test compression with CSV format
            start_time = time.time()
            
            results = production_loader.load(
                data=large_australian_dataset,
                output_path=temp_export_directory / f'compression_{algorithm}',
                formats=['csv'],
                compress=True,
                compression_algorithm=algorithm,
                partition=False
            )
            
            end_time = time.time()
            compression_time = end_time - start_time
            
            # Calculate performance metrics
            export_info = results['export_results']['formats']['csv']
            compressed_file = export_info['files'][0]
            
            # Estimate uncompressed size by temporarily creating uncompressed version
            uncompressed_results = production_loader.load(
                data=large_australian_dataset,
                output_path=temp_export_directory / f'uncompressed_{algorithm}',
                formats=['csv'],
                compress=False,
                partition=False
            )
            
            uncompressed_size = uncompressed_results['export_results']['formats']['csv']['files'][0]['size_bytes']
            compressed_size = compressed_file['size_bytes']
            
            compression_ratio = compressed_size / uncompressed_size
            data_size_mb = uncompressed_size / 1024 / 1024
            compression_speed = data_size_mb / compression_time
            
            compression_results[algorithm] = {
                'ratio': compression_ratio,
                'speed_mb_s': compression_speed,
                'time_seconds': compression_time
            }
            
            # Validate compression performance
            assert compression_ratio <= target['ratio'], (
                f"{algorithm} compression ratio {compression_ratio:.3f} exceeds target {target['ratio']}"
            )
            
            assert compression_speed >= target['speed_mb_s'] * 0.7, (
                f"{algorithm} speed {compression_speed:.1f} MB/s below target {target['speed_mb_s']} MB/s"
            )
        
        # Log compression comparison
        print("\nCompression Algorithm Performance:")
        for algo, metrics in compression_results.items():
            print(f"{algo}: {metrics['ratio']:.3f} ratio, {metrics['speed_mb_s']:.1f} MB/s")
    
    def test_data_partitioning_strategies(
        self, production_loader, large_australian_dataset, temp_export_directory
    ):
        """Test different data partitioning strategies and their effectiveness."""
        partitioning_strategies = [
            {'strategy': 'state_based', 'column': 'state_code'},
            {'strategy': 'size_based', 'max_rows': 10000},
            {'strategy': 'temporal', 'column': 'financial_year'},
            {'strategy': 'auto', 'column': None}
        ]
        
        for strategy_config in partitioning_strategies:
            strategy_name = strategy_config['strategy']
            
            results = production_loader.load(
                data=large_australian_dataset,
                output_path=temp_export_directory / f'partition_{strategy_name}',
                formats=['parquet'],
                compress=True,
                partition=True,
                partition_strategy=strategy_config,
                optimise_for_web=False
            )
            
            assert results['pipeline_status'] == 'completed'
            
            # Validate partitioning effectiveness
            export_info = results['export_results']['formats']['parquet']
            partition_info = export_info.get('partition_info', {})
            
            if strategy_name != 'auto':
                # Should have created multiple partitions for large dataset
                assert len(export_info['files']) > 1, (
                    f"Partitioning strategy {strategy_name} should create multiple files"
                )
            
            # Validate partition completeness by loading all partitions
            all_partitioned_data = []
            for file_info in export_info['files']:
                partition_data = pd.read_parquet(file_info['path'])
                all_partitioned_data.append(partition_data)
            
            combined_data = pd.concat(all_partitioned_data, ignore_index=True)
            
            # Check data completeness
            assert len(combined_data) == len(large_australian_dataset), (
                f"Partitioning {strategy_name} lost data: {len(combined_data)} vs {len(large_australian_dataset)}"
            )
            
            # Check that partitioning logic was applied correctly
            if strategy_name == 'state_based':
                # Each partition should contain only one state
                for file_info in export_info['files']:
                    partition_data = pd.read_parquet(file_info['path'])
                    if len(partition_data) > 0:
                        unique_states = partition_data['state_code'].nunique()
                        assert unique_states <= 2, (  # Allow some flexibility
                            f"State-based partition has {unique_states} states"
                        )
    
    def test_target_output_schema_compliance(
        self, production_loader, large_australian_dataset, temp_export_directory
    ):
        """Test compliance with target output schema specifications."""
        # Define target export specifications
        export_specs = [
            ExportSpecification(
                export_name="master_health_data",
                export_description="Complete health and geographic dataset",
                export_type="full",
                source_tables=["master_health_records"],
                output_format=ExportFormat.PARQUET,
                compression=CompressionType.SNAPPY,
                file_naming_pattern="health_data_{date}_{partition}.parquet",
                include_headers=True,
                encoding="utf-8"
            ),
            ExportSpecification(
                export_name="web_health_api",
                export_description="Web API format health data",
                export_type="filtered",
                source_tables=["master_health_records"],
                output_format=ExportFormat.JSON,
                compression=CompressionType.GZIP,
                file_naming_pattern="api_data_{date}.json.gz",
                include_headers=True,
                encoding="utf-8"
            )
        ]
        
        for spec in export_specs:
            # Export according to specification
            results = production_loader.load(
                data=large_australian_dataset,
                output_path=temp_export_directory / spec.export_name,
                formats=[spec.output_format.value],
                compress=spec.compression != CompressionType.NONE,
                compression_algorithm=spec.compression.value if spec.compression != CompressionType.NONE else None,
                encoding=spec.encoding,
                partition=False  # Simplified for testing
            )
            
            assert results['pipeline_status'] == 'completed'
            
            # Validate export specification compliance
            export_info = results['export_results']['formats'][spec.output_format.value]
            
            # Check file naming pattern compliance (basic validation)
            for file_info in export_info['files']:
                filename = file_info['filename']
                expected_extension = f".{spec.output_format.value}"
                
                if spec.compression == CompressionType.GZIP:
                    assert filename.endswith('.gz'), f"Gzip compression not reflected in filename: {filename}"
                else:
                    assert expected_extension in filename, f"File extension mismatch: {filename}"
            
            # Validate compression specification
            if spec.compression != CompressionType.NONE:
                compression_info = export_info.get('compression_info', {})
                # Should have compression information
                assert len([f for f in export_info['files'] if f.get('compression')]) > 0
            
            # Validate encoding for text formats
            if spec.output_format in [ExportFormat.CSV, ExportFormat.JSON]:
                # Test file can be read with specified encoding
                test_file = Path(export_info['files'][0]['path'])
                try:
                    with open(test_file, 'r', encoding=spec.encoding) as f:
                        content = f.read(100)  # Read first 100 chars
                    assert len(content) > 0, "File appears to be empty or encoding issues"
                except UnicodeDecodeError:
                    pytest.fail(f"File cannot be read with specified encoding {spec.encoding}")
    
    def test_web_optimisation_effectiveness(
        self, production_loader, large_australian_dataset, temp_export_directory
    ):
        """Test web optimisation features and their effectiveness."""
        # Test web-optimised vs standard export
        standard_results = production_loader.load(
            data=large_australian_dataset,
            output_path=temp_export_directory / 'standard',
            formats=['json'],
            compress=False,
            optimise_for_web=False,
            partition=False
        )
        
        web_optimised_results = production_loader.load(
            data=large_australian_dataset,
            output_path=temp_export_directory / 'web_optimised',
            formats=['json'],
            compress=True,
            optimise_for_web=True,
            target_size_kb=500,
            partition=False
        )
        
        # Both should complete successfully
        assert standard_results['pipeline_status'] == 'completed'
        assert web_optimised_results['pipeline_status'] == 'completed'
        
        # Compare file sizes
        standard_file = Path(standard_results['export_results']['formats']['json']['files'][0]['path'])
        web_file_info = web_optimised_results['export_results']['formats']['json']['files'][0]
        web_file = Path(web_file_info['path'])
        
        standard_size = standard_file.stat().st_size
        web_size = web_file.stat().st_size
        
        # Web optimised should be smaller
        size_reduction = (standard_size - web_size) / standard_size
        assert size_reduction > 0.1, f"Web optimisation achieved only {size_reduction:.1%} reduction"
        
        # Web file should be reasonably sized for web delivery
        web_size_kb = web_size / 1024
        assert web_size_kb < 2000, f"Web-optimised file {web_size_kb:.1f}KB too large for web"
        
        # Test that optimised file is still valid JSON
        try:
            with open(web_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            assert isinstance(json_data, (list, dict)), "Web-optimised JSON structure invalid"
        except json.JSONDecodeError:
            pytest.fail("Web-optimised file is not valid JSON")
    
    def test_concurrent_export_operations(
        self, production_loader, large_australian_dataset, temp_export_directory
    ):
        """Test concurrent export operations and thread safety."""
        def export_task(task_id: int, format_type: str) -> Dict[str, Any]:
            """Individual export task for concurrent testing."""
            task_data = large_australian_dataset.sample(n=10000, random_state=task_id)
            
            results = production_loader.load(
                data=task_data,
                output_path=temp_export_directory / f'concurrent_{task_id}_{format_type}',
                formats=[format_type],
                compress=True,
                partition=False
            )
            
            return {
                'task_id': task_id,
                'format': format_type,
                'status': results['pipeline_status'],
                'file_count': len(results['export_results']['formats'][format_type]['files'])
            }
        
        # Run multiple concurrent exports
        tasks = []
        formats = ['parquet', 'csv', 'json']
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            # Submit tasks
            for i in range(9):  # 3 tasks per format
                format_type = formats[i % 3]
                future = executor.submit(export_task, i, format_type)
                tasks.append(future)
            
            # Collect results
            results = []
            for future in as_completed(tasks, timeout=60):
                result = future.result()
                results.append(result)
        
        # Validate all tasks completed successfully
        assert len(results) == 9, f"Expected 9 results, got {len(results)}"
        
        for result in results:
            assert result['status'] == 'completed', f"Task {result['task_id']} failed"
            assert result['file_count'] > 0, f"Task {result['task_id']} produced no files"
    
    def test_export_data_quality_reporting(
        self, production_loader, large_australian_dataset, temp_export_directory
    ):
        """Test data quality reporting during export process."""
        results = production_loader.load(
            data=large_australian_dataset,
            output_path=temp_export_directory,
            formats=['parquet', 'csv'],
            compress=True,
            partition=False,
            quality_validation=True,
            generate_quality_report=True
        )
        
        assert results['pipeline_status'] == 'completed'
        assert 'quality_results' in results
        
        quality_results = results['quality_results']
        
        # Check quality report structure
        required_quality_metrics = [
            'completeness', 'consistency', 'validity', 'overall_score'
        ]
        
        for metric in required_quality_metrics:
            assert metric in quality_results, f"Quality metric {metric} missing"
        
        # Validate quality scores are reasonable
        completeness = quality_results['completeness']['percentage']
        assert 0 <= completeness <= 100, f"Invalid completeness score: {completeness}"
        
        overall_score = quality_results['overall_score']
        assert 0 <= overall_score <= 100, f"Invalid overall quality score: {overall_score}"
        
        # Check that quality report identifies the controlled issues we introduced
        if 'issues' in quality_results:
            issues = quality_results['issues']
            # Should detect missing values in 'notes' field
            missing_value_issues = [
                issue for issue in issues 
                if 'missing' in issue.get('type', '').lower() or 'null' in issue.get('type', '').lower()
            ]
            assert len(missing_value_issues) > 0, "Should detect missing values in test data"
    
    def test_large_dataset_memory_efficiency(
        self, production_loader, temp_export_directory
    ):
        """Test memory efficiency with very large datasets."""
        # Create a larger dataset for memory testing
        np.random.seed(42)
        n_records = 100000  # 100k records
        
        large_data = pd.DataFrame({
            'id': range(n_records),
            'large_text': ['This is a large text field with substantial content ' * 10] * n_records,
            'numeric_data': np.random.randn(n_records),
            'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_records),
            'timestamp': pd.date_range('2020-01-01', periods=n_records, freq='H')
        })
        
        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        
        results = production_loader.load(
            data=large_data,
            output_path=temp_export_directory,
            formats=['parquet'],
            compress=True,
            partition=True,
            memory_efficient=True,
            streaming=True
        )
        
        end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = final_memory - initial_memory
        data_size_mb = large_data.memory_usage(deep=True).sum() / 1024 / 1024
        memory_multiplier = memory_increase / data_size_mb
        
        # Validate memory efficiency
        assert memory_multiplier < 3.0, (
            f"Memory usage {memory_multiplier:.2f}x data size exceeds acceptable limit"
        )
        
        # Validate export completed successfully
        assert results['pipeline_status'] == 'completed'
        
        # Validate performance
        processing_time = end_time - start_time
        throughput = data_size_mb / processing_time
        
        assert throughput > 10, f"Large dataset throughput {throughput:.2f} MB/s too low"
    
    def test_export_validation_and_integrity(
        self, production_loader, large_australian_dataset, temp_export_directory
    ):
        """Test export validation and data integrity preservation."""
        # Calculate original data characteristics
        original_hash = hashlib.md5(str(large_australian_dataset.values).encode()).hexdigest()
        original_row_count = len(large_australian_dataset)
        original_column_count = len(large_australian_dataset.columns)
        original_null_count = large_australian_dataset.isnull().sum().sum()
        
        # Export with validation enabled
        results = production_loader.load(
            data=large_australian_dataset,
            output_path=temp_export_directory,
            formats=['parquet', 'csv'],
            compress=True,
            partition=False,
            validate_export=True,
            integrity_checks=True
        )
        
        assert results['pipeline_status'] == 'completed'
        
        # Test data integrity for each format
        for format_type in ['parquet', 'csv']:
            export_info = results['export_results']['formats'][format_type]
            file_path = Path(export_info['files'][0]['path'])
            
            # Load exported data
            if format_type == 'parquet':
                exported_data = pd.read_parquet(file_path)
            else:  # csv
                if file_path.suffix == '.gz':
                    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                        exported_data = pd.read_csv(f)
                else:
                    exported_data = pd.read_csv(file_path)
            
            # Validate basic structure
            assert len(exported_data) == original_row_count, (
                f"{format_type}: Row count mismatch {len(exported_data)} vs {original_row_count}"
            )
            
            # Allow for some column differences in CSV due to formatting
            if format_type == 'parquet':
                assert len(exported_data.columns) == original_column_count, (
                    f"{format_type}: Column count mismatch"
                )
            
            # Check that key columns are preserved
            key_columns = ['sa2_code', 'state_code', 'total_population', 'life_expectancy']
            for col in key_columns:
                if col in large_australian_dataset.columns:
                    assert col in exported_data.columns, f"{format_type}: Missing key column {col}"
                    
                    # For numeric columns, check values are preserved (allowing for precision differences)
                    if col in ['total_population', 'life_expectancy']:
                        original_values = large_australian_dataset[col].fillna(0)
                        exported_values = exported_data[col].fillna(0)
                        
                        # Allow small differences for floating point
                        if col == 'life_expectancy':
                            max_diff = abs(original_values - exported_values).max()
                            assert max_diff < 0.01, f"{format_type}: {col} values differ by {max_diff}"
                        else:
                            # Integer values should be exact
                            assert original_values.equals(exported_values), (
                                f"{format_type}: {col} values not preserved"
                            )
        
        # Check validation results
        validation_results = results.get('validation_results', {})
        assert validation_results.get('integrity_check_passed', False), "Integrity check should pass"