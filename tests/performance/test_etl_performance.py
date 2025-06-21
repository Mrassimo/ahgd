"""
Performance tests for ETL pipeline components.

This module contains comprehensive performance tests for:
- Extract performance
- Transform performance
- Load performance
- End-to-end pipeline performance
- Scalability testing
"""

import asyncio
import json
import pytest
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np

from src.performance.benchmarks import ETLBenchmarkSuite, BenchmarkSuite
from src.performance.profiler import PerformanceProfiler
from src.performance.monitoring import SystemMonitor, PerformanceMonitor
from src.utils.logging import get_logger

logger = get_logger()


class TestETLPerformance:
    """Test suite for ETL pipeline performance."""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.benchmark_suite = ETLBenchmarkSuite(output_dir="tests/performance/results")
        cls.profiler = PerformanceProfiler()
        cls.system_monitor = SystemMonitor(collection_interval=1.0)
        cls.performance_monitor = PerformanceMonitor()
        
        # Start monitoring
        cls.system_monitor.start_monitoring()
    
    @classmethod
    def teardown_class(cls):
        """Clean up test environment."""
        cls.system_monitor.stop_monitoring()
    
    def test_extract_csv_performance(self):
        """Test CSV extraction performance across different data sizes."""
        data_sizes = [1000, 5000, 10000]
        results = []
        
        for size in data_sizes:
            with self.profiler.profile_operation(f"extract_csv_{size}"):
                start_time = time.time()
                
                # Generate test CSV data
                test_data = self._generate_test_dataframe(size)
                temp_file = self._create_temp_csv(test_data)
                
                try:
                    # Test CSV reading performance
                    read_data = pd.read_csv(temp_file)
                    
                    execution_time = time.time() - start_time
                    throughput = size / execution_time
                    
                    results.append({
                        'data_size': size,
                        'execution_time': execution_time,
                        'throughput': throughput,
                        'records_processed': len(read_data)
                    })
                    
                    # Performance assertions
                    assert execution_time < 30.0, f"CSV extraction too slow for {size} records: {execution_time}s"
                    assert throughput > 100, f"CSV extraction throughput too low: {throughput} records/s"
                    assert len(read_data) == size, f"Data loss during extraction: expected {size}, got {len(read_data)}"
                    
                finally:
                    temp_file.unlink()
        
        # Log performance summary
        logger.info("CSV extraction performance test completed", results=results)
    
    def test_extract_json_performance(self):
        """Test JSON extraction performance."""
        data_sizes = [1000, 5000, 10000]
        results = []
        
        for size in data_sizes:
            with self.profiler.profile_operation(f"extract_json_{size}"):
                start_time = time.time()
                
                # Generate test JSON data
                test_data = self._generate_test_data(size)
                temp_file = self._create_temp_json(test_data)
                
                try:
                    # Test JSON reading performance
                    with open(temp_file, 'r') as f:
                        read_data = json.load(f)
                    
                    execution_time = time.time() - start_time
                    throughput = size / execution_time
                    
                    results.append({
                        'data_size': size,
                        'execution_time': execution_time,
                        'throughput': throughput,
                        'records_processed': len(read_data)
                    })
                    
                    # Performance assertions
                    assert execution_time < 45.0, f"JSON extraction too slow for {size} records: {execution_time}s"
                    assert throughput > 50, f"JSON extraction throughput too low: {throughput} records/s"
                    
                finally:
                    temp_file.unlink()
        
        logger.info("JSON extraction performance test completed", results=results)
    
    def test_transform_standardisation_performance(self):
        """Test data standardisation transformation performance."""
        data_sizes = [1000, 5000, 10000]
        results = []
        
        for size in data_sizes:
            with self.profiler.profile_operation(f"transform_standardise_{size}"):
                start_time = time.time()
                
                # Generate test data
                test_data = self._generate_test_dataframe(size)
                
                # Apply standardisation transformations
                transformed_data = self._apply_standardisation(test_data)
                
                execution_time = time.time() - start_time
                throughput = size / execution_time
                
                results.append({
                    'data_size': size,
                    'execution_time': execution_time,
                    'throughput': throughput,
                    'records_processed': len(transformed_data)
                })
                
                # Performance assertions
                assert execution_time < 20.0, f"Standardisation too slow for {size} records: {execution_time}s"
                assert throughput > 200, f"Standardisation throughput too low: {throughput} records/s"
                assert len(transformed_data) == len(test_data), "Data loss during transformation"
        
        logger.info("Transform standardisation performance test completed", results=results)
    
    def test_transform_validation_performance(self):
        """Test data validation transformation performance."""
        data_sizes = [1000, 5000, 10000]
        results = []
        
        for size in data_sizes:
            with self.profiler.profile_operation(f"transform_validate_{size}"):
                start_time = time.time()
                
                # Generate test data with some invalid records
                test_data = self._generate_test_dataframe_with_errors(size, error_rate=0.1)
                
                # Apply validation
                validation_results = self._apply_validation(test_data)
                
                execution_time = time.time() - start_time
                throughput = size / execution_time
                
                results.append({
                    'data_size': size,
                    'execution_time': execution_time,
                    'throughput': throughput,
                    'records_validated': len(test_data),
                    'validation_results': validation_results
                })
                
                # Performance assertions
                assert execution_time < 25.0, f"Validation too slow for {size} records: {execution_time}s"
                assert throughput > 150, f"Validation throughput too low: {throughput} records/s"
        
        logger.info("Transform validation performance test completed", results=results)
    
    def test_load_parquet_performance(self):
        """Test Parquet loading performance."""
        data_sizes = [1000, 5000, 10000]
        results = []
        
        for size in data_sizes:
            with self.profiler.profile_operation(f"load_parquet_{size}"):
                start_time = time.time()
                
                # Generate test data
                test_data = self._generate_test_dataframe(size)
                temp_file = self._create_temp_file(".parquet")
                
                try:
                    # Test Parquet writing performance
                    test_data.to_parquet(temp_file, index=False)
                    
                    # Verify by reading back
                    read_data = pd.read_parquet(temp_file)
                    
                    execution_time = time.time() - start_time
                    throughput = size / execution_time
                    file_size_mb = temp_file.stat().st_size / 1024 / 1024
                    
                    results.append({
                        'data_size': size,
                        'execution_time': execution_time,
                        'throughput': throughput,
                        'file_size_mb': file_size_mb,
                        'records_written': len(read_data)
                    })
                    
                    # Performance assertions
                    assert execution_time < 15.0, f"Parquet loading too slow for {size} records: {execution_time}s"
                    assert throughput > 300, f"Parquet loading throughput too low: {throughput} records/s"
                    assert len(read_data) == size, "Data integrity issue during Parquet loading"
                    
                finally:
                    temp_file.unlink()
        
        logger.info("Parquet loading performance test completed", results=results)
    
    def test_load_database_performance(self):
        """Test database loading performance."""
        data_sizes = [1000, 5000, 10000]
        results = []
        
        for size in data_sizes:
            with self.profiler.profile_operation(f"load_database_{size}"):
                start_time = time.time()
                
                # Generate test data
                test_data = self._generate_test_dataframe(size)
                temp_db = self._create_temp_file(".db")
                
                try:
                    # Test database loading performance
                    import sqlite3
                    conn = sqlite3.connect(str(temp_db))
                    
                    # Create table and insert data
                    test_data.to_sql('test_table', conn, index=False, if_exists='replace')
                    
                    # Verify data
                    result = conn.execute("SELECT COUNT(*) FROM test_table").fetchone()[0]
                    conn.close()
                    
                    execution_time = time.time() - start_time
                    throughput = size / execution_time
                    
                    results.append({
                        'data_size': size,
                        'execution_time': execution_time,
                        'throughput': throughput,
                        'records_loaded': result
                    })
                    
                    # Performance assertions
                    assert execution_time < 30.0, f"Database loading too slow for {size} records: {execution_time}s"
                    assert throughput > 100, f"Database loading throughput too low: {throughput} records/s"
                    assert result == size, f"Data integrity issue: expected {size}, loaded {result}"
                    
                finally:
                    temp_db.unlink()
        
        logger.info("Database loading performance test completed", results=results)
    
    def test_full_pipeline_performance(self):
        """Test end-to-end ETL pipeline performance."""
        data_sizes = [1000, 5000]  # Smaller sizes for full pipeline
        results = []
        
        for size in data_sizes:
            with self.profiler.profile_operation(f"full_pipeline_{size}"):
                start_time = time.time()
                
                # Extract: Generate and write test data
                source_data = self._generate_test_dataframe(size)
                temp_csv = self._create_temp_csv(source_data)
                temp_parquet = self._create_temp_file(".parquet")
                
                try:
                    # Extract
                    extracted_data = pd.read_csv(temp_csv)
                    
                    # Transform
                    transformed_data = self._apply_standardisation(extracted_data)
                    validation_results = self._apply_validation(transformed_data)
                    
                    # Load
                    final_data = transformed_data[transformed_data['is_valid']]
                    final_data.to_parquet(temp_parquet, index=False)
                    
                    # Verify final result
                    result_data = pd.read_parquet(temp_parquet)
                    
                    execution_time = time.time() - start_time
                    throughput = size / execution_time
                    success_rate = len(result_data) / size
                    
                    results.append({
                        'data_size': size,
                        'execution_time': execution_time,
                        'throughput': throughput,
                        'success_rate': success_rate,
                        'input_records': size,
                        'output_records': len(result_data)
                    })
                    
                    # Performance assertions
                    assert execution_time < 60.0, f"Full pipeline too slow for {size} records: {execution_time}s"
                    assert throughput > 50, f"Full pipeline throughput too low: {throughput} records/s"
                    assert success_rate > 0.8, f"Success rate too low: {success_rate}"
                    
                finally:
                    temp_csv.unlink()
                    if temp_parquet.exists():
                        temp_parquet.unlink()
        
        logger.info("Full pipeline performance test completed", results=results)
    
    def test_memory_usage_during_etl(self):
        """Test memory usage patterns during ETL operations."""
        data_size = 10000
        
        with self.profiler.profile_operation("memory_usage_etl"):
            # Generate large test data
            test_data = self._generate_test_dataframe(data_size)
            
            # Monitor memory during operations
            initial_memory = self.system_monitor.get_current_metrics().memory_percent
            
            # Perform memory-intensive operations
            temp_csv = self._create_temp_csv(test_data)
            extracted_data = pd.read_csv(temp_csv)
            
            # Multiple transformations
            for i in range(5):
                transformed_data = self._apply_standardisation(extracted_data)
                validation_results = self._apply_validation(transformed_data)
            
            final_memory = self.system_monitor.get_current_metrics().memory_percent
            memory_growth = final_memory - initial_memory
            
            # Clean up
            temp_csv.unlink()
            
            # Memory assertions
            assert memory_growth < 20.0, f"Excessive memory growth during ETL: {memory_growth}%"
            logger.info("Memory usage test completed", 
                       initial_memory=initial_memory,
                       final_memory=final_memory,
                       memory_growth=memory_growth)
    
    @pytest.mark.slow
    def test_scalability_performance(self):
        """Test ETL scalability with increasing data sizes."""
        data_sizes = [1000, 5000, 10000, 25000, 50000]
        scalability_results = []
        
        for size in data_sizes:
            with self.profiler.profile_operation(f"scalability_test_{size}"):
                start_time = time.time()
                
                # Full ETL pipeline
                source_data = self._generate_test_dataframe(size)
                temp_csv = self._create_temp_csv(source_data)
                
                try:
                    extracted_data = pd.read_csv(temp_csv)
                    transformed_data = self._apply_standardisation(extracted_data)
                    validation_results = self._apply_validation(transformed_data)
                    
                    execution_time = time.time() - start_time
                    throughput = size / execution_time
                    
                    scalability_results.append({
                        'data_size': size,
                        'execution_time': execution_time,
                        'throughput': throughput
                    })
                    
                finally:
                    temp_csv.unlink()
        
        # Analyse scalability
        self._analyse_scalability(scalability_results)
        logger.info("Scalability performance test completed", results=scalability_results)
    
    # Helper methods
    def _generate_test_dataframe(self, size: int) -> pd.DataFrame:
        """Generate test DataFrame."""
        return pd.DataFrame({
            'id': range(size),
            'name': [f'Record_{i}' for i in range(size)],
            'value': np.random.random(size),
            'category': np.random.choice(['A', 'B', 'C', 'D'], size),
            'timestamp': pd.date_range('2023-01-01', periods=size, freq='1H'),
            'coordinates_lat': np.random.uniform(-90, 90, size),
            'coordinates_lon': np.random.uniform(-180, 180, size)
        })
    
    def _generate_test_dataframe_with_errors(self, size: int, error_rate: float = 0.1) -> pd.DataFrame:
        """Generate test DataFrame with intentional errors."""
        df = self._generate_test_dataframe(size)
        
        # Introduce errors
        error_count = int(size * error_rate)
        error_indices = np.random.choice(size, error_count, replace=False)
        
        for idx in error_indices:
            # Randomly introduce different types of errors
            error_type = np.random.choice(['missing_name', 'invalid_value', 'invalid_coordinates'])
            
            if error_type == 'missing_name':
                df.loc[idx, 'name'] = None
            elif error_type == 'invalid_value':
                df.loc[idx, 'value'] = -999  # Invalid value
            elif error_type == 'invalid_coordinates':
                df.loc[idx, 'coordinates_lat'] = 999  # Invalid latitude
        
        return df
    
    def _generate_test_data(self, size: int) -> List[Dict[str, Any]]:
        """Generate test data as list of dictionaries."""
        return [
            {
                'id': i,
                'name': f'Record_{i}',
                'value': np.random.random(),
                'category': np.random.choice(['A', 'B', 'C', 'D']),
                'timestamp': '2023-01-01T00:00:00Z'
            }
            for i in range(size)
        ]
    
    def _create_temp_csv(self, data: pd.DataFrame) -> Path:
        """Create temporary CSV file."""
        temp_file = self._create_temp_file(".csv")
        data.to_csv(temp_file, index=False)
        return temp_file
    
    def _create_temp_json(self, data: List[Dict[str, Any]]) -> Path:
        """Create temporary JSON file."""
        temp_file = self._create_temp_file(".json")
        with open(temp_file, 'w') as f:
            json.dump(data, f, default=str)
        return temp_file
    
    def _create_temp_file(self, suffix: str) -> Path:
        """Create temporary file."""
        temp_dir = Path("tests/performance/temp")
        temp_dir.mkdir(exist_ok=True)
        return temp_dir / f"test_data_{int(time.time() * 1000)}{suffix}"
    
    def _apply_standardisation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply data standardisation transformations."""
        result = data.copy()
        
        # Standardise name column
        result['name'] = result['name'].str.upper()
        
        # Standardise category column
        result['category'] = result['category'].str.lower()
        
        # Normalise coordinates
        result['coordinates_lat'] = result['coordinates_lat'].clip(-90, 90)
        result['coordinates_lon'] = result['coordinates_lon'].clip(-180, 180)
        
        return result
    
    def _apply_validation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply data validation rules."""
        result = data.copy()
        
        # Validation flags
        result['is_valid'] = True
        
        # Check required fields
        result.loc[result['name'].isna(), 'is_valid'] = False
        result.loc[result['id'].isna(), 'is_valid'] = False
        
        # Check value ranges
        result.loc[(result['value'] < 0) | (result['value'] > 1), 'is_valid'] = False
        
        # Check coordinate validity
        result.loc[(result['coordinates_lat'] < -90) | (result['coordinates_lat'] > 90), 'is_valid'] = False
        result.loc[(result['coordinates_lon'] < -180) | (result['coordinates_lon'] > 180), 'is_valid'] = False
        
        return result
    
    def _analyse_scalability(self, results: List[Dict[str, Any]]):
        """Analyse scalability characteristics."""
        data_sizes = [r['data_size'] for r in results]
        execution_times = [r['execution_time'] for r in results]
        throughputs = [r['throughput'] for r in results]
        
        # Calculate complexity (simplified)
        # In practice, you'd use more sophisticated curve fitting
        if len(results) >= 3:
            # Check if execution time grows linearly with data size
            time_ratio = execution_times[-1] / execution_times[0]
            size_ratio = data_sizes[-1] / data_sizes[0]
            
            complexity_factor = time_ratio / size_ratio
            
            # Assert reasonable scalability
            assert complexity_factor < 2.0, f"Poor scalability: complexity factor {complexity_factor}"
            
            logger.info("Scalability analysis completed",
                       complexity_factor=complexity_factor,
                       time_ratio=time_ratio,
                       size_ratio=size_ratio)