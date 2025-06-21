"""
Performance benchmarking suite for the AHGD ETL pipeline.

This module provides comprehensive benchmarking capabilities including:
- ETL pipeline performance benchmarks
- Data processing performance tests
- Validation performance benchmarks
- Loading performance tests
- Regression testing framework
"""

import asyncio
import json
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from ..utils.logging import get_logger, track_lineage
from ..utils.interfaces import ProcessingMetadata, ProcessingStatus, DataRecord, DataBatch
from .profiler import PerformanceProfiler

logger = get_logger()


@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""
    benchmark_id: str
    benchmark_name: str
    execution_time: float
    throughput: float  # records per second
    memory_usage_mb: float
    cpu_usage_percent: float
    success_rate: float
    error_count: int
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'benchmark_id': self.benchmark_id,
            'benchmark_name': self.benchmark_name,
            'execution_time': self.execution_time,
            'throughput': self.throughput,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'success_rate': self.success_rate,
            'error_count': self.error_count,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'performance_metrics': self.performance_metrics
        }


@dataclass
class BenchmarkSuite:
    """Configuration for a benchmark suite."""
    suite_name: str
    benchmarks: List[str]
    data_sizes: List[int] = field(default_factory=lambda: [1000, 5000, 10000])
    iterations: int = 3
    warmup_iterations: int = 1
    timeout_seconds: float = 300.0
    parallel_execution: bool = False
    max_workers: int = 4


class ETLBenchmarkSuite:
    """
    Comprehensive benchmarking suite for ETL pipeline components.
    
    Features:
    - Extract performance benchmarks
    - Transform performance benchmarks  
    - Load performance benchmarks
    - End-to-end pipeline benchmarks
    - Scalability testing
    - Regression detection
    """
    
    def __init__(self, output_dir: str = "benchmarks", enable_profiling: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.enable_profiling = enable_profiling
        self.results = []
        self.baseline_results = {}
        self.profiler = PerformanceProfiler() if enable_profiling else None
        
    def run_benchmark_suite(self, suite: BenchmarkSuite) -> List[BenchmarkResult]:
        """Run a complete benchmark suite."""
        logger.info(f"Starting benchmark suite: {suite.suite_name}")
        suite_results = []
        
        for benchmark_name in suite.benchmarks:
            for data_size in suite.data_sizes:
                benchmark_results = self._run_benchmark_with_iterations(
                    benchmark_name, data_size, suite.iterations, suite.warmup_iterations
                )
                suite_results.extend(benchmark_results)
        
        # Save results
        self._save_suite_results(suite.suite_name, suite_results)
        
        logger.info(f"Completed benchmark suite: {suite.suite_name}",
                   total_benchmarks=len(suite_results))
        
        return suite_results
    
    def _run_benchmark_with_iterations(self, benchmark_name: str, data_size: int, 
                                     iterations: int, warmup_iterations: int) -> List[BenchmarkResult]:
        """Run a benchmark with multiple iterations."""
        results = []
        
        # Warmup iterations
        for i in range(warmup_iterations):
            logger.debug(f"Warmup iteration {i+1}/{warmup_iterations} for {benchmark_name}")
            self._run_single_benchmark(benchmark_name, data_size, warmup=True)
        
        # Actual benchmark iterations
        for i in range(iterations):
            logger.debug(f"Benchmark iteration {i+1}/{iterations} for {benchmark_name}")
            result = self._run_single_benchmark(benchmark_name, data_size)
            if result:
                results.append(result)
        
        return results
    
    def _run_single_benchmark(self, benchmark_name: str, data_size: int, 
                            warmup: bool = False) -> Optional[BenchmarkResult]:
        """Run a single benchmark iteration."""
        benchmark_id = f"{benchmark_name}_{data_size}_{uuid.uuid4().hex[:8]}"
        
        # Generate test data
        test_data = self._generate_test_data(data_size)
        
        # Start profiling if enabled
        if self.profiler and not warmup:
            self.profiler.start_profiling()
        
        start_time = time.time()
        success_count = 0
        error_count = 0
        
        try:
            # Run the specific benchmark
            success_count, error_count = self._execute_benchmark(benchmark_name, test_data)
            
        except Exception as e:
            logger.error(f"Benchmark {benchmark_name} failed", error=str(e))
            error_count = data_size
            
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Stop profiling and get metrics
            performance_metrics = {}
            if self.profiler and not warmup:
                performance_metrics = self.profiler.stop_profiling()
            
            # Calculate metrics
            throughput = success_count / execution_time if execution_time > 0 else 0
            success_rate = success_count / data_size if data_size > 0 else 0
            
            # Extract resource usage from performance metrics
            memory_usage_mb = 0
            cpu_usage_percent = 0
            
            if performance_metrics.get('results'):
                memory_data = performance_metrics['results'].get('memory', {})
                cpu_data = performance_metrics['results'].get('cpu', {})
                
                memory_usage_mb = memory_data.get('peak_memory_mb', 0)
                cpu_usage_percent = cpu_data.get('cpu_usage_percent', 0)
            
            if not warmup:
                # Create benchmark result
                result = BenchmarkResult(
                    benchmark_id=benchmark_id,
                    benchmark_name=benchmark_name,
                    execution_time=execution_time,
                    throughput=throughput,
                    memory_usage_mb=memory_usage_mb,
                    cpu_usage_percent=cpu_usage_percent,
                    success_rate=success_rate,
                    error_count=error_count,
                    timestamp=datetime.now(timezone.utc),
                    metadata={'data_size': data_size},
                    performance_metrics=performance_metrics
                )
                
                self.results.append(result)
                
                # Track in data lineage
                track_lineage(
                    f"benchmark_input_{benchmark_name}",
                    f"benchmark_result_{benchmark_id}",
                    "performance_benchmark",
                    benchmark_result=result.to_dict()
                )
                
                return result
        
        return None    
    def _generate_test_data(self, size: int) -> DataBatch:
        """Generate test data for benchmarking."""
        data = []
        for i in range(size):
            record = {
                'id': i,
                'name': f'Record_{i}',
                'value': np.random.random(),
                'category': np.random.choice(['A', 'B', 'C', 'D']),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'coordinates': [np.random.uniform(-180, 180), np.random.uniform(-90, 90)],
                'metadata': {
                    'source': 'benchmark',
                    'batch_id': str(uuid.uuid4()),
                    'processed': False
                }
            }
            data.append(record)
        return data
    
    def _execute_benchmark(self, benchmark_name: str, test_data: DataBatch) -> Tuple[int, int]:
        """Execute a specific benchmark."""
        success_count = 0
        error_count = 0
        
        if benchmark_name == 'extract_csv':
            success_count, error_count = self._benchmark_extract_csv(test_data)
        elif benchmark_name == 'extract_json':
            success_count, error_count = self._benchmark_extract_json(test_data)
        elif benchmark_name == 'transform_standardise':
            success_count, error_count = self._benchmark_transform_standardise(test_data)
        elif benchmark_name == 'transform_validate':
            success_count, error_count = self._benchmark_transform_validate(test_data)
        elif benchmark_name == 'load_parquet':
            success_count, error_count = self._benchmark_load_parquet(test_data)
        elif benchmark_name == 'load_database':
            success_count, error_count = self._benchmark_load_database(test_data)
        elif benchmark_name == 'full_pipeline':
            success_count, error_count = self._benchmark_full_pipeline(test_data)
        else:
            logger.warning(f"Unknown benchmark: {benchmark_name}")
            error_count = len(test_data)
        
        return success_count, error_count
    
    def _benchmark_extract_csv(self, test_data: DataBatch) -> Tuple[int, int]:
        """Benchmark CSV extraction performance."""
        success_count = 0
        error_count = 0
        
        # Create temporary CSV file
        temp_file = self.output_dir / f"benchmark_temp_{uuid.uuid4().hex}.csv"
        
        try:
            # Write data to CSV
            df = pd.DataFrame(test_data)
            df.to_csv(temp_file, index=False)
            
            # Read data back
            read_df = pd.read_csv(temp_file)
            success_count = len(read_df)
            
        except Exception as e:
            logger.error(f"CSV extraction benchmark failed", error=str(e))
            error_count = len(test_data)
        
        finally:
            # Clean up
            if temp_file.exists():
                temp_file.unlink()
        
        return success_count, error_count
    
    def _benchmark_extract_json(self, test_data: DataBatch) -> Tuple[int, int]:
        """Benchmark JSON extraction performance."""
        success_count = 0
        error_count = 0
        
        # Create temporary JSON file
        temp_file = self.output_dir / f"benchmark_temp_{uuid.uuid4().hex}.json"
        
        try:
            # Write data to JSON
            with open(temp_file, 'w') as f:
                json.dump(test_data, f, default=str)
            
            # Read data back
            with open(temp_file, 'r') as f:
                read_data = json.load(f)
            
            success_count = len(read_data)
            
        except Exception as e:
            logger.error(f"JSON extraction benchmark failed", error=str(e))
            error_count = len(test_data)
        
        finally:
            # Clean up
            if temp_file.exists():
                temp_file.unlink()
        
        return success_count, error_count
    
    def _benchmark_transform_standardise(self, test_data: DataBatch) -> Tuple[int, int]:
        """Benchmark data standardisation performance."""
        success_count = 0
        error_count = 0
        
        try:
            for record in test_data:
                # Simulate data standardisation
                standardised_record = {
                    'id': record.get('id'),
                    'name': str(record.get('name', '')).upper(),
                    'value': float(record.get('value', 0)),
                    'category': str(record.get('category', '')).lower(),
                    'timestamp': record.get('timestamp'),
                    'coordinates': record.get('coordinates', [0, 0])
                }
                
                # Validate required fields
                if standardised_record['id'] is not None and standardised_record['name']:
                    success_count += 1
                else:
                    error_count += 1
                    
        except Exception as e:
            logger.error(f"Transform standardise benchmark failed", error=str(e))
            error_count = len(test_data)
        
        return success_count, error_count
    
    def _benchmark_transform_validate(self, test_data: DataBatch) -> Tuple[int, int]:
        """Benchmark data validation performance."""
        success_count = 0
        error_count = 0
        
        try:
            for record in test_data:
                # Simulate data validation
                is_valid = True
                
                # Check required fields
                if not record.get('id') or not record.get('name'):
                    is_valid = False
                
                # Check data types
                if not isinstance(record.get('value'), (int, float)):
                    is_valid = False
                
                # Check value ranges
                if record.get('value', 0) < 0 or record.get('value', 0) > 1:
                    is_valid = False
                
                # Check coordinate validity
                coords = record.get('coordinates', [])
                if len(coords) != 2 or not all(isinstance(c, (int, float)) for c in coords):
                    is_valid = False
                
                if is_valid:
                    success_count += 1
                else:
                    error_count += 1
                    
        except Exception as e:
            logger.error(f"Transform validate benchmark failed", error=str(e))
            error_count = len(test_data)
        
        return success_count, error_count
    
    def _benchmark_load_parquet(self, test_data: DataBatch) -> Tuple[int, int]:
        """Benchmark Parquet loading performance."""
        success_count = 0
        error_count = 0
        
        temp_file = self.output_dir / f"benchmark_temp_{uuid.uuid4().hex}.parquet"
        
        try:
            # Write to Parquet
            df = pd.DataFrame(test_data)
            df.to_parquet(temp_file, index=False)
            
            # Read back to verify
            read_df = pd.read_parquet(temp_file)
            success_count = len(read_df)
            
        except Exception as e:
            logger.error(f"Parquet loading benchmark failed", error=str(e))
            error_count = len(test_data)
        
        finally:
            # Clean up
            if temp_file.exists():
                temp_file.unlink()
        
        return success_count, error_count
    
    def _benchmark_load_database(self, test_data: DataBatch) -> Tuple[int, int]:
        """Benchmark database loading performance."""
        success_count = 0
        error_count = 0
        
        temp_file = self.output_dir / f"benchmark_temp_{uuid.uuid4().hex}.db"
        
        try:
            # Create in-memory SQLite database for testing
            import sqlite3
            conn = sqlite3.connect(str(temp_file))
            cursor = conn.cursor()
            
            # Create table
            cursor.execute('''
                CREATE TABLE benchmark_data (
                    id INTEGER,
                    name TEXT,
                    value REAL,
                    category TEXT,
                    timestamp TEXT
                )
            ''')
            
            # Insert data
            for record in test_data:
                cursor.execute('''
                    INSERT INTO benchmark_data (id, name, value, category, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    record.get('id'),
                    record.get('name'),
                    record.get('value'),
                    record.get('category'),
                    record.get('timestamp')
                ))
                success_count += 1
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Database loading benchmark failed", error=str(e))
            error_count = len(test_data)
        
        finally:
            # Clean up
            if temp_file.exists():
                temp_file.unlink()
        
        return success_count, error_count
    
    def _benchmark_full_pipeline(self, test_data: DataBatch) -> Tuple[int, int]:
        """Benchmark full ETL pipeline performance."""
        success_count = 0
        error_count = 0
        
        try:
            # Extract (simulate reading from source)
            extracted_data = test_data.copy()
            
            # Transform
            transformed_data = []
            for record in extracted_data:
                try:
                    # Standardise
                    standardised = {
                        'id': record.get('id'),
                        'name': str(record.get('name', '')).upper(),
                        'value': float(record.get('value', 0)),
                        'category': str(record.get('category', '')).lower(),
                        'timestamp': record.get('timestamp'),
                        'coordinates': record.get('coordinates', [0, 0])
                    }
                    
                    # Validate
                    if (standardised['id'] is not None and 
                        standardised['name'] and 
                        0 <= standardised['value'] <= 1):
                        transformed_data.append(standardised)
                        success_count += 1
                    else:
                        error_count += 1
                        
                except Exception:
                    error_count += 1
            
            # Load (simulate writing to destination)
            # This would normally write to the actual destination
            
        except Exception as e:
            logger.error(f"Full pipeline benchmark failed", error=str(e))
            error_count = len(test_data)
        
        return success_count, error_count
    
    def _save_suite_results(self, suite_name: str, results: List[BenchmarkResult]):
        """Save benchmark suite results to file."""
        results_file = self.output_dir / f"{suite_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results_data = {
            'suite_name': suite_name,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'results': [result.to_dict() for result in results],
            'summary': self._generate_results_summary(results)
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Benchmark results saved to {results_file}")
    
    def _generate_results_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate summary statistics for benchmark results."""
        if not results:
            return {}
        
        execution_times = [r.execution_time for r in results]
        throughputs = [r.throughput for r in results]
        memory_usages = [r.memory_usage_mb for r in results]
        success_rates = [r.success_rate for r in results]
        
        return {
            'total_benchmarks': len(results),
            'execution_time': {
                'mean': statistics.mean(execution_times),
                'median': statistics.median(execution_times),
                'min': min(execution_times),
                'max': max(execution_times),
                'stdev': statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            },
            'throughput': {
                'mean': statistics.mean(throughputs),
                'median': statistics.median(throughputs),
                'min': min(throughputs),
                'max': max(throughputs),
                'stdev': statistics.stdev(throughputs) if len(throughputs) > 1 else 0
            },
            'memory_usage': {
                'mean': statistics.mean(memory_usages),
                'median': statistics.median(memory_usages),
                'min': min(memory_usages),
                'max': max(memory_usages),
                'stdev': statistics.stdev(memory_usages) if len(memory_usages) > 1 else 0
            },
            'success_rate': {
                'mean': statistics.mean(success_rates),
                'median': statistics.median(success_rates),
                'min': min(success_rates),
                'max': max(success_rates)
            }
        }class DataProcessingBenchmarks:
    """
    Specialised benchmarks for data processing operations.
    
    Features:
    - DataFrame operations benchmarks
    - Array processing benchmarks
    - String processing benchmarks
    - Numerical computation benchmarks
    - Geographic processing benchmarks
    """
    
    def __init__(self, output_dir: str = "benchmarks/data_processing"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def benchmark_dataframe_operations(self, data_sizes: List[int]) -> List[BenchmarkResult]:
        """Benchmark pandas DataFrame operations."""
        results = []
        
        for size in data_sizes:
            # Generate test data
            df = pd.DataFrame({
                'id': range(size),
                'value': np.random.random(size),
                'category': np.random.choice(['A', 'B', 'C'], size),
                'timestamp': pd.date_range('2023-01-01', periods=size, freq='1H')
            })
            
            # Benchmark various operations
            operations = {
                'groupby_sum': lambda df: df.groupby('category')['value'].sum(),
                'filter_data': lambda df: df[df['value'] > 0.5],
                'merge_self': lambda df: df.merge(df, on='id', suffixes=('_left', '_right')),
                'pivot_table': lambda df: df.pivot_table(values='value', index='category', aggfunc='mean'),
                'sort_values': lambda df: df.sort_values(['category', 'value'])
            }
            
            for op_name, operation in operations.items():
                start_time = time.time()
                try:
                    result = operation(df)
                    execution_time = time.time() - start_time
                    
                    # Create benchmark result
                    benchmark_result = BenchmarkResult(
                        benchmark_id=f"dataframe_{op_name}_{size}",
                        benchmark_name=f"dataframe_{op_name}",
                        execution_time=execution_time,
                        throughput=size / execution_time,
                        memory_usage_mb=df.memory_usage(deep=True).sum() / 1024 / 1024,
                        cpu_usage_percent=0,  # Would need separate monitoring
                        success_rate=1.0,
                        error_count=0,
                        timestamp=datetime.now(timezone.utc),
                        metadata={'data_size': size, 'operation': op_name}
                    )
                    
                    results.append(benchmark_result)
                    
                except Exception as e:
                    logger.error(f"DataFrame operation benchmark failed: {op_name}", error=str(e))
        
        return results
    
    def benchmark_array_processing(self, data_sizes: List[int]) -> List[BenchmarkResult]:
        """Benchmark NumPy array processing operations."""
        results = []
        
        for size in data_sizes:
            # Generate test arrays
            arr1 = np.random.random(size)
            arr2 = np.random.random(size)
            
            # Benchmark various operations
            operations = {
                'vector_addition': lambda a1, a2: a1 + a2,
                'vector_multiplication': lambda a1, a2: a1 * a2,
                'statistical_ops': lambda a1, a2: np.mean(a1) + np.std(a2),
                'matrix_operations': lambda a1, a2: np.dot(a1.reshape(-1, 1), a2.reshape(1, -1)),
                'sorting': lambda a1, a2: np.sort(a1)
            }
            
            for op_name, operation in operations.items():
                start_time = time.time()
                try:
                    result = operation(arr1, arr2)
                    execution_time = time.time() - start_time
                    
                    # Create benchmark result
                    benchmark_result = BenchmarkResult(
                        benchmark_id=f"array_{op_name}_{size}",
                        benchmark_name=f"array_{op_name}",
                        execution_time=execution_time,
                        throughput=size / execution_time,
                        memory_usage_mb=(arr1.nbytes + arr2.nbytes) / 1024 / 1024,
                        cpu_usage_percent=0,
                        success_rate=1.0,
                        error_count=0,
                        timestamp=datetime.now(timezone.utc),
                        metadata={'data_size': size, 'operation': op_name}
                    )
                    
                    results.append(benchmark_result)
                    
                except Exception as e:
                    logger.error(f"Array operation benchmark failed: {op_name}", error=str(e))
        
        return results


class ValidationBenchmarks:
    """
    Benchmarks for data validation operations.
    
    Features:
    - Schema validation benchmarks
    - Business rule validation benchmarks
    - Statistical validation benchmarks
    - Geographic validation benchmarks
    """
    
    def __init__(self, output_dir: str = "benchmarks/validation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def benchmark_schema_validation(self, data_sizes: List[int]) -> List[BenchmarkResult]:
        """Benchmark schema validation performance."""
        results = []
        
        for size in data_sizes:
            # Generate test data with some invalid records
            test_data = []
            for i in range(size):
                # Make some records invalid (10%)
                if i % 10 == 0:
                    record = {'id': None, 'name': '', 'value': 'invalid'}  # Invalid record
                else:
                    record = {'id': i, 'name': f'Record_{i}', 'value': np.random.random()}
                test_data.append(record)
            
            # Schema validation logic
            def validate_schema(data):
                valid_count = 0
                invalid_count = 0
                
                for record in data:
                    is_valid = True
                    
                    # Check required fields
                    if record.get('id') is None or not record.get('name'):
                        is_valid = False
                    
                    # Check data types
                    if not isinstance(record.get('value'), (int, float)):
                        is_valid = False
                    
                    if is_valid:
                        valid_count += 1
                    else:
                        invalid_count += 1
                
                return valid_count, invalid_count
            
            # Run benchmark
            start_time = time.time()
            try:
                valid_count, invalid_count = validate_schema(test_data)
                execution_time = time.time() - start_time
                
                # Create benchmark result
                benchmark_result = BenchmarkResult(
                    benchmark_id=f"schema_validation_{size}",
                    benchmark_name="schema_validation",
                    execution_time=execution_time,
                    throughput=size / execution_time,
                    memory_usage_mb=0,  # Would need memory profiling
                    cpu_usage_percent=0,
                    success_rate=valid_count / size,
                    error_count=invalid_count,
                    timestamp=datetime.now(timezone.utc),
                    metadata={'data_size': size, 'valid_records': valid_count, 'invalid_records': invalid_count}
                )
                
                results.append(benchmark_result)
                
            except Exception as e:
                logger.error(f"Schema validation benchmark failed", error=str(e))
        
        return results


class LoadingBenchmarks:
    """
    Benchmarks for data loading operations.
    
    Features:
    - File format loading benchmarks
    - Database loading benchmarks
    - Batch vs streaming loading benchmarks
    - Compression benchmarks
    """
    
    def __init__(self, output_dir: str = "benchmarks/loading"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def benchmark_file_formats(self, data_sizes: List[int]) -> List[BenchmarkResult]:
        """Benchmark different file format loading performance."""
        results = []
        
        for size in data_sizes:
            # Generate test data
            df = pd.DataFrame({
                'id': range(size),
                'name': [f'Record_{i}' for i in range(size)],
                'value': np.random.random(size),
                'timestamp': pd.date_range('2023-01-01', periods=size, freq='1H')
            })
            
            # Test different formats
            formats = {
                'csv': lambda df, file: df.to_csv(file, index=False),
                'parquet': lambda df, file: df.to_parquet(file, index=False),
                'json': lambda df, file: df.to_json(file, orient='records'),
                'pickle': lambda df, file: df.to_pickle(file)
            }
            
            for format_name, write_func in formats.items():
                temp_file = self.output_dir / f"benchmark_temp_{format_name}_{size}.{format_name}"
                
                start_time = time.time()
                try:
                    write_func(df, temp_file)
                    execution_time = time.time() - start_time
                    
                    # Get file size
                    file_size_mb = temp_file.stat().st_size / 1024 / 1024
                    
                    # Create benchmark result
                    benchmark_result = BenchmarkResult(
                        benchmark_id=f"file_format_{format_name}_{size}",
                        benchmark_name=f"file_format_{format_name}",
                        execution_time=execution_time,
                        throughput=size / execution_time,
                        memory_usage_mb=file_size_mb,
                        cpu_usage_percent=0,
                        success_rate=1.0,
                        error_count=0,
                        timestamp=datetime.now(timezone.utc),
                        metadata={'data_size': size, 'format': format_name, 'file_size_mb': file_size_mb}
                    )
                    
                    results.append(benchmark_result)
                    
                except Exception as e:
                    logger.error(f"File format benchmark failed: {format_name}", error=str(e))
                
                finally:
                    # Clean up
                    if temp_file.exists():
                        temp_file.unlink()
        
        return results


class RegressionTestFramework:
    """
    Framework for detecting performance regressions.
    
    Features:
    - Baseline performance tracking
    - Regression detection algorithms
    - Performance trend analysis
    - Automated alerting
    - Historical comparison
    """
    
    def __init__(self, baseline_dir: str = "benchmarks/baseline", threshold_percent: float = 20.0):
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.threshold_percent = threshold_percent
        self.baselines = {}
        self.load_baselines()
    
    def load_baselines(self):
        """Load baseline performance data."""
        baseline_file = self.baseline_dir / "baselines.json"
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                self.baselines = json.load(f)
    
    def save_baselines(self):
        """Save baseline performance data."""
        baseline_file = self.baseline_dir / "baselines.json"
        with open(baseline_file, 'w') as f:
            json.dump(self.baselines, f, indent=2)
    
    def set_baseline(self, benchmark_name: str, results: List[BenchmarkResult]):
        """Set baseline performance for a benchmark."""
        if not results:
            return
        
        # Calculate baseline metrics
        execution_times = [r.execution_time for r in results]
        throughputs = [r.throughput for r in results]
        memory_usages = [r.memory_usage_mb for r in results]
        
        baseline = {
            'benchmark_name': benchmark_name,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'sample_size': len(results),
            'execution_time': {
                'mean': statistics.mean(execution_times),
                'median': statistics.median(execution_times),
                'stdev': statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            },
            'throughput': {
                'mean': statistics.mean(throughputs),
                'median': statistics.median(throughputs),
                'stdev': statistics.stdev(throughputs) if len(throughputs) > 1 else 0
            },
            'memory_usage': {
                'mean': statistics.mean(memory_usages),
                'median': statistics.median(memory_usages),
                'stdev': statistics.stdev(memory_usages) if len(memory_usages) > 1 else 0
            }
        }
        
        self.baselines[benchmark_name] = baseline
        self.save_baselines()
        
        logger.info(f"Baseline set for {benchmark_name}", baseline=baseline)
    
    def detect_regression(self, benchmark_name: str, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Detect performance regression compared to baseline."""
        if benchmark_name not in self.baselines or not results:
            return {'has_regression': False, 'reason': 'No baseline or results'}
        
        baseline = self.baselines[benchmark_name]
        
        # Calculate current metrics
        execution_times = [r.execution_time for r in results]
        throughputs = [r.throughput for r in results]
        memory_usages = [r.memory_usage_mb for r in results]
        
        current_metrics = {
            'execution_time': statistics.mean(execution_times),
            'throughput': statistics.mean(throughputs),
            'memory_usage': statistics.mean(memory_usages)
        }
        
        # Compare with baseline
        regressions = []
        
        # Check execution time regression (higher is worse)
        exec_time_change = ((current_metrics['execution_time'] - baseline['execution_time']['mean']) / 
                           baseline['execution_time']['mean']) * 100
        if exec_time_change > self.threshold_percent:
            regressions.append({
                'metric': 'execution_time',
                'change_percent': exec_time_change,
                'baseline': baseline['execution_time']['mean'],
                'current': current_metrics['execution_time']
            })
        
        # Check throughput regression (lower is worse)
        throughput_change = ((baseline['throughput']['mean'] - current_metrics['throughput']) / 
                           baseline['throughput']['mean']) * 100
        if throughput_change > self.threshold_percent:
            regressions.append({
                'metric': 'throughput',
                'change_percent': throughput_change,
                'baseline': baseline['throughput']['mean'],
                'current': current_metrics['throughput']
            })
        
        # Check memory regression (higher is worse)
        memory_change = ((current_metrics['memory_usage'] - baseline['memory_usage']['mean']) / 
                        baseline['memory_usage']['mean']) * 100
        if memory_change > self.threshold_percent:
            regressions.append({
                'metric': 'memory_usage',
                'change_percent': memory_change,
                'baseline': baseline['memory_usage']['mean'],
                'current': current_metrics['memory_usage']
            })
        
        regression_result = {
            'has_regression': len(regressions) > 0,
            'regressions': regressions,
            'benchmark_name': benchmark_name,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'threshold_percent': self.threshold_percent
        }
        
        if regression_result['has_regression']:
            logger.warning(f"Performance regression detected for {benchmark_name}",
                         regressions=regressions)
        
        return regression_result