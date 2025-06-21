"""Performance tests for export pipeline speed and efficiency.

Tests export performance across different data sizes, formats, and configurations
to ensure the pipeline meets production performance requirements.

British English spelling is used throughout (optimise, standardise, etc.).
"""

import pytest
import time
import tempfile
import psutil
import os
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np
from memory_profiler import profile

from src.pipelines.export_pipeline import ExportPipeline
from src.loaders.production_loader import ProductionLoader
from src.loaders.format_exporters import ParquetExporter, CSVExporter, JSONExporter
from src.utils.compression_utils import CompressionAnalyzer, SizeCalculator


class TestExportPerformance:
    """Performance tests for export pipeline."""
    
    @pytest.fixture(scope="class")
    def performance_config(self):
        """Configuration for performance testing."""
        return {
            'target_throughput_mb_per_second': 50,
            'max_memory_multiplier': 3.0,
            'timeout_minutes': 10,
            'compression_ratio_targets': {
                'gzip': 0.5,
                'brotli': 0.4,
                'lz4': 0.6
            }
        }
    
    @pytest.fixture
    def small_dataset(self):
        """Small dataset (< 1MB) for baseline performance."""
        return self._create_test_dataset(1000)  # ~1000 records
    
    @pytest.fixture
    def medium_dataset(self):
        """Medium dataset (~10MB) for standard performance testing."""
        return self._create_test_dataset(50000)  # ~50K records
    
    @pytest.fixture
    def large_dataset(self):
        """Large dataset (~100MB) for stress testing."""
        return self._create_test_dataset(500000)  # ~500K records
    
    @pytest.fixture
    def export_pipeline(self):
        """Export pipeline instance for performance testing."""
        config = {
            'export_pipeline': {
                'performance_mode': True,
                'parallel_processing': True,
                'memory_optimised': True
            }
        }
        return ExportPipeline(config)
    
    @pytest.fixture
    def temp_output_dir(self):
        """Temporary output directory for test exports."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def _create_test_dataset(self, n_records: int) -> pd.DataFrame:
        """Create test dataset with specified number of records."""
        np.random.seed(42)  # Consistent test data
        
        # Australian states with realistic distribution
        state_codes = np.random.choice(
            ['NSW', 'VIC', 'QLD', 'WA', 'SA', 'TAS', 'ACT', 'NT'],
            size=n_records,
            p=[0.32, 0.25, 0.20, 0.10, 0.07, 0.02, 0.02, 0.02]
        )
        
        # Generate realistic data patterns
        data = pd.DataFrame({
            'record_id': range(n_records),
            'sa2_code': [f"{np.random.randint(10000, 99999)}" for _ in range(n_records)],
            'state_code': state_codes,
            'postcode': [f"{np.random.randint(1000, 9999):04d}" for _ in range(n_records)],
            'latitude': np.random.uniform(-43.6, -10.7, n_records),
            'longitude': np.random.uniform(113.3, 153.6, n_records),
            'population': np.random.randint(100, 50000, n_records),
            'health_score': np.random.normal(75, 15, n_records),
            'obesity_rate': np.random.beta(2, 5, n_records) * 40,
            'diabetes_rate': np.random.gamma(2, 2, n_records),
            'income_median': np.random.normal(65000, 20000, n_records),
            'education_score': np.random.uniform(60, 95, n_records),
            'remoteness_category': np.random.choice(
                ['Major Cities', 'Inner Regional', 'Outer Regional', 'Remote', 'Very Remote'],
                n_records,
                p=[0.7, 0.18, 0.09, 0.02, 0.01]
            ),
            'data_quality_flag': np.random.choice(['High', 'Medium', 'Low'], n_records, p=[0.8, 0.15, 0.05]),
            'last_updated': pd.date_range('2023-01-01', periods=n_records, freq='H')[:n_records],
            'notes': [f"Record {i} with extended descriptive text to increase data size" for i in range(n_records)]
        })
        
        return data
    
    def _measure_performance(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Measure performance metrics for a function call."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        start_cpu = process.cpu_percent()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_cpu = process.cpu_percent()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'result': result,
            'execution_time_seconds': end_time - start_time,
            'memory_usage_mb': final_memory - initial_memory,
            'peak_memory_mb': final_memory,
            'cpu_usage_percent': end_cpu - start_cpu
        }
    
    def test_export_throughput_small_dataset(self, export_pipeline, small_dataset, temp_output_dir, performance_config):
        """Test export throughput with small dataset."""
        metrics = self._measure_performance(
            export_pipeline.export_data,
            data=small_dataset,
            output_path=temp_output_dir / 'small',
            formats=['parquet', 'csv', 'json']
        )
        
        # Calculate throughput
        data_size_mb = small_dataset.memory_usage(deep=True).sum() / 1024 / 1024
        throughput = data_size_mb / metrics['execution_time_seconds']
        
        # Verify completion
        assert metrics['result']['pipeline_status'] == 'completed'
        
        # Performance assertions
        assert throughput > performance_config['target_throughput_mb_per_second'] / 2  # Allow lower for small files
        assert metrics['execution_time_seconds'] < 30  # Should complete within 30 seconds
        
        print(f"Small dataset throughput: {throughput:.2f} MB/s")
    
    def test_export_throughput_medium_dataset(self, export_pipeline, medium_dataset, temp_output_dir, performance_config):
        """Test export throughput with medium dataset."""
        metrics = self._measure_performance(
            export_pipeline.export_data,
            data=medium_dataset,
            output_path=temp_output_dir / 'medium',
            formats=['parquet', 'csv'],
            export_options={'compress': True, 'partition': True}
        )
        
        # Calculate throughput
        data_size_mb = medium_dataset.memory_usage(deep=True).sum() / 1024 / 1024
        throughput = data_size_mb / metrics['execution_time_seconds']
        
        # Verify completion
        assert metrics['result']['pipeline_status'] == 'completed'
        
        # Performance assertions
        target_throughput = performance_config['target_throughput_mb_per_second']
        assert throughput > target_throughput * 0.8  # Allow 20% variance
        assert metrics['execution_time_seconds'] < 120  # Should complete within 2 minutes
        
        print(f"Medium dataset throughput: {throughput:.2f} MB/s (target: {target_throughput} MB/s)")
    
    def test_export_throughput_large_dataset(self, export_pipeline, large_dataset, temp_output_dir, performance_config):
        """Test export throughput with large dataset."""
        metrics = self._measure_performance(
            export_pipeline.export_data,
            data=large_dataset,
            output_path=temp_output_dir / 'large',
            formats=['parquet'],  # Only Parquet for large datasets
            export_options={
                'compress': True,
                'partition': True,
                'parallel_processing': True
            }
        )
        
        # Calculate throughput
        data_size_mb = large_dataset.memory_usage(deep=True).sum() / 1024 / 1024
        throughput = data_size_mb / metrics['execution_time_seconds']
        
        # Verify completion
        assert metrics['result']['pipeline_status'] == 'completed'
        
        # Performance assertions for large datasets
        target_throughput = performance_config['target_throughput_mb_per_second']
        assert throughput > target_throughput * 0.6  # Allow more variance for large datasets
        assert metrics['execution_time_seconds'] < 600  # Should complete within 10 minutes
        
        print(f"Large dataset throughput: {throughput:.2f} MB/s")
        print(f"Large dataset size: {data_size_mb:.2f} MB")
        print(f"Large dataset export time: {metrics['execution_time_seconds']:.2f} seconds")
    
    def test_memory_efficiency(self, export_pipeline, medium_dataset, temp_output_dir, performance_config):
        """Test memory efficiency during export."""
        data_size_mb = medium_dataset.memory_usage(deep=True).sum() / 1024 / 1024
        
        metrics = self._measure_performance(
            export_pipeline.export_data,
            data=medium_dataset,
            output_path=temp_output_dir / 'memory_test',
            formats=['parquet', 'csv'],
            export_options={'memory_efficient': True}
        )
        
        # Memory efficiency checks
        memory_multiplier = metrics['memory_usage_mb'] / data_size_mb
        max_multiplier = performance_config['max_memory_multiplier']
        
        assert memory_multiplier < max_multiplier, (
            f"Memory usage {memory_multiplier:.2f}x exceeds limit {max_multiplier}x"
        )
        
        print(f"Memory efficiency: {memory_multiplier:.2f}x data size")
        print(f"Peak memory: {metrics['peak_memory_mb']:.2f} MB")
    
    @pytest.mark.parametrize('format_type', ['parquet', 'csv', 'json'])
    def test_format_specific_performance(self, export_pipeline, medium_dataset, temp_output_dir, format_type):
        """Test performance for specific export formats."""
        metrics = self._measure_performance(
            export_pipeline.export_data,
            data=medium_dataset,
            output_path=temp_output_dir / f'format_{format_type}',
            formats=[format_type],
            export_options={'compress': True}
        )
        
        # Calculate format-specific metrics
        data_size_mb = medium_dataset.memory_usage(deep=True).sum() / 1024 / 1024
        throughput = data_size_mb / metrics['execution_time_seconds']
        
        # Format-specific performance expectations
        format_expectations = {
            'parquet': {'min_throughput': 40, 'max_time': 60},
            'csv': {'min_throughput': 30, 'max_time': 90},
            'json': {'min_throughput': 20, 'max_time': 120}
        }
        
        expectations = format_expectations[format_type]
        
        assert throughput > expectations['min_throughput'], (
            f"{format_type} throughput {throughput:.2f} MB/s below minimum {expectations['min_throughput']}"
        )
        
        assert metrics['execution_time_seconds'] < expectations['max_time'], (
            f"{format_type} export time {metrics['execution_time_seconds']:.2f}s exceeds maximum {expectations['max_time']}s"
        )
        
        print(f"{format_type.upper()} performance: {throughput:.2f} MB/s in {metrics['execution_time_seconds']:.2f}s")
    
    @pytest.mark.parametrize('compression_type', ['gzip', 'brotli', 'lz4'])
    def test_compression_performance(self, medium_dataset, temp_output_dir, compression_type, performance_config):
        """Test compression algorithm performance."""
        from src.loaders.production_loader import CompressionManager
        
        compression_manager = CompressionManager()
        
        # Export CSV first
        csv_exporter = CSVExporter()
        csv_path = temp_output_dir / 'test_data.csv'
        csv_exporter.export(medium_dataset, csv_path)
        
        # Measure compression performance
        metrics = self._measure_performance(
            compression_manager.compress_file,
            input_path=csv_path,
            algorithm=compression_type,
            level=6
        )
        
        # Calculate compression metrics
        original_size = csv_path.stat().st_size
        compressed_size = metrics['result'].stat().st_size
        compression_ratio = compressed_size / original_size
        
        # Performance assertions
        target_ratio = performance_config['compression_ratio_targets'][compression_type]
        
        assert compression_ratio < target_ratio, (
            f"{compression_type} compression ratio {compression_ratio:.2f} exceeds target {target_ratio}"
        )
        
        assert metrics['execution_time_seconds'] < 60, (
            f"{compression_type} compression took {metrics['execution_time_seconds']:.2f}s, expected < 60s"
        )
        
        print(f"{compression_type.upper()} compression: {compression_ratio:.2f} ratio in {metrics['execution_time_seconds']:.2f}s")
    
    def test_partitioning_performance(self, export_pipeline, large_dataset, temp_output_dir):
        """Test performance impact of partitioning strategies."""
        # Test without partitioning
        metrics_no_partition = self._measure_performance(
            export_pipeline.export_data,
            data=large_dataset,
            output_path=temp_output_dir / 'no_partition',
            formats=['parquet'],
            export_options={'partition': False}
        )
        
        # Test with state-based partitioning
        metrics_with_partition = self._measure_performance(
            export_pipeline.export_data,
            data=large_dataset,
            output_path=temp_output_dir / 'with_partition',
            formats=['parquet'],
            export_options={'partition': True, 'partition_strategy': 'state_based'}
        )
        
        # Compare performance
        no_partition_time = metrics_no_partition['execution_time_seconds']
        with_partition_time = metrics_with_partition['execution_time_seconds']
        
        # Partitioning might be slower but should not be dramatically worse
        time_ratio = with_partition_time / no_partition_time
        assert time_ratio < 2.0, f"Partitioning slowed export by {time_ratio:.2f}x"
        
        print(f"No partitioning: {no_partition_time:.2f}s")
        print(f"With partitioning: {with_partition_time:.2f}s")
        print(f"Time ratio: {time_ratio:.2f}x")
    
    def test_parallel_processing_performance(self, export_pipeline, medium_dataset, temp_output_dir):
        """Test performance benefit of parallel processing."""
        # Sequential processing
        metrics_sequential = self._measure_performance(
            export_pipeline.export_data,
            data=medium_dataset,
            output_path=temp_output_dir / 'sequential',
            formats=['parquet', 'csv', 'json'],
            export_options={'parallel_processing': False}
        )
        
        # Parallel processing
        metrics_parallel = self._measure_performance(
            export_pipeline.export_data,
            data=medium_dataset,
            output_path=temp_output_dir / 'parallel',
            formats=['parquet', 'csv', 'json'],
            export_options={'parallel_processing': True}
        )
        
        # Parallel should be faster or at least not much slower
        sequential_time = metrics_sequential['execution_time_seconds']
        parallel_time = metrics_parallel['execution_time_seconds']
        
        speedup_ratio = sequential_time / parallel_time
        
        # Expect at least some benefit or no significant slowdown
        assert speedup_ratio > 0.8, f"Parallel processing provided {speedup_ratio:.2f}x speedup (expected > 0.8x)"
        
        print(f"Sequential: {sequential_time:.2f}s")
        print(f"Parallel: {parallel_time:.2f}s")
        print(f"Speedup: {speedup_ratio:.2f}x")
    
    def test_streaming_performance(self, export_pipeline, large_dataset, temp_output_dir):
        """Test streaming export performance for large datasets."""
        metrics = self._measure_performance(
            export_pipeline.export_data,
            data=large_dataset,
            output_path=temp_output_dir / 'streaming',
            formats=['csv'],
            export_options={
                'streaming': True,
                'chunk_size': 10000,
                'memory_efficient': True
            }
        )
        
        # Streaming should complete successfully
        assert metrics['result']['pipeline_status'] == 'completed'
        
        # Memory usage should be controlled
        data_size_mb = large_dataset.memory_usage(deep=True).sum() / 1024 / 1024
        memory_multiplier = metrics['memory_usage_mb'] / data_size_mb
        
        # Streaming should use significantly less memory
        assert memory_multiplier < 2.0, f"Streaming used {memory_multiplier:.2f}x memory (expected < 2.0x)"
        
        print(f"Streaming memory efficiency: {memory_multiplier:.2f}x data size")
    
    def test_size_estimation_accuracy(self, medium_dataset):
        """Test accuracy of size estimation vs actual export sizes."""
        size_calculator = SizeCalculator()
        
        # Test different formats
        formats_to_test = ['parquet', 'csv', 'json']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            for format_type in formats_to_test:
                # Get size estimate
                estimate = size_calculator.estimate_export_size(
                    data=medium_dataset,
                    target_format=format_type,
                    compression=None
                )
                
                # Perform actual export
                if format_type == 'parquet':
                    exporter = ParquetExporter()
                elif format_type == 'csv':
                    exporter = CSVExporter()
                elif format_type == 'json':
                    exporter = JSONExporter()
                
                export_path = temp_path / f'test.{format_type}'
                export_info = exporter.export(medium_dataset, export_path)
                
                # Compare estimate vs actual
                estimated_size = estimate['estimated_size_bytes']
                actual_size = export_info['file_size_bytes']
                
                accuracy_ratio = actual_size / estimated_size
                
                # Estimate should be within 50% of actual size
                assert 0.5 < accuracy_ratio < 2.0, (
                    f"{format_type} size estimate off by {accuracy_ratio:.2f}x (estimated: {estimated_size}, actual: {actual_size})"
                )
                
                print(f"{format_type.upper()} size estimation accuracy: {accuracy_ratio:.2f}x")
    
    def test_concurrent_exports(self, export_pipeline, medium_dataset, temp_output_dir):
        """Test performance under concurrent export requests."""
        import threading
        import concurrent.futures
        
        def export_task(task_id):
            """Individual export task."""
            return export_pipeline.export_data(
                data=medium_dataset,
                output_path=temp_output_dir / f'concurrent_{task_id}',
                formats=['parquet'],
                export_options={'compress': True}
            )
        
        # Run multiple concurrent exports
        num_concurrent = 3
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(export_task, i) for i in range(num_concurrent)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All exports should complete successfully
        for result in results:
            assert result['pipeline_status'] == 'completed'
        
        # Concurrent execution should not take much longer than sequential
        # (allowing for some overhead)
        expected_sequential_time = 60 * num_concurrent  # Rough estimate
        assert total_time < expected_sequential_time, (
            f"Concurrent exports took {total_time:.2f}s, expected < {expected_sequential_time}s"
        )
        
        print(f"Concurrent exports ({num_concurrent} tasks): {total_time:.2f}s")
    
    def test_performance_regression(self, export_pipeline, medium_dataset, temp_output_dir):
        """Test for performance regression by running baseline performance test."""
        # This test establishes baseline performance metrics
        # In CI/CD, compare against historical baselines
        
        baseline_metrics = self._measure_performance(
            export_pipeline.export_data,
            data=medium_dataset,
            output_path=temp_output_dir / 'baseline',
            formats=['parquet', 'csv'],
            export_options={'compress': True, 'partition': False}
        )
        
        # Calculate key metrics
        data_size_mb = medium_dataset.memory_usage(deep=True).sum() / 1024 / 1024
        throughput = data_size_mb / baseline_metrics['execution_time_seconds']
        memory_efficiency = baseline_metrics['memory_usage_mb'] / data_size_mb
        
        # Store metrics for regression testing
        performance_report = {
            'timestamp': time.time(),
            'data_size_mb': data_size_mb,
            'execution_time_seconds': baseline_metrics['execution_time_seconds'],
            'throughput_mb_per_second': throughput,
            'memory_efficiency_multiplier': memory_efficiency,
            'peak_memory_mb': baseline_metrics['peak_memory_mb']
        }
        
        # Save performance report
        report_path = temp_output_dir / 'performance_report.json'
        import json
        with open(report_path, 'w') as f:
            json.dump(performance_report, f, indent=2)
        
        # Basic performance assertions
        assert throughput > 20, f"Baseline throughput {throughput:.2f} MB/s below minimum 20 MB/s"
        assert memory_efficiency < 5, f"Memory efficiency {memory_efficiency:.2f}x exceeds limit 5x"
        
        print(f"Baseline performance report saved to {report_path}")
        print(f"Throughput: {throughput:.2f} MB/s")
        print(f"Memory efficiency: {memory_efficiency:.2f}x")
