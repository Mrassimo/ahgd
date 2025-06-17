"""
Comprehensive performance benchmark tests for Australian Health Analytics platform.

Tests performance targets across all major components:
- Data processing performance (SEIFA, Health, Boundary processors)
- Storage optimization performance (Parquet compression, memory usage)
- Analysis module performance (Risk calculator, access scorer)
- Memory efficiency and resource utilisation
- Scalability with realistic Australian health data volumes

Validates enterprise-grade performance requirements.
"""

import pytest
import polars as pl
import numpy as np
import time
import psutil
import os
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import shutil
import gc

from src.data_processing.seifa_processor import SEIFAProcessor
from src.data_processing.health_processor import HealthDataProcessor
from src.data_processing.simple_boundary_processor import SimpleBoundaryProcessor
from src.data_processing.storage.parquet_storage_manager import ParquetStorageManager
from src.data_processing.storage.memory_optimizer import MemoryOptimizer
from src.analysis.risk.health_risk_calculator import HealthRiskCalculator


class TestPerformanceBenchmarks:
    """Performance benchmark tests for all major components."""
    
    def test_seifa_processing_performance(self, mock_excel_seifa_file, mock_data_paths, performance_benchmarks):
        """Test SEIFA processor meets performance benchmarks."""
        processor = SEIFAProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        
        # Create realistic-sized test file (2000+ SA2 areas)
        excel_file = mock_excel_seifa_file(num_areas=2000)
        expected_path = processor.raw_dir / "SEIFA_2021_SA2_Indexes.xlsx"
        shutil.copy(excel_file, expected_path)
        
        # Benchmark processing
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        result_df = processor.process_seifa_file()
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        processing_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        # Performance assertions
        assert processing_time < 30.0, f"SEIFA processing took {processing_time:.1f}s, expected <30s"
        assert memory_usage < 500, f"SEIFA processing used {memory_usage:.1f}MB, expected <500MB"
        
        # Verify data integrity
        assert len(result_df) > 1800, "Should process majority of SA2 areas"
        assert isinstance(result_df, pl.DataFrame)
        
        # Calculate throughput
        throughput = len(result_df) / processing_time
        assert throughput > 50, f"Processing throughput {throughput:.1f} SA2/s, expected >50 SA2/s"
    
    def test_health_data_processing_performance(self, mock_health_data, mock_data_paths, performance_benchmarks):
        """Test health data processor meets performance benchmarks."""
        processor = HealthDataProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        
        # Create realistic health dataset (50K+ records)
        large_health_df = mock_health_data(num_records=50000, num_sa2_areas=1000)
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Process through validation and aggregation pipeline
        validated_df = processor._validate_health_data(large_health_df)
        aggregated_df = processor._aggregate_by_sa2(validated_df)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        processing_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        # Performance assertions
        assert processing_time < 45.0, f"Health processing took {processing_time:.1f}s, expected <45s"
        assert memory_usage < 800, f"Health processing used {memory_usage:.1f}MB, expected <800MB"
        
        # Verify processing results
        assert len(aggregated_df) > 0
        assert len(aggregated_df) <= 1000  # Should aggregate to SA2 level
        
        # Calculate throughput
        throughput = len(large_health_df) / processing_time
        assert throughput > 1000, f"Processing throughput {throughput:.1f} records/s, expected >1000 records/s"
    
    def test_boundary_processing_performance(self, mock_boundary_data, mock_data_paths, performance_benchmarks):
        """Test boundary processor meets performance benchmarks."""
        processor = SimpleBoundaryProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        
        # Create realistic boundary dataset (2000+ SA2 areas)
        large_boundary_df = mock_boundary_data(num_areas=2000)
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Process through validation and enhancement pipeline
        validated_df = processor._validate_boundary_data(large_boundary_df)
        density_df = processor._calculate_population_density(validated_df)
        classified_df = processor._classify_remoteness(density_df)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        processing_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        # Performance assertions
        assert processing_time < 20.0, f"Boundary processing took {processing_time:.1f}s, expected <20s"
        assert memory_usage < 400, f"Boundary processing used {memory_usage:.1f}MB, expected <400MB"
        
        # Verify processing results
        assert len(classified_df) > 1500  # Should retain most areas after validation
        
        # Calculate throughput
        throughput = len(large_boundary_df) / processing_time
        assert throughput > 100, f"Processing throughput {throughput:.1f} areas/s, expected >100 areas/s"
    
    def test_parquet_compression_performance(self, mock_seifa_data, mock_data_paths, performance_benchmarks):
        """Test Parquet storage meets compression and I/O benchmarks."""
        manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        
        # Create large test dataset
        large_df = mock_seifa_data(num_areas=5000)
        
        # Save as CSV for size comparison
        csv_path = mock_data_paths["parquet_dir"] / "benchmark.csv"
        csv_start = time.time()
        large_df.write_csv(csv_path)
        csv_write_time = time.time() - csv_start
        csv_size = csv_path.stat().st_size
        
        # Save as optimized Parquet
        parquet_path = mock_data_paths["parquet_dir"] / "benchmark.parquet"
        parquet_start = time.time()
        manager.save_optimized_parquet(large_df, parquet_path, data_type="seifa")
        parquet_write_time = time.time() - parquet_start
        parquet_size = parquet_path.stat().st_size
        
        # Test read performance
        read_start = time.time()
        loaded_df = pl.read_parquet(parquet_path)
        read_time = time.time() - read_start
        
        # Calculate metrics
        compression_ratio = parquet_size / csv_size
        file_size_mb = parquet_size / (1024 * 1024)
        write_speed = parquet_write_time / file_size_mb
        read_speed = read_time / file_size_mb
        
        # Performance assertions from benchmarks
        benchmarks = performance_benchmarks["parquet_compression"]
        
        assert compression_ratio <= benchmarks["min_compression_ratio"], \
            f"Compression ratio {compression_ratio:.2f} exceeds target {benchmarks['min_compression_ratio']}"
        
        assert write_speed <= benchmarks["max_write_time_per_mb"], \
            f"Write speed {write_speed:.3f}s/MB exceeds target {benchmarks['max_write_time_per_mb']}s/MB"
        
        assert read_speed <= benchmarks["max_read_time_per_mb"], \
            f"Read speed {read_speed:.3f}s/MB exceeds target {benchmarks['max_read_time_per_mb']}s/MB"
        
        # Verify data integrity
        assert len(loaded_df) == len(large_df)
        assert loaded_df.shape == large_df.shape
    
    def test_memory_optimization_performance(self, mock_health_data, performance_benchmarks, memory_profiler):
        """Test memory optimizer meets efficiency benchmarks."""
        optimizer = MemoryOptimizer()
        
        # Create memory-intensive dataset
        large_df = mock_health_data(num_records=20000, num_sa2_areas=800)
        
        # Force inefficient data types
        large_df = large_df.with_columns([
            pl.col("prescription_count").cast(pl.Int64),
            pl.col("chronic_medication").cast(pl.Int64),
            pl.col("sa2_code").cast(pl.Utf8),
            pl.col("state").cast(pl.Utf8)
        ])
        
        memory_profiler.start()
        initial_memory = large_df.estimated_size("mb")
        
        # Optimize memory usage
        start_time = time.time()
        optimized_df = optimizer.optimize_data_types(large_df, data_category="health")
        optimization_time = time.time() - start_time
        
        final_memory = optimized_df.estimated_size("mb")
        memory_reduction = (initial_memory - final_memory) / initial_memory
        
        # Performance assertions from benchmarks
        benchmarks = performance_benchmarks["memory_optimization"]
        
        assert memory_reduction >= benchmarks["min_memory_reduction"], \
            f"Memory reduction {memory_reduction:.2f} below target {benchmarks['min_memory_reduction']}"
        
        assert optimization_time < 15.0, f"Memory optimization took {optimization_time:.1f}s, expected <15s"
        
        # Verify data integrity
        assert len(optimized_df) == len(large_df)
        assert optimized_df.shape == large_df.shape
        
        # Verify specific optimizations
        assert optimized_df["sa2_code"].dtype == pl.Categorical
        assert optimized_df["state"].dtype == pl.Categorical
    
    def test_risk_calculator_performance(self, integration_test_data, performance_benchmarks):
        """Test health risk calculator meets performance benchmarks."""
        calculator = HealthRiskCalculator()
        
        # Create large integrated dataset
        large_data = integration_test_data(num_sa2_areas=2000, num_health_records=10000)
        
        seifa_df = large_data["seifa"]
        health_df = large_data["health"]
        boundary_df = large_data["boundaries"]
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Calculate risk components
        seifa_risk = calculator._calculate_seifa_risk_score(seifa_df)
        
        # Aggregate health data
        health_agg = health_df.group_by("sa2_code").agg([
            pl.col("prescription_count").sum().alias("total_prescriptions"),
            pl.col("chronic_medication").mean().alias("chronic_rate"),
            pl.col("cost_government").sum().alias("total_cost")
        ])
        
        health_risk = calculator._calculate_health_utilisation_risk(health_agg)
        geographic_risk = calculator._calculate_geographic_accessibility_risk(boundary_df)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        processing_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        # Performance assertions
        assert processing_time < 30.0, f"Risk calculation took {processing_time:.1f}s, expected <30s"
        assert memory_usage < 600, f"Risk calculation used {memory_usage:.1f}MB, expected <600MB"
        
        # Verify calculation results
        assert len(seifa_risk) > 1500
        assert len(health_risk) > 0
        assert len(geographic_risk) > 1500
        
        # Calculate throughput
        throughput = len(seifa_df) / processing_time
        assert throughput > 60, f"Risk calculation throughput {throughput:.1f} areas/s, expected >60 areas/s"
    
    def test_concurrent_processing_performance(self, mock_seifa_data, mock_health_data, mock_data_paths):
        """Test concurrent processing performance and scalability."""
        import concurrent.futures
        import threading
        
        def process_seifa_batch(batch_id, num_areas):
            """Process SEIFA data batch."""
            processor = SEIFAProcessor(data_dir=mock_data_paths["raw_dir"].parent)
            df = mock_seifa_data(num_areas=num_areas)
            return processor._validate_seifa_data(df)
        
        def process_health_batch(batch_id, num_records):
            """Process health data batch."""
            processor = HealthDataProcessor(data_dir=mock_data_paths["raw_dir"].parent)
            df = mock_health_data(num_records=num_records, num_sa2_areas=50)
            return processor._validate_health_data(df)
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Run concurrent processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            seifa_futures = [
                executor.submit(process_seifa_batch, i, 500) 
                for i in range(4)
            ]
            health_futures = [
                executor.submit(process_health_batch, i, 1000) 
                for i in range(4)
            ]
            
            # Collect results
            seifa_results = [f.result() for f in seifa_futures]
            health_results = [f.result() for f in health_futures]
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        concurrent_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        # Performance assertions
        assert concurrent_time < 20.0, f"Concurrent processing took {concurrent_time:.1f}s, expected <20s"
        assert memory_usage < 1000, f"Concurrent processing used {memory_usage:.1f}MB, expected <1GB"
        
        # Verify all operations succeeded
        assert len(seifa_results) == 4
        assert len(health_results) == 4
        
        for result in seifa_results + health_results:
            assert isinstance(result, pl.DataFrame)
            assert len(result) > 0
    
    def test_large_dataset_scalability(self, mock_health_data, mock_data_paths, performance_benchmarks):
        """Test system scalability with large realistic datasets."""
        # Simulate realistic Australian health data volumes
        # Total PBS records: ~500K annually, SEIFA: 2.4K SA2 areas
        
        optimizer = MemoryOptimizer(chunk_size=5000)
        storage_manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        
        # Create very large health dataset
        very_large_df = mock_health_data(num_records=100000, num_sa2_areas=2000)
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Process in chunks for memory efficiency
        processed_chunks = []
        for chunk_df in optimizer.process_in_chunks(very_large_df):
            optimized_chunk = optimizer.optimize_data_types(chunk_df, data_category="health")
            processed_chunks.append(optimized_chunk)
        
        # Combine and save
        final_df = pl.concat(processed_chunks)
        output_path = mock_data_paths["parquet_dir"] / "large_dataset.parquet"
        storage_manager.save_optimized_parquet(final_df, output_path, data_type="health")
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        total_time = end_time - start_time
        peak_memory = end_memory - start_memory
        
        # Scalability assertions
        assert total_time < 120.0, f"Large dataset processing took {total_time:.1f}s, expected <120s"
        assert peak_memory < 1500, f"Peak memory usage {peak_memory:.1f}MB, expected <1.5GB"
        
        # Verify processing integrity
        assert len(final_df) == len(very_large_df)
        assert output_path.exists()
        
        # Test read performance of large file
        read_start = time.time()
        loaded_df = pl.read_parquet(output_path)
        read_time = time.time() - read_start
        
        assert read_time < 5.0, f"Large file read took {read_time:.1f}s, expected <5s"
        assert len(loaded_df) == len(very_large_df)
    
    def test_memory_leak_detection(self, mock_seifa_data, memory_profiler):
        """Test for memory leaks during repeated operations."""
        memory_profiler.start()
        baseline_memory = memory_profiler.get_current_usage()
        
        # Perform repeated operations that should not leak memory
        for iteration in range(20):
            # Create and process data
            df = mock_seifa_data(num_areas=200)
            
            # Simulate various operations
            processed_df = df.with_columns([
                pl.col("irsd_decile").cast(pl.Int8),
                pl.col("sa2_code_2021").cast(pl.Categorical)
            ])
            
            # Force aggregation
            summary = processed_df.group_by("state_name").agg([
                pl.col("irsd_decile").mean(),
                pl.col("usual_resident_population").sum()
            ])
            
            # Clean up references
            del df, processed_df, summary
            
            # Periodic garbage collection
            if iteration % 5 == 0:
                gc.collect()
                
                # Check for memory growth
                current_memory = memory_profiler.get_current_usage()
                memory_growth = current_memory - baseline_memory
                
                # Memory growth should be minimal
                assert memory_growth < 200, f"Potential memory leak: {memory_growth}MB growth at iteration {iteration}"
        
        # Final memory check
        final_memory = memory_profiler.get_current_usage()
        total_growth = final_memory - baseline_memory
        
        assert total_growth < 100, f"Total memory growth {total_growth}MB suggests memory leak"
    
    def test_io_performance_benchmarks(self, mock_seifa_data, mock_data_paths):
        """Test I/O performance across different file formats."""
        test_df = mock_seifa_data(num_areas=3000)
        
        file_formats = {
            "csv": lambda df, path: df.write_csv(path),
            "parquet": lambda df, path: df.write_parquet(path),
            "json": lambda df, path: df.write_json(path)
        }
        
        read_functions = {
            "csv": pl.read_csv,
            "parquet": pl.read_parquet,
            "json": pl.read_json
        }
        
        performance_results = {}
        
        for format_name, write_func in file_formats.items():
            file_path = mock_data_paths["parquet_dir"] / f"benchmark.{format_name}"
            
            # Test write performance
            write_start = time.time()
            write_func(test_df, file_path)
            write_time = time.time() - write_start
            
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            
            # Test read performance
            read_start = time.time()
            loaded_df = read_functions[format_name](file_path)
            read_time = time.time() - read_start
            
            performance_results[format_name] = {
                "write_time": write_time,
                "read_time": read_time,
                "file_size_mb": file_size,
                "write_speed_mb_s": file_size / write_time,
                "read_speed_mb_s": file_size / read_time
            }
            
            # Verify data integrity
            assert len(loaded_df) == len(test_df)
        
        # Parquet should be most efficient for our use case
        parquet_perf = performance_results["parquet"]
        csv_perf = performance_results["csv"]
        
        # Parquet should be smaller and faster to read
        assert parquet_perf["file_size_mb"] < csv_perf["file_size_mb"]
        assert parquet_perf["read_speed_mb_s"] > csv_perf["read_speed_mb_s"]
    
    def test_query_performance_benchmarks(self, mock_health_data, mock_data_paths):
        """Test query performance on large datasets."""
        storage_manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        
        # Create and save large dataset
        large_df = mock_health_data(num_records=50000, num_sa2_areas=1000)
        parquet_path = mock_data_paths["parquet_dir"] / "query_benchmark.parquet"
        storage_manager.save_optimized_parquet(large_df, parquet_path, data_type="health")
        
        # Test various query patterns
        query_tests = [
            {
                "name": "Simple filter",
                "query": lambda: pl.read_parquet(parquet_path).filter(pl.col("prescription_count") > 10)
            },
            {
                "name": "Groupby aggregation",
                "query": lambda: pl.read_parquet(parquet_path).group_by("sa2_code").agg([
                    pl.col("prescription_count").sum(),
                    pl.col("cost_government").mean()
                ])
            },
            {
                "name": "Complex filter and sort",
                "query": lambda: pl.read_parquet(parquet_path)
                    .filter((pl.col("chronic_medication") == 1) & (pl.col("cost_government") > 100))
                    .sort("cost_government", descending=True)
                    .head(100)
            }
        ]
        
        for test in query_tests:
            start_time = time.time()
            result_df = test["query"]()
            query_time = time.time() - start_time
            
            # Query performance assertions
            assert query_time < 5.0, f"{test['name']} took {query_time:.2f}s, expected <5s"
            assert isinstance(result_df, pl.DataFrame)
            assert len(result_df) > 0


class TestPerformanceRegression:
    """Performance regression tests to ensure performance doesn't degrade."""
    
    def test_performance_regression_seifa_processing(self, mock_excel_seifa_file, mock_data_paths):
        """Test SEIFA processing performance hasn't regressed."""
        processor = SEIFAProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        
        # Standard test dataset
        excel_file = mock_excel_seifa_file(num_areas=1000)
        expected_path = processor.raw_dir / "SEIFA_2021_SA2_Indexes.xlsx"
        shutil.copy(excel_file, expected_path)
        
        # Baseline performance expectations
        baseline_time = 15.0  # seconds
        baseline_memory = 300  # MB
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        result_df = processor.process_seifa_file()
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        actual_time = end_time - start_time
        actual_memory = end_memory - start_memory
        
        # Regression checks (allow 20% variance)
        assert actual_time <= baseline_time * 1.2, \
            f"Performance regression: {actual_time:.1f}s vs baseline {baseline_time}s"
        
        assert actual_memory <= baseline_memory * 1.2, \
            f"Memory regression: {actual_memory:.1f}MB vs baseline {baseline_memory}MB"
        
        # Ensure functionality maintained
        assert len(result_df) > 800
        assert "sa2_code_2021" in result_df.columns
    
    def test_performance_regression_storage_optimization(self, mock_seifa_data, mock_data_paths):
        """Test storage optimization performance hasn't regressed."""
        storage_manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        optimizer = MemoryOptimizer()
        
        # Standard test dataset
        test_df = mock_seifa_data(num_areas=2000)
        
        # Baseline expectations
        baseline_compression_ratio = 0.7
        baseline_optimization_time = 5.0
        baseline_memory_reduction = 0.2
        
        # Test compression
        csv_path = mock_data_paths["parquet_dir"] / "regression_test.csv"
        parquet_path = mock_data_paths["parquet_dir"] / "regression_test.parquet"
        
        test_df.write_csv(csv_path)
        csv_size = csv_path.stat().st_size
        
        storage_manager.save_optimized_parquet(test_df, parquet_path, data_type="seifa")
        parquet_size = parquet_path.stat().st_size
        
        compression_ratio = parquet_size / csv_size
        
        # Test memory optimization
        initial_memory = test_df.estimated_size("mb")
        
        start_time = time.time()
        optimized_df = optimizer.optimize_data_types(test_df, data_category="seifa")
        optimization_time = time.time() - start_time
        
        final_memory = optimized_df.estimated_size("mb")
        memory_reduction = (initial_memory - final_memory) / initial_memory
        
        # Regression checks
        assert compression_ratio <= baseline_compression_ratio * 1.1, \
            f"Compression regression: {compression_ratio:.2f} vs baseline {baseline_compression_ratio}"
        
        assert optimization_time <= baseline_optimization_time * 1.2, \
            f"Optimization time regression: {optimization_time:.1f}s vs baseline {baseline_optimization_time}s"
        
        assert memory_reduction >= baseline_memory_reduction * 0.8, \
            f"Memory reduction regression: {memory_reduction:.2f} vs baseline {baseline_memory_reduction}"