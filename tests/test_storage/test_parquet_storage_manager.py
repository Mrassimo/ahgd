"""
Comprehensive unit tests for Parquet Storage Manager.

Tests efficient Parquet storage with compression optimization for Australian health data:
- Optimal compression algorithms and configurations
- Column-specific optimizations (SA2 codes, categorical data)
- Schema optimization and data type conversions
- Lazy loading capabilities and performance
- Benchmark validation for compression ratios
- Large dataset handling and memory efficiency

Validates 60-70% compression rates and performance targets.
"""

import pytest
import polars as pl
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import time
import os
from datetime import datetime, date, timedelta

from src.data_processing.storage.parquet_storage_manager import ParquetStorageManager


class TestParquetStorageManager:
    """Comprehensive test suite for Parquet storage manager."""
    
    def test_parquet_manager_initialization(self, mock_data_paths):
        """Test Parquet manager initializes with proper directory structure."""
        manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        
        assert manager.base_path == mock_data_paths["parquet_dir"]
        assert manager.performance_metrics == {}
        
        # Check directory structure was created
        expected_dirs = ["health", "geographic", "seifa", "risk_assessments"]
        for dir_name in expected_dirs:
            assert (manager.base_path / dir_name).exists()
    
    def test_parquet_manager_default_path(self):
        """Test Parquet manager with default base path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            try:
                import os
                os.chdir(temp_dir)
                
                manager = ParquetStorageManager()
                assert manager.base_path.name == "parquet"
                assert manager.base_path.exists()
            finally:
                import os
                os.chdir(original_cwd)
    
    def test_optimize_schema_for_australian_data(self, mock_seifa_data, mock_health_data):
        """Test schema optimization for Australian health data patterns."""
        manager = ParquetStorageManager()
        
        # Test with SEIFA data
        seifa_df = mock_seifa_data(num_areas=50)
        optimized_seifa = manager._optimize_schema(seifa_df, data_type="seifa")
        
        # Verify SA2 codes converted to categorical
        if "sa2_code_2021" in optimized_seifa.columns:
            assert optimized_seifa["sa2_code_2021"].dtype == pl.Categorical
        
        # Verify deciles converted to Int8
        decile_columns = [col for col in optimized_seifa.columns if "decile" in col]
        for col in decile_columns:
            assert optimized_seifa[col].dtype == pl.Int8
        
        # Test with health data
        health_df = mock_health_data(num_records=100, num_sa2_areas=20)
        optimized_health = manager._optimize_schema(health_df, data_type="health")
        
        # Verify categorical optimizations
        if "sa2_code" in optimized_health.columns:
            assert optimized_health["sa2_code"].dtype == pl.Categorical
        
        if "state" in optimized_health.columns:
            assert optimized_health["state"].dtype == pl.Categorical
    
    def test_save_optimized_parquet_seifa(self, mock_seifa_data, mock_data_paths):
        """Test saving SEIFA data with optimal Parquet configuration."""
        manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        seifa_df = mock_seifa_data(num_areas=100)
        
        output_path = manager.base_path / "seifa" / "seifa_optimized.parquet"
        
        # Save with optimization
        result_path = manager.save_optimized_parquet(
            seifa_df, 
            output_path, 
            data_type="seifa"
        )
        
        assert result_path.exists()
        assert result_path == output_path
        
        # Verify file was created with compression
        file_size = result_path.stat().st_size
        assert file_size > 0
        assert file_size < 1024 * 1024  # Should be well compressed for 100 records
        
        # Test reading back
        loaded_df = pl.read_parquet(result_path)
        assert len(loaded_df) == len(seifa_df)
        
        # Verify schema optimizations were applied
        if "sa2_code_2021" in loaded_df.columns:
            assert loaded_df["sa2_code_2021"].dtype == pl.Categorical
    
    def test_save_optimized_parquet_health(self, mock_health_data, mock_data_paths):
        """Test saving health data with optimal Parquet configuration."""
        manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        health_df = mock_health_data(num_records=500, num_sa2_areas=50)
        
        output_path = manager.base_path / "health" / "health_optimized.parquet"
        
        # Save with optimization
        result_path = manager.save_optimized_parquet(
            health_df,
            output_path,
            data_type="health"
        )
        
        assert result_path.exists()
        
        # Test reading back
        loaded_df = pl.read_parquet(result_path)
        assert len(loaded_df) == len(health_df)
        
        # Verify categorical optimizations
        if "sa2_code" in loaded_df.columns:
            assert loaded_df["sa2_code"].dtype == pl.Categorical
    
    def test_compression_benchmarking(self, mock_seifa_data, mock_data_paths, performance_benchmarks):
        """Test compression ratio benchmarking meets performance targets."""
        manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        
        # Create larger dataset for meaningful compression testing
        large_df = mock_seifa_data(num_areas=1000)
        
        # Save uncompressed CSV for comparison
        csv_path = mock_data_paths["parquet_dir"] / "uncompressed.csv"
        large_df.write_csv(csv_path)
        csv_size = csv_path.stat().st_size
        
        # Save optimized Parquet
        parquet_path = manager.base_path / "seifa" / "compressed.parquet"
        manager.save_optimized_parquet(large_df, parquet_path, data_type="seifa")
        parquet_size = parquet_path.stat().st_size
        
        # Calculate compression ratio
        compression_ratio = parquet_size / csv_size
        
        # Verify meets benchmark (at least 60% compression)
        target_ratio = performance_benchmarks["parquet_compression"]["min_compression_ratio"]
        assert compression_ratio <= target_ratio, f"Compression ratio {compression_ratio:.2f} exceeds target {target_ratio}"
        
        # Store performance metrics
        manager.performance_metrics["compression_test"] = {
            "csv_size_mb": csv_size / (1024 * 1024),
            "parquet_size_mb": parquet_size / (1024 * 1024),
            "compression_ratio": compression_ratio
        }
    
    def test_read_performance_benchmarking(self, mock_seifa_data, mock_data_paths, performance_benchmarks):
        """Test read performance meets benchmark targets."""
        manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        
        # Create test dataset
        test_df = mock_seifa_data(num_areas=2000)  # Larger dataset
        parquet_path = manager.base_path / "seifa" / "performance_test.parquet"
        manager.save_optimized_parquet(test_df, parquet_path, data_type="seifa")
        
        file_size_mb = parquet_path.stat().st_size / (1024 * 1024)
        
        # Benchmark read performance
        start_time = time.time()
        loaded_df = pl.read_parquet(parquet_path)
        read_time = time.time() - start_time
        
        # Calculate read speed
        read_speed = read_time / file_size_mb  # seconds per MB
        
        # Verify meets benchmark
        max_read_time = performance_benchmarks["parquet_compression"]["max_read_time_per_mb"]
        assert read_speed <= max_read_time, f"Read speed {read_speed:.3f}s/MB exceeds target {max_read_time}s/MB"
        
        # Verify data integrity
        assert len(loaded_df) == len(test_df)
        
        # Store performance metrics
        manager.performance_metrics["read_performance"] = {
            "file_size_mb": file_size_mb,
            "read_time_seconds": read_time,
            "read_speed_seconds_per_mb": read_speed
        }
    
    def test_write_performance_benchmarking(self, mock_health_data, mock_data_paths, performance_benchmarks):
        """Test write performance meets benchmark targets."""
        manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        
        # Create test dataset
        test_df = mock_health_data(num_records=5000, num_sa2_areas=200)
        parquet_path = manager.base_path / "health" / "write_performance_test.parquet"
        
        # Benchmark write performance
        start_time = time.time()
        manager.save_optimized_parquet(test_df, parquet_path, data_type="health")
        write_time = time.time() - start_time
        
        file_size_mb = parquet_path.stat().st_size / (1024 * 1024)
        write_speed = write_time / file_size_mb  # seconds per MB
        
        # Verify meets benchmark
        max_write_time = performance_benchmarks["parquet_compression"]["max_write_time_per_mb"]
        assert write_speed <= max_write_time, f"Write speed {write_speed:.3f}s/MB exceeds target {max_write_time}s/MB"
        
        # Store performance metrics
        manager.performance_metrics["write_performance"] = {
            "records": len(test_df),
            "file_size_mb": file_size_mb,
            "write_time_seconds": write_time,
            "write_speed_seconds_per_mb": write_speed
        }
    
    def test_lazy_loading_capabilities(self, mock_seifa_data, mock_data_paths):
        """Test lazy loading capabilities for memory efficiency."""
        manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        
        # Create and save test data
        test_df = mock_seifa_data(num_areas=1000)
        parquet_path = manager.base_path / "seifa" / "lazy_test.parquet"
        manager.save_optimized_parquet(test_df, parquet_path, data_type="seifa")
        
        # Test lazy frame creation
        lazy_frame = manager.create_lazy_frame(parquet_path)
        assert isinstance(lazy_frame, pl.LazyFrame)
        
        # Test selective column loading
        selected_columns = ["sa2_code_2021", "irsd_decile"]
        if all(col in test_df.columns for col in selected_columns):
            lazy_subset = lazy_frame.select(selected_columns)
            result_df = lazy_subset.collect()
            
            assert len(result_df.columns) == 2
            assert all(col in result_df.columns for col in selected_columns)
            assert len(result_df) == len(test_df)
    
    def test_batch_processing_large_datasets(self, mock_health_data, mock_data_paths):
        """Test batch processing for large datasets that don't fit in memory."""
        manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        
        # Simulate large dataset processing
        batch_size = 1000
        total_records = 5000
        output_dir = manager.base_path / "health" / "batched"
        output_dir.mkdir(exist_ok=True)
        
        # Process in batches
        batch_files = []
        for i in range(0, total_records, batch_size):
            batch_df = mock_health_data(num_records=min(batch_size, total_records - i), num_sa2_areas=50)
            batch_file = output_dir / f"batch_{i//batch_size:03d}.parquet"
            
            manager.save_optimized_parquet(batch_df, batch_file, data_type="health")
            batch_files.append(batch_file)
        
        # Verify all batch files created
        assert len(batch_files) == 5  # 5000 / 1000
        
        # Test reading batches back
        total_loaded = 0
        for batch_file in batch_files:
            assert batch_file.exists()
            batch_df = pl.read_parquet(batch_file)
            total_loaded += len(batch_df)
        
        assert total_loaded == total_records
    
    def test_data_partitioning_by_state(self, mock_health_data, mock_data_paths):
        """Test data partitioning by Australian state for performance."""
        manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        
        # Create health data with explicit state distribution
        health_df = mock_health_data(num_records=1000, num_sa2_areas=100)
        
        # Ensure state column exists and has Australian states
        australian_states = ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT"]
        health_df = health_df.with_columns(
            pl.Series("state", np.random.choice(australian_states, len(health_df)))
        )
        
        # Partition and save by state
        partitioned_files = manager.save_partitioned_parquet(
            health_df,
            manager.base_path / "health" / "partitioned",
            partition_columns=["state"],
            data_type="health"
        )
        
        # Verify partitions were created
        assert len(partitioned_files) > 0
        
        # Test reading specific partition
        nsw_files = [f for f in partitioned_files if "NSW" in str(f)]
        if nsw_files:
            nsw_df = pl.read_parquet(nsw_files[0])
            assert len(nsw_df) > 0
            # All records should be from NSW
            assert all(state == "NSW" for state in nsw_df["state"].to_list())
    
    def test_column_encoding_optimization(self, mock_seifa_data, mock_data_paths):
        """Test column-specific encoding optimizations."""
        manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        
        # Create data with high cardinality and categorical columns
        test_df = mock_seifa_data(num_areas=500)
        
        # Add some high-cardinality columns for testing
        test_df = test_df.with_columns([
            # Low cardinality - should use dictionary encoding
            pl.Series("risk_category", np.random.choice(["Low", "Medium", "High"], len(test_df))),
            pl.Series("access_category", np.random.choice(["Good", "Fair", "Poor"], len(test_df))),
            
            # High cardinality - should use different encoding
            pl.Series("unique_id", [f"ID_{i:06d}" for i in range(len(test_df))])
        ])
        
        # Save with encoding optimizations
        output_path = manager.base_path / "seifa" / "encoded_test.parquet"
        manager.save_optimized_parquet(test_df, output_path, data_type="seifa")
        
        # Verify file was created and is reasonably sized
        assert output_path.exists()
        file_size = output_path.stat().st_size
        assert file_size > 0
        
        # Test reading back with correct types
        loaded_df = pl.read_parquet(output_path)
        
        # Categorical columns should be preserved
        if "risk_category" in loaded_df.columns:
            assert loaded_df["risk_category"].dtype == pl.Categorical
        if "sa2_code_2021" in loaded_df.columns:
            assert loaded_df["sa2_code_2021"].dtype == pl.Categorical
    
    def test_memory_usage_optimization(self, mock_seifa_data, mock_data_paths, memory_profiler):
        """Test memory usage during Parquet operations."""
        manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        
        # Start memory profiling
        memory_profiler.start()
        initial_memory = memory_profiler.get_current_usage()
        
        # Create and process large dataset
        large_df = mock_seifa_data(num_areas=2000)
        output_path = manager.base_path / "seifa" / "memory_test.parquet"
        
        # Save with optimization
        manager.save_optimized_parquet(large_df, output_path, data_type="seifa")
        
        # Check memory usage
        peak_memory = memory_profiler.get_peak_usage()
        memory_increase = memory_profiler.get_increase()
        
        # Memory increase should be reasonable
        assert memory_increase < 500  # Less than 500MB increase
        
        # Clean up large objects
        del large_df
        
        # Load back efficiently
        loaded_df = pl.read_parquet(output_path)
        final_memory = memory_profiler.get_current_usage()
        
        # Verify data integrity
        assert len(loaded_df) == 2000
        
        # Store memory metrics
        manager.performance_metrics["memory_usage"] = {
            "initial_memory_mb": initial_memory,
            "peak_memory_mb": peak_memory,
            "memory_increase_mb": memory_increase,
            "final_memory_mb": final_memory
        }
    
    def test_concurrent_parquet_operations(self, mock_health_data, mock_data_paths):
        """Test thread safety of concurrent Parquet operations."""
        import concurrent.futures
        import threading
        
        manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        
        def save_parquet_batch(batch_id):
            """Save a Parquet file in a separate thread."""
            batch_df = mock_health_data(num_records=200, num_sa2_areas=20)
            output_path = manager.base_path / "health" / f"concurrent_batch_{batch_id}.parquet"
            
            result_path = manager.save_optimized_parquet(
                batch_df, 
                output_path, 
                data_type="health"
            )
            
            return result_path, len(batch_df)
        
        # Run multiple saves concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(save_parquet_batch, i) for i in range(8)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Verify all operations succeeded
        assert len(results) == 8
        
        for result_path, record_count in results:
            assert result_path.exists()
            assert record_count == 200
            
            # Verify file integrity
            loaded_df = pl.read_parquet(result_path)
            assert len(loaded_df) == record_count
    
    def test_parquet_metadata_preservation(self, mock_seifa_data, mock_data_paths):
        """Test preservation of metadata in Parquet files."""
        manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        
        # Create dataset with metadata
        test_df = mock_seifa_data(num_areas=100)
        
        # Add custom metadata
        metadata = {
            "source": "Australian Bureau of Statistics",
            "dataset": "SEIFA 2021",
            "processing_date": datetime.now().isoformat(),
            "record_count": len(test_df),
            "sa2_areas": test_df["sa2_code_2021"].n_unique() if "sa2_code_2021" in test_df.columns else 0
        }
        
        output_path = manager.base_path / "seifa" / "metadata_test.parquet"
        
        # Save with metadata
        manager.save_with_metadata(test_df, output_path, metadata, data_type="seifa")
        
        # Read back and verify metadata
        parquet_file = pq.ParquetFile(output_path)
        stored_metadata = parquet_file.metadata.metadata
        
        if stored_metadata:
            # Verify some metadata was preserved
            assert b"source" in stored_metadata or "source" in str(stored_metadata)
    
    def test_schema_evolution_handling(self, mock_seifa_data, mock_data_paths):
        """Test handling of schema evolution in Parquet files."""
        manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        
        # Create initial dataset
        initial_df = mock_seifa_data(num_areas=50)
        output_path = manager.base_path / "seifa" / "schema_evolution_test.parquet"
        
        # Save initial version
        manager.save_optimized_parquet(initial_df, output_path, data_type="seifa")
        
        # Create evolved dataset with additional columns
        evolved_df = initial_df.with_columns([
            pl.lit("2024").alias("data_year"),
            pl.Series("quality_score", np.random.uniform(0.8, 1.0, len(initial_df)))
        ])
        
        # Save evolved version (should handle schema differences)
        evolved_path = manager.base_path / "seifa" / "schema_evolved.parquet"
        manager.save_optimized_parquet(evolved_df, evolved_path, data_type="seifa")
        
        # Verify both can be read
        initial_loaded = pl.read_parquet(output_path)
        evolved_loaded = pl.read_parquet(evolved_path)
        
        assert len(initial_loaded.columns) < len(evolved_loaded.columns)
        assert "data_year" in evolved_loaded.columns
        assert "quality_score" in evolved_loaded.columns
    
    def test_error_handling_invalid_data(self, mock_data_paths):
        """Test error handling with invalid data types."""
        manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        
        # Create DataFrame with problematic data
        problematic_df = pl.DataFrame({
            "sa2_code": ["123456789", "INVALID_CODE", None],
            "bad_numeric": ["not_a_number", "123", "456"],
            "mixed_types": [123, "string", None]
        })
        
        output_path = manager.base_path / "test" / "error_handling.parquet"
        
        # Should handle errors gracefully
        try:
            result_path = manager.save_optimized_parquet(
                problematic_df, 
                output_path, 
                data_type="health"
            )
            # If successful, verify file was created
            assert result_path.exists()
        except Exception as e:
            # If it fails, ensure it's a reasonable error
            assert isinstance(e, (ValueError, TypeError, pl.ComputeError))
    
    def test_cleanup_temporary_files(self, mock_seifa_data, mock_data_paths):
        """Test cleanup of temporary files during processing."""
        manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        
        # Count initial files
        initial_files = list(manager.base_path.rglob("*"))
        initial_count = len(initial_files)
        
        # Process data that might create temporary files
        test_df = mock_seifa_data(num_areas=100)
        output_path = manager.base_path / "seifa" / "cleanup_test.parquet"
        
        manager.save_optimized_parquet(test_df, output_path, data_type="seifa")
        
        # Count final files
        final_files = list(manager.base_path.rglob("*"))
        final_count = len(final_files)
        
        # Should not have excessive temporary files
        temp_files = [f for f in final_files if f.name.startswith(".") or "temp" in f.name.lower()]
        assert len(temp_files) <= 2  # Allow for minimal temporary files


class TestParquetStorageConfiguration:
    """Test Parquet storage configuration and schema optimizations."""
    
    def test_parquet_config_structure(self):
        """Test Parquet configuration has expected structure."""
        config = ParquetStorageManager.PARQUET_CONFIG
        
        required_keys = [
            "compression", "row_group_size", "use_pyarrow", "pyarrow_options"
        ]
        
        for key in required_keys:
            assert key in config
        
        # Verify data types
        assert isinstance(config["compression"], str)
        assert isinstance(config["row_group_size"], int)
        assert isinstance(config["use_pyarrow"], bool)
        assert isinstance(config["pyarrow_options"], dict)
    
    def test_schema_optimizations_complete(self):
        """Test schema optimizations cover Australian health data patterns."""
        optimizations = ParquetStorageManager.SCHEMA_OPTIMIZATIONS
        
        # Check SA2 code optimizations
        assert "sa2_code" in optimizations
        assert "sa2_code_2021" in optimizations
        assert optimizations["sa2_code"] == pl.Categorical
        
        # Check SEIFA decile optimizations
        seifa_deciles = ["irsd_decile", "irsad_decile", "ier_decile", "ieo_decile"]
        for decile_col in seifa_deciles:
            assert decile_col in optimizations
            assert optimizations[decile_col] == pl.Int8
        
        # Check categorical optimizations
        categorical_cols = ["state_name", "risk_category", "access_category"]
        for cat_col in categorical_cols:
            assert cat_col in optimizations
            assert optimizations[cat_col] == pl.Categorical
    
    def test_compression_algorithm_selection(self):
        """Test compression algorithm is appropriate for health data."""
        config = ParquetStorageManager.PARQUET_CONFIG
        
        # Snappy is good balance of speed/compression for health data
        assert config["compression"] == "snappy"
        
        # Row group size should be optimized for SA2-level queries
        assert 10000 <= config["row_group_size"] <= 100000
        
        # PyArrow should be enabled for better compression
        assert config["use_pyarrow"] is True