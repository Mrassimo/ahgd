"""
Test suite for Storage Optimization modules (Phase 4)

Tests the Parquet storage optimization, lazy loading, and performance monitoring
for Australian health data at production scale.
"""

import pytest
import polars as pl
import numpy as np
from pathlib import Path
import tempfile
import time
import json

from src.data_processing.storage.parquet_storage_manager import ParquetStorageManager
from src.data_processing.storage.lazy_data_loader import LazyDataLoader, MemoryMonitor
from src.data_processing.storage.storage_performance_monitor import StoragePerformanceMonitor


@pytest.fixture
def sample_health_data():
    """Create realistic Australian health data for testing."""
    np.random.seed(42)
    n_rows = 5000
    
    return pl.DataFrame({
        "sa2_code": np.random.choice([f"1{str(i).zfill(8)}" for i in range(1000, 1500)], n_rows),
        "sa2_name": [f"Health Area {i}" for i in range(n_rows)],
        "state_name": np.random.choice(['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT'], n_rows),
        "risk_score": np.random.uniform(1, 10, n_rows),
        "access_score": np.random.uniform(1, 10, n_rows),
        "population": np.random.randint(100, 5000, n_rows),
        "prescription_count": np.random.poisson(3, n_rows),
        "total_cost": np.random.exponential(45, n_rows),
        "chronic_medication": np.random.choice([0, 1], n_rows, p=[0.7, 0.3]),
        "dispensing_date": ["2023-01-01"] * n_rows
    })


@pytest.fixture
def temp_storage_dir():
    """Create temporary directory for storage testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestParquetStorageManager:
    """Test suite for ParquetStorageManager class."""
    
    def test_initialization(self, temp_storage_dir):
        """Test storage manager initialization."""
        manager = ParquetStorageManager(temp_storage_dir)
        
        assert manager.base_path == temp_storage_dir
        assert (temp_storage_dir / "health").exists()
        assert (temp_storage_dir / "geographic").exists()
        assert (temp_storage_dir / "seifa").exists()
        assert (temp_storage_dir / "_metadata").exists()
    
    def test_schema_optimization(self, sample_health_data):
        """Test DataFrame schema optimization for compression."""
        manager = ParquetStorageManager()
        
        optimized_df = manager.optimize_dataframe_schema(sample_health_data)
        
        # Check that categorical columns were optimized
        assert optimized_df["sa2_code"].dtype == pl.Categorical
        assert optimized_df["state_name"].dtype == pl.Categorical
        
        # Check that date column was converted
        assert optimized_df["dispensing_date"].dtype == pl.Date
        
        # Check data integrity
        assert optimized_df.shape == sample_health_data.shape
        assert optimized_df["sa2_code"].n_unique() == sample_health_data["sa2_code"].n_unique()
    
    def test_write_parquet_optimized(self, sample_health_data, temp_storage_dir):
        """Test optimized Parquet writing with compression metrics."""
        manager = ParquetStorageManager(temp_storage_dir)
        file_path = temp_storage_dir / "test_health_data.parquet"
        
        metrics = manager.write_parquet_optimized(sample_health_data, file_path)
        
        # Check file was created
        assert file_path.exists()
        assert file_path.stat().st_size > 0
        
        # Check metrics
        assert "write_time_seconds" in metrics
        assert "file_size_mb" in metrics
        assert "compression_ratio" in metrics
        assert "row_count" in metrics
        assert metrics["row_count"] == sample_health_data.shape[0]
        
        # Check compression ratio is reasonable
        assert 0 <= metrics["compression_ratio"] <= 1
        
        # Check metadata file exists
        metadata_file = temp_storage_dir / "_metadata" / "test_health_data_metadata.json"
        assert metadata_file.exists()
    
    def test_read_parquet_lazy(self, sample_health_data, temp_storage_dir):
        """Test lazy Parquet reading."""
        manager = ParquetStorageManager(temp_storage_dir)
        file_path = temp_storage_dir / "test_lazy_read.parquet"
        
        # Write test file
        manager.write_parquet_optimized(sample_health_data, file_path)
        
        # Read as lazy frame
        lazy_df = manager.read_parquet_lazy(file_path)
        
        assert isinstance(lazy_df, pl.LazyFrame)
        
        # Collect and verify data
        result_df = lazy_df.collect()
        assert result_df.shape[0] == sample_health_data.shape[0]
        assert set(result_df.columns) == set(sample_health_data.columns)
    
    def test_csv_to_parquet_conversion(self, sample_health_data, temp_storage_dir):
        """Test CSV to Parquet conversion with metrics."""
        manager = ParquetStorageManager(temp_storage_dir)
        
        # Create test CSV file
        csv_path = temp_storage_dir / "test_data.csv"
        sample_health_data.write_csv(csv_path)
        
        parquet_path = temp_storage_dir / "converted_data.parquet"
        
        # Convert CSV to Parquet
        conversion_metrics = manager.convert_csv_to_parquet(csv_path, parquet_path)
        
        # Check conversion completed
        assert parquet_path.exists()
        assert "conversion_time_seconds" in conversion_metrics
        assert "size_reduction" in conversion_metrics
        assert "original_csv_size_mb" in conversion_metrics
        
        # Verify data integrity
        converted_df = pl.read_parquet(parquet_path)
        assert converted_df.shape[0] == sample_health_data.shape[0]
        
        # Check size reduction (Parquet should be smaller than CSV)
        assert conversion_metrics["size_reduction"] > 0
    
    def test_storage_summary(self, sample_health_data, temp_storage_dir):
        """Test storage summary generation."""
        manager = ParquetStorageManager(temp_storage_dir)
        
        # Create multiple test files
        for i in range(3):
            file_path = temp_storage_dir / f"health/test_file_{i}.parquet"
            manager.write_parquet_optimized(sample_health_data, file_path)
        
        summary = manager.get_storage_summary()
        
        assert "total_files" in summary
        assert "total_size_mb" in summary
        assert "files_by_category" in summary
        assert summary["total_files"] == 3
        assert "health" in summary["files_by_category"]
        assert summary["files_by_category"]["health"]["count"] == 3
    
    def test_benchmark_storage_performance(self):
        """Test storage performance benchmarking."""
        manager = ParquetStorageManager()
        
        benchmark_results = manager.benchmark_storage_performance("small")
        
        assert "test_data_rows" in benchmark_results
        assert "best_compression" in benchmark_results
        assert "compression_results" in benchmark_results
        assert "recommendation" in benchmark_results
        
        # Check that compression algorithms were tested
        compression_results = benchmark_results["compression_results"]
        assert "snappy" in compression_results
        assert "gzip" in compression_results
        
        # Check that each compression test has required metrics
        for compression, results in compression_results.items():
            assert "write_time" in results
            assert "read_time" in results
            assert "file_size_mb" in results
            assert "compression_ratio" in results


class TestLazyDataLoader:
    """Test suite for LazyDataLoader class."""
    
    def test_initialization(self, temp_storage_dir):
        """Test lazy data loader initialization."""
        cache_dir = temp_storage_dir / "cache"
        loader = LazyDataLoader(cache_dir)
        
        assert loader.cache_dir == cache_dir
        assert cache_dir.exists()
        assert isinstance(loader.memory_monitor, MemoryMonitor)
    
    def test_load_lazy_dataset_parquet(self, sample_health_data, temp_storage_dir):
        """Test loading Parquet dataset as lazy frame."""
        # Create test Parquet file
        parquet_path = temp_storage_dir / "test_data.parquet"
        sample_health_data.write_parquet(parquet_path)
        
        loader = LazyDataLoader()
        lazy_df = loader.load_lazy_dataset(parquet_path, "parquet")
        
        assert isinstance(lazy_df, pl.LazyFrame)
        
        # Verify data integrity
        result_df = lazy_df.collect()
        assert result_df.shape[0] == sample_health_data.shape[0]
    
    def test_load_lazy_dataset_csv(self, sample_health_data, temp_storage_dir):
        """Test loading CSV dataset as lazy frame."""
        # Create test CSV file
        csv_path = temp_storage_dir / "test_data.csv"
        sample_health_data.write_csv(csv_path)
        
        loader = LazyDataLoader()
        lazy_df = loader.load_lazy_dataset(csv_path, "csv")
        
        assert isinstance(lazy_df, pl.LazyFrame)
        
        # Verify data integrity
        result_df = lazy_df.collect()
        assert result_df.shape[0] == sample_health_data.shape[0]
    
    def test_execute_lazy_query(self, sample_health_data, temp_storage_dir):
        """Test lazy query execution with monitoring."""
        # Create test Parquet file
        parquet_path = temp_storage_dir / "test_data.parquet"
        sample_health_data.write_parquet(parquet_path)
        
        loader = LazyDataLoader()
        lazy_df = loader.load_lazy_dataset(parquet_path, "parquet")
        
        # Add some operations
        query_df = lazy_df.filter(pl.col("risk_score") > 5.0).group_by("state_name").agg([
            pl.col("population").sum().alias("total_population"),
            pl.col("risk_score").mean().alias("avg_risk")
        ])
        
        # Execute query
        result_df = loader.execute_lazy_query(query_df, cache_key="test_query")
        
        assert isinstance(result_df, pl.DataFrame)
        assert "state_name" in result_df.columns
        assert "total_population" in result_df.columns
        assert "avg_risk" in result_df.columns
        assert result_df.shape[0] <= 8  # Max 8 states/territories
    
    def test_batch_processing_lazy(self, sample_health_data, temp_storage_dir):
        """Test batch processing of lazy datasets."""
        # Create test Parquet file
        parquet_path = temp_storage_dir / "test_data.parquet"
        sample_health_data.write_parquet(parquet_path)
        
        loader = LazyDataLoader()
        lazy_df = loader.load_lazy_dataset(parquet_path, "parquet")
        
        # Process in batches
        batch_size = 1000
        processed_rows = 0
        batch_count = 0
        
        for batch_df in loader.batch_process_lazy(lazy_df, batch_size=batch_size):
            assert isinstance(batch_df, pl.DataFrame)
            assert batch_df.shape[0] <= batch_size
            processed_rows += batch_df.shape[0]
            batch_count += 1
            
            if batch_count > 10:  # Prevent infinite loop in tests
                break
        
        assert processed_rows > 0
        assert batch_count > 0
    
    def test_optimize_query_plan(self, sample_health_data, temp_storage_dir):
        """Test query plan optimization."""
        # Create test Parquet file
        parquet_path = temp_storage_dir / "test_data.parquet"
        sample_health_data.write_parquet(parquet_path)
        
        loader = LazyDataLoader()
        lazy_df = loader.load_lazy_dataset(parquet_path, "parquet")
        
        # Create a complex query
        complex_query = lazy_df.filter(pl.col("risk_score") > 5.0).group_by("state_name").agg([
            pl.col("population").sum(),
            pl.col("prescription_count").mean()
        ]).sort("population", descending=True)
        
        optimized_df, optimization_stats = loader.optimize_query_plan(complex_query)
        
        assert isinstance(optimized_df, pl.LazyFrame)
        assert "original_plan_lines" in optimization_stats
        assert "optimizations_applied" in optimization_stats
        assert "estimated_improvement" in optimization_stats
    
    def test_cache_functionality(self, sample_health_data, temp_storage_dir):
        """Test query result caching."""
        # Create test Parquet file
        parquet_path = temp_storage_dir / "test_data.parquet"
        sample_health_data.write_parquet(parquet_path)
        
        loader = LazyDataLoader()
        lazy_df = loader.load_lazy_dataset(parquet_path, "parquet")
        
        # Execute query with caching
        cache_key = "test_cache_query"
        query_df = lazy_df.select(["sa2_code", "state_name", "risk_score"])
        
        # First execution - should cache result
        result1 = loader.execute_lazy_query(query_df, cache_key=cache_key)
        
        # Second execution - should use cache
        result2 = loader.execute_lazy_query(query_df, cache_key=cache_key)
        
        # Results should be identical
        assert result1.shape == result2.shape
        assert result1.columns == result2.columns
        
        # Check cache contains the result
        assert cache_key in loader.query_cache
    
    def test_loader_statistics(self):
        """Test loader statistics generation."""
        loader = LazyDataLoader()
        
        stats = loader.get_loader_statistics()
        
        assert "cache_entries" in stats
        assert "cache_size_mb" in stats
        assert "current_memory_usage_gb" in stats
        assert "memory_limit_gb" in stats
        assert "cached_queries" in stats


class TestMemoryMonitor:
    """Test suite for MemoryMonitor class."""
    
    def test_memory_monitor_initialization(self):
        """Test memory monitor initialization."""
        monitor = MemoryMonitor()
        assert monitor.process is not None
    
    def test_get_memory_usage(self):
        """Test memory usage reporting."""
        monitor = MemoryMonitor()
        
        memory_gb = monitor.get_memory_usage_gb()
        assert isinstance(memory_gb, float)
        assert memory_gb >= 0
    
    def test_get_system_memory_info(self):
        """Test system memory information."""
        monitor = MemoryMonitor()
        
        memory_info = monitor.get_system_memory_info()
        assert "total_gb" in memory_info
        assert "available_gb" in memory_info
        assert "used_gb" in memory_info
        assert "percent_used" in memory_info
        
        # Check values are reasonable
        assert memory_info["total_gb"] > 0
        assert 0 <= memory_info["percent_used"] <= 100
    
    def test_is_memory_available(self):
        """Test memory availability checking."""
        monitor = MemoryMonitor()
        
        # Test with small requirement (should be available)
        assert monitor.is_memory_available(0.1) is True
        
        # Test with very large requirement (should not be available)
        assert monitor.is_memory_available(1000.0) is False


class TestStoragePerformanceMonitor:
    """Test suite for StoragePerformanceMonitor class."""
    
    def test_initialization(self, temp_storage_dir):
        """Test performance monitor initialization."""
        metrics_dir = temp_storage_dir / "metrics"
        monitor = StoragePerformanceMonitor(metrics_dir)
        
        assert monitor.metrics_dir == metrics_dir
        assert metrics_dir.exists()
        assert len(monitor.storage_metrics) == 0
        assert len(monitor.system_metrics) == 0
    
    def test_operation_tracking(self):
        """Test storage operation tracking."""
        monitor = StoragePerformanceMonitor()
        
        operation_id = "test_operation"
        
        # Start tracking
        monitor.start_operation(operation_id, "read", "/test/file.parquet")
        assert operation_id in monitor.active_operations
        
        # Simulate some processing time
        time.sleep(0.1)
        
        # End tracking
        metrics = monitor.end_operation(operation_id, rows_processed=1000, file_size_mb=10.0, compression_ratio=0.6)
        
        assert operation_id not in monitor.active_operations
        assert metrics is not None
        assert metrics.operation_type == "read"
        assert metrics.rows_processed == 1000
        assert metrics.file_size_mb == 10.0
        assert metrics.compression_ratio == 0.6
        assert metrics.duration_seconds > 0
    
    def test_query_profiling(self, sample_health_data, temp_storage_dir):
        """Test query performance profiling."""
        # Create test data
        parquet_path = temp_storage_dir / "profile_test.parquet"
        sample_health_data.write_parquet(parquet_path)
        
        monitor = StoragePerformanceMonitor()
        
        # Create lazy query
        lazy_df = pl.scan_parquet(parquet_path).filter(pl.col("risk_score") > 5.0)
        
        # Profile query
        profile = monitor.profile_query_performance("test_query", lazy_df, collect_result=True)
        
        assert "query_name" in profile
        assert "execution_time_seconds" in profile
        assert "query_plan_lines" in profile
        assert "rows_processed" in profile
        assert profile["query_name"] == "test_query"
        assert profile["execution_time_seconds"] >= 0
    
    def test_benchmark_storage_operations(self):
        """Test comprehensive storage benchmarking."""
        monitor = StoragePerformanceMonitor()
        
        benchmark_results = monitor.benchmark_storage_operations()
        
        assert "timestamp" in benchmark_results
        assert "read_performance" in benchmark_results
        assert "write_performance" in benchmark_results
        assert "system_capabilities" in benchmark_results
        
        # Check system capabilities
        capabilities = benchmark_results["system_capabilities"]
        assert "cpu_count" in capabilities
        assert "memory_total_gb" in capabilities
        assert capabilities["cpu_count"] > 0
        assert capabilities["memory_total_gb"] > 0
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        monitor = StoragePerformanceMonitor()
        
        # Add some mock metrics
        monitor.start_operation("op1", "read", "/test1.parquet")
        time.sleep(0.01)
        monitor.end_operation("op1", 1000, 5.0, 0.5)
        
        monitor.start_operation("op2", "write", "/test2.parquet")
        time.sleep(0.01)
        monitor.end_operation("op2", 2000, 10.0, 0.6)
        
        summary = monitor.get_performance_summary(hours_back=1)
        
        assert "time_period_hours" in summary
        assert "storage_operations" in summary
        assert "performance_summary" in summary
        assert summary["storage_operations"] == 2
    
    def test_optimization_recommendations(self):
        """Test optimization recommendations generation."""
        monitor = StoragePerformanceMonitor()
        
        # Add some metrics that should trigger recommendations
        monitor.start_operation("slow_op", "read", "/test.parquet")
        time.sleep(0.01)
        # Simulate slow operation
        slow_metrics = monitor.end_operation("slow_op", 1000, 5.0, 0.1)  # Poor compression
        if slow_metrics:
            slow_metrics.duration_seconds = 10.0  # Force slow operation
        
        recommendations = monitor.get_optimization_recommendations()
        
        assert isinstance(recommendations, list)
        for rec in recommendations:
            assert "type" in rec
            assert "priority" in rec
            assert "title" in rec
            assert "description" in rec


@pytest.mark.integration
class TestStorageIntegration:
    """Integration tests for storage optimization components."""
    
    def test_complete_storage_pipeline(self, sample_health_data, temp_storage_dir):
        """Test complete storage optimization pipeline."""
        # Initialize all components
        storage_manager = ParquetStorageManager(temp_storage_dir)
        lazy_loader = LazyDataLoader(temp_storage_dir / "cache")
        performance_monitor = StoragePerformanceMonitor(temp_storage_dir / "metrics")
        
        # Start performance monitoring
        performance_monitor.start_monitoring()
        
        try:
            # Step 1: Write data with optimization
            file_path = temp_storage_dir / "health/integrated_test.parquet"
            write_metrics = storage_manager.write_parquet_optimized(sample_health_data, file_path)
            
            assert write_metrics["compression_ratio"] > 0
            
            # Step 2: Load data lazily
            lazy_df = lazy_loader.load_lazy_dataset(file_path, "parquet")
            
            # Step 3: Execute query with monitoring
            operation_id = "integration_query"
            performance_monitor.start_operation(operation_id, "query", str(file_path))
            
            query_df = lazy_df.filter(pl.col("risk_score") > 7.0).group_by("state_name").agg([
                pl.col("population").sum().alias("total_population")
            ])
            
            result_df = lazy_loader.execute_lazy_query(query_df, cache_key="integration_test")
            
            query_metrics = performance_monitor.end_operation(
                operation_id, 
                result_df.shape[0], 
                write_metrics["file_size_mb"]
            )
            
            # Step 4: Verify results
            assert isinstance(result_df, pl.DataFrame)
            assert "state_name" in result_df.columns
            assert "total_population" in result_df.columns
            assert query_metrics is not None
            
            # Step 5: Get performance summary
            summary = performance_monitor.get_performance_summary()
            assert summary["storage_operations"] >= 1
            
        finally:
            performance_monitor.stop_monitoring()
    
    def test_storage_optimization_workflow(self, sample_health_data, temp_storage_dir):
        """Test storage optimization workflow with real data."""
        storage_manager = ParquetStorageManager(temp_storage_dir)
        
        # Create CSV files to optimize
        csv_files = []
        for i in range(3):
            csv_path = temp_storage_dir / f"raw_data_{i}.csv"
            sample_health_data.write_csv(csv_path)
            csv_files.append(csv_path)
        
        # Run optimization analysis
        optimization_plan = storage_manager.optimize_existing_storage(dry_run=True)
        
        assert "csv_files_to_convert" in optimization_plan
        assert "estimated_space_savings_mb" in optimization_plan
        assert optimization_plan["estimated_space_savings_mb"] > 0
        
        # Execute optimization (convert one file)
        if optimization_plan["csv_files_to_convert"]:
            csv_info = optimization_plan["csv_files_to_convert"][0]
            csv_path = Path(csv_info["file"])
            parquet_path = temp_storage_dir / "optimized" / "converted_data.parquet"
            parquet_path.parent.mkdir(exist_ok=True)
            
            conversion_metrics = storage_manager.convert_csv_to_parquet(csv_path, parquet_path)
            
            # Verify optimization worked
            assert conversion_metrics["size_reduction"] > 0
            assert parquet_path.exists()
            
            # Verify data integrity
            original_df = pl.read_csv(csv_path)
            converted_df = pl.read_parquet(parquet_path)
            assert original_df.shape[0] == converted_df.shape[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])