"""
Test suite for Memory Optimization (Phase 4.3)

Tests the advanced memory optimization strategies for Australian health data
processing with large datasets (497,181+ records).
"""

import pytest
import polars as pl
import numpy as np
from pathlib import Path
import tempfile
import time
import psutil

from src.data_processing.storage.memory_optimizer import (
    MemoryOptimizer, SystemMemoryMonitor, MemoryPressureDetector,
    MemoryProfile, MemoryRecommendation
)


@pytest.fixture
def large_health_dataset():
    """Create large health dataset for memory optimization testing."""
    np.random.seed(42)
    n_rows = 50000  # Large enough to test memory optimization
    
    return pl.DataFrame({
        "sa2_code": np.random.choice([f"1{str(i).zfill(8)}" for i in range(1000, 2000)], n_rows),
        "sa2_name": [f"Health Area {i}" for i in range(n_rows)],
        "state_name": np.random.choice(['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT'], n_rows),
        "postcode": np.random.choice([str(i) for i in range(2000, 8000)], n_rows),
        "age_group": np.random.choice(['0-17', '18-34', '35-49', '50-64', '65+'], n_rows),
        "prescription_count": np.random.randint(0, 100, n_rows),  # int64 by default
        "total_cost": np.random.uniform(10.0, 1000.0, n_rows),   # float64 by default
        "risk_score": np.random.uniform(1.0, 10.0, n_rows),     # float64 by default
        "seifa_irsd_decile": np.random.randint(1, 11, n_rows),   # int64 by default
        "seifa_irsad_decile": np.random.randint(1, 11, n_rows),  # int64 by default
        "population": np.random.randint(100, 10000, n_rows),     # int64 by default
        "chronic_medication": np.random.choice([0, 1], n_rows),  # int64 by default
        "dispensing_date": ["2023-01-01"] * n_rows
    })


@pytest.fixture
def temp_storage_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def memory_optimizer():
    """Create memory optimizer instance for testing."""
    return MemoryOptimizer(memory_limit_gb=2.0, enable_profiling=True)


class TestMemoryOptimizer:
    """Test suite for MemoryOptimizer class."""
    
    def test_initialization(self):
        """Test memory optimizer initialization."""
        optimizer = MemoryOptimizer(memory_limit_gb=4.0, enable_profiling=True)
        
        assert optimizer.memory_limit_gb == 4.0
        assert optimizer.enable_profiling is True
        assert len(optimizer.memory_profiles) == 0
        assert len(optimizer.active_operations) == 0
        assert isinstance(optimizer.system_monitor, SystemMemoryMonitor)
        assert isinstance(optimizer.memory_pressure_detector, MemoryPressureDetector)
    
    def test_safe_memory_limit_calculation(self):
        """Test automatic memory limit calculation."""
        optimizer = MemoryOptimizer()  # Should auto-calculate
        
        # Should be reasonable bounds
        assert 1.0 <= optimizer.memory_limit_gb <= 16.0
        
        # Should be related to system memory
        system_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        expected_limit = max(1.0, min(16.0, system_memory_gb * 0.6))
        assert abs(optimizer.memory_limit_gb - expected_limit) < 0.1
    
    def test_dataframe_memory_optimization(self, large_health_dataset, memory_optimizer):
        """Test DataFrame memory optimization with health data."""
        original_size_mb = large_health_dataset.estimated_size("mb")
        
        optimized_df, stats = memory_optimizer.optimize_dataframe_memory(
            large_health_dataset, "health"
        )
        
        # Check that optimization occurred
        assert "memory_savings_mb" in stats
        assert "memory_savings_percent" in stats
        assert "optimizations_applied" in stats
        
        # Should have some memory savings
        assert stats["memory_savings_mb"] >= 0
        
        # Check data integrity
        assert optimized_df.shape == large_health_dataset.shape
        assert set(optimized_df.columns) == set(large_health_dataset.columns)
        
        # Check specific optimizations for health data
        optimizations = stats["optimizations_applied"]
        optimization_text = " ".join(optimizations)
        
        # Should optimize categorical columns
        assert optimized_df["sa2_code"].dtype == pl.Categorical
        assert optimized_df["state_name"].dtype == pl.Categorical
        assert optimized_df["age_group"].dtype == pl.Categorical
        
        # Should optimize integer sizes
        assert optimized_df["seifa_irsd_decile"].dtype in [pl.Int8, pl.Int16]
        assert optimized_df["seifa_irsad_decile"].dtype in [pl.Int8, pl.Int16]
        
        # Should optimize prescription counts to int32 or smaller
        assert optimized_df["prescription_count"].dtype in [pl.Int8, pl.Int16, pl.Int32]
    
    def test_health_data_specific_optimizations(self, memory_optimizer):
        """Test health data specific optimization patterns."""
        # Create DataFrame with specific health data patterns
        health_df = pl.DataFrame({
            "sa2_code": ["101234567", "102345678", "103456789"] * 100,
            "state_name": ["NSW", "VIC", "QLD"] * 100,
            "seifa_decile": [1, 5, 10] * 100,
            "age_group": ["18-34", "35-49", "50-64"] * 100,
            "population_count": [1000, 2000, 3000] * 100,
            "prescription_count": [5, 10, 15] * 100
        })
        
        optimized_df, stats = memory_optimizer.optimize_dataframe_memory(health_df, "health")
        
        # Check health-specific optimizations
        assert optimized_df["sa2_code"].dtype == pl.Categorical
        assert optimized_df["state_name"].dtype == pl.Categorical
        assert optimized_df["age_group"].dtype == pl.Categorical
        assert optimized_df["seifa_decile"].dtype == pl.Int8  # 1-10 range fits in int8
        
        # Check optimization log
        optimizations = stats["optimizations_applied"]
        optimization_text = " ".join(optimizations)
        assert "SA2" in optimization_text or "sa2" in optimization_text.lower()
        assert "state" in optimization_text.lower()
    
    def test_streaming_processing_large_dataset(self, large_health_dataset, temp_storage_dir, memory_optimizer):
        """Test streaming processing for large datasets."""
        # Save test data to file
        test_file = temp_storage_dir / "large_health_data.parquet"
        large_health_dataset.write_parquet(test_file)
        
        # Define simple processing function
        def simple_processing(df: pl.DataFrame) -> pl.DataFrame:
            return df.filter(pl.col("prescription_count") > 10).with_columns([
                (pl.col("total_cost") * 1.1).alias("adjusted_cost")
            ])
        
        # Process with streaming
        result = memory_optimizer.process_large_dataset_streaming(
            test_file,
            simple_processing,
            batch_size=5000  # Small batches for testing
        )
        
        # Check result structure
        assert "result" in result or "stats" in result
        assert "stats" in result
        
        stats = result["stats"]
        assert "total_rows_processed" in stats
        assert "batches_processed" in stats
        assert stats["total_rows_processed"] > 0
        assert stats["batches_processed"] > 0
        
        # If we got a result DataFrame, verify it
        if "result" in result and result["result"] is not None:
            result_df = result["result"]
            assert isinstance(result_df, pl.DataFrame)
            assert "adjusted_cost" in result_df.columns
            # Should only contain rows with prescription_count > 10
            assert (result_df["prescription_count"] > 10).all()
    
    def test_optimal_batch_size_calculation(self, temp_storage_dir, memory_optimizer):
        """Test optimal batch size calculation."""
        # Create test file of known size
        test_data = pl.DataFrame({
            "col1": range(10000),
            "col2": ["test"] * 10000
        })
        test_file = temp_storage_dir / "test_data.parquet"
        test_data.write_parquet(test_file)
        
        batch_size = memory_optimizer._calculate_optimal_batch_size(test_file)
        
        # Should return reasonable batch size
        assert 1000 <= batch_size <= 100000
        assert isinstance(batch_size, int)
    
    def test_memory_efficient_lazy_query(self, temp_storage_dir, large_health_dataset, memory_optimizer):
        """Test memory-efficient lazy query creation."""
        # Create multiple test files
        file_paths = []
        for i in range(3):
            file_path = temp_storage_dir / f"test_data_{i}.parquet"
            large_health_dataset.slice(i * 10000, 10000).write_parquet(file_path)
            file_paths.append(file_path)
        
        # Create lazy query with operations
        query_operations = [
            "select:sa2_code,state_name,prescription_count",
            "filter:prescription_count>5"
        ]
        
        lazy_query = memory_optimizer.create_memory_efficient_lazy_query(
            file_paths, query_operations
        )
        
        assert isinstance(lazy_query, pl.LazyFrame)
        
        # Execute query to verify it works
        result_df = lazy_query.collect()
        assert isinstance(result_df, pl.DataFrame)
        assert result_df.shape[0] > 0
    
    def test_memory_tracking(self, memory_optimizer):
        """Test memory tracking functionality."""
        operation_id = "test_operation"
        
        # Start tracking
        memory_optimizer._start_memory_tracking(operation_id, "test")
        assert operation_id in memory_optimizer.active_operations
        
        # Simulate some processing
        time.sleep(0.1)
        
        # End tracking
        additional_stats = {
            "optimized_size_mb": 10.0,
            "optimizations_applied": ["test_optimization"]
        }
        memory_optimizer._end_memory_tracking(operation_id, 1000, additional_stats)
        
        # Check that profile was created
        assert operation_id not in memory_optimizer.active_operations
        assert len(memory_optimizer.memory_profiles) == 1
        
        profile = memory_optimizer.memory_profiles[0]
        assert profile.operation_id == operation_id
        assert profile.operation_type == "test"
        assert profile.rows_processed == 1000
        assert profile.processing_time_seconds > 0
    
    def test_memory_optimization_recommendations(self, memory_optimizer):
        """Test memory optimization recommendation generation."""
        # Add some mock profiles to trigger recommendations
        from datetime import datetime
        
        # High memory usage profile
        high_memory_profile = MemoryProfile(
            operation_id="high_memory_op",
            operation_type="test",
            timestamp=datetime.now().isoformat(),
            peak_memory_mb=2000.0,  # 2GB
            memory_efficiency_ratio=0.05,  # Low efficiency
            processing_time_seconds=15.0,  # Slow
            rows_processed=1000,
            columns_processed=10,
            optimization_applied=[]
        )
        memory_optimizer.memory_profiles.append(high_memory_profile)
        
        recommendations = memory_optimizer.get_memory_optimization_recommendations()
        
        assert isinstance(recommendations, list)
        
        # Should have recommendations for high memory usage
        high_priority_recs = [r for r in recommendations if r.priority == "high"]
        assert len(high_priority_recs) > 0
        
        # Check recommendation structure
        for rec in recommendations:
            assert hasattr(rec, 'priority')
            assert hasattr(rec, 'category')
            assert hasattr(rec, 'title')
            assert hasattr(rec, 'description')
            assert hasattr(rec, 'estimated_memory_savings_mb')
            assert hasattr(rec, 'implementation_effort')
    
    def test_memory_optimization_summary(self, memory_optimizer):
        """Test memory optimization summary generation."""
        # Add some test profiles
        test_profile = MemoryProfile(
            operation_id="test_summary",
            operation_type="optimization",
            timestamp="2023-01-01T12:00:00",
            peak_memory_mb=500.0,
            memory_efficiency_ratio=0.3,
            processing_time_seconds=5.0,
            rows_processed=10000,
            columns_processed=15,
            optimization_applied=["categorical", "Downcasted"]
        )
        memory_optimizer.memory_profiles.append(test_profile)
        
        summary = memory_optimizer.get_memory_optimization_summary()
        
        # Check summary structure
        assert "total_operations_tracked" in summary
        assert "average_peak_memory_mb" in summary
        assert "average_efficiency_ratio" in summary
        assert "current_memory_usage_gb" in summary
        assert "memory_limit_gb" in summary
        assert "optimization_categories" in summary
        
        # Check values
        assert summary["total_operations_tracked"] >= 1
        assert summary["memory_limit_gb"] == memory_optimizer.memory_limit_gb
        
        # Check optimization categories
        categories = summary["optimization_categories"]
        assert "data_type_optimization" in categories
        assert "categorical_encoding" in categories
        assert categories["data_type_optimization"] >= 1  # Should count "Downcasted"
        assert categories["categorical_encoding"] >= 1    # Should count "categorical"


class TestSystemMemoryMonitor:
    """Test suite for SystemMemoryMonitor class."""
    
    def test_memory_monitor_initialization(self):
        """Test memory monitor initialization."""
        monitor = SystemMemoryMonitor()
        assert monitor.process is not None
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        monitor = SystemMemoryMonitor()
        
        memory_gb = monitor.get_memory_usage_gb()
        assert isinstance(memory_gb, float)
        assert memory_gb >= 0
        
        available_gb = monitor.get_available_memory_gb()
        assert isinstance(available_gb, float)
        assert available_gb >= 0
    
    def test_comprehensive_memory_info(self):
        """Test comprehensive memory information."""
        monitor = SystemMemoryMonitor()
        
        memory_info = monitor.get_memory_info()
        
        required_keys = [
            "total_gb", "available_gb", "used_gb", "percent_used",
            "process_rss_gb", "process_vms_gb"
        ]
        
        for key in required_keys:
            assert key in memory_info
            assert isinstance(memory_info[key], (int, float))
            assert memory_info[key] >= 0


class TestMemoryPressureDetector:
    """Test suite for MemoryPressureDetector class."""
    
    def test_pressure_detector_initialization(self):
        """Test pressure detector initialization."""
        detector = MemoryPressureDetector(pressure_threshold=0.8)
        assert detector.pressure_threshold == 0.8
        assert len(detector.pressure_history) == 0
    
    def test_pressure_detection(self):
        """Test memory pressure detection."""
        detector = MemoryPressureDetector(pressure_threshold=0.9)  # High threshold for testing
        
        is_under_pressure = detector.is_under_pressure()
        assert isinstance(is_under_pressure, bool)
        
        pressure_level = detector.get_pressure_level()
        assert isinstance(pressure_level, float)
        assert 0.0 <= pressure_level <= 1.0
    
    def test_pressure_trend_analysis(self):
        """Test pressure trend analysis."""
        detector = MemoryPressureDetector()
        
        # Add some pressure history manually for testing
        for i in range(10):
            detector.pressure_history.append(0.5 + i * 0.02)  # Increasing trend
        
        trend = detector.get_pressure_trend()
        assert trend in ["increasing", "stable", "decreasing", "unknown"]


@pytest.mark.integration
class TestMemoryOptimizationIntegration:
    """Integration tests for memory optimization components."""
    
    def test_complete_memory_optimization_workflow(self, large_health_dataset, temp_storage_dir):
        """Test complete memory optimization workflow."""
        # Initialize optimizer
        optimizer = MemoryOptimizer(memory_limit_gb=3.0, enable_profiling=True)
        
        # Step 1: Optimize DataFrame memory
        optimized_df, opt_stats = optimizer.optimize_dataframe_memory(large_health_dataset, "health")
        
        assert opt_stats["memory_savings_mb"] >= 0
        assert len(opt_stats["optimizations_applied"]) > 0
        
        # Step 2: Save to file for streaming test
        test_file = temp_storage_dir / "optimized_data.parquet"
        optimized_df.write_parquet(test_file)
        
        # Step 3: Process with streaming
        def analysis_function(df: pl.DataFrame) -> pl.DataFrame:
            return df.group_by("state_name").agg([
                pl.col("prescription_count").sum().alias("total_prescriptions"),
                pl.col("total_cost").mean().alias("avg_cost"),
                pl.col("risk_score").mean().alias("avg_risk")
            ])
        
        streaming_result = optimizer.process_large_dataset_streaming(
            test_file,
            analysis_function,
            batch_size=10000
        )
        
        assert "stats" in streaming_result
        assert streaming_result["stats"]["total_rows_processed"] > 0
        
        # Step 4: Get optimization summary
        summary = optimizer.get_memory_optimization_summary()
        assert summary["total_operations_tracked"] >= 2  # DataFrame opt + streaming
        
        # Step 5: Get recommendations
        recommendations = optimizer.get_memory_optimization_recommendations()
        assert isinstance(recommendations, list)
    
    def test_memory_pressure_response(self, temp_storage_dir):
        """Test system response to memory pressure."""
        # Create optimizer with low memory limit to trigger pressure responses
        optimizer = MemoryOptimizer(memory_limit_gb=0.5, enable_profiling=True)
        
        # Create data that should trigger memory pressure
        large_data = pl.DataFrame({
            "col1": range(100000),
            "col2": ["test_string_" + str(i) for i in range(100000)],
            "col3": np.random.random(100000)
        })
        
        # Test optimization under pressure
        optimized_df, stats = optimizer.optimize_dataframe_memory(large_data)
        
        # Should still complete successfully
        assert "memory_savings_mb" in stats
        assert optimized_df.shape[0] == large_data.shape[0]
        
        # Memory pressure should trigger more aggressive optimizations
        assert len(stats["optimizations_applied"]) > 0
    
    def test_real_world_australian_health_data_optimization(self):
        """Test optimization with realistic Australian health data patterns."""
        # Create realistic Australian health dataset
        np.random.seed(42)
        n_records = 25000  # Subset of the 497,181 records
        
        realistic_data = pl.DataFrame({
            # Geographic identifiers
            "sa2_code": np.random.choice([f"1{str(i).zfill(8)}" for i in range(1001, 3000)], n_records),
            "sa2_name": [f"Statistical Area {i}" for i in range(n_records)],
            "state_territory": np.random.choice(['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT'], n_records),
            "postcode": np.random.choice([str(i) for i in range(2000, 8000)], n_records),
            
            # SEIFA indices (1-10 deciles)
            "seifa_irsd_decile": np.random.randint(1, 11, n_records),
            "seifa_irsad_decile": np.random.randint(1, 11, n_records),
            "seifa_ier_decile": np.random.randint(1, 11, n_records),
            "seifa_ieo_decile": np.random.randint(1, 11, n_records),
            
            # Health metrics
            "prescription_count": np.random.poisson(5, n_records),
            "total_cost_aud": np.random.exponential(50, n_records),
            "chronic_conditions": np.random.randint(0, 8, n_records),
            "age_group": np.random.choice(['0-17', '18-34', '35-49', '50-64', '65-79', '80+'], n_records),
            "gender": np.random.choice(['M', 'F', 'O'], n_records),
            
            # Population data
            "usual_resident_population": np.random.randint(50, 15000, n_records),
            
            # Risk scores (1-10 scale)
            "health_risk_score": np.random.uniform(1.0, 10.0, n_records),
            "access_score": np.random.uniform(1.0, 10.0, n_records),
            
            # Dates
            "service_date": ["2023-01-01"] * n_records,
            "data_extraction_date": ["2023-12-01"] * n_records
        })
        
        # Initialize optimizer
        optimizer = MemoryOptimizer(enable_profiling=True)
        
        # Test optimization
        original_size = realistic_data.estimated_size("mb")
        optimized_df, stats = optimizer.optimize_dataframe_memory(realistic_data, "health")
        optimized_size = optimized_df.estimated_size("mb")
        
        # Should achieve significant memory savings with this data pattern
        assert stats["memory_savings_mb"] > 0
        assert stats["memory_savings_percent"] > 10  # At least 10% savings expected
        
        # Verify Australian health data optimizations
        assert optimized_df["sa2_code"].dtype == pl.Categorical
        assert optimized_df["state_territory"].dtype == pl.Categorical
        assert optimized_df["age_group"].dtype == pl.Categorical
        assert optimized_df["gender"].dtype == pl.Categorical
        
        # SEIFA deciles should be optimized to int8
        for col in ["seifa_irsd_decile", "seifa_irsad_decile", "seifa_ier_decile", "seifa_ieo_decile"]:
            assert optimized_df[col].dtype == pl.Int8
        
        # Health metrics should be appropriately sized
        assert optimized_df["prescription_count"].dtype in [pl.Int8, pl.Int16, pl.Int32]
        assert optimized_df["chronic_conditions"].dtype == pl.Int8  # 0-8 range
        
        # Verify data integrity
        assert optimized_df.shape == realistic_data.shape
        assert set(optimized_df.columns) == set(realistic_data.columns)
        
        # Check that categorical mappings are preserved
        assert optimized_df["sa2_code"].n_unique() == realistic_data["sa2_code"].n_unique()
        assert optimized_df["state_territory"].n_unique() == realistic_data["state_territory"].n_unique()
        
        print(f"✅ Real-world optimization test: {original_size:.2f}MB → {optimized_size:.2f}MB "
              f"({stats['memory_savings_percent']:.1f}% reduction)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])