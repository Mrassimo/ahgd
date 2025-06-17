"""
Comprehensive unit tests for Memory Optimizer.

Tests memory optimization strategies for Australian health data processing:
- Memory usage monitoring and profiling
- Data type optimization for reduced memory footprint
- Chunked processing for large datasets
- Garbage collection and memory cleanup
- Performance benchmarks for memory efficiency
- Memory leak detection and prevention

Validates memory reduction targets and processing efficiency.
"""

import pytest
import polars as pl
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import time
import gc
import psutil
import os
from datetime import datetime, date, timedelta

from src.data_processing.storage.memory_optimizer import MemoryOptimizer


class TestMemoryOptimizer:
    """Comprehensive test suite for memory optimizer."""
    
    def test_memory_optimizer_initialization(self):
        """Test memory optimizer initializes with proper configuration."""
        optimizer = MemoryOptimizer()
        
        assert hasattr(optimizer, 'memory_threshold')
        assert hasattr(optimizer, 'chunk_size')
        assert hasattr(optimizer, 'optimization_strategies')
        
        # Check default thresholds are reasonable
        assert 0 < optimizer.memory_threshold < 10000  # MB
        assert 1000 <= optimizer.chunk_size <= 100000  # Records
    
    def test_memory_optimizer_custom_config(self):
        """Test memory optimizer with custom configuration."""
        custom_config = {
            'memory_threshold': 1000,  # MB
            'chunk_size': 5000,        # Records
            'aggressive_optimization': True
        }
        
        optimizer = MemoryOptimizer(**custom_config)
        
        assert optimizer.memory_threshold == 1000
        assert optimizer.chunk_size == 5000
        assert optimizer.aggressive_optimization is True
    
    def test_optimize_data_types_seifa(self, mock_seifa_data, memory_profiler):
        """Test data type optimization for SEIFA data."""
        optimizer = MemoryOptimizer()
        
        # Create SEIFA data with suboptimal types
        seifa_df = mock_seifa_data(num_areas=1000)
        
        # Ensure some columns are using inefficient types
        seifa_df = seifa_df.with_columns([
            pl.col("irsd_decile").cast(pl.Int64),  # Should be Int8
            pl.col("irsad_decile").cast(pl.Int64), # Should be Int8
            pl.col("sa2_code_2021").cast(pl.Utf8)  # Should be Categorical
        ])
        
        memory_profiler.start()
        initial_memory = seifa_df.estimated_size("mb")
        
        # Optimize data types
        optimized_df = optimizer.optimize_data_types(seifa_df, data_category="seifa")
        
        final_memory = optimized_df.estimated_size("mb")
        memory_reduction = (initial_memory - final_memory) / initial_memory
        
        # Verify memory reduction
        assert memory_reduction > 0, "Memory optimization should reduce memory usage"
        
        # Verify data types were optimized
        if "irsd_decile" in optimized_df.columns:
            assert optimized_df["irsd_decile"].dtype == pl.Int8
        if "sa2_code_2021" in optimized_df.columns:
            assert optimized_df["sa2_code_2021"].dtype == pl.Categorical
        
        # Verify data integrity
        assert len(optimized_df) == len(seifa_df)
    
    def test_optimize_data_types_health(self, mock_health_data, memory_profiler):
        """Test data type optimization for health data."""
        optimizer = MemoryOptimizer()
        
        # Create health data with suboptimal types
        health_df = mock_health_data(num_records=2000, num_sa2_areas=100)
        
        # Force inefficient types
        health_df = health_df.with_columns([
            pl.col("prescription_count").cast(pl.Int64),  # Should be Int32
            pl.col("chronic_medication").cast(pl.Int64),  # Should be Int8
            pl.col("sa2_code").cast(pl.Utf8),             # Should be Categorical
            pl.col("state").cast(pl.Utf8)                 # Should be Categorical
        ])
        
        initial_memory = health_df.estimated_size("mb")
        
        # Optimize data types
        optimized_df = optimizer.optimize_data_types(health_df, data_category="health")
        
        final_memory = optimized_df.estimated_size("mb")
        memory_reduction = (initial_memory - final_memory) / initial_memory
        
        # Verify memory reduction
        assert memory_reduction > 0
        
        # Verify specific optimizations
        if "prescription_count" in optimized_df.columns:
            assert optimized_df["prescription_count"].dtype in [pl.Int32, pl.Int16]
        if "chronic_medication" in optimized_df.columns:
            assert optimized_df["chronic_medication"].dtype == pl.Int8
        if "sa2_code" in optimized_df.columns:
            assert optimized_df["sa2_code"].dtype == pl.Categorical
        if "state" in optimized_df.columns:
            assert optimized_df["state"].dtype == pl.Categorical
    
    def test_chunked_processing_large_dataset(self, mock_health_data, memory_profiler, performance_benchmarks):
        """Test chunked processing for memory-efficient handling of large datasets."""
        optimizer = MemoryOptimizer(chunk_size=1000)
        
        # Create large dataset
        large_df = mock_health_data(num_records=10000, num_sa2_areas=500)
        
        memory_profiler.start()
        initial_memory = memory_profiler.get_current_usage()
        
        # Process in chunks
        processed_chunks = []
        for chunk_df in optimizer.process_in_chunks(large_df):
            # Simulate processing on each chunk
            processed_chunk = optimizer.optimize_data_types(chunk_df, data_category="health")
            processed_chunks.append(processed_chunk)
            
            # Monitor memory usage during processing
            current_memory = memory_profiler.get_current_usage()
            memory_increase = current_memory - initial_memory
            
            # Memory increase should be limited during chunked processing
            max_increase = performance_benchmarks["memory_optimization"]["max_memory_increase"]
            assert memory_increase < (initial_memory * max_increase), \
                f"Memory increase {memory_increase}MB exceeds threshold"
        
        # Recombine chunks
        final_df = pl.concat(processed_chunks)
        
        # Verify processing integrity
        assert len(final_df) == len(large_df)
        
        # Verify memory efficiency
        peak_memory = memory_profiler.get_peak_usage()
        memory_efficiency = (peak_memory - initial_memory) / initial_memory
        
        assert memory_efficiency < 2.0, "Memory usage should not double during processing"
    
    def test_memory_monitoring_alerts(self, mock_seifa_data):
        """Test memory monitoring and alert system."""
        # Set low memory threshold for testing
        optimizer = MemoryOptimizer(memory_threshold=100)  # Very low threshold
        
        large_df = mock_seifa_data(num_areas=5000)
        
        # Monitor should detect high memory usage
        with patch.object(optimizer, '_get_current_memory_usage', return_value=150):
            memory_alert = optimizer.check_memory_threshold()
            assert memory_alert is True, "Should alert when memory exceeds threshold"
        
        # Monitor should be fine with low memory usage
        with patch.object(optimizer, '_get_current_memory_usage', return_value=50):
            memory_alert = optimizer.check_memory_threshold()
            assert memory_alert is False, "Should not alert when memory is below threshold"
    
    def test_garbage_collection_strategies(self, mock_health_data, memory_profiler):
        """Test garbage collection and memory cleanup strategies."""
        optimizer = MemoryOptimizer()
        
        memory_profiler.start()
        initial_memory = memory_profiler.get_current_usage()
        
        # Create multiple large datasets
        datasets = []
        for i in range(5):
            df = mock_health_data(num_records=2000, num_sa2_areas=100)
            datasets.append(df)
        
        peak_memory = memory_profiler.get_current_usage()
        
        # Trigger cleanup
        datasets.clear()  # Remove references
        optimizer.trigger_garbage_collection()
        
        # Allow time for cleanup
        time.sleep(0.1)
        
        final_memory = memory_profiler.get_current_usage()
        memory_freed = peak_memory - final_memory
        
        # Should have freed some memory
        assert memory_freed > 0, "Garbage collection should free memory"
        
        # Final memory should be closer to initial
        memory_recovery = (peak_memory - final_memory) / (peak_memory - initial_memory)
        assert memory_recovery > 0.3, "Should recover at least 30% of allocated memory"
    
    def test_string_memory_optimization(self, mock_boundary_data):
        """Test string column memory optimization."""
        optimizer = MemoryOptimizer()
        
        # Create data with repetitive string patterns (good for categorical conversion)
        boundary_df = mock_boundary_data(num_areas=1000)
        
        # Add columns with different string patterns
        boundary_df = boundary_df.with_columns([
            # High repetition - good for categorical
            pl.Series("state_abbrev", np.random.choice(["NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT"], 1000)),
            
            # Medium repetition - might benefit from categorical
            pl.Series("region_type", np.random.choice(["Urban", "Regional", "Remote"], 1000)),
            
            # Low repetition - might not benefit
            pl.Series("unique_ids", [f"ID_{i:06d}" for i in range(1000)])
        ])
        
        initial_memory = boundary_df.estimated_size("mb")
        
        # Optimize string columns
        optimized_df = optimizer.optimize_string_columns(boundary_df)
        
        final_memory = optimized_df.estimated_size("mb")
        memory_reduction = (initial_memory - final_memory) / initial_memory
        
        # Should achieve some memory reduction
        assert memory_reduction >= 0, "String optimization should not increase memory"
        
        # High repetition columns should be categorical
        if "state_abbrev" in optimized_df.columns:
            assert optimized_df["state_abbrev"].dtype == pl.Categorical
        
        if "region_type" in optimized_df.columns:
            assert optimized_df["region_type"].dtype == pl.Categorical
    
    def test_memory_profiling_detailed(self, mock_seifa_data, memory_profiler):
        """Test detailed memory profiling capabilities."""
        optimizer = MemoryOptimizer()
        
        # Start profiling
        optimizer.start_memory_profiling()
        
        # Perform various operations
        df1 = mock_seifa_data(num_areas=500)
        profile_1 = optimizer.get_memory_profile("after_data_creation")
        
        df2 = optimizer.optimize_data_types(df1, data_category="seifa")
        profile_2 = optimizer.get_memory_profile("after_optimization")
        
        df3 = df2.clone()
        profile_3 = optimizer.get_memory_profile("after_cloning")
        
        # Get complete profile
        complete_profile = optimizer.get_complete_memory_profile()
        
        # Verify profiling captured operations
        assert "after_data_creation" in complete_profile
        assert "after_optimization" in complete_profile
        assert "after_cloning" in complete_profile
        
        # Verify memory tracking
        for stage, profile in complete_profile.items():
            assert "memory_mb" in profile
            assert "timestamp" in profile
            assert profile["memory_mb"] > 0
    
    def test_memory_leak_detection(self, mock_health_data, memory_profiler):
        """Test memory leak detection during repeated operations."""
        optimizer = MemoryOptimizer()
        
        memory_profiler.start()
        baseline_memory = memory_profiler.get_current_usage()
        
        # Perform repeated operations that should not leak memory
        for i in range(10):
            df = mock_health_data(num_records=500, num_sa2_areas=50)
            optimized_df = optimizer.optimize_data_types(df, data_category="health")
            
            # Force cleanup
            del df, optimized_df
            
            if i % 3 == 0:  # Periodic garbage collection
                optimizer.trigger_garbage_collection()
        
        final_memory = memory_profiler.get_current_usage()
        memory_growth = final_memory - baseline_memory
        
        # Memory growth should be minimal (allow for some overhead)
        assert memory_growth < 100, f"Potential memory leak detected: {memory_growth}MB growth"
    
    def test_optimize_for_specific_operations(self, mock_seifa_data):
        """Test optimization for specific operation patterns."""
        optimizer = MemoryOptimizer()
        
        seifa_df = mock_seifa_data(num_areas=1000)
        
        # Optimize for aggregation operations
        agg_optimized = optimizer.optimize_for_operation(seifa_df, operation="aggregation")
        
        # Should preserve numeric columns as efficient types
        numeric_columns = [col for col in agg_optimized.columns if "score" in col]
        for col in numeric_columns:
            if col in agg_optimized.columns:
                # Should use efficient numeric types for aggregation
                assert agg_optimized[col].dtype in [pl.Int32, pl.Int16, pl.Float32, pl.Float64]
        
        # Optimize for filtering operations
        filter_optimized = optimizer.optimize_for_operation(seifa_df, operation="filtering")
        
        # Should optimize categorical columns for filtering
        if "sa2_code_2021" in filter_optimized.columns:
            assert filter_optimized["sa2_code_2021"].dtype == pl.Categorical
    
    def test_memory_budget_management(self, mock_health_data, memory_profiler, performance_benchmarks):
        """Test memory budget management for constrained environments."""
        # Set strict memory budget
        memory_budget_mb = 200
        optimizer = MemoryOptimizer(memory_budget=memory_budget_mb)
        
        memory_profiler.start()
        
        # Try to process dataset that might exceed budget
        large_df = mock_health_data(num_records=5000, num_sa2_areas=200)
        
        # Process with budget constraints
        result_df = optimizer.process_with_budget_constraint(large_df, data_category="health")
        
        # Verify processing completed
        assert isinstance(result_df, pl.DataFrame)
        assert len(result_df) > 0
        
        # Check memory usage stayed within budget (with some tolerance)
        peak_memory = memory_profiler.get_peak_usage()
        memory_increase = memory_profiler.get_increase()
        
        # Should respect memory budget
        tolerance = 1.2  # Allow 20% tolerance
        assert memory_increase < (memory_budget_mb * tolerance), \
            f"Memory usage {memory_increase}MB exceeded budget {memory_budget_mb}MB"
    
    def test_lazy_evaluation_optimization(self, mock_seifa_data):
        """Test lazy evaluation strategies for memory efficiency."""
        optimizer = MemoryOptimizer()
        
        seifa_df = mock_seifa_data(num_areas=2000)
        
        # Convert to lazy frame for efficient operations
        lazy_df = optimizer.create_lazy_optimized_frame(seifa_df, data_category="seifa")
        
        assert isinstance(lazy_df, pl.LazyFrame)
        
        # Perform lazy operations
        lazy_filtered = lazy_df.filter(pl.col("irsd_decile") <= 5)
        lazy_selected = lazy_filtered.select(["sa2_code_2021", "irsd_decile", "irsd_score"])
        
        # Collect only when needed
        result_df = lazy_selected.collect()
        
        # Verify lazy operations worked
        assert isinstance(result_df, pl.DataFrame)
        assert len(result_df.columns) == 3
        
        # All IRSD deciles should be <= 5
        if len(result_df) > 0:
            max_decile = result_df["irsd_decile"].max()
            assert max_decile <= 5
    
    def test_memory_optimization_benchmarks(self, mock_health_data, memory_profiler, performance_benchmarks):
        """Test memory optimization meets performance benchmarks."""
        optimizer = MemoryOptimizer()
        
        # Create test dataset
        test_df = mock_health_data(num_records=3000, num_sa2_areas=150)
        
        memory_profiler.start()
        initial_memory = test_df.estimated_size("mb")
        
        # Apply optimization
        optimized_df = optimizer.optimize_data_types(test_df, data_category="health")
        final_memory = optimized_df.estimated_size("mb")
        
        # Calculate reduction
        memory_reduction_ratio = (initial_memory - final_memory) / initial_memory
        
        # Verify meets benchmark
        min_reduction = performance_benchmarks["memory_optimization"]["min_memory_reduction"]
        assert memory_reduction_ratio >= min_reduction, \
            f"Memory reduction {memory_reduction_ratio:.2f} below target {min_reduction}"
        
        # Verify data integrity maintained
        assert len(optimized_df) == len(test_df)
        
        # Verify processing time is reasonable
        start_time = time.time()
        _ = optimizer.optimize_data_types(test_df, data_category="health")
        optimization_time = time.time() - start_time
        
        assert optimization_time < 10.0, "Optimization should complete within 10 seconds"
    
    def test_concurrent_memory_optimization(self, mock_seifa_data, mock_health_data):
        """Test thread safety of memory optimization operations."""
        import concurrent.futures
        
        optimizer = MemoryOptimizer()
        
        def optimize_dataset(dataset_type, data_size):
            """Optimize a dataset in a separate thread."""
            if dataset_type == "seifa":
                df = mock_seifa_data(num_areas=data_size)
                return optimizer.optimize_data_types(df, data_category="seifa")
            else:
                df = mock_health_data(num_records=data_size*2, num_sa2_areas=data_size//2)
                return optimizer.optimize_data_types(df, data_category="health")
        
        # Run multiple optimizations concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(optimize_dataset, "seifa", 200),
                executor.submit(optimize_dataset, "health", 300),
                executor.submit(optimize_dataset, "seifa", 150),
                executor.submit(optimize_dataset, "health", 250)
            ]
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Verify all operations succeeded
        assert len(results) == 4
        
        for result_df in results:
            assert isinstance(result_df, pl.DataFrame)
            assert len(result_df) > 0
    
    def test_adaptive_optimization_strategies(self, mock_boundary_data):
        """Test adaptive optimization based on data characteristics."""
        optimizer = MemoryOptimizer()
        
        # Create datasets with different characteristics
        
        # High cardinality dataset
        high_cardinality_df = mock_boundary_data(num_areas=1000)
        high_cardinality_df = high_cardinality_df.with_columns(
            pl.Series("unique_ids", [f"ID_{i:08d}" for i in range(1000)])
        )
        
        # Low cardinality dataset
        low_cardinality_df = mock_boundary_data(num_areas=1000)
        low_cardinality_df = low_cardinality_df.with_columns(
            pl.Series("categories", np.random.choice(["A", "B", "C"], 1000))
        )
        
        # Adaptive optimization should choose different strategies
        high_card_optimized = optimizer.adaptive_optimization(high_cardinality_df)
        low_card_optimized = optimizer.adaptive_optimization(low_cardinality_df)
        
        # Verify different optimization strategies were applied
        assert isinstance(high_card_optimized, pl.DataFrame)
        assert isinstance(low_card_optimized, pl.DataFrame)
        
        # Low cardinality column should be categorical
        if "categories" in low_card_optimized.columns:
            assert low_card_optimized["categories"].dtype == pl.Categorical
        
        # High cardinality column might remain string
        if "unique_ids" in high_card_optimized.columns:
            # Could be string or categorical depending on optimization strategy
            assert high_card_optimized["unique_ids"].dtype in [pl.Utf8, pl.Categorical]


class TestMemoryOptimizerConfiguration:
    """Test memory optimizer configuration and strategies."""
    
    def test_default_optimization_strategies(self):
        """Test default optimization strategies are comprehensive."""
        optimizer = MemoryOptimizer()
        
        strategies = optimizer.optimization_strategies
        
        required_strategies = [
            "categorical_conversion", "numeric_downcasting", 
            "string_optimization", "null_handling"
        ]
        
        for strategy in required_strategies:
            assert strategy in strategies or any(strategy in str(s) for s in strategies)
    
    def test_memory_thresholds_reasonable(self):
        """Test memory thresholds are set to reasonable values."""
        optimizer = MemoryOptimizer()
        
        # Memory threshold should be reasonable for typical systems
        assert 100 <= optimizer.memory_threshold <= 32000  # 100MB to 32GB
        
        # Chunk size should be reasonable for processing
        assert 100 <= optimizer.chunk_size <= 1000000  # 100 to 1M records
    
    def test_data_type_mappings_complete(self):
        """Test data type optimization mappings are complete."""
        optimizer = MemoryOptimizer()
        
        # Should have mappings for Australian health data patterns
        seifa_mappings = optimizer._get_optimization_mappings("seifa")
        health_mappings = optimizer._get_optimization_mappings("health")
        
        assert isinstance(seifa_mappings, dict)
        assert isinstance(health_mappings, dict)
        
        # Should include key Australian patterns
        assert any("sa2" in str(key).lower() for key in seifa_mappings.keys())
        assert any("decile" in str(key).lower() for key in seifa_mappings.keys())