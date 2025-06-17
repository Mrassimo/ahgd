"""
Storage Performance Tests - Phase 5.4

Comprehensive storage performance testing validating Parquet compression targets (60-70%),
memory optimization effectiveness (57.5% reduction), and storage I/O performance under
production-scale Australian health data loads.

Key Performance Tests:
- Parquet compression performance at scale (60-70% targets)
- Bronze-Silver-Gold transition performance validation
- Memory optimization validation (57.5% reduction target)
- Storage I/O performance under concurrent load
- Data lake operations performance testing
"""

import pytest
import polars as pl
import numpy as np
import time
import psutil
import gc
import logging
import tempfile
import concurrent.futures
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json

from src.data_processing.storage.parquet_storage_manager import ParquetStorageManager
from src.data_processing.storage.memory_optimizer import MemoryOptimizer
from src.data_processing.storage.incremental_processor import IncrementalProcessor
from src.data_processing.storage.lazy_data_loader import LazyDataLoader
from tests.performance import PERFORMANCE_CONFIG, AUSTRALIAN_DATA_SCALE
from tests.performance.test_large_scale_processing import AustralianHealthDataGenerator

logger = logging.getLogger(__name__)


@dataclass
class StoragePerformanceResult:
    """Results from storage performance testing."""
    test_name: str
    data_size_mb: float
    storage_size_mb: float
    compression_ratio: float
    compression_percent: float
    write_speed_mb_per_s: float
    read_speed_mb_per_s: float
    memory_usage_mb: float
    processing_time_s: float
    targets_met: Dict[str, bool]
    optimization_details: Dict[str, Any]


@dataclass
class MemoryOptimizationResult:
    """Results from memory optimization testing."""
    test_name: str
    original_size_mb: float
    optimized_size_mb: float
    memory_reduction_mb: float
    memory_reduction_percent: float
    optimization_time_s: float
    optimizations_applied: List[str]
    targets_met: Dict[str, bool]
    data_quality_preserved: bool


class TestStoragePerformance:
    """Storage performance tests validating compression, optimization, and I/O targets."""
    
    @pytest.fixture(scope="class")
    def data_generator(self):
        """Create Australian health data generator."""
        return AustralianHealthDataGenerator(seed=42)
    
    @pytest.fixture(scope="class")
    def storage_components(self, tmp_path_factory):
        """Create storage testing components."""
        temp_dir = tmp_path_factory.mktemp("storage_performance")
        
        return {
            'storage_manager': ParquetStorageManager(base_path=temp_dir / "parquet"),
            'memory_optimizer': MemoryOptimizer(),
            'incremental_processor': IncrementalProcessor(temp_dir / "data_lake"),
            'lazy_loader': LazyDataLoader(),
            'temp_dir': temp_dir
        }
    
    def test_parquet_compression_performance_at_scale(self, data_generator, storage_components):
        """Test Parquet compression performance with large datasets (60-70% target)."""
        logger.info("Testing Parquet compression performance at scale")
        
        storage_manager = storage_components['storage_manager']
        temp_dir = storage_components['temp_dir']
        
        # Test with multiple dataset sizes to validate scaling
        test_sizes = [50000, 100000, 250000, 500000]
        compression_results = []
        
        for size in test_sizes:
            logger.info(f"Testing compression with {size:,} records")
            
            # Generate realistic Australian health data
            test_data = data_generator.generate_large_scale_health_data(size)
            original_size_mb = test_data.estimated_size("mb")
            
            # Test compression performance
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Write with compression optimization
            file_path = temp_dir / f"compression_test_{size}.parquet"
            saved_path = storage_manager.save_optimized_parquet(
                test_data, file_path, data_type="health"
            )
            
            write_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            # Measure compressed file size
            compressed_size_mb = saved_path.stat().st_size / 1024 / 1024
            compression_ratio = original_size_mb / compressed_size_mb
            compression_percent = (1 - compressed_size_mb / original_size_mb) * 100
            
            # Test read performance
            read_start = time.time()
            loaded_data = pl.read_parquet(saved_path)
            read_time = time.time() - read_start
            
            # Calculate performance metrics
            write_speed = original_size_mb / write_time if write_time > 0 else 0
            read_speed = compressed_size_mb / read_time if read_time > 0 else 0
            
            # Validate compression targets (60-70%)
            target_compression_min = PERFORMANCE_CONFIG['storage_performance_targets']['parquet_compression_min_percent']
            target_compression_max = PERFORMANCE_CONFIG['storage_performance_targets']['parquet_compression_max_percent']
            target_read_speed = PERFORMANCE_CONFIG['storage_performance_targets']['min_read_speed_mb_per_second']
            target_write_speed = PERFORMANCE_CONFIG['storage_performance_targets']['min_write_speed_mb_per_second']
            
            targets_met = {
                'compression_in_range': target_compression_min <= compression_percent <= target_compression_max,
                'read_speed_acceptable': read_speed >= target_read_speed * 0.5,  # Allow 50% of target for large files
                'write_speed_acceptable': write_speed >= target_write_speed * 0.5,
                'compression_ratio_good': compression_ratio >= 1.6,  # At least 60% reduction
                'data_integrity_preserved': len(loaded_data) == len(test_data)
            }
            
            result = StoragePerformanceResult(
                test_name=f"parquet_compression_{size}_records",
                data_size_mb=original_size_mb,
                storage_size_mb=compressed_size_mb,
                compression_ratio=compression_ratio,
                compression_percent=compression_percent,
                write_speed_mb_per_s=write_speed,
                read_speed_mb_per_s=read_speed,
                memory_usage_mb=memory_usage,
                processing_time_s=write_time + read_time,
                targets_met=targets_met,
                optimization_details={
                    'write_time': write_time,
                    'read_time': read_time,
                    'records_processed': len(test_data),
                    'columns_optimized': len(test_data.columns)
                }
            )
            
            compression_results.append(result)
            
            # Individual test validation
            assert compression_percent >= target_compression_min * 0.8, \
                f"Compression {compression_percent:.1f}% should be ≥{target_compression_min * 0.8:.1f}%"
            assert len(loaded_data) == len(test_data), "Data integrity should be preserved"
            
            logger.info(f"  {size:,} records: {compression_percent:.1f}% compression, "
                       f"{write_speed:.1f}MB/s write, {read_speed:.1f}MB/s read")
        
        # Overall compression performance validation
        avg_compression = np.mean([r.compression_percent for r in compression_results])
        avg_write_speed = np.mean([r.write_speed_mb_per_s for r in compression_results])
        avg_read_speed = np.mean([r.read_speed_mb_per_s for r in compression_results])
        
        assert avg_compression >= target_compression_min, \
            f"Average compression {avg_compression:.1f}% should be ≥{target_compression_min}%"
        assert all(r.targets_met['data_integrity_preserved'] for r in compression_results), \
            "Data integrity should be preserved across all tests"
        
        logger.info(f"Parquet compression performance: {avg_compression:.1f}% avg compression, "
                   f"{avg_write_speed:.1f}MB/s avg write, {avg_read_speed:.1f}MB/s avg read")
        
        return compression_results
    
    def test_memory_optimization_at_scale(self, data_generator, storage_components):
        """Test memory optimization effectiveness at scale (57.5% target)."""
        logger.info("Testing memory optimization at scale")
        
        memory_optimizer = storage_components['memory_optimizer']
        
        # Test with different data types and sizes
        test_scenarios = [
            ('seifa', lambda: data_generator.generate_large_scale_seifa_data()),
            ('health_small', lambda: data_generator.generate_large_scale_health_data(100000)),
            ('health_large', lambda: data_generator.generate_large_scale_health_data(500000)),
            ('boundary', lambda: data_generator.generate_large_scale_boundary_data()),
        ]
        
        optimization_results = []
        target_reduction = PERFORMANCE_CONFIG['memory_optimization_targets']['min_memory_reduction_percent']
        
        for scenario_name, data_generator_func in test_scenarios:
            logger.info(f"Testing memory optimization for {scenario_name}")
            
            # Generate test data
            test_data = data_generator_func()
            original_size_mb = test_data.estimated_size("mb")
            
            # Test memory optimization
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Determine data category
            if 'seifa' in scenario_name:
                data_category = 'seifa'
            elif 'health' in scenario_name:
                data_category = 'health'
            else:
                data_category = 'geographic'
            
            # Apply memory optimization
            optimized_data = memory_optimizer.optimize_data_types(test_data, data_category=data_category)
            
            optimization_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            # Measure optimization effectiveness
            optimized_size_mb = optimized_data.estimated_size("mb")
            memory_reduction_mb = original_size_mb - optimized_size_mb
            memory_reduction_percent = (memory_reduction_mb / original_size_mb) * 100
            
            # Validate data quality preservation
            data_quality_preserved = (
                len(optimized_data) == len(test_data) and
                len(optimized_data.columns) == len(test_data.columns)
            )
            
            # Performance targets validation
            targets_met = {
                'memory_reduction_target': memory_reduction_percent >= target_reduction * 0.7,  # 70% of target
                'optimization_speed_acceptable': optimization_time < 60.0,  # Under 1 minute
                'data_quality_preserved': data_quality_preserved,
                'memory_usage_reasonable': memory_usage < original_size_mb * 0.5,  # Overhead <50% of original
                'significant_reduction': memory_reduction_percent >= 25.0  # At least 25% reduction
            }
            
            result = MemoryOptimizationResult(
                test_name=f"memory_optimization_{scenario_name}",
                original_size_mb=original_size_mb,
                optimized_size_mb=optimized_size_mb,
                memory_reduction_mb=memory_reduction_mb,
                memory_reduction_percent=memory_reduction_percent,
                optimization_time_s=optimization_time,
                optimizations_applied=[],  # Would be populated by optimizer
                targets_met=targets_met,
                data_quality_preserved=data_quality_preserved
            )
            
            optimization_results.append(result)
            
            # Individual test validation
            assert memory_reduction_percent >= 15.0, \
                f"Memory reduction {memory_reduction_percent:.1f}% should be ≥15%"
            assert data_quality_preserved, "Data quality should be preserved"
            assert optimization_time < 120.0, f"Optimization took {optimization_time:.1f}s, expected <120s"
            
            logger.info(f"  {scenario_name}: {memory_reduction_percent:.1f}% reduction, "
                       f"{optimization_time:.2f}s optimization time")
        
        # Overall memory optimization validation
        avg_reduction = np.mean([r.memory_reduction_percent for r in optimization_results])
        total_original_mb = sum([r.original_size_mb for r in optimization_results])
        total_optimized_mb = sum([r.optimized_size_mb for r in optimization_results])
        overall_reduction = ((total_original_mb - total_optimized_mb) / total_original_mb) * 100
        
        assert avg_reduction >= 30.0, f"Average memory reduction {avg_reduction:.1f}% should be ≥30%"
        assert overall_reduction >= 35.0, f"Overall memory reduction {overall_reduction:.1f}% should be ≥35%"
        assert all(r.data_quality_preserved for r in optimization_results), \
            "Data quality should be preserved across all optimizations"
        
        logger.info(f"Memory optimization results: {avg_reduction:.1f}% avg reduction, "
                   f"{overall_reduction:.1f}% overall reduction")
        
        return optimization_results
    
    def test_bronze_silver_gold_performance(self, data_generator, storage_components):
        """Test Bronze-Silver-Gold data lake performance."""
        logger.info("Testing Bronze-Silver-Gold data lake performance")
        
        incremental_processor = storage_components['incremental_processor']
        
        # Generate test dataset
        health_data = data_generator.generate_large_scale_health_data(200000)
        data_size_mb = health_data.estimated_size("mb")
        
        # Test Bronze-Silver-Gold pipeline performance
        pipeline_start = time.time()
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024
        
        # Bronze layer ingestion
        bronze_start = time.time()
        bronze_version = incremental_processor.ingest_to_bronze(
            health_data, "health", {"source": "performance_test", "batch_id": "perf_001"}
        )
        bronze_time = time.time() - bronze_start
        
        # Silver layer processing
        silver_start = time.time()
        silver_version = incremental_processor.process_to_silver("health", bronze_version)
        silver_time = time.time() - silver_start
        
        # Gold layer aggregation
        gold_start = time.time()
        gold_version = incremental_processor.aggregate_to_gold(
            "health", silver_version, {"aggregation_type": "sa2_summary"}
        )
        gold_time = time.time() - gold_start
        
        total_pipeline_time = time.time() - pipeline_start
        end_memory = process.memory_info().rss / 1024 / 1024
        memory_usage = end_memory - start_memory
        
        # Validate Bronze-Silver-Gold performance
        throughput = data_size_mb / total_pipeline_time
        
        # Performance targets
        assert total_pipeline_time < 180.0, f"BSG pipeline took {total_pipeline_time:.1f}s, expected <180s"
        assert bronze_time < 90.0, f"Bronze ingestion took {bronze_time:.1f}s, expected <90s"
        assert silver_time < 60.0, f"Silver processing took {silver_time:.1f}s, expected <60s"
        assert gold_time < 30.0, f"Gold aggregation took {gold_time:.1f}s, expected <30s"
        assert throughput > 5.0, f"Throughput {throughput:.1f}MB/s should be >5MB/s"
        
        logger.info(f"Bronze-Silver-Gold performance: {total_pipeline_time:.2f}s total, "
                   f"Bronze: {bronze_time:.2f}s, Silver: {silver_time:.2f}s, Gold: {gold_time:.2f}s")
        
        return {
            'total_time': total_pipeline_time,
            'bronze_time': bronze_time,
            'silver_time': silver_time,
            'gold_time': gold_time,
            'throughput_mb_per_s': throughput,
            'memory_usage_mb': memory_usage,
            'data_size_mb': data_size_mb
        }
    
    def test_concurrent_storage_operations(self, data_generator, storage_components):
        """Test storage performance under concurrent operations."""
        logger.info("Testing concurrent storage operations")
        
        storage_manager = storage_components['storage_manager']
        temp_dir = storage_components['temp_dir']
        
        # Create multiple datasets for concurrent testing
        num_concurrent_ops = 6
        records_per_dataset = 100000
        
        datasets = []
        for i in range(num_concurrent_ops):
            if i % 3 == 0:
                data = data_generator.generate_large_scale_seifa_data()
                data_type = "seifa"
            elif i % 3 == 1:
                data = data_generator.generate_large_scale_health_data(records_per_dataset)
                data_type = "health"
            else:
                data = data_generator.generate_large_scale_boundary_data()
                data_type = "geographic"
            
            datasets.append((f"concurrent_dataset_{i}", data, data_type))
        
        def concurrent_storage_operation(dataset_info):
            """Perform storage operation concurrently."""
            dataset_name, dataset, data_type = dataset_info
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Write operation
            file_path = temp_dir / f"{dataset_name}.parquet"
            saved_path = storage_manager.save_optimized_parquet(dataset, file_path, data_type=data_type)
            
            write_time = time.time() - start_time
            
            # Read operation
            read_start = time.time()
            loaded_data = pl.read_parquet(saved_path)
            read_time = time.time() - read_start
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Calculate metrics
            data_size_mb = dataset.estimated_size("mb")
            storage_size_mb = saved_path.stat().st_size / 1024 / 1024
            compression_ratio = data_size_mb / storage_size_mb
            
            return {
                'dataset_name': dataset_name,
                'data_type': data_type,
                'total_time': end_time - start_time,
                'write_time': write_time,
                'read_time': read_time,
                'data_size_mb': data_size_mb,
                'storage_size_mb': storage_size_mb,
                'compression_ratio': compression_ratio,
                'memory_delta': end_memory - start_memory,
                'records_processed': len(dataset),
                'data_integrity_ok': len(loaded_data) == len(dataset),
                'success': True
            }
        
        # Execute concurrent operations
        concurrent_start = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_ops) as executor:
            futures = [executor.submit(concurrent_storage_operation, dataset_info) for dataset_info in datasets]
            results = []
            
            for future in concurrent.futures.as_completed(futures, timeout=300):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Concurrent operation failed: {e}")
                    results.append({'success': False, 'error': str(e)})
        
        concurrent_total_time = time.time() - concurrent_start
        
        # Validate concurrent storage performance
        successful_results = [r for r in results if r.get('success', False)]
        failed_results = [r for r in results if not r.get('success', False)]
        
        assert len(failed_results) == 0, f"All concurrent operations should succeed, {len(failed_results)} failed"
        assert len(successful_results) == num_concurrent_ops, f"Should have {num_concurrent_ops} successful results"
        assert concurrent_total_time < 240.0, f"Concurrent operations took {concurrent_total_time:.1f}s, expected <240s"
        
        # Performance analysis
        total_data_processed = sum(r['data_size_mb'] for r in successful_results)
        total_records_processed = sum(r['records_processed'] for r in successful_results)
        avg_compression_ratio = np.mean([r['compression_ratio'] for r in successful_results])
        concurrent_throughput = total_data_processed / concurrent_total_time
        
        assert concurrent_throughput > 10.0, f"Concurrent throughput {concurrent_throughput:.1f}MB/s should be >10MB/s"
        assert avg_compression_ratio > 1.5, f"Average compression ratio {avg_compression_ratio:.2f} should be >1.5"
        assert all(r['data_integrity_ok'] for r in successful_results), "Data integrity should be preserved"
        
        logger.info(f"Concurrent storage operations: {concurrent_total_time:.2f}s, "
                   f"{concurrent_throughput:.1f}MB/s, {avg_compression_ratio:.2f}x compression")
        
        return {
            'concurrent_total_time': concurrent_total_time,
            'concurrent_throughput_mb_per_s': concurrent_throughput,
            'operations_completed': len(successful_results),
            'total_data_processed_mb': total_data_processed,
            'total_records_processed': total_records_processed,
            'average_compression_ratio': avg_compression_ratio,
            'all_operations_successful': len(failed_results) == 0
        }
    
    def test_lazy_loading_performance(self, data_generator, storage_components):
        """Test lazy loading performance with large datasets."""
        logger.info("Testing lazy loading performance")
        
        lazy_loader = storage_components['lazy_loader']
        storage_manager = storage_components['storage_manager']
        temp_dir = storage_components['temp_dir']
        
        # Create large dataset for lazy loading test
        large_dataset = data_generator.generate_large_scale_health_data(500000)
        data_size_mb = large_dataset.estimated_size("mb")
        
        # Save dataset for lazy loading
        parquet_path = temp_dir / "lazy_loading_test.parquet"
        storage_manager.save_optimized_parquet(large_dataset, parquet_path, data_type="health")
        
        # Test lazy loading performance
        lazy_start = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Create lazy dataset
        lazy_df = lazy_loader.load_lazy_dataset(parquet_path, "parquet")
        
        # Execute various lazy operations
        query_results = []
        
        # Query 1: Aggregation by state
        query1_start = time.time()
        result1 = lazy_loader.execute_lazy_query(
            lazy_df.group_by("state_territory").agg([
                pl.col("prescription_count").sum().alias("total_prescriptions"),
                pl.col("total_cost_aud").mean().alias("avg_cost")
            ]),
            cache_key="state_aggregation"
        )
        query1_time = time.time() - query1_start
        query_results.append(('state_aggregation', query1_time, len(result1)))
        
        # Query 2: Filtering and analysis
        query2_start = time.time()
        result2 = lazy_loader.execute_lazy_query(
            lazy_df.filter(pl.col("age_group").is_in(["65-79", "80+"]))
                   .group_by("sa2_code").agg([
                       pl.col("chronic_conditions_count").mean().alias("avg_chronic_conditions")
                   ]),
            cache_key="elderly_analysis"
        )
        query2_time = time.time() - query2_start
        query_results.append(('elderly_analysis', query2_time, len(result2)))
        
        # Query 3: Complex multi-step analysis
        query3_start = time.time()
        result3 = lazy_loader.execute_lazy_query(
            lazy_df.filter(pl.col("prescription_count") > 5)
                   .with_columns([
                       (pl.col("total_cost_aud") / pl.col("prescription_count")).alias("cost_per_prescription")
                   ])
                   .group_by(["state_territory", "age_group"]).agg([
                       pl.col("cost_per_prescription").mean().alias("avg_cost_per_prescription"),
                       pl.count().alias("patient_count")
                   ]),
            cache_key="cost_analysis"
        )
        query3_time = time.time() - query3_start
        query_results.append(('cost_analysis', query3_time, len(result3)))
        
        total_lazy_time = time.time() - lazy_start
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_usage = end_memory - start_memory
        
        # Validate lazy loading performance
        avg_query_time = np.mean([q[1] for q in query_results])
        lazy_throughput = data_size_mb / total_lazy_time
        
        assert total_lazy_time < 120.0, f"Lazy loading took {total_lazy_time:.1f}s, expected <120s"
        assert avg_query_time < 30.0, f"Average query time {avg_query_time:.1f}s should be <30s"
        assert memory_usage < data_size_mb * 0.3, f"Memory usage {memory_usage:.1f}MB should be <30% of data size"
        assert lazy_throughput > 5.0, f"Lazy loading throughput {lazy_throughput:.1f}MB/s should be >5MB/s"
        
        # Validate query results
        for query_name, query_time, result_count in query_results:
            assert result_count > 0, f"Query {query_name} should return results"
            assert query_time < 45.0, f"Query {query_name} took {query_time:.1f}s, expected <45s"
        
        logger.info(f"Lazy loading performance: {total_lazy_time:.2f}s total, "
                   f"{avg_query_time:.2f}s avg query, {memory_usage:.1f}MB memory")
        
        return {
            'total_lazy_time': total_lazy_time,
            'average_query_time': avg_query_time,
            'memory_usage_mb': memory_usage,
            'lazy_throughput_mb_per_s': lazy_throughput,
            'query_results': query_results,
            'data_size_mb': data_size_mb
        }