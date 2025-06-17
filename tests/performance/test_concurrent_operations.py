"""
Concurrent Operations Performance Tests - Phase 5.4

Tests the platform's ability to handle concurrent processing, multi-user scenarios,
and parallel data operations with realistic Australian health data loads. Validates
thread scaling, resource contention handling, and performance under concurrent access.

Key Performance Tests:
- Multi-threaded data processing scalability
- Concurrent user access simulation
- Resource contention and lock performance
- Thread pool optimization and scaling
- Parallel I/O operations performance
- Cross-component concurrent integration
"""

import pytest
import polars as pl
import numpy as np
import time
import psutil
import gc
import logging
import threading
import asyncio
import queue
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from contextlib import contextmanager

from src.data_processing.seifa_processor import SEIFAProcessor
from src.data_processing.health_processor import HealthDataProcessor
from src.data_processing.simple_boundary_processor import SimpleBoundaryProcessor
from src.data_processing.storage.parquet_storage_manager import ParquetStorageManager
from src.data_processing.storage.memory_optimizer import MemoryOptimizer
from src.data_processing.storage.incremental_processor import IncrementalProcessor
from src.analysis.risk.health_risk_calculator import HealthRiskCalculator
from tests.performance import PERFORMANCE_CONFIG, AUSTRALIAN_DATA_SCALE
from tests.performance.test_large_scale_processing import AustralianHealthDataGenerator

logger = logging.getLogger(__name__)


@dataclass
class ConcurrentPerformanceResult:
    """Results from concurrent operations performance testing."""
    test_name: str
    concurrent_operations: int
    total_time_seconds: float
    average_operation_time: float
    throughput_operations_per_second: float
    resource_utilization: Dict[str, float]
    success_rate: float
    scaling_efficiency: float
    targets_met: Dict[str, bool]
    operation_details: List[Dict[str, Any]]


@dataclass
class ThreadScalingResult:
    """Results from thread scaling performance tests."""
    thread_counts: List[int]
    throughput_per_thread_count: List[float]
    efficiency_per_thread_count: List[float]
    optimal_thread_count: int
    linear_scaling_achieved: bool
    scaling_efficiency_score: float


class ConcurrentOperationManager:
    """Manages concurrent operations and resource monitoring."""
    
    def __init__(self):
        self.operation_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.operation_lock = threading.Lock()
        self.resource_monitor = {}
        self.active_operations = 0
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager for monitoring individual operations."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_cpu = psutil.Process().cpu_percent()
        
        with self.operation_lock:
            self.active_operations += 1
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            end_cpu = psutil.Process().cpu_percent()
            
            with self.operation_lock:
                self.active_operations -= 1
                self.resource_monitor[operation_name] = {
                    'duration': end_time - start_time,
                    'memory_delta': end_memory - start_memory,
                    'cpu_delta': end_cpu - start_cpu,
                    'timestamp': datetime.now().isoformat()
                }
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization metrics."""
        process = psutil.Process()
        return {
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'memory_percent': process.memory_percent(),
            'active_operations': self.active_operations,
            'thread_count': threading.active_count()
        }


class TestConcurrentOperations:
    """Concurrent operations performance tests for Australian Health Analytics platform."""
    
    @pytest.fixture(scope="class")
    def data_generator(self):
        """Create Australian health data generator."""
        return AustralianHealthDataGenerator(seed=42)
    
    @pytest.fixture(scope="class")
    def concurrent_processors(self, tmp_path_factory):
        """Create processors for concurrent testing."""
        temp_dir = tmp_path_factory.mktemp("concurrent_test")
        
        return {
            'seifa_processor': SEIFAProcessor(data_dir=temp_dir),
            'health_processor': HealthDataProcessor(data_dir=temp_dir),
            'boundary_processor': SimpleBoundaryProcessor(data_dir=temp_dir),
            'storage_manager': ParquetStorageManager(base_path=temp_dir / "parquet"),
            'memory_optimizer': MemoryOptimizer(),
            'incremental_processor': IncrementalProcessor(temp_dir / "lake"),
            'risk_calculator': HealthRiskCalculator(data_dir=temp_dir / "processed"),
            'temp_dir': temp_dir
        }
    
    @pytest.fixture(scope="class")
    def operation_manager(self):
        """Create concurrent operation manager."""
        return ConcurrentOperationManager()
    
    def test_thread_scaling_performance(self, data_generator, concurrent_processors, operation_manager):
        """Test thread scaling performance and efficiency."""
        logger.info("Testing thread scaling performance")
        
        health_processor = concurrent_processors['health_processor']
        memory_optimizer = concurrent_processors['memory_optimizer']
        
        # Test with different thread counts
        thread_counts = [1, 2, 4, 8, 12, 16]
        scaling_results = []
        
        # Generate test datasets
        num_datasets = 16
        dataset_size = 50000
        test_datasets = [
            data_generator.generate_large_scale_health_data(dataset_size) 
            for _ in range(num_datasets)
        ]
        
        def process_dataset_task(dataset_id_and_data):
            """Process a single dataset (for thread scaling test)."""
            dataset_id, dataset = dataset_id_and_data
            task_name = f"thread_scaling_task_{dataset_id}"
            
            with operation_manager.monitor_operation(task_name):
                # Full processing pipeline
                validated = health_processor._validate_health_data(dataset)
                aggregated = health_processor._aggregate_by_sa2(validated)
                optimized = memory_optimizer.optimize_data_types(aggregated, data_category="health")
                
                return {
                    'dataset_id': dataset_id,
                    'input_records': len(dataset),
                    'output_records': len(optimized),
                    'success': True
                }
        
        for thread_count in thread_counts:
            logger.info(f"Testing with {thread_count} threads")
            
            gc.collect()  # Clean up before test
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Execute tasks with specified thread count
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                dataset_tasks = [(i, dataset) for i, dataset in enumerate(test_datasets[:thread_count*2])]
                
                futures = [executor.submit(process_dataset_task, task) for task in dataset_tasks]
                results = []
                
                for future in concurrent.futures.as_completed(futures, timeout=300):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Task failed with {thread_count} threads: {e}")
                        results.append({'success': False, 'error': str(e)})
            
            total_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            # Calculate performance metrics
            successful_results = [r for r in results if r.get('success', False)]
            success_rate = len(successful_results) / len(results)
            total_records_processed = sum(r['input_records'] for r in successful_results)
            throughput = total_records_processed / total_time
            
            # Calculate scaling efficiency (vs single thread baseline)
            if thread_count == 1:
                baseline_throughput = throughput
                efficiency = 1.0
            else:
                efficiency = throughput / (baseline_throughput * thread_count)
            
            scaling_results.append({
                'thread_count': thread_count,
                'total_time': total_time,
                'throughput': throughput,
                'efficiency': efficiency,
                'success_rate': success_rate,
                'memory_usage_mb': memory_usage,
                'records_processed': total_records_processed,
                'tasks_completed': len(successful_results)
            })
            
            # Validate thread scaling performance
            assert success_rate >= 0.95, f"Success rate {success_rate:.1%} should be ≥95% with {thread_count} threads"
            assert total_time < 180, f"Processing with {thread_count} threads took {total_time:.1f}s, expected <180s"
        
        # Analyze thread scaling efficiency
        throughputs = [r['throughput'] for r in scaling_results]
        efficiencies = [r['efficiency'] for r in scaling_results]
        
        # Find optimal thread count (best throughput)
        optimal_thread_count = scaling_results[np.argmax(throughputs)]['thread_count']
        
        # Check for linear scaling (efficiency should be reasonable)
        linear_scaling_achieved = all(eff >= 0.7 for eff in efficiencies[1:4])  # Check first few thread counts
        scaling_efficiency_score = np.mean(efficiencies[1:]) * 10  # Score out of 10
        
        thread_scaling_result = ThreadScalingResult(
            thread_counts=thread_counts,
            throughput_per_thread_count=throughputs,
            efficiency_per_thread_count=efficiencies,
            optimal_thread_count=optimal_thread_count,
            linear_scaling_achieved=linear_scaling_achieved,
            scaling_efficiency_score=scaling_efficiency_score
        )
        
        # Thread scaling validation
        assert optimal_thread_count >= 4, f"Optimal thread count {optimal_thread_count} should be ≥4"
        assert linear_scaling_achieved, "Should achieve reasonable linear scaling for initial thread counts"
        assert scaling_efficiency_score >= 6.0, f"Scaling efficiency score {scaling_efficiency_score:.1f} should be ≥6.0"
        
        logger.info(f"Thread scaling: optimal {optimal_thread_count} threads, "
                   f"efficiency score: {scaling_efficiency_score:.1f}")
        
        return thread_scaling_result
    
    def test_concurrent_data_processing_pipeline(self, data_generator, concurrent_processors, operation_manager):
        """Test concurrent end-to-end data processing pipelines."""
        logger.info("Testing concurrent data processing pipelines")
        
        # Create multiple complete datasets for concurrent processing
        num_pipelines = 8
        datasets_per_pipeline = {
            'seifa': lambda: data_generator.generate_large_scale_seifa_data(),
            'health': lambda: data_generator.generate_large_scale_health_data(100000),
            'boundary': lambda: data_generator.generate_large_scale_boundary_data()
        }
        
        def execute_complete_pipeline(pipeline_id: int):
            """Execute a complete data processing pipeline."""
            pipeline_name = f"concurrent_pipeline_{pipeline_id}"
            
            with operation_manager.monitor_operation(pipeline_name):
                # Generate pipeline-specific data
                seifa_data = datasets_per_pipeline['seifa']()
                health_data = datasets_per_pipeline['health']()
                boundary_data = datasets_per_pipeline['boundary']()
                
                # Get processors
                seifa_processor = concurrent_processors['seifa_processor']
                health_processor = concurrent_processors['health_processor']
                boundary_processor = concurrent_processors['boundary_processor']
                memory_optimizer = concurrent_processors['memory_optimizer']
                storage_manager = concurrent_processors['storage_manager']
                temp_dir = concurrent_processors['temp_dir']
                
                # Execute full pipeline
                # Stage 1: Validation
                validated_seifa = seifa_processor._validate_seifa_data(seifa_data)
                validated_health = health_processor._validate_health_data(health_data)
                validated_boundary = boundary_processor._validate_boundary_data(boundary_data)
                
                # Stage 2: Memory optimization
                optimized_seifa = memory_optimizer.optimize_data_types(validated_seifa, data_category="seifa")
                optimized_health = memory_optimizer.optimize_data_types(validated_health, data_category="health")
                optimized_boundary = memory_optimizer.optimize_data_types(validated_boundary, data_category="geographic")
                
                # Stage 3: Processing and integration
                aggregated_health = health_processor._aggregate_by_sa2(optimized_health)
                enhanced_boundary = boundary_processor._calculate_population_density(optimized_boundary)
                
                # Stage 4: Storage
                seifa_path = storage_manager.save_optimized_parquet(
                    optimized_seifa, temp_dir / f"concurrent_seifa_{pipeline_id}.parquet", data_type="seifa"
                )
                health_path = storage_manager.save_optimized_parquet(
                    aggregated_health, temp_dir / f"concurrent_health_{pipeline_id}.parquet", data_type="health"
                )
                boundary_path = storage_manager.save_optimized_parquet(
                    enhanced_boundary, temp_dir / f"concurrent_boundary_{pipeline_id}.parquet", data_type="geographic"
                )
                
                return {
                    'pipeline_id': pipeline_id,
                    'seifa_records': len(optimized_seifa),
                    'health_records': len(aggregated_health),
                    'boundary_records': len(enhanced_boundary),
                    'files_created': [seifa_path, health_path, boundary_path],
                    'total_records': len(optimized_seifa) + len(aggregated_health) + len(enhanced_boundary),
                    'success': True
                }
        
        # Execute concurrent pipelines
        concurrent_start = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        with ThreadPoolExecutor(max_workers=num_pipelines) as executor:
            futures = [executor.submit(execute_complete_pipeline, i) for i in range(num_pipelines)]
            pipeline_results = []
            
            for future in concurrent.futures.as_completed(futures, timeout=600):
                try:
                    result = future.result()
                    pipeline_results.append(result)
                except Exception as e:
                    logger.error(f"Pipeline failed: {e}")
                    pipeline_results.append({'success': False, 'error': str(e)})
        
        concurrent_total_time = time.time() - concurrent_start
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_usage = end_memory - start_memory
        
        # Analyze concurrent pipeline performance
        successful_pipelines = [r for r in pipeline_results if r.get('success', False)]
        failed_pipelines = [r for r in pipeline_results if not r.get('success', False)]
        
        success_rate = len(successful_pipelines) / num_pipelines
        total_records_processed = sum(r['total_records'] for r in successful_pipelines)
        average_pipeline_time = concurrent_total_time / num_pipelines
        throughput = total_records_processed / concurrent_total_time
        
        # Resource utilization analysis
        resource_utilization = operation_manager.get_resource_utilization()
        
        concurrent_result = ConcurrentPerformanceResult(
            test_name="concurrent_pipeline_processing",
            concurrent_operations=num_pipelines,
            total_time_seconds=concurrent_total_time,
            average_operation_time=average_pipeline_time,
            throughput_operations_per_second=throughput,
            resource_utilization=resource_utilization,
            success_rate=success_rate,
            scaling_efficiency=success_rate * (throughput / 1000),  # Normalize efficiency metric
            targets_met={
                'all_pipelines_successful': len(failed_pipelines) == 0,
                'completion_time_acceptable': concurrent_total_time < 300,
                'throughput_adequate': throughput > 5000,
                'memory_usage_reasonable': memory_usage < 2048,
                'success_rate_high': success_rate >= 0.95
            },
            operation_details=[{'resource_monitor': operation_manager.resource_monitor}]
        )
        
        # Concurrent pipeline validation
        assert len(failed_pipelines) == 0, f"All {num_pipelines} pipelines should succeed, {len(failed_pipelines)} failed"
        assert concurrent_total_time < 300, f"Concurrent pipelines took {concurrent_total_time:.1f}s, expected <300s"
        assert success_rate >= 0.95, f"Success rate {success_rate:.1%} should be ≥95%"
        assert throughput > 5000, f"Throughput {throughput:.0f} records/s should be >5000"
        
        logger.info(f"Concurrent pipelines: {num_pipelines} pipelines, {concurrent_total_time:.1f}s, "
                   f"{success_rate:.1%} success rate, {throughput:.0f} records/s")
        
        return concurrent_result
    
    def test_concurrent_storage_operations(self, data_generator, concurrent_processors, operation_manager):
        """Test concurrent storage I/O operations."""
        logger.info("Testing concurrent storage I/O operations")
        
        storage_manager = concurrent_processors['storage_manager']
        temp_dir = concurrent_processors['temp_dir']
        
        # Create datasets for concurrent I/O testing
        num_concurrent_ops = 12
        storage_datasets = []
        
        for i in range(num_concurrent_ops):
            if i % 3 == 0:
                data = data_generator.generate_large_scale_seifa_data()
                data_type = "seifa"
            elif i % 3 == 1:
                data = data_generator.generate_large_scale_health_data(75000)
                data_type = "health"
            else:
                data = data_generator.generate_large_scale_boundary_data()
                data_type = "geographic"
            
            storage_datasets.append((i, data, data_type))
        
        def concurrent_storage_operation(dataset_info):
            """Perform concurrent storage read/write operations."""
            dataset_id, dataset, data_type = dataset_info
            operation_name = f"storage_op_{dataset_id}"
            
            with operation_manager.monitor_operation(operation_name):
                # Write operation
                write_start = time.time()
                file_path = temp_dir / f"concurrent_storage_{dataset_id}.parquet"
                saved_path = storage_manager.save_optimized_parquet(dataset, file_path, data_type=data_type)
                write_time = time.time() - write_start
                
                # Read operation
                read_start = time.time()
                loaded_data = pl.read_parquet(saved_path)
                read_time = time.time() - read_start
                
                # Verify data integrity
                data_integrity_ok = len(loaded_data) == len(dataset)
                
                # Calculate metrics
                data_size_mb = dataset.estimated_size("mb")
                file_size_mb = saved_path.stat().st_size / 1024 / 1024
                compression_ratio = data_size_mb / file_size_mb
                write_speed = data_size_mb / write_time
                read_speed = file_size_mb / read_time
                
                return {
                    'dataset_id': dataset_id,
                    'data_type': data_type,
                    'write_time': write_time,
                    'read_time': read_time,
                    'total_time': write_time + read_time,
                    'data_size_mb': data_size_mb,
                    'file_size_mb': file_size_mb,
                    'compression_ratio': compression_ratio,
                    'write_speed_mb_s': write_speed,
                    'read_speed_mb_s': read_speed,
                    'data_integrity_ok': data_integrity_ok,
                    'records_processed': len(dataset),
                    'success': True
                }
        
        # Execute concurrent storage operations
        storage_start = time.time()
        
        with ThreadPoolExecutor(max_workers=num_concurrent_ops) as executor:
            futures = [executor.submit(concurrent_storage_operation, dataset_info) for dataset_info in storage_datasets]
            storage_results = []
            
            for future in concurrent.futures.as_completed(futures, timeout=300):
                try:
                    result = future.result()
                    storage_results.append(result)
                except Exception as e:
                    logger.error(f"Storage operation failed: {e}")
                    storage_results.append({'success': False, 'error': str(e)})
        
        storage_total_time = time.time() - storage_start
        
        # Analyze concurrent storage performance
        successful_ops = [r for r in storage_results if r.get('success', False)]
        failed_ops = [r for r in storage_results if not r.get('success', False)]
        
        success_rate = len(successful_ops) / num_concurrent_ops
        total_data_processed = sum(r['data_size_mb'] for r in successful_ops)
        avg_write_speed = np.mean([r['write_speed_mb_s'] for r in successful_ops])
        avg_read_speed = np.mean([r['read_speed_mb_s'] for r in successful_ops])
        avg_compression_ratio = np.mean([r['compression_ratio'] for r in successful_ops])
        storage_throughput = total_data_processed / storage_total_time
        
        # Concurrent storage validation
        assert len(failed_ops) == 0, f"All {num_concurrent_ops} storage operations should succeed"
        assert success_rate >= 0.95, f"Storage success rate {success_rate:.1%} should be ≥95%"
        assert storage_total_time < 180, f"Concurrent storage took {storage_total_time:.1f}s, expected <180s"
        assert avg_write_speed > 20, f"Average write speed {avg_write_speed:.1f}MB/s should be >20MB/s"
        assert avg_read_speed > 50, f"Average read speed {avg_read_speed:.1f}MB/s should be >50MB/s"
        assert all(r['data_integrity_ok'] for r in successful_ops), "Data integrity should be preserved"
        
        logger.info(f"Concurrent storage: {num_concurrent_ops} operations, {storage_total_time:.1f}s, "
                   f"{avg_write_speed:.1f}MB/s write, {avg_read_speed:.1f}MB/s read")
        
        return {
            'total_operations': num_concurrent_ops,
            'success_rate': success_rate,
            'total_time': storage_total_time,
            'average_write_speed_mb_s': avg_write_speed,
            'average_read_speed_mb_s': avg_read_speed,
            'average_compression_ratio': avg_compression_ratio,
            'storage_throughput_mb_s': storage_throughput,
            'data_integrity_preserved': all(r['data_integrity_ok'] for r in successful_ops)
        }
    
    def test_resource_contention_handling(self, data_generator, concurrent_processors, operation_manager):
        """Test resource contention and lock performance."""
        logger.info("Testing resource contention handling")
        
        # Create shared resource access scenario
        memory_optimizer = concurrent_processors['memory_optimizer']
        shared_datasets = [
            data_generator.generate_large_scale_health_data(50000) for _ in range(10)
        ]
        
        # Shared resource for contention testing
        shared_resource_lock = threading.Lock()
        shared_counter = {'value': 0}
        contention_results = []
        
        def contended_operation(operation_id: int):
            """Operation that creates resource contention."""
            operation_name = f"contention_op_{operation_id}"
            
            with operation_manager.monitor_operation(operation_name):
                dataset = shared_datasets[operation_id % len(shared_datasets)]
                
                # Simulate contended resource access
                with shared_resource_lock:
                    shared_counter['value'] += 1
                    current_count = shared_counter['value']
                    time.sleep(0.01)  # Simulate brief locked operation
                
                # Perform actual processing (non-contended)
                optimized_data = memory_optimizer.optimize_data_types(dataset, data_category="health")
                
                # Another contended access
                with shared_resource_lock:
                    shared_counter['value'] += len(optimized_data)
                    final_count = shared_counter['value']
                
                return {
                    'operation_id': operation_id,
                    'start_count': current_count,
                    'final_count': final_count,
                    'records_processed': len(optimized_data),
                    'success': True
                }
        
        # Test resource contention with varying thread counts
        thread_counts = [2, 4, 8, 16]
        contention_test_results = []
        
        for thread_count in thread_counts:
            shared_counter['value'] = 0  # Reset counter
            
            contention_start = time.time()
            
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = [executor.submit(contended_operation, i) for i in range(thread_count * 2)]
                results = []
                
                for future in concurrent.futures.as_completed(futures, timeout=120):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Contended operation failed: {e}")
                        results.append({'success': False, 'error': str(e)})
            
            contention_time = time.time() - contention_start
            
            # Analyze contention results
            successful_ops = [r for r in results if r.get('success', False)]
            success_rate = len(successful_ops) / len(results)
            total_records = sum(r['records_processed'] for r in successful_ops)
            contention_throughput = total_records / contention_time
            
            contention_test_results.append({
                'thread_count': thread_count,
                'total_time': contention_time,
                'success_rate': success_rate,
                'throughput': contention_throughput,
                'operations_completed': len(successful_ops),
                'contention_handled_well': success_rate >= 0.95 and contention_time < 60
            })
        
        # Resource contention validation
        avg_success_rate = np.mean([r['success_rate'] for r in contention_test_results])
        contention_handling_effective = all(r['contention_handled_well'] for r in contention_test_results)
        
        assert avg_success_rate >= 0.95, f"Average success rate {avg_success_rate:.1%} should be ≥95%"
        assert contention_handling_effective, "Should handle resource contention effectively at all thread counts"
        assert shared_counter['value'] > 0, "Shared resource should have been accessed"
        
        logger.info(f"Resource contention: {avg_success_rate:.1%} avg success rate, "
                   f"effective handling: {contention_handling_effective}")
        
        return {
            'average_success_rate': avg_success_rate,
            'contention_handling_effective': contention_handling_effective,
            'thread_test_results': contention_test_results,
            'final_shared_counter': shared_counter['value']
        }
    
    def test_cross_component_concurrent_integration(self, data_generator, concurrent_processors, operation_manager):
        """Test concurrent integration across all platform components."""
        logger.info("Testing cross-component concurrent integration")
        
        # Define integrated workflow scenarios
        workflow_scenarios = [
            'seifa_analysis_workflow',
            'health_risk_assessment_workflow',
            'geographic_analytics_workflow',
            'comprehensive_integration_workflow'
        ]
        
        def execute_integrated_workflow(scenario_name: str, scenario_id: int):
            """Execute integrated workflow scenario."""
            workflow_name = f"{scenario_name}_{scenario_id}"
            
            with operation_manager.monitor_operation(workflow_name):
                if 'seifa' in scenario_name:
                    # SEIFA-focused workflow
                    seifa_data = data_generator.generate_large_scale_seifa_data()
                    seifa_processor = concurrent_processors['seifa_processor']
                    memory_optimizer = concurrent_processors['memory_optimizer']
                    
                    processed_seifa = seifa_processor._validate_seifa_data(seifa_data)
                    optimized_seifa = memory_optimizer.optimize_data_types(processed_seifa, data_category="seifa")
                    
                    result_data = optimized_seifa
                    workflow_type = "seifa"
                
                elif 'health_risk' in scenario_name:
                    # Health risk assessment workflow
                    health_data = data_generator.generate_large_scale_health_data(75000)
                    health_processor = concurrent_processors['health_processor']
                    risk_calculator = concurrent_processors['risk_calculator']
                    
                    validated_health = health_processor._validate_health_data(health_data)
                    aggregated_health = health_processor._aggregate_by_sa2(validated_health)
                    
                    result_data = aggregated_health
                    workflow_type = "health"
                
                elif 'geographic' in scenario_name:
                    # Geographic analytics workflow
                    boundary_data = data_generator.generate_large_scale_boundary_data()
                    boundary_processor = concurrent_processors['boundary_processor']
                    
                    validated_boundaries = boundary_processor._validate_boundary_data(boundary_data)
                    enhanced_boundaries = boundary_processor._calculate_population_density(validated_boundaries)
                    
                    result_data = enhanced_boundaries
                    workflow_type = "geographic"
                
                else:
                    # Comprehensive integration workflow
                    seifa_data = data_generator.generate_large_scale_seifa_data()
                    health_data = data_generator.generate_large_scale_health_data(50000)
                    
                    seifa_processor = concurrent_processors['seifa_processor']
                    health_processor = concurrent_processors['health_processor']
                    memory_optimizer = concurrent_processors['memory_optimizer']
                    
                    # Integrated processing
                    processed_seifa = seifa_processor._validate_seifa_data(seifa_data)
                    validated_health = health_processor._validate_health_data(health_data)
                    
                    optimized_seifa = memory_optimizer.optimize_data_types(processed_seifa, data_category="seifa")
                    aggregated_health = health_processor._aggregate_by_sa2(validated_health)
                    
                    # Integration
                    integrated_data = optimized_seifa.join(
                        aggregated_health, left_on="sa2_code_2021", right_on="sa2_code", how="left"
                    )
                    
                    result_data = integrated_data
                    workflow_type = "integrated"
                
                return {
                    'scenario_name': scenario_name,
                    'scenario_id': scenario_id,
                    'workflow_type': workflow_type,
                    'records_processed': len(result_data),
                    'columns_count': len(result_data.columns),
                    'success': True
                }
        
        # Execute workflows concurrently
        num_concurrent_workflows = 16
        workflow_tasks = []
        
        for i in range(num_concurrent_workflows):
            scenario = workflow_scenarios[i % len(workflow_scenarios)]
            workflow_tasks.append((scenario, i))
        
        integration_start = time.time()
        
        with ThreadPoolExecutor(max_workers=num_concurrent_workflows) as executor:
            futures = [
                executor.submit(execute_integrated_workflow, scenario, scenario_id)
                for scenario, scenario_id in workflow_tasks
            ]
            
            integration_results = []
            for future in concurrent.futures.as_completed(futures, timeout=480):
                try:
                    result = future.result()
                    integration_results.append(result)
                except Exception as e:
                    logger.error(f"Integrated workflow failed: {e}")
                    integration_results.append({'success': False, 'error': str(e)})
        
        integration_total_time = time.time() - integration_start
        
        # Analyze cross-component integration performance
        successful_workflows = [r for r in integration_results if r.get('success', False)]
        failed_workflows = [r for r in integration_results if not r.get('success', False)]
        
        success_rate = len(successful_workflows) / num_concurrent_workflows
        total_records_processed = sum(r['records_processed'] for r in successful_workflows)
        integration_throughput = total_records_processed / integration_total_time
        
        # Workflow type distribution
        workflow_types = {}
        for result in successful_workflows:
            wf_type = result['workflow_type']
            workflow_types[wf_type] = workflow_types.get(wf_type, 0) + 1
        
        # Cross-component integration validation
        assert len(failed_workflows) == 0, f"All {num_concurrent_workflows} workflows should succeed"
        assert success_rate >= 0.95, f"Integration success rate {success_rate:.1%} should be ≥95%"
        assert integration_total_time < 360, f"Integration took {integration_total_time:.1f}s, expected <360s"
        assert integration_throughput > 2000, f"Integration throughput {integration_throughput:.0f} records/s should be >2000"
        assert len(workflow_types) >= 3, "Should successfully execute multiple workflow types"
        
        logger.info(f"Cross-component integration: {num_concurrent_workflows} workflows, "
                   f"{integration_total_time:.1f}s, {success_rate:.1%} success rate")
        
        return {
            'total_workflows': num_concurrent_workflows,
            'success_rate': success_rate,
            'total_time': integration_total_time,
            'integration_throughput': integration_throughput,
            'workflow_type_distribution': workflow_types,
            'total_records_processed': total_records_processed,
            'cross_component_integration_successful': True
        }