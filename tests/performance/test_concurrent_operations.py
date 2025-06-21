"""
Concurrent operations performance tests for the AHGD ETL pipeline.

This module contains tests for:
- Thread safety and performance
- Parallel processing efficiency
- Resource contention
- Deadlock detection
- Scalability under concurrent load
"""

import asyncio
import concurrent.futures
import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Any, Callable
import pandas as pd
import numpy as np

from src.performance.profiler import PerformanceProfiler
from src.performance.monitoring import SystemMonitor, ResourceTracker
from src.utils.logging import get_logger

logger = get_logger()


class TestConcurrentOperations:
    """Test suite for concurrent operations performance."""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.profiler = PerformanceProfiler()
        cls.system_monitor = SystemMonitor(collection_interval=1.0)
        cls.resource_tracker = ResourceTracker()
        
        # Start monitoring
        cls.system_monitor.start_monitoring()
    
    @classmethod
    def teardown_class(cls):
        """Clean up test environment."""
        cls.system_monitor.stop_monitoring()
    
    def test_thread_pool_performance(self):
        """Test performance scaling with different thread pool sizes."""
        data_size = 1000
        pool_sizes = [1, 2, 4, 8]
        results = []
        
        def process_chunk(chunk_data):
            """Process a chunk of data."""
            return self._process_dataframe(chunk_data)
        
        for pool_size in pool_sizes:
            with self.profiler.profile_operation(f"thread_pool_{pool_size}"):
                start_time = time.time()
                
                # Create test data and split into chunks
                test_data = self._create_test_dataframe(data_size)
                chunks = self._split_dataframe(test_data, pool_size)
                
                # Process chunks in parallel
                with ThreadPoolExecutor(max_workers=pool_size) as executor:
                    futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
                    processed_chunks = [future.result() for future in concurrent.futures.as_completed(futures)]
                
                execution_time = time.time() - start_time
                throughput = data_size / execution_time
                
                results.append({
                    'pool_size': pool_size,
                    'execution_time': execution_time,
                    'throughput': throughput,
                    'processed_chunks': len(processed_chunks)
                })
                
                # Performance assertions
                assert len(processed_chunks) == pool_size, f"Lost chunks: expected {pool_size}, got {len(processed_chunks)}"
                
        # Analyse scaling efficiency
        self._analyse_parallel_scaling(results, "thread_pool")
        logger.info("Thread pool performance test completed", results=results)
    
    def test_process_pool_performance(self):
        """Test performance scaling with different process pool sizes."""
        data_size = 5000  # Larger data for process overhead to be worthwhile
        pool_sizes = [1, 2, 4]  # Fewer processes due to overhead
        results = []
        
        for pool_size in pool_sizes:
            with self.profiler.profile_operation(f"process_pool_{pool_size}"):
                start_time = time.time()
                
                # Create test data and split into chunks
                test_data = self._create_test_dataframe(data_size)
                chunks = self._split_dataframe(test_data, pool_size)
                
                # Process chunks in separate processes
                with ProcessPoolExecutor(max_workers=pool_size) as executor:
                    futures = [executor.submit(self._process_dataframe_static, chunk) for chunk in chunks]
                    processed_chunks = [future.result() for future in concurrent.futures.as_completed(futures)]
                
                execution_time = time.time() - start_time
                throughput = data_size / execution_time
                
                results.append({
                    'pool_size': pool_size,
                    'execution_time': execution_time,
                    'throughput': throughput,
                    'processed_chunks': len(processed_chunks)
                })
                
                # Performance assertions
                assert len(processed_chunks) == pool_size, f"Lost chunks: expected {pool_size}, got {len(processed_chunks)}"
        
        # Analyse scaling efficiency
        self._analyse_parallel_scaling(results, "process_pool")
        logger.info("Process pool performance test completed", results=results)
    
    def test_resource_contention(self):
        """Test performance under resource contention."""
        num_threads = 8
        data_size = 2000
        contention_results = []
        
        # Test with shared resource (file I/O)
        def file_io_task(thread_id):
            """Task that performs file I/O."""
            data = self._create_test_dataframe(data_size)
            filename = f"tests/performance/temp/contention_test_{thread_id}.csv"
            
            start_time = time.time()
            data.to_csv(filename, index=False)
            read_data = pd.read_csv(filename)
            execution_time = time.time() - start_time
            
            # Clean up
            import os
            if os.path.exists(filename):
                os.remove(filename)
            
            return {
                'thread_id': thread_id,
                'execution_time': execution_time,
                'records_processed': len(read_data)
            }
        
        # Test memory-intensive task
        def memory_intensive_task(thread_id):
            """Task that uses significant memory."""
            start_time = time.time()
            
            # Create multiple datasets
            datasets = []
            for i in range(5):
                data = self._create_test_dataframe(data_size)
                processed = self._process_dataframe(data)
                datasets.append(processed)
            
            execution_time = time.time() - start_time
            
            # Clean up
            del datasets
            
            return {
                'thread_id': thread_id,
                'execution_time': execution_time,
                'task_type': 'memory_intensive'
            }
        
        # Test file I/O contention
        with self.profiler.profile_operation("file_io_contention"):
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(file_io_task, i) for i in range(num_threads)]
                io_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Test memory contention
        with self.profiler.profile_operation("memory_contention"):
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(memory_intensive_task, i) for i in range(num_threads)]
                memory_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Analyse contention impact
        io_times = [r['execution_time'] for r in io_results]
        memory_times = [r['execution_time'] for r in memory_results]
        
        io_variance = np.var(io_times)
        memory_variance = np.var(memory_times)
        
        # Contention assertions
        assert io_variance < 2.0, f"High I/O contention variance: {io_variance:.2f}"
        assert memory_variance < 5.0, f"High memory contention variance: {memory_variance:.2f}"
        
        logger.info("Resource contention test completed",
                   io_results=io_results,
                   memory_results=memory_results,
                   io_variance=io_variance,
                   memory_variance=memory_variance)
    
    def test_async_operations_performance(self):
        """Test asynchronous operations performance."""
        num_tasks = 10
        data_size = 1000
        
        async def async_data_processing(task_id):
            """Asynchronous data processing task."""
            start_time = time.time()
            
            # Simulate I/O bound operation
            await asyncio.sleep(0.1)  # Simulate network/disk I/O
            
            # CPU bound operation
            data = self._create_test_dataframe(data_size)
            processed = self._process_dataframe(data)
            
            execution_time = time.time() - start_time
            
            return {
                'task_id': task_id,
                'execution_time': execution_time,
                'records_processed': len(processed)
            }
        
        async def run_async_tasks():
            """Run multiple async tasks."""
            tasks = [async_data_processing(i) for i in range(num_tasks)]
            return await asyncio.gather(*tasks)
        
        # Test async performance
        with self.profiler.profile_operation("async_operations"):
            start_time = time.time()
            results = asyncio.run(run_async_tasks())
            total_time = time.time() - start_time
        
        # Compare with synchronous execution
        with self.profiler.profile_operation("sync_operations"):
            sync_start_time = time.time()
            sync_results = []
            for i in range(num_tasks):
                # Simulate synchronous equivalent
                time.sleep(0.1)  # Simulate I/O
                data = self._create_test_dataframe(data_size)
                processed = self._process_dataframe(data)
                sync_results.append({
                    'task_id': i,
                    'records_processed': len(processed)
                })
            sync_total_time = time.time() - sync_start_time
        
        # Performance comparison
        async_efficiency = sync_total_time / total_time
        
        # Async performance assertions
        assert len(results) == num_tasks, f"Lost async tasks: expected {num_tasks}, got {len(results)}"
        assert async_efficiency > 1.5, f"Async efficiency too low: {async_efficiency:.2f}x"
        assert total_time < sync_total_time, "Async should be faster than sync for I/O bound tasks"
        
        logger.info("Async operations performance test completed",
                   async_time=total_time,
                   sync_time=sync_total_time,
                   efficiency=async_efficiency,
                   async_results=results)
    
    def test_thread_safety(self):
        """Test thread safety of operations."""
        num_threads = 8
        iterations_per_thread = 100
        shared_counter = {'value': 0}
        counter_lock = threading.Lock()
        
        def thread_safe_operation(thread_id):
            """Thread-safe operation that modifies shared state."""
            local_results = []
            
            for i in range(iterations_per_thread):
                # Create some data
                data = self._create_small_dataframe(100)
                
                # Thread-safe counter increment
                with counter_lock:
                    shared_counter['value'] += 1
                    current_count = shared_counter['value']
                
                local_results.append({
                    'thread_id': thread_id,
                    'iteration': i,
                    'counter_value': current_count,
                    'data_size': len(data)
                })
            
            return local_results
        
        # Run thread-safe operations
        with self.profiler.profile_operation("thread_safety_test"):
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(thread_safe_operation, i) for i in range(num_threads)]
                all_results = [result for future in concurrent.futures.as_completed(futures) 
                             for result in future.result()]
        
        # Verify thread safety
        expected_count = num_threads * iterations_per_thread
        final_count = shared_counter['value']
        
        assert final_count == expected_count, f"Thread safety violation: expected {expected_count}, got {final_count}"
        assert len(all_results) == expected_count, f"Lost results: expected {expected_count}, got {len(all_results)}"
        
        # Check for race conditions in counter values
        counter_values = [r['counter_value'] for r in all_results]
        unique_values = set(counter_values)
        
        assert len(unique_values) == len(counter_values), "Duplicate counter values indicate race condition"
        
        logger.info("Thread safety test completed",
                   expected_count=expected_count,
                   final_count=final_count,
                   results_count=len(all_results))
    
    def test_deadlock_detection(self):
        """Test for potential deadlocks in concurrent operations."""
        num_threads = 4
        lock1 = threading.Lock()
        lock2 = threading.Lock()
        
        def potential_deadlock_task(thread_id, reverse_order=False):
            """Task that could potentially cause deadlock."""
            try:
                if reverse_order:
                    # Acquire locks in reverse order to create potential deadlock
                    with lock2:
                        time.sleep(0.01)  # Small delay to increase deadlock chance
                        with lock1:
                            data = self._create_small_dataframe(50)
                            time.sleep(0.01)
                            return len(data)
                else:
                    # Normal lock order
                    with lock1:
                        time.sleep(0.01)
                        with lock2:
                            data = self._create_small_dataframe(50)
                            time.sleep(0.01)
                            return len(data)
            except Exception as e:
                logger.error(f"Thread {thread_id} failed", error=str(e))
                raise
        
        # Test with timeout to detect deadlocks
        with self.profiler.profile_operation("deadlock_detection"):
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Submit tasks with different lock orders
                futures = []
                for i in range(num_threads):
                    reverse = i % 2 == 1  # Alternate lock order
                    future = executor.submit(potential_deadlock_task, i, reverse)
                    futures.append(future)
                
                # Wait for completion with timeout
                try:
                    results = []
                    for future in concurrent.futures.as_completed(futures, timeout=10.0):
                        results.append(future.result())
                    
                    execution_time = time.time() - start_time
                    
                    # Deadlock assertions
                    assert len(results) == num_threads, f"Deadlock detected: only {len(results)}/{num_threads} completed"
                    assert execution_time < 5.0, f"Execution too slow, possible deadlock: {execution_time:.2f}s"
                    
                except concurrent.futures.TimeoutError:
                    pytest.fail("Deadlock detected: operations timed out")
        
        logger.info("Deadlock detection test completed", execution_time=execution_time)
    
    def test_resource_tracking_under_load(self):
        """Test resource tracking accuracy under concurrent load."""
        num_operations = 10
        operation_results = []
        
        def tracked_operation(operation_id):
            """Operation with resource tracking."""
            self.resource_tracker.start_operation_tracking(
                f"concurrent_op_{operation_id}", 
                "concurrent_test_operation"
            )
            
            try:
                # Perform resource-intensive operation
                data = self._create_test_dataframe(2000)
                processed = self._process_dataframe(data)
                
                # Simulate additional resource usage
                temp_data = [self._create_small_dataframe(100) for _ in range(10)]
                
                return len(processed)
                
            finally:
                summary = self.resource_tracker.stop_operation_tracking(f"concurrent_op_{operation_id}")
                operation_results.append(summary)
        
        # Run concurrent tracked operations
        with self.profiler.profile_operation("resource_tracking_load"):
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(tracked_operation, i) for i in range(num_operations)]
                processing_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Verify resource tracking
        assert len(operation_results) == num_operations, f"Tracking failed: {len(operation_results)}/{num_operations}"
        
        # Check resource tracking accuracy
        for result in operation_results:
            assert 'memory_growth_mb' in result, "Memory tracking missing"
            assert 'duration_seconds' in result, "Duration tracking missing"
            assert 'resource_efficiency' in result, "Efficiency calculation missing"
            
            # Reasonable bounds checks
            assert result['duration_seconds'] > 0, "Invalid duration"
            assert 0 <= result['resource_efficiency'] <= 100, "Invalid efficiency score"
        
        logger.info("Resource tracking under load test completed",
                   tracked_operations=len(operation_results),
                   processing_results=len(processing_results))
    
    @pytest.mark.slow
    def test_long_running_concurrent_operations(self):
        """Test performance of long-running concurrent operations."""
        duration_seconds = 30  # Run for 30 seconds
        num_workers = 4
        
        def long_running_task(worker_id):
            """Long-running task that processes data continuously."""
            start_time = time.time()
            iterations = 0
            total_records = 0
            
            while time.time() - start_time < duration_seconds:
                data = self._create_test_dataframe(1000)
                processed = self._process_dataframe(data)
                
                iterations += 1
                total_records += len(processed)
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
            
            execution_time = time.time() - start_time
            
            return {
                'worker_id': worker_id,
                'iterations': iterations,
                'total_records': total_records,
                'execution_time': execution_time,
                'throughput': total_records / execution_time
            }
        
        # Run long-running concurrent operations
        with self.profiler.profile_operation("long_running_concurrent"):
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(long_running_task, i) for i in range(num_workers)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Analyse long-running performance
        total_iterations = sum(r['iterations'] for r in results)
        total_records = sum(r['total_records'] for r in results)
        avg_throughput = sum(r['throughput'] for r in results) / len(results)
        
        # Long-running performance assertions
        assert len(results) == num_workers, f"Worker failure: {len(results)}/{num_workers} completed"
        assert total_iterations > 0, "No iterations completed"
        assert avg_throughput > 10, f"Throughput too low: {avg_throughput:.2f} records/s"
        
        # Check performance consistency
        throughputs = [r['throughput'] for r in results]
        throughput_variance = np.var(throughputs)
        throughput_cv = np.std(throughputs) / np.mean(throughputs)
        
        assert throughput_cv < 0.5, f"High throughput variance: {throughput_cv:.2f}"
        
        logger.info("Long-running concurrent operations test completed",
                   total_iterations=total_iterations,
                   total_records=total_records,
                   avg_throughput=avg_throughput,
                   throughput_variance=throughput_variance)
    
    # Helper methods
    def _create_test_dataframe(self, size: int) -> pd.DataFrame:
        """Create test DataFrame."""
        return pd.DataFrame({
            'id': range(size),
            'value': np.random.random(size),
            'category': np.random.choice(['A', 'B', 'C'], size),
            'timestamp': pd.date_range('2023-01-01', periods=size, freq='1H')
        })
    
    def _create_small_dataframe(self, size: int) -> pd.DataFrame:
        """Create small test DataFrame for quick operations."""
        return pd.DataFrame({
            'id': range(size),
            'value': np.random.random(size)
        })
    
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process DataFrame with typical operations."""
        result = df.copy()
        result['value_squared'] = result['value'] ** 2
        result['category_encoded'] = pd.Categorical(result.get('category', 'A')).codes
        return result
    
    @staticmethod
    def _process_dataframe_static(df: pd.DataFrame) -> pd.DataFrame:
        """Static method for process pool executor."""
        result = df.copy()
        result['value_squared'] = result['value'] ** 2
        if 'category' in result.columns:
            result['category_encoded'] = pd.Categorical(result['category']).codes
        return result
    
    def _split_dataframe(self, df: pd.DataFrame, num_chunks: int) -> List[pd.DataFrame]:
        """Split DataFrame into chunks."""
        chunk_size = len(df) // num_chunks
        chunks = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            if i == num_chunks - 1:  # Last chunk gets remaining rows
                end_idx = len(df)
            else:
                end_idx = (i + 1) * chunk_size
            
            chunks.append(df.iloc[start_idx:end_idx].copy())
        
        return chunks
    
    def _analyse_parallel_scaling(self, results: List[Dict[str, Any]], test_type: str):
        """Analyse parallel scaling efficiency."""
        if len(results) < 2:
            return
        
        # Calculate scaling efficiency
        baseline = results[0]  # Single worker performance
        
        for result in results[1:]:
            pool_size = result['pool_size']
            speedup = baseline['execution_time'] / result['execution_time']
            efficiency = speedup / pool_size
            
            result['speedup'] = speedup
            result['efficiency'] = efficiency
            
            # Efficiency should be reasonable (above 50% for thread pools)
            min_efficiency = 0.3 if test_type == "process_pool" else 0.5
            assert efficiency > min_efficiency, f"Poor {test_type} efficiency: {efficiency:.2f} for {pool_size} workers"
        
        logger.info(f"{test_type} scaling analysis completed", 
                   scaling_results=results)