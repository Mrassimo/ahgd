"""
Memory usage performance tests for the AHGD ETL pipeline.

This module contains tests for:
- Memory leak detection
- Memory usage patterns
- Memory efficiency optimization
- Garbage collection impact
- Large dataset memory handling
"""

import gc
import pytest
import psutil
import time
import tracemalloc
from typing import Dict, List, Any
import pandas as pd
import numpy as np

from src.performance.profiler import MemoryProfiler
from src.performance.optimisation import MemoryOptimiser
from src.utils.logging import get_logger

logger = get_logger()


class TestMemoryUsage:
    """Test suite for memory usage analysis and optimization."""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.memory_profiler = MemoryProfiler()
        cls.memory_optimiser = MemoryOptimiser()
        cls.process = psutil.Process()
        
        # Start tracemalloc for detailed memory tracking
        if not tracemalloc.is_tracing():
            tracemalloc.start()
    
    @classmethod
    def teardown_class(cls):
        """Clean up test environment."""
        if tracemalloc.is_tracing():
            tracemalloc.stop()
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated operations."""
        initial_memory = self.process.memory_info().rss
        memory_measurements = []
        
        # Perform repeated operations that should not leak memory
        for i in range(10):
            with self.memory_profiler.profile_memory(f"leak_test_iteration_{i}"):
                # Create and process data
                data = self._create_test_dataframe(1000)
                processed_data = self._process_dataframe(data)
                
                # Explicitly delete references
                del data
                del processed_data
                
                # Force garbage collection
                gc.collect()
                
                # Measure memory
                current_memory = self.process.memory_info().rss
                memory_measurements.append(current_memory)
        
        # Analyse memory trend
        memory_growth = memory_measurements[-1] - memory_measurements[0]
        memory_growth_mb = memory_growth / 1024 / 1024
        
        # Check for memory leaks
        assert memory_growth_mb < 50, f"Potential memory leak detected: {memory_growth_mb:.2f} MB growth"
        
        # Check that memory doesn't continuously grow
        last_three = memory_measurements[-3:]
        if len(last_three) == 3:
            trend = last_three[-1] - last_three[0]
            trend_mb = trend / 1024 / 1024
            assert trend_mb < 10, f"Memory continues to grow: {trend_mb:.2f} MB in last 3 iterations"
        
        logger.info("Memory leak test completed",
                   initial_memory_mb=initial_memory / 1024 / 1024,
                   final_memory_mb=memory_measurements[-1] / 1024 / 1024,
                   total_growth_mb=memory_growth_mb)
    
    def test_large_dataset_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        data_sizes = [10000, 50000, 100000]
        memory_results = []
        
        for size in data_sizes:
            with self.memory_profiler.profile_memory(f"large_dataset_{size}"):
                initial_memory = self.process.memory_info().rss
                
                # Create large dataset
                large_data = self._create_test_dataframe(size)
                
                # Measure memory after creation
                after_creation_memory = self.process.memory_info().rss
                
                # Process the data
                processed_data = self._process_dataframe(large_data)
                
                # Measure peak memory
                peak_memory = self.process.memory_info().rss
                
                # Clean up
                del large_data
                del processed_data
                gc.collect()
                
                # Measure memory after cleanup
                after_cleanup_memory = self.process.memory_info().rss
                
                # Calculate memory metrics
                creation_memory_mb = (after_creation_memory - initial_memory) / 1024 / 1024
                peak_memory_mb = (peak_memory - initial_memory) / 1024 / 1024
                cleanup_efficiency = (peak_memory - after_cleanup_memory) / (peak_memory - initial_memory)
                
                memory_results.append({
                    'data_size': size,
                    'creation_memory_mb': creation_memory_mb,
                    'peak_memory_mb': peak_memory_mb,
                    'cleanup_efficiency': cleanup_efficiency,
                    'memory_per_record_kb': (creation_memory_mb * 1024) / size
                })
                
                # Memory efficiency assertions
                memory_per_record_kb = (creation_memory_mb * 1024) / size
                assert memory_per_record_kb < 10, f"Memory per record too high: {memory_per_record_kb:.2f} KB"
                assert cleanup_efficiency > 0.8, f"Poor memory cleanup: {cleanup_efficiency:.2f} efficiency"
        
        # Analyse memory scaling
        self._analyse_memory_scaling(memory_results)
        logger.info("Large dataset memory test completed", results=memory_results)
    
    def test_streaming_vs_batch_memory_usage(self):
        """Compare memory usage between streaming and batch processing."""
        data_size = 50000
        
        # Test batch processing
        with self.memory_profiler.profile_memory("batch_processing"):
            initial_memory = self.process.memory_info().rss
            
            # Load all data at once
            all_data = self._create_test_dataframe(data_size)
            batch_peak_memory = self.process.memory_info().rss
            
            # Process all data
            processed_data = self._process_dataframe(all_data)
            batch_processing_memory = self.process.memory_info().rss
            
            del all_data
            del processed_data
            gc.collect()
        
        # Test streaming processing
        with self.memory_profiler.profile_memory("streaming_processing"):
            streaming_initial_memory = self.process.memory_info().rss
            streaming_peak_memory = streaming_initial_memory
            
            # Process data in chunks
            chunk_size = 5000
            for i in range(0, data_size, chunk_size):
                chunk_data = self._create_test_dataframe(min(chunk_size, data_size - i))
                processed_chunk = self._process_dataframe(chunk_data)
                
                current_memory = self.process.memory_info().rss
                streaming_peak_memory = max(streaming_peak_memory, current_memory)
                
                del chunk_data
                del processed_chunk
                gc.collect()
        
        # Compare memory usage
        batch_memory_mb = (batch_processing_memory - initial_memory) / 1024 / 1024
        streaming_memory_mb = (streaming_peak_memory - streaming_initial_memory) / 1024 / 1024
        memory_savings = (batch_memory_mb - streaming_memory_mb) / batch_memory_mb
        
        # Memory efficiency assertions
        assert streaming_memory_mb < batch_memory_mb, "Streaming should use less memory than batch"
        assert memory_savings > 0.5, f"Streaming memory savings insufficient: {memory_savings:.2%}"
        
        logger.info("Streaming vs batch memory comparison completed",
                   batch_memory_mb=batch_memory_mb,
                   streaming_memory_mb=streaming_memory_mb,
                   memory_savings=memory_savings)
    
    def test_garbage_collection_impact(self):
        """Test the impact of garbage collection on performance."""
        data_size = 10000
        gc_results = []
        
        # Test with different GC settings
        original_threshold = gc.get_threshold()
        
        gc_configurations = [
            (700, 10, 10),    # Default-like
            (1000, 15, 15),   # Less frequent GC
            (500, 5, 5),      # More frequent GC
        ]
        
        for threshold in gc_configurations:
            gc.set_threshold(*threshold)
            
            # Measure performance with this GC configuration
            start_time = time.time()
            initial_memory = self.process.memory_info().rss
            gc_count_before = sum(gc.get_stats())
            
            # Perform memory-intensive operations
            for i in range(20):
                data = self._create_test_dataframe(data_size)
                processed = self._process_dataframe(data)
                del data
                del processed
            
            gc.collect()  # Final cleanup
            
            end_time = time.time()
            final_memory = self.process.memory_info().rss
            gc_count_after = sum(gc.get_stats())
            
            execution_time = end_time - start_time
            memory_growth = (final_memory - initial_memory) / 1024 / 1024
            gc_collections = gc_count_after - gc_count_before
            
            gc_results.append({
                'threshold': threshold,
                'execution_time': execution_time,
                'memory_growth_mb': memory_growth,
                'gc_collections': gc_collections,
                'throughput': (20 * data_size) / execution_time
            })
        
        # Restore original GC threshold
        gc.set_threshold(*original_threshold)
        
        # Analyse GC impact
        best_config = min(gc_results, key=lambda x: x['execution_time'])
        worst_config = max(gc_results, key=lambda x: x['execution_time'])
        performance_difference = (worst_config['execution_time'] - best_config['execution_time']) / best_config['execution_time']
        
        # GC impact assertions
        assert performance_difference < 0.5, f"GC configuration impact too high: {performance_difference:.2%}"
        
        logger.info("Garbage collection impact test completed", 
                   gc_results=gc_results,
                   performance_difference=performance_difference)
    
    def test_memory_profiling_accuracy(self):
        """Test the accuracy of memory profiling tools."""
        data_size = 5000
        
        with self.memory_profiler.profile_memory("profiling_accuracy_test"):
            # Get baseline memory
            baseline_memory = self.process.memory_info().rss
            
            # Create data with known memory footprint
            # DataFrame with known structure
            test_data = pd.DataFrame({
                'integers': np.arange(data_size, dtype=np.int64),      # 8 bytes per int
                'floats': np.random.random(data_size).astype(np.float64),  # 8 bytes per float
                'strings': [f'test_string_{i:06d}' for i in range(data_size)]  # Variable size
            })
            
            # Measure memory after creation
            after_creation_memory = self.process.memory_info().rss
            measured_memory_mb = (after_creation_memory - baseline_memory) / 1024 / 1024
            
            # Calculate expected memory usage (rough estimate)
            expected_memory_mb = (
                (data_size * 8 * 2) +  # integers and floats
                (data_size * 20)       # approximate string overhead
            ) / 1024 / 1024
            
            # Get profiler measurements
            profiler_summary = self.memory_profiler.stop_profiling()
            profiler_memory_mb = profiler_summary.get('peak_memory_mb', 0)
            
            # Clean up
            del test_data
            gc.collect()
            
            # Accuracy checks (allow for overhead and measurement differences)
            measurement_accuracy = abs(measured_memory_mb - expected_memory_mb) / expected_memory_mb
            assert measurement_accuracy < 0.5, f"Memory measurement inaccurate: {measurement_accuracy:.2%} error"
            
            logger.info("Memory profiling accuracy test completed",
                       expected_memory_mb=expected_memory_mb,
                       measured_memory_mb=measured_memory_mb,
                       profiler_memory_mb=profiler_memory_mb,
                       measurement_accuracy=measurement_accuracy)
    
    def test_memory_optimisation_recommendations(self):
        """Test memory optimisation recommendation generation."""
        # Create inefficient data structures for testing
        inefficient_data = {
            'large_list': list(range(100000)),
            'dict_with_duplicates': {f'key_{i % 1000}': f'value_{i}' for i in range(10000)},
            'unused_dataframe': pd.DataFrame(np.random.random((10000, 50)))
        }
        
        # Analyse memory usage
        analysis = self.memory_optimiser.analyse_memory_usage("optimisation_test")
        
        # Generate recommendations
        data_info = {
            'type': 'mixed_structures',
            'size': 100000,
            'contains_duplicates': True
        }
        
        recommendations = self.memory_optimiser.suggest_data_structure_optimisations(data_info)
        
        # Verify recommendations are generated
        assert len(recommendations) > 0, "No memory optimisation recommendations generated"
        assert any('memory' in rec.lower() for rec in recommendations), "No memory-specific recommendations"
        
        # Clean up
        del inefficient_data
        gc.collect()
        
        logger.info("Memory optimisation recommendations test completed",
                   recommendations=recommendations,
                   analysis_summary=analysis)
    
    @pytest.mark.slow
    def test_memory_stress_test(self):
        """Stress test memory usage under extreme conditions."""
        stress_results = []
        
        # Gradually increase memory pressure
        for multiplier in [1, 2, 4, 8]:
            base_size = 10000 * multiplier
            
            try:
                with self.memory_profiler.profile_memory(f"stress_test_{multiplier}x"):
                    initial_memory = self.process.memory_info().rss
                    
                    # Create multiple large datasets simultaneously
                    datasets = []
                    for i in range(3):
                        data = self._create_test_dataframe(base_size)
                        datasets.append(data)
                    
                    # Process all datasets
                    processed_datasets = []
                    for data in datasets:
                        processed = self._process_dataframe(data)
                        processed_datasets.append(processed)
                    
                    peak_memory = self.process.memory_info().rss
                    memory_usage_mb = (peak_memory - initial_memory) / 1024 / 1024
                    
                    # Clean up
                    del datasets
                    del processed_datasets
                    gc.collect()
                    
                    after_cleanup_memory = self.process.memory_info().rss
                    cleanup_effectiveness = (peak_memory - after_cleanup_memory) / (peak_memory - initial_memory)
                    
                    stress_results.append({
                        'multiplier': multiplier,
                        'base_size': base_size,
                        'peak_memory_mb': memory_usage_mb,
                        'cleanup_effectiveness': cleanup_effectiveness,
                        'success': True
                    })
                    
                    # Check memory limits
                    assert memory_usage_mb < 2000, f"Memory usage too high: {memory_usage_mb:.2f} MB"
                    assert cleanup_effectiveness > 0.7, f"Poor cleanup: {cleanup_effectiveness:.2f}"
                    
            except MemoryError:
                stress_results.append({
                    'multiplier': multiplier,
                    'base_size': base_size,
                    'success': False,
                    'error': 'MemoryError'
                })
                break
            
            except Exception as e:
                stress_results.append({
                    'multiplier': multiplier,
                    'base_size': base_size,
                    'success': False,
                    'error': str(e)
                })
        
        # Analyse stress test results
        successful_tests = [r for r in stress_results if r['success']]
        assert len(successful_tests) >= 2, "Too few successful stress tests"
        
        logger.info("Memory stress test completed", results=stress_results)
    
    # Helper methods
    def _create_test_dataframe(self, size: int) -> pd.DataFrame:
        """Create a test DataFrame with known memory characteristics."""
        return pd.DataFrame({
            'id': np.arange(size, dtype=np.int64),
            'value': np.random.random(size).astype(np.float64),
            'category': np.random.choice(['A', 'B', 'C', 'D'], size),
            'text': [f'text_data_{i:06d}' for i in range(size)],
            'timestamp': pd.date_range('2023-01-01', periods=size, freq='1H'),
            'coordinates': [(np.random.random(), np.random.random()) for _ in range(size)]
        })
    
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process DataFrame to simulate real ETL operations."""
        # Create a copy to simulate transformation
        result = df.copy()
        
        # Perform various operations that use memory
        result['value_squared'] = result['value'] ** 2
        result['category_encoded'] = pd.Categorical(result['category']).codes
        result['text_length'] = result['text'].str.len()
        
        # Group operations
        grouped = result.groupby('category').agg({
            'value': ['mean', 'std', 'count'],
            'value_squared': 'sum'
        })
        
        # Merge back some aggregated data
        result = result.merge(
            grouped['value']['mean'].reset_index().rename(columns={'mean': 'category_mean'}),
            on='category',
            how='left'
        )
        
        return result
    
    def _analyse_memory_scaling(self, results: List[Dict[str, Any]]):
        """Analyse how memory usage scales with data size."""
        if len(results) < 3:
            return
        
        data_sizes = [r['data_size'] for r in results]
        memory_usages = [r['creation_memory_mb'] for r in results]
        
        # Calculate scaling factor
        size_ratio = data_sizes[-1] / data_sizes[0]
        memory_ratio = memory_usages[-1] / memory_usages[0]
        scaling_factor = memory_ratio / size_ratio
        
        # Check for reasonable scaling (should be close to linear)
        assert 0.8 <= scaling_factor <= 1.5, f"Poor memory scaling: factor {scaling_factor:.2f}"
        
        logger.info("Memory scaling analysis completed",
                   size_ratio=size_ratio,
                   memory_ratio=memory_ratio,
                   scaling_factor=scaling_factor)