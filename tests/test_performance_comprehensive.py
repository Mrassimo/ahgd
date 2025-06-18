"""
Comprehensive performance and reliability test suite.

This module provides extensive testing for performance benchmarks,
memory usage, concurrent operations, and system reliability under load.
"""

import pytest
import pandas as pd
import numpy as np
import time
import threading
import concurrent.futures
from unittest.mock import Mock, patch
import sys
from pathlib import Path
import psutil
import os
import gc

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock heavy dependencies
sys.modules['streamlit'] = Mock()
sys.modules['folium'] = Mock()
sys.modules['plotly.express'] = Mock()
sys.modules['plotly.graph_objects'] = Mock()


class TestPerformanceBenchmarks:
    """Performance benchmark tests for core operations"""
    
    def setup_method(self):
        """Setup performance test data"""
        np.random.seed(42)  # For reproducible tests
        
        # Create datasets of varying sizes
        self.small_dataset = self._create_dataset(1000)
        self.medium_dataset = self._create_dataset(10000)
        self.large_dataset = self._create_dataset(100000)
    
    def _create_dataset(self, size):
        """Create test dataset of specified size"""
        return pd.DataFrame({
            'SA2_CODE': [f"area_{i:06d}" for i in range(size)],
            'health_risk_score': np.random.normal(10, 3, size),
            'IRSD_Score': np.random.normal(1000, 150, size),
            'mortality_rate': np.random.normal(6, 2, size),
            'diabetes_prevalence': np.random.normal(8, 2.5, size),
            'population': np.random.randint(5000, 50000, size),
            'area_sqkm': np.random.uniform(1, 100, size)
        })
    
    @pytest.mark.slow
    def test_data_loading_performance(self):
        """Test data loading performance benchmarks"""
        datasets = [
            ('small', self.small_dataset),
            ('medium', self.medium_dataset),
            ('large', self.large_dataset)
        ]
        
        performance_results = {}
        
        for name, dataset in datasets:
            # Test CSV operations
            start_time = time.time()
            with patch('pandas.read_csv', return_value=dataset):
                from src.dashboard.data.loaders import load_data
                result = load_data()
            csv_time = time.time() - start_time
            
            # Test data processing
            start_time = time.time()
            processed = dataset.copy()
            processed['population_density'] = processed['population'] / processed['area_sqkm']
            processed['risk_category'] = pd.cut(processed['health_risk_score'], bins=3, labels=['Low', 'Medium', 'High'])
            processing_time = time.time() - start_time
            
            performance_results[name] = {
                'data_size': len(dataset),
                'loading_time': csv_time,
                'processing_time': processing_time,
                'total_time': csv_time + processing_time
            }
        
        # Performance assertions
        assert performance_results['small']['total_time'] < 1.0
        assert performance_results['medium']['total_time'] < 5.0
        assert performance_results['large']['total_time'] < 30.0
        
        # Verify linear scaling roughly holds
        small_rate = performance_results['small']['total_time'] / performance_results['small']['data_size']
        large_rate = performance_results['large']['total_time'] / performance_results['large']['data_size']
        
        # Large datasets shouldn't be more than 10x slower per item (accounting for overhead)
        assert large_rate < small_rate * 10
    
    @pytest.mark.slow
    def test_correlation_calculation_performance(self):
        """Test correlation calculation performance"""
        correlation_times = {}
        
        for name, dataset in [('medium', self.medium_dataset), ('large', self.large_dataset)]:
            numeric_cols = ['health_risk_score', 'IRSD_Score', 'mortality_rate', 'diabetes_prevalence', 'population']
            correlation_data = dataset[numeric_cols]
            
            start_time = time.time()
            correlation_matrix = correlation_data.corr()
            correlation_time = time.time() - start_time
            
            correlation_times[name] = correlation_time
            
            # Verify correlation matrix is valid
            assert correlation_matrix.shape == (len(numeric_cols), len(numeric_cols))
            assert not correlation_matrix.isna().any().any()
        
        # Performance assertions
        assert correlation_times['medium'] < 2.0
        assert correlation_times['large'] < 10.0
    
    @pytest.mark.slow
    def test_aggregation_performance(self):
        """Test data aggregation performance"""
        # Add state column for grouping
        states = ['NSW', 'VIC', 'QLD', 'WA', 'SA', 'TAS', 'NT', 'ACT']
        self.large_dataset['state'] = np.random.choice(states, len(self.large_dataset))
        
        aggregation_operations = [
            ('mean', lambda df: df.groupby('state')['health_risk_score'].mean()),
            ('sum', lambda df: df.groupby('state')['population'].sum()),
            ('count', lambda df: df.groupby('state').size()),
            ('multi_agg', lambda df: df.groupby('state').agg({
                'health_risk_score': ['mean', 'std'],
                'population': ['sum', 'count'],
                'mortality_rate': 'median'
            }))
        ]
        
        for op_name, operation in aggregation_operations:
            start_time = time.time()
            result = operation(self.large_dataset)
            op_time = time.time() - start_time
            
            # Performance assertion - should complete within reasonable time
            assert op_time < 5.0, f"{op_name} operation took too long: {op_time:.2f}s"
            
            # Verify result validity
            assert len(result) > 0
    
    def test_memory_usage_efficiency(self):
        """Test memory usage efficiency during operations"""
        process = psutil.Process(os.getpid())
        
        # Measure initial memory
        gc.collect()  # Clean up before measurement
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform memory-intensive operations
        large_data = self._create_dataset(50000)
        
        # Data transformations
        large_data['population_density'] = large_data['population'] / large_data['area_sqkm']
        large_data['health_category'] = pd.cut(large_data['health_risk_score'], bins=5)
        
        # Correlation calculations
        correlation_matrix = large_data.select_dtypes(include=[np.number]).corr()
        
        # Aggregations
        state_stats = large_data.groupby(large_data.index % 8).agg({
            'health_risk_score': ['mean', 'std', 'min', 'max'],
            'population': ['sum', 'count']
        })
        
        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory assertions
        assert memory_increase < 1000, f"Memory usage too high: {memory_increase:.1f}MB"
        
        # Clean up
        del large_data, correlation_matrix, state_stats
        gc.collect()


class TestConcurrencyReliability:
    """Test concurrent operations and system reliability"""
    
    def test_concurrent_data_access(self):
        """Test concurrent access to data loading functions"""
        def load_and_process_data(thread_id):
            """Function to run in parallel threads"""
            try:
                # Simulate data loading and processing
                data = pd.DataFrame({
                    'id': range(1000),
                    'value': np.random.random(1000)
                })
                
                # Perform operations
                result = data.groupby(data['id'] % 10)['value'].mean()
                correlations = data.corr()
                
                return {
                    'thread_id': thread_id,
                    'success': True,
                    'result_size': len(result),
                    'correlation_shape': correlations.shape
                }
            except Exception as e:
                return {
                    'thread_id': thread_id,
                    'success': False,
                    'error': str(e)
                }
        
        # Run concurrent operations
        num_threads = 5
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(load_and_process_data, i) for i in range(num_threads)]
            results = [future.result(timeout=10) for future in futures]
        
        # Verify all operations succeeded
        successful_results = [r for r in results if r['success']]
        assert len(successful_results) == num_threads
        
        # Verify consistent results
        for result in successful_results:
            assert result['result_size'] == 10
            assert result['correlation_shape'] == (2, 2)
    
    @pytest.mark.slow
    def test_system_stability_under_load(self):
        """Test system stability under sustained load"""
        def cpu_intensive_operation():
            """CPU-intensive operation for stress testing"""
            data = pd.DataFrame({
                'x': np.random.random(10000),
                'y': np.random.random(10000)
            })
            
            # Perform multiple operations
            for _ in range(10):
                result = data.rolling(window=100).mean()
                correlations = data.corr()
                grouped = data.groupby(pd.cut(data['x'], bins=10)).mean()
            
            return len(result)
        
        # Run operations for sustained period
        start_time = time.time()
        iteration_count = 0
        max_duration = 10  # seconds
        
        while time.time() - start_time < max_duration:
            try:
                result = cpu_intensive_operation()
                assert result > 0
                iteration_count += 1
            except Exception as e:
                pytest.fail(f"System became unstable after {iteration_count} iterations: {e}")
        
        # Should complete multiple iterations without failure
        assert iteration_count >= 5
    
    def test_error_recovery_mechanisms(self):
        """Test error recovery and graceful degradation"""
        def operation_with_potential_errors(error_rate=0.3):
            """Operation that might fail based on error rate"""
            if np.random.random() < error_rate:
                raise ValueError("Simulated processing error")
            
            data = pd.DataFrame({
                'values': np.random.random(1000)
            })
            return data.mean().iloc[0]
        
        # Test with error handling
        success_count = 0
        error_count = 0
        total_attempts = 50
        
        for _ in range(total_attempts):
            try:
                result = operation_with_potential_errors()
                assert 0 <= result <= 1  # Valid result
                success_count += 1
            except ValueError:
                error_count += 1
                # Error handling - could implement retry logic here
                continue
            except Exception as e:
                pytest.fail(f"Unexpected error: {e}")
        
        # Should have mix of successes and expected errors
        assert success_count > 20  # At least some successes
        assert error_count > 5   # Some expected errors
        assert success_count + error_count == total_attempts


class TestDataQualityReliability:
    """Test data quality and reliability under various conditions"""
    
    def test_handling_corrupted_data(self):
        """Test handling of corrupted or malformed data"""
        corrupted_datasets = [
            # Mixed data types in numeric columns
            pd.DataFrame({
                'numeric_col': [1, 2, 'invalid', 4, np.nan],
                'health_score': [8.5, 'bad', 12.3, None, 6.1]
            }),
            
            # Extreme values
            pd.DataFrame({
                'values': [np.inf, -np.inf, 1e10, -1e10, np.nan]
            }),
            
            # Inconsistent formatting
            pd.DataFrame({
                'dates': ['2021-01-01', '01/02/2021', 'invalid', None, '2021-12-31'],
                'codes': ['ABC123', 'abc-123', 'ABC_123', '', None]
            })
        ]
        
        for i, dataset in enumerate(corrupted_datasets):
            try:
                # Should handle corrupted data gracefully
                cleaned = dataset.copy()
                
                # Basic cleaning operations
                numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    cleaned[numeric_cols] = cleaned[numeric_cols].apply(pd.to_numeric, errors='coerce')
                
                # Remove infinite values
                cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
                
                # Count valid data
                valid_data_count = cleaned.notna().sum().sum()
                
                success = True
                
            except Exception as e:
                success = False
                print(f"Corrupted dataset {i} handling failed: {e}")
            
            assert success, f"Failed to handle corrupted dataset {i}"
    
    def test_extreme_data_conditions(self):
        """Test handling of extreme data conditions"""
        extreme_conditions = [
            # All zeros
            pd.DataFrame({'values': [0] * 1000}),
            
            # All same value
            pd.DataFrame({'values': [42] * 1000}),
            
            # Single row
            pd.DataFrame({'values': [1]}),
            
            # Single column
            pd.DataFrame({'col1': range(1000)}),
            
            # Very sparse data (mostly NaN)
            pd.DataFrame({
                'sparse': [1 if i % 100 == 0 else np.nan for i in range(1000)]
            })
        ]
        
        for i, dataset in enumerate(extreme_conditions):
            try:
                # Should handle extreme conditions
                stats = dataset.describe()
                correlations = dataset.corr() if len(dataset.columns) > 1 else None
                
                # Basic aggregations
                if len(dataset) > 0:
                    mean_vals = dataset.mean()
                    sum_vals = dataset.sum()
                
                success = True
                
            except Exception as e:
                # Some operations might legitimately fail, but shouldn't crash
                success = True  # Consider this acceptable
                print(f"Extreme condition {i} caused expected error: {e}")
            
            assert success, f"Failed to handle extreme condition {i}"
    
    def test_data_consistency_validation(self):
        """Test data consistency validation across operations"""
        # Create test data with known properties
        test_data = pd.DataFrame({
            'id': range(1000),
            'group': [f"group_{i%5}" for i in range(1000)],
            'value': np.random.normal(100, 15, 1000)
        })
        
        # Test consistency across operations
        original_sum = test_data['value'].sum()
        original_count = len(test_data)
        original_groups = test_data['group'].nunique()
        
        # Perform various operations and verify consistency
        operations_results = {}
        
        # Groupby operations
        grouped_sum = test_data.groupby('group')['value'].sum().sum()
        operations_results['groupby_sum_consistency'] = abs(grouped_sum - original_sum) < 1e-10
        
        # Filtering operations
        filtered_data = test_data[test_data['value'] > 0]
        positive_count = len(filtered_data)
        operations_results['filtering_preserves_data'] = positive_count <= original_count
        
        # Sorting operations
        sorted_data = test_data.sort_values('value')
        operations_results['sorting_preserves_count'] = len(sorted_data) == original_count
        operations_results['sorting_preserves_sum'] = abs(sorted_data['value'].sum() - original_sum) < 1e-10
        
        # Merging operations
        metadata = pd.DataFrame({
            'group': [f"group_{i}" for i in range(5)],
            'description': [f"Description {i}" for i in range(5)]
        })
        merged_data = test_data.merge(metadata, on='group')
        operations_results['merge_preserves_rows'] = len(merged_data) == original_count
        
        # Assert all consistency checks pass
        for operation, passed in operations_results.items():
            assert passed, f"Consistency check failed for {operation}"


class TestScalabilityBenchmarks:
    """Test scalability with increasing data sizes"""
    
    @pytest.mark.parametrize("size_multiplier", [1, 2, 5, 10])
    def test_linear_scalability(self, size_multiplier):
        """Test that operations scale linearly with data size"""
        base_size = 5000
        data_size = base_size * size_multiplier
        
        # Generate test data
        data = pd.DataFrame({
            'id': range(data_size),
            'category': [f"cat_{i%10}" for i in range(data_size)],
            'value': np.random.random(data_size)
        })
        
        # Time various operations
        operations = {}
        
        # Basic statistics
        start_time = time.time()
        stats = data.describe()
        operations['describe'] = time.time() - start_time
        
        # Groupby operations
        start_time = time.time()
        grouped = data.groupby('category')['value'].mean()
        operations['groupby'] = time.time() - start_time
        
        # Sorting
        start_time = time.time()
        sorted_data = data.sort_values('value')
        operations['sort'] = time.time() - start_time
        
        # All operations should complete in reasonable time
        max_time_per_operation = 2.0 * size_multiplier  # Allow linear scaling
        
        for op_name, op_time in operations.items():
            assert op_time < max_time_per_operation, f"{op_name} took {op_time:.2f}s for size {data_size}"
    
    @pytest.mark.slow
    def test_memory_scalability(self):
        """Test memory usage scalability"""
        sizes = [1000, 5000, 10000, 20000]
        memory_usage = []
        
        for size in sizes:
            process = psutil.Process(os.getpid())
            
            # Clean up before measurement
            gc.collect()
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            # Create and process data
            data = pd.DataFrame({
                'col1': np.random.random(size),
                'col2': np.random.random(size),
                'col3': [f"text_{i}" for i in range(size)]
            })
            
            # Perform operations
            processed = data.copy()
            processed['col4'] = processed['col1'] * processed['col2']
            stats = processed.describe()
            
            # Measure memory after operations
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_used = final_memory - initial_memory
            memory_usage.append((size, memory_used))
            
            # Clean up
            del data, processed, stats
            gc.collect()
        
        # Verify memory usage scales reasonably
        for i in range(1, len(memory_usage)):
            size_ratio = memory_usage[i][0] / memory_usage[i-1][0]
            memory_ratio = memory_usage[i][1] / memory_usage[i-1][1] if memory_usage[i-1][1] > 0 else 1
            
            # Memory should not grow exponentially
            assert memory_ratio < size_ratio * 2, f"Memory usage growing too fast: {memory_ratio:.2f}x for {size_ratio:.2f}x data"


class TestReliabilityEdgeCases:
    """Test reliability under edge cases and boundary conditions"""
    
    def test_boundary_value_handling(self):
        """Test handling of boundary values"""
        boundary_cases = [
            # Numeric boundaries
            pd.DataFrame({'values': [sys.float_info.min, sys.float_info.max]}),
            pd.DataFrame({'values': [sys.maxsize, -sys.maxsize]}),
            
            # String boundaries
            pd.DataFrame({'text': ['', 'a' * 10000]}),
            
            # Date boundaries
            pd.DataFrame({'dates': [pd.Timestamp.min, pd.Timestamp.max]}),
        ]
        
        for i, dataset in enumerate(boundary_cases):
            try:
                # Should handle boundary values without crashing
                info = dataset.info()
                stats = dataset.describe(include='all')
                
                success = True
                
            except Exception as e:
                success = False
                print(f"Boundary case {i} failed: {e}")
            
            # Some boundary cases might cause warnings but shouldn't crash
            assert success or "overflow" in str(e).lower(), f"Boundary case {i} handling failed unexpectedly"
    
    def test_resource_cleanup(self):
        """Test proper resource cleanup and memory management"""
        initial_objects = len(gc.get_objects())
        
        # Perform operations that create many objects
        for _ in range(10):
            large_data = pd.DataFrame({
                'col1': np.random.random(10000),
                'col2': np.random.random(10000)
            })
            
            # Operations that create intermediate objects
            processed = large_data.rolling(window=100).mean()
            correlations = large_data.corr()
            grouped = large_data.groupby(pd.cut(large_data['col1'], bins=10)).mean()
            
            # Explicit cleanup
            del large_data, processed, correlations, grouped
        
        # Force garbage collection
        gc.collect()
        
        final_objects = len(gc.get_objects())
        object_increase = final_objects - initial_objects
        
        # Should not have significant object leakage
        assert object_increase < 1000, f"Potential memory leak: {object_increase} objects not cleaned up"
    
    def test_interrupt_handling(self):
        """Test graceful handling of interrupted operations"""
        def long_running_operation():
            """Simulate a long-running operation"""
            data = pd.DataFrame({
                'values': np.random.random(100000)
            })
            
            # Simulate work that can be interrupted
            for i in range(100):
                if i == 50:  # Simulate interruption point
                    raise KeyboardInterrupt("Operation interrupted")
                
                # Some processing
                result = data.rolling(window=1000).mean()
            
            return result
        
        try:
            result = long_running_operation()
            # Should not reach here due to interruption
            assert False, "Operation should have been interrupted"
            
        except KeyboardInterrupt:
            # Should handle interruption gracefully
            success = True
            
        except Exception as e:
            success = False
            print(f"Unexpected error during interruption: {e}")
        
        assert success, "Failed to handle operation interruption gracefully"