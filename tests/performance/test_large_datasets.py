"""
Large dataset performance tests for the AHGD ETL pipeline.

This module contains tests for:
- Processing very large datasets efficiently
- Memory management with large data
- Streaming vs batch processing comparison
- Scalability limits
- Performance degradation analysis
"""

import gc
import pytest
import psutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Generator
import pandas as pd
import numpy as np

from src.performance.profiler import PerformanceProfiler
from src.performance.monitoring import SystemMonitor
from src.performance.optimisation import MemoryOptimiser
from src.utils.logging import get_logger

logger = get_logger()


class TestLargeDatasets:
    """Test suite for large dataset processing performance."""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.profiler = PerformanceProfiler()
        cls.system_monitor = SystemMonitor(collection_interval=5.0)
        cls.memory_optimiser = MemoryOptimiser()
        cls.process = psutil.Process()
        cls.temp_dir = Path("tests/performance/temp")
        cls.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Start monitoring
        cls.system_monitor.start_monitoring()
    
    @classmethod
    def teardown_class(cls):
        """Clean up test environment."""
        cls.system_monitor.stop_monitoring()
        
        # Clean up temp files
        for temp_file in cls.temp_dir.glob("large_dataset_*"):
            temp_file.unlink()
    
    def test_large_csv_processing(self):
        """Test processing of large CSV files."""
        # Test with increasing file sizes
        test_sizes = [100000, 500000, 1000000]  # 100K, 500K, 1M records
        results = []
        
        for size in test_sizes:
            with self.profiler.profile_operation(f"large_csv_{size}"):
                # Create large test file
                temp_file = self._create_large_csv_file(size)
                
                try:
                    start_time = time.time()
                    initial_memory = self.process.memory_info().rss
                    
                    # Test different reading strategies
                    
                    # 1. Load all at once
                    start_full_load = time.time()
                    full_data = pd.read_csv(temp_file)
                    full_load_time = time.time() - start_full_load
                    full_load_memory = self.process.memory_info().rss
                    
                    # Process the data
                    processed_full = self._process_large_dataframe(full_data)
                    full_processing_time = time.time() - start_full_load
                    
                    del full_data, processed_full
                    gc.collect()
                    
                    # 2. Chunked processing
                    start_chunked = time.time()
                    chunk_size = min(10000, size // 10)
                    processed_records = 0
                    
                    for chunk in pd.read_csv(temp_file, chunksize=chunk_size):
                        processed_chunk = self._process_large_dataframe(chunk)
                        processed_records += len(processed_chunk)
                        del processed_chunk
                    
                    chunked_processing_time = time.time() - start_chunked
                    after_chunked_memory = self.process.memory_info().rss
                    
                    total_time = time.time() - start_time
                    
                    # Calculate metrics
                    full_memory_mb = (full_load_memory - initial_memory) / 1024 / 1024
                    chunked_memory_mb = (after_chunked_memory - initial_memory) / 1024 / 1024
                    memory_savings = (full_memory_mb - chunked_memory_mb) / full_memory_mb * 100
                    
                    result = {
                        'size': size,
                        'full_load_time': full_load_time,
                        'full_processing_time': full_processing_time,
                        'chunked_processing_time': chunked_processing_time,
                        'full_memory_mb': full_memory_mb,
                        'chunked_memory_mb': chunked_memory_mb,
                        'memory_savings_percent': memory_savings,
                        'processed_records': processed_records,
                        'throughput_full': size / full_processing_time,
                        'throughput_chunked': size / chunked_processing_time
                    }
                    
                    results.append(result)
                    
                    # Performance assertions
                    assert processed_records == size, f"Data loss in chunked processing: {processed_records} vs {size}"
                    assert memory_savings > 50, f"Insufficient memory savings: {memory_savings:.1f}%"
                    assert chunked_processing_time < full_processing_time * 2, "Chunked processing too slow"
                    
                finally:
                    temp_file.unlink()
        
        self._analyse_scaling_performance(results)
        logger.info("Large CSV processing test completed", results=results)
    
    def test_streaming_data_processing(self):
        """Test streaming data processing performance."""
        total_records = 1000000
        batch_size = 10000
        
        def data_generator() -> Generator[pd.DataFrame, None, None]:
            """Generate data in batches."""
            for i in range(0, total_records, batch_size):
                current_batch_size = min(batch_size, total_records - i)
                yield self._create_test_dataframe(current_batch_size, start_id=i)
        
        with self.profiler.profile_operation("streaming_processing"):
            start_time = time.time()
            initial_memory = self.process.memory_info().rss
            peak_memory = initial_memory
            
            processed_count = 0
            batch_count = 0
            
            for batch in data_generator():
                # Process batch
                processed_batch = self._process_large_dataframe(batch)
                processed_count += len(processed_batch)
                batch_count += 1
                
                # Monitor memory
                current_memory = self.process.memory_info().rss
                peak_memory = max(peak_memory, current_memory)
                
                # Clean up batch
                del batch, processed_batch
                
                # Periodic garbage collection
                if batch_count % 10 == 0:
                    gc.collect()
            
            total_time = time.time() - start_time
            final_memory = self.process.memory_info().rss
            
            # Calculate streaming metrics
            peak_memory_mb = (peak_memory - initial_memory) / 1024 / 1024
            final_memory_mb = (final_memory - initial_memory) / 1024 / 1024
            throughput = processed_count / total_time
            memory_efficiency = processed_count / peak_memory_mb  # records per MB
            
            # Streaming performance assertions
            assert processed_count == total_records, f"Data loss: {processed_count} vs {total_records}"
            assert peak_memory_mb < 500, f"Excessive peak memory: {peak_memory_mb:.1f} MB"
            assert throughput > 10000, f"Low throughput: {throughput:.1f} records/s"
            assert final_memory_mb < peak_memory_mb * 0.5, "Poor memory cleanup"
            
            logger.info("Streaming data processing test completed",
                       processed_count=processed_count,
                       total_time=total_time,
                       peak_memory_mb=peak_memory_mb,
                       throughput=throughput,
                       memory_efficiency=memory_efficiency)
    
    def test_memory_mapped_file_processing(self):
        """Test memory-mapped file processing for very large datasets."""
        file_size = 2000000  # 2M records
        
        with self.profiler.profile_operation("memory_mapped_processing"):
            # Create large dataset file
            temp_file = self._create_large_csv_file(file_size)
            
            try:
                start_time = time.time()
                initial_memory = self.process.memory_info().rss
                
                # Use memory mapping for large file processing
                # Note: This is a simplified example - in practice you'd use more sophisticated memory mapping
                
                # Process file in chunks without loading entire file
                chunk_size = 50000
                processed_records = 0
                
                # Read file statistics first
                total_lines = sum(1 for _ in open(temp_file)) - 1  # Subtract header
                
                # Process in chunks
                for chunk_start in range(0, total_lines, chunk_size):
                    chunk_data = pd.read_csv(
                        temp_file,
                        skiprows=range(1, chunk_start + 1) if chunk_start > 0 else None,
                        nrows=chunk_size
                    )
                    
                    processed_chunk = self._process_large_dataframe(chunk_data)
                    processed_records += len(processed_chunk)
                    
                    # Monitor memory usage
                    current_memory = self.process.memory_info().rss
                    memory_usage_mb = (current_memory - initial_memory) / 1024 / 1024
                    
                    # Assert memory usage stays reasonable
                    assert memory_usage_mb < 200, f"Memory usage too high: {memory_usage_mb:.1f} MB"
                    
                    del chunk_data, processed_chunk
                    
                    # Periodic cleanup
                    if (chunk_start // chunk_size) % 5 == 0:
                        gc.collect()
                
                total_time = time.time() - start_time
                final_memory = self.process.memory_info().rss
                final_memory_mb = (final_memory - initial_memory) / 1024 / 1024
                
                # Memory-mapped processing assertions
                assert processed_records <= file_size, f"Processing error: {processed_records} vs {file_size}"
                assert total_time < 300, f"Processing too slow: {total_time:.1f}s"
                assert final_memory_mb < 100, f"Memory not cleaned up: {final_memory_mb:.1f} MB"
                
                throughput = processed_records / total_time
                
                logger.info("Memory-mapped file processing test completed",
                           file_size=file_size,
                           processed_records=processed_records,
                           total_time=total_time,
                           throughput=throughput,
                           final_memory_mb=final_memory_mb)
                
            finally:
                temp_file.unlink()
    
    def test_data_compression_impact(self):
        """Test impact of data compression on large dataset processing."""
        dataset_size = 500000
        compression_formats = ['gzip', 'bz2', None]  # None = uncompressed
        results = []
        
        for compression in compression_formats:
            with self.profiler.profile_operation(f"compression_{compression or 'none'}"):
                # Create test data
                test_data = self._create_test_dataframe(dataset_size)
                
                # Save with different compression
                file_suffix = '.csv'
                if compression == 'gzip':
                    file_suffix = '.csv.gz'
                elif compression == 'bz2':
                    file_suffix = '.csv.bz2'
                
                temp_file = self.temp_dir / f"compression_test_{compression or 'none'}{file_suffix}"
                
                # Write file
                write_start = time.time()
                test_data.to_csv(temp_file, index=False, compression=compression)
                write_time = time.time() - write_start
                
                file_size_mb = temp_file.stat().st_size / 1024 / 1024
                
                # Read file
                read_start = time.time()
                read_data = pd.read_csv(temp_file, compression=compression)
                read_time = time.time() - read_start
                
                # Process data
                process_start = time.time()
                processed_data = self._process_large_dataframe(read_data)
                process_time = time.time() - process_start
                
                total_time = write_time + read_time + process_time
                
                result = {
                    'compression': compression or 'none',
                    'write_time': write_time,
                    'read_time': read_time,
                    'process_time': process_time,
                    'total_time': total_time,
                    'file_size_mb': file_size_mb,
                    'records_processed': len(processed_data),
                    'throughput': dataset_size / total_time
                }
                
                results.append(result)
                
                # Clean up
                temp_file.unlink()
                del test_data, read_data, processed_data
                gc.collect()
        
        # Analyse compression impact
        uncompressed = next(r for r in results if r['compression'] == 'none')
        
        for result in results:
            if result['compression'] != 'none':
                size_reduction = (uncompressed['file_size_mb'] - result['file_size_mb']) / uncompressed['file_size_mb']
                time_overhead = (result['total_time'] - uncompressed['total_time']) / uncompressed['total_time']
                
                result['size_reduction_percent'] = size_reduction * 100
                result['time_overhead_percent'] = time_overhead * 100
                
                # Compression should provide significant size reduction
                assert size_reduction > 0.3, f"Insufficient compression: {size_reduction:.1%}"
        
        logger.info("Data compression impact test completed", results=results)
    
    @pytest.mark.slow
    def test_extreme_dataset_limits(self):
        """Test processing at the limits of system capability."""
        # Start with available memory consideration
        available_memory_gb = psutil.virtual_memory().available / 1024 / 1024 / 1024
        
        # Test with dataset sizes that approach memory limits
        # Use 50% of available memory as safe limit
        safe_memory_mb = available_memory_gb * 1024 * 0.5
        
        # Estimate records per MB (rough calculation)
        # Assuming ~100 bytes per record average
        max_records = int(safe_memory_mb * 1024 * 1024 / 100)
        
        # Test sizes: 25%, 50%, 75% of estimated maximum
        test_sizes = [
            max_records // 4,
            max_records // 2,
            int(max_records * 0.75)
        ]
        
        logger.info(f"Testing extreme dataset limits with sizes: {test_sizes}")
        
        for size in test_sizes:
            try:
                with self.profiler.profile_operation(f"extreme_dataset_{size}"):
                    start_time = time.time()
                    initial_memory = self.process.memory_info().rss
                    
                    # Use streaming approach for extreme sizes
                    chunk_size = min(50000, size // 20)
                    processed_count = 0
                    peak_memory = initial_memory
                    
                    # Process in chunks
                    for i in range(0, size, chunk_size):
                        current_chunk_size = min(chunk_size, size - i)
                        
                        # Create chunk
                        chunk = self._create_test_dataframe(current_chunk_size, start_id=i)
                        
                        # Process chunk
                        processed_chunk = self._process_large_dataframe(chunk)
                        processed_count += len(processed_chunk)
                        
                        # Monitor memory
                        current_memory = self.process.memory_info().rss
                        peak_memory = max(peak_memory, current_memory)
                        
                        # Clean up immediately
                        del chunk, processed_chunk
                        
                        # Aggressive garbage collection for extreme sizes
                        if i % (chunk_size * 5) == 0:
                            gc.collect()
                        
                        # Safety check for memory usage
                        memory_usage_mb = (current_memory - initial_memory) / 1024 / 1024
                        if memory_usage_mb > safe_memory_mb:
                            logger.warning(f"Approaching memory limit: {memory_usage_mb:.1f} MB")
                            break
                    
                    total_time = time.time() - start_time
                    peak_memory_mb = (peak_memory - initial_memory) / 1024 / 1024
                    throughput = processed_count / total_time
                    
                    # Extreme dataset assertions
                    assert processed_count > 0, "No records processed"
                    assert peak_memory_mb < safe_memory_mb, f"Memory limit exceeded: {peak_memory_mb:.1f} MB"
                    assert total_time < 600, f"Processing too slow: {total_time:.1f}s"  # 10 minutes max
                    
                    logger.info(f"Extreme dataset test completed for size {size}",
                               processed_count=processed_count,
                               total_time=total_time,
                               peak_memory_mb=peak_memory_mb,
                               throughput=throughput)
                    
            except MemoryError:
                logger.warning(f"Memory limit reached at size {size}")
                break
            except Exception as e:
                logger.error(f"Extreme dataset test failed at size {size}", error=str(e))
                break
    
    def test_performance_degradation_analysis(self):
        """Analyse performance degradation patterns with increasing dataset size."""
        base_sizes = [10000, 50000, 100000, 250000, 500000]
        degradation_results = []
        
        for size in base_sizes:
            with self.profiler.profile_operation(f"degradation_analysis_{size}"):
                start_time = time.time()
                initial_memory = self.process.memory_info().rss
                
                # Create and process dataset
                data = self._create_test_dataframe(size)
                processed_data = self._process_large_dataframe(data)
                
                processing_time = time.time() - start_time
                peak_memory = self.process.memory_info().rss
                memory_usage_mb = (peak_memory - initial_memory) / 1024 / 1024
                
                # Calculate per-record metrics
                time_per_record = processing_time / size
                memory_per_record = memory_usage_mb / size
                throughput = size / processing_time
                
                degradation_results.append({
                    'size': size,
                    'processing_time': processing_time,
                    'memory_usage_mb': memory_usage_mb,
                    'time_per_record': time_per_record,
                    'memory_per_record_mb': memory_per_record,
                    'throughput': throughput
                })
                
                # Clean up
                del data, processed_data
                gc.collect()
        
        # Analyse degradation patterns
        self._analyse_performance_degradation(degradation_results)
        logger.info("Performance degradation analysis completed", results=degradation_results)
    
    # Helper methods
    def _create_large_csv_file(self, size: int) -> Path:
        """Create a large CSV file for testing."""
        filename = self.temp_dir / f"large_dataset_{size}_{int(time.time())}.csv"
        
        # Generate data in chunks to avoid memory issues
        chunk_size = min(50000, size)
        
        with open(filename, 'w') as f:
            # Write header
            f.write("id,name,value,category,timestamp,latitude,longitude,description\n")
            
            # Write data in chunks
            for i in range(0, size, chunk_size):
                current_chunk_size = min(chunk_size, size - i)
                chunk_data = self._create_test_dataframe(current_chunk_size, start_id=i)
                
                # Write without header for subsequent chunks
                chunk_data.to_csv(f, index=False, header=False)
                
                del chunk_data
        
        return filename
    
    def _create_test_dataframe(self, size: int, start_id: int = 0) -> pd.DataFrame:
        """Create test DataFrame with realistic data structure."""
        return pd.DataFrame({
            'id': range(start_id, start_id + size),
            'name': [f'Entity_{i:08d}' for i in range(start_id, start_id + size)],
            'value': np.random.random(size),
            'category': np.random.choice(['Category_A', 'Category_B', 'Category_C', 'Category_D'], size),
            'timestamp': pd.date_range('2023-01-01', periods=size, freq='1H'),
            'latitude': np.random.uniform(-90, 90, size),
            'longitude': np.random.uniform(-180, 180, size),
            'description': [f'Description for entity {i} with some additional text content' 
                          for i in range(start_id, start_id + size)]
        })
    
    def _process_large_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process DataFrame with realistic transformations."""
        result = df.copy()
        
        # Data transformations
        result['name_upper'] = result['name'].str.upper()
        result['value_squared'] = result['value'] ** 2
        result['value_log'] = np.log1p(result['value'])
        result['category_encoded'] = pd.Categorical(result['category']).codes
        
        # Derived fields
        result['coordinate_distance'] = np.sqrt(
            result['latitude']**2 + result['longitude']**2
        )
        
        # Text processing
        result['description_length'] = result['description'].str.len()
        result['description_word_count'] = result['description'].str.split().str.len()
        
        # Conditional logic
        result['value_category'] = pd.cut(
            result['value'], 
            bins=[0, 0.25, 0.5, 0.75, 1.0], 
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        return result
    
    def _analyse_scaling_performance(self, results: List[Dict[str, Any]]):
        """Analyse how performance scales with dataset size."""
        if len(results) < 3:
            return
        
        sizes = [r['size'] for r in results]
        chunked_times = [r['chunked_processing_time'] for r in results]
        
        # Calculate scaling factors
        size_ratios = [sizes[i] / sizes[0] for i in range(len(sizes))]
        time_ratios = [chunked_times[i] / chunked_times[0] for i in range(len(chunked_times))]
        
        # Analyse complexity (simplified linear regression)
        avg_complexity = sum(time_ratios[i] / size_ratios[i] for i in range(1, len(size_ratios))) / (len(size_ratios) - 1)
        
        # Performance should scale roughly linearly
        assert avg_complexity < 2.0, f"Poor scaling: complexity factor {avg_complexity:.2f}"
        
        logger.info("Scaling performance analysis completed",
                   size_ratios=size_ratios,
                   time_ratios=time_ratios,
                   avg_complexity=avg_complexity)
    
    def _analyse_performance_degradation(self, results: List[Dict[str, Any]]):
        """Analyse performance degradation patterns."""
        if len(results) < 3:
            return
        
        # Check if time per record increases significantly
        times_per_record = [r['time_per_record'] for r in results]
        degradation_factor = times_per_record[-1] / times_per_record[0]
        
        # Check memory efficiency
        memory_per_record = [r['memory_per_record_mb'] for r in results]
        memory_efficiency_degradation = memory_per_record[-1] / memory_per_record[0]
        
        # Assertions for reasonable degradation
        assert degradation_factor < 3.0, f"Severe time degradation: {degradation_factor:.2f}x"
        assert memory_efficiency_degradation < 2.0, f"Severe memory degradation: {memory_efficiency_degradation:.2f}x"
        
        logger.info("Performance degradation analysis completed",
                   time_degradation_factor=degradation_factor,
                   memory_degradation_factor=memory_efficiency_degradation)