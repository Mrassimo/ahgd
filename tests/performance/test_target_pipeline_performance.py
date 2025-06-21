"""Performance tests for target pipeline requirements.

This module implements Test-Driven Development for performance requirements,
validating processing speed, memory efficiency, and scalability targets.
"""

import pytest
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime, timedelta
import pandas as pd
import memory_profiler

from src.utils.logging import get_logger, monitor_performance
from src.performance.monitoring import PerformanceMonitor

logger = get_logger(__name__)


@dataclass
class PerformanceTarget:
    """Performance target specification."""
    metric_name: str
    target_value: float
    unit: str
    tolerance_pct: float
    measurement_method: str
    description: str


@dataclass
class MemoryConstraint:
    """Memory usage constraint specification."""
    operation: str
    max_memory_mb: int
    max_memory_growth_mb: int
    gc_efficiency_threshold: float
    description: str


@dataclass
class ConcurrencyTarget:
    """Concurrent processing target specification."""
    operation: str
    max_workers: int
    target_throughput: float
    max_latency_ms: int
    error_tolerance_pct: float


class TestSA2ProcessingPerformance:
    """Test processing speed for 2,473 SA2s across Australia."""
    
    @pytest.fixture
    def performance_targets(self):
        """Define performance targets for SA2 processing."""
        return [
            PerformanceTarget(
                metric_name="sa2_extraction_time",
                target_value=30.0,
                unit="seconds",
                tolerance_pct=20.0,
                measurement_method="wall_clock",
                description="Time to extract data for all 2,473 SA2s"
            ),
            PerformanceTarget(
                metric_name="sa2_transformation_time",
                target_value=45.0,
                unit="seconds",
                tolerance_pct=25.0,
                measurement_method="wall_clock",
                description="Time to transform and integrate all SA2 data"
            ),
            PerformanceTarget(
                metric_name="sa2_validation_time",
                target_value=20.0,
                unit="seconds",
                tolerance_pct=30.0,
                measurement_method="wall_clock",
                description="Time to validate all SA2 data quality"
            ),
            PerformanceTarget(
                metric_name="sa2_export_time",
                target_value=60.0,
                unit="seconds",
                tolerance_pct=20.0,
                measurement_method="wall_clock",
                description="Time to export all SA2 data in multiple formats"
            ),
            PerformanceTarget(
                metric_name="sa2_processing_throughput",
                target_value=15.0,
                unit="sa2s_per_second",
                tolerance_pct=15.0,
                measurement_method="throughput",
                description="SA2s processed per second during peak processing"
            )
        ]
    
    def test_sa2_processing_performance(self, performance_targets):
        """Test that SA2 processing meets performance targets.
        
        Validates processing speed for all 2,473 SA2 areas across Australia
        meets specified performance requirements for production use.
        """
        from src.etl.sa2_processor import SA2Processor
        from src.performance.monitoring import PerformanceMonitor
        
        processor = SA2Processor()
        monitor = PerformanceMonitor()
        
        # Get list of all Australian SA2 codes
        all_sa2_codes = processor.get_all_sa2_codes()
        assert len(all_sa2_codes) == 2473, f"Expected 2473 SA2s, got {len(all_sa2_codes)}"
        
        # Test each performance target
        for target in performance_targets:
            
            if target.metric_name == "sa2_extraction_time":
                # Test extraction performance
                start_time = time.time()
                extraction_result = processor.extract_all_sa2_data()
                end_time = time.time()
                
                actual_time = end_time - start_time
                max_allowed_time = target.target_value * (1 + target.tolerance_pct / 100)
                
                assert actual_time <= max_allowed_time, \
                    f"SA2 extraction took {actual_time:.2f}s, exceeds target of {max_allowed_time:.2f}s"
                
                assert extraction_result.success, "SA2 extraction failed"
                assert len(extraction_result.processed_sa2s) == 2473, \
                    "Not all SA2s were extracted"
            
            elif target.metric_name == "sa2_transformation_time":
                # Test transformation performance
                start_time = time.time()
                transformation_result = processor.transform_all_sa2_data()
                end_time = time.time()
                
                actual_time = end_time - start_time
                max_allowed_time = target.target_value * (1 + target.tolerance_pct / 100)
                
                assert actual_time <= max_allowed_time, \
                    f"SA2 transformation took {actual_time:.2f}s, exceeds target of {max_allowed_time:.2f}s"
                
                assert transformation_result.success, "SA2 transformation failed"
                assert len(transformation_result.integrated_records) == 2473, \
                    "Not all SA2s were transformed"
            
            elif target.metric_name == "sa2_validation_time":
                # Test validation performance
                start_time = time.time()
                validation_result = processor.validate_all_sa2_data()
                end_time = time.time()
                
                actual_time = end_time - start_time
                max_allowed_time = target.target_value * (1 + target.tolerance_pct / 100)
                
                assert actual_time <= max_allowed_time, \
                    f"SA2 validation took {actual_time:.2f}s, exceeds target of {max_allowed_time:.2f}s"
                
                assert validation_result.overall_validation_passed, "SA2 validation failed"
                assert validation_result.validated_records == 2473, \
                    "Not all SA2s were validated"
            
            elif target.metric_name == "sa2_export_time":
                # Test export performance
                export_formats = ['json', 'csv', 'parquet', 'geojson']
                
                start_time = time.time()
                export_results = processor.export_all_sa2_data(export_formats)
                end_time = time.time()
                
                actual_time = end_time - start_time
                max_allowed_time = target.target_value * (1 + target.tolerance_pct / 100)
                
                assert actual_time <= max_allowed_time, \
                    f"SA2 export took {actual_time:.2f}s, exceeds target of {max_allowed_time:.2f}s"
                
                for format_name, result in export_results.items():
                    assert result.success, f"Export failed for format {format_name}"
                    assert result.record_count == 2473, \
                        f"Not all SA2s exported for format {format_name}"
            
            elif target.metric_name == "sa2_processing_throughput":
                # Test processing throughput
                sample_sa2s = all_sa2_codes[:100]  # Test with sample for throughput
                
                start_time = time.time()
                throughput_result = processor.process_sa2_batch(sample_sa2s)
                end_time = time.time()
                
                processing_time = end_time - start_time
                actual_throughput = len(sample_sa2s) / processing_time
                min_required_throughput = target.target_value * (1 - target.tolerance_pct / 100)
                
                assert actual_throughput >= min_required_throughput, \
                    f"SA2 throughput {actual_throughput:.2f} SA2s/s below target of {min_required_throughput:.2f}"
    
    def test_sa2_processing_scalability(self):
        """Test that processing scales linearly with SA2 count."""
        from src.etl.sa2_processor import SA2Processor
        
        processor = SA2Processor()
        all_sa2_codes = processor.get_all_sa2_codes()
        
        # Test with different batch sizes
        batch_sizes = [10, 50, 100, 500, 1000]
        processing_times = []
        
        for batch_size in batch_sizes:
            sample_sa2s = all_sa2_codes[:batch_size]
            
            start_time = time.time()
            result = processor.process_sa2_batch(sample_sa2s)
            end_time = time.time()
            
            processing_time = end_time - start_time
            processing_times.append(processing_time)
            
            assert result.success, f"Processing failed for batch size {batch_size}"
        
        # Check for linear scaling (allowing some overhead)
        for i in range(1, len(batch_sizes)):
            expected_ratio = batch_sizes[i] / batch_sizes[0]
            actual_ratio = processing_times[i] / processing_times[0]
            
            # Allow up to 50% overhead for larger batches
            max_allowed_ratio = expected_ratio * 1.5
            
            assert actual_ratio <= max_allowed_ratio, \
                f"Processing doesn't scale linearly: batch {batch_sizes[i]} took {actual_ratio:.2f}x " \
                f"time vs expected {expected_ratio:.2f}x (max allowed {max_allowed_ratio:.2f}x)"
    
    def test_sa2_processing_consistency(self):
        """Test that processing performance is consistent across runs."""
        from src.etl.sa2_processor import SA2Processor
        
        processor = SA2Processor()
        sample_sa2s = processor.get_all_sa2_codes()[:100]
        
        # Run multiple iterations
        processing_times = []
        num_iterations = 5
        
        for i in range(num_iterations):
            start_time = time.time()
            result = processor.process_sa2_batch(sample_sa2s)
            end_time = time.time()
            
            processing_time = end_time - start_time
            processing_times.append(processing_time)
            
            assert result.success, f"Processing failed on iteration {i+1}"
        
        # Check consistency (coefficient of variation < 20%)
        mean_time = sum(processing_times) / len(processing_times)
        variance = sum((t - mean_time) ** 2 for t in processing_times) / len(processing_times)
        std_dev = variance ** 0.5
        cv = std_dev / mean_time
        
        assert cv < 0.20, \
            f"Processing time too variable: CV = {cv:.3f} (>20%), times = {processing_times}"


class TestMemoryUsageConstraints:
    """Test memory efficiency for large datasets."""
    
    @pytest.fixture
    def memory_constraints(self):
        """Define memory usage constraints for different operations."""
        return [
            MemoryConstraint(
                operation="sa2_data_loading",
                max_memory_mb=2048,  # 2GB max for loading all SA2 data
                max_memory_growth_mb=1024,  # 1GB max growth during operation
                gc_efficiency_threshold=0.80,  # 80% memory should be reclaimable
                description="Memory usage when loading all SA2 data into memory"
            ),
            MemoryConstraint(
                operation="data_transformation",
                max_memory_mb=3072,  # 3GB max during transformation
                max_memory_growth_mb=1536,  # 1.5GB max growth
                gc_efficiency_threshold=0.75,
                description="Memory usage during data transformation operations"
            ),
            MemoryConstraint(
                operation="export_generation",
                max_memory_mb=4096,  # 4GB max during export
                max_memory_growth_mb=2048,  # 2GB max growth
                gc_efficiency_threshold=0.70,
                description="Memory usage during multi-format export generation"
            ),
            MemoryConstraint(
                operation="validation_processing",
                max_memory_mb=1536,  # 1.5GB max for validation
                max_memory_growth_mb=768,  # 768MB max growth
                gc_efficiency_threshold=0.85,
                description="Memory usage during data quality validation"
            )
        ]
    
    def test_memory_usage_constraints(self, memory_constraints):
        """Test that memory usage stays within defined constraints.
        
        Validates memory efficiency for processing large health datasets
        and ensures memory doesn't grow beyond acceptable limits.
        """
        from src.etl.sa2_processor import SA2Processor
        import gc
        
        processor = SA2Processor()
        
        for constraint in memory_constraints:
            # Get baseline memory usage
            gc.collect()  # Force garbage collection
            baseline_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            
            if constraint.operation == "sa2_data_loading":
                # Test memory usage during data loading
                peak_memory = self._monitor_memory_usage(
                    lambda: processor.load_all_sa2_data()
                )
                
            elif constraint.operation == "data_transformation":
                # Test memory usage during transformation
                processor.load_all_sa2_data()  # Pre-load data
                peak_memory = self._monitor_memory_usage(
                    lambda: processor.transform_all_sa2_data()
                )
                
            elif constraint.operation == "export_generation":
                # Test memory usage during export
                processor.prepare_export_data()  # Pre-prepare data
                peak_memory = self._monitor_memory_usage(
                    lambda: processor.export_all_formats()
                )
                
            elif constraint.operation == "validation_processing":
                # Test memory usage during validation
                processor.load_validation_data()  # Pre-load data
                peak_memory = self._monitor_memory_usage(
                    lambda: processor.validate_all_sa2_data()
                )
            
            # Check maximum memory usage
            assert peak_memory <= constraint.max_memory_mb, \
                f"Peak memory {peak_memory:.1f}MB exceeds limit of {constraint.max_memory_mb}MB " \
                f"for operation {constraint.operation}"
            
            # Check memory growth
            memory_growth = peak_memory - baseline_memory
            assert memory_growth <= constraint.max_memory_growth_mb, \
                f"Memory growth {memory_growth:.1f}MB exceeds limit of {constraint.max_memory_growth_mb}MB " \
                f"for operation {constraint.operation}"
            
            # Test garbage collection efficiency
            gc.collect()
            post_gc_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            memory_freed = peak_memory - post_gc_memory
            gc_efficiency = memory_freed / (peak_memory - baseline_memory) if peak_memory > baseline_memory else 1.0
            
            assert gc_efficiency >= constraint.gc_efficiency_threshold, \
                f"GC efficiency {gc_efficiency:.2f} below threshold {constraint.gc_efficiency_threshold} " \
                f"for operation {constraint.operation}"
    
    def _monitor_memory_usage(self, operation_func):
        """Monitor peak memory usage during operation execution."""
        peak_memory = 0
        monitoring = True
        
        def memory_monitor():
            nonlocal peak_memory
            while monitoring:
                current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                peak_memory = max(peak_memory, current_memory)
                time.sleep(0.1)  # Check every 100ms
        
        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=memory_monitor)
        monitor_thread.start()
        
        try:
            # Execute the operation
            result = operation_func()
            return peak_memory
        finally:
            monitoring = False
            monitor_thread.join()
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated operations."""
        from src.etl.sa2_processor import SA2Processor
        import gc
        
        processor = SA2Processor()
        sample_sa2s = processor.get_all_sa2_codes()[:50]
        
        # Record initial memory
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Perform multiple iterations
        memory_readings = []
        num_iterations = 10
        
        for i in range(num_iterations):
            processor.process_sa2_batch(sample_sa2s)
            gc.collect()
            current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            memory_readings.append(current_memory)
        
        # Check for memory leaks (memory should not consistently grow)
        final_memory = memory_readings[-1]
        memory_growth = final_memory - initial_memory
        
        # Allow some memory growth but not excessive
        max_allowed_growth = 100  # 100MB max growth over 10 iterations
        
        assert memory_growth <= max_allowed_growth, \
            f"Possible memory leak detected: grew {memory_growth:.1f}MB over {num_iterations} iterations"
        
        # Check that memory isn't consistently increasing
        increasing_trend = sum(1 for i in range(1, len(memory_readings)) 
                             if memory_readings[i] > memory_readings[i-1])
        
        # Should not increase in more than 70% of iterations
        assert increasing_trend / (num_iterations - 1) <= 0.70, \
            f"Memory consistently increasing in {increasing_trend}/{num_iterations-1} iterations"


class TestExportPerformanceStandards:
    """Test export speed requirements."""
    
    def test_export_performance_standards(self):
        """Test that data exports meet performance standards.
        
        Validates export speed for different formats and sizes
        meets production requirements for user experience.
        """
        from src.web.data_export_engine import MultiFormatExporter
        
        exporter = MultiFormatExporter()
        
        # Export performance targets
        export_targets = {
            'csv': {'max_time_seconds': 30, 'max_size_mb': 50},
            'json': {'max_time_seconds': 45, 'max_size_mb': 75},
            'parquet': {'max_time_seconds': 20, 'max_size_mb': 25},
            'geojson': {'max_time_seconds': 60, 'max_size_mb': 200}
        }
        
        # Get full dataset for export testing
        full_dataset = exporter.get_master_dataset()
        assert len(full_dataset) == 2473, "Full dataset should contain all SA2s"
        
        for export_format, targets in export_targets.items():
            # Test export performance
            start_time = time.time()
            export_result = exporter.export_data(
                data=full_dataset,
                format=export_format,
                compression='gzip' if export_format != 'geojson' else None
            )
            end_time = time.time()
            
            export_time = end_time - start_time
            file_size_mb = export_result.file_size_bytes / (1024 * 1024)
            
            # Validate export success
            assert export_result.success, f"Export failed for format {export_format}"
            
            # Validate performance targets
            assert export_time <= targets['max_time_seconds'], \
                f"Export time {export_time:.2f}s exceeds target {targets['max_time_seconds']}s for {export_format}"
            
            assert file_size_mb <= targets['max_size_mb'], \
                f"Export size {file_size_mb:.2f}MB exceeds target {targets['max_size_mb']}MB for {export_format}"
            
            # Test export quality
            quality_check = exporter.validate_export_quality(export_result.file_path)
            assert quality_check.data_integrity_verified, \
                f"Data integrity check failed for {export_format} export"
    
    def test_incremental_export_performance(self):
        """Test performance of incremental exports for updated data."""
        from src.web.data_export_engine import MultiFormatExporter
        from src.etl.incremental_processor import IncrementalProcessor
        
        exporter = MultiFormatExporter()
        incremental_processor = IncrementalProcessor()
        
        # Simulate data updates for subset of SA2s
        updated_sa2s = ['101011007', '201011021', '301011001']  # 3 SA2s
        
        # Test incremental export performance
        start_time = time.time()
        incremental_result = incremental_processor.process_incremental_updates(
            updated_sa2s=updated_sa2s,
            export_formats=['json', 'csv']
        )
        end_time = time.time()
        
        incremental_time = end_time - start_time
        
        # Incremental updates should be very fast
        assert incremental_time <= 5.0, \
            f"Incremental export took {incremental_time:.2f}s, should be <5s"
        
        assert incremental_result.success, "Incremental export failed"
        assert len(incremental_result.updated_exports) > 0, "No exports were updated"
    
    def test_concurrent_export_performance(self):
        """Test performance when multiple exports run concurrently."""
        from src.web.data_export_engine import MultiFormatExporter
        from concurrent.futures import ThreadPoolExecutor
        
        exporter = MultiFormatExporter()
        dataset = exporter.get_master_dataset()
        
        # Test concurrent exports
        export_formats = ['csv', 'json', 'parquet']
        
        def export_format(fmt):
            start_time = time.time()
            result = exporter.export_data(data=dataset, format=fmt)
            end_time = time.time()
            return {
                'format': fmt,
                'time': end_time - start_time,
                'success': result.success,
                'size_mb': result.file_size_bytes / (1024 * 1024)
            }
        
        # Run concurrent exports
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(export_format, export_formats))
        end_time = time.time()
        
        total_concurrent_time = end_time - start_time
        
        # Validate all exports succeeded
        for result in results:
            assert result['success'], f"Concurrent export failed for {result['format']}"
        
        # Concurrent execution should be faster than sequential
        sequential_time_estimate = sum(result['time'] for result in results)
        speedup = sequential_time_estimate / total_concurrent_time
        
        assert speedup >= 1.5, \
            f"Concurrent exports not efficient: speedup {speedup:.2f}x (expected >1.5x)"


class TestConcurrentProcessingCapacity:
    """Test parallel processing capabilities."""
    
    @pytest.fixture
    def concurrency_targets(self):
        """Define concurrent processing targets."""
        return [
            ConcurrencyTarget(
                operation="parallel_sa2_processing",
                max_workers=8,
                target_throughput=50.0,  # SA2s per second
                max_latency_ms=2000,  # 2 second max latency per SA2
                error_tolerance_pct=1.0  # 1% error tolerance
            ),
            ConcurrencyTarget(
                operation="concurrent_validation",
                max_workers=4,
                target_throughput=100.0,  # Validations per second
                max_latency_ms=500,  # 500ms max latency per validation
                error_tolerance_pct=0.5  # 0.5% error tolerance
            ),
            ConcurrencyTarget(
                operation="parallel_export_generation",
                max_workers=6,
                target_throughput=20.0,  # Exports per second
                max_latency_ms=3000,  # 3 second max latency per export
                error_tolerance_pct=2.0  # 2% error tolerance
            )
        ]
    
    def test_concurrent_processing_capacity(self, concurrency_targets):
        """Test parallel processing capabilities and limits.
        
        Validates that the system can handle concurrent processing
        with multiple workers while maintaining performance and reliability.
        """
        from src.etl.sa2_processor import SA2Processor
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        processor = SA2Processor()
        all_sa2_codes = processor.get_all_sa2_codes()
        
        for target in concurrency_targets:
            
            if target.operation == "parallel_sa2_processing":
                # Test parallel SA2 processing
                batch_size = 50  # Process 50 SA2s per worker
                sa2_batches = [all_sa2_codes[i:i+batch_size] 
                              for i in range(0, len(all_sa2_codes), batch_size)]
                
                start_time = time.time()
                results = []
                errors = 0
                
                with ThreadPoolExecutor(max_workers=target.max_workers) as executor:
                    future_to_batch = {executor.submit(processor.process_sa2_batch, batch): batch 
                                     for batch in sa2_batches[:10]}  # Test with first 10 batches
                    
                    for future in as_completed(future_to_batch):
                        try:
                            result = future.result(timeout=target.max_latency_ms/1000)
                            results.append(result)
                        except Exception as e:
                            errors += 1
                            logger.error(f"Batch processing failed: {e}")
                
                end_time = time.time()
                
                # Calculate performance metrics
                total_time = end_time - start_time
                total_sa2s_processed = sum(len(r.processed_sa2s) for r in results if r.success)
                actual_throughput = total_sa2s_processed / total_time
                error_rate = errors / len(sa2_batches[:10]) * 100
                
                # Validate performance targets
                assert actual_throughput >= target.target_throughput, \
                    f"Parallel processing throughput {actual_throughput:.1f} SA2s/s " \
                    f"below target {target.target_throughput} SA2s/s"
                
                assert error_rate <= target.error_tolerance_pct, \
                    f"Error rate {error_rate:.1f}% exceeds tolerance {target.error_tolerance_pct}%"
            
            elif target.operation == "concurrent_validation":
                # Test concurrent validation
                validation_tasks = all_sa2_codes[:100]  # Test with 100 SA2s
                
                start_time = time.time()
                validation_results = []
                errors = 0
                
                with ThreadPoolExecutor(max_workers=target.max_workers) as executor:
                    future_to_sa2 = {executor.submit(processor.validate_sa2_record, sa2): sa2 
                                    for sa2 in validation_tasks}
                    
                    for future in as_completed(future_to_sa2):
                        try:
                            result = future.result(timeout=target.max_latency_ms/1000)
                            validation_results.append(result)
                        except Exception as e:
                            errors += 1
                            logger.error(f"Validation failed: {e}")
                
                end_time = time.time()
                
                total_time = end_time - start_time
                actual_throughput = len(validation_results) / total_time
                error_rate = errors / len(validation_tasks) * 100
                
                assert actual_throughput >= target.target_throughput, \
                    f"Validation throughput {actual_throughput:.1f} validations/s " \
                    f"below target {target.target_throughput} validations/s"
                
                assert error_rate <= target.error_tolerance_pct, \
                    f"Validation error rate {error_rate:.1f}% exceeds tolerance {target.error_tolerance_pct}%"
    
    def test_system_resource_utilisation(self):
        """Test efficient utilisation of system resources during concurrent processing."""
        from src.etl.sa2_processor import SA2Processor
        from concurrent.futures import ThreadPoolExecutor
        import psutil
        
        processor = SA2Processor()
        
        # Monitor system resources during concurrent processing
        cpu_usage_readings = []
        memory_usage_readings = []
        monitoring = True
        
        def resource_monitor():
            while monitoring:
                cpu_usage_readings.append(psutil.cpu_percent(interval=0.1))
                memory_usage_readings.append(psutil.virtual_memory().percent)
                time.sleep(0.5)
        
        # Start resource monitoring
        monitor_thread = threading.Thread(target=resource_monitor)
        monitor_thread.start()
        
        try:
            # Run concurrent processing
            sa2_batches = [processor.get_all_sa2_codes()[i:i+25] 
                          for i in range(0, 200, 25)]  # 8 batches of 25 SA2s
            
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(processor.process_sa2_batch, sa2_batches))
            
        finally:
            monitoring = False
            monitor_thread.join()
        
        # Analyse resource utilisation
        avg_cpu_usage = sum(cpu_usage_readings) / len(cpu_usage_readings)
        max_cpu_usage = max(cpu_usage_readings)
        avg_memory_usage = sum(memory_usage_readings) / len(memory_usage_readings)
        max_memory_usage = max(memory_usage_readings)
        
        # CPU should be well utilised but not maxed out
        assert 30 <= avg_cpu_usage <= 85, \
            f"CPU utilisation {avg_cpu_usage:.1f}% not in optimal range (30-85%)"
        
        assert max_cpu_usage <= 95, \
            f"Peak CPU usage {max_cpu_usage:.1f}% too high (>95%)"
        
        # Memory usage should be reasonable
        assert max_memory_usage <= 80, \
            f"Peak memory usage {max_memory_usage:.1f}% too high (>80%)"
        
        # All processing should succeed
        successful_batches = sum(1 for r in results if r.success)
        assert successful_batches == len(sa2_batches), \
            f"Only {successful_batches}/{len(sa2_batches)} batches processed successfully"
    
    def test_concurrent_processing_stability(self):
        """Test stability of concurrent processing under sustained load."""
        from src.etl.sa2_processor import SA2Processor
        from concurrent.futures import ThreadPoolExecutor
        
        processor = SA2Processor()
        
        # Run sustained concurrent processing for several minutes
        test_duration_seconds = 120  # 2 minutes
        batch_size = 20
        max_workers = 6
        
        start_time = time.time()
        total_batches_processed = 0
        total_errors = 0
        
        while time.time() - start_time < test_duration_seconds:
            # Get random sample of SA2s for processing
            sample_sa2s = processor.get_random_sa2_sample(batch_size)
            
            try:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    result = executor.submit(processor.process_sa2_batch, sample_sa2s).result(timeout=10)
                    
                if result.success:
                    total_batches_processed += 1
                else:
                    total_errors += 1
                    
            except Exception as e:
                total_errors += 1
                logger.error(f"Concurrent processing error: {e}")
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.1)
        
        # Validate stability metrics
        error_rate = total_errors / (total_batches_processed + total_errors) * 100 if total_batches_processed + total_errors > 0 else 100
        
        assert total_batches_processed > 0, "No batches were successfully processed"
        assert error_rate <= 5.0, \
            f"Error rate {error_rate:.1f}% too high during sustained load (>5%)"
        
        # System should maintain reasonable performance
        avg_batches_per_minute = total_batches_processed / (test_duration_seconds / 60)
        assert avg_batches_per_minute >= 10, \
            f"Processing rate {avg_batches_per_minute:.1f} batches/min too low during sustained load"
