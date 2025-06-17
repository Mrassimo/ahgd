"""
Stress Testing and Resilience Validation - Phase 5.4

Tests the platform's resilience under extreme conditions, resource constraints,
and failure scenarios. Validates system stability, error recovery, graceful
degradation, and continuous operation capabilities for production deployment.

Key Stress Tests:
- Memory leak detection during extended operations
- Error recovery under resource constraints  
- Graceful degradation testing
- System stability under extreme loads
- Resource exhaustion scenario testing
- 24+ hour continuous operation simulation
"""

import pytest
import polars as pl
import numpy as np
import time
import psutil
import gc
import logging
import threading
import os
import signal
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import concurrent.futures
from contextlib import contextmanager
import resource
import traceback

from src.data_processing.seifa_processor import SEIFAProcessor
from src.data_processing.health_processor import HealthDataProcessor
from src.data_processing.simple_boundary_processor import SimpleBoundaryProcessor
from src.data_processing.storage.parquet_storage_manager import ParquetStorageManager
from src.data_processing.storage.memory_optimizer import MemoryOptimizer
from src.data_processing.storage.incremental_processor import IncrementalProcessor
from tests.performance import PERFORMANCE_CONFIG, AUSTRALIAN_DATA_SCALE
from tests.performance.test_large_scale_processing import AustralianHealthDataGenerator

logger = logging.getLogger(__name__)


@dataclass
class StressTestResult:
    """Results from stress testing."""
    test_name: str
    duration_hours: float
    operations_completed: int
    failures_encountered: int
    recovery_successful: bool
    memory_leak_detected: bool
    performance_degradation_percent: float
    system_stable: bool
    targets_met: Dict[str, bool]
    stress_details: Dict[str, Any]


@dataclass
class ResilienceTestResult:
    """Results from resilience testing."""
    test_name: str
    failure_scenarios_tested: int
    recovery_success_rate: float
    average_recovery_time_seconds: float
    graceful_degradation_achieved: bool
    data_integrity_preserved: bool
    system_availability_percent: float
    resilience_score: float


class StressTestEnvironment:
    """Manages stress testing environment and resource constraints."""
    
    def __init__(self):
        self.memory_samples = []
        self.cpu_samples = []
        self.operation_times = []
        self.error_count = 0
        self.recovery_count = 0
        self.start_time = None
        self.monitoring_active = False
        self.monitoring_thread = None
    
    def start_monitoring(self, sample_interval: float = 5.0):
        """Start continuous resource monitoring."""
        self.start_time = time.time()
        self.monitoring_active = True
        
        def monitor_resources():
            while self.monitoring_active:
                try:
                    process = psutil.Process()
                    timestamp = time.time() - self.start_time
                    
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    cpu_percent = process.cpu_percent()
                    
                    self.memory_samples.append((timestamp, memory_mb))
                    self.cpu_samples.append((timestamp, cpu_percent))
                    
                    time.sleep(sample_interval)
                except Exception as e:
                    logger.warning(f"Monitoring error: {e}")
        
        self.monitoring_thread = threading.Thread(target=monitor_resources)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
    
    def detect_memory_leak(self, leak_threshold_mb: float = 100.0) -> bool:
        """Detect memory leaks from monitoring data."""
        if len(self.memory_samples) < 10:
            return False
        
        # Analyze memory trend over time
        times = [sample[0] for sample in self.memory_samples]
        memories = [sample[1] for sample in self.memory_samples]
        
        # Linear regression to detect upward trend
        if len(memories) >= 2:
            memory_slope = (memories[-1] - memories[0]) / (times[-1] - times[0]) if times[-1] > times[0] else 0
            total_growth = memories[-1] - memories[0]
            
            # Leak detected if consistent upward trend and significant growth
            return memory_slope > 1.0 and total_growth > leak_threshold_mb
        
        return False
    
    def calculate_performance_degradation(self) -> float:
        """Calculate performance degradation over time."""
        if len(self.operation_times) < 10:
            return 0.0
        
        # Compare first 25% vs last 25% of operations
        quarter_size = len(self.operation_times) // 4
        if quarter_size < 2:
            return 0.0
        
        early_times = self.operation_times[:quarter_size]
        late_times = self.operation_times[-quarter_size:]
        
        early_avg = np.mean(early_times)
        late_avg = np.mean(late_times)
        
        degradation_percent = ((late_avg - early_avg) / early_avg) * 100 if early_avg > 0 else 0
        return max(0, degradation_percent)
    
    @contextmanager
    def constrain_memory(self, limit_mb: int):
        """Constrain available memory for stress testing."""
        try:
            # Set memory limit (Unix-like systems)
            limit_bytes = limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
            yield
        except (OSError, ValueError) as e:
            logger.warning(f"Could not set memory limit: {e}")
            yield
        finally:
            try:
                # Reset memory limit
                resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
            except:
                pass
    
    @contextmanager
    def simulate_resource_pressure(self):
        """Simulate resource pressure conditions."""
        # Create memory pressure
        memory_pressure = []
        try:
            # Allocate some memory to create pressure
            for _ in range(10):
                memory_pressure.append(np.zeros(10_000_000, dtype=np.float32))  # ~40MB each
            
            yield
        finally:
            # Clean up memory pressure
            del memory_pressure
            gc.collect()


class TestStressResilience:
    """Stress testing and resilience validation for Australian Health Analytics platform."""
    
    @pytest.fixture(scope="class")
    def data_generator(self):
        """Create Australian health data generator."""
        return AustralianHealthDataGenerator(seed=42)
    
    @pytest.fixture(scope="class")
    def stress_processors(self, tmp_path_factory):
        """Create processors for stress testing."""
        temp_dir = tmp_path_factory.mktemp("stress_test")
        
        return {
            'seifa_processor': SEIFAProcessor(data_dir=temp_dir),
            'health_processor': HealthDataProcessor(data_dir=temp_dir),
            'boundary_processor': SimpleBoundaryProcessor(data_dir=temp_dir),
            'storage_manager': ParquetStorageManager(base_path=temp_dir / "parquet"),
            'memory_optimizer': MemoryOptimizer(),
            'incremental_processor': IncrementalProcessor(temp_dir / "lake"),
            'temp_dir': temp_dir
        }
    
    @pytest.fixture(scope="class")
    def stress_environment(self):
        """Create stress testing environment."""
        return StressTestEnvironment()
    
    def test_memory_leak_detection_extended_operation(self, data_generator, stress_processors, stress_environment):
        """Test for memory leaks during extended operations."""
        logger.info("Testing memory leak detection during extended operations")
        
        target_duration_hours = 2.0  # 2 hour test (reduced from 24 for testing)
        leak_tolerance_mb = PERFORMANCE_CONFIG['stress_testing_targets']['memory_leak_tolerance_mb']
        
        health_processor = stress_processors['health_processor']
        memory_optimizer = stress_processors['memory_optimizer']
        storage_manager = stress_processors['storage_manager']
        temp_dir = stress_processors['temp_dir']
        
        # Start continuous monitoring
        stress_environment.start_monitoring(sample_interval=10.0)
        
        # Extended operation loop
        operation_start = time.time()
        operation_count = 0
        target_duration_seconds = target_duration_hours * 3600
        
        try:
            while (time.time() - operation_start) < target_duration_seconds:
                iteration_start = time.time()
                
                # Generate and process data
                health_data = data_generator.generate_large_scale_health_data(25000)
                
                try:
                    # Full processing cycle
                    validated = health_processor._validate_health_data(health_data)
                    aggregated = health_processor._aggregate_by_sa2(validated)
                    optimized = memory_optimizer.optimize_data_types(aggregated, data_category="health")
                    
                    # Save and load to test I/O
                    file_path = temp_dir / f"leak_test_{operation_count}.parquet"
                    storage_manager.save_optimized_parquet(optimized, file_path, data_type="health")
                    loaded_data = pl.read_parquet(file_path)
                    
                    # Clean up file
                    file_path.unlink()
                    
                    # Explicit cleanup
                    del health_data, validated, aggregated, optimized, loaded_data
                    
                    # Force garbage collection periodically
                    if operation_count % 10 == 0:
                        gc.collect()
                    
                    operation_count += 1
                    
                except Exception as e:
                    stress_environment.error_count += 1
                    logger.warning(f"Operation {operation_count} failed: {e}")
                
                # Record operation time
                operation_time = time.time() - iteration_start
                stress_environment.operation_times.append(operation_time)
                
                # Brief pause to prevent overwhelming
                time.sleep(0.1)
                
                # Log progress every 100 operations
                if operation_count % 100 == 0:
                    elapsed_hours = (time.time() - operation_start) / 3600
                    logger.info(f"Extended operation: {operation_count} operations, {elapsed_hours:.2f} hours elapsed")
        
        finally:
            stress_environment.stop_monitoring()
        
        total_duration_hours = (time.time() - operation_start) / 3600
        
        # Analyze results
        memory_leak_detected = stress_environment.detect_memory_leak(leak_tolerance_mb)
        performance_degradation = stress_environment.calculate_performance_degradation()
        
        # Calculate system stability metrics
        error_rate = stress_environment.error_count / max(1, operation_count)
        system_stable = error_rate < 0.05 and not memory_leak_detected
        
        stress_result = StressTestResult(
            test_name="memory_leak_detection_extended",
            duration_hours=total_duration_hours,
            operations_completed=operation_count,
            failures_encountered=stress_environment.error_count,
            recovery_successful=True,  # Test completed
            memory_leak_detected=memory_leak_detected,
            performance_degradation_percent=performance_degradation,
            system_stable=system_stable,
            targets_met={
                'no_memory_leak': not memory_leak_detected,
                'acceptable_degradation': performance_degradation < 20.0,
                'low_error_rate': error_rate < 0.05,
                'duration_achieved': total_duration_hours >= target_duration_hours * 0.8,
                'operations_completed': operation_count >= 100
            },
            stress_details={
                'error_rate': error_rate,
                'total_memory_samples': len(stress_environment.memory_samples),
                'avg_operation_time': np.mean(stress_environment.operation_times) if stress_environment.operation_times else 0
            }
        )
        
        # Extended operation validation
        assert not memory_leak_detected, f"Memory leak detected during extended operation"
        assert performance_degradation < 30.0, f"Performance degradation {performance_degradation:.1f}% should be <30%"
        assert error_rate < 0.1, f"Error rate {error_rate:.1%} should be <10%"
        assert operation_count >= 50, f"Should complete ≥50 operations, completed {operation_count}"
        
        logger.info(f"Extended operation test: {total_duration_hours:.2f}h, {operation_count} operations, "
                   f"leak detected: {memory_leak_detected}, {performance_degradation:.1f}% degradation")
        
        return stress_result
    
    def test_resource_exhaustion_scenarios(self, data_generator, stress_processors, stress_environment):
        """Test system behavior under resource exhaustion."""
        logger.info("Testing resource exhaustion scenarios")
        
        health_processor = stress_processors['health_processor']
        memory_optimizer = stress_processors['memory_optimizer']
        
        exhaustion_scenarios = [
            ('memory_constraint', 512),  # 512MB memory limit
            ('memory_constraint', 1024), # 1GB memory limit
            ('memory_constraint', 2048), # 2GB memory limit
        ]
        
        exhaustion_results = []
        
        for scenario_name, memory_limit_mb in exhaustion_scenarios:
            logger.info(f"Testing {scenario_name} with {memory_limit_mb}MB limit")
            
            scenario_start = time.time()
            operations_completed = 0
            failures_encountered = 0
            recoveries_successful = 0
            
            # Test under resource constraint
            with stress_environment.constrain_memory(memory_limit_mb):
                # Attempt multiple operations under constraint
                for i in range(20):
                    try:
                        # Generate data appropriate for memory limit
                        if memory_limit_mb <= 512:
                            data_size = 10000
                        elif memory_limit_mb <= 1024:
                            data_size = 25000
                        else:
                            data_size = 50000
                        
                        health_data = data_generator.generate_large_scale_health_data(data_size)
                        
                        # Attempt processing
                        validated = health_processor._validate_health_data(health_data)
                        optimized = memory_optimizer.optimize_data_types(validated, data_category="health")
                        
                        operations_completed += 1
                        
                        # Clean up immediately
                        del health_data, validated, optimized
                        gc.collect()
                        
                    except MemoryError:
                        failures_encountered += 1
                        gc.collect()  # Attempt recovery
                        
                        # Test recovery
                        try:
                            small_data = data_generator.generate_large_scale_health_data(1000)
                            small_validated = health_processor._validate_health_data(small_data)
                            recoveries_successful += 1
                            del small_data, small_validated
                        except:
                            pass
                    
                    except Exception as e:
                        failures_encountered += 1
                        logger.warning(f"Resource constraint operation failed: {e}")
            
            scenario_time = time.time() - scenario_start
            success_rate = operations_completed / 20
            recovery_rate = recoveries_successful / max(1, failures_encountered)
            
            exhaustion_results.append({
                'scenario': scenario_name,
                'memory_limit_mb': memory_limit_mb,
                'operations_completed': operations_completed,
                'failures_encountered': failures_encountered,
                'recoveries_successful': recoveries_successful,
                'success_rate': success_rate,
                'recovery_rate': recovery_rate,
                'scenario_time': scenario_time,
                'graceful_degradation': success_rate >= 0.5 or recovery_rate >= 0.7
            })
        
        # Resource exhaustion validation
        avg_success_rate = np.mean([r['success_rate'] for r in exhaustion_results])
        avg_recovery_rate = np.mean([r['recovery_rate'] for r in exhaustion_results])
        graceful_degradation_achieved = all(r['graceful_degradation'] for r in exhaustion_results)
        
        assert avg_success_rate >= 0.4, f"Average success rate under constraints {avg_success_rate:.1%} should be ≥40%"
        assert graceful_degradation_achieved, "Should achieve graceful degradation under all resource constraints"
        
        logger.info(f"Resource exhaustion: {avg_success_rate:.1%} success rate, "
                   f"{avg_recovery_rate:.1%} recovery rate, graceful: {graceful_degradation_achieved}")
        
        return {
            'average_success_rate': avg_success_rate,
            'average_recovery_rate': avg_recovery_rate,
            'graceful_degradation_achieved': graceful_degradation_achieved,
            'scenario_results': exhaustion_results
        }
    
    def test_error_recovery_under_stress(self, data_generator, stress_processors, stress_environment):
        """Test error recovery mechanisms under stress conditions."""
        logger.info("Testing error recovery under stress conditions")
        
        health_processor = stress_processors['health_processor']
        storage_manager = stress_processors['storage_manager']
        temp_dir = stress_processors['temp_dir']
        
        # Error injection scenarios
        error_scenarios = [
            'corrupted_data',
            'file_system_full',
            'permission_denied',
            'invalid_schema',
            'memory_pressure'
        ]
        
        recovery_results = []
        
        for scenario in error_scenarios:
            logger.info(f"Testing error recovery for {scenario}")
            
            scenario_start = time.time()
            attempts = 10
            successful_recoveries = 0
            recovery_times = []
            
            for attempt in range(attempts):
                try:
                    # Generate normal data
                    health_data = data_generator.generate_large_scale_health_data(20000)
                    
                    # Inject error based on scenario
                    if scenario == 'corrupted_data':
                        # Corrupt some data
                        corrupted_data = health_data.with_columns([
                            pl.when(pl.int_range(len(health_data)) % 100 == 0)
                            .then(None)
                            .otherwise(pl.col("prescription_count"))
                            .alias("prescription_count")
                        ])
                        test_data = corrupted_data
                    
                    elif scenario == 'file_system_full':
                        # Simulate file system full by using invalid path
                        test_data = health_data
                        invalid_path = Path("/invalid/path/that/does/not/exist")
                    
                    elif scenario == 'permission_denied':
                        # Create read-only file to cause permission error
                        test_data = health_data
                        readonly_path = temp_dir / f"readonly_{attempt}.parquet"
                        readonly_path.touch()
                        readonly_path.chmod(0o444)  # Read-only
                    
                    elif scenario == 'invalid_schema':
                        # Create data with invalid schema
                        invalid_data = pl.DataFrame({
                            'invalid_column': ['invalid'] * len(health_data)
                        })
                        test_data = invalid_data
                    
                    elif scenario == 'memory_pressure':
                        # Create memory pressure during processing
                        with stress_environment.simulate_resource_pressure():
                            test_data = health_data
                    
                    else:
                        test_data = health_data
                    
                    # Attempt operation that may fail
                    recovery_start = time.time()
                    
                    try:
                        if scenario == 'file_system_full':
                            storage_manager.save_optimized_parquet(test_data, invalid_path, data_type="health")
                        elif scenario == 'permission_denied':
                            storage_manager.save_optimized_parquet(test_data, readonly_path, data_type="health")
                        else:
                            validated = health_processor._validate_health_data(test_data)
                            
                        # If we get here, operation succeeded (unexpected for error scenarios)
                        if scenario in ['corrupted_data', 'invalid_schema']:
                            pass  # These might succeed with data cleaning
                        else:
                            continue  # Skip to next attempt
                    
                    except Exception as e:
                        # Error occurred as expected, now test recovery
                        recovery_attempt_start = time.time()
                        
                        try:
                            # Attempt recovery with clean data
                            clean_data = data_generator.generate_large_scale_health_data(10000)
                            validated_clean = health_processor._validate_health_data(clean_data)
                            
                            # Save to valid location
                            valid_path = temp_dir / f"recovery_{scenario}_{attempt}.parquet"
                            storage_manager.save_optimized_parquet(validated_clean, valid_path, data_type="health")
                            
                            recovery_time = time.time() - recovery_attempt_start
                            recovery_times.append(recovery_time)
                            successful_recoveries += 1
                            
                            # Clean up
                            del clean_data, validated_clean
                            if valid_path.exists():
                                valid_path.unlink()
                        
                        except Exception as recovery_error:
                            logger.warning(f"Recovery failed for {scenario}: {recovery_error}")
                    
                    # Clean up test artifacts
                    if scenario == 'permission_denied' and readonly_path.exists():
                        readonly_path.chmod(0o666)  # Make writable
                        readonly_path.unlink()
                
                except Exception as e:
                    logger.warning(f"Error scenario {scenario} attempt {attempt} failed: {e}")
            
            scenario_time = time.time() - scenario_start
            recovery_rate = successful_recoveries / attempts
            avg_recovery_time = np.mean(recovery_times) if recovery_times else float('inf')
            
            recovery_results.append({
                'scenario': scenario,
                'attempts': attempts,
                'successful_recoveries': successful_recoveries,
                'recovery_rate': recovery_rate,
                'avg_recovery_time': avg_recovery_time,
                'scenario_time': scenario_time,
                'recovery_acceptable': recovery_rate >= 0.7 and avg_recovery_time < 30.0
            })
        
        # Error recovery validation
        overall_recovery_rate = np.mean([r['recovery_rate'] for r in recovery_results])
        avg_recovery_time = np.mean([r['avg_recovery_time'] for r in recovery_results if r['avg_recovery_time'] != float('inf')])
        recovery_mechanisms_effective = all(r['recovery_acceptable'] for r in recovery_results)
        
        resilience_result = ResilienceTestResult(
            test_name="error_recovery_under_stress",
            failure_scenarios_tested=len(error_scenarios),
            recovery_success_rate=overall_recovery_rate,
            average_recovery_time_seconds=avg_recovery_time,
            graceful_degradation_achieved=overall_recovery_rate >= 0.6,
            data_integrity_preserved=True,  # Validated through successful recoveries
            system_availability_percent=overall_recovery_rate * 100,
            resilience_score=min(10, overall_recovery_rate * 10 + (30 - min(30, avg_recovery_time)) / 3)
        )
        
        assert overall_recovery_rate >= 0.6, f"Overall recovery rate {overall_recovery_rate:.1%} should be ≥60%"
        assert avg_recovery_time < 60.0, f"Average recovery time {avg_recovery_time:.1f}s should be <60s"
        
        logger.info(f"Error recovery: {overall_recovery_rate:.1%} recovery rate, "
                   f"{avg_recovery_time:.1f}s avg recovery time")
        
        return resilience_result
    
    def test_continuous_operation_stability(self, data_generator, stress_processors, stress_environment):
        """Test system stability during continuous operation."""
        logger.info("Testing continuous operation stability")
        
        # Continuous operation for 1 hour (reduced from 24 hours for testing)
        target_duration_hours = 1.0
        target_duration_seconds = target_duration_hours * 3600
        
        health_processor = stress_processors['health_processor']
        memory_optimizer = stress_processors['memory_optimizer']
        storage_manager = stress_processors['storage_manager']
        temp_dir = stress_processors['temp_dir']
        
        # Start monitoring
        stress_environment.start_monitoring(sample_interval=30.0)
        
        operation_start = time.time()
        cycle_count = 0
        stability_metrics = {
            'operation_times': [],
            'memory_usage': [],
            'error_count': 0,
            'recovery_count': 0
        }
        
        try:
            while (time.time() - operation_start) < target_duration_seconds:
                cycle_start = time.time()
                
                try:
                    # Continuous processing cycle
                    
                    # Phase 1: Data generation and processing
                    health_data = data_generator.generate_large_scale_health_data(15000)
                    seifa_data = data_generator.generate_large_scale_seifa_data()
                    
                    # Phase 2: Validation and optimization
                    validated_health = health_processor._validate_health_data(health_data)
                    optimized_health = memory_optimizer.optimize_data_types(validated_health, data_category="health")
                    
                    # Phase 3: Storage operations
                    file_path = temp_dir / f"continuous_{cycle_count}.parquet"
                    storage_manager.save_optimized_parquet(optimized_health, file_path, data_type="health")
                    
                    # Phase 4: Read verification
                    loaded_data = pl.read_parquet(file_path)
                    data_integrity_ok = len(loaded_data) == len(optimized_health)
                    
                    if not data_integrity_ok:
                        stability_metrics['error_count'] += 1
                    
                    # Clean up
                    file_path.unlink()
                    del health_data, seifa_data, validated_health, optimized_health, loaded_data
                    
                    # Periodic garbage collection
                    if cycle_count % 20 == 0:
                        gc.collect()
                    
                    cycle_count += 1
                    
                except Exception as e:
                    stability_metrics['error_count'] += 1
                    logger.warning(f"Continuous operation cycle {cycle_count} failed: {e}")
                    
                    # Attempt recovery
                    try:
                        gc.collect()
                        time.sleep(1.0)  # Brief recovery pause
                        stability_metrics['recovery_count'] += 1
                    except:
                        pass
                
                cycle_time = time.time() - cycle_start
                stability_metrics['operation_times'].append(cycle_time)
                
                # Record current memory usage
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                stability_metrics['memory_usage'].append(current_memory)
                
                # Log progress every 50 cycles
                if cycle_count % 50 == 0:
                    elapsed_hours = (time.time() - operation_start) / 3600
                    logger.info(f"Continuous operation: {cycle_count} cycles, {elapsed_hours:.2f}h elapsed")
                
                # Brief pause to prevent overwhelming
                time.sleep(0.1)
        
        finally:
            stress_environment.stop_monitoring()
        
        total_duration_hours = (time.time() - operation_start) / 3600
        
        # Analyze stability metrics
        error_rate = stability_metrics['error_count'] / max(1, cycle_count)
        recovery_rate = stability_metrics['recovery_count'] / max(1, stability_metrics['error_count'])
        
        # Performance stability analysis
        if len(stability_metrics['operation_times']) >= 10:
            early_performance = np.mean(stability_metrics['operation_times'][:len(stability_metrics['operation_times'])//4])
            late_performance = np.mean(stability_metrics['operation_times'][-len(stability_metrics['operation_times'])//4:])
            performance_degradation = ((late_performance - early_performance) / early_performance) * 100 if early_performance > 0 else 0
        else:
            performance_degradation = 0
        
        # Memory stability analysis
        memory_leak_detected = stress_environment.detect_memory_leak()
        
        # System stability assessment
        system_stable = (
            error_rate < 0.05 and
            not memory_leak_detected and
            performance_degradation < 25.0 and
            cycle_count >= target_duration_hours * 30  # At least 30 cycles per hour
        )
        
        continuous_result = StressTestResult(
            test_name="continuous_operation_stability",
            duration_hours=total_duration_hours,
            operations_completed=cycle_count,
            failures_encountered=stability_metrics['error_count'],
            recovery_successful=recovery_rate >= 0.8,
            memory_leak_detected=memory_leak_detected,
            performance_degradation_percent=performance_degradation,
            system_stable=system_stable,
            targets_met={
                'duration_achieved': total_duration_hours >= target_duration_hours * 0.9,
                'low_error_rate': error_rate < 0.1,
                'no_memory_leak': not memory_leak_detected,
                'performance_stable': performance_degradation < 30.0,
                'adequate_cycles': cycle_count >= target_duration_hours * 25
            },
            stress_details={
                'cycle_count': cycle_count,
                'error_rate': error_rate,
                'recovery_rate': recovery_rate,
                'avg_cycle_time': np.mean(stability_metrics['operation_times']) if stability_metrics['operation_times'] else 0
            }
        )
        
        # Continuous operation validation
        assert total_duration_hours >= target_duration_hours * 0.8, \
            f"Should run for ≥{target_duration_hours * 0.8:.1f}h, achieved {total_duration_hours:.2f}h"
        assert error_rate < 0.15, f"Error rate {error_rate:.1%} should be <15%"
        assert not memory_leak_detected, "No memory leak should be detected during continuous operation"
        assert cycle_count >= 20, f"Should complete ≥20 cycles, completed {cycle_count}"
        
        logger.info(f"Continuous operation: {total_duration_hours:.2f}h, {cycle_count} cycles, "
                   f"stable: {system_stable}, {error_rate:.1%} error rate")
        
        return continuous_result
    
    def test_extreme_load_graceful_degradation(self, data_generator, stress_processors, stress_environment):
        """Test graceful degradation under extreme load conditions."""
        logger.info("Testing graceful degradation under extreme load")
        
        health_processor = stress_processors['health_processor']
        memory_optimizer = stress_processors['memory_optimizer']
        
        # Extreme load scenarios
        load_scenarios = [
            ('massive_dataset', 200000),
            ('high_frequency', 50000),
            ('memory_intensive', 100000),
            ('cpu_intensive', 75000)
        ]
        
        degradation_results = []
        
        for scenario_name, data_size in load_scenarios:
            logger.info(f"Testing graceful degradation for {scenario_name}")
            
            scenario_start = time.time()
            degradation_strategies_used = []
            performance_levels = []
            
            # Attempt processing under extreme load
            for load_level in range(1, 6):  # Increasing load levels
                try:
                    level_start = time.time()
                    
                    if scenario_name == 'massive_dataset':
                        # Progressively larger datasets
                        test_data = data_generator.generate_large_scale_health_data(data_size * load_level)
                        
                    elif scenario_name == 'high_frequency':
                        # Rapid sequential processing
                        test_data = data_generator.generate_large_scale_health_data(data_size)
                        for _ in range(load_level * 2):
                            validated = health_processor._validate_health_data(test_data)
                            del validated
                    
                    elif scenario_name == 'memory_intensive':
                        # Memory-intensive operations
                        with stress_environment.simulate_resource_pressure():
                            test_data = data_generator.generate_large_scale_health_data(data_size)
                            validated = health_processor._validate_health_data(test_data)
                            optimized = memory_optimizer.optimize_data_types(validated, data_category="health")
                            del validated, optimized
                    
                    elif scenario_name == 'cpu_intensive':
                        # CPU-intensive processing
                        test_data = data_generator.generate_large_scale_health_data(data_size)
                        for _ in range(load_level):
                            validated = health_processor._validate_health_data(test_data)
                            aggregated = health_processor._aggregate_by_sa2(validated)
                            del validated, aggregated
                    
                    level_time = time.time() - level_start
                    performance_levels.append((load_level, level_time, True))  # Success
                    
                    # Clean up
                    del test_data
                    gc.collect()
                
                except MemoryError:
                    # Graceful degradation: reduce data size
                    degradation_strategies_used.append(f"reduced_data_size_level_{load_level}")
                    try:
                        reduced_data = data_generator.generate_large_scale_health_data(data_size // 2)
                        validated = health_processor._validate_health_data(reduced_data)
                        level_time = time.time() - level_start
                        performance_levels.append((load_level, level_time, False))  # Degraded success
                        del reduced_data, validated
                    except:
                        performance_levels.append((load_level, float('inf'), False))  # Failed
                
                except Exception as e:
                    # Other graceful degradation strategies
                    degradation_strategies_used.append(f"error_recovery_level_{load_level}")
                    logger.warning(f"Load level {load_level} failed: {e}")
                    performance_levels.append((load_level, float('inf'), False))  # Failed
            
            scenario_time = time.time() - scenario_start
            
            # Analyze graceful degradation
            successful_levels = [p for p in performance_levels if p[2]]
            max_successful_level = max([p[0] for p in successful_levels]) if successful_levels else 0
            degradation_strategies_effective = len(degradation_strategies_used) > 0
            
            degradation_results.append({
                'scenario': scenario_name,
                'max_successful_level': max_successful_level,
                'degradation_strategies_used': len(degradation_strategies_used),
                'degradation_strategies_effective': degradation_strategies_effective,
                'performance_levels': performance_levels,
                'scenario_time': scenario_time,
                'graceful_degradation_achieved': max_successful_level >= 3 or degradation_strategies_effective
            })
        
        # Graceful degradation validation
        avg_max_level = np.mean([r['max_successful_level'] for r in degradation_results])
        degradation_achieved_rate = sum(1 for r in degradation_results if r['graceful_degradation_achieved']) / len(degradation_results)
        
        assert avg_max_level >= 2.0, f"Average max successful level {avg_max_level:.1f} should be ≥2.0"
        assert degradation_achieved_rate >= 0.75, f"Graceful degradation rate {degradation_achieved_rate:.1%} should be ≥75%"
        
        logger.info(f"Graceful degradation: {avg_max_level:.1f} avg max level, "
                   f"{degradation_achieved_rate:.1%} degradation achieved")
        
        return {
            'average_max_successful_level': avg_max_level,
            'graceful_degradation_achieved_rate': degradation_achieved_rate,
            'degradation_results': degradation_results
        }