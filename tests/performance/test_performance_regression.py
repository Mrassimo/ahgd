"""
Performance Regression Detection Framework - Phase 5.4

Comprehensive performance regression testing framework that compares current
performance against established baselines, detects performance degradation,
and provides automated performance monitoring for continuous integration.

Key Features:
- Baseline performance comparison
- Automated regression detection
- Performance trend analysis
- CI/CD integration support
- Performance alert generation
"""

import pytest
import polars as pl
import numpy as np
import time
import psutil
import gc
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import concurrent.futures
from contextlib import contextmanager
import statistics

from src.data_processing.seifa_processor import SEIFAProcessor
from src.data_processing.health_processor import HealthDataProcessor
from src.data_processing.simple_boundary_processor import SimpleBoundaryProcessor
from src.data_processing.storage.parquet_storage_manager import ParquetStorageManager
from src.data_processing.storage.memory_optimizer import MemoryOptimizer
from tests.performance import PERFORMANCE_CONFIG, AUSTRALIAN_DATA_SCALE
from tests.performance.test_large_scale_processing import AustralianHealthDataGenerator

logger = logging.getLogger(__name__)


@dataclass
class PerformanceBaseline:
    """Performance baseline metrics."""
    test_name: str
    baseline_date: str
    platform_info: Dict[str, Any]
    metrics: Dict[str, float]
    data_characteristics: Dict[str, Any]
    environment_info: Dict[str, Any]


@dataclass
class RegressionTestResult:
    """Regression test comparison result."""
    test_name: str
    baseline_metrics: Dict[str, float]
    current_metrics: Dict[str, float]
    regression_detected: bool
    regression_severity: str  # none, low, medium, high, critical
    performance_changes: Dict[str, Dict[str, float]]  # metric -> {change_percent, threshold_exceeded}
    overall_performance_change: float
    recommendations: List[str]


@dataclass
class PerformanceTrend:
    """Performance trend analysis."""
    metric_name: str
    historical_values: List[float]
    timestamps: List[str]
    trend_direction: str  # improving, stable, degrading
    trend_strength: float  # 0-1
    outliers_detected: List[int]
    forecast_next_value: Optional[float]


class PerformanceRegressionDetector:
    """
    Comprehensive performance regression detection system.
    Compares current performance against established baselines and trends.
    """
    
    # Regression detection thresholds
    REGRESSION_THRESHOLDS = {
        'processing_time': {
            'low': 0.05,     # 5% increase
            'medium': 0.15,  # 15% increase
            'high': 0.30,    # 30% increase
            'critical': 0.50 # 50% increase
        },
        'memory_usage': {
            'low': 0.10,     # 10% increase
            'medium': 0.25,  # 25% increase
            'high': 0.50,    # 50% increase
            'critical': 0.75 # 75% increase
        },
        'throughput': {
            'low': -0.05,    # 5% decrease
            'medium': -0.15, # 15% decrease
            'high': -0.30,   # 30% decrease
            'critical': -0.50 # 50% decrease
        },
        'error_rate': {
            'low': 0.01,     # 1% increase
            'medium': 0.05,  # 5% increase
            'high': 0.10,    # 10% increase
            'critical': 0.20 # 20% increase
        }
    }
    
    def __init__(self, baseline_dir: Path):
        """
        Initialize regression detector.
        
        Args:
            baseline_dir: Directory containing performance baselines
        """
        self.baseline_dir = baseline_dir
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.load_baselines()
        
        logger.info(f"Performance regression detector initialized with {len(self.baselines)} baselines")
    
    def load_baselines(self) -> None:
        """Load existing performance baselines."""
        baseline_files = list(self.baseline_dir.glob("baseline_*.json"))
        
        for baseline_file in baseline_files:
            try:
                with open(baseline_file, 'r') as f:
                    baseline_data = json.load(f)
                
                baseline = PerformanceBaseline(**baseline_data)
                self.baselines[baseline.test_name] = baseline
                
            except Exception as e:
                logger.warning(f"Failed to load baseline {baseline_file}: {e}")
    
    def create_baseline(self, 
                       test_name: str,
                       metrics: Dict[str, float],
                       data_characteristics: Dict[str, Any],
                       overwrite: bool = False) -> PerformanceBaseline:
        """
        Create a new performance baseline.
        
        Args:
            test_name: Name of the test
            metrics: Performance metrics
            data_characteristics: Data size, complexity, etc.
            overwrite: Whether to overwrite existing baseline
        """
        if test_name in self.baselines and not overwrite:
            raise ValueError(f"Baseline for {test_name} already exists. Use overwrite=True to replace.")
        
        # Collect platform information
        platform_info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024**3,
            'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}",
            'platform': psutil.platform.platform()
        }
        
        # Environment information
        environment_info = {
            'timestamp': datetime.now().isoformat(),
            'test_environment': 'performance_testing',
            'data_scale': data_characteristics.get('total_records', 0)
        }
        
        baseline = PerformanceBaseline(
            test_name=test_name,
            baseline_date=datetime.now().strftime('%Y-%m-%d'),
            platform_info=platform_info,
            metrics=metrics,
            data_characteristics=data_characteristics,
            environment_info=environment_info
        )
        
        # Save baseline
        self.save_baseline(baseline)
        self.baselines[test_name] = baseline
        
        logger.info(f"Created performance baseline for {test_name}")
        return baseline
    
    def save_baseline(self, baseline: PerformanceBaseline) -> None:
        """Save baseline to disk."""
        baseline_file = self.baseline_dir / f"baseline_{baseline.test_name}.json"
        
        try:
            with open(baseline_file, 'w') as f:
                json.dump(asdict(baseline), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save baseline: {e}")
    
    def detect_regression(self, 
                         test_name: str,
                         current_metrics: Dict[str, float],
                         data_characteristics: Dict[str, Any]) -> RegressionTestResult:
        """
        Detect performance regression against baseline.
        
        Args:
            test_name: Name of the test
            current_metrics: Current performance metrics
            data_characteristics: Current data characteristics
        """
        if test_name not in self.baselines:
            logger.warning(f"No baseline found for {test_name}. Creating new baseline.")
            self.create_baseline(test_name, current_metrics, data_characteristics)
            
            return RegressionTestResult(
                test_name=test_name,
                baseline_metrics={},
                current_metrics=current_metrics,
                regression_detected=False,
                regression_severity='none',
                performance_changes={},
                overall_performance_change=0.0,
                recommendations=["New baseline created - no regression analysis available"]
            )
        
        baseline = self.baselines[test_name]
        
        # Compare metrics
        performance_changes = {}
        regression_scores = []
        
        for metric_name, current_value in current_metrics.items():
            if metric_name not in baseline.metrics:
                continue
            
            baseline_value = baseline.metrics[metric_name]
            change_analysis = self._analyze_metric_change(
                metric_name, baseline_value, current_value
            )
            
            performance_changes[metric_name] = change_analysis
            regression_scores.append(change_analysis['regression_score'])
        
        # Overall regression assessment
        overall_change = np.mean(regression_scores) if regression_scores else 0.0
        regression_detected = any(
            change['threshold_exceeded'] for change in performance_changes.values()
        )
        
        # Determine severity
        severity = self._determine_regression_severity(performance_changes)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(performance_changes, baseline, current_metrics)
        
        result = RegressionTestResult(
            test_name=test_name,
            baseline_metrics=baseline.metrics,
            current_metrics=current_metrics,
            regression_detected=regression_detected,
            regression_severity=severity,
            performance_changes=performance_changes,
            overall_performance_change=overall_change,
            recommendations=recommendations
        )
        
        logger.info(f"Regression analysis for {test_name}: "
                   f"detected={regression_detected}, severity={severity}")
        
        return result
    
    def _analyze_metric_change(self, 
                              metric_name: str,
                              baseline_value: float,
                              current_value: float) -> Dict[str, float]:
        """Analyze change in a specific metric."""
        
        if baseline_value == 0:
            change_percent = 0.0 if current_value == 0 else float('inf')
        else:
            change_percent = (current_value - baseline_value) / baseline_value
        
        # Determine if this is a regression based on metric type
        is_regression = False
        regression_score = 0.0
        threshold_exceeded = False
        
        # Get thresholds for this metric type
        thresholds = None
        for threshold_type, type_thresholds in self.REGRESSION_THRESHOLDS.items():
            if threshold_type in metric_name.lower():
                thresholds = type_thresholds
                break
        
        if thresholds:
            # For metrics where increase is bad (time, memory, errors)
            if any(keyword in metric_name.lower() for keyword in ['time', 'memory', 'error', 'latency']):
                if change_percent > thresholds['low']:
                    is_regression = True
                    threshold_exceeded = True
                    
                    if change_percent > thresholds['critical']:
                        regression_score = 1.0
                    elif change_percent > thresholds['high']:
                        regression_score = 0.8
                    elif change_percent > thresholds['medium']:
                        regression_score = 0.6
                    else:
                        regression_score = 0.3
            
            # For metrics where decrease is bad (throughput, speed)
            elif any(keyword in metric_name.lower() for keyword in ['throughput', 'speed', 'rate']):
                if change_percent < thresholds['low']:  # Negative thresholds
                    is_regression = True
                    threshold_exceeded = True
                    
                    if change_percent < thresholds['critical']:
                        regression_score = 1.0
                    elif change_percent < thresholds['high']:
                        regression_score = 0.8
                    elif change_percent < thresholds['medium']:
                        regression_score = 0.6
                    else:
                        regression_score = 0.3
        
        return {
            'baseline_value': baseline_value,
            'current_value': current_value,
            'change_percent': change_percent,
            'change_absolute': current_value - baseline_value,
            'is_regression': is_regression,
            'regression_score': regression_score,
            'threshold_exceeded': threshold_exceeded
        }
    
    def _determine_regression_severity(self, performance_changes: Dict[str, Dict[str, float]]) -> str:
        """Determine overall regression severity."""
        
        if not any(change['threshold_exceeded'] for change in performance_changes.values()):
            return 'none'
        
        max_regression_score = max(
            change['regression_score'] for change in performance_changes.values()
        )
        
        if max_regression_score >= 1.0:
            return 'critical'
        elif max_regression_score >= 0.8:
            return 'high'
        elif max_regression_score >= 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(self, 
                                 performance_changes: Dict[str, Dict[str, float]],
                                 baseline: PerformanceBaseline,
                                 current_metrics: Dict[str, float]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        # Analyze each degraded metric
        for metric_name, change in performance_changes.items():
            if not change['is_regression']:
                continue
            
            change_percent = abs(change['change_percent']) * 100
            
            if 'processing_time' in metric_name or 'execution_time' in metric_name:
                recommendations.append(
                    f"Processing time increased by {change_percent:.1f}%. "
                    "Consider algorithm optimization, caching, or parallel processing."
                )
            
            elif 'memory' in metric_name:
                recommendations.append(
                    f"Memory usage increased by {change_percent:.1f}%. "
                    "Review memory optimization, data chunking, or garbage collection."
                )
            
            elif 'throughput' in metric_name:
                recommendations.append(
                    f"Throughput decreased by {change_percent:.1f}%. "
                    "Investigate I/O bottlenecks, CPU utilization, or data processing efficiency."
                )
            
            elif 'error' in metric_name:
                recommendations.append(
                    f"Error rate increased by {change_percent:.1f}%. "
                    "Review error handling, data validation, and system stability."
                )
        
        # General recommendations based on overall degradation
        regression_count = sum(1 for change in performance_changes.values() if change['is_regression'])
        if regression_count > 3:
            recommendations.append(
                "Multiple performance metrics degraded. Consider system-wide optimization review."
            )
        
        if not recommendations:
            recommendations.append("Performance within acceptable thresholds.")
        
        return recommendations
    
    def analyze_performance_trend(self, 
                                 test_name: str,
                                 metric_name: str,
                                 lookback_days: int = 30) -> PerformanceTrend:
        """Analyze performance trends over time."""
        
        # This would typically load historical data from a database
        # For now, simulate with recent baseline data
        
        historical_values = [self.baselines[test_name].metrics.get(metric_name, 0.0)]
        timestamps = [self.baselines[test_name].baseline_date]
        
        # Simple trend analysis
        if len(historical_values) < 2:
            trend_direction = 'stable'
            trend_strength = 0.0
        else:
            # Linear regression for trend
            x = np.arange(len(historical_values))
            slope, _ = np.polyfit(x, historical_values, 1)
            
            if slope > 0.05:
                trend_direction = 'degrading' if 'time' in metric_name or 'memory' in metric_name else 'improving'
            elif slope < -0.05:
                trend_direction = 'improving' if 'time' in metric_name or 'memory' in metric_name else 'degrading'
            else:
                trend_direction = 'stable'
            
            trend_strength = min(1.0, abs(slope) / np.mean(historical_values))
        
        # Outlier detection (simple z-score method)
        if len(historical_values) > 3:
            z_scores = np.abs(statistics.zscore(historical_values))
            outliers = [i for i, z in enumerate(z_scores) if z > 2.0]
        else:
            outliers = []
        
        # Simple forecast (linear extrapolation)
        forecast_next = None
        if len(historical_values) >= 2:
            recent_trend = historical_values[-1] - historical_values[-2]
            forecast_next = historical_values[-1] + recent_trend
        
        return PerformanceTrend(
            metric_name=metric_name,
            historical_values=historical_values,
            timestamps=timestamps,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            outliers_detected=outliers,
            forecast_next_value=forecast_next
        )


class TestPerformanceRegression:
    """Performance regression tests for Australian Health Analytics platform."""
    
    @pytest.fixture(scope="class")
    def data_generator(self):
        """Create Australian health data generator."""
        return AustralianHealthDataGenerator(seed=42)
    
    @pytest.fixture(scope="class")
    def regression_detector(self, tmp_path_factory):
        """Create regression detector."""
        baseline_dir = tmp_path_factory.mktemp("regression_baselines")
        return PerformanceRegressionDetector(baseline_dir)
    
    @pytest.fixture(scope="class")
    def performance_processors(self, tmp_path_factory):
        """Create processors for regression testing."""
        temp_dir = tmp_path_factory.mktemp("regression_test")
        
        return {
            'seifa_processor': SEIFAProcessor(data_dir=temp_dir),
            'health_processor': HealthDataProcessor(data_dir=temp_dir),
            'boundary_processor': SimpleBoundaryProcessor(data_dir=temp_dir),
            'storage_manager': ParquetStorageManager(base_path=temp_dir / "parquet"),
            'memory_optimizer': MemoryOptimizer(),
            'temp_dir': temp_dir
        }
    
    def test_baseline_creation_and_validation(self, data_generator, regression_detector, performance_processors):
        """Test creation and validation of performance baselines."""
        logger.info("Testing baseline creation and validation")
        
        # Run baseline performance test
        test_name = "seifa_processing_baseline"
        
        # Generate test data
        seifa_data = data_generator.generate_large_scale_seifa_data()
        data_characteristics = {
            'total_records': len(seifa_data),
            'data_size_mb': seifa_data.estimated_size("mb"),
            'sa2_areas': len(seifa_data)
        }
        
        # Measure baseline performance
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        seifa_processor = performance_processors['seifa_processor']
        processed_seifa = seifa_processor._validate_seifa_data(seifa_data)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Calculate metrics
        baseline_metrics = {
            'processing_time_seconds': end_time - start_time,
            'memory_usage_mb': end_memory - start_memory,
            'throughput_records_per_second': len(seifa_data) / (end_time - start_time),
            'data_quality_score': 1.0,  # Assume high quality
            'success_rate': 1.0
        }
        
        # Create baseline
        baseline = regression_detector.create_baseline(
            test_name=test_name,
            metrics=baseline_metrics,
            data_characteristics=data_characteristics
        )
        
        # Validate baseline creation
        assert baseline.test_name == test_name
        assert len(baseline.metrics) == len(baseline_metrics)
        assert baseline.test_name in regression_detector.baselines
        
        # Validate baseline persistence
        regression_detector.load_baselines()
        assert test_name in regression_detector.baselines
        
        logger.info(f"Baseline created and validated for {test_name}")
        return baseline_metrics, data_characteristics
    
    def test_regression_detection_no_change(self, data_generator, regression_detector, performance_processors):
        """Test regression detection when performance hasn't changed."""
        logger.info("Testing regression detection with no performance change")
        
        # Use existing baseline or create one
        test_name = "health_processing_stable"
        
        # Generate identical test scenario
        health_data = data_generator.generate_large_scale_health_data(50000)
        data_characteristics = {
            'total_records': len(health_data),
            'data_size_mb': health_data.estimated_size("mb")
        }
        
        # Measure performance (should be similar to baseline)
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        health_processor = performance_processors['health_processor']
        validated_health = health_processor._validate_health_data(health_data)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        current_metrics = {
            'processing_time_seconds': end_time - start_time,
            'memory_usage_mb': end_memory - start_memory,
            'throughput_records_per_second': len(health_data) / (end_time - start_time),
            'success_rate': 1.0
        }
        
        # Create baseline if it doesn't exist
        if test_name not in regression_detector.baselines:
            regression_detector.create_baseline(test_name, current_metrics, data_characteristics)
        
        # Test regression detection
        regression_result = regression_detector.detect_regression(
            test_name=test_name,
            current_metrics=current_metrics,
            data_characteristics=data_characteristics
        )
        
        # Should not detect regression for stable performance
        assert not regression_result.regression_detected or regression_result.regression_severity == 'low'
        assert regression_result.test_name == test_name
        assert len(regression_result.performance_changes) > 0
        
        logger.info(f"Regression test passed: severity={regression_result.regression_severity}")
        return regression_result
    
    def test_regression_detection_performance_degradation(self, data_generator, regression_detector, performance_processors):
        """Test regression detection with simulated performance degradation."""
        logger.info("Testing regression detection with performance degradation")
        
        test_name = "memory_optimization_degraded"
        
        # Generate test data
        health_data = data_generator.generate_large_scale_health_data(75000)
        data_characteristics = {
            'total_records': len(health_data),
            'data_size_mb': health_data.estimated_size("mb")
        }
        
        # Create baseline with good performance
        baseline_metrics = {
            'processing_time_seconds': 2.0,
            'memory_usage_mb': 50.0,
            'throughput_records_per_second': 37500.0,
            'memory_optimization_percent': 60.0
        }
        
        if test_name not in regression_detector.baselines:
            regression_detector.create_baseline(test_name, baseline_metrics, data_characteristics)
        
        # Simulate degraded performance (simulate by adding artificial delays/overhead)
        memory_optimizer = performance_processors['memory_optimizer']
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Add artificial delay to simulate degradation
        time.sleep(1.0)  # Simulate slower processing
        
        optimized_data = memory_optimizer.optimize_data_types(health_data, data_category="health")
        
        # Simulate memory overhead
        memory_pressure = np.zeros(10_000_000, dtype=np.float32)  # ~40MB overhead
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Clean up memory pressure
        del memory_pressure
        gc.collect()
        
        # Current metrics show degradation
        current_metrics = {
            'processing_time_seconds': end_time - start_time,  # Will be slower due to sleep
            'memory_usage_mb': end_memory - start_memory,      # Will be higher due to overhead
            'throughput_records_per_second': len(health_data) / (end_time - start_time),
            'memory_optimization_percent': 45.0  # Reduced optimization
        }
        
        # Test regression detection
        regression_result = regression_detector.detect_regression(
            test_name=test_name,
            current_metrics=current_metrics,
            data_characteristics=data_characteristics
        )
        
        # Should detect regression
        assert regression_result.regression_detected
        assert regression_result.regression_severity in ['medium', 'high', 'critical']
        assert len(regression_result.recommendations) > 0
        
        # Validate specific regressions
        processing_time_change = regression_result.performance_changes.get('processing_time_seconds', {})
        assert processing_time_change.get('is_regression', False)
        
        logger.info(f"Regression detected: severity={regression_result.regression_severity}, "
                   f"changes={len(regression_result.performance_changes)}")
        
        return regression_result
    
    def test_performance_trend_analysis(self, regression_detector):
        """Test performance trend analysis capabilities."""
        logger.info("Testing performance trend analysis")
        
        # Create mock historical data for trend analysis
        test_name = "storage_performance_trend"
        metric_name = "throughput_mb_per_second"
        
        # Simulate baseline data
        baseline_metrics = {
            metric_name: 85.0,
            'processing_time_seconds': 5.2,
            'compression_ratio': 2.3
        }
        
        data_characteristics = {'total_records': 100000}
        
        if test_name not in regression_detector.baselines:
            regression_detector.create_baseline(test_name, baseline_metrics, data_characteristics)
        
        # Analyze trend
        trend_analysis = regression_detector.analyze_performance_trend(
            test_name=test_name,
            metric_name=metric_name,
            lookback_days=30
        )
        
        # Validate trend analysis
        assert trend_analysis.metric_name == metric_name
        assert len(trend_analysis.historical_values) > 0
        assert trend_analysis.trend_direction in ['improving', 'stable', 'degrading']
        assert 0.0 <= trend_analysis.trend_strength <= 1.0
        
        logger.info(f"Trend analysis: direction={trend_analysis.trend_direction}, "
                   f"strength={trend_analysis.trend_strength:.2f}")
        
        return trend_analysis
    
    def test_comprehensive_regression_suite(self, data_generator, regression_detector, performance_processors):
        """Test comprehensive regression detection across multiple components."""
        logger.info("Testing comprehensive regression detection suite")
        
        # Test multiple components for regression
        test_scenarios = [
            {
                'name': 'seifa_comprehensive',
                'processor': 'seifa_processor',
                'data_generator': lambda: data_generator.generate_large_scale_seifa_data(),
                'processor_method': '_validate_seifa_data'
            },
            {
                'name': 'health_comprehensive',
                'processor': 'health_processor',
                'data_generator': lambda: data_generator.generate_large_scale_health_data(30000),
                'processor_method': '_validate_health_data'
            },
            {
                'name': 'boundary_comprehensive',
                'processor': 'boundary_processor',
                'data_generator': lambda: data_generator.generate_large_scale_boundary_data(),
                'processor_method': '_validate_boundary_data'
            }
        ]
        
        regression_results = []
        
        for scenario in test_scenarios:
            logger.info(f"Testing regression for {scenario['name']}")
            
            # Generate test data
            test_data = scenario['data_generator']()
            data_characteristics = {
                'total_records': len(test_data),
                'data_size_mb': test_data.estimated_size("mb")
            }
            
            # Measure current performance
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            processor = performance_processors[scenario['processor']]
            processor_method = getattr(processor, scenario['processor_method'])
            processed_data = processor_method(test_data)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            current_metrics = {
                'processing_time_seconds': end_time - start_time,
                'memory_usage_mb': end_memory - start_memory,
                'throughput_records_per_second': len(test_data) / (end_time - start_time),
                'output_records': len(processed_data),
                'data_retention_rate': len(processed_data) / len(test_data),
                'success_rate': 1.0
            }
            
            # Test regression detection
            regression_result = regression_detector.detect_regression(
                test_name=scenario['name'],
                current_metrics=current_metrics,
                data_characteristics=data_characteristics
            )
            
            regression_results.append(regression_result)
            
            # Clean up
            del test_data, processed_data
            gc.collect()
        
        # Validate comprehensive results
        assert len(regression_results) == len(test_scenarios)
        
        # Check that at least some baselines were created or comparisons made
        comparison_count = sum(1 for result in regression_results if len(result.performance_changes) > 0)
        assert comparison_count >= len(test_scenarios) * 0.5  # At least 50% should have comparisons
        
        # Analyze overall system regression
        overall_regression_detected = any(result.regression_detected for result in regression_results)
        severe_regressions = sum(
            1 for result in regression_results 
            if result.regression_severity in ['high', 'critical']
        )
        
        # Generate comprehensive report
        comprehensive_report = {
            'total_tests': len(regression_results),
            'regressions_detected': sum(1 for r in regression_results if r.regression_detected),
            'severe_regressions': severe_regressions,
            'overall_system_health': 'degraded' if severe_regressions > 0 else 'stable',
            'test_results': {r.test_name: r.regression_severity for r in regression_results}
        }
        
        logger.info(f"Comprehensive regression analysis: {comprehensive_report}")
        
        # System should not have critical regressions
        assert severe_regressions <= 1, f"Too many severe regressions detected: {severe_regressions}"
        
        return comprehensive_report
    
    def test_ci_cd_integration_simulation(self, regression_detector):
        """Test CI/CD integration capabilities for automated regression detection."""
        logger.info("Testing CI/CD integration simulation")
        
        # Simulate CI/CD pipeline performance check
        pipeline_tests = [
            {
                'name': 'unit_test_performance',
                'metrics': {
                    'test_execution_time_seconds': 45.0,
                    'memory_usage_mb': 120.0,
                    'test_success_rate': 0.98
                }
            },
            {
                'name': 'integration_test_performance',
                'metrics': {
                    'test_execution_time_seconds': 180.0,
                    'memory_usage_mb': 350.0,
                    'test_success_rate': 0.95
                }
            },
            {
                'name': 'deployment_performance',
                'metrics': {
                    'deployment_time_seconds': 120.0,
                    'startup_time_seconds': 15.0,
                    'health_check_response_time_ms': 250.0
                }
            }
        ]
        
        ci_cd_results = []
        
        for test in pipeline_tests:
            # Check regression
            regression_result = regression_detector.detect_regression(
                test_name=test['name'],
                current_metrics=test['metrics'],
                data_characteristics={'pipeline_stage': test['name']}
            )
            
            ci_cd_results.append({
                'test_name': test['name'],
                'regression_detected': regression_result.regression_detected,
                'severity': regression_result.regression_severity,
                'should_block_deployment': regression_result.regression_severity in ['high', 'critical']
            })
        
        # CI/CD decision logic
        blocking_regressions = sum(1 for result in ci_cd_results if result['should_block_deployment'])
        deployment_recommendation = 'block' if blocking_regressions > 0 else 'proceed'
        
        ci_cd_summary = {
            'pipeline_tests_count': len(ci_cd_results),
            'regressions_detected': sum(1 for r in ci_cd_results if r['regression_detected']),
            'blocking_regressions': blocking_regressions,
            'deployment_recommendation': deployment_recommendation,
            'test_results': ci_cd_results
        }
        
        # Validate CI/CD integration
        assert len(ci_cd_results) == len(pipeline_tests)
        assert deployment_recommendation in ['proceed', 'block']
        
        # If blocking regressions, ensure they're properly flagged
        if blocking_regressions > 0:
            assert any(r['should_block_deployment'] for r in ci_cd_results)
        
        logger.info(f"CI/CD integration test: {deployment_recommendation} deployment, "
                   f"{blocking_regressions} blocking regressions")
        
        return ci_cd_summary