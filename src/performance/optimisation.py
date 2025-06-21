"""
Performance analysis and optimisation tools for the AHGD ETL pipeline.

This module provides tools for:
- Performance bottleneck identification
- Optimisation recommendations
- Memory usage optimisation
- Query optimisation helpers
- Automated performance tuning
"""

import ast
import gc
import json
import re
import sqlite3
import sys
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from ..utils.logging import get_logger
from ..utils.interfaces import DataRecord, DataBatch
from .profiler import PerformanceProfiler
from .monitoring import SystemMonitor, PerformanceMonitor

logger = get_logger()


@dataclass
class PerformanceBottleneck:
    """Identified performance bottleneck."""
    bottleneck_id: str
    bottleneck_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    affected_operation: str
    impact_score: float  # 0-100
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'bottleneck_id': self.bottleneck_id,
            'bottleneck_type': self.bottleneck_type,
            'severity': self.severity,
            'description': self.description,
            'affected_operation': self.affected_operation,
            'impact_score': self.impact_score,
            'recommendations': self.recommendations,
            'metadata': self.metadata,
            'detected_at': self.detected_at.isoformat()
        }


@dataclass
class OptimisationRecommendation:
    """Performance optimisation recommendation."""
    recommendation_id: str
    category: str
    priority: str  # 'low', 'medium', 'high', 'critical'
    title: str
    description: str
    implementation_effort: str  # 'minimal', 'moderate', 'significant'
    expected_improvement: str
    code_changes: List[str] = field(default_factory=list)
    configuration_changes: List[str] = field(default_factory=list)
    estimated_impact: float = 0.0  # 0-100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'recommendation_id': self.recommendation_id,
            'category': self.category,
            'priority': self.priority,
            'title': self.title,
            'description': self.description,
            'implementation_effort': self.implementation_effort,
            'expected_improvement': self.expected_improvement,
            'code_changes': self.code_changes,
            'configuration_changes': self.configuration_changes,
            'estimated_impact': self.estimated_impact
        }


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis tools.
    
    Features:
    - Performance data analysis
    - Statistical performance evaluation
    - Trend analysis
    - Performance comparison
    - Automated insights generation
    """
    
    def __init__(self, data_retention_days: int = 30):
        self.data_retention_days = data_retention_days
        self.analysis_cache = {}
        self.insights = []
    
    def analyse_performance_data(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse performance data and generate insights."""
        analysis_results = {
            'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
            'data_summary': self._summarise_performance_data(performance_data),
            'statistical_analysis': self._perform_statistical_analysis(performance_data),
            'trend_analysis': self._analyse_trends(performance_data),
            'anomaly_detection': self._detect_anomalies(performance_data),
            'insights': self._generate_insights(performance_data)
        }
        
        return analysis_results
    
    def _summarise_performance_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for performance data."""
        summary = {
            'total_operations': 0,
            'unique_operations': 0,
            'time_range': {},
            'performance_metrics': {}
        }
        
        # Extract operation data
        operations = data.get('operations', [])
        if operations:
            durations = [op.get('duration', 0) for op in operations]
            throughputs = [op.get('throughput', 0) for op in operations]
            memory_usage = [op.get('memory_usage_mb', 0) for op in operations]
            
            summary.update({
                'total_operations': len(operations),
                'unique_operations': len(set(op.get('operation_name', '') for op in operations)),
                'performance_metrics': {
                    'duration': {
                        'mean': np.mean(durations) if durations else 0,
                        'median': np.median(durations) if durations else 0,
                        'std': np.std(durations) if durations else 0,
                        'min': np.min(durations) if durations else 0,
                        'max': np.max(durations) if durations else 0
                    },
                    'throughput': {
                        'mean': np.mean(throughputs) if throughputs else 0,
                        'median': np.median(throughputs) if throughputs else 0,
                        'std': np.std(throughputs) if throughputs else 0
                    },
                    'memory_usage': {
                        'mean': np.mean(memory_usage) if memory_usage else 0,
                        'median': np.median(memory_usage) if memory_usage else 0,
                        'max': np.max(memory_usage) if memory_usage else 0
                    }
                }
            })
        
        return summary
    
    def _perform_statistical_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis on performance data."""
        statistical_results = {}
        
        operations = data.get('operations', [])
        if not operations:
            return statistical_results
        
        # Group by operation type
        operation_groups = defaultdict(list)
        for op in operations:
            operation_groups[op.get('operation_name', 'unknown')].append(op)
        
        # Analyse each operation type
        for op_name, op_data in operation_groups.items():
            durations = [op.get('duration', 0) for op in op_data]
            
            # Calculate percentiles
            percentiles = {}
            for p in [50, 75, 90, 95, 99]:
                percentiles[f'p{p}'] = np.percentile(durations, p) if durations else 0
            
            # Identify outliers (using IQR method)
            q1 = np.percentile(durations, 25) if durations else 0
            q3 = np.percentile(durations, 75) if durations else 0
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = [d for d in durations if d < lower_bound or d > upper_bound]
            
            statistical_results[op_name] = {
                'sample_size': len(durations),
                'percentiles': percentiles,
                'outliers_count': len(outliers),
                'outliers_percentage': (len(outliers) / len(durations)) * 100 if durations else 0,
                'coefficient_of_variation': (np.std(durations) / np.mean(durations)) * 100 if durations and np.mean(durations) > 0 else 0
            }
        
        return statistical_results
    
    def _analyse_trends(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse performance trends over time."""
        trend_analysis = {}
        
        operations = data.get('operations', [])
        if not operations:
            return trend_analysis
        
        # Sort by timestamp
        sorted_ops = sorted(operations, key=lambda x: x.get('timestamp', ''))
        
        # Group by operation type and analyse trends
        operation_groups = defaultdict(list)
        for op in sorted_ops:
            operation_groups[op.get('operation_name', 'unknown')].append(op)
        
        for op_name, op_data in operation_groups.items():
            if len(op_data) < 5:  # Need at least 5 data points
                continue
            
            durations = [op.get('duration', 0) for op in op_data]
            timestamps = range(len(durations))  # Simplified time index
            
            # Simple linear regression for trend
            if len(durations) > 1:
                correlation = np.corrcoef(timestamps, durations)[0, 1] if len(set(durations)) > 1 else 0
                slope = np.polyfit(timestamps, durations, 1)[0] if len(durations) > 1 else 0
                
                # Determine trend direction
                if abs(correlation) < 0.3:
                    trend_direction = 'stable'
                elif slope > 0:
                    trend_direction = 'increasing'
                else:
                    trend_direction = 'decreasing'
                
                trend_analysis[op_name] = {
                    'trend_direction': trend_direction,
                    'correlation': correlation,
                    'slope': slope,
                    'data_points': len(durations),
                    'trend_strength': abs(correlation)
                }
        
        return trend_analysis
    
    def _detect_anomalies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect performance anomalies."""
        anomalies = {}
        
        operations = data.get('operations', [])
        if not operations:
            return anomalies
        
        # Group by operation type
        operation_groups = defaultdict(list)
        for op in operations:
            operation_groups[op.get('operation_name', 'unknown')].append(op)
        
        for op_name, op_data in operation_groups.items():
            durations = [op.get('duration', 0) for op in op_data]
            
            if len(durations) < 10:  # Need sufficient data
                continue
            
            # Use Z-score method for anomaly detection
            mean_duration = np.mean(durations)
            std_duration = np.std(durations)
            
            anomalous_operations = []
            for i, op in enumerate(op_data):
                duration = op.get('duration', 0)
                z_score = abs((duration - mean_duration) / std_duration) if std_duration > 0 else 0
                
                if z_score > 3:  # More than 3 standard deviations
                    anomalous_operations.append({
                        'operation_index': i,
                        'duration': duration,
                        'z_score': z_score,
                        'timestamp': op.get('timestamp', '')
                    })
            
            if anomalous_operations:
                anomalies[op_name] = {
                    'anomalous_operations': anomalous_operations,
                    'anomaly_rate': (len(anomalous_operations) / len(durations)) * 100,
                    'mean_duration': mean_duration,
                    'std_duration': std_duration
                }
        
        return anomalies
    
    def _generate_insights(self, data: Dict[str, Any]) -> List[str]:
        """Generate performance insights from analysis."""
        insights = []
        
        operations = data.get('operations', [])
        if not operations:
            return insights
        
        # Overall performance insights
        durations = [op.get('duration', 0) for op in operations]
        memory_usage = [op.get('memory_usage_mb', 0) for op in operations]
        
        if durations:
            avg_duration = np.mean(durations)
            max_duration = np.max(durations)
            
            if max_duration > avg_duration * 5:
                insights.append(f"Some operations take significantly longer than average (max: {max_duration:.2f}s vs avg: {avg_duration:.2f}s)")
        
        if memory_usage:
            max_memory = np.max(memory_usage)
            if max_memory > 1000:  # More than 1GB
                insights.append(f"High memory usage detected: {max_memory:.1f} MB peak usage")
        
        # Operation-specific insights
        operation_groups = defaultdict(list)
        for op in operations:
            operation_groups[op.get('operation_name', 'unknown')].append(op)
        
        for op_name, op_data in operation_groups.items():
            op_durations = [op.get('duration', 0) for op in op_data]
            
            if len(op_durations) > 5:
                cv = (np.std(op_durations) / np.mean(op_durations)) * 100 if np.mean(op_durations) > 0 else 0
                
                if cv > 50:  # High variability
                    insights.append(f"Operation '{op_name}' shows high performance variability (CV: {cv:.1f}%)")
                
                if np.mean(op_durations) > 30:  # Slow operation
                    insights.append(f"Operation '{op_name}' is potentially slow (avg: {np.mean(op_durations):.2f}s)")
        
        return insights


class BottleneckDetector:
    """
    Automatic detection of performance bottlenecks.
    
    Features:
    - CPU bottleneck detection
    - Memory bottleneck detection
    - I/O bottleneck detection
    - Database bottleneck detection
    - Application-level bottleneck detection
    """
    
    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or {
            'cpu_high': 80.0,
            'cpu_critical': 95.0,
            'memory_high': 80.0,
            'memory_critical': 95.0,
            'disk_high': 85.0,
            'disk_critical': 95.0,
            'slow_operation': 30.0,  # seconds
            'very_slow_operation': 120.0  # seconds
        }
        self.detected_bottlenecks = []
    
    def detect_bottlenecks(self, 
                          system_metrics: Dict[str, Any] = None,
                          performance_data: Dict[str, Any] = None) -> List[PerformanceBottleneck]:
        """Detect performance bottlenecks from system and performance data."""
        bottlenecks = []
        
        # System-level bottleneck detection
        if system_metrics:
            bottlenecks.extend(self._detect_system_bottlenecks(system_metrics))
        
        # Application-level bottleneck detection
        if performance_data:
            bottlenecks.extend(self._detect_application_bottlenecks(performance_data))
        
        self.detected_bottlenecks.extend(bottlenecks)
        
        if bottlenecks:
            logger.warning(f"Detected {len(bottlenecks)} performance bottlenecks")
        
        return bottlenecks
    
    def _detect_system_bottlenecks(self, metrics: Dict[str, Any]) -> List[PerformanceBottleneck]:
        """Detect system-level bottlenecks."""
        bottlenecks = []
        
        # CPU bottlenecks
        cpu_percent = metrics.get('cpu_percent', 0)
        if cpu_percent >= self.thresholds['cpu_critical']:
            bottlenecks.append(PerformanceBottleneck(
                bottleneck_id=f"cpu_critical_{int(time.time())}",
                bottleneck_type="cpu",
                severity="critical",
                description=f"Critical CPU usage: {cpu_percent:.1f}%",
                affected_operation="system",
                impact_score=95.0,
                recommendations=[
                    "Investigate CPU-intensive processes",
                    "Consider horizontal scaling",
                    "Optimise algorithms for CPU efficiency",
                    "Review thread pool configurations"
                ],
                metadata={'cpu_percent': cpu_percent}
            ))
        elif cpu_percent >= self.thresholds['cpu_high']:
            bottlenecks.append(PerformanceBottleneck(
                bottleneck_id=f"cpu_high_{int(time.time())}",
                bottleneck_type="cpu",
                severity="high",
                description=f"High CPU usage: {cpu_percent:.1f}%",
                affected_operation="system",
                impact_score=70.0,
                recommendations=[
                    "Monitor CPU usage trends",
                    "Profile CPU-intensive operations",
                    "Consider performance optimisations"
                ],
                metadata={'cpu_percent': cpu_percent}
            ))
        
        # Memory bottlenecks
        memory_percent = metrics.get('memory_percent', 0)
        if memory_percent >= self.thresholds['memory_critical']:
            bottlenecks.append(PerformanceBottleneck(
                bottleneck_id=f"memory_critical_{int(time.time())}",
                bottleneck_type="memory",
                severity="critical",
                description=f"Critical memory usage: {memory_percent:.1f}%",
                affected_operation="system",
                impact_score=90.0,
                recommendations=[
                    "Investigate memory leaks",
                    "Implement memory pooling",
                    "Add more RAM or scale horizontally",
                    "Optimise data structures",
                    "Implement memory-efficient algorithms"
                ],
                metadata={'memory_percent': memory_percent}
            ))
        elif memory_percent >= self.thresholds['memory_high']:
            bottlenecks.append(PerformanceBottleneck(
                bottleneck_id=f"memory_high_{int(time.time())}",
                bottleneck_type="memory",
                severity="high",
                description=f"High memory usage: {memory_percent:.1f}%",
                affected_operation="system",
                impact_score=65.0,
                recommendations=[
                    "Monitor memory usage patterns",
                    "Profile memory allocation",
                    "Consider memory optimisations"
                ],
                metadata={'memory_percent': memory_percent}
            ))
        
        # Disk bottlenecks
        disk_percent = metrics.get('disk_usage_percent', 0)
        if disk_percent >= self.thresholds['disk_critical']:
            bottlenecks.append(PerformanceBottleneck(
                bottleneck_id=f"disk_critical_{int(time.time())}",
                bottleneck_type="disk",
                severity="critical",
                description=f"Critical disk usage: {disk_percent:.1f}%",
                affected_operation="system",
                impact_score=85.0,
                recommendations=[
                    "Clean up temporary files",
                    "Archive old data",
                    "Add more storage capacity",
                    "Implement data compression",
                    "Move data to external storage"
                ],
                metadata={'disk_usage_percent': disk_percent}
            ))
        
        return bottlenecks
    
    def _detect_application_bottlenecks(self, data: Dict[str, Any]) -> List[PerformanceBottleneck]:
        """Detect application-level bottlenecks."""
        bottlenecks = []
        
        operations = data.get('operations', [])
        if not operations:
            return bottlenecks
        
        # Group by operation type
        operation_groups = defaultdict(list)
        for op in operations:
            operation_groups[op.get('operation_name', 'unknown')].append(op)
        
        for op_name, op_data in operation_groups.items():
            durations = [op.get('duration', 0) for op in op_data]
            
            if not durations:
                continue
            
            avg_duration = np.mean(durations)
            max_duration = np.max(durations)
            
            # Slow operation bottlenecks
            if max_duration >= self.thresholds['very_slow_operation']:
                bottlenecks.append(PerformanceBottleneck(
                    bottleneck_id=f"slow_op_critical_{op_name}_{int(time.time())}",
                    bottleneck_type="slow_operation",
                    severity="critical",
                    description=f"Very slow operation: {op_name} (max: {max_duration:.1f}s)",
                    affected_operation=op_name,
                    impact_score=80.0,
                    recommendations=[
                        "Profile the operation for bottlenecks",
                        "Optimise database queries",
                        "Implement caching strategies",
                        "Consider algorithmic improvements",
                        "Add parallel processing"
                    ],
                    metadata={'max_duration': max_duration, 'avg_duration': avg_duration}
                ))
            elif avg_duration >= self.thresholds['slow_operation']:
                bottlenecks.append(PerformanceBottleneck(
                    bottleneck_id=f"slow_op_{op_name}_{int(time.time())}",
                    bottleneck_type="slow_operation",
                    severity="high",
                    description=f"Slow operation: {op_name} (avg: {avg_duration:.1f}s)",
                    affected_operation=op_name,
                    impact_score=60.0,
                    recommendations=[
                        "Analyse operation performance",
                        "Look for optimisation opportunities",
                        "Consider performance tuning"
                    ],
                    metadata={'avg_duration': avg_duration}
                ))
            
            # High variability bottleneck
            if len(durations) > 5:
                cv = (np.std(durations) / np.mean(durations)) * 100 if np.mean(durations) > 0 else 0
                if cv > 75:  # High coefficient of variation
                    bottlenecks.append(PerformanceBottleneck(
                        bottleneck_id=f"variable_perf_{op_name}_{int(time.time())}",
                        bottleneck_type="performance_variability",
                        severity="medium",
                        description=f"High performance variability: {op_name} (CV: {cv:.1f}%)",
                        affected_operation=op_name,
                        impact_score=40.0,
                        recommendations=[
                            "Investigate causes of performance variation",
                            "Check for resource contention",
                            "Analyse input data patterns",
                            "Consider performance stabilisation"
                        ],
                        metadata={'coefficient_of_variation': cv}
                    ))
        
        return bottlenecks


class OptimisationRecommender:
    """
    Generate specific optimisation recommendations based on performance analysis.
    
    Features:
    - Bottleneck-specific recommendations
    - Code optimisation suggestions
    - Configuration optimisation
    - Architecture recommendations
    - Prioritised improvement plans
    """
    
    def __init__(self):
        self.recommendation_rules = self._load_recommendation_rules()
        self.generated_recommendations = []
    
    def generate_recommendations(self, 
                               bottlenecks: List[PerformanceBottleneck],
                               performance_data: Dict[str, Any] = None) -> List[OptimisationRecommendation]:
        """Generate optimisation recommendations based on detected bottlenecks."""
        recommendations = []
        
        for bottleneck in bottlenecks:
            recs = self._generate_bottleneck_recommendations(bottleneck, performance_data)
            recommendations.extend(recs)
        
        # Add general recommendations based on performance data
        if performance_data:
            general_recs = self._generate_general_recommendations(performance_data)
            recommendations.extend(general_recs)
        
        # Remove duplicates and prioritise
        recommendations = self._deduplicate_and_prioritise(recommendations)
        
        self.generated_recommendations.extend(recommendations)
        
        logger.info(f"Generated {len(recommendations)} optimisation recommendations")
        
        return recommendations
    
    def _load_recommendation_rules(self) -> Dict[str, Any]:
        """Load recommendation rules for different bottleneck types."""
        return {
            'cpu': {
                'critical': [
                    {
                        'title': 'Implement CPU-efficient algorithms',
                        'description': 'Replace CPU-intensive algorithms with more efficient alternatives',
                        'effort': 'significant',
                        'impact': 85.0,
                        'code_changes': [
                            'Profile CPU hotspots with cProfile',
                            'Replace nested loops with vectorized operations',
                            'Use NumPy for numerical computations',
                            'Implement caching for expensive calculations'
                        ]
                    },
                    {
                        'title': 'Enable parallel processing',
                        'description': 'Utilise multiple CPU cores for parallel execution',
                        'effort': 'moderate',
                        'impact': 70.0,
                        'code_changes': [
                            'Use multiprocessing.Pool for CPU-bound tasks',
                            'Implement concurrent.futures for parallel execution',
                            'Consider using Dask for large-scale parallelism'
                        ]
                    }
                ],
                'high': [
                    {
                        'title': 'Optimise algorithm complexity',
                        'description': 'Reduce algorithmic complexity of operations',
                        'effort': 'moderate',
                        'impact': 60.0,
                        'code_changes': [
                            'Analyse Big O complexity of algorithms',
                            'Replace O(n²) algorithms with O(n log n) alternatives',
                            'Use appropriate data structures (dict vs list for lookups)'
                        ]
                    }
                ]
            },
            'memory': {
                'critical': [
                    {
                        'title': 'Implement memory pooling',
                        'description': 'Use object pooling to reduce memory allocation overhead',
                        'effort': 'moderate',
                        'impact': 75.0,
                        'code_changes': [
                            'Implement object pools for frequently created objects',
                            'Use memory-mapped files for large datasets',
                            'Consider using generators instead of lists'
                        ]
                    },
                    {
                        'title': 'Add streaming processing',
                        'description': 'Process data in chunks to reduce memory footprint',
                        'effort': 'significant',
                        'impact': 80.0,
                        'code_changes': [
                            'Implement chunk-based processing',
                            'Use pandas.read_csv with chunksize parameter',
                            'Process data in batches instead of loading all at once'
                        ]
                    }
                ],
                'high': [
                    {
                        'title': 'Optimise data structures',
                        'description': 'Use memory-efficient data structures',
                        'effort': 'minimal',
                        'impact': 50.0,
                        'code_changes': [
                            'Use appropriate data types (int32 vs int64)',
                            'Consider using category dtype for string columns',
                            'Use sparse data structures where applicable'
                        ]
                    }
                ]
            },
            'slow_operation': {
                'critical': [
                    {
                        'title': 'Implement caching strategy',
                        'description': 'Cache expensive operation results',
                        'effort': 'moderate',
                        'impact': 85.0,
                        'code_changes': [
                            'Use functools.lru_cache for function results',
                            'Implement Redis caching for database queries',
                            'Cache intermediate processing results'
                        ]
                    },
                    {
                        'title': 'Optimise database operations',
                        'description': 'Improve database query performance',
                        'effort': 'moderate',
                        'impact': 75.0,
                        'code_changes': [
                            'Add database indexes for frequently queried columns',
                            'Use bulk operations instead of row-by-row processing',
                            'Implement connection pooling'
                        ]
                    }
                ]
            },
            'disk': {
                'critical': [
                    {
                        'title': 'Implement data compression',
                        'description': 'Compress data to reduce storage requirements',
                        'effort': 'minimal',
                        'impact': 60.0,
                        'configuration_changes': [
                            'Enable gzip compression for text files',
                            'Use Parquet format for structured data',
                            'Implement automatic data archival'
                        ]
                    }
                ]
            }
        }
    
    def _generate_bottleneck_recommendations(self, 
                                           bottleneck: PerformanceBottleneck,
                                           performance_data: Dict[str, Any] = None) -> List[OptimisationRecommendation]:
        """Generate recommendations for a specific bottleneck."""
        recommendations = []
        
        rules = self.recommendation_rules.get(bottleneck.bottleneck_type, {})
        severity_rules = rules.get(bottleneck.severity, [])
        
        for rule in severity_rules:
            rec_id = f"{bottleneck.bottleneck_type}_{bottleneck.severity}_{rule['title'].replace(' ', '_').lower()}"
            
            recommendation = OptimisationRecommendation(
                recommendation_id=rec_id,
                category=bottleneck.bottleneck_type,
                priority=bottleneck.severity,
                title=rule['title'],
                description=rule['description'],
                implementation_effort=rule['effort'],
                expected_improvement=f"{rule['impact']:.0f}% improvement potential",
                code_changes=rule.get('code_changes', []),
                configuration_changes=rule.get('configuration_changes', []),
                estimated_impact=rule['impact']
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_general_recommendations(self, performance_data: Dict[str, Any]) -> List[OptimisationRecommendation]:
        """Generate general performance recommendations."""
        recommendations = []
        
        operations = performance_data.get('operations', [])
        if not operations:
            return recommendations
        
        # Analyse operation patterns
        operation_counts = Counter(op.get('operation_name', 'unknown') for op in operations)
        durations = [op.get('duration', 0) for op in operations]
        
        # Recommend monitoring if many operations
        if len(operations) > 100:
            recommendations.append(OptimisationRecommendation(
                recommendation_id="general_monitoring",
                category="monitoring",
                priority="medium",
                title="Implement comprehensive monitoring",
                description="Add detailed performance monitoring for high-volume operations",
                implementation_effort="minimal",
                expected_improvement="Better visibility into performance issues",
                code_changes=[
                    "Add performance decorators to key functions",
                    "Implement structured logging",
                    "Set up performance dashboards"
                ],
                estimated_impact=30.0
            ))
        
        # Recommend optimisation for frequent operations
        most_frequent_op = operation_counts.most_common(1)
        if most_frequent_op and most_frequent_op[0][1] > 50:
            op_name = most_frequent_op[0][0]
            recommendations.append(OptimisationRecommendation(
                recommendation_id=f"optimise_frequent_{op_name}",
                category="optimisation",
                priority="high",
                title=f"Optimise frequent operation: {op_name}",
                description=f"Focus optimisation efforts on the most frequently executed operation",
                implementation_effort="moderate",
                expected_improvement="Significant overall performance improvement",
                code_changes=[
                    f"Profile {op_name} operation in detail",
                    "Implement operation-specific caching",
                    "Consider batch processing for multiple instances"
                ],
                estimated_impact=65.0
            ))
        
        return recommendations
    
    def _deduplicate_and_prioritise(self, recommendations: List[OptimisationRecommendation]) -> List[OptimisationRecommendation]:
        """Remove duplicate recommendations and sort by priority."""
        # Remove duplicates based on recommendation_id
        seen_ids = set()
        unique_recommendations = []
        
        for rec in recommendations:
            if rec.recommendation_id not in seen_ids:
                unique_recommendations.append(rec)
                seen_ids.add(rec.recommendation_id)
        
        # Sort by priority and estimated impact
        priority_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        
        sorted_recommendations = sorted(
            unique_recommendations,
            key=lambda x: (priority_order.get(x.priority, 0), x.estimated_impact),
            reverse=True
        )
        
        return sorted_recommendations


class MemoryOptimiser:
    """
    Memory usage optimisation tools.
    
    Features:
    - Memory leak detection
    - Memory usage analysis
    - Garbage collection optimisation
    - Memory-efficient data structure recommendations
    """
    
    def __init__(self):
        self.memory_snapshots = []
        self.gc_stats = []
        
    def analyse_memory_usage(self, operation_name: str = None) -> Dict[str, Any]:
        """Analyse current memory usage and provide recommendations."""
        import tracemalloc
        import gc
        import psutil
        
        # Get current memory info
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Get garbage collection stats
        gc_stats = {
            'collections': gc.get_stats(),
            'garbage_count': len(gc.garbage),
            'threshold': gc.get_threshold()
        }
        
        # Get tracemalloc stats if available
        tracemalloc_stats = {}
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc_stats = {
                'current_mb': current / 1024 / 1024,
                'peak_mb': peak / 1024 / 1024
            }
        
        analysis = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'operation_name': operation_name,
            'memory_info': {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent()
            },
            'gc_stats': gc_stats,
            'tracemalloc_stats': tracemalloc_stats,
            'recommendations': self._generate_memory_recommendations(memory_info, gc_stats)
        }
        
        return analysis
    
    def _generate_memory_recommendations(self, memory_info, gc_stats) -> List[str]:
        """Generate memory optimisation recommendations."""
        recommendations = []
        
        memory_mb = memory_info.rss / 1024 / 1024
        
        if memory_mb > 1000:  # More than 1GB
            recommendations.append("High memory usage detected. Consider implementing streaming processing.")
            recommendations.append("Use generators instead of lists where possible.")
            recommendations.append("Implement object pooling for frequently created objects.")
        
        if memory_mb > 500:  # More than 500MB
            recommendations.append("Consider using memory-mapped files for large datasets.")
            recommendations.append("Implement data chunking for large processing operations.")
        
        # Check garbage collection
        total_collections = sum(stat['collections'] for stat in gc_stats['collections'])
        if total_collections > 1000:
            recommendations.append("High number of garbage collections. Check for memory leaks.")
            recommendations.append("Consider increasing garbage collection thresholds.")
        
        if gc_stats['garbage_count'] > 0:
            recommendations.append("Uncollectable objects detected. Check for circular references.")
        
        return recommendations
    
    def suggest_data_structure_optimisations(self, data_info: Dict[str, Any]) -> List[str]:
        """Suggest optimisations for data structures."""
        suggestions = []
        
        data_type = data_info.get('type', '')
        size = data_info.get('size', 0)
        
        if data_type == 'pandas.DataFrame':
            suggestions.extend([
                "Use appropriate data types (int32 instead of int64 where possible)",
                "Convert string columns to category dtype if they have limited unique values",
                "Use sparse arrays for datasets with many zeros",
                "Consider using Parquet format for storage"
            ])
        
        elif data_type == 'list' and size > 10000:
            suggestions.extend([
                "Consider using NumPy arrays for numerical data",
                "Use deque for frequent append/pop operations",
                "Implement generators for one-time iteration"
            ])
        
        elif data_type == 'dict' and size > 10000:
            suggestions.extend([
                "Consider using slots for object classes",
                "Use frozendict for immutable dictionaries",
                "Implement dictionary compression for sparse data"
            ])
        
        return suggestions


class QueryOptimiser:
    """
    Database query optimisation tools.
    
    Features:
    - Query performance analysis
    - Index recommendation
    - Query plan analysis
    - Connection pooling optimisation
    """
    
    def __init__(self):
        self.query_patterns = {}
        self.slow_queries = []
        
    def analyse_query_performance(self, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyse query performance and provide optimisation recommendations."""
        if not queries:
            return {}
        
        # Group queries by pattern
        query_groups = defaultdict(list)
        for query in queries:
            pattern = self._extract_query_pattern(query.get('query_text', ''))
            query_groups[pattern].append(query)
        
        analysis = {
            'total_queries': len(queries),
            'unique_patterns': len(query_groups),
            'query_analysis': {},
            'recommendations': []
        }
        
        for pattern, pattern_queries in query_groups.items():
            durations = [q.get('execution_time', 0) for q in pattern_queries]
            avg_duration = np.mean(durations) if durations else 0
            
            analysis['query_analysis'][pattern] = {
                'count': len(pattern_queries),
                'avg_duration': avg_duration,
                'max_duration': max(durations) if durations else 0,
                'total_time': sum(durations)
            }
            
            # Generate recommendations for slow queries
            if avg_duration > 1.0:  # Slower than 1 second
                analysis['recommendations'].extend(
                    self._generate_query_recommendations(pattern, pattern_queries)
                )
        
        return analysis
    
    def _extract_query_pattern(self, query_text: str) -> str:
        """Extract a generalised pattern from a query."""
        if not query_text:
            return 'unknown'
        
        # Remove string literals and numbers
        pattern = re.sub(r"'[^']*'", "'?'", query_text)
        pattern = re.sub(r'\b\d+\b', '?', pattern)
        
        # Extract main operation
        query_lower = pattern.lower().strip()
        if query_lower.startswith('select'):
            return 'SELECT'
        elif query_lower.startswith('insert'):
            return 'INSERT'
        elif query_lower.startswith('update'):
            return 'UPDATE'
        elif query_lower.startswith('delete'):
            return 'DELETE'
        else:
            return 'OTHER'
    
    def _generate_query_recommendations(self, pattern: str, queries: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for slow queries."""
        recommendations = []
        
        avg_duration = np.mean([q.get('execution_time', 0) for q in queries])
        
        if pattern == 'SELECT':
            recommendations.extend([
                "Add indexes on frequently queried columns",
                "Consider using LIMIT to reduce result set size",
                "Use appropriate WHERE clauses to filter data early",
                "Consider query result caching"
            ])
        
        elif pattern == 'INSERT':
            recommendations.extend([
                "Use bulk insert operations instead of individual inserts",
                "Consider disabling indexes during bulk operations",
                "Use batch processing for large datasets"
            ])
        
        elif pattern == 'UPDATE':
            recommendations.extend([
                "Add indexes on columns used in WHERE clauses",
                "Consider batch updates instead of row-by-row operations",
                "Use appropriate transaction sizes"
            ])
        
        if avg_duration > 5.0:  # Very slow queries
            recommendations.extend([
                "Analyse query execution plan",
                "Consider table partitioning",
                "Review database statistics and update if needed"
            ])
        
        return recommendations
    
    def recommend_indexes(self, query_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend database indexes based on query patterns."""
        index_recommendations = []
        
        for pattern, analysis in query_patterns.items():
            if analysis.get('avg_duration', 0) > 1.0:  # Slow queries
                # Extract table and column information (simplified)
                # In practice, this would parse SQL more thoroughly
                recommendation = {
                    'query_pattern': pattern,
                    'recommended_action': 'Add index',
                    'estimated_improvement': '50-80% query time reduction',
                    'implementation_effort': 'minimal'
                }
                index_recommendations.append(recommendation)
        
        return index_recommendations