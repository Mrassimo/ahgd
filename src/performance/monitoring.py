"""
Real-time performance monitoring for the AHGD ETL pipeline.

This module provides comprehensive monitoring capabilities including:
- Real-time system resource monitoring
- Performance alerts and thresholds
- Historical performance tracking
- Performance degradation detection
- Automatic performance reporting
"""

import asyncio
import json
import psutil
import threading
import time
import warnings
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union
import sqlite3

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not available. Install with: pip install matplotlib")

from ..utils.logging import get_logger
from ..utils.interfaces import ProcessingStatus

logger = get_logger()


@dataclass
class SystemMetrics:
    """System resource metrics at a point in time."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    load_average: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_available_mb': self.memory_available_mb,
            'disk_usage_percent': self.disk_usage_percent,
            'disk_free_gb': self.disk_free_gb,
            'network_bytes_sent': self.network_bytes_sent,
            'network_bytes_recv': self.network_bytes_recv,
            'process_count': self.process_count,
            'load_average': self.load_average
        }


@dataclass
class PerformanceAlert:
    """Performance alert configuration and state."""
    alert_id: str
    alert_name: str
    metric_name: str
    threshold_value: float
    comparison_operator: str  # 'gt', 'lt', 'eq', 'gte', 'lte'
    is_active: bool = False
    triggered_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    trigger_count: int = 0
    callback: Optional[Callable] = None
    
    def check_threshold(self, value: float) -> bool:
        """Check if the metric value crosses the threshold."""
        if self.comparison_operator == 'gt':
            return value > self.threshold_value
        elif self.comparison_operator == 'gte':
            return value >= self.threshold_value
        elif self.comparison_operator == 'lt':
            return value < self.threshold_value
        elif self.comparison_operator == 'lte':
            return value <= self.threshold_value
        elif self.comparison_operator == 'eq':
            return value == self.threshold_value
        return False


@dataclass
class PerformanceTrend:
    """Performance trend analysis."""
    metric_name: str
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_strength: float  # 0-1, how strong the trend is
    slope: float
    correlation: float
    duration_hours: float
    data_points: int


class SystemMonitor:
    """
    Real-time system resource monitoring.
    
    Features:
    - CPU, memory, disk, network monitoring
    - Historical data collection
    - Resource usage trends
    - System health checks
    """
    
    def __init__(self, collection_interval: float = 60.0, max_history: int = 1440):
        self.collection_interval = collection_interval
        self.max_history = max_history  # 24 hours at 1-minute intervals
        self.metrics_history = deque(maxlen=max_history)
        self.is_monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process()
        
    def start_monitoring(self):
        """Start system monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("System monitoring started",
                   collection_interval=self.collection_interval,
                   max_history=self.max_history)
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Log current system state periodically
                if len(self.metrics_history) % 10 == 0:  # Every 10 intervals
                    logger.debug("System metrics collected",
                               cpu_percent=metrics.cpu_percent,
                               memory_percent=metrics.memory_percent,
                               disk_usage_percent=metrics.disk_usage_percent)
                
            except Exception as e:
                logger.error("Error collecting system metrics", error=str(e))
            
            time.sleep(self.collection_interval)
    
    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_mb = memory.available / 1024 / 1024
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_usage_percent = disk.percent
        disk_free_gb = disk.free / 1024 / 1024 / 1024
        
        # Network metrics
        network = psutil.net_io_counters()
        network_bytes_sent = network.bytes_sent
        network_bytes_recv = network.bytes_recv
        
        # Process count
        process_count = len(psutil.pids())
        
        # Load average (Unix only)
        load_average = []
        try:
            load_average = list(psutil.getloadavg())
        except AttributeError:
            pass  # Windows doesn't have getloadavg
        
        return SystemMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_mb=memory_available_mb,
            disk_usage_percent=disk_usage_percent,
            disk_free_gb=disk_free_gb,
            network_bytes_sent=network_bytes_sent,
            network_bytes_recv=network_bytes_recv,
            process_count=process_count,
            load_average=load_average
        )
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, hours: float = 1.0) -> List[SystemMetrics]:
        """Get metrics history for the specified time period."""
        if not self.metrics_history:
            return []
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health assessment."""
        current = self.get_current_metrics()
        if not current:
            return {'status': 'unknown', 'reason': 'No metrics available'}
        
        health_issues = []
        
        # Check CPU usage
        if current.cpu_percent > 90:
            health_issues.append('CPU usage critically high')
        elif current.cpu_percent > 70:
            health_issues.append('CPU usage high')
        
        # Check memory usage
        if current.memory_percent > 95:
            health_issues.append('Memory usage critically high')
        elif current.memory_percent > 80:
            health_issues.append('Memory usage high')
        
        # Check disk space
        if current.disk_usage_percent > 95:
            health_issues.append('Disk space critically low')
        elif current.disk_usage_percent > 85:
            health_issues.append('Disk space low')
        
        # Determine overall status
        if any('critically' in issue for issue in health_issues):
            status = 'critical'
        elif health_issues:
            status = 'warning'
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'timestamp': current.timestamp.isoformat(),
            'issues': health_issues,
            'metrics': current.to_dict()
        }class PerformanceMonitor:
    """
    Application-level performance monitoring.
    
    Features:
    - ETL operation performance tracking
    - Custom metric collection
    - Performance trend analysis
    - Degradation detection
    """
    
    def __init__(self, storage_file: Optional[str] = None):
        self.storage_file = storage_file
        self.metrics = defaultdict(list)
        self.operation_timings = {}
        self.custom_metrics = defaultdict(list)
        self.is_monitoring = True
        
        if storage_file:
            self._init_storage()
    
    def _init_storage(self):
        """Initialise persistent storage for metrics."""
        if not self.storage_file:
            return
        
        conn = sqlite3.connect(self.storage_file)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                operation_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS operation_timings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                operation_name TEXT NOT NULL,
                duration_seconds REAL NOT NULL,
                status TEXT NOT NULL,
                metadata TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_operation_timing(self, operation_name: str, duration: float, 
                              status: str = 'completed', metadata: Dict[str, Any] = None):
        """Record timing for an operation."""
        timing_record = {
            'timestamp': datetime.now(timezone.utc),
            'operation_name': operation_name,
            'duration': duration,
            'status': status,
            'metadata': metadata or {}
        }
        
        self.operation_timings[operation_name] = self.operation_timings.get(operation_name, [])
        self.operation_timings[operation_name].append(timing_record)
        
        # Store in database if configured
        if self.storage_file:
            self._store_operation_timing(timing_record)
        
        logger.debug(f"Operation timing recorded: {operation_name}",
                    duration=duration, status=status)
    
    def record_custom_metric(self, metric_name: str, value: float, 
                           operation_name: str = None, metadata: Dict[str, Any] = None):
        """Record a custom performance metric."""
        metric_record = {
            'timestamp': datetime.now(timezone.utc),
            'metric_name': metric_name,
            'value': value,
            'operation_name': operation_name,
            'metadata': metadata or {}
        }
        
        self.custom_metrics[metric_name].append(metric_record)
        
        # Store in database if configured
        if self.storage_file:
            self._store_custom_metric(metric_record)
        
        logger.debug(f"Custom metric recorded: {metric_name}",
                    value=value, operation=operation_name)
    
    def _store_operation_timing(self, timing_record: Dict[str, Any]):
        """Store operation timing in database."""
        conn = sqlite3.connect(self.storage_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO operation_timings (timestamp, operation_name, duration_seconds, status, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            timing_record['timestamp'].isoformat(),
            timing_record['operation_name'],
            timing_record['duration'],
            timing_record['status'],
            json.dumps(timing_record['metadata'])
        ))
        
        conn.commit()
        conn.close()
    
    def _store_custom_metric(self, metric_record: Dict[str, Any]):
        """Store custom metric in database."""
        conn = sqlite3.connect(self.storage_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance_metrics (timestamp, operation_name, metric_name, metric_value, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            metric_record['timestamp'].isoformat(),
            metric_record.get('operation_name', ''),
            metric_record['metric_name'],
            metric_record['value'],
            json.dumps(metric_record['metadata'])
        ))
        
        conn.commit()
        conn.close()
    
    def get_operation_statistics(self, operation_name: str, hours: float = 24.0) -> Dict[str, Any]:
        """Get statistics for an operation over a time period."""
        if operation_name not in self.operation_timings:
            return {}
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_timings = [
            t for t in self.operation_timings[operation_name]
            if t['timestamp'] >= cutoff_time
        ]
        
        if not recent_timings:
            return {}
        
        durations = [t['duration'] for t in recent_timings]
        successful = [t for t in recent_timings if t['status'] == 'completed']
        
        return {
            'operation_name': operation_name,
            'time_period_hours': hours,
            'total_executions': len(recent_timings),
            'successful_executions': len(successful),
            'success_rate': len(successful) / len(recent_timings),
            'average_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'total_duration': sum(durations),
            'executions_per_hour': len(recent_timings) / hours
        }
    
    def detect_performance_degradation(self, operation_name: str, 
                                     baseline_hours: float = 168.0,  # 1 week
                                     recent_hours: float = 24.0) -> Dict[str, Any]:
        """Detect performance degradation by comparing recent vs baseline performance."""
        baseline_stats = self.get_operation_statistics(operation_name, baseline_hours)
        recent_stats = self.get_operation_statistics(operation_name, recent_hours)
        
        if not baseline_stats or not recent_stats:
            return {'has_degradation': False, 'reason': 'Insufficient data'}
        
        # Compare average durations
        baseline_avg = baseline_stats['average_duration']
        recent_avg = recent_stats['average_duration']
        duration_change = ((recent_avg - baseline_avg) / baseline_avg) * 100
        
        # Compare success rates
        baseline_success = baseline_stats['success_rate']
        recent_success = recent_stats['success_rate']
        success_change = ((baseline_success - recent_success) / baseline_success) * 100
        
        # Detect degradation (20% threshold)
        has_duration_degradation = duration_change > 20
        has_success_degradation = success_change > 5  # 5% drop in success rate
        
        return {
            'has_degradation': has_duration_degradation or has_success_degradation,
            'operation_name': operation_name,
            'duration_change_percent': duration_change,
            'success_rate_change_percent': success_change,
            'baseline_avg_duration': baseline_avg,
            'recent_avg_duration': recent_avg,
            'baseline_success_rate': baseline_success,
            'recent_success_rate': recent_success,
            'degradation_types': {
                'duration': has_duration_degradation,
                'success_rate': has_success_degradation
            }
        }


class ResourceTracker:
    """
    Track resource usage for specific operations or processes.
    
    Features:
    - Per-operation resource tracking
    - Resource usage trends
    - Resource leak detection
    - Resource efficiency metrics
    """
    
    def __init__(self):
        self.active_operations = {}
        self.completed_operations = []
        self.resource_snapshots = defaultdict(list)
    
    def start_operation_tracking(self, operation_id: str, operation_name: str):
        """Start tracking resources for an operation."""
        start_time = datetime.now(timezone.utc)
        process = psutil.Process()
        
        initial_snapshot = {
            'timestamp': start_time,
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent(),
            'threads': process.num_threads(),
            'open_files': len(process.open_files())
        }
        
        self.active_operations[operation_id] = {
            'operation_name': operation_name,
            'start_time': start_time,
            'initial_snapshot': initial_snapshot,
            'snapshots': [initial_snapshot]
        }
        
        logger.debug(f"Started resource tracking for operation: {operation_name}",
                    operation_id=operation_id)
    
    def stop_operation_tracking(self, operation_id: str) -> Dict[str, Any]:
        """Stop tracking resources for an operation and return summary."""
        if operation_id not in self.active_operations:
            return {}
        
        operation = self.active_operations.pop(operation_id)
        end_time = datetime.now(timezone.utc)
        process = psutil.Process()
        
        final_snapshot = {
            'timestamp': end_time,
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent(),
            'threads': process.num_threads(),
            'open_files': len(process.open_files())
        }
        
        operation['end_time'] = end_time
        operation['final_snapshot'] = final_snapshot
        operation['duration'] = (end_time - operation['start_time']).total_seconds()
        
        # Calculate resource usage
        initial = operation['initial_snapshot']
        final = final_snapshot
        
        resource_summary = {
            'operation_id': operation_id,
            'operation_name': operation['operation_name'],
            'duration_seconds': operation['duration'],
            'memory_growth_mb': final['memory_mb'] - initial['memory_mb'],
            'peak_memory_mb': max(s['memory_mb'] for s in operation['snapshots'] + [final]),
            'average_cpu_percent': sum(s['cpu_percent'] for s in operation['snapshots']) / len(operation['snapshots']),
            'thread_count_change': final['threads'] - initial['threads'],
            'file_descriptor_change': final['open_files'] - initial['open_files'],
            'resource_efficiency': self._calculate_efficiency(operation)
        }
        
        self.completed_operations.append(resource_summary)
        
        logger.debug(f"Stopped resource tracking for operation: {operation['operation_name']}",
                    resource_summary=resource_summary)
        
        return resource_summary
    
    def _calculate_efficiency(self, operation: Dict[str, Any]) -> float:
        """Calculate resource efficiency score (0-100)."""
        score = 100.0
        
        initial = operation['initial_snapshot']
        final = operation['final_snapshot']
        
        # Penalise excessive memory growth
        memory_growth = final['memory_mb'] - initial['memory_mb']
        if memory_growth > 100:  # More than 100MB growth
            score -= min(50, memory_growth / 10)
        
        # Penalise high CPU usage
        avg_cpu = sum(s['cpu_percent'] for s in operation['snapshots']) / len(operation['snapshots'])
        if avg_cpu > 80:
            score -= min(30, (avg_cpu - 80) * 2)
        
        # Penalise resource leaks
        thread_leak = final['threads'] - initial['threads']
        if thread_leak > 0:
            score -= min(20, thread_leak * 5)
        
        file_leak = final['open_files'] - initial['open_files']
        if file_leak > 0:
            score -= min(20, file_leak * 2)
        
        return max(0.0, score)


class AlertManager:
    """
    Manage performance alerts and notifications.
    
    Features:
    - Threshold-based alerting
    - Alert escalation
    - Alert suppression
    - Custom alert callbacks
    """
    
    def __init__(self):
        self.alerts = {}
        self.alert_history = []
        self.suppressed_alerts = set()
        self.is_monitoring = False
        
    def add_alert(self, alert: PerformanceAlert):
        """Add a performance alert."""
        self.alerts[alert.alert_id] = alert
        logger.info(f"Performance alert added: {alert.alert_name}",
                   alert_id=alert.alert_id,
                   metric=alert.metric_name,
                   threshold=alert.threshold_value)
    
    def remove_alert(self, alert_id: str):
        """Remove a performance alert."""
        if alert_id in self.alerts:
            del self.alerts[alert_id]
            logger.info(f"Performance alert removed: {alert_id}")
    
    def check_alerts(self, metrics: Dict[str, float]):
        """Check all alerts against current metrics."""
        for alert in self.alerts.values():
            if alert.alert_id in self.suppressed_alerts:
                continue
                
            metric_value = metrics.get(alert.metric_name)
            if metric_value is None:
                continue
            
            threshold_crossed = alert.check_threshold(metric_value)
            
            if threshold_crossed and not alert.is_active:
                # Alert triggered
                self._trigger_alert(alert, metric_value)
            elif not threshold_crossed and alert.is_active:
                # Alert resolved
                self._resolve_alert(alert, metric_value)
    
    def _trigger_alert(self, alert: PerformanceAlert, metric_value: float):
        """Trigger an alert."""
        alert.is_active = True
        alert.triggered_at = datetime.now(timezone.utc)
        alert.trigger_count += 1
        
        alert_event = {
            'alert_id': alert.alert_id,
            'alert_name': alert.alert_name,
            'event_type': 'triggered',
            'timestamp': alert.triggered_at,
            'metric_name': alert.metric_name,
            'metric_value': metric_value,
            'threshold_value': alert.threshold_value,
            'trigger_count': alert.trigger_count
        }
        
        self.alert_history.append(alert_event)
        
        logger.warning(f"Performance alert triggered: {alert.alert_name}",
                      alert_event=alert_event)
        
        # Call custom callback if provided
        if alert.callback:
            try:
                alert.callback(alert_event)
            except Exception as e:
                logger.error(f"Alert callback failed for {alert.alert_id}", error=str(e))
    
    def _resolve_alert(self, alert: PerformanceAlert, metric_value: float):
        """Resolve an alert."""
        alert.is_active = False
        alert.resolved_at = datetime.now(timezone.utc)
        
        alert_event = {
            'alert_id': alert.alert_id,
            'alert_name': alert.alert_name,
            'event_type': 'resolved',
            'timestamp': alert.resolved_at,
            'metric_name': alert.metric_name,
            'metric_value': metric_value,
            'threshold_value': alert.threshold_value,
            'duration_minutes': (alert.resolved_at - alert.triggered_at).total_seconds() / 60
        }
        
        self.alert_history.append(alert_event)
        
        logger.info(f"Performance alert resolved: {alert.alert_name}",
                   alert_event=alert_event)
    
    def suppress_alert(self, alert_id: str, duration_minutes: int = 60):
        """Suppress an alert for a specified duration."""
        self.suppressed_alerts.add(alert_id)
        logger.info(f"Alert suppressed: {alert_id}",
                   duration_minutes=duration_minutes)
        
        # Schedule unsuppression (simplified - in production use proper scheduler)
        def unsuppress():
            time.sleep(duration_minutes * 60)
            self.suppressed_alerts.discard(alert_id)
            logger.info(f"Alert suppression expired: {alert_id}")
        
        threading.Thread(target=unsuppress, daemon=True).start()
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all currently active alerts."""
        return [
            {
                'alert_id': alert.alert_id,
                'alert_name': alert.alert_name,
                'metric_name': alert.metric_name,
                'threshold_value': alert.threshold_value,
                'triggered_at': alert.triggered_at.isoformat() if alert.triggered_at else None,
                'trigger_count': alert.trigger_count
            }
            for alert in self.alerts.values()
            if alert.is_active
        ]


class PerformanceReporter:
    """
    Generate performance reports and visualisations.
    
    Features:
    - Automated report generation
    - Performance visualisations
    - Trend analysis reports
    - Executive summaries
    """
    
    def __init__(self, output_dir: str = "reports/performance"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_system_report(self, system_monitor: SystemMonitor, 
                             hours: float = 24.0) -> str:
        """Generate a comprehensive system performance report."""
        metrics_history = system_monitor.get_metrics_history(hours)
        if not metrics_history:
            return "No metrics data available for report generation."
        
        report_data = {
            'report_type': 'system_performance',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'time_period_hours': hours,
            'metrics_count': len(metrics_history),
            'summary': self._generate_system_summary(metrics_history),
            'trends': self._analyse_system_trends(metrics_history)
        }
        
        # Save report
        report_file = self.output_dir / f"system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Generate visualisation if matplotlib available
        if MATPLOTLIB_AVAILABLE:
            self._create_system_visualisation(metrics_history, report_file.stem)
        
        logger.info(f"System performance report generated: {report_file}")
        return str(report_file)
    
    def _generate_system_summary(self, metrics_history: List[SystemMetrics]) -> Dict[str, Any]:
        """Generate system performance summary."""
        cpu_values = [m.cpu_percent for m in metrics_history]
        memory_values = [m.memory_percent for m in metrics_history]
        disk_values = [m.disk_usage_percent for m in metrics_history]
        
        return {
            'cpu_usage': {
                'average': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory_usage': {
                'average': sum(memory_values) / len(memory_values),
                'max': max(memory_values),
                'min': min(memory_values)
            },
            'disk_usage': {
                'average': sum(disk_values) / len(disk_values),
                'max': max(disk_values),
                'min': min(disk_values)
            }
        }
    
    def _analyse_system_trends(self, metrics_history: List[SystemMetrics]) -> List[PerformanceTrend]:
        """Analyse performance trends in system metrics."""
        trends = []
        
        # Simple trend analysis (in production, use more sophisticated algorithms)
        cpu_values = [m.cpu_percent for m in metrics_history]
        memory_values = [m.memory_percent for m in metrics_history]
        
        # CPU trend
        if len(cpu_values) > 5:
            cpu_slope = (cpu_values[-1] - cpu_values[0]) / len(cpu_values)
            cpu_trend = PerformanceTrend(
                metric_name='cpu_percent',
                trend_direction='increasing' if cpu_slope > 0.5 else 'decreasing' if cpu_slope < -0.5 else 'stable',
                trend_strength=min(1.0, abs(cpu_slope) / 10),
                slope=cpu_slope,
                correlation=0.0,  # Would calculate actual correlation
                duration_hours=(metrics_history[-1].timestamp - metrics_history[0].timestamp).total_seconds() / 3600,
                data_points=len(cpu_values)
            )
            trends.append(cpu_trend)
        
        # Memory trend
        if len(memory_values) > 5:
            memory_slope = (memory_values[-1] - memory_values[0]) / len(memory_values)
            memory_trend = PerformanceTrend(
                metric_name='memory_percent',
                trend_direction='increasing' if memory_slope > 0.5 else 'decreasing' if memory_slope < -0.5 else 'stable',
                trend_strength=min(1.0, abs(memory_slope) / 10),
                slope=memory_slope,
                correlation=0.0,
                duration_hours=(metrics_history[-1].timestamp - metrics_history[0].timestamp).total_seconds() / 3600,
                data_points=len(memory_values)
            )
            trends.append(memory_trend)
        
        return trends
    
    def _create_system_visualisation(self, metrics_history: List[SystemMetrics], filename_base: str):
        """Create visualisation of system metrics."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        timestamps = [m.timestamp for m in metrics_history]
        cpu_values = [m.cpu_percent for m in metrics_history]
        memory_values = [m.memory_percent for m in metrics_history]
        disk_values = [m.disk_usage_percent for m in metrics_history]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('System Performance Metrics', fontsize=16)
        
        # CPU usage
        axes[0].plot(timestamps, cpu_values, 'b-', linewidth=2)
        axes[0].set_title('CPU Usage (%)')
        axes[0].set_ylabel('CPU %')
        axes[0].grid(True, alpha=0.3)
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # Memory usage
        axes[1].plot(timestamps, memory_values, 'r-', linewidth=2)
        axes[1].set_title('Memory Usage (%)')
        axes[1].set_ylabel('Memory %')
        axes[1].grid(True, alpha=0.3)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # Disk usage
        axes[2].plot(timestamps, disk_values, 'g-', linewidth=2)
        axes[2].set_title('Disk Usage (%)')
        axes[2].set_ylabel('Disk %')
        axes[2].set_xlabel('Time')
        axes[2].grid(True, alpha=0.3)
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / f"{filename_base}_visualisation.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"System performance visualisation saved: {plot_file}")