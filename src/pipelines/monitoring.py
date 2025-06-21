"""
Pipeline performance monitoring and metrics collection.

This module provides comprehensive monitoring capabilities for pipeline execution
including performance metrics, resource usage tracking, and alert generation.
"""

import json
import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import psutil

from src.utils.logging import get_logger
from src.utils.config import get_config

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class Alert:
    """System alert."""
    id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    pipeline_name: Optional[str] = None
    stage_name: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class PipelineExecutionSummary:
    """Summary of pipeline execution."""
    pipeline_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_stages: int = 0
    completed_stages: int = 0
    failed_stages: int = 0
    skipped_stages: int = 0
    total_records_processed: int = 0
    average_records_per_second: float = 0.0
    peak_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    total_disk_io_mb: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate execution duration."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_stages == 0:
            return 0.0
        return (self.completed_stages / self.total_stages) * 100


class MetricsCollector:
    """Collects and stores pipeline metrics."""
    
    def __init__(self, max_metrics: int = 10000):
        """
        Initialise metrics collector.
        
        Args:
            max_metrics: Maximum metrics to retain in memory
        """
        self.max_metrics = max_metrics
        self.metrics: deque = deque(maxlen=max_metrics)
        self.metric_series: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.RLock()
    
    def record_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        metric_type: MetricType = MetricType.GAUGE
    ) -> None:
        """
        Record a metric.
        
        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels
            metric_type: Type of metric
        """
        with self._lock:
            metric = Metric(
                name=name,
                value=value,
                timestamp=datetime.now(),
                labels=labels or {},
                metric_type=metric_type
            )
            
            self.metrics.append(metric)
            self.metric_series[name].append((metric.timestamp, value))
    
    def get_metrics(
        self,
        name_pattern: Optional[str] = None,
        since: Optional[datetime] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> List[Metric]:
        """
        Retrieve metrics based on filters.
        
        Args:
            name_pattern: Pattern to match metric names
            since: Only return metrics after this time
            labels: Labels to filter by
            
        Returns:
            Filtered metrics
        """
        with self._lock:
            filtered_metrics = []
            
            for metric in self.metrics:
                # Filter by name pattern
                if name_pattern and name_pattern not in metric.name:
                    continue
                
                # Filter by timestamp
                if since and metric.timestamp < since:
                    continue
                
                # Filter by labels
                if labels:
                    if not all(
                        metric.labels.get(k) == v
                        for k, v in labels.items()
                    ):
                        continue
                
                filtered_metrics.append(metric)
            
            return filtered_metrics
    
    def get_metric_series(self, name: str) -> List[Tuple[datetime, float]]:
        """Get time series data for a metric."""
        with self._lock:
            return list(self.metric_series.get(name, []))
    
    def get_latest_value(self, name: str) -> Optional[float]:
        """Get latest value for a metric."""
        series = self.get_metric_series(name)
        return series[-1][1] if series else None
    
    def calculate_statistics(
        self,
        name: str,
        since: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Calculate statistics for a metric.
        
        Args:
            name: Metric name
            since: Calculate stats since this time
            
        Returns:
            Statistics dictionary
        """
        metrics = self.get_metrics(name, since=since)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "sum": sum(values),
            "latest": values[-1] if values else 0.0
        }


class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self, max_alerts: int = 1000):
        """
        Initialise alert manager.
        
        Args:
            max_alerts: Maximum alerts to retain
        """
        self.max_alerts = max_alerts
        self.alerts: deque = deque(maxlen=max_alerts)
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self._lock = threading.RLock()
        self._alert_counter = 0
    
    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add alert handler."""
        self.alert_handlers.append(handler)
    
    def create_alert(
        self,
        severity: AlertSeverity,
        title: str,
        description: str,
        pipeline_name: Optional[str] = None,
        stage_name: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """
        Create and process an alert.
        
        Args:
            severity: Alert severity
            title: Alert title
            description: Alert description
            pipeline_name: Related pipeline
            stage_name: Related stage
            metrics: Associated metrics
            
        Returns:
            Created alert
        """
        with self._lock:
            self._alert_counter += 1
            
            alert = Alert(
                id=f"alert_{self._alert_counter}_{int(time.time())}",
                severity=severity,
                title=title,
                description=description,
                timestamp=datetime.now(),
                pipeline_name=pipeline_name,
                stage_name=stage_name,
                metrics=metrics or {}
            )
            
            self.alerts.append(alert)
            
            # Notify handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.log.error(
                        "Alert handler failed",
                        handler=handler.__name__,
                        error=str(e)
                    )
            
            logger.log.info(
                "Alert created",
                alert_id=alert.id,
                severity=severity.value,
                title=title
            )
            
            return alert
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: Alert ID to resolve
            
        Returns:
            Whether alert was found and resolved
        """
        with self._lock:
            for alert in self.alerts:
                if alert.id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    
                    logger.log.info(
                        "Alert resolved",
                        alert_id=alert_id
                    )
                    return True
            
            return False
    
    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        pipeline_name: Optional[str] = None
    ) -> List[Alert]:
        """Get active (unresolved) alerts."""
        with self._lock:
            active_alerts = [a for a in self.alerts if not a.resolved]
            
            if severity:
                active_alerts = [a for a in active_alerts if a.severity == severity]
            
            if pipeline_name:
                active_alerts = [a for a in active_alerts if a.pipeline_name == pipeline_name]
            
            return active_alerts


class SystemMonitor:
    """Monitors system resources and performance."""
    
    def __init__(self, interval_seconds: int = 5):
        """
        Initialise system monitor.
        
        Args:
            interval_seconds: Monitoring interval
        """
        self.interval = interval_seconds
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        
        # Resource thresholds
        self.cpu_threshold = get_config("monitoring.cpu_threshold", 80.0)
        self.memory_threshold = get_config("monitoring.memory_threshold", 80.0)
        self.disk_threshold = get_config("monitoring.disk_threshold", 90.0)
    
    def start(self) -> None:
        """Start system monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.log.info("System monitoring started")
    
    def stop(self) -> None:
        """Stop system monitoring."""
        self.is_monitoring = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10.0)
        
        logger.log.info("System monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics_collector.record_metric(
                    "system.cpu_percent",
                    cpu_percent,
                    metric_type=MetricType.GAUGE
                )
                
                if cpu_percent > self.cpu_threshold:
                    self.alert_manager.create_alert(
                        AlertSeverity.WARNING,
                        "High CPU Usage",
                        f"CPU usage is {cpu_percent:.1f}%",
                        metrics={"cpu_percent": cpu_percent}
                    )
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_mb = memory.used / 1024 / 1024
                
                self.metrics_collector.record_metric(
                    "system.memory_percent",
                    memory_percent,
                    metric_type=MetricType.GAUGE
                )
                self.metrics_collector.record_metric(
                    "system.memory_mb",
                    memory_mb,
                    metric_type=MetricType.GAUGE
                )
                
                if memory_percent > self.memory_threshold:
                    self.alert_manager.create_alert(
                        AlertSeverity.WARNING,
                        "High Memory Usage",
                        f"Memory usage is {memory_percent:.1f}%",
                        metrics={"memory_percent": memory_percent}
                    )
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                
                self.metrics_collector.record_metric(
                    "system.disk_percent",
                    disk_percent,
                    metric_type=MetricType.GAUGE
                )
                
                if disk_percent > self.disk_threshold:
                    self.alert_manager.create_alert(
                        AlertSeverity.ERROR,
                        "High Disk Usage",
                        f"Disk usage is {disk_percent:.1f}%",
                        metrics={"disk_percent": disk_percent}
                    )
                
                # Network I/O
                network = psutil.net_io_counters()
                network_mb = (network.bytes_sent + network.bytes_recv) / 1024 / 1024
                
                self.metrics_collector.record_metric(
                    "system.network_io_mb",
                    network_mb,
                    metric_type=MetricType.COUNTER
                )
                
            except Exception as e:
                logger.log.error(
                    "System monitoring error",
                    error=str(e)
                )
            
            time.sleep(self.interval)


class PipelineMonitor:
    """
    Comprehensive pipeline monitoring with metrics collection and alerting.
    
    Features:
    - Real-time performance monitoring
    - Resource usage tracking
    - Alert generation
    - Dashboard integration
    - Historical metrics storage
    """
    
    def __init__(
        self,
        metrics_dir: Optional[Path] = None,
        enable_system_monitoring: bool = True
    ):
        """
        Initialise pipeline monitor.
        
        Args:
            metrics_dir: Directory for storing metrics
            enable_system_monitoring: Whether to monitor system resources
        """
        self.metrics_dir = Path(metrics_dir or "metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.system_monitor = SystemMonitor() if enable_system_monitoring else None
        
        self.pipeline_summaries: Dict[str, PipelineExecutionSummary] = {}
        self.active_pipelines: Set[str] = set()
        
        self._lock = threading.RLock()
        
        # Set up alert handlers
        self.alert_manager.add_handler(self._log_alert)
        
        logger.log.info(
            "Pipeline monitor initialised",
            metrics_dir=str(self.metrics_dir)
        )
    
    def start_system_monitoring(self) -> None:
        """Start system resource monitoring."""
        if self.system_monitor:
            self.system_monitor.start()
    
    def stop_system_monitoring(self) -> None:
        """Stop system resource monitoring."""
        if self.system_monitor:
            self.system_monitor.stop()
    
    def start_pipeline_monitoring(self, pipeline_name: str) -> None:
        """
        Start monitoring a pipeline.
        
        Args:
            pipeline_name: Pipeline to monitor
        """
        with self._lock:
            self.active_pipelines.add(pipeline_name)
            
            summary = PipelineExecutionSummary(
                pipeline_name=pipeline_name,
                start_time=datetime.now()
            )
            self.pipeline_summaries[pipeline_name] = summary
            
            logger.log.info(
                "Started pipeline monitoring",
                pipeline=pipeline_name
            )
    
    def stop_pipeline_monitoring(self, pipeline_name: str) -> None:
        """
        Stop monitoring a pipeline.
        
        Args:
            pipeline_name: Pipeline to stop monitoring
        """
        with self._lock:
            self.active_pipelines.discard(pipeline_name)
            
            if pipeline_name in self.pipeline_summaries:
                summary = self.pipeline_summaries[pipeline_name]
                summary.end_time = datetime.now()
                
                # Save summary to file
                self._save_pipeline_summary(summary)
                
                logger.log.info(
                    "Stopped pipeline monitoring",
                    pipeline=pipeline_name,
                    duration=summary.duration.total_seconds() if summary.duration else 0
                )
    
    def record_stage_start(
        self,
        pipeline_name: str,
        stage_name: str
    ) -> None:
        """Record stage start."""
        self.metrics_collector.record_metric(
            "pipeline.stage.started",
            1,
            labels={
                "pipeline": pipeline_name,
                "stage": stage_name
            },
            metric_type=MetricType.COUNTER
        )
    
    def record_stage_completion(
        self,
        pipeline_name: str,
        stage_name: str,
        duration_seconds: float,
        records_processed: int = 0,
        stage_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record stage completion.
        
        Args:
            pipeline_name: Pipeline name
            stage_name: Stage name
            duration_seconds: Stage duration
            records_processed: Number of records processed
            stage_metrics: Additional stage metrics
        """
        # Record basic metrics
        self.metrics_collector.record_metric(
            "pipeline.stage.completed",
            1,
            labels={
                "pipeline": pipeline_name,
                "stage": stage_name
            },
            metric_type=MetricType.COUNTER
        )
        
        self.metrics_collector.record_metric(
            "pipeline.stage.duration",
            duration_seconds,
            labels={
                "pipeline": pipeline_name,
                "stage": stage_name
            },
            metric_type=MetricType.TIMER
        )
        
        if records_processed > 0:
            self.metrics_collector.record_metric(
                "pipeline.stage.records_processed",
                records_processed,
                labels={
                    "pipeline": pipeline_name,
                    "stage": stage_name
                },
                metric_type=MetricType.COUNTER
            )
            
            # Calculate processing rate
            if duration_seconds > 0:
                rate = records_processed / duration_seconds
                self.metrics_collector.record_metric(
                    "pipeline.stage.records_per_second",
                    rate,
                    labels={
                        "pipeline": pipeline_name,
                        "stage": stage_name
                    },
                    metric_type=MetricType.GAUGE
                )
        
        # Record additional stage metrics
        if stage_metrics:
            for metric_name, value in stage_metrics.items():
                if isinstance(value, (int, float)):
                    self.metrics_collector.record_metric(
                        f"pipeline.stage.{metric_name}",
                        value,
                        labels={
                            "pipeline": pipeline_name,
                            "stage": stage_name
                        }
                    )
        
        # Update pipeline summary
        with self._lock:
            if pipeline_name in self.pipeline_summaries:
                summary = self.pipeline_summaries[pipeline_name]
                summary.completed_stages += 1
                summary.total_records_processed += records_processed
                
                # Update resource usage peaks
                if stage_metrics:
                    cpu_percent = stage_metrics.get("cpu_percent", 0)
                    memory_mb = stage_metrics.get("memory_mb", 0)
                    disk_io_mb = stage_metrics.get("disk_io_mb", 0)
                    
                    if cpu_percent > summary.peak_cpu_percent:
                        summary.peak_cpu_percent = cpu_percent
                    
                    if memory_mb > summary.peak_memory_mb:
                        summary.peak_memory_mb = memory_mb
                    
                    summary.total_disk_io_mb += disk_io_mb
    
    def record_stage_failure(
        self,
        pipeline_name: str,
        stage_name: str,
        error: str
    ) -> None:
        """Record stage failure."""
        self.metrics_collector.record_metric(
            "pipeline.stage.failed",
            1,
            labels={
                "pipeline": pipeline_name,
                "stage": stage_name
            },
            metric_type=MetricType.COUNTER
        )
        
        # Create alert
        self.alert_manager.create_alert(
            AlertSeverity.ERROR,
            "Stage Execution Failed",
            f"Stage {stage_name} in pipeline {pipeline_name} failed: {error}",
            pipeline_name=pipeline_name,
            stage_name=stage_name
        )
        
        # Update pipeline summary
        with self._lock:
            if pipeline_name in self.pipeline_summaries:
                summary = self.pipeline_summaries[pipeline_name]
                summary.failed_stages += 1
                summary.errors.append(f"{stage_name}: {error}")
    
    def get_pipeline_status(self, pipeline_name: str) -> Dict[str, Any]:
        """Get current status of a pipeline."""
        with self._lock:
            is_active = pipeline_name in self.active_pipelines
            summary = self.pipeline_summaries.get(pipeline_name)
            
            status = {
                "pipeline_name": pipeline_name,
                "is_active": is_active,
                "summary": None
            }
            
            if summary:
                status["summary"] = {
                    "start_time": summary.start_time.isoformat(),
                    "end_time": summary.end_time.isoformat() if summary.end_time else None,
                    "duration_seconds": summary.duration.total_seconds() if summary.duration else None,
                    "total_stages": summary.total_stages,
                    "completed_stages": summary.completed_stages,
                    "failed_stages": summary.failed_stages,
                    "success_rate": summary.success_rate,
                    "total_records_processed": summary.total_records_processed,
                    "peak_memory_mb": summary.peak_memory_mb,
                    "peak_cpu_percent": summary.peak_cpu_percent,
                    "errors": summary.errors,
                    "warnings": summary.warnings
                }
            
            return status
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        with self._lock:
            # System metrics
            system_metrics = {}
            if self.system_monitor:
                system_metrics = {
                    "cpu_percent": self.system_monitor.metrics_collector.get_latest_value("system.cpu_percent"),
                    "memory_percent": self.system_monitor.metrics_collector.get_latest_value("system.memory_percent"),
                    "disk_percent": self.system_monitor.metrics_collector.get_latest_value("system.disk_percent")
                }
            
            # Pipeline summaries
            pipeline_summaries = []
            for summary in self.pipeline_summaries.values():
                pipeline_summaries.append({
                    "pipeline_name": summary.pipeline_name,
                    "is_active": summary.pipeline_name in self.active_pipelines,
                    "completed_stages": summary.completed_stages,
                    "failed_stages": summary.failed_stages,
                    "success_rate": summary.success_rate,
                    "duration": summary.duration.total_seconds() if summary.duration else None
                })
            
            # Active alerts
            active_alerts = []
            all_alerts = self.alert_manager.get_active_alerts()
            for alert in all_alerts[-10:]:  # Last 10 alerts
                active_alerts.append({
                    "id": alert.id,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "timestamp": alert.timestamp.isoformat(),
                    "pipeline_name": alert.pipeline_name
                })
            
            return {
                "timestamp": datetime.now().isoformat(),
                "system_metrics": system_metrics,
                "pipeline_summaries": pipeline_summaries,
                "active_alerts": active_alerts,
                "total_active_pipelines": len(self.active_pipelines)
            }
    
    def export_metrics(
        self,
        format: str = "json",
        since: Optional[datetime] = None
    ) -> str:
        """
        Export metrics in specified format.
        
        Args:
            format: Export format ('json', 'csv', 'prometheus')
            since: Export metrics since this time
            
        Returns:
            Exported metrics string
        """
        metrics = self.metrics_collector.get_metrics(since=since)
        
        if format == "json":
            return json.dumps([
                {
                    "name": m.name,
                    "value": m.value,
                    "timestamp": m.timestamp.isoformat(),
                    "labels": m.labels,
                    "type": m.metric_type.value
                }
                for m in metrics
            ], indent=2)
        
        elif format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Header
            writer.writerow(["name", "value", "timestamp", "labels", "type"])
            
            # Data
            for m in metrics:
                writer.writerow([
                    m.name,
                    m.value,
                    m.timestamp.isoformat(),
                    json.dumps(m.labels),
                    m.metric_type.value
                ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _log_alert(self, alert: Alert) -> None:
        """Log alert to logger."""
        log_level = {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "error",
            AlertSeverity.CRITICAL: "error"
        }[alert.severity]
        
        getattr(logger.log, log_level)(
            alert.description,
            alert_id=alert.id,
            pipeline=alert.pipeline_name,
            stage=alert.stage_name,
            **alert.metrics
        )
    
    def _save_pipeline_summary(self, summary: PipelineExecutionSummary) -> None:
        """Save pipeline summary to file."""
        try:
            summary_file = self.metrics_dir / f"pipeline_summary_{summary.pipeline_name}_{summary.start_time.strftime('%Y%m%d_%H%M%S')}.json"
            
            data = {
                "pipeline_name": summary.pipeline_name,
                "start_time": summary.start_time.isoformat(),
                "end_time": summary.end_time.isoformat() if summary.end_time else None,
                "duration_seconds": summary.duration.total_seconds() if summary.duration else None,
                "total_stages": summary.total_stages,
                "completed_stages": summary.completed_stages,
                "failed_stages": summary.failed_stages,
                "skipped_stages": summary.skipped_stages,
                "success_rate": summary.success_rate,
                "total_records_processed": summary.total_records_processed,
                "average_records_per_second": summary.average_records_per_second,
                "peak_memory_mb": summary.peak_memory_mb,
                "peak_cpu_percent": summary.peak_cpu_percent,
                "total_disk_io_mb": summary.total_disk_io_mb,
                "errors": summary.errors,
                "warnings": summary.warnings
            }
            
            with open(summary_file, "w") as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            logger.log.error(
                "Failed to save pipeline summary",
                pipeline=summary.pipeline_name,
                error=str(e)
            )