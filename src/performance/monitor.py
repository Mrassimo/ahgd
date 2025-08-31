#!/usr/bin/env python3
"""
AHGD V3: Real-Time Performance Monitor
Continuous monitoring and alerting for the modern health analytics platform.

Features:
- Real-time performance metrics collection
- Automatic alerting for performance degradation
- Historical performance tracking
- System health monitoring
- Resource utilization tracking
"""

import time
import psutil
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import json
import sqlite3
from pathlib import Path
import threading

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.logging import get_logger
from src.storage.parquet_manager import ParquetStorageManager

logger = get_logger("performance_monitor")


@dataclass
class MetricDataPoint:
    """Single performance metric data point."""
    
    timestamp: datetime
    metric_name: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric_name": self.metric_name,
            "value": self.value,
            "metadata": self.metadata
        }


@dataclass
class PerformanceAlert:
    """Performance alert definition."""
    
    alert_id: str
    metric_name: str
    condition: str  # "gt", "lt", "eq", "ne"
    threshold: float
    severity: str  # "low", "medium", "high", "critical"
    message: str
    enabled: bool = True
    consecutive_violations: int = 0
    last_triggered: Optional[datetime] = None
    
    def check_violation(self, value: float) -> bool:
        """Check if metric value violates the threshold."""
        if not self.enabled:
            return False
        
        if self.condition == "gt":
            return value > self.threshold
        elif self.condition == "lt":
            return value < self.threshold
        elif self.condition == "eq":
            return value == self.threshold
        elif self.condition == "ne":
            return value != self.threshold
        
        return False


class PerformanceMetricsCollector:
    """
    Collects comprehensive performance metrics for AHGD V3.
    
    Monitors:
    - System resources (CPU, memory, disk, network)
    - Application performance (response times, throughput)
    - Data processing metrics (Polars operations)
    - Storage performance (Parquet read/write)
    - Database performance (DuckDB queries)
    """
    
    def __init__(self, collection_interval: float = 30.0):
        """
        Initialize performance metrics collector.
        
        Args:
            collection_interval: Metrics collection interval in seconds
        """
        self.collection_interval = collection_interval
        self.metrics_history = deque(maxlen=2880)  # 24 hours at 30s intervals
        self.alerts: Dict[str, PerformanceAlert] = {}
        self.alert_handlers: List[Callable] = []
        
        # Initialize storage
        self.db_path = Path("data/performance_metrics.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # System monitoring
        self.process = psutil.Process()
        self.system_boot_time = psutil.boot_time()
        
        # Performance counters
        self.request_counts = deque(maxlen=100)  # Last 100 requests
        self.response_times = deque(maxlen=1000)  # Last 1000 response times
        self.error_counts = deque(maxlen=100)  # Last 100 errors
        
        # Default alerts
        self._setup_default_alerts()
        
        logger.info(f"Performance monitor initialized with {collection_interval}s collection interval")
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                alert_id TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                metric_value REAL,
                resolved BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)")
        
        conn.close()
    
    def _setup_default_alerts(self):
        """Setup default performance alerts."""
        
        default_alerts = [
            PerformanceAlert(
                alert_id="high_cpu_usage",
                metric_name="cpu_percent",
                condition="gt",
                threshold=80.0,
                severity="high",
                message="CPU usage above 80%"
            ),
            PerformanceAlert(
                alert_id="high_memory_usage",
                metric_name="memory_percent",
                condition="gt",
                threshold=85.0,
                severity="high",
                message="Memory usage above 85%"
            ),
            PerformanceAlert(
                alert_id="slow_response_time",
                metric_name="avg_response_time_ms",
                condition="gt",
                threshold=1000.0,
                severity="medium",
                message="Average response time above 1 second"
            ),
            PerformanceAlert(
                alert_id="high_error_rate",
                metric_name="error_rate_percent",
                condition="gt",
                threshold=5.0,
                severity="high",
                message="Error rate above 5%"
            ),
            PerformanceAlert(
                alert_id="low_disk_space",
                metric_name="disk_usage_percent",
                condition="gt",
                threshold=90.0,
                severity="critical",
                message="Disk usage above 90%"
            )
        ]
        
        for alert in default_alerts:
            self.alerts[alert.alert_id] = alert
    
    def collect_system_metrics(self) -> List[MetricDataPoint]:
        """Collect system-level performance metrics."""
        
        timestamp = datetime.now()
        metrics = []
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
        
        metrics.extend([
            MetricDataPoint(timestamp, "cpu_percent", cpu_percent),
            MetricDataPoint(timestamp, "cpu_count", cpu_count),
            MetricDataPoint(timestamp, "load_avg_1m", load_avg[0]),
            MetricDataPoint(timestamp, "load_avg_5m", load_avg[1]),
            MetricDataPoint(timestamp, "load_avg_15m", load_avg[2])
        ])
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        metrics.extend([
            MetricDataPoint(timestamp, "memory_total_gb", memory.total / (1024**3)),
            MetricDataPoint(timestamp, "memory_used_gb", memory.used / (1024**3)),
            MetricDataPoint(timestamp, "memory_percent", memory.percent),
            MetricDataPoint(timestamp, "memory_available_gb", memory.available / (1024**3)),
            MetricDataPoint(timestamp, "swap_percent", swap.percent)
        ])
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        metrics.extend([
            MetricDataPoint(timestamp, "disk_total_gb", disk.total / (1024**3)),
            MetricDataPoint(timestamp, "disk_used_gb", disk.used / (1024**3)),
            MetricDataPoint(timestamp, "disk_usage_percent", (disk.used / disk.total) * 100),
            MetricDataPoint(timestamp, "disk_read_mb_s", disk_io.read_bytes / (1024**2) if disk_io else 0),
            MetricDataPoint(timestamp, "disk_write_mb_s", disk_io.write_bytes / (1024**2) if disk_io else 0)
        ])
        
        # Network metrics
        network = psutil.net_io_counters()
        if network:
            metrics.extend([
                MetricDataPoint(timestamp, "network_sent_mb", network.bytes_sent / (1024**2)),
                MetricDataPoint(timestamp, "network_recv_mb", network.bytes_recv / (1024**2)),
                MetricDataPoint(timestamp, "network_packets_sent", network.packets_sent),
                MetricDataPoint(timestamp, "network_packets_recv", network.packets_recv)
            ])
        
        # Process-specific metrics
        try:
            process_memory = self.process.memory_info()
            process_cpu = self.process.cpu_percent()
            
            metrics.extend([
                MetricDataPoint(timestamp, "process_memory_mb", process_memory.rss / (1024**2)),
                MetricDataPoint(timestamp, "process_cpu_percent", process_cpu),
                MetricDataPoint(timestamp, "process_threads", self.process.num_threads())
            ])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            logger.warning("Could not collect process-specific metrics")
        
        return metrics
    
    def collect_application_metrics(self) -> List[MetricDataPoint]:
        """Collect application-level performance metrics."""
        
        timestamp = datetime.now()
        metrics = []
        
        # Request metrics
        if self.request_counts:
            recent_requests = len([t for t in self.request_counts if t > time.time() - 60])  # Last minute
            metrics.append(MetricDataPoint(timestamp, "requests_per_minute", recent_requests))
        
        # Response time metrics
        if self.response_times:
            recent_times = [t for t in self.response_times if t > 0]
            if recent_times:
                avg_response_time = sum(recent_times) / len(recent_times)
                p95_response_time = sorted(recent_times)[int(len(recent_times) * 0.95)]
                p99_response_time = sorted(recent_times)[int(len(recent_times) * 0.99)]
                
                metrics.extend([
                    MetricDataPoint(timestamp, "avg_response_time_ms", avg_response_time),
                    MetricDataPoint(timestamp, "p95_response_time_ms", p95_response_time),
                    MetricDataPoint(timestamp, "p99_response_time_ms", p99_response_time)
                ])
        
        # Error rate metrics
        if self.error_counts and self.request_counts:
            recent_errors = len([t for t in self.error_counts if t > time.time() - 300])  # Last 5 minutes
            recent_requests = len([t for t in self.request_counts if t > time.time() - 300])
            
            if recent_requests > 0:
                error_rate = (recent_errors / recent_requests) * 100
                metrics.append(MetricDataPoint(timestamp, "error_rate_percent", error_rate))
        
        return metrics
    
    def record_request(self, response_time_ms: float, is_error: bool = False):
        """Record an API request for performance tracking."""
        
        current_time = time.time()
        self.request_counts.append(current_time)
        self.response_times.append(response_time_ms)
        
        if is_error:
            self.error_counts.append(current_time)
    
    def collect_all_metrics(self) -> List[MetricDataPoint]:
        """Collect all available metrics."""
        
        all_metrics = []
        
        try:
            # System metrics
            all_metrics.extend(self.collect_system_metrics())
            
            # Application metrics
            all_metrics.extend(self.collect_application_metrics())
            
            # Add to history
            self.metrics_history.extend(all_metrics)
            
            # Store in database
            self._store_metrics(all_metrics)
            
            # Check for alerts
            self._check_alerts(all_metrics)
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
        
        return all_metrics
    
    def _store_metrics(self, metrics: List[MetricDataPoint]):
        """Store metrics in database."""
        
        conn = sqlite3.connect(self.db_path)
        
        for metric in metrics:
            conn.execute(
                "INSERT INTO metrics (timestamp, metric_name, value, metadata) VALUES (?, ?, ?, ?)",
                (metric.timestamp, metric.metric_name, metric.value, json.dumps(metric.metadata))
            )
        
        conn.commit()
        conn.close()
    
    def _check_alerts(self, metrics: List[MetricDataPoint]):
        """Check metrics against alert thresholds."""
        
        for metric in metrics:
            for alert_id, alert in self.alerts.items():
                if alert.metric_name == metric.metric_name:
                    if alert.check_violation(metric.value):
                        alert.consecutive_violations += 1
                        
                        # Trigger alert if consecutive violations exceed threshold
                        if alert.consecutive_violations >= 2:  # Require 2 consecutive violations
                            self._trigger_alert(alert, metric.value)
                    else:
                        alert.consecutive_violations = 0
    
    def _trigger_alert(self, alert: PerformanceAlert, metric_value: float):
        """Trigger a performance alert."""
        
        # Avoid duplicate alerts within 5 minutes
        if alert.last_triggered and (datetime.now() - alert.last_triggered).total_seconds() < 300:
            return
        
        alert.last_triggered = datetime.now()
        
        # Store alert in database
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO alerts (timestamp, alert_id, severity, message, metric_value) VALUES (?, ?, ?, ?, ?)",
            (datetime.now(), alert.alert_id, alert.severity, alert.message, metric_value)
        )
        conn.commit()
        conn.close()
        
        # Log alert
        logger.warning(f"🚨 PERFORMANCE ALERT [{alert.severity.upper()}]: {alert.message} (value: {metric_value})")
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert, metric_value)
            except Exception as e:
                logger.error(f"Alert handler failed: {str(e)}")
    
    def add_alert_handler(self, handler: Callable):
        """Add a custom alert handler function."""
        self.alert_handlers.append(handler)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics summary."""
        
        if not self.metrics_history:
            return {}
        
        # Get latest metrics
        latest_metrics = {}
        for metric in reversed(list(self.metrics_history)):
            if metric.metric_name not in latest_metrics:
                latest_metrics[metric.metric_name] = metric.value
        
        # Calculate derived metrics
        summary = {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": {},
            "application_metrics": {},
            "alerts": {
                "active": len([a for a in self.alerts.values() if a.consecutive_violations > 0]),
                "total": len(self.alerts)
            }
        }
        
        # Categorize metrics
        for metric_name, value in latest_metrics.items():
            if metric_name.startswith(("cpu_", "memory_", "disk_", "network_", "process_")):
                summary["system_metrics"][metric_name] = value
            else:
                summary["application_metrics"][metric_name] = value
        
        return summary
    
    def get_historical_metrics(
        self, 
        metric_names: List[str], 
        hours_back: int = 24
    ) -> Dict[str, List[Dict]]:
        """Get historical metrics data."""
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        conn = sqlite3.connect(self.db_path)
        
        results = {}
        for metric_name in metric_names:
            cursor = conn.execute(
                "SELECT timestamp, value FROM metrics WHERE metric_name = ? AND timestamp > ? ORDER BY timestamp",
                (metric_name, cutoff_time)
            )
            
            data_points = []
            for row in cursor.fetchall():
                data_points.append({
                    "timestamp": row[0],
                    "value": row[1]
                })
            
            results[metric_name] = data_points
        
        conn.close()
        return results
    
    def start_continuous_monitoring(self):
        """Start continuous performance monitoring in a separate thread."""
        
        def monitor_loop():
            logger.info("Starting continuous performance monitoring")
            
            while True:
                try:
                    self.collect_all_metrics()
                    time.sleep(self.collection_interval)
                except KeyboardInterrupt:
                    logger.info("Performance monitoring stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {str(e)}")
                    time.sleep(self.collection_interval)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
        return monitor_thread


def create_performance_dashboard():
    """Create a simple web dashboard for performance monitoring."""
    
    try:
        from flask import Flask, jsonify, render_template_string
        
        app = Flask(__name__)
        monitor = PerformanceMetricsCollector()
        
        # Start monitoring
        monitor.start_continuous_monitoring()
        
        @app.route('/metrics')
        def get_metrics():
            """API endpoint for current metrics."""
            return jsonify(monitor.get_current_metrics())
        
        @app.route('/historical/<metric_name>')
        def get_historical(metric_name):
            """API endpoint for historical metrics."""
            hours = request.args.get('hours', 24, type=int)
            data = monitor.get_historical_metrics([metric_name], hours)
            return jsonify(data)
        
        @app.route('/')
        def dashboard():
            """Simple dashboard."""
            dashboard_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>AHGD V3 Performance Dashboard</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .metric-card { 
                        border: 1px solid #ddd; 
                        border-radius: 5px; 
                        padding: 15px; 
                        margin: 10px; 
                        display: inline-block; 
                        min-width: 200px; 
                    }
                    .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
                    .metric-label { color: #7f8c8d; margin-bottom: 5px; }
                </style>
            </head>
            <body>
                <h1>🚀 AHGD V3 Performance Dashboard</h1>
                <div id="metrics-container"></div>
                
                <script>
                function updateMetrics() {
                    fetch('/metrics')
                        .then(response => response.json())
                        .then(data => {
                            let html = '';
                            
                            // System metrics
                            if (data.system_metrics) {
                                html += '<h2>System Metrics</h2>';
                                for (const [key, value] of Object.entries(data.system_metrics)) {
                                    const displayValue = typeof value === 'number' ? value.toFixed(2) : value;
                                    html += `
                                        <div class="metric-card">
                                            <div class="metric-label">${key.replace(/_/g, ' ')}</div>
                                            <div class="metric-value">${displayValue}</div>
                                        </div>
                                    `;
                                }
                            }
                            
                            // Application metrics
                            if (data.application_metrics) {
                                html += '<h2>Application Metrics</h2>';
                                for (const [key, value] of Object.entries(data.application_metrics)) {
                                    const displayValue = typeof value === 'number' ? value.toFixed(2) : value;
                                    html += `
                                        <div class="metric-card">
                                            <div class="metric-label">${key.replace(/_/g, ' ')}</div>
                                            <div class="metric-value">${displayValue}</div>
                                        </div>
                                    `;
                                }
                            }
                            
                            document.getElementById('metrics-container').innerHTML = html;
                        })
                        .catch(error => console.error('Error:', error));
                }
                
                // Update every 30 seconds
                updateMetrics();
                setInterval(updateMetrics, 30000);
                </script>
            </body>
            </html>
            """
            return render_template_string(dashboard_html)
        
        logger.info("Performance dashboard starting on http://localhost:5001")
        app.run(host='0.0.0.0', port=5001, debug=False)
        
    except ImportError:
        logger.error("Flask not available. Install with: pip install flask")
        return None


def main():
    """Run the performance monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AHGD V3 Performance Monitor")
    parser.add_argument("--interval", type=float, default=30.0, help="Collection interval in seconds")
    parser.add_argument("--dashboard", action="store_true", help="Start web dashboard")
    parser.add_argument("--duration", type=int, help="Monitor duration in minutes (default: continuous)")
    
    args = parser.parse_args()
    
    if args.dashboard:
        create_performance_dashboard()
    else:
        monitor = PerformanceMetricsCollector(collection_interval=args.interval)
        
        # Add a simple alert handler
        def alert_handler(alert, value):
            print(f"🚨 ALERT: {alert.message} (value: {value:.2f})")
        
        monitor.add_alert_handler(alert_handler)
        
        # Start monitoring
        if args.duration:
            logger.info(f"Starting performance monitoring for {args.duration} minutes")
            end_time = time.time() + (args.duration * 60)
            
            while time.time() < end_time:
                metrics = monitor.collect_all_metrics()
                print(f"Collected {len(metrics)} metrics at {datetime.now().strftime('%H:%M:%S')}")
                time.sleep(args.interval)
        else:
            logger.info("Starting continuous performance monitoring (Ctrl+C to stop)")
            monitor_thread = monitor.start_continuous_monitoring()
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")


if __name__ == "__main__":
    main()