"""
AHGD Monitoring and Health Check Framework

This module provides comprehensive monitoring capabilities including:
- Health check utilities for system components
- Performance metrics collection and analysis
- Error tracking and alerting mechanisms
- System resource monitoring (CPU, memory, disk, network)
- Automated alerting and notification systems
"""

import psutil
import time
import json
import os
import threading
import smtplib
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
from collections import defaultdict, deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import requests
from loguru import logger
import yaml

from .logging import get_logger


@dataclass
class HealthCheckResult:
    """Result of a health check operation"""
    component: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    message: str
    timestamp: datetime
    duration: float
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class PerformanceMetrics:
    """System performance metrics snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available: int
    disk_usage_percent: float
    disk_free: int
    network_io: Dict[str, int]
    process_count: int
    load_average: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class Alert:
    """Alert configuration and tracking"""
    name: str
    condition: str
    threshold: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    cooldown_minutes: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        if self.last_triggered:
            data['last_triggered'] = self.last_triggered.isoformat()
        return data


class SystemMonitor:
    """
    System resource monitoring and health checking
    
    Provides comprehensive monitoring of:
    - CPU usage and load average
    - Memory usage and availability
    - Disk usage and I/O
    - Network I/O
    - Process counts and status
    - Custom application metrics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics_history = deque(maxlen=self.config.get('history_size', 1000))
        self.alerts = {}
        self.health_checks = {}
        self.monitoring_active = False
        self.monitor_thread = None
        self.logger = get_logger()
        
        # Load alerts from configuration
        self._load_alerts()
    
    def _load_alerts(self):
        """Load alert configurations"""
        default_alerts = [
            Alert('high_cpu', 'cpu_percent > threshold', 80.0, 'high'),
            Alert('high_memory', 'memory_percent > threshold', 85.0, 'high'),
            Alert('low_disk_space', 'disk_usage_percent > threshold', 90.0, 'critical'),
            Alert('high_load', 'load_average[0] > threshold', 4.0, 'medium'),
        ]
        
        for alert in default_alerts:
            self.alerts[alert.name] = alert
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current system performance metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        
        # Network metrics
        network = psutil.net_io_counters()
        network_io = {
            'bytes_sent': network.bytes_sent,
            'bytes_recv': network.bytes_recv,
            'packets_sent': network.packets_sent,
            'packets_recv': network.packets_recv
        }
        
        # Process count
        process_count = len(psutil.pids())
        
        # Load average (Unix/Linux only)
        load_average = None
        if hasattr(os, 'getloadavg'):
            try:
                load_average = list(os.getloadavg())
            except OSError:
                pass
        
        return PerformanceMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available=memory.available,
            disk_usage_percent=disk.percent,
            disk_free=disk.free,
            network_io=network_io,
            process_count=process_count,
            load_average=load_average
        )
    
    def collect_metrics(self):
        """Collect and store current metrics"""
        metrics = self.get_current_metrics()
        self.metrics_history.append(metrics)
        
        # Check alerts
        self._check_alerts(metrics)
        
        return metrics
    
    def _check_alerts(self, metrics: PerformanceMetrics):
        """Check alert conditions against current metrics"""
        for alert_name, alert in self.alerts.items():
            if not alert.enabled:
                continue
            
            # Check cooldown period
            if alert.last_triggered:
                cooldown_end = alert.last_triggered + timedelta(minutes=alert.cooldown_minutes)
                if datetime.now(timezone.utc) < cooldown_end:
                    continue
            
            # Evaluate alert condition
            triggered = self._evaluate_alert_condition(alert, metrics)
            
            if triggered:
                alert.last_triggered = datetime.now(timezone.utc)
                alert.trigger_count += 1
                self._trigger_alert(alert, metrics)
    
    def _evaluate_alert_condition(self, alert: Alert, metrics: PerformanceMetrics) -> bool:
        """Evaluate if alert condition is met"""
        try:
            # Create evaluation context
            context = {
                'cpu_percent': metrics.cpu_percent,
                'memory_percent': metrics.memory_percent,
                'disk_usage_percent': metrics.disk_usage_percent,
                'process_count': metrics.process_count,
                'load_average': metrics.load_average or [0, 0, 0],
                'threshold': alert.threshold
            }
            
            # Evaluate condition
            return eval(alert.condition, {"__builtins__": {}}, context)
            
        except Exception as e:
            self.logger.error(f"Error evaluating alert condition: {alert.name}: {e}")
            return False
    
    def _trigger_alert(self, alert: Alert, metrics: PerformanceMetrics):
        """Trigger alert notification"""
        alert_data = {
            'alert': alert.to_dict(),
            'metrics': metrics.to_dict(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        self.logger.warning(
            f"Alert triggered: {alert.name}",
            alert_name=alert.name,
            severity=alert.severity,
            alert_data=alert_data
        )
        
        # Send notifications
        self._send_notifications(alert, alert_data)
    
    def _send_notifications(self, alert: Alert, alert_data: Dict[str, Any]):
        """Send alert notifications"""
        notification_config = self.config.get('notifications', {})
        
        # Email notifications
        if notification_config.get('email', {}).get('enabled', False):
            self._send_email_notification(alert, alert_data, notification_config['email'])
        
        # Webhook notifications
        if notification_config.get('webhook', {}).get('enabled', False):
            self._send_webhook_notification(alert, alert_data, notification_config['webhook'])
        
        # Slack notifications
        if notification_config.get('slack', {}).get('enabled', False):
            self._send_slack_notification(alert, alert_data, notification_config['slack'])
    
    def _send_email_notification(self, alert: Alert, alert_data: Dict[str, Any], email_config: Dict[str, Any]):
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = email_config['from']
            msg['To'] = ', '.join(email_config['to'])
            msg['Subject'] = f"AHGD Alert: {alert.name} ({alert.severity})"
            
            body = f"""
            Alert: {alert.name}
            Severity: {alert.severity}
            Condition: {alert.condition}
            Threshold: {alert.threshold}
            
            Current Metrics:
            {json.dumps(alert_data['metrics'], indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                if email_config.get('use_tls', True):
                    server.starttls()
                if email_config.get('username') and email_config.get('password'):
                    server.login(email_config['username'], email_config['password'])
                server.send_message(msg)
                
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
    
    def _send_webhook_notification(self, alert: Alert, alert_data: Dict[str, Any], webhook_config: Dict[str, Any]):
        """Send webhook notification"""
        try:
            payload = {
                'alert_name': alert.name,
                'severity': alert.severity,
                'alert_data': alert_data
            }
            
            response = requests.post(
                webhook_config['url'],
                json=payload,
                headers=webhook_config.get('headers', {}),
                timeout=30
            )
            response.raise_for_status()
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
    
    def _send_slack_notification(self, alert: Alert, alert_data: Dict[str, Any], slack_config: Dict[str, Any]):
        """Send Slack notification"""
        try:
            payload = {
                'text': f"🚨 AHGD Alert: {alert.name}",
                'attachments': [{
                    'color': 'danger' if alert.severity == 'critical' else 'warning',
                    'fields': [
                        {'title': 'Severity', 'value': alert.severity, 'short': True},
                        {'title': 'Condition', 'value': alert.condition, 'short': True},
                        {'title': 'Threshold', 'value': str(alert.threshold), 'short': True},
                        {'title': 'Timestamp', 'value': alert_data['timestamp'], 'short': True}
                    ]
                }]
            }
            
            response = requests.post(
                slack_config['webhook_url'],
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {e}")
    
    def start_monitoring(self, interval: int = 60):
        """Start continuous monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info(f"System monitoring started with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("System monitoring stopped")
    
    def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self.collect_metrics()
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of metrics over specified time period"""
        if not self.metrics_history:
            return {}
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {}
        
        # Calculate statistics
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        disk_values = [m.disk_usage_percent for m in recent_metrics]
        
        return {
            'timespan_hours': hours,
            'sample_count': len(recent_metrics),
            'cpu': {
                'avg': sum(cpu_values) / len(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values)
            },
            'memory': {
                'avg': sum(memory_values) / len(memory_values),
                'min': min(memory_values),
                'max': max(memory_values)
            },
            'disk': {
                'avg': sum(disk_values) / len(disk_values),
                'min': min(disk_values),
                'max': max(disk_values)
            },
            'latest_metrics': recent_metrics[-1].to_dict() if recent_metrics else None
        }


class HealthChecker:
    """
    Health check framework for application components
    
    Provides health checking for:
    - Database connections
    - External service availability
    - File system access
    - Configuration validity
    - Custom application components
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.health_checks = {}
        self.last_results = {}
        self.logger = get_logger()
    
    def register_health_check(self, name: str, check_func: Callable[[], bool], 
                            description: str = "", timeout: int = 30):
        """Register a health check function"""
        self.health_checks[name] = {
            'function': check_func,
            'description': description,
            'timeout': timeout
        }
    
    def run_health_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check"""
        if name not in self.health_checks:
            return HealthCheckResult(
                component=name,
                status='unhealthy',
                message=f"Health check '{name}' not found",
                timestamp=datetime.now(timezone.utc),
                duration=0.0
            )
        
        check_config = self.health_checks[name]
        start_time = time.time()
        
        try:
            # Run the health check with timeout
            result = self._run_with_timeout(
                check_config['function'],
                check_config['timeout']
            )
            
            duration = time.time() - start_time
            
            health_result = HealthCheckResult(
                component=name,
                status='healthy' if result else 'unhealthy',
                message=check_config['description'] or f"Health check for {name}",
                timestamp=datetime.now(timezone.utc),
                duration=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            health_result = HealthCheckResult(
                component=name,
                status='unhealthy',
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.now(timezone.utc),
                duration=duration,
                metadata={'error': str(e)}
            )
        
        self.last_results[name] = health_result
        
        self.logger.info(
            f"Health check completed: {name}",
            component=name,
            status=health_result.status,
            duration=health_result.duration
        )
        
        return health_result
    
    def _run_with_timeout(self, func: Callable, timeout: int) -> Any:
        """Run function with timeout"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Health check timed out after {timeout}s")
        
        # Set timeout (Unix/Linux only)
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
        
        try:
            result = func()
            return result
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
    
    def run_all_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        results = {}
        
        for name in self.health_checks:
            results[name] = self.run_health_check(name)
        
        return results
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary"""
        results = self.run_all_health_checks()
        
        total_checks = len(results)
        healthy_checks = sum(1 for r in results.values() if r.status == 'healthy')
        unhealthy_checks = sum(1 for r in results.values() if r.status == 'unhealthy')
        
        overall_status = 'healthy'
        if unhealthy_checks > 0:
            overall_status = 'degraded' if healthy_checks > unhealthy_checks else 'unhealthy'
        
        return {
            'overall_status': overall_status,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'total_checks': total_checks,
                'healthy': healthy_checks,
                'unhealthy': unhealthy_checks
            },
            'details': {name: result.to_dict() for name, result in results.items()}
        }


class ErrorTracker:
    """
    Error tracking and analysis system
    
    Tracks and analyzes:
    - Exception frequency and patterns
    - Error rates over time
    - Critical error alerting
    - Error classification and categorization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.error_history = deque(maxlen=self.config.get('history_size', 10000))
        self.error_counts = defaultdict(int)
        self.logger = get_logger()
    
    def track_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Track an error occurrence"""
        error_data = {
            'timestamp': datetime.now(timezone.utc),
            'type': type(error).__name__,
            'message': str(error),
            'context': context or {},
            'traceback': str(error.__traceback__) if error.__traceback__ else None
        }
        
        self.error_history.append(error_data)
        self.error_counts[error_data['type']] += 1
        
        self.logger.error(
            f"Error tracked: {error_data['type']}",
            error_type=error_data['type'],
            error_message=error_data['message'],
            context=context
        )
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for specified time period"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_errors = [e for e in self.error_history if e['timestamp'] >= cutoff_time]
        
        # Count by type
        type_counts = defaultdict(int)
        for error in recent_errors:
            type_counts[error['type']] += 1
        
        return {
            'timespan_hours': hours,
            'total_errors': len(recent_errors),
            'error_types': dict(type_counts),
            'error_rate': len(recent_errors) / hours if hours > 0 else 0,
            'recent_errors': recent_errors[-10:]  # Last 10 errors
        }


# Global instances
_system_monitor = None
_health_checker = None
_error_tracker = None


def get_system_monitor(config: Optional[Dict[str, Any]] = None) -> SystemMonitor:
    """Get or create global system monitor"""
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMonitor(config)
    return _system_monitor


def get_health_checker(config: Optional[Dict[str, Any]] = None) -> HealthChecker:
    """Get or create global health checker"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker(config)
    return _health_checker


def get_error_tracker(config: Optional[Dict[str, Any]] = None) -> ErrorTracker:
    """Get or create global error tracker"""
    global _error_tracker
    if _error_tracker is None:
        _error_tracker = ErrorTracker(config)
    return _error_tracker


# Convenience functions for common health checks
def check_database_connection(connection_string: str) -> bool:
    """Check database connection health"""
    try:
        import sqlalchemy
        engine = sqlalchemy.create_engine(connection_string)
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1"))
        return True
    except Exception:
        return False


def check_file_system_access(path: str) -> bool:
    """Check file system access"""
    try:
        test_path = Path(path)
        if not test_path.exists():
            test_path.mkdir(parents=True, exist_ok=True)
        
        # Test write access
        test_file = test_path / f".health_check_{int(time.time())}"
        test_file.write_text("health check")
        test_file.unlink()
        return True
    except Exception:
        return False


def check_external_service(url: str, timeout: int = 10) -> bool:
    """Check external service availability"""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False