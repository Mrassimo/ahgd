"""
Automated Alerting and Notification System for Australian Health Analytics Dashboard

Features:
- Performance threshold monitoring
- Multi-channel alert delivery (email, webhook, log)
- Alert aggregation and rate limiting
- Escalation policies
- Alert history and analytics
- Custom alert rules and conditions
"""

import json
import logging
import smtplib
import threading
import time

try:
    from email.mime.multipart import MimeMultipart
    from email.mime.text import MimeText
except ImportError:
    MimeText = None
    MimeMultipart = None
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Optional

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from .health import HealthCheck
from .health import HealthStatus
from .monitoring import PerformanceMetric

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels"""

    LOG = "log"
    EMAIL = "email"
    WEBHOOK = "webhook"
    CONSOLE = "console"
    FILE = "file"


@dataclass
class AlertRule:
    """Alert rule configuration"""

    name: str
    condition: str  # Python expression
    severity: AlertSeverity
    channels: list[AlertChannel]
    message_template: str
    cooldown_minutes: int = 15
    max_alerts_per_hour: int = 10
    enabled: bool = True
    tags: dict[str, str] = field(default_factory=dict)
    escalation_delay_minutes: int = 60
    escalation_channels: list[AlertChannel] = field(default_factory=list)

    # State tracking
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    last_escalated: Optional[datetime] = None


@dataclass
class Alert:
    """Individual alert instance"""

    rule_name: str
    message: str
    severity: AlertSeverity
    timestamp: datetime = field(default_factory=datetime.now)
    channels: list[AlertChannel] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    escalated: bool = False
    escalated_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "rule_name": self.rule_name,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "channels": [ch.value for ch in self.channels],
            "context": self.context,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "escalated": self.escalated,
            "escalated_at": self.escalated_at.isoformat() if self.escalated_at else None,
        }


@dataclass
class AlertConfig:
    """Alert system configuration"""

    # Email settings
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: bool = True
    email_from: str = "ahgd-alerts@example.com"
    email_to: list[str] = field(default_factory=list)

    # Webhook settings
    webhook_urls: list[str] = field(default_factory=list)
    webhook_timeout: int = 10
    webhook_retry_count: int = 3

    # File settings
    alert_log_file: Optional[Path] = None
    max_log_size_mb: int = 10

    # Rate limiting
    global_rate_limit: int = 100  # Max alerts per hour
    rate_limit_window_minutes: int = 60

    # Alert aggregation
    aggregation_enabled: bool = True
    aggregation_window_minutes: int = 5

    # Storage
    alert_history_enabled: bool = True
    max_alert_history: int = 10000
    alert_storage_file: Optional[Path] = None


class AlertChannel_Interface(ABC):
    """Abstract base class for alert delivery channels"""

    @abstractmethod
    def send_alert(self, alert: Alert) -> bool:
        """Send alert through this channel"""
        pass

    @abstractmethod
    def get_channel_type(self) -> AlertChannel:
        """Get channel type"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if channel is available"""
        pass


class LogAlertChannel(AlertChannel_Interface):
    """Log-based alert channel"""

    def __init__(self, logger_name: str = "alerts"):
        self.logger = logging.getLogger(logger_name)

    def send_alert(self, alert: Alert) -> bool:
        """Send alert to log"""
        try:
            severity_levels = {
                AlertSeverity.LOW: logging.INFO,
                AlertSeverity.MEDIUM: logging.WARNING,
                AlertSeverity.HIGH: logging.ERROR,
                AlertSeverity.CRITICAL: logging.CRITICAL,
            }

            level = severity_levels.get(alert.severity, logging.WARNING)

            log_message = (
                f"ALERT [{alert.severity.value.upper()}] {alert.rule_name}: {alert.message}"
            )
            if alert.context:
                log_message += f" | Context: {json.dumps(alert.context)}"

            self.logger.log(level, log_message)
            return True

        except Exception as e:
            logger.error(f"Failed to send log alert: {e}")
            return False

    def get_channel_type(self) -> AlertChannel:
        return AlertChannel.LOG

    def is_available(self) -> bool:
        return True


class EmailAlertChannel(AlertChannel_Interface):
    """Email-based alert channel"""

    def __init__(self, config: AlertConfig):
        self.config = config

    def send_alert(self, alert: Alert) -> bool:
        """Send alert via email"""
        if not self.config.email_to:
            return False

        try:
            # Check if email modules are available
            if MimeMultipart is None or MimeText is None:
                self.logger.warning(
                    "Email functionality not available - MimeText/MimeMultipart not imported"
                )
                return False

            # Create message
            msg = MimeMultipart()
            msg["From"] = self.config.email_from
            msg["To"] = ", ".join(self.config.email_to)
            msg["Subject"] = f"[{alert.severity.value.upper()}] AHGD Alert: {alert.rule_name}"

            # Email body
            body = self._format_email_body(alert)
            msg.attach(MimeText(body, "html"))

            # Send email
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                if self.config.smtp_use_tls:
                    server.starttls()

                if self.config.smtp_username and self.config.smtp_password:
                    server.login(self.config.smtp_username, self.config.smtp_password)

                server.send_message(msg)

            logger.info(f"Email alert sent for {alert.rule_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    def _format_email_body(self, alert: Alert) -> str:
        """Format email body HTML"""
        severity_colors = {
            AlertSeverity.LOW: "#17a2b8",
            AlertSeverity.MEDIUM: "#ffc107",
            AlertSeverity.HIGH: "#fd7e14",
            AlertSeverity.CRITICAL: "#dc3545",
        }

        color = severity_colors.get(alert.severity, "#6c757d")

        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="border-left: 4px solid {color}; padding-left: 20px; margin: 20px 0;">
                <h2 style="color: {color}; margin-top: 0;">
                    Alert: {alert.rule_name}
                </h2>
                <p><strong>Severity:</strong> {alert.severity.value.upper()}</p>
                <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Message:</strong> {alert.message}</p>
        """

        if alert.context:
            html += "<h3>Context:</h3><ul>"
            for key, value in alert.context.items():
                html += f"<li><strong>{key}:</strong> {value}</li>"
            html += "</ul>"

        html += """
            </div>
            <hr>
            <p style="color: #6c757d; font-size: 12px;">
                This alert was generated by the Australian Health Analytics Dashboard monitoring system.
            </p>
        </body>
        </html>
        """

        return html

    def get_channel_type(self) -> AlertChannel:
        return AlertChannel.EMAIL

    def is_available(self) -> bool:
        return bool(self.config.email_to and self.config.smtp_host)


class WebhookAlertChannel(AlertChannel_Interface):
    """Webhook-based alert channel"""

    def __init__(self, config: AlertConfig):
        self.config = config

    def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook"""
        if not REQUESTS_AVAILABLE or not self.config.webhook_urls:
            return False

        payload = {
            "alert": alert.to_dict(),
            "system": "Australian Health Analytics Dashboard",
            "timestamp": datetime.now().isoformat(),
        }

        success_count = 0

        for url in self.config.webhook_urls:
            if self._send_webhook(url, payload):
                success_count += 1

        return success_count > 0

    def _send_webhook(self, url: str, payload: dict[str, Any]) -> bool:
        """Send individual webhook"""
        for attempt in range(self.config.webhook_retry_count):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.config.webhook_timeout,
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code < 400:
                    logger.info(f"Webhook alert sent to {url}")
                    return True
                else:
                    logger.warning(f"Webhook failed with status {response.status_code}: {url}")

            except Exception as e:
                logger.error(f"Webhook attempt {attempt + 1} failed for {url}: {e}")

                if attempt < self.config.webhook_retry_count - 1:
                    time.sleep(2**attempt)  # Exponential backoff

        return False

    def get_channel_type(self) -> AlertChannel:
        return AlertChannel.WEBHOOK

    def is_available(self) -> bool:
        return REQUESTS_AVAILABLE and bool(self.config.webhook_urls)


class FileAlertChannel(AlertChannel_Interface):
    """File-based alert channel"""

    def __init__(self, config: AlertConfig):
        self.config = config
        self.log_file = config.alert_log_file or Path("alerts.log")
        self._ensure_log_file()

    def _ensure_log_file(self):
        """Ensure log file exists and is writable"""
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self.log_file.touch(exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create alert log file: {e}")

    def send_alert(self, alert: Alert) -> bool:
        """Send alert to file"""
        try:
            # Check file size and rotate if needed
            if (
                self.log_file.exists()
                and self.log_file.stat().st_size > self.config.max_log_size_mb * 1024 * 1024
            ):
                self._rotate_log_file()

            # Write alert
            alert_line = json.dumps(alert.to_dict()) + "\n"

            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(alert_line)

            return True

        except Exception as e:
            logger.error(f"Failed to write alert to file: {e}")
            return False

    def _rotate_log_file(self):
        """Rotate log file when it gets too large"""
        try:
            backup_file = self.log_file.with_suffix(f".{int(time.time())}.log")
            self.log_file.rename(backup_file)
            self.log_file.touch()
            logger.info(f"Rotated alert log file to {backup_file}")
        except Exception as e:
            logger.error(f"Failed to rotate alert log file: {e}")

    def get_channel_type(self) -> AlertChannel:
        return AlertChannel.FILE

    def is_available(self) -> bool:
        return True


class ConsoleAlertChannel(AlertChannel_Interface):
    """Console output alert channel"""

    def send_alert(self, alert: Alert) -> bool:
        """Send alert to console"""
        try:
            severity_symbols = {
                AlertSeverity.LOW: "ℹ️",
                AlertSeverity.MEDIUM: "⚠️",
                AlertSeverity.HIGH: "🚨",
                AlertSeverity.CRITICAL: "🔥",
            }

            symbol = severity_symbols.get(alert.severity, "⚠️")

            print(
                f"\n{symbol} ALERT [{alert.severity.value.upper()}] {alert.timestamp.strftime('%H:%M:%S')}"
            )
            print(f"Rule: {alert.rule_name}")
            print(f"Message: {alert.message}")

            if alert.context:
                print("Context:")
                for key, value in alert.context.items():
                    print(f"  {key}: {value}")
            print("-" * 50)

            return True

        except Exception as e:
            logger.error(f"Failed to send console alert: {e}")
            return False

    def get_channel_type(self) -> AlertChannel:
        return AlertChannel.CONSOLE

    def is_available(self) -> bool:
        return True


class AlertManager:
    """Main alert management system"""

    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self.rules: dict[str, AlertRule] = {}
        self.channels: dict[AlertChannel, AlertChannel_Interface] = {}
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=self.config.max_alert_history)
        self.rate_limiter = defaultdict(deque)
        self.aggregation_buffer: dict[str, list[Alert]] = defaultdict(list)

        # Threading
        self._lock = threading.Lock()
        self.background_thread = None
        self.running = False

        # Initialize channels
        self._init_channels()

        # Load alert storage
        if self.config.alert_storage_file:
            self._load_alert_storage()

        # Start background processing
        self.start_background_processing()

    def _init_channels(self):
        """Initialize alert delivery channels"""
        self.channels[AlertChannel.LOG] = LogAlertChannel()
        self.channels[AlertChannel.EMAIL] = EmailAlertChannel(self.config)
        self.channels[AlertChannel.WEBHOOK] = WebhookAlertChannel(self.config)
        self.channels[AlertChannel.FILE] = FileAlertChannel(self.config)
        self.channels[AlertChannel.CONSOLE] = ConsoleAlertChannel()

        # Log available channels
        available_channels = [
            ch.value for ch, handler in self.channels.items() if handler.is_available()
        ]
        logger.info(f"Alert channels available: {available_channels}")

    def add_rule(self, rule: AlertRule):
        """Add alert rule"""
        with self._lock:
            self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, rule_name: str):
        """Remove alert rule"""
        with self._lock:
            if rule_name in self.rules:
                del self.rules[rule_name]
                logger.info(f"Removed alert rule: {rule_name}")

    def evaluate_metric(self, metric: PerformanceMetric):
        """Evaluate metric against alert rules"""
        with self._lock:
            for rule in self.rules.values():
                if not rule.enabled:
                    continue

                try:
                    # Create evaluation context
                    context = {
                        "metric": metric,
                        "value": metric.value,
                        "name": metric.name,
                        "category": metric.category,
                        "timestamp": metric.timestamp,
                        "tags": metric.tags,
                        "metadata": metric.metadata,
                    }

                    # Evaluate condition
                    if self._evaluate_condition(rule.condition, context):
                        self._trigger_alert(rule, metric, context)

                except Exception as e:
                    logger.error(f"Error evaluating rule {rule.name}: {e}")

    def evaluate_health_check(self, health_check: HealthCheck):
        """Evaluate health check against alert rules"""
        # Convert health check to metric-like structure for evaluation
        metric_value = 0 if health_check.status == HealthStatus.HEALTHY else 1

        context = {
            "health_check": health_check,
            "value": metric_value,
            "name": health_check.name,
            "status": health_check.status.value,
            "message": health_check.message,
            "duration_ms": health_check.duration_ms,
            "timestamp": health_check.timestamp,
            "metadata": health_check.metadata,
        }

        with self._lock:
            for rule in self.rules.values():
                if not rule.enabled:
                    continue

                try:
                    if self._evaluate_condition(rule.condition, context):
                        self._trigger_health_alert(rule, health_check, context)

                except Exception as e:
                    logger.error(f"Error evaluating health rule {rule.name}: {e}")

    def _evaluate_condition(self, condition: str, context: dict[str, Any]) -> bool:
        """Safely evaluate alert condition"""
        try:
            # Define safe functions for use in conditions
            safe_functions = {
                "abs": abs,
                "min": min,
                "max": max,
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
            }

            # Merge context with safe functions
            eval_context = {**context, **safe_functions}

            # Evaluate condition
            result = eval(condition, {"__builtins__": {}}, eval_context)
            return bool(result)

        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False

    def _trigger_alert(self, rule: AlertRule, metric: PerformanceMetric, context: dict[str, Any]):
        """Trigger alert for metric"""
        # Check rate limiting
        if not self._check_rate_limit(rule):
            return

        # Create alert
        alert_message = rule.message_template.format(**context)

        alert = Alert(
            rule_name=rule.name,
            message=alert_message,
            severity=rule.severity,
            channels=rule.channels,
            context={
                "metric_name": metric.name,
                "metric_value": metric.value,
                "metric_category": metric.category,
                "metric_timestamp": metric.timestamp.isoformat(),
            },
        )

        self._process_alert(alert, rule)

    def _trigger_health_alert(
        self, rule: AlertRule, health_check: HealthCheck, context: dict[str, Any]
    ):
        """Trigger alert for health check"""
        # Check rate limiting
        if not self._check_rate_limit(rule):
            return

        # Create alert
        alert_message = rule.message_template.format(**context)

        alert = Alert(
            rule_name=rule.name,
            message=alert_message,
            severity=rule.severity,
            channels=rule.channels,
            context={
                "check_name": health_check.name,
                "check_status": health_check.status.value,
                "check_message": health_check.message,
                "check_duration_ms": health_check.duration_ms,
                "check_timestamp": health_check.timestamp.isoformat(),
            },
        )

        self._process_alert(alert, rule)

    def _check_rate_limit(self, rule: AlertRule) -> bool:
        """Check if alert is rate limited"""
        now = datetime.now()

        # Check rule-specific cooldown
        if rule.last_triggered:
            cooldown_delta = timedelta(minutes=rule.cooldown_minutes)
            if now - rule.last_triggered < cooldown_delta:
                return False

        # Check rule-specific rate limit
        rule_alerts = self.rate_limiter[f"rule_{rule.name}"]
        cutoff_time = now - timedelta(hours=1)

        # Remove old alerts
        while rule_alerts and rule_alerts[0] < cutoff_time:
            rule_alerts.popleft()

        if len(rule_alerts) >= rule.max_alerts_per_hour:
            return False

        # Check global rate limit
        global_alerts = self.rate_limiter["global"]
        cutoff_time = now - timedelta(minutes=self.config.rate_limit_window_minutes)

        while global_alerts and global_alerts[0] < cutoff_time:
            global_alerts.popleft()

        if len(global_alerts) >= self.config.global_rate_limit:
            return False

        # Update rate limiters
        rule_alerts.append(now)
        global_alerts.append(now)
        rule.last_triggered = now
        rule.trigger_count += 1

        return True

    def _process_alert(self, alert: Alert, rule: AlertRule):
        """Process and deliver alert"""
        # Add to active alerts
        self.active_alerts[f"{rule.name}_{alert.timestamp.isoformat()}"] = alert

        # Add to history
        self.alert_history.append(alert)

        # Handle aggregation
        if self.config.aggregation_enabled:
            self.aggregation_buffer[rule.name].append(alert)
        else:
            self._deliver_alert(alert)

        # Save to storage
        if self.config.alert_storage_file:
            self._save_alert_to_storage(alert)

        logger.info(f"Alert triggered: {rule.name} - {alert.message}")

    def _deliver_alert(self, alert: Alert):
        """Deliver alert through configured channels"""
        for channel_type in alert.channels:
            if channel_type in self.channels:
                channel = self.channels[channel_type]
                if channel.is_available():
                    try:
                        success = channel.send_alert(alert)
                        if not success:
                            logger.error(f"Failed to deliver alert via {channel_type.value}")
                    except Exception as e:
                        logger.error(f"Error delivering alert via {channel_type.value}: {e}")

    def _deliver_aggregated_alerts(self):
        """Deliver aggregated alerts"""
        with self._lock:
            for rule_name, alerts in self.aggregation_buffer.items():
                if not alerts:
                    continue

                # Create aggregated alert
                if len(alerts) == 1:
                    self._deliver_alert(alerts[0])
                else:
                    aggregated_alert = self._create_aggregated_alert(rule_name, alerts)
                    self._deliver_alert(aggregated_alert)

                # Clear buffer
                alerts.clear()

    def _create_aggregated_alert(self, rule_name: str, alerts: list[Alert]) -> Alert:
        """Create aggregated alert from multiple alerts"""
        first_alert = alerts[0]

        message = f"Multiple alerts for {rule_name} ({len(alerts)} occurrences in {self.config.aggregation_window_minutes} minutes)"

        # Aggregate context
        context = {
            "aggregated": True,
            "alert_count": len(alerts),
            "first_alert_time": alerts[0].timestamp.isoformat(),
            "last_alert_time": alerts[-1].timestamp.isoformat(),
            "individual_messages": [alert.message for alert in alerts],
        }

        return Alert(
            rule_name=f"{rule_name}_aggregated",
            message=message,
            severity=max(alert.severity for alert in alerts),
            channels=first_alert.channels,
            context=context,
        )

    def start_background_processing(self):
        """Start background processing thread"""
        if self.running:
            return

        self.running = True
        self.background_thread = threading.Thread(target=self._background_worker, daemon=True)
        self.background_thread.start()
        logger.info("Started alert background processing")

    def stop_background_processing(self):
        """Stop background processing thread"""
        self.running = False
        if self.background_thread:
            self.background_thread.join(timeout=2)
        logger.info("Stopped alert background processing")

    def _background_worker(self):
        """Background worker for alert processing"""
        while self.running:
            try:
                # Process aggregated alerts
                if self.config.aggregation_enabled:
                    self._deliver_aggregated_alerts()

                # Check for escalations
                self._check_escalations()

                # Cleanup old alerts
                self._cleanup_old_alerts()

                time.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in alert background processing: {e}")
                time.sleep(30)

    def _check_escalations(self):
        """Check for alert escalations"""
        now = datetime.now()

        with self._lock:
            for rule in self.rules.values():
                if (
                    rule.escalation_channels
                    and rule.last_triggered
                    and not rule.last_escalated
                    and now - rule.last_triggered > timedelta(minutes=rule.escalation_delay_minutes)
                ):
                    # Create escalation alert
                    escalation_alert = Alert(
                        rule_name=f"{rule.name}_escalation",
                        message=f"ESCALATION: Alert {rule.name} has not been resolved after {rule.escalation_delay_minutes} minutes",
                        severity=AlertSeverity.CRITICAL,
                        channels=rule.escalation_channels,
                        context={"original_rule": rule.name, "escalated": True},
                    )

                    self._deliver_alert(escalation_alert)
                    rule.last_escalated = now

                    logger.warning(f"Alert escalated: {rule.name}")

    def _cleanup_old_alerts(self):
        """Cleanup old active alerts"""
        cutoff_time = datetime.now() - timedelta(hours=24)

        with self._lock:
            old_alert_keys = [
                key for key, alert in self.active_alerts.items() if alert.timestamp < cutoff_time
            ]

            for key in old_alert_keys:
                del self.active_alerts[key]

    def _load_alert_storage(self):
        """Load alerts from storage file"""
        try:
            if self.config.alert_storage_file and self.config.alert_storage_file.exists():
                with open(self.config.alert_storage_file) as f:
                    data = json.load(f)

                # Load alert history
                for alert_data in data.get("history", []):
                    alert = Alert(**alert_data)
                    self.alert_history.append(alert)

                logger.info(f"Loaded {len(self.alert_history)} alerts from storage")

        except Exception as e:
            logger.error(f"Failed to load alert storage: {e}")

    def _save_alert_to_storage(self, alert: Alert):
        """Save alert to storage file"""
        if not self.config.alert_storage_file:
            return

        try:
            # Load existing data
            data = {"history": []}
            if self.config.alert_storage_file.exists():
                with open(self.config.alert_storage_file) as f:
                    data = json.load(f)

            # Add new alert
            data["history"].append(alert.to_dict())

            # Limit history size
            if len(data["history"]) > self.config.max_alert_history:
                data["history"] = data["history"][-self.config.max_alert_history :]

            # Save back
            self.config.alert_storage_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.alert_storage_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save alert to storage: {e}")

    def resolve_alert(self, rule_name: str):
        """Manually resolve active alerts for a rule"""
        now = datetime.now()
        resolved_count = 0

        with self._lock:
            for alert in self.active_alerts.values():
                if alert.rule_name == rule_name and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = now
                    resolved_count += 1

            # Reset rule state
            if rule_name in self.rules:
                rule = self.rules[rule_name]
                rule.last_triggered = None
                rule.last_escalated = None
                rule.trigger_count = 0

        logger.info(f"Resolved {resolved_count} alerts for rule: {rule_name}")
        return resolved_count

    def get_alert_statistics(self) -> dict[str, Any]:
        """Get alert system statistics"""
        with self._lock:
            stats = {
                "total_rules": len(self.rules),
                "enabled_rules": sum(1 for rule in self.rules.values() if rule.enabled),
                "active_alerts": len(self.active_alerts),
                "total_history": len(self.alert_history),
                "available_channels": [
                    ch.value for ch, handler in self.channels.items() if handler.is_available()
                ],
                "rules_triggered_24h": 0,
                "alerts_by_severity": defaultdict(int),
                "recent_rules": [],
            }

            # Analyze recent alerts
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_alerts = [alert for alert in self.alert_history if alert.timestamp > cutoff_time]

            stats["rules_triggered_24h"] = len(set(alert.rule_name for alert in recent_alerts))

            for alert in recent_alerts:
                stats["alerts_by_severity"][alert.severity.value] += 1

            # Get most recently triggered rules
            stats["recent_rules"] = list(
                set(alert.rule_name for alert in list(self.alert_history)[-10:])
            )

            return dict(stats)


# Default alert rules for common scenarios
def create_default_alert_rules() -> list[AlertRule]:
    """Create default alert rules for common monitoring scenarios"""
    return [
        AlertRule(
            name="high_cpu_usage",
            condition="name == 'cpu_usage_percent' and value > 80",
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.LOG, AlertChannel.CONSOLE],
            message_template="High CPU usage detected: {value:.1f}%",
            cooldown_minutes=10,
        ),
        AlertRule(
            name="critical_memory_usage",
            condition="name == 'memory_usage_percent' and value > 90",
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.WEBHOOK],
            message_template="Critical memory usage: {value:.1f}%",
            cooldown_minutes=5,
        ),
        AlertRule(
            name="slow_database_query",
            condition="name.startswith('db_query_duration_') and value > 5.0",
            severity=AlertSeverity.MEDIUM,
            channels=[AlertChannel.LOG],
            message_template="Slow database query detected: {name} took {value:.2f}s",
            cooldown_minutes=15,
        ),
        AlertRule(
            name="health_check_failure",
            condition="health_check and status in ['critical', 'unknown']",
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.LOG, AlertChannel.CONSOLE, AlertChannel.EMAIL],
            message_template="Health check failed: {name} - {message}",
            cooldown_minutes=10,
        ),
        AlertRule(
            name="page_load_slow",
            condition="name.startswith('page_load_time_') and value > 10.0",
            severity=AlertSeverity.MEDIUM,
            channels=[AlertChannel.LOG],
            message_template="Slow page load: {name} took {value:.2f}s",
            cooldown_minutes=20,
        ),
    ]


# Global alert manager instance
_global_alert_manager: Optional[AlertManager] = None


def get_alert_manager(config: Optional[AlertConfig] = None) -> AlertManager:
    """Get or create global alert manager"""
    global _global_alert_manager
    if _global_alert_manager is None:
        _global_alert_manager = AlertManager(config)

        # Add default rules
        for rule in create_default_alert_rules():
            _global_alert_manager.add_rule(rule)

    return _global_alert_manager


if __name__ == "__main__":
    # Test alert system
    from .health import HealthCheck
    from .health import HealthStatus
    from .monitoring import PerformanceMetric

    print("Testing alert system...")

    # Create alert manager
    config = AlertConfig(
        email_to=["admin@example.com"],
        webhook_urls=["http://localhost:8080/webhook"],
        alert_log_file=Path("test_alerts.log"),
    )

    alert_manager = AlertManager(config)

    # Add test rule
    test_rule = AlertRule(
        name="test_high_value",
        condition="value > 50",
        severity=AlertSeverity.HIGH,
        channels=[AlertChannel.LOG, AlertChannel.CONSOLE, AlertChannel.FILE],
        message_template="Test alert: value is {value}",
    )

    alert_manager.add_rule(test_rule)

    # Test with metric
    test_metric = PerformanceMetric(
        name="test_metric", value=75, timestamp=datetime.now(), category="test"
    )

    alert_manager.evaluate_metric(test_metric)

    # Test with health check
    test_health = HealthCheck(
        name="test_check", status=HealthStatus.CRITICAL, message="Test failure"
    )

    # Add health check rule
    health_rule = AlertRule(
        name="test_health_failure",
        condition="health_check and status == 'critical'",
        severity=AlertSeverity.CRITICAL,
        channels=[AlertChannel.LOG, AlertChannel.CONSOLE],
        message_template="Health check failed: {name}",
    )

    alert_manager.add_rule(health_rule)
    alert_manager.evaluate_health_check(test_health)

    # Get statistics
    stats = alert_manager.get_alert_statistics()
    print(f"Alert statistics: {json.dumps(stats, indent=2)}")

    # Wait a moment for background processing
    time.sleep(2)

    # Cleanup
    alert_manager.stop_background_processing()

    # Remove test files
    test_log = Path("test_alerts.log")
    if test_log.exists():
        test_log.unlink()

    print("Alert system test completed!")
