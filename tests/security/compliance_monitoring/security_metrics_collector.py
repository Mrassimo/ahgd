"""
Security Metrics Collector - Phase 5.6

Comprehensive security metrics collection and analysis for the Australian Health Analytics platform.
Provides real-time security monitoring, trend analysis, and performance metrics for
security controls and compliance requirements.

Key Features:
- Real-time security metrics collection
- Security performance trend analysis
- Compliance metrics monitoring
- Automated security alerting
- Security dashboard data generation
"""

import time
import json
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import threading
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of security metrics."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    ENCRYPTION = "encryption"
    VULNERABILITY = "vulnerability"
    INCIDENT = "incident"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"
    AVAILABILITY = "availability"


class MetricSeverity(Enum):
    """Metric severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Security alert types."""
    THRESHOLD_BREACH = "threshold_breach"
    ANOMALY_DETECTED = "anomaly_detected"
    COMPLIANCE_VIOLATION = "compliance_violation"
    SECURITY_INCIDENT = "security_incident"
    SYSTEM_FAILURE = "system_failure"


@dataclass
class SecurityMetric:
    """Individual security metric."""
    metric_id: str
    metric_type: MetricType
    name: str
    value: float
    unit: str
    timestamp: datetime
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    source_system: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityAlert:
    """Security alert."""
    alert_id: str
    alert_type: AlertType
    severity: MetricSeverity
    title: str
    description: str
    affected_metrics: List[str]
    detection_timestamp: datetime
    source_system: str
    remediation_suggestions: List[str]
    acknowledged: bool = False
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None


@dataclass
class SecurityTrend:
    """Security metric trend analysis."""
    metric_id: str
    time_period: str
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_strength: float  # 0-1 scale
    change_percentage: float
    significance_level: float
    anomalies_detected: int
    predictions: Dict[str, float]


@dataclass
class SecurityMetrics:
    """Collection of security metrics."""
    collection_timestamp: datetime
    authentication_metrics: Dict[str, float]
    authorization_metrics: Dict[str, float]
    data_access_metrics: Dict[str, float]
    encryption_metrics: Dict[str, float]
    vulnerability_metrics: Dict[str, float]
    incident_metrics: Dict[str, float]
    compliance_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    availability_metrics: Dict[str, float]
    overall_security_score: float


class SecurityMetricsCollector:
    """
    Comprehensive security metrics collector for Australian health data systems.
    Collects, analyzes, and reports on security performance and compliance metrics.
    """
    
    def __init__(self, 
                 collection_interval_seconds: int = 60,
                 enable_real_time_monitoring: bool = True,
                 metric_retention_days: int = 90):
        """
        Initialize security metrics collector.
        
        Args:
            collection_interval_seconds: Interval between metric collections
            enable_real_time_monitoring: Enable real-time monitoring
            metric_retention_days: Number of days to retain metrics
        """
        self.collection_interval = collection_interval_seconds
        self.enable_real_time_monitoring = enable_real_time_monitoring
        self.retention_days = metric_retention_days
        
        # Metrics storage
        self.metrics: deque = deque(maxlen=10000)  # Rolling buffer
        self.alerts: List[SecurityAlert] = []
        self.trends: Dict[str, SecurityTrend] = {}
        
        # Collection state
        self.is_collecting = False
        self.collection_thread = None
        self.last_collection_time = None
        
        # Metric definitions and thresholds
        self.metric_definitions = self._initialize_metric_definitions()
        
        # Security baselines for anomaly detection
        self.baselines: Dict[str, Dict[str, float]] = {}
        
        # Australian health data specific metrics
        self.health_security_metrics = {
            'patient_data_access_rate': {'threshold_warning': 100, 'threshold_critical': 200},
            'health_record_encryption_coverage': {'threshold_warning': 0.95, 'threshold_critical': 0.90},
            'clinical_data_breach_incidents': {'threshold_warning': 1, 'threshold_critical': 3},
            'health_system_availability': {'threshold_warning': 0.95, 'threshold_critical': 0.90},
            'privacy_violation_rate': {'threshold_warning': 0.01, 'threshold_critical': 0.05},
            'consent_management_compliance': {'threshold_warning': 0.95, 'threshold_critical': 0.90}
        }
        
        logger.info("Security metrics collector initialized")
    
    def _initialize_metric_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize metric definitions and configurations."""
        return {
            'authentication_success_rate': {
                'type': MetricType.AUTHENTICATION,
                'unit': 'percentage',
                'threshold_warning': 0.95,
                'threshold_critical': 0.90,
                'description': 'Percentage of successful authentication attempts'
            },
            'authentication_failure_rate': {
                'type': MetricType.AUTHENTICATION,
                'unit': 'percentage',
                'threshold_warning': 0.05,
                'threshold_critical': 0.10,
                'description': 'Percentage of failed authentication attempts'
            },
            'multi_factor_auth_coverage': {
                'type': MetricType.AUTHENTICATION,
                'unit': 'percentage',
                'threshold_warning': 0.90,
                'threshold_critical': 0.80,
                'description': 'Percentage of users with MFA enabled'
            },
            'unauthorized_access_attempts': {
                'type': MetricType.AUTHORIZATION,
                'unit': 'count',
                'threshold_warning': 10,
                'threshold_critical': 25,
                'description': 'Number of unauthorized access attempts'
            },
            'privilege_escalation_attempts': {
                'type': MetricType.AUTHORIZATION,
                'unit': 'count',
                'threshold_warning': 1,
                'threshold_critical': 5,
                'description': 'Number of privilege escalation attempts'
            },
            'data_encryption_coverage': {
                'type': MetricType.ENCRYPTION,
                'unit': 'percentage',
                'threshold_warning': 0.95,
                'threshold_critical': 0.90,
                'description': 'Percentage of data encrypted at rest'
            },
            'tls_compliance_rate': {
                'type': MetricType.ENCRYPTION,
                'unit': 'percentage',
                'threshold_warning': 0.95,
                'threshold_critical': 0.90,
                'description': 'Percentage of connections using TLS 1.3+'
            },
            'vulnerability_scan_coverage': {
                'type': MetricType.VULNERABILITY,
                'unit': 'percentage',
                'threshold_warning': 0.95,
                'threshold_critical': 0.90,
                'description': 'Percentage of systems covered by vulnerability scans'
            },
            'high_severity_vulnerabilities': {
                'type': MetricType.VULNERABILITY,
                'unit': 'count',
                'threshold_warning': 5,
                'threshold_critical': 10,
                'description': 'Number of high severity vulnerabilities'
            },
            'security_incident_count': {
                'type': MetricType.INCIDENT,
                'unit': 'count',
                'threshold_warning': 1,
                'threshold_critical': 3,
                'description': 'Number of security incidents'
            },
            'incident_response_time': {
                'type': MetricType.INCIDENT,
                'unit': 'minutes',
                'threshold_warning': 60,
                'threshold_critical': 120,
                'description': 'Average incident response time'
            },
            'compliance_score': {
                'type': MetricType.COMPLIANCE,
                'unit': 'percentage',
                'threshold_warning': 0.85,
                'threshold_critical': 0.75,
                'description': 'Overall compliance score'
            },
            'audit_log_completeness': {
                'type': MetricType.COMPLIANCE,
                'unit': 'percentage',
                'threshold_warning': 0.95,
                'threshold_critical': 0.90,
                'description': 'Percentage of complete audit logs'
            },
            'system_response_time': {
                'type': MetricType.PERFORMANCE,
                'unit': 'milliseconds',
                'threshold_warning': 2000,
                'threshold_critical': 5000,
                'description': 'Average system response time'
            },
            'system_availability': {
                'type': MetricType.AVAILABILITY,
                'unit': 'percentage',
                'threshold_warning': 0.95,
                'threshold_critical': 0.90,
                'description': 'System availability percentage'
            }
        }
    
    def start_collection(self):
        """Start real-time metrics collection."""
        if self.is_collecting:
            logger.warning("Metrics collection already running")
            return
        
        self.is_collecting = True
        if self.enable_real_time_monitoring:
            self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
            self.collection_thread.start()
            logger.info("Started real-time metrics collection")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=10)
        logger.info("Stopped metrics collection")
    
    def _collection_loop(self):
        """Main collection loop for real-time monitoring."""
        while self.is_collecting:
            try:
                metrics = self.collect_current_metrics()
                self._process_metrics(metrics)
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(self.collection_interval)
    
    def collect_current_metrics(self) -> SecurityMetrics:
        """
        Collect current security metrics from all monitored systems.
        
        Returns:
            SecurityMetrics: Current security metrics
        """
        collection_time = datetime.now()
        
        # Collect authentication metrics
        auth_metrics = self._collect_authentication_metrics()
        
        # Collect authorization metrics
        authz_metrics = self._collect_authorization_metrics()
        
        # Collect data access metrics
        data_access_metrics = self._collect_data_access_metrics()
        
        # Collect encryption metrics
        encryption_metrics = self._collect_encryption_metrics()
        
        # Collect vulnerability metrics
        vulnerability_metrics = self._collect_vulnerability_metrics()
        
        # Collect incident metrics
        incident_metrics = self._collect_incident_metrics()
        
        # Collect compliance metrics
        compliance_metrics = self._collect_compliance_metrics()
        
        # Collect performance metrics
        performance_metrics = self._collect_performance_metrics()
        
        # Collect availability metrics
        availability_metrics = self._collect_availability_metrics()
        
        # Calculate overall security score
        overall_score = self._calculate_overall_security_score(
            auth_metrics, authz_metrics, data_access_metrics,
            encryption_metrics, vulnerability_metrics, incident_metrics,
            compliance_metrics, performance_metrics, availability_metrics
        )
        
        metrics = SecurityMetrics(
            collection_timestamp=collection_time,
            authentication_metrics=auth_metrics,
            authorization_metrics=authz_metrics,
            data_access_metrics=data_access_metrics,
            encryption_metrics=encryption_metrics,
            vulnerability_metrics=vulnerability_metrics,
            incident_metrics=incident_metrics,
            compliance_metrics=compliance_metrics,
            performance_metrics=performance_metrics,
            availability_metrics=availability_metrics,
            overall_security_score=overall_score
        )
        
        self.last_collection_time = collection_time
        return metrics
    
    def _collect_authentication_metrics(self) -> Dict[str, float]:
        """Collect authentication-related metrics."""
        # Simulate metric collection - in practice, these would come from actual systems
        import random
        
        return {
            'authentication_success_rate': random.uniform(0.92, 0.99),
            'authentication_failure_rate': random.uniform(0.01, 0.08),
            'multi_factor_auth_coverage': random.uniform(0.85, 0.98),
            'password_policy_compliance': random.uniform(0.88, 0.96),
            'session_timeout_compliance': random.uniform(0.90, 0.99),
            'account_lockout_events': random.randint(0, 5),
            'suspicious_login_attempts': random.randint(0, 10)
        }
    
    def _collect_authorization_metrics(self) -> Dict[str, float]:
        """Collect authorization-related metrics."""
        import random
        
        return {
            'unauthorized_access_attempts': random.randint(0, 15),
            'privilege_escalation_attempts': random.randint(0, 3),
            'role_based_access_compliance': random.uniform(0.90, 0.99),
            'permission_violations': random.randint(0, 8),
            'access_control_effectiveness': random.uniform(0.88, 0.97),
            'administrative_access_monitoring': random.uniform(0.92, 0.99)
        }
    
    def _collect_data_access_metrics(self) -> Dict[str, float]:
        """Collect data access-related metrics."""
        import random
        
        return {
            'patient_data_access_rate': random.randint(50, 150),
            'health_record_access_violations': random.randint(0, 3),
            'data_export_monitoring': random.uniform(0.95, 1.0),
            'clinical_data_access_compliance': random.uniform(0.90, 0.98),
            'after_hours_access_events': random.randint(0, 12),
            'bulk_data_access_events': random.randint(0, 5)
        }
    
    def _collect_encryption_metrics(self) -> Dict[str, float]:
        """Collect encryption-related metrics."""
        import random
        
        return {
            'data_encryption_coverage': random.uniform(0.92, 0.99),
            'tls_compliance_rate': random.uniform(0.94, 1.0),
            'key_management_compliance': random.uniform(0.88, 0.96),
            'encryption_key_rotation_compliance': random.uniform(0.85, 0.95),
            'certificate_validity_monitoring': random.uniform(0.90, 0.99),
            'cryptographic_strength_compliance': random.uniform(0.92, 0.99)
        }
    
    def _collect_vulnerability_metrics(self) -> Dict[str, float]:
        """Collect vulnerability-related metrics."""
        import random
        
        return {
            'vulnerability_scan_coverage': random.uniform(0.88, 0.98),
            'high_severity_vulnerabilities': random.randint(0, 8),
            'medium_severity_vulnerabilities': random.randint(5, 25),
            'vulnerability_remediation_time': random.uniform(24, 168),  # hours
            'patch_management_compliance': random.uniform(0.85, 0.96),
            'security_configuration_compliance': random.uniform(0.90, 0.98)
        }
    
    def _collect_incident_metrics(self) -> Dict[str, float]:
        """Collect incident-related metrics."""
        import random
        
        return {
            'security_incident_count': random.randint(0, 2),
            'incident_response_time': random.uniform(15, 90),  # minutes
            'incident_resolution_time': random.uniform(60, 480),  # minutes
            'false_positive_rate': random.uniform(0.05, 0.20),
            'security_alert_volume': random.randint(10, 100),
            'threat_detection_accuracy': random.uniform(0.80, 0.95)
        }
    
    def _collect_compliance_metrics(self) -> Dict[str, float]:
        """Collect compliance-related metrics."""
        import random
        
        return {
            'compliance_score': random.uniform(0.82, 0.96),
            'audit_log_completeness': random.uniform(0.92, 0.99),
            'privacy_policy_compliance': random.uniform(0.88, 0.97),
            'data_retention_compliance': random.uniform(0.90, 0.98),
            'consent_management_compliance': random.uniform(0.85, 0.95),
            'app_compliance_score': random.uniform(0.80, 0.94)
        }
    
    def _collect_performance_metrics(self) -> Dict[str, float]:
        """Collect performance-related metrics."""
        import random
        
        # Get actual system metrics where possible
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        
        return {
            'system_response_time': random.uniform(200, 1500),  # milliseconds
            'cpu_utilization': cpu_usage,
            'memory_utilization': memory_usage,
            'disk_utilization': disk_usage,
            'network_latency': random.uniform(10, 100),  # milliseconds
            'database_response_time': random.uniform(50, 500)  # milliseconds
        }
    
    def _collect_availability_metrics(self) -> Dict[str, float]:
        """Collect availability-related metrics."""
        import random
        
        return {
            'system_availability': random.uniform(0.92, 0.999),
            'service_uptime': random.uniform(0.95, 1.0),
            'planned_downtime_compliance': random.uniform(0.90, 1.0),
            'backup_success_rate': random.uniform(0.88, 0.99),
            'disaster_recovery_readiness': random.uniform(0.85, 0.95),
            'business_continuity_score': random.uniform(0.80, 0.94)
        }
    
    def _calculate_overall_security_score(self, *metric_groups) -> float:
        """Calculate overall security score from all metric groups."""
        all_metrics = {}
        for group in metric_groups:
            all_metrics.update(group)
        
        # Weight different metric types
        weights = {
            'authentication_success_rate': 0.10,
            'unauthorized_access_attempts': -0.05,  # Negative impact
            'data_encryption_coverage': 0.15,
            'high_severity_vulnerabilities': -0.10,  # Negative impact
            'security_incident_count': -0.10,  # Negative impact
            'compliance_score': 0.20,
            'system_availability': 0.15,
            'tls_compliance_rate': 0.10
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric_name, value in all_metrics.items():
            if metric_name in weights:
                weight = weights[metric_name]
                
                # Handle negative impact metrics
                if weight < 0:
                    # Convert counts/rates to impact scores
                    if isinstance(value, int):
                        impact = min(1.0, value / 10.0)  # Scale counts
                    else:
                        impact = value
                    weighted_score += abs(weight) * (1.0 - impact)
                else:
                    # Normalize percentage metrics
                    if value > 1.0:
                        normalized_value = min(1.0, value / 100.0)
                    else:
                        normalized_value = value
                    weighted_score += weight * normalized_value
                
                total_weight += abs(weight)
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _process_metrics(self, metrics: SecurityMetrics):
        """Process collected metrics for analysis and alerting."""
        # Store metrics
        self.metrics.append(metrics)
        
        # Check thresholds and generate alerts
        self._check_thresholds(metrics)
        
        # Update trends
        self._update_trends(metrics)
        
        # Clean old metrics
        self._cleanup_old_metrics()
    
    def _check_thresholds(self, metrics: SecurityMetrics):
        """Check metrics against thresholds and generate alerts."""
        all_metrics = {
            **metrics.authentication_metrics,
            **metrics.authorization_metrics,
            **metrics.data_access_metrics,
            **metrics.encryption_metrics,
            **metrics.vulnerability_metrics,
            **metrics.incident_metrics,
            **metrics.compliance_metrics,
            **metrics.performance_metrics,
            **metrics.availability_metrics
        }
        
        for metric_name, value in all_metrics.items():
            definition = self.metric_definitions.get(metric_name)
            if not definition:
                continue
            
            threshold_critical = definition.get('threshold_critical')
            threshold_warning = definition.get('threshold_warning')
            
            alert_severity = None
            
            # Determine if threshold is breached
            if threshold_critical is not None:
                if (definition.get('type') in [MetricType.AUTHENTICATION, MetricType.ENCRYPTION, 
                                             MetricType.COMPLIANCE, MetricType.AVAILABILITY] and 
                    value < threshold_critical):
                    alert_severity = MetricSeverity.CRITICAL
                elif (definition.get('type') in [MetricType.VULNERABILITY, MetricType.INCIDENT] and 
                      value > threshold_critical):
                    alert_severity = MetricSeverity.CRITICAL
            
            if alert_severity is None and threshold_warning is not None:
                if (definition.get('type') in [MetricType.AUTHENTICATION, MetricType.ENCRYPTION, 
                                             MetricType.COMPLIANCE, MetricType.AVAILABILITY] and 
                    value < threshold_warning):
                    alert_severity = MetricSeverity.WARNING
                elif (definition.get('type') in [MetricType.VULNERABILITY, MetricType.INCIDENT] and 
                      value > threshold_warning):
                    alert_severity = MetricSeverity.WARNING
            
            # Generate alert if threshold breached
            if alert_severity:
                self._generate_alert(
                    alert_type=AlertType.THRESHOLD_BREACH,
                    severity=alert_severity,
                    title=f"Threshold breach: {metric_name}",
                    description=f"Metric {metric_name} value {value} breached threshold",
                    affected_metrics=[metric_name],
                    source_system="metrics_collector"
                )
    
    def _update_trends(self, metrics: SecurityMetrics):
        """Update trend analysis for metrics."""
        if len(self.metrics) < 10:
            return  # Need sufficient data for trend analysis
        
        # Get recent metrics for trend calculation
        recent_metrics = list(self.metrics)[-10:]
        
        all_metrics = {
            **metrics.authentication_metrics,
            **metrics.authorization_metrics,
            **metrics.data_access_metrics,
            **metrics.encryption_metrics,
            **metrics.vulnerability_metrics,
            **metrics.incident_metrics,
            **metrics.compliance_metrics,
            **metrics.performance_metrics,
            **metrics.availability_metrics
        }
        
        for metric_name, current_value in all_metrics.items():
            # Extract historical values
            historical_values = []
            for historical_metrics in recent_metrics:
                for attr_name in ['authentication_metrics', 'authorization_metrics', 
                                'data_access_metrics', 'encryption_metrics',
                                'vulnerability_metrics', 'incident_metrics',
                                'compliance_metrics', 'performance_metrics',
                                'availability_metrics']:
                    attr_dict = getattr(historical_metrics, attr_name, {})
                    if metric_name in attr_dict:
                        historical_values.append(attr_dict[metric_name])
                        break
            
            if len(historical_values) >= 5:
                trend = self._calculate_trend(historical_values)
                self.trends[metric_name] = trend
    
    def _calculate_trend(self, values: List[float]) -> SecurityTrend:
        """Calculate trend from historical values."""
        if len(values) < 2:
            return SecurityTrend(
                metric_id="",
                time_period="insufficient_data",
                trend_direction="stable",
                trend_strength=0.0,
                change_percentage=0.0,
                significance_level=0.0,
                anomalies_detected=0,
                predictions={}
            )
        
        # Calculate basic trend
        first_value = values[0]
        last_value = values[-1]
        change_percentage = ((last_value - first_value) / first_value * 100) if first_value != 0 else 0.0
        
        # Determine trend direction
        if abs(change_percentage) < 5:  # 5% threshold for stability
            trend_direction = "stable"
        elif change_percentage > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"
        
        # Calculate trend strength (correlation coefficient approximation)
        if len(values) >= 3:
            x = list(range(len(values)))
            mean_x = statistics.mean(x)
            mean_y = statistics.mean(values)
            
            numerator = sum((x[i] - mean_x) * (values[i] - mean_y) for i in range(len(values)))
            denominator_x = sum((x[i] - mean_x) ** 2 for i in range(len(values)))
            denominator_y = sum((values[i] - mean_y) ** 2 for i in range(len(values)))
            
            if denominator_x > 0 and denominator_y > 0:
                trend_strength = abs(numerator / (denominator_x * denominator_y) ** 0.5)
            else:
                trend_strength = 0.0
        else:
            trend_strength = 0.0
        
        # Detect anomalies (simple outlier detection)
        if len(values) >= 5:
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            anomalies = sum(1 for v in values if abs(v - mean_val) > 2 * std_val) if std_val > 0 else 0
        else:
            anomalies = 0
        
        return SecurityTrend(
            metric_id="",
            time_period="last_10_collections",
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            change_percentage=change_percentage,
            significance_level=trend_strength,
            anomalies_detected=anomalies,
            predictions={"next_value": last_value}  # Simplified prediction
        )
    
    def _generate_alert(self, 
                       alert_type: AlertType,
                       severity: MetricSeverity,
                       title: str,
                       description: str,
                       affected_metrics: List[str],
                       source_system: str):
        """Generate a security alert."""
        alert = SecurityAlert(
            alert_id=f"alert_{int(time.time())}_{len(self.alerts)}",
            alert_type=alert_type,
            severity=severity,
            title=title,
            description=description,
            affected_metrics=affected_metrics,
            detection_timestamp=datetime.now(),
            source_system=source_system,
            remediation_suggestions=self._get_remediation_suggestions(alert_type, affected_metrics)
        )
        
        self.alerts.append(alert)
        logger.warning(f"Security alert generated: {title}")
    
    def _get_remediation_suggestions(self, 
                                   alert_type: AlertType,
                                   affected_metrics: List[str]) -> List[str]:
        """Get remediation suggestions for alerts."""
        suggestions = []
        
        for metric in affected_metrics:
            if 'authentication' in metric:
                suggestions.extend([
                    "Review authentication logs for suspicious activity",
                    "Check multi-factor authentication coverage",
                    "Verify password policy compliance"
                ])
            elif 'encryption' in metric:
                suggestions.extend([
                    "Verify encryption configuration",
                    "Check certificate validity",
                    "Review key management procedures"
                ])
            elif 'vulnerability' in metric:
                suggestions.extend([
                    "Prioritize vulnerability remediation",
                    "Update vulnerability scanning schedule",
                    "Review patch management process"
                ])
            elif 'incident' in metric:
                suggestions.extend([
                    "Activate incident response procedures",
                    "Review security monitoring configuration",
                    "Escalate to security team"
                ])
        
        return list(set(suggestions))  # Remove duplicates
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics based on retention policy."""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        # Remove old metrics
        while self.metrics and self.metrics[0].collection_timestamp < cutoff_date:
            self.metrics.popleft()
        
        # Remove old alerts
        self.alerts = [
            alert for alert in self.alerts
            if alert.detection_timestamp > cutoff_date
        ]
    
    def get_security_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for security monitoring dashboard."""
        if not self.metrics:
            return {"error": "No metrics available"}
        
        latest_metrics = self.metrics[-1]
        
        # Calculate summary statistics
        active_alerts = [alert for alert in self.alerts if not alert.resolved]
        critical_alerts = [alert for alert in active_alerts if alert.severity == MetricSeverity.CRITICAL]
        
        # Get trending metrics
        trending_up = [name for name, trend in self.trends.items() 
                      if trend.trend_direction == "increasing" and trend.trend_strength > 0.5]
        trending_down = [name for name, trend in self.trends.items() 
                        if trend.trend_direction == "decreasing" and trend.trend_strength > 0.5]
        
        return {
            'overall_security_score': latest_metrics.overall_security_score,
            'collection_timestamp': latest_metrics.collection_timestamp.isoformat(),
            'alerts_summary': {
                'total_active': len(active_alerts),
                'critical': len(critical_alerts),
                'warning': len([a for a in active_alerts if a.severity == MetricSeverity.WARNING])
            },
            'key_metrics': {
                'authentication_success_rate': latest_metrics.authentication_metrics.get('authentication_success_rate', 0),
                'data_encryption_coverage': latest_metrics.encryption_metrics.get('data_encryption_coverage', 0),
                'compliance_score': latest_metrics.compliance_metrics.get('compliance_score', 0),
                'system_availability': latest_metrics.availability_metrics.get('system_availability', 0),
                'security_incident_count': latest_metrics.incident_metrics.get('security_incident_count', 0)
            },
            'trends': {
                'improving_metrics': len(trending_up),
                'declining_metrics': len(trending_down),
                'stable_metrics': len(self.trends) - len(trending_up) - len(trending_down)
            },
            'health_data_metrics': {
                'patient_data_access_rate': latest_metrics.data_access_metrics.get('patient_data_access_rate', 0),
                'health_record_encryption_coverage': latest_metrics.encryption_metrics.get('data_encryption_coverage', 0),
                'clinical_data_access_compliance': latest_metrics.data_access_metrics.get('clinical_data_access_compliance', 0),
                'consent_management_compliance': latest_metrics.compliance_metrics.get('consent_management_compliance', 0)
            },
            'recent_alerts': [
                {
                    'title': alert.title,
                    'severity': alert.severity.value,
                    'timestamp': alert.detection_timestamp.isoformat(),
                    'resolved': alert.resolved
                }
                for alert in sorted(self.alerts[-10:], key=lambda x: x.detection_timestamp, reverse=True)
            ]
        }
    
    def generate_security_report(self, 
                               time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive security metrics report.
        
        Args:
            time_range: Optional time range for report
            
        Returns:
            Dict[str, Any]: Comprehensive security report
        """
        if time_range:
            start_time, end_time = time_range
            relevant_metrics = [
                m for m in self.metrics
                if start_time <= m.collection_timestamp <= end_time
            ]
        else:
            relevant_metrics = list(self.metrics)
            start_time = relevant_metrics[0].collection_timestamp if relevant_metrics else datetime.now()
            end_time = relevant_metrics[-1].collection_timestamp if relevant_metrics else datetime.now()
        
        if not relevant_metrics:
            return {"error": "No metrics available for specified time range"}
        
        # Calculate averages
        avg_metrics = self._calculate_average_metrics(relevant_metrics)
        
        # Calculate trends
        trend_summary = self._calculate_trend_summary(relevant_metrics)
        
        # Alert statistics
        period_alerts = [
            alert for alert in self.alerts
            if start_time <= alert.detection_timestamp <= end_time
        ]
        
        return {
            'report_period': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_hours': (end_time - start_time).total_seconds() / 3600
            },
            'executive_summary': {
                'overall_security_score': avg_metrics.get('overall_security_score', 0),
                'total_metrics_collected': len(relevant_metrics),
                'security_incidents': sum(m.incident_metrics.get('security_incident_count', 0) for m in relevant_metrics),
                'compliance_status': 'compliant' if avg_metrics.get('compliance_score', 0) >= 0.85 else 'non_compliant'
            },
            'average_metrics': avg_metrics,
            'trend_analysis': trend_summary,
            'alert_statistics': {
                'total_alerts': len(period_alerts),
                'critical_alerts': len([a for a in period_alerts if a.severity == MetricSeverity.CRITICAL]),
                'resolved_alerts': len([a for a in period_alerts if a.resolved]),
                'avg_resolution_time': self._calculate_avg_resolution_time(period_alerts)
            },
            'recommendations': self._generate_security_recommendations(avg_metrics, trend_summary, period_alerts)
        }
    
    def _calculate_average_metrics(self, metrics_list: List[SecurityMetrics]) -> Dict[str, float]:
        """Calculate average values for all metrics."""
        if not metrics_list:
            return {}
        
        totals = defaultdict(list)
        
        for metrics in metrics_list:
            totals['overall_security_score'].append(metrics.overall_security_score)
            
            for attr_name in ['authentication_metrics', 'authorization_metrics', 
                            'data_access_metrics', 'encryption_metrics',
                            'vulnerability_metrics', 'incident_metrics',
                            'compliance_metrics', 'performance_metrics',
                            'availability_metrics']:
                attr_dict = getattr(metrics, attr_name, {})
                for key, value in attr_dict.items():
                    totals[key].append(value)
        
        return {key: statistics.mean(values) for key, values in totals.items()}
    
    def _calculate_trend_summary(self, metrics_list: List[SecurityMetrics]) -> Dict[str, Any]:
        """Calculate trend summary for metrics."""
        if len(metrics_list) < 3:
            return {"insufficient_data": True}
        
        improving_count = 0
        declining_count = 0
        stable_count = 0
        
        for trend in self.trends.values():
            if trend.trend_direction == "increasing":
                improving_count += 1
            elif trend.trend_direction == "decreasing":
                declining_count += 1
            else:
                stable_count += 1
        
        return {
            'total_metrics_analyzed': len(self.trends),
            'improving_metrics': improving_count,
            'declining_metrics': declining_count,
            'stable_metrics': stable_count,
            'high_confidence_trends': len([t for t in self.trends.values() if t.trend_strength > 0.7])
        }
    
    def _calculate_avg_resolution_time(self, alerts: List[SecurityAlert]) -> float:
        """Calculate average alert resolution time."""
        resolved_alerts = [a for a in alerts if a.resolved and a.resolution_timestamp]
        
        if not resolved_alerts:
            return 0.0
        
        resolution_times = [
            (alert.resolution_timestamp - alert.detection_timestamp).total_seconds() / 60
            for alert in resolved_alerts
        ]
        
        return statistics.mean(resolution_times)
    
    def _generate_security_recommendations(self, 
                                         avg_metrics: Dict[str, float],
                                         trend_summary: Dict[str, Any],
                                         alerts: List[SecurityAlert]) -> List[str]:
        """Generate security recommendations based on analysis."""
        recommendations = []
        
        # Check critical metrics
        if avg_metrics.get('compliance_score', 0) < 0.85:
            recommendations.append("Improve compliance posture - score below 85% threshold")
        
        if avg_metrics.get('security_incident_count', 0) > 1:
            recommendations.append("Review incident response procedures - elevated incident count")
        
        if avg_metrics.get('data_encryption_coverage', 0) < 0.95:
            recommendations.append("Increase data encryption coverage to meet 95% target")
        
        # Check trends
        declining_count = trend_summary.get('declining_metrics', 0)
        if declining_count > 5:
            recommendations.append(f"Address {declining_count} declining security metrics")
        
        # Check alerts
        critical_alerts = len([a for a in alerts if a.severity == MetricSeverity.CRITICAL])
        if critical_alerts > 0:
            recommendations.append(f"Investigate and resolve {critical_alerts} critical alerts")
        
        return recommendations