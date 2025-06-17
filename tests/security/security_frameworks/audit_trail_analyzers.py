"""
Audit Trail Analysis Framework - Phase 5.6

Comprehensive audit trail analysis utilities for the Australian Health Analytics platform.
Provides robust analysis of audit logs, data lineage tracking, and compliance monitoring
to ensure complete accountability for all health data operations.

Key Features:
- Complete audit trail validation for all data operations
- Data lineage and provenance tracking security
- Compliance monitoring and alerting
- Audit log integrity and tamper detection
- Australian health data audit compliance
"""

import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_CREATION = "data_creation"
    DATA_DELETION = "data_deletion"
    USER_AUTHENTICATION = "user_authentication"
    PERMISSION_CHANGE = "permission_change"
    SYSTEM_CONFIGURATION = "system_configuration"
    EXPORT_OPERATION = "export_operation"
    BACKUP_OPERATION = "backup_operation"
    SECURITY_EVENT = "security_event"


class AuditSeverity(Enum):
    """Audit event severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY_ALERT = "security_alert"


class ComplianceStandard(Enum):
    """Compliance standards for audit requirements."""
    AUSTRALIAN_PRIVACY_PRINCIPLES = "app"
    HEALTH_RECORDS_ACT = "hra"
    PRIVACY_ACT_1988 = "privacy_act_1988"
    HEALTH_INSURANCE_ACT = "health_insurance_act"
    ISO_27001 = "iso_27001"
    NIST_FRAMEWORK = "nist_framework"


@dataclass
class AuditLogEntry:
    """Individual audit log entry."""
    timestamp: datetime
    event_type: AuditEventType
    event_id: str
    user_id: str
    session_id: str
    ip_address: str
    user_agent: str
    resource: str
    action: str
    outcome: str
    severity: AuditSeverity
    details: Dict[str, Any]
    data_classification: str
    checksum: str = field(default="")
    
    def __post_init__(self):
        """Calculate checksum after initialization."""
        if not self.checksum:
            self.checksum = self.calculate_checksum()
    
    def calculate_checksum(self) -> str:
        """Calculate integrity checksum for the audit entry."""
        data_string = (
            f"{self.timestamp.isoformat()}{self.event_type.value}{self.event_id}"
            f"{self.user_id}{self.resource}{self.action}{self.outcome}"
        )
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify the integrity of this audit entry."""
        expected_checksum = self.calculate_checksum()
        return self.checksum == expected_checksum


@dataclass
class DataLineageEntry:
    """Data lineage tracking entry."""
    data_id: str
    operation_id: str
    operation_type: str
    timestamp: datetime
    user_id: str
    source_data: List[str]
    transformation_applied: str
    output_data: List[str]
    validation_status: str
    lineage_checksum: str = field(default="")
    
    def __post_init__(self):
        """Calculate lineage checksum after initialization."""
        if not self.lineage_checksum:
            self.lineage_checksum = self.calculate_lineage_checksum()
    
    def calculate_lineage_checksum(self) -> str:
        """Calculate checksum for lineage integrity."""
        lineage_string = (
            f"{self.data_id}{self.operation_id}{self.operation_type}"
            f"{self.timestamp.isoformat()}{self.user_id}"
            f"{'|'.join(sorted(self.source_data))}{'|'.join(sorted(self.output_data))}"
        )
        return hashlib.sha256(lineage_string.encode()).hexdigest()


@dataclass
class ComplianceViolation:
    """Compliance violation detected in audit trail."""
    violation_id: str
    violation_type: str
    compliance_standard: ComplianceStandard
    severity: AuditSeverity
    description: str
    affected_entries: List[str]
    detection_timestamp: datetime
    remediation_required: bool
    remediation_suggestions: List[str]


@dataclass
class AuditAnalysisResult:
    """Result of audit trail analysis."""
    analysis_id: str
    analysis_timestamp: datetime
    total_entries_analyzed: int
    integrity_violations: List[str]
    compliance_violations: List[ComplianceViolation]
    security_anomalies: List[Dict[str, Any]]
    coverage_analysis: Dict[str, float]
    recommendations: List[str]
    overall_compliance_score: float


class AuditTrailAnalyzer:
    """
    Comprehensive audit trail analyzer for Australian health data systems.
    Validates audit completeness, integrity, and compliance with Australian regulations.
    """
    
    def __init__(self, 
                 audit_log_path: Optional[Path] = None,
                 compliance_config_path: Optional[Path] = None):
        """
        Initialize audit trail analyzer.
        
        Args:
            audit_log_path: Path to audit log files
            compliance_config_path: Path to compliance configuration
        """
        self.audit_log_path = audit_log_path
        self.compliance_config_path = compliance_config_path
        
        # Audit log storage
        self.audit_entries: List[AuditLogEntry] = []
        self.lineage_entries: List[DataLineageEntry] = []
        
        # Compliance requirements
        self.compliance_requirements = self._load_compliance_requirements()
        
        # Analysis configuration
        self.analysis_config = {
            'integrity_check_enabled': True,
            'compliance_validation_enabled': True,
            'anomaly_detection_enabled': True,
            'retention_policy_days': 2557,  # 7 years for health data
            'critical_events_retention_days': 3652,  # 10 years for critical events
            'suspicious_activity_threshold': 5,
            'data_access_frequency_threshold': 100
        }
        
        # Australian health data specific settings
        self.health_data_audit_requirements = {
            'mandatory_events': [
                AuditEventType.DATA_ACCESS,
                AuditEventType.DATA_MODIFICATION,
                AuditEventType.DATA_DELETION,
                AuditEventType.EXPORT_OPERATION
            ],
            'privacy_sensitive_actions': [
                'patient_data_access',
                'medical_record_view',
                'prescription_access',
                'genetic_data_access',
                'mental_health_data_access'
            ],
            'required_audit_fields': [
                'timestamp', 'user_id', 'resource', 'action', 
                'outcome', 'data_classification', 'ip_address'
            ]
        }
        
        logger.info("Audit trail analyzer initialized")
    
    def _load_compliance_requirements(self) -> Dict[str, Any]:
        """Load compliance requirements configuration."""
        default_requirements = {
            ComplianceStandard.AUSTRALIAN_PRIVACY_PRINCIPLES: {
                'required_audit_events': [
                    'personal_information_collection',
                    'personal_information_use',
                    'personal_information_disclosure',
                    'data_quality_checks',
                    'security_safeguards',
                    'access_requests',
                    'correction_requests'
                ],
                'retention_requirements': {
                    'minimum_days': 2557,  # 7 years
                    'critical_events_days': 3652  # 10 years
                },
                'integrity_requirements': {
                    'tamper_protection': True,
                    'digital_signatures': True,
                    'backup_verification': True
                }
            },
            ComplianceStandard.HEALTH_RECORDS_ACT: {
                'required_audit_events': [
                    'health_record_access',
                    'health_record_modification',
                    'health_record_sharing',
                    'consent_verification',
                    'emergency_access'
                ],
                'access_logging': {
                    'log_all_access': True,
                    'include_read_operations': True,
                    'track_data_lineage': True
                }
            },
            ComplianceStandard.PRIVACY_ACT_1988: {
                'notification_requirements': {
                    'data_breach_reporting': True,
                    'unauthorised_access_reporting': True,
                    'system_compromise_reporting': True
                },
                'audit_trail_requirements': {
                    'complete_audit_trail': True,
                    'immutable_logs': True,
                    'real_time_monitoring': True
                }
            }
        }
        
        return default_requirements
    
    def add_audit_entry(self, entry: AuditLogEntry) -> bool:
        """
        Add an audit entry to the analyzer.
        
        Args:
            entry: Audit log entry to add
            
        Returns:
            bool: Success status
        """
        try:
            # Verify entry integrity
            if not entry.verify_integrity():
                logger.warning(f"Audit entry integrity check failed: {entry.event_id}")
                return False
            
            # Validate required fields
            if not self._validate_audit_entry(entry):
                logger.warning(f"Audit entry validation failed: {entry.event_id}")
                return False
            
            self.audit_entries.append(entry)
            logger.debug(f"Added audit entry: {entry.event_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add audit entry: {e}")
            return False
    
    def add_lineage_entry(self, entry: DataLineageEntry) -> bool:
        """
        Add a data lineage entry to the analyzer.
        
        Args:
            entry: Data lineage entry to add
            
        Returns:
            bool: Success status
        """
        try:
            self.lineage_entries.append(entry)
            logger.debug(f"Added lineage entry: {entry.operation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add lineage entry: {e}")
            return False
    
    def analyze_audit_trail_completeness(self, 
                                       time_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """
        Analyze audit trail completeness for a given time range.
        
        Args:
            time_range: Tuple of (start_time, end_time)
            
        Returns:
            Dict[str, Any]: Completeness analysis results
        """
        start_time, end_time = time_range
        
        # Filter entries within time range
        relevant_entries = [
            entry for entry in self.audit_entries
            if start_time <= entry.timestamp <= end_time
        ]
        
        # Analyze coverage by event type
        event_type_coverage = defaultdict(int)
        for entry in relevant_entries:
            event_type_coverage[entry.event_type] += 1
        
        # Check for mandatory events
        mandatory_events = self.health_data_audit_requirements['mandatory_events']
        missing_mandatory_events = [
            event_type for event_type in mandatory_events
            if event_type not in event_type_coverage
        ]
        
        # Analyze coverage by user activity
        user_activity = defaultdict(int)
        for entry in relevant_entries:
            user_activity[entry.user_id] += 1
        
        # Check for suspicious gaps in logging
        time_gaps = self._detect_logging_gaps(relevant_entries, start_time, end_time)
        
        # Calculate completeness score
        completeness_score = self._calculate_completeness_score(
            event_type_coverage, missing_mandatory_events, time_gaps
        )
        
        return {
            'analysis_period': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_hours': (end_time - start_time).total_seconds() / 3600
            },
            'entries_analyzed': len(relevant_entries),
            'event_type_coverage': dict(event_type_coverage),
            'missing_mandatory_events': [event.value for event in missing_mandatory_events],
            'user_activity_summary': {
                'total_users': len(user_activity),
                'most_active_users': sorted(
                    user_activity.items(), key=lambda x: x[1], reverse=True
                )[:10]
            },
            'time_gaps_detected': len(time_gaps),
            'significant_gaps': [gap for gap in time_gaps if gap['duration_minutes'] > 60],
            'completeness_score': completeness_score,
            'recommendations': self._generate_completeness_recommendations(
                missing_mandatory_events, time_gaps, completeness_score
            )
        }
    
    def analyze_audit_trail_integrity(self) -> Dict[str, Any]:
        """
        Analyze audit trail integrity and detect tampering.
        
        Returns:
            Dict[str, Any]: Integrity analysis results
        """
        integrity_violations = []
        checksum_failures = []
        sequence_anomalies = []
        
        # Verify individual entry integrity
        for entry in self.audit_entries:
            if not entry.verify_integrity():
                integrity_violations.append({
                    'entry_id': entry.event_id,
                    'violation_type': 'checksum_mismatch',
                    'timestamp': entry.timestamp.isoformat(),
                    'severity': 'critical'
                })
                checksum_failures.append(entry.event_id)
        
        # Check for sequence anomalies
        sorted_entries = sorted(self.audit_entries, key=lambda x: x.timestamp)
        for i in range(1, len(sorted_entries)):
            prev_entry = sorted_entries[i-1]
            curr_entry = sorted_entries[i]
            
            # Check for timestamp anomalies
            time_diff = (curr_entry.timestamp - prev_entry.timestamp).total_seconds()
            if time_diff < 0:
                sequence_anomalies.append({
                    'anomaly_type': 'timestamp_reversal',
                    'entry_ids': [prev_entry.event_id, curr_entry.event_id],
                    'time_difference_seconds': time_diff
                })
        
        # Analyze patterns for potential tampering
        tampering_indicators = self._detect_tampering_patterns()
        
        # Calculate integrity score
        total_entries = len(self.audit_entries)
        integrity_score = 1.0 - (len(integrity_violations) / total_entries) if total_entries > 0 else 0.0
        
        return {
            'total_entries_checked': total_entries,
            'integrity_violations': integrity_violations,
            'checksum_failures': len(checksum_failures),
            'sequence_anomalies': sequence_anomalies,
            'tampering_indicators': tampering_indicators,
            'integrity_score': integrity_score,
            'recommendations': self._generate_integrity_recommendations(
                integrity_violations, sequence_anomalies, tampering_indicators
            )
        }
    
    def analyze_compliance_adherence(self, 
                                   standards: List[ComplianceStandard]) -> Dict[str, Any]:
        """
        Analyze adherence to compliance standards.
        
        Args:
            standards: List of compliance standards to check
            
        Returns:
            Dict[str, Any]: Compliance analysis results
        """
        compliance_results = {}
        overall_violations = []
        
        for standard in standards:
            standard_requirements = self.compliance_requirements.get(standard, {})
            violations = []
            
            # Check required audit events
            required_events = standard_requirements.get('required_audit_events', [])
            logged_actions = set(entry.action for entry in self.audit_entries)
            
            for required_event in required_events:
                if required_event not in logged_actions:
                    violation = ComplianceViolation(
                        violation_id=f"{standard.value}_{required_event}_missing",
                        violation_type="missing_required_event",
                        compliance_standard=standard,
                        severity=AuditSeverity.ERROR,
                        description=f"Required audit event '{required_event}' not found in logs",
                        affected_entries=[],
                        detection_timestamp=datetime.now(),
                        remediation_required=True,
                        remediation_suggestions=[
                            f"Ensure all '{required_event}' actions are properly logged",
                            "Review audit configuration for completeness",
                            "Implement mandatory logging for this event type"
                        ]
                    )
                    violations.append(violation)
                    overall_violations.append(violation)
            
            # Check retention requirements
            retention_reqs = standard_requirements.get('retention_requirements', {})
            if retention_reqs:
                retention_violations = self._check_retention_compliance(retention_reqs)
                violations.extend(retention_violations)
                overall_violations.extend(retention_violations)
            
            # Check integrity requirements
            integrity_reqs = standard_requirements.get('integrity_requirements', {})
            if integrity_reqs:
                integrity_violations = self._check_integrity_compliance(integrity_reqs)
                violations.extend(integrity_violations)
                overall_violations.extend(integrity_violations)
            
            # Calculate compliance score for this standard
            total_requirements = (
                len(required_events) + 
                len(retention_reqs) + 
                len(integrity_reqs)
            )
            compliance_score = max(0.0, 1.0 - (len(violations) / max(total_requirements, 1)))
            
            compliance_results[standard.value] = {
                'compliance_score': compliance_score,
                'violations': [
                    {
                        'violation_id': v.violation_id,
                        'type': v.violation_type,
                        'severity': v.severity.value,
                        'description': v.description,
                        'remediation_required': v.remediation_required
                    }
                    for v in violations
                ],
                'requirements_checked': total_requirements,
                'violations_found': len(violations)
            }
        
        # Calculate overall compliance score
        if compliance_results:
            overall_score = sum(
                result['compliance_score'] for result in compliance_results.values()
            ) / len(compliance_results)
        else:
            overall_score = 0.0
        
        return {
            'overall_compliance_score': overall_score,
            'standards_analyzed': [standard.value for standard in standards],
            'compliance_by_standard': compliance_results,
            'total_violations': len(overall_violations),
            'critical_violations': [
                v for v in overall_violations 
                if v.severity in [AuditSeverity.CRITICAL, AuditSeverity.SECURITY_ALERT]
            ],
            'recommendations': self._generate_compliance_recommendations(overall_violations)
        }
    
    def analyze_data_lineage_integrity(self) -> Dict[str, Any]:
        """
        Analyze data lineage integrity and completeness.
        
        Returns:
            Dict[str, Any]: Data lineage analysis results
        """
        lineage_violations = []
        orphaned_data = []
        broken_chains = []
        
        # Build lineage graph
        lineage_graph = self._build_lineage_graph()
        
        # Check for orphaned data (data without lineage)
        all_data_ids = set()
        tracked_data_ids = set()
        
        for entry in self.lineage_entries:
            all_data_ids.update(entry.source_data)
            all_data_ids.update(entry.output_data)
            tracked_data_ids.add(entry.data_id)
        
        orphaned_data = list(all_data_ids - tracked_data_ids)
        
        # Check for broken lineage chains
        for data_id in tracked_data_ids:
            if not self._validate_lineage_chain(data_id, lineage_graph):
                broken_chains.append(data_id)
        
        # Verify lineage entry integrity
        for entry in self.lineage_entries:
            expected_checksum = entry.calculate_lineage_checksum()
            if entry.lineage_checksum != expected_checksum:
                lineage_violations.append({
                    'operation_id': entry.operation_id,
                    'violation_type': 'checksum_mismatch',
                    'data_id': entry.data_id,
                    'timestamp': entry.timestamp.isoformat()
                })
        
        # Calculate lineage integrity score
        total_lineage_entries = len(self.lineage_entries)
        lineage_score = 1.0 - (
            len(lineage_violations) / max(total_lineage_entries, 1)
        )
        
        return {
            'total_lineage_entries': total_lineage_entries,
            'unique_data_items': len(all_data_ids),
            'tracked_data_items': len(tracked_data_ids),
            'orphaned_data_count': len(orphaned_data),
            'broken_chains_count': len(broken_chains),
            'lineage_violations': lineage_violations,
            'lineage_integrity_score': lineage_score,
            'lineage_coverage_percentage': (
                len(tracked_data_ids) / max(len(all_data_ids), 1) * 100
            ),
            'recommendations': self._generate_lineage_recommendations(
                orphaned_data, broken_chains, lineage_violations
            )
        }
    
    def perform_comprehensive_audit_analysis(self, 
                                           time_range: Optional[Tuple[datetime, datetime]] = None,
                                           compliance_standards: Optional[List[ComplianceStandard]] = None) -> AuditAnalysisResult:
        """
        Perform comprehensive audit trail analysis.
        
        Args:
            time_range: Optional time range for analysis
            compliance_standards: Optional list of compliance standards to check
            
        Returns:
            AuditAnalysisResult: Comprehensive analysis results
        """
        analysis_start = time.time()
        
        # Set default parameters
        if time_range is None:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)  # Last 30 days
            time_range = (start_time, end_time)
        
        if compliance_standards is None:
            compliance_standards = [
                ComplianceStandard.AUSTRALIAN_PRIVACY_PRINCIPLES,
                ComplianceStandard.HEALTH_RECORDS_ACT,
                ComplianceStandard.PRIVACY_ACT_1988
            ]
        
        logger.info("Starting comprehensive audit analysis")
        
        # Perform individual analyses
        completeness_analysis = self.analyze_audit_trail_completeness(time_range)
        integrity_analysis = self.analyze_audit_trail_integrity()
        compliance_analysis = self.analyze_compliance_adherence(compliance_standards)
        lineage_analysis = self.analyze_data_lineage_integrity()
        
        # Detect security anomalies
        security_anomalies = self._detect_security_anomalies()
        
        # Calculate overall compliance score
        overall_score = (
            completeness_analysis['completeness_score'] * 0.25 +
            integrity_analysis['integrity_score'] * 0.25 +
            compliance_analysis['overall_compliance_score'] * 0.30 +
            lineage_analysis['lineage_integrity_score'] * 0.20
        )
        
        # Generate comprehensive recommendations
        all_recommendations = (
            completeness_analysis.get('recommendations', []) +
            integrity_analysis.get('recommendations', []) +
            compliance_analysis.get('recommendations', []) +
            lineage_analysis.get('recommendations', [])
        )
        
        # Remove duplicates and prioritize
        unique_recommendations = list(set(all_recommendations))
        prioritized_recommendations = self._prioritize_recommendations(unique_recommendations)
        
        analysis_duration = time.time() - analysis_start
        
        result = AuditAnalysisResult(
            analysis_id=f"audit_analysis_{int(time.time())}",
            analysis_timestamp=datetime.now(),
            total_entries_analyzed=len(self.audit_entries),
            integrity_violations=integrity_analysis.get('integrity_violations', []),
            compliance_violations=compliance_analysis.get('critical_violations', []),
            security_anomalies=security_anomalies,
            coverage_analysis={
                'completeness_score': completeness_analysis['completeness_score'],
                'integrity_score': integrity_analysis['integrity_score'],
                'compliance_score': compliance_analysis['overall_compliance_score'],
                'lineage_score': lineage_analysis['lineage_integrity_score']
            },
            recommendations=prioritized_recommendations,
            overall_compliance_score=overall_score
        )
        
        logger.info(f"Comprehensive audit analysis completed in {analysis_duration:.2f} seconds")
        logger.info(f"Overall compliance score: {overall_score:.2f}")
        
        return result
    
    def _validate_audit_entry(self, entry: AuditLogEntry) -> bool:
        """Validate audit entry against requirements."""
        required_fields = self.health_data_audit_requirements['required_audit_fields']
        
        # Check required fields
        for field in required_fields:
            if not hasattr(entry, field) or getattr(entry, field) is None:
                return False
        
        # Validate timestamp
        if entry.timestamp > datetime.now():
            return False
        
        # Validate user_id format
        if not entry.user_id or len(entry.user_id) < 3:
            return False
        
        return True
    
    def _detect_logging_gaps(self, 
                           entries: List[AuditLogEntry], 
                           start_time: datetime, 
                           end_time: datetime) -> List[Dict[str, Any]]:
        """Detect gaps in audit logging."""
        gaps = []
        
        if not entries:
            return [{
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_minutes': (end_time - start_time).total_seconds() / 60,
                'type': 'complete_gap'
            }]
        
        sorted_entries = sorted(entries, key=lambda x: x.timestamp)
        
        # Check gap before first entry
        if sorted_entries[0].timestamp > start_time + timedelta(minutes=30):
            gaps.append({
                'start_time': start_time.isoformat(),
                'end_time': sorted_entries[0].timestamp.isoformat(),
                'duration_minutes': (sorted_entries[0].timestamp - start_time).total_seconds() / 60,
                'type': 'initial_gap'
            })
        
        # Check gaps between entries
        for i in range(1, len(sorted_entries)):
            time_diff = sorted_entries[i].timestamp - sorted_entries[i-1].timestamp
            if time_diff > timedelta(hours=2):  # Gap longer than 2 hours
                gaps.append({
                    'start_time': sorted_entries[i-1].timestamp.isoformat(),
                    'end_time': sorted_entries[i].timestamp.isoformat(),
                    'duration_minutes': time_diff.total_seconds() / 60,
                    'type': 'intermediate_gap'
                })
        
        # Check gap after last entry
        if sorted_entries[-1].timestamp < end_time - timedelta(minutes=30):
            gaps.append({
                'start_time': sorted_entries[-1].timestamp.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_minutes': (end_time - sorted_entries[-1].timestamp).total_seconds() / 60,
                'type': 'final_gap'
            })
        
        return gaps
    
    def _calculate_completeness_score(self, 
                                    event_coverage: Dict[AuditEventType, int],
                                    missing_events: List[AuditEventType],
                                    time_gaps: List[Dict[str, Any]]) -> float:
        """Calculate audit trail completeness score."""
        base_score = 1.0
        
        # Penalty for missing mandatory events
        mandatory_events = self.health_data_audit_requirements['mandatory_events']
        missing_mandatory = len(missing_events)
        total_mandatory = len(mandatory_events)
        
        if total_mandatory > 0:
            base_score -= (missing_mandatory / total_mandatory) * 0.4
        
        # Penalty for significant time gaps
        significant_gaps = len([gap for gap in time_gaps if gap['duration_minutes'] > 60])
        base_score -= min(significant_gaps * 0.1, 0.3)
        
        # Penalty for low event coverage
        if len(event_coverage) < 3:  # Minimum expected event types
            base_score -= 0.2
        
        return max(0.0, base_score)
    
    def _detect_tampering_patterns(self) -> List[Dict[str, Any]]:
        """Detect patterns that might indicate log tampering."""
        indicators = []
        
        # Check for unusual patterns in timestamps
        timestamps = [entry.timestamp for entry in self.audit_entries]
        if len(timestamps) > 1:
            time_diffs = [
                (timestamps[i] - timestamps[i-1]).total_seconds()
                for i in range(1, len(timestamps))
            ]
            
            # Look for suspiciously regular intervals
            if len(set(time_diffs)) < len(time_diffs) * 0.5:  # Too many identical intervals
                indicators.append({
                    'type': 'suspicious_timing_pattern',
                    'description': 'Unusually regular time intervals detected',
                    'confidence': 0.7
                })
        
        # Check for missing event IDs or patterns
        event_ids = [entry.event_id for entry in self.audit_entries]
        if len(set(event_ids)) != len(event_ids):
            indicators.append({
                'type': 'duplicate_event_ids',
                'description': 'Duplicate event IDs detected',
                'confidence': 0.9
            })
        
        # Check for inconsistent user patterns
        user_activity = defaultdict(list)
        for entry in self.audit_entries:
            user_activity[entry.user_id].append(entry.timestamp)
        
        for user_id, timestamps in user_activity.items():
            # Check for impossible activity patterns
            sorted_times = sorted(timestamps)
            for i in range(1, len(sorted_times)):
                time_diff = (sorted_times[i] - sorted_times[i-1]).total_seconds()
                if time_diff < 1:  # Actions less than 1 second apart
                    indicators.append({
                        'type': 'impossible_user_activity',
                        'description': f'Impossible activity pattern for user {user_id}',
                        'confidence': 0.8,
                        'user_id': user_id
                    })
                    break
        
        return indicators
    
    def _check_retention_compliance(self, 
                                  retention_requirements: Dict[str, Any]) -> List[ComplianceViolation]:
        """Check compliance with data retention requirements."""
        violations = []
        
        min_retention_days = retention_requirements.get('minimum_days', 0)
        critical_retention_days = retention_requirements.get('critical_events_days', 0)
        
        cutoff_date = datetime.now() - timedelta(days=min_retention_days)
        critical_cutoff_date = datetime.now() - timedelta(days=critical_retention_days)
        
        # Check for entries that should be retained but are missing
        # This is a simplified check - in practice, you'd compare against expected data
        old_entries = [
            entry for entry in self.audit_entries 
            if entry.timestamp < cutoff_date
        ]
        
        if len(old_entries) == 0 and min_retention_days > 0:
            violations.append(ComplianceViolation(
                violation_id="retention_gap_detected",
                violation_type="insufficient_retention",
                compliance_standard=ComplianceStandard.AUSTRALIAN_PRIVACY_PRINCIPLES,
                severity=AuditSeverity.WARNING,
                description="Insufficient audit log retention detected",
                affected_entries=[],
                detection_timestamp=datetime.now(),
                remediation_required=True,
                remediation_suggestions=[
                    "Review audit log retention policy",
                    "Ensure logs are properly archived",
                    "Implement automated retention management"
                ]
            ))
        
        return violations
    
    def _check_integrity_compliance(self, 
                                  integrity_requirements: Dict[str, Any]) -> List[ComplianceViolation]:
        """Check compliance with integrity requirements."""
        violations = []
        
        # Check tamper protection
        if integrity_requirements.get('tamper_protection', False):
            tampering_indicators = self._detect_tampering_patterns()
            if tampering_indicators:
                violations.append(ComplianceViolation(
                    violation_id="tampering_detected",
                    violation_type="integrity_compromise",
                    compliance_standard=ComplianceStandard.AUSTRALIAN_PRIVACY_PRINCIPLES,
                    severity=AuditSeverity.CRITICAL,
                    description="Potential log tampering detected",
                    affected_entries=[],
                    detection_timestamp=datetime.now(),
                    remediation_required=True,
                    remediation_suggestions=[
                        "Investigate tampering indicators immediately",
                        "Review log integrity mechanisms",
                        "Implement stronger tamper protection"
                    ]
                ))
        
        return violations
    
    def _build_lineage_graph(self) -> Dict[str, List[str]]:
        """Build data lineage graph for analysis."""
        graph = defaultdict(list)
        
        for entry in self.lineage_entries:
            for source in entry.source_data:
                graph[source].append(entry.data_id)
        
        return dict(graph)
    
    def _validate_lineage_chain(self, data_id: str, lineage_graph: Dict[str, List[str]]) -> bool:
        """Validate completeness of a data lineage chain."""
        # Simplified validation - check if data has proper lineage connections
        return data_id in lineage_graph or any(
            data_id in outputs for outputs in lineage_graph.values()
        )
    
    def _detect_security_anomalies(self) -> List[Dict[str, Any]]:
        """Detect security anomalies in audit logs."""
        anomalies = []
        
        # Detect unusual access patterns
        user_access_patterns = defaultdict(list)
        for entry in self.audit_entries:
            if entry.event_type == AuditEventType.DATA_ACCESS:
                user_access_patterns[entry.user_id].append(entry)
        
        for user_id, accesses in user_access_patterns.items():
            if len(accesses) > self.analysis_config['data_access_frequency_threshold']:
                anomalies.append({
                    'type': 'excessive_data_access',
                    'user_id': user_id,
                    'access_count': len(accesses),
                    'severity': 'medium',
                    'description': f'User {user_id} accessed data {len(accesses)} times'
                })
        
        # Detect failed authentication clusters
        failed_auth_events = [
            entry for entry in self.audit_entries
            if entry.event_type == AuditEventType.USER_AUTHENTICATION and entry.outcome == 'failed'
        ]
        
        if len(failed_auth_events) > self.analysis_config['suspicious_activity_threshold']:
            anomalies.append({
                'type': 'authentication_failure_cluster',
                'event_count': len(failed_auth_events),
                'severity': 'high',
                'description': f'{len(failed_auth_events)} failed authentication attempts detected'
            })
        
        # Detect after-hours access
        after_hours_access = [
            entry for entry in self.audit_entries
            if entry.timestamp.hour < 6 or entry.timestamp.hour > 22
        ]
        
        if len(after_hours_access) > 10:  # Threshold for concern
            anomalies.append({
                'type': 'after_hours_access',
                'event_count': len(after_hours_access),
                'severity': 'medium',
                'description': f'{len(after_hours_access)} after-hours access events detected'
            })
        
        return anomalies
    
    def _generate_completeness_recommendations(self, 
                                             missing_events: List[AuditEventType],
                                             time_gaps: List[Dict[str, Any]],
                                             score: float) -> List[str]:
        """Generate recommendations for audit completeness."""
        recommendations = []
        
        if missing_events:
            recommendations.append(
                f"Implement logging for missing mandatory events: {[e.value for e in missing_events]}"
            )
        
        if time_gaps:
            significant_gaps = [gap for gap in time_gaps if gap['duration_minutes'] > 60]
            if significant_gaps:
                recommendations.append(
                    f"Investigate {len(significant_gaps)} significant logging gaps"
                )
        
        if score < 0.8:
            recommendations.append("Audit trail completeness is below acceptable threshold")
        
        return recommendations
    
    def _generate_integrity_recommendations(self, 
                                          violations: List[Dict[str, Any]],
                                          anomalies: List[Dict[str, Any]],
                                          tampering: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for audit integrity."""
        recommendations = []
        
        if violations:
            recommendations.append("Address audit log integrity violations immediately")
        
        if anomalies:
            recommendations.append("Investigate sequence anomalies in audit logs")
        
        if tampering:
            recommendations.append("Implement stronger tamper protection mechanisms")
        
        return recommendations
    
    def _generate_compliance_recommendations(self, 
                                           violations: List[ComplianceViolation]) -> List[str]:
        """Generate recommendations for compliance adherence."""
        recommendations = []
        
        critical_violations = [v for v in violations if v.severity == AuditSeverity.CRITICAL]
        if critical_violations:
            recommendations.append("Address critical compliance violations immediately")
        
        missing_events = [v for v in violations if v.violation_type == "missing_required_event"]
        if missing_events:
            recommendations.append("Implement logging for all required compliance events")
        
        return recommendations
    
    def _generate_lineage_recommendations(self, 
                                        orphaned_data: List[str],
                                        broken_chains: List[str],
                                        violations: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for data lineage."""
        recommendations = []
        
        if orphaned_data:
            recommendations.append(f"Establish lineage tracking for {len(orphaned_data)} orphaned data items")
        
        if broken_chains:
            recommendations.append(f"Repair {len(broken_chains)} broken lineage chains")
        
        if violations:
            recommendations.append("Address data lineage integrity violations")
        
        return recommendations
    
    def _prioritize_recommendations(self, recommendations: List[str]) -> List[str]:
        """Prioritize recommendations by importance."""
        priority_keywords = {
            'critical': 10,
            'immediately': 9,
            'security': 8,
            'compliance': 7,
            'integrity': 6,
            'missing': 5
        }
        
        def get_priority(recommendation: str) -> int:
            score = 0
            for keyword, weight in priority_keywords.items():
                if keyword in recommendation.lower():
                    score += weight
            return score
        
        return sorted(recommendations, key=get_priority, reverse=True)