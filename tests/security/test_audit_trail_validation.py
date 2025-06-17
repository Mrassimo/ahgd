"""
Audit Trail Validation Testing

Comprehensive testing suite for audit trail validation including:
- Complete audit trail validation for all data operations
- Data lineage and provenance tracking security
- Compliance monitoring and alerting testing
- Security incident detection and response testing
- Regulatory compliance reporting validation
- Audit log integrity and tamper detection

This test suite ensures comprehensive audit trail coverage for regulatory
compliance and security monitoring of the Australian Health Analytics platform.
"""

import json
import pytest
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Union, Any
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass, field
from enum import Enum
import logging
import hmac

import polars as pl
import numpy as np
from loguru import logger


class AuditEventType(Enum):
    """Types of audit events."""
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_EXPORT = "data_export"
    DATA_DELETION = "data_deletion"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    PERMISSION_CHANGE = "permission_change"
    SYSTEM_CONFIGURATION = "system_configuration"
    PRIVACY_VIOLATION = "privacy_violation"
    SECURITY_INCIDENT = "security_incident"
    COMPLIANCE_CHECK = "compliance_check"
    DATA_PROCESSING = "data_processing"


class AuditLevel(Enum):
    """Audit logging levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    DEBUG = "debug"


class ComplianceFramework(Enum):
    """Compliance frameworks."""
    PRIVACY_ACT_1988 = "privacy_act_1988"
    HEALTH_RECORDS_ACT = "health_records_act"
    NOTIFIABLE_DATA_BREACHES = "notifiable_data_breaches"
    ISO_27001 = "iso_27001"
    NIST_CYBERSECURITY = "nist_cybersecurity"
    AUSTRALIAN_GOVERNMENT_ISM = "australian_government_ism"


@dataclass
class AuditRecord:
    """Individual audit record."""
    audit_id: str
    timestamp: str
    event_type: AuditEventType
    level: AuditLevel
    user_id: Optional[str]
    session_id: Optional[str]
    resource: str
    action: str
    outcome: str  # "success", "failure", "warning"
    ip_address: str
    user_agent: str
    data_elements: List[str]
    privacy_classification: str
    compliance_frameworks: List[ComplianceFramework]
    checksum: str
    details: Dict[str, Any]
    correlation_id: Optional[str] = None


@dataclass
class DataLineageRecord:
    """Data lineage tracking record."""
    lineage_id: str
    source_dataset: str
    target_dataset: str
    transformation_type: str
    transformation_timestamp: str
    processor_id: str
    data_elements_affected: List[str]
    privacy_impact: str
    compliance_notes: str
    parent_lineage_ids: List[str]
    child_lineage_ids: List[str]
    quality_metrics: Dict[str, float]
    provenance_chain: List[Dict[str, Any]]


@dataclass
class ComplianceMonitoringRule:
    """Compliance monitoring rule."""
    rule_id: str
    name: str
    description: str
    compliance_framework: ComplianceFramework
    event_types: List[AuditEventType]
    conditions: Dict[str, Any]
    alert_threshold: int
    time_window_minutes: int
    severity: str
    notification_recipients: List[str]
    automated_response: Optional[str] = None


@dataclass
class AuditTrailViolation:
    """Audit trail validation violation."""
    violation_type: str
    severity: str  # "critical", "high", "medium", "low"
    description: str
    affected_records: List[str]
    missing_elements: List[str]
    compliance_impact: str
    remediation_required: bool
    remediation_timeframe: str
    evidence: List[Dict[str, Any]]
    regulatory_risk: str
    details: Dict[str, Any]


class AuditTrailValidator:
    """Audit trail validation and compliance monitoring."""
    
    def __init__(self):
        """Initialise audit trail validator."""
        self.logger = logger.bind(component="audit_trail_validator")
        
        # Required audit elements for different operations
        self.required_audit_elements = {
            AuditEventType.DATA_ACCESS: [
                "audit_id", "timestamp", "user_id", "session_id", "resource", 
                "action", "outcome", "ip_address", "data_elements", "privacy_classification"
            ],
            AuditEventType.DATA_MODIFICATION: [
                "audit_id", "timestamp", "user_id", "session_id", "resource", 
                "action", "outcome", "ip_address", "data_elements", "privacy_classification",
                "before_state", "after_state", "change_reason"
            ],
            AuditEventType.DATA_EXPORT: [
                "audit_id", "timestamp", "user_id", "session_id", "resource", 
                "action", "outcome", "ip_address", "data_elements", "privacy_classification",
                "export_format", "recipient", "purpose"
            ],
            AuditEventType.DATA_DELETION: [
                "audit_id", "timestamp", "user_id", "session_id", "resource", 
                "action", "outcome", "ip_address", "data_elements", "privacy_classification",
                "deletion_reason", "approval_reference", "recovery_possible"
            ],
            AuditEventType.USER_LOGIN: [
                "audit_id", "timestamp", "user_id", "outcome", "ip_address", 
                "user_agent", "authentication_method", "mfa_status"
            ],
            AuditEventType.PRIVACY_VIOLATION: [
                "audit_id", "timestamp", "violation_type", "affected_data", 
                "detection_method", "severity", "notification_sent", "remediation_actions"
            ]
        }
        
        # Compliance monitoring rules
        self.compliance_rules = {
            ComplianceFramework.PRIVACY_ACT_1988: [
                {
                    "rule": "data_access_logging",
                    "description": "All personal information access must be logged",
                    "required_events": [AuditEventType.DATA_ACCESS],
                    "retention_years": 7
                },
                {
                    "rule": "consent_tracking",
                    "description": "Consent collection and withdrawal must be tracked",
                    "required_events": [AuditEventType.DATA_MODIFICATION],
                    "retention_years": 7
                }
            ],
            ComplianceFramework.NOTIFIABLE_DATA_BREACHES: [
                {
                    "rule": "incident_response_logging",
                    "description": "All security incidents must be logged within 30 days",
                    "required_events": [AuditEventType.SECURITY_INCIDENT],
                    "notification_timeframe_hours": 72
                }
            ]
        }
        
        # Audit integrity configuration
        self.integrity_settings = {
            "checksum_algorithm": "sha256",
            "tamper_detection_enabled": True,
            "log_encryption_required": True,
            "backup_frequency_hours": 24,
            "retention_period_years": 7,
            "write_once_storage": True
        }
    
    def validate_audit_completeness(self, audit_records: List[AuditRecord], 
                                  data_operations: List[Dict[str, Any]]) -> List[AuditTrailViolation]:
        """
        Validate completeness of audit trail coverage.
        
        Args:
            audit_records: List of audit records
            data_operations: List of data operations that should be audited
            
        Returns:
            List of audit completeness violations
        """
        violations = []
        
        # Create audit lookup by correlation ID and timestamp
        audit_lookup = {}
        for record in audit_records:
            key = f"{record.correlation_id}_{record.resource}_{record.action}"
            if key not in audit_lookup:
                audit_lookup[key] = []
            audit_lookup[key].append(record)
        
        # Check each data operation has corresponding audit record
        for operation in data_operations:
            operation_key = f"{operation.get('correlation_id')}_{operation.get('resource')}_{operation.get('action')}"
            
            if operation_key not in audit_lookup:
                violations.append(AuditTrailViolation(
                    violation_type="missing_audit_record",
                    severity="high",
                    description=f"Data operation not audited: {operation.get('resource')} - {operation.get('action')}",
                    affected_records=[operation.get('operation_id', 'unknown')],
                    missing_elements=["audit_record"],
                    compliance_impact="high",
                    remediation_required=True,
                    remediation_timeframe="immediate",
                    evidence=[{"operation": operation}],
                    regulatory_risk="high",
                    details={"operation": operation, "expected_audit_key": operation_key}
                ))
            else:
                # Check audit record completeness for the operation
                audit_record = audit_lookup[operation_key][0]
                event_type = audit_record.event_type
                required_elements = self.required_audit_elements.get(event_type, [])
                
                missing_elements = []
                for element in required_elements:
                    if not hasattr(audit_record, element) or getattr(audit_record, element) is None:
                        if element not in audit_record.details:
                            missing_elements.append(element)
                
                if missing_elements:
                    violations.append(AuditTrailViolation(
                        violation_type="incomplete_audit_record",
                        severity="medium",
                        description=f"Audit record missing required elements: {missing_elements}",
                        affected_records=[audit_record.audit_id],
                        missing_elements=missing_elements,
                        compliance_impact="medium",
                        remediation_required=True,
                        remediation_timeframe="24 hours",
                        evidence=[{"audit_record": audit_record.__dict__}],
                        regulatory_risk="medium",
                        details={"required_elements": required_elements, "missing_elements": missing_elements}
                    ))
        
        return violations
    
    def validate_audit_integrity(self, audit_records: List[AuditRecord]) -> List[AuditTrailViolation]:
        """
        Validate integrity of audit records (tamper detection).
        
        Args:
            audit_records: List of audit records to validate
            
        Returns:
            List of audit integrity violations
        """
        violations = []
        
        for record in audit_records:
            # Verify checksum integrity
            if record.checksum:
                # Reconstruct checksum from record content
                record_content = {
                    "timestamp": record.timestamp,
                    "event_type": record.event_type.value,
                    "user_id": record.user_id,
                    "resource": record.resource,
                    "action": record.action,
                    "outcome": record.outcome,
                    "details": record.details
                }
                
                content_string = json.dumps(record_content, sort_keys=True, separators=(',', ':'))
                calculated_checksum = hashlib.sha256(content_string.encode()).hexdigest()
                
                if calculated_checksum != record.checksum:
                    violations.append(AuditTrailViolation(
                        violation_type="audit_record_tampering",
                        severity="critical",
                        description=f"Audit record checksum mismatch - possible tampering detected",
                        affected_records=[record.audit_id],
                        missing_elements=[],
                        compliance_impact="critical",
                        remediation_required=True,
                        remediation_timeframe="immediate",
                        evidence=[{
                            "audit_id": record.audit_id,
                            "stored_checksum": record.checksum,
                            "calculated_checksum": calculated_checksum
                        }],
                        regulatory_risk="critical",
                        details={
                            "stored_checksum": record.checksum,
                            "calculated_checksum": calculated_checksum,
                            "record_content": record_content
                        }
                    ))
            else:
                violations.append(AuditTrailViolation(
                    violation_type="missing_audit_checksum",
                    severity="high",
                    description="Audit record missing integrity checksum",
                    affected_records=[record.audit_id],
                    missing_elements=["checksum"],
                    compliance_impact="high",
                    remediation_required=True,
                    remediation_timeframe="24 hours",
                    evidence=[{"audit_id": record.audit_id}],
                    regulatory_risk="high",
                    details={"audit_record": record.__dict__}
                ))
        
        # Check for chronological consistency
        sorted_records = sorted(audit_records, key=lambda x: x.timestamp)
        for i in range(1, len(sorted_records)):
            current_time = datetime.fromisoformat(sorted_records[i].timestamp.replace("Z", "+00:00"))
            previous_time = datetime.fromisoformat(sorted_records[i-1].timestamp.replace("Z", "+00:00"))
            
            if current_time < previous_time:
                violations.append(AuditTrailViolation(
                    violation_type="chronological_inconsistency",
                    severity="medium",
                    description="Audit records not in chronological order - possible tampering",
                    affected_records=[sorted_records[i].audit_id, sorted_records[i-1].audit_id],
                    missing_elements=[],
                    compliance_impact="medium",
                    remediation_required=True,
                    remediation_timeframe="24 hours",
                    evidence=[{
                        "current_record": sorted_records[i].audit_id,
                        "current_timestamp": sorted_records[i].timestamp,
                        "previous_record": sorted_records[i-1].audit_id,
                        "previous_timestamp": sorted_records[i-1].timestamp
                    }],
                    regulatory_risk="medium",
                    details={"chronological_order_violated": True}
                ))
        
        return violations
    
    def validate_data_lineage(self, lineage_records: List[DataLineageRecord]) -> List[AuditTrailViolation]:
        """
        Validate data lineage and provenance tracking.
        
        Args:
            lineage_records: List of data lineage records
            
        Returns:
            List of data lineage violations
        """
        violations = []
        
        # Create lineage graph for validation
        lineage_graph = {}
        for record in lineage_records:
            lineage_graph[record.lineage_id] = record
        
        for record in lineage_records:
            # Validate parent-child relationships
            for parent_id in record.parent_lineage_ids:
                if parent_id not in lineage_graph:
                    violations.append(AuditTrailViolation(
                        violation_type="broken_lineage_chain",
                        severity="high",
                        description=f"Missing parent lineage record: {parent_id}",
                        affected_records=[record.lineage_id],
                        missing_elements=["parent_lineage_record"],
                        compliance_impact="high",
                        remediation_required=True,
                        remediation_timeframe="24 hours",
                        evidence=[{"lineage_id": record.lineage_id, "missing_parent": parent_id}],
                        regulatory_risk="high",
                        details={"lineage_record": record.__dict__, "missing_parent_id": parent_id}
                    ))
            
            # Validate provenance chain completeness
            if not record.provenance_chain:
                violations.append(AuditTrailViolation(
                    violation_type="missing_provenance_chain",
                    severity="medium",
                    description="Data lineage record missing provenance chain",
                    affected_records=[record.lineage_id],
                    missing_elements=["provenance_chain"],
                    compliance_impact="medium",
                    remediation_required=True,
                    remediation_timeframe="7 days",
                    evidence=[{"lineage_id": record.lineage_id}],
                    regulatory_risk="medium",
                    details={"lineage_record": record.__dict__}
                ))
            
            # Validate privacy impact assessment
            if record.privacy_impact not in ["none", "low", "medium", "high", "critical"]:
                violations.append(AuditTrailViolation(
                    violation_type="invalid_privacy_impact_assessment",
                    severity="medium",
                    description=f"Invalid privacy impact assessment: {record.privacy_impact}",
                    affected_records=[record.lineage_id],
                    missing_elements=["valid_privacy_impact"],
                    compliance_impact="medium",
                    remediation_required=True,
                    remediation_timeframe="7 days",
                    evidence=[{"lineage_id": record.lineage_id, "privacy_impact": record.privacy_impact}],
                    regulatory_risk="medium",
                    details={"invalid_privacy_impact": record.privacy_impact}
                ))
        
        return violations
    
    def validate_compliance_monitoring(self, audit_records: List[AuditRecord], 
                                     monitoring_rules: List[ComplianceMonitoringRule]) -> List[AuditTrailViolation]:
        """
        Validate compliance monitoring and alerting.
        
        Args:
            audit_records: List of audit records
            monitoring_rules: List of compliance monitoring rules
            
        Returns:
            List of compliance monitoring violations
        """
        violations = []
        
        for rule in monitoring_rules:
            relevant_records = [
                record for record in audit_records 
                if record.event_type in rule.event_types
            ]
            
            # Check if rule has been triggered
            time_window = timedelta(minutes=rule.time_window_minutes)
            now = datetime.now()
            
            recent_events = [
                record for record in relevant_records
                if (now - datetime.fromisoformat(record.timestamp.replace("Z", "+00:00"))) <= time_window
            ]
            
            # Apply rule conditions
            triggered_events = []
            for record in recent_events:
                conditions_met = True
                
                for condition_key, condition_value in rule.conditions.items():
                    record_value = getattr(record, condition_key, None)
                    if record_value is None:
                        record_value = record.details.get(condition_key)
                    
                    if record_value != condition_value:
                        conditions_met = False
                        break
                
                if conditions_met:
                    triggered_events.append(record)
            
            # Check if alert threshold exceeded
            if len(triggered_events) >= rule.alert_threshold:
                violations.append(AuditTrailViolation(
                    violation_type="compliance_rule_triggered",
                    severity=rule.severity,
                    description=f"Compliance rule '{rule.name}' triggered: {len(triggered_events)} events in {rule.time_window_minutes} minutes",
                    affected_records=[event.audit_id for event in triggered_events],
                    missing_elements=[],
                    compliance_impact=rule.severity,
                    remediation_required=True,
                    remediation_timeframe="immediate" if rule.severity == "critical" else "24 hours",
                    evidence=[{
                        "rule": rule.__dict__,
                        "triggered_events": len(triggered_events),
                        "time_window": rule.time_window_minutes
                    }],
                    regulatory_risk=rule.severity,
                    details={
                        "rule_id": rule.rule_id,
                        "triggered_events": [event.__dict__ for event in triggered_events],
                        "compliance_framework": rule.compliance_framework.value
                    }
                ))
        
        return violations
    
    def validate_retention_compliance(self, audit_records: List[AuditRecord]) -> List[AuditTrailViolation]:
        """
        Validate audit record retention compliance.
        
        Args:
            audit_records: List of audit records
            
        Returns:
            List of retention compliance violations
        """
        violations = []
        
        retention_years = self.integrity_settings["retention_period_years"]
        retention_cutoff = datetime.now() - timedelta(days=retention_years * 365)
        
        # Check for records that should have been deleted
        expired_records = [
            record for record in audit_records
            if datetime.fromisoformat(record.timestamp.replace("Z", "+00:00")) < retention_cutoff
        ]
        
        if expired_records:
            violations.append(AuditTrailViolation(
                violation_type="retention_policy_violation",
                severity="medium",
                description=f"{len(expired_records)} audit records exceed retention period",
                affected_records=[record.audit_id for record in expired_records],
                missing_elements=[],
                compliance_impact="medium",
                remediation_required=True,
                remediation_timeframe="30 days",
                evidence=[{
                    "expired_records_count": len(expired_records),
                    "retention_period_years": retention_years
                }],
                regulatory_risk="medium",
                details={
                    "retention_cutoff": retention_cutoff.isoformat(),
                    "expired_records": [record.audit_id for record in expired_records[:10]]  # Sample
                }
            ))
        
        # Check for gaps in audit coverage
        if audit_records:
            sorted_records = sorted(audit_records, key=lambda x: x.timestamp)
            
            for i in range(1, len(sorted_records)):
                current_time = datetime.fromisoformat(sorted_records[i].timestamp.replace("Z", "+00:00"))
                previous_time = datetime.fromisoformat(sorted_records[i-1].timestamp.replace("Z", "+00:00"))
                time_gap = current_time - previous_time
                
                # Flag gaps longer than 24 hours (possible missing records)
                if time_gap > timedelta(hours=24):
                    violations.append(AuditTrailViolation(
                        violation_type="audit_coverage_gap",
                        severity="medium",
                        description=f"Audit coverage gap detected: {time_gap.days} days",
                        affected_records=[sorted_records[i-1].audit_id, sorted_records[i].audit_id],
                        missing_elements=["continuous_audit_coverage"],
                        compliance_impact="medium",
                        remediation_required=True,
                        remediation_timeframe="7 days",
                        evidence=[{
                            "gap_start": sorted_records[i-1].timestamp,
                            "gap_end": sorted_records[i].timestamp,
                            "gap_duration_hours": time_gap.total_seconds() / 3600
                        }],
                        regulatory_risk="medium",
                        details={"gap_duration_days": time_gap.days}
                    ))
        
        return violations
    
    def conduct_comprehensive_audit_assessment(self, 
                                             audit_records: List[AuditRecord],
                                             data_operations: List[Dict[str, Any]],
                                             lineage_records: List[DataLineageRecord],
                                             monitoring_rules: List[ComplianceMonitoringRule]) -> Dict[str, Any]:
        """
        Conduct comprehensive audit trail assessment.
        
        Args:
            audit_records: List of audit records
            data_operations: List of data operations
            lineage_records: List of data lineage records
            monitoring_rules: List of compliance monitoring rules
            
        Returns:
            Comprehensive audit assessment results
        """
        assessment_id = f"audit_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        all_violations = []
        
        # Run all audit validation tests
        completeness_violations = self.validate_audit_completeness(audit_records, data_operations)
        all_violations.extend(completeness_violations)
        
        integrity_violations = self.validate_audit_integrity(audit_records)
        all_violations.extend(integrity_violations)
        
        lineage_violations = self.validate_data_lineage(lineage_records)
        all_violations.extend(lineage_violations)
        
        compliance_violations = self.validate_compliance_monitoring(audit_records, monitoring_rules)
        all_violations.extend(compliance_violations)
        
        retention_violations = self.validate_retention_compliance(audit_records)
        all_violations.extend(retention_violations)
        
        # Categorise violations by severity
        violations_by_severity = {
            "critical": [v for v in all_violations if v.severity == "critical"],
            "high": [v for v in all_violations if v.severity == "high"],
            "medium": [v for v in all_violations if v.severity == "medium"],
            "low": [v for v in all_violations if v.severity == "low"]
        }
        
        # Calculate audit coverage metrics
        audit_coverage = self._calculate_audit_coverage(audit_records, data_operations)
        
        # Calculate compliance scores
        compliance_scores = self._calculate_compliance_scores(all_violations, monitoring_rules)
        
        # Generate recommendations
        recommendations = self._generate_audit_recommendations(all_violations, violations_by_severity)
        
        return {
            "assessment_id": assessment_id,
            "assessment_timestamp": datetime.now().isoformat(),
            "total_violations": len(all_violations),
            "violations_by_severity": {
                severity: len(violations) for severity, violations in violations_by_severity.items()
            },
            "violations_by_type": self._categorise_violations_by_type(all_violations),
            "audit_coverage": audit_coverage,
            "compliance_scores": compliance_scores,
            "integrity_status": {
                "tamper_incidents": len([v for v in integrity_violations if v.violation_type == "audit_record_tampering"]),
                "checksum_failures": len([v for v in integrity_violations if v.violation_type == "missing_audit_checksum"]),
                "chronological_issues": len([v for v in integrity_violations if v.violation_type == "chronological_inconsistency"])
            },
            "lineage_status": {
                "broken_chains": len([v for v in lineage_violations if v.violation_type == "broken_lineage_chain"]),
                "missing_provenance": len([v for v in lineage_violations if v.violation_type == "missing_provenance_chain"])
            },
            "recommendations": recommendations,
            "detailed_violations": [violation.__dict__ for violation in all_violations],
            "regulatory_risk_assessment": self._assess_regulatory_risk(all_violations),
            "next_assessment_date": (datetime.now() + timedelta(days=30)).isoformat()
        }
    
    def _calculate_audit_coverage(self, audit_records: List[AuditRecord], data_operations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate audit coverage metrics."""
        if not data_operations:
            return {"overall_coverage": 0.0, "by_operation_type": {}}
        
        # Overall coverage
        audited_operations = set()
        for record in audit_records:
            if record.correlation_id:
                audited_operations.add(record.correlation_id)
        
        total_operations = len(data_operations)
        overall_coverage = len(audited_operations) / total_operations if total_operations > 0 else 0.0
        
        # Coverage by operation type
        operations_by_type = {}
        audited_by_type = {}
        
        for operation in data_operations:
            op_type = operation.get("operation_type", "unknown")
            if op_type not in operations_by_type:
                operations_by_type[op_type] = 0
                audited_by_type[op_type] = 0
            operations_by_type[op_type] += 1
            
            if operation.get("correlation_id") in audited_operations:
                audited_by_type[op_type] += 1
        
        coverage_by_type = {
            op_type: audited_by_type[op_type] / operations_by_type[op_type]
            for op_type in operations_by_type
        }
        
        return {
            "overall_coverage": overall_coverage,
            "by_operation_type": coverage_by_type,
            "total_operations": total_operations,
            "audited_operations": len(audited_operations)
        }
    
    def _calculate_compliance_scores(self, violations: List[AuditTrailViolation], 
                                   monitoring_rules: List[ComplianceMonitoringRule]) -> Dict[str, float]:
        """Calculate compliance scores by framework."""
        compliance_scores = {}
        
        # Group violations by compliance framework
        for rule in monitoring_rules:
            framework = rule.compliance_framework.value
            if framework not in compliance_scores:
                compliance_scores[framework] = {"violations": 0, "critical_violations": 0}
        
        for violation in violations:
            # Map violations to frameworks (simplified)
            if "privacy" in violation.violation_type or "consent" in violation.violation_type:
                framework = ComplianceFramework.PRIVACY_ACT_1988.value
            elif "security_incident" in violation.violation_type:
                framework = ComplianceFramework.NOTIFIABLE_DATA_BREACHES.value
            else:
                framework = "general"
            
            if framework not in compliance_scores:
                compliance_scores[framework] = {"violations": 0, "critical_violations": 0}
            
            compliance_scores[framework]["violations"] += 1
            if violation.severity == "critical":
                compliance_scores[framework]["critical_violations"] += 1
        
        # Calculate scores (100 - violations penalty)
        for framework in compliance_scores:
            violations_count = compliance_scores[framework]["violations"]
            critical_count = compliance_scores[framework]["critical_violations"]
            
            # Penalty: 10 points per violation, 20 points per critical
            penalty = violations_count * 10 + critical_count * 20
            score = max(0, 100 - penalty)
            
            compliance_scores[framework]["score"] = score
        
        return compliance_scores
    
    def _categorise_violations_by_type(self, violations: List[AuditTrailViolation]) -> Dict[str, int]:
        """Categorise violations by type."""
        type_counts = {}
        for violation in violations:
            if violation.violation_type not in type_counts:
                type_counts[violation.violation_type] = 0
            type_counts[violation.violation_type] += 1
        return type_counts
    
    def _generate_audit_recommendations(self, all_violations: List[AuditTrailViolation], 
                                      violations_by_severity: Dict[str, List]) -> List[str]:
        """Generate audit trail recommendations."""
        recommendations = []
        
        if violations_by_severity["critical"]:
            recommendations.append("URGENT: Address critical audit trail violations immediately to prevent regulatory action")
        
        violation_types = [v.violation_type for v in all_violations]
        
        if "missing_audit_record" in violation_types:
            recommendations.append("Implement comprehensive audit logging for all data operations")
        
        if "audit_record_tampering" in violation_types:
            recommendations.append("Strengthen audit log integrity protection with immutable storage")
        
        if "broken_lineage_chain" in violation_types:
            recommendations.append("Repair data lineage tracking and implement automated validation")
        
        if "compliance_rule_triggered" in violation_types:
            recommendations.append("Review and tune compliance monitoring rules to reduce false positives")
        
        if "retention_policy_violation" in violation_types:
            recommendations.append("Implement automated audit record retention and disposal policies")
        
        if len(all_violations) > 10:
            recommendations.append("Consider comprehensive audit system redesign and automation")
        
        return recommendations
    
    def _assess_regulatory_risk(self, violations: List[AuditTrailViolation]) -> Dict[str, Any]:
        """Assess regulatory risk from audit violations."""
        critical_violations = [v for v in violations if v.severity == "critical"]
        high_violations = [v for v in violations if v.severity == "high"]
        
        risk_level = "low"
        if len(critical_violations) > 0:
            risk_level = "critical"
        elif len(high_violations) > 3:
            risk_level = "high"
        elif len(violations) > 5:
            risk_level = "medium"
        
        return {
            "overall_risk_level": risk_level,
            "critical_violations": len(critical_violations),
            "high_violations": len(high_violations),
            "regulatory_frameworks_affected": list(set([
                v.details.get("compliance_framework", "unknown") for v in violations
            ])),
            "immediate_action_required": len(critical_violations) > 0 or len(high_violations) > 5
        }


# Test suite
class TestAuditTrailValidation:
    """Test suite for audit trail validation."""
    
    @pytest.fixture
    def audit_validator(self):
        """Create audit trail validator instance."""
        return AuditTrailValidator()
    
    @pytest.fixture
    def test_audit_records(self):
        """Test audit records."""
        now = datetime.now()
        return [
            AuditRecord(
                audit_id="audit001",
                timestamp=now.isoformat(),
                event_type=AuditEventType.DATA_ACCESS,
                level=AuditLevel.INFO,
                user_id="user001",
                session_id="sess001",
                resource="health_data_table",
                action="select",
                outcome="success",
                ip_address="192.168.1.100",
                user_agent="Mozilla/5.0",
                data_elements=["patient_id", "diagnosis"],
                privacy_classification="confidential",
                compliance_frameworks=[ComplianceFramework.PRIVACY_ACT_1988],
                checksum="abc123def456",
                details={"query": "SELECT * FROM health_data WHERE sa2_code = '101021007'"},
                correlation_id="op001"
            ),
            AuditRecord(
                audit_id="audit002",
                timestamp=(now + timedelta(minutes=5)).isoformat(),
                event_type=AuditEventType.DATA_EXPORT,
                level=AuditLevel.WARNING,
                user_id="user002",
                session_id="sess002",
                resource="aggregated_health_data",
                action="export",
                outcome="success",
                ip_address="192.168.1.101",
                user_agent="Chrome/91.0",
                data_elements=["sa2_code", "condition_count"],
                privacy_classification="public",
                compliance_frameworks=[ComplianceFramework.PRIVACY_ACT_1988],
                checksum="def456ghi789",
                details={
                    "export_format": "csv",
                    "recipient": "researcher@university.edu",
                    "purpose": "health_research"
                },
                correlation_id="op002"
            )
        ]
    
    @pytest.fixture
    def test_data_operations(self):
        """Test data operations."""
        return [
            {
                "operation_id": "op001",
                "operation_type": "data_access",
                "resource": "health_data_table",
                "action": "select",
                "correlation_id": "op001",
                "timestamp": datetime.now().isoformat()
            },
            {
                "operation_id": "op002",
                "operation_type": "data_export",
                "resource": "aggregated_health_data",
                "action": "export",
                "correlation_id": "op002",
                "timestamp": (datetime.now() + timedelta(minutes=5)).isoformat()
            },
            {
                "operation_id": "op003",
                "operation_type": "data_modification",
                "resource": "patient_records",
                "action": "update",
                "correlation_id": "op003",
                "timestamp": (datetime.now() + timedelta(minutes=10)).isoformat()
            }
        ]
    
    @pytest.fixture
    def test_lineage_records(self):
        """Test data lineage records."""
        return [
            DataLineageRecord(
                lineage_id="lineage001",
                source_dataset="raw_health_data",
                target_dataset="processed_health_data",
                transformation_type="deidentification",
                transformation_timestamp=datetime.now().isoformat(),
                processor_id="processor001",
                data_elements_affected=["patient_id", "name", "address"],
                privacy_impact="high",
                compliance_notes="Removed direct identifiers per APP 11",
                parent_lineage_ids=[],
                child_lineage_ids=["lineage002"],
                quality_metrics={"completeness": 0.98, "accuracy": 0.95},
                provenance_chain=[
                    {"step": 1, "process": "data_extraction", "timestamp": "2023-06-01T10:00:00Z"},
                    {"step": 2, "process": "deidentification", "timestamp": "2023-06-01T10:30:00Z"}
                ]
            )
        ]
    
    @pytest.fixture
    def test_monitoring_rules(self):
        """Test compliance monitoring rules."""
        return [
            ComplianceMonitoringRule(
                rule_id="rule001",
                name="High-volume data access monitoring",
                description="Monitor for unusual data access patterns",
                compliance_framework=ComplianceFramework.PRIVACY_ACT_1988,
                event_types=[AuditEventType.DATA_ACCESS],
                conditions={"outcome": "success"},
                alert_threshold=10,
                time_window_minutes=60,
                severity="high",
                notification_recipients=["security@example.com"]
            )
        ]
    
    def test_audit_completeness_validation(self, audit_validator, test_audit_records, test_data_operations):
        """Test audit completeness validation."""
        violations = audit_validator.validate_audit_completeness(test_audit_records, test_data_operations)
        
        # Should detect missing audit record for op003
        missing_audit_violations = [v for v in violations if v.violation_type == "missing_audit_record"]
        assert len(missing_audit_violations) > 0, "Should detect missing audit records"
        
        # Verify violation details
        for violation in missing_audit_violations:
            assert violation.severity in ["high", "medium"]
            assert violation.compliance_impact in ["high", "medium"]
            assert violation.remediation_required is True
            assert len(violation.affected_records) > 0
    
    def test_audit_integrity_validation(self, audit_validator):
        """Test audit integrity validation."""
        # Create test records with integrity issues
        now = datetime.now()
        tampered_records = [
            AuditRecord(
                audit_id="audit_tampered",
                timestamp=now.isoformat(),
                event_type=AuditEventType.DATA_ACCESS,
                level=AuditLevel.INFO,
                user_id="user001",
                session_id="sess001",
                resource="health_data_table",
                action="select",
                outcome="success",
                ip_address="192.168.1.100",
                user_agent="Mozilla/5.0",
                data_elements=["patient_id"],
                privacy_classification="confidential",
                compliance_frameworks=[ComplianceFramework.PRIVACY_ACT_1988],
                checksum="invalid_checksum",  # Wrong checksum
                details={"query": "SELECT * FROM health_data"},
                correlation_id="op001"
            ),
            AuditRecord(
                audit_id="audit_no_checksum",
                timestamp=(now + timedelta(minutes=5)).isoformat(),
                event_type=AuditEventType.DATA_ACCESS,
                level=AuditLevel.INFO,
                user_id="user001",
                session_id="sess001",
                resource="health_data_table",
                action="select",
                outcome="success",
                ip_address="192.168.1.100",
                user_agent="Mozilla/5.0",
                data_elements=["patient_id"],
                privacy_classification="confidential",
                compliance_frameworks=[ComplianceFramework.PRIVACY_ACT_1988],
                checksum="",  # Missing checksum
                details={"query": "SELECT * FROM health_data"},
                correlation_id="op002"
            )
        ]
        
        violations = audit_validator.validate_audit_integrity(tampered_records)
        
        # Should detect integrity violations
        assert len(violations) > 0, "Should detect audit integrity violations"
        
        violation_types = [v.violation_type for v in violations]
        assert "audit_record_tampering" in violation_types or "missing_audit_checksum" in violation_types
        
        # Check for critical violations
        critical_violations = [v for v in violations if v.severity == "critical"]
        if critical_violations:
            assert critical_violations[0].remediation_timeframe == "immediate"
    
    def test_data_lineage_validation(self, audit_validator, test_lineage_records):
        """Test data lineage validation."""
        # Add problematic lineage record
        problematic_lineage = DataLineageRecord(
            lineage_id="lineage_broken",
            source_dataset="unknown_source",
            target_dataset="processed_data",
            transformation_type="aggregation",
            transformation_timestamp=datetime.now().isoformat(),
            processor_id="processor002",
            data_elements_affected=["health_metrics"],
            privacy_impact="invalid_level",  # Invalid privacy impact
            compliance_notes="",
            parent_lineage_ids=["missing_parent"],  # Non-existent parent
            child_lineage_ids=[],
            quality_metrics={},
            provenance_chain=[]  # Missing provenance chain
        )
        
        test_records = test_lineage_records + [problematic_lineage]
        violations = audit_validator.validate_data_lineage(test_records)
        
        # Should detect lineage violations
        assert len(violations) > 0, "Should detect data lineage violations"
        
        violation_types = [v.violation_type for v in violations]
        expected_types = ["broken_lineage_chain", "missing_provenance_chain", "invalid_privacy_impact_assessment"]
        
        detected_types = [vtype for vtype in expected_types if vtype in violation_types]
        assert len(detected_types) > 0, "Should detect lineage validation issues"
    
    def test_compliance_monitoring_validation(self, audit_validator, test_audit_records, test_monitoring_rules):
        """Test compliance monitoring validation."""
        # Create records that trigger the monitoring rule
        now = datetime.now()
        high_volume_records = []
        
        for i in range(12):  # Above threshold of 10
            record = AuditRecord(
                audit_id=f"audit_volume_{i}",
                timestamp=(now + timedelta(minutes=i)).isoformat(),
                event_type=AuditEventType.DATA_ACCESS,
                level=AuditLevel.INFO,
                user_id="user001",
                session_id="sess001",
                resource="health_data_table",
                action="select",
                outcome="success",
                ip_address="192.168.1.100",
                user_agent="Mozilla/5.0",
                data_elements=["patient_id"],
                privacy_classification="confidential",
                compliance_frameworks=[ComplianceFramework.PRIVACY_ACT_1988],
                checksum=f"checksum_{i}",
                details={"query": f"SELECT * FROM health_data LIMIT {i}"},
                correlation_id=f"op_{i}"
            )
            high_volume_records.append(record)
        
        all_records = test_audit_records + high_volume_records
        violations = audit_validator.validate_compliance_monitoring(all_records, test_monitoring_rules)
        
        # Should trigger compliance rule
        rule_violations = [v for v in violations if v.violation_type == "compliance_rule_triggered"]
        assert len(rule_violations) > 0, "Should trigger compliance monitoring rule"
        
        for violation in rule_violations:
            assert violation.severity in ["high", "critical"]
            assert len(violation.affected_records) >= test_monitoring_rules[0].alert_threshold
    
    def test_retention_compliance_validation(self, audit_validator):
        """Test retention compliance validation."""
        # Create old records that should have been deleted
        old_date = datetime.now() - timedelta(days=8*365)  # 8 years old
        
        expired_records = [
            AuditRecord(
                audit_id="audit_expired",
                timestamp=old_date.isoformat(),
                event_type=AuditEventType.DATA_ACCESS,
                level=AuditLevel.INFO,
                user_id="user001",
                session_id="sess001",
                resource="health_data_table",
                action="select",
                outcome="success",
                ip_address="192.168.1.100",
                user_agent="Mozilla/5.0",
                data_elements=["patient_id"],
                privacy_classification="confidential",
                compliance_frameworks=[ComplianceFramework.PRIVACY_ACT_1988],
                checksum="expired_checksum",
                details={"query": "OLD QUERY"},
                correlation_id="old_op"
            )
        ]
        
        violations = audit_validator.validate_retention_compliance(expired_records)
        
        # Should detect retention violations
        retention_violations = [v for v in violations if v.violation_type == "retention_policy_violation"]
        assert len(retention_violations) > 0, "Should detect retention policy violations"
        
        for violation in retention_violations:
            assert violation.severity in ["medium", "high"]
            assert violation.remediation_timeframe in ["30 days", "7 days"]
    
    def test_comprehensive_audit_assessment(self, audit_validator, test_audit_records, test_data_operations, 
                                          test_lineage_records, test_monitoring_rules):
        """Test comprehensive audit trail assessment."""
        assessment = audit_validator.conduct_comprehensive_audit_assessment(
            test_audit_records, test_data_operations, test_lineage_records, test_monitoring_rules
        )
        
        # Verify assessment structure
        required_keys = [
            "assessment_id", "assessment_timestamp", "total_violations",
            "violations_by_severity", "violations_by_type", "audit_coverage",
            "compliance_scores", "integrity_status", "lineage_status",
            "recommendations", "detailed_violations", "regulatory_risk_assessment",
            "next_assessment_date"
        ]
        
        for key in required_keys:
            assert key in assessment, f"Assessment should include {key}"
        
        # Verify assessment quality
        assert assessment["total_violations"] >= 0
        assert "overall_coverage" in assessment["audit_coverage"]
        assert 0.0 <= assessment["audit_coverage"]["overall_coverage"] <= 1.0
        assert isinstance(assessment["recommendations"], list)
        assert assessment["regulatory_risk_assessment"]["overall_risk_level"] in ["low", "medium", "high", "critical"]
    
    def test_required_audit_elements_configuration(self, audit_validator):
        """Test required audit elements are properly configured."""
        required_elements = audit_validator.required_audit_elements
        
        # Verify critical event types have requirements
        critical_events = [
            AuditEventType.DATA_ACCESS,
            AuditEventType.DATA_MODIFICATION,
            AuditEventType.DATA_EXPORT,
            AuditEventType.DATA_DELETION
        ]
        
        for event_type in critical_events:
            assert event_type in required_elements, f"Event type {event_type} should have required elements"
            assert len(required_elements[event_type]) > 5, f"Event type {event_type} should have comprehensive requirements"
        
        # Verify common elements are present
        common_elements = ["audit_id", "timestamp", "user_id", "resource", "action", "outcome"]
        for event_type, elements in required_elements.items():
            for common_element in common_elements:
                if event_type != AuditEventType.USER_LOGIN:  # Some elements may not apply to all events
                    assert common_element in elements or any(common_element in elem for elem in elements), f"Event {event_type} should include {common_element}"
    
    def test_compliance_rules_configuration(self, audit_validator):
        """Test compliance rules are properly configured."""
        compliance_rules = audit_validator.compliance_rules
        
        # Verify key frameworks have rules
        key_frameworks = [ComplianceFramework.PRIVACY_ACT_1988, ComplianceFramework.NOTIFIABLE_DATA_BREACHES]
        for framework in key_frameworks:
            assert framework in compliance_rules, f"Framework {framework} should have compliance rules"
            assert len(compliance_rules[framework]) > 0, f"Framework {framework} should have rules defined"
        
        # Verify rule structure
        for framework, rules in compliance_rules.items():
            for rule in rules:
                assert "rule" in rule, "Compliance rule should have 'rule' field"
                assert "description" in rule, "Compliance rule should have 'description' field"
                assert "required_events" in rule, "Compliance rule should have 'required_events' field"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])