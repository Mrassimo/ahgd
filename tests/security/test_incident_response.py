"""
Security Incident Response Testing

Comprehensive testing suite for security incident response including:
- Security incident detection and response testing
- Incident response plan validation
- Security alert and notification testing
- Breach response procedure testing
- Recovery and business continuity testing
- Post-incident analysis and learning validation

This test suite ensures the platform has robust security incident response
capabilities that meet regulatory requirements for health data protection.
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
import time

import polars as pl
import numpy as np
from loguru import logger


class IncidentType(Enum):
    """Types of security incidents."""
    DATA_BREACH = "data_breach"
    UNAUTHORISED_ACCESS = "unauthorised_access"
    MALWARE_INFECTION = "malware_infection"
    DENIAL_OF_SERVICE = "denial_of_service"
    SYSTEM_COMPROMISE = "system_compromise"
    INSIDER_THREAT = "insider_threat"
    PRIVACY_VIOLATION = "privacy_violation"
    RANSOMWARE = "ransomware"
    PHISHING_ATTACK = "phishing_attack"
    API_ABUSE = "api_abuse"


class IncidentSeverity(Enum):
    """Incident severity levels."""
    CRITICAL = "critical"  # Critical systems compromised, major data exposure
    HIGH = "high"         # Significant impact, limited data exposure
    MEDIUM = "medium"     # Moderate impact, potential for escalation
    LOW = "low"          # Minor impact, contained incident
    INFO = "info"        # Informational, no immediate threat


class IncidentStatus(Enum):
    """Incident status."""
    NEW = "new"
    INVESTIGATING = "investigating"
    CONTAINMENT = "containment"
    ERADICATION = "eradication"
    RECOVERY = "recovery"
    POST_INCIDENT = "post_incident"
    CLOSED = "closed"


class NotificationChannel(Enum):
    """Incident notification channels."""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    PHONE = "phone"
    DASHBOARD = "dashboard"


@dataclass
class SecurityIncident:
    """Security incident record."""
    incident_id: str
    incident_type: IncidentType
    severity: IncidentSeverity
    status: IncidentStatus
    title: str
    description: str
    affected_systems: List[str]
    affected_data_types: List[str]
    detection_timestamp: str
    reported_timestamp: str
    response_team: List[str]
    stakeholders_notified: List[str]
    containment_actions: List[str]
    eradication_actions: List[str]
    recovery_actions: List[str]
    lessons_learned: List[str]
    estimated_impact: Dict[str, Any]
    regulatory_notification_required: bool
    regulatory_notifications_sent: List[Dict[str, str]]
    evidence_collected: List[Dict[str, Any]]
    timeline: List[Dict[str, str]]
    root_cause: Optional[str] = None
    resolution_timestamp: Optional[str] = None


@dataclass
class IncidentResponse:
    """Incident response record."""
    response_id: str
    incident_id: str
    response_timestamp: str
    responder_id: str
    action_type: str  # "detection", "containment", "eradication", "recovery", "notification"
    action_description: str
    outcome: str
    duration_minutes: int
    success: bool
    evidence_collected: List[str]
    next_actions: List[str]


@dataclass
class IncidentAlert:
    """Security incident alert."""
    alert_id: str
    alert_timestamp: str
    source_system: str
    alert_type: str
    severity: IncidentSeverity
    message: str
    indicators: Dict[str, Any]
    false_positive: bool
    escalated_to_incident: bool
    incident_id: Optional[str]
    acknowledged_by: Optional[str]
    acknowledged_timestamp: Optional[str]


@dataclass
class IncidentResponseTest:
    """Incident response test case."""
    test_id: str
    test_name: str
    incident_scenario: Dict[str, Any]
    expected_response_time: int  # minutes
    expected_actions: List[str]
    expected_notifications: List[str]
    actual_response_time: Optional[int]
    actual_actions: List[str]
    actual_notifications: List[str]
    test_passed: bool
    gaps_identified: List[str]
    recommendations: List[str]


class SecurityIncidentResponseTester:
    """Security incident response testing and validation."""
    
    def __init__(self):
        """Initialise security incident response tester."""
        self.logger = logger.bind(component="security_incident_response_tester")
        
        # Incident response time requirements (in minutes)
        self.response_time_requirements = {
            IncidentSeverity.CRITICAL: {
                "detection_to_response": 15,
                "initial_containment": 30,
                "stakeholder_notification": 60,
                "regulatory_notification": 4320  # 72 hours
            },
            IncidentSeverity.HIGH: {
                "detection_to_response": 30,
                "initial_containment": 60,
                "stakeholder_notification": 120,
                "regulatory_notification": 4320  # 72 hours
            },
            IncidentSeverity.MEDIUM: {
                "detection_to_response": 60,
                "initial_containment": 240,
                "stakeholder_notification": 480,
                "regulatory_notification": 10080  # 7 days
            },
            IncidentSeverity.LOW: {
                "detection_to_response": 240,
                "initial_containment": 480,
                "stakeholder_notification": 1440,
                "regulatory_notification": 20160  # 14 days
            }
        }
        
        # Required incident response actions by type
        self.required_response_actions = {
            IncidentType.DATA_BREACH: [
                "immediate_containment",
                "affected_data_assessment",
                "stakeholder_notification",
                "regulatory_notification",
                "forensic_investigation",
                "remediation_plan",
                "communication_plan"
            ],
            IncidentType.UNAUTHORISED_ACCESS: [
                "access_revocation",
                "system_isolation",
                "credential_reset",
                "access_review",
                "monitoring_enhancement"
            ],
            IncidentType.MALWARE_INFECTION: [
                "system_isolation",
                "malware_analysis",
                "antivirus_scan",
                "system_restoration",
                "patch_management"
            ],
            IncidentType.RANSOMWARE: [
                "immediate_isolation",
                "backup_restoration",
                "law_enforcement_notification",
                "business_continuity_activation",
                "communication_management"
            ]
        }
        
        # Regulatory notification requirements
        self.regulatory_requirements = {
            "notifiable_data_breaches_scheme": {
                "notification_timeframe_hours": 72,
                "authority": "Office of the Australian Information Commissioner (OAIC)",
                "triggers": ["likely_serious_harm", "personal_information_exposure"]
            },
            "health_records_act": {
                "notification_timeframe_hours": 72,
                "authority": "State Health Department",
                "triggers": ["health_information_breach", "unauthorised_disclosure"]
            },
            "therapeutic_goods_administration": {
                "notification_timeframe_hours": 24,
                "authority": "TGA",
                "triggers": ["medical_device_compromise", "therapeutic_data_breach"]
            }
        }
        
        # Incident response team roles
        self.response_team_roles = {
            "incident_commander": "Overall incident coordination and decision making",
            "security_analyst": "Technical investigation and containment",
            "forensics_specialist": "Evidence collection and analysis",
            "legal_counsel": "Legal and regulatory guidance",
            "communications_lead": "Internal and external communications",
            "business_continuity": "Service restoration and continuity",
            "privacy_officer": "Privacy impact assessment and notification"
        }
    
    def test_incident_detection_capabilities(self, security_events: List[Dict[str, Any]]) -> List[IncidentAlert]:
        """
        Test security incident detection capabilities.
        
        Args:
            security_events: List of security events to test detection against
            
        Returns:
            List of generated alerts
        """
        alerts = []
        
        for event in security_events:
            event_type = event.get("event_type", "unknown")
            event_timestamp = event.get("timestamp", datetime.now().isoformat())
            source_ip = event.get("source_ip", "unknown")
            user_id = event.get("user_id", "unknown")
            
            # Simulate detection logic for different event types
            if event_type == "failed_login_attempts":
                attempts = event.get("attempt_count", 0)
                if attempts >= 5:  # Threshold for suspicious activity
                    alert = IncidentAlert(
                        alert_id=f"ALERT_{len(alerts) + 1:04d}",
                        alert_timestamp=datetime.now().isoformat(),
                        source_system="authentication_system",
                        alert_type="brute_force_attempt",
                        severity=IncidentSeverity.MEDIUM,
                        message=f"Multiple failed login attempts detected: {attempts} attempts from {source_ip}",
                        indicators={
                            "source_ip": source_ip,
                            "user_id": user_id,
                            "attempt_count": attempts,
                            "time_window": "5 minutes"
                        },
                        false_positive=False,
                        escalated_to_incident=attempts >= 10,  # Escalate to incident if >= 10 attempts
                        incident_id=f"INC_{len(alerts) + 1:04d}" if attempts >= 10 else None,
                        acknowledged_by=None,
                        acknowledged_timestamp=None
                    )
                    alerts.append(alert)
            
            elif event_type == "data_access_anomaly":
                access_volume = event.get("records_accessed", 0)
                if access_volume >= 1000:  # Large data access threshold
                    alert = IncidentAlert(
                        alert_id=f"ALERT_{len(alerts) + 1:04d}",
                        alert_timestamp=datetime.now().isoformat(),
                        source_system="data_access_monitor",
                        alert_type="abnormal_data_access",
                        severity=IncidentSeverity.HIGH,
                        message=f"Abnormal data access detected: {access_volume} records accessed by {user_id}",
                        indicators={
                            "user_id": user_id,
                            "records_accessed": access_volume,
                            "access_pattern": "bulk_download",
                            "time_period": "10 minutes"
                        },
                        false_positive=False,
                        escalated_to_incident=True,  # Always escalate large data access
                        incident_id=f"INC_{len(alerts) + 1:04d}",
                        acknowledged_by=None,
                        acknowledged_timestamp=None
                    )
                    alerts.append(alert)
            
            elif event_type == "privilege_escalation":
                new_privileges = event.get("privileges_granted", [])
                if "admin" in str(new_privileges).lower():
                    alert = IncidentAlert(
                        alert_id=f"ALERT_{len(alerts) + 1:04d}",
                        alert_timestamp=datetime.now().isoformat(),
                        source_system="access_control_system",
                        alert_type="privilege_escalation",
                        severity=IncidentSeverity.HIGH,
                        message=f"Administrative privileges granted to {user_id}",
                        indicators={
                            "user_id": user_id,
                            "privileges_granted": new_privileges,
                            "granted_by": event.get("granted_by", "unknown"),
                            "approval_status": event.get("approved", False)
                        },
                        false_positive=event.get("approved", False),
                        escalated_to_incident=not event.get("approved", False),
                        incident_id=f"INC_{len(alerts) + 1:04d}" if not event.get("approved", False) else None,
                        acknowledged_by=None,
                        acknowledged_timestamp=None
                    )
                    alerts.append(alert)
        
        return alerts
    
    def test_incident_response_procedures(self, incident: SecurityIncident) -> List[IncidentResponseTest]:
        """
        Test incident response procedures for a given incident.
        
        Args:
            incident: Security incident to test response procedures
            
        Returns:
            List of incident response test results
        """
        test_results = []
        
        # Test 1: Initial Response Time
        response_time_test = self._test_response_time(incident)
        test_results.append(response_time_test)
        
        # Test 2: Containment Actions
        containment_test = self._test_containment_actions(incident)
        test_results.append(containment_test)
        
        # Test 3: Stakeholder Notifications
        notification_test = self._test_stakeholder_notifications(incident)
        test_results.append(notification_test)
        
        # Test 4: Evidence Collection
        evidence_test = self._test_evidence_collection(incident)
        test_results.append(evidence_test)
        
        # Test 5: Regulatory Compliance
        regulatory_test = self._test_regulatory_compliance(incident)
        test_results.append(regulatory_test)
        
        return test_results
    
    def test_notification_systems(self, incident: SecurityIncident) -> List[Dict[str, Any]]:
        """
        Test incident notification systems.
        
        Args:
            incident: Security incident for notification testing
            
        Returns:
            List of notification test results
        """
        notification_results = []
        
        # Test email notifications
        email_result = self._test_email_notifications(incident)
        notification_results.append(email_result)
        
        # Test SMS notifications
        sms_result = self._test_sms_notifications(incident)
        notification_results.append(sms_result)
        
        # Test dashboard alerts
        dashboard_result = self._test_dashboard_notifications(incident)
        notification_results.append(dashboard_result)
        
        # Test escalation procedures
        escalation_result = self._test_escalation_procedures(incident)
        notification_results.append(escalation_result)
        
        return notification_results
    
    def test_business_continuity_response(self, incident: SecurityIncident) -> Dict[str, Any]:
        """
        Test business continuity response procedures.
        
        Args:
            incident: Security incident affecting business operations
            
        Returns:
            Business continuity test results
        """
        continuity_test = {
            "test_id": f"BC_{incident.incident_id}",
            "incident_id": incident.incident_id,
            "incident_type": incident.incident_type.value,
            "severity": incident.severity.value,
            "tests_performed": [],
            "overall_success": False,
            "critical_failures": [],
            "recommendations": []
        }
        
        # Test backup system activation
        backup_test = self._test_backup_activation(incident)
        continuity_test["tests_performed"].append(backup_test)
        
        # Test failover procedures
        failover_test = self._test_failover_procedures(incident)
        continuity_test["tests_performed"].append(failover_test)
        
        # Test data recovery capabilities
        recovery_test = self._test_data_recovery(incident)
        continuity_test["tests_performed"].append(recovery_test)
        
        # Test communication continuity
        communication_test = self._test_communication_continuity(incident)
        continuity_test["tests_performed"].append(communication_test)
        
        # Assess overall success
        successful_tests = len([t for t in continuity_test["tests_performed"] if t["success"]])
        total_tests = len(continuity_test["tests_performed"])
        continuity_test["overall_success"] = (successful_tests / total_tests) >= 0.8
        
        # Identify critical failures
        for test in continuity_test["tests_performed"]:
            if not test["success"] and test["criticality"] == "high":
                continuity_test["critical_failures"].append(test["test_name"])
        
        # Generate recommendations
        if not continuity_test["overall_success"]:
            continuity_test["recommendations"].append("Review and update business continuity procedures")
        if continuity_test["critical_failures"]:
            continuity_test["recommendations"].append("Address critical business continuity failures immediately")
        
        return continuity_test
    
    def test_post_incident_procedures(self, incident: SecurityIncident) -> Dict[str, Any]:
        """
        Test post-incident analysis and learning procedures.
        
        Args:
            incident: Completed security incident
            
        Returns:
            Post-incident procedure test results
        """
        post_incident_test = {
            "test_id": f"POST_{incident.incident_id}",
            "incident_id": incident.incident_id,
            "completion_date": datetime.now().isoformat(),
            "required_activities": [],
            "completed_activities": [],
            "gaps_identified": [],
            "lessons_learned": [],
            "improvement_actions": [],
            "compliance_met": False
        }
        
        # Required post-incident activities
        required_activities = [
            "root_cause_analysis",
            "timeline_documentation",
            "impact_assessment",
            "response_effectiveness_review",
            "lessons_learned_documentation",
            "process_improvement_recommendations",
            "stakeholder_debrief",
            "documentation_archival"
        ]
        
        post_incident_test["required_activities"] = required_activities
        
        # Simulate checking completed activities
        completed_activities = self._simulate_post_incident_activities(incident)
        post_incident_test["completed_activities"] = completed_activities
        
        # Identify gaps
        gaps = [activity for activity in required_activities if activity not in completed_activities]
        post_incident_test["gaps_identified"] = gaps
        
        # Check lessons learned documentation
        if len(incident.lessons_learned) > 0:
            post_incident_test["lessons_learned"] = incident.lessons_learned
        else:
            post_incident_test["gaps_identified"].append("lessons_learned_not_documented")
        
        # Generate improvement actions
        if gaps:
            post_incident_test["improvement_actions"].append("Complete missing post-incident activities")
        if not incident.root_cause:
            post_incident_test["improvement_actions"].append("Conduct thorough root cause analysis")
        
        # Assess compliance
        post_incident_test["compliance_met"] = len(gaps) == 0 and len(incident.lessons_learned) > 0
        
        return post_incident_test
    
    def conduct_comprehensive_incident_response_assessment(self, 
                                                         incidents: List[SecurityIncident],
                                                         security_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Conduct comprehensive incident response capability assessment.
        
        Args:
            incidents: List of security incidents to assess
            security_events: List of security events for detection testing
            
        Returns:
            Comprehensive incident response assessment results
        """
        assessment_id = f"ir_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Test incident detection
        detection_alerts = self.test_incident_detection_capabilities(security_events)
        
        # Test incident response procedures for each incident
        all_response_tests = []
        all_notification_tests = []
        all_continuity_tests = []
        all_post_incident_tests = []
        
        for incident in incidents:
            response_tests = self.test_incident_response_procedures(incident)
            all_response_tests.extend(response_tests)
            
            notification_tests = self.test_notification_systems(incident)
            all_notification_tests.extend(notification_tests)
            
            continuity_test = self.test_business_continuity_response(incident)
            all_continuity_tests.append(continuity_test)
            
            post_incident_test = self.test_post_incident_procedures(incident)
            all_post_incident_tests.append(post_incident_test)
        
        # Calculate metrics
        metrics = self._calculate_incident_response_metrics(
            detection_alerts, all_response_tests, all_notification_tests, 
            all_continuity_tests, all_post_incident_tests
        )
        
        # Assess maturity level
        maturity_level = self._assess_incident_response_maturity(metrics)
        
        # Generate recommendations
        recommendations = self._generate_incident_response_recommendations(
            metrics, all_response_tests, all_continuity_tests
        )
        
        return {
            "assessment_id": assessment_id,
            "assessment_date": datetime.now().isoformat(),
            "incidents_assessed": len(incidents),
            "security_events_tested": len(security_events),
            "detection_alerts_generated": len(detection_alerts),
            "response_tests_performed": len(all_response_tests),
            "notification_tests_performed": len(all_notification_tests),
            "continuity_tests_performed": len(all_continuity_tests),
            "post_incident_tests_performed": len(all_post_incident_tests),
            "metrics": metrics,
            "maturity_level": maturity_level,
            "recommendations": recommendations,
            "detailed_results": {
                "detection_alerts": [alert.__dict__ for alert in detection_alerts],
                "response_tests": [test.__dict__ for test in all_response_tests],
                "notification_tests": all_notification_tests,
                "continuity_tests": all_continuity_tests,
                "post_incident_tests": all_post_incident_tests
            },
            "compliance_assessment": self._assess_incident_response_compliance(
                all_response_tests, all_post_incident_tests
            )
        }
    
    def _test_response_time(self, incident: SecurityIncident) -> IncidentResponseTest:
        """Test incident response time compliance."""
        severity = incident.severity
        required_time = self.response_time_requirements[severity]["detection_to_response"]
        
        # Calculate actual response time
        detection_time = datetime.fromisoformat(incident.detection_timestamp.replace("Z", "+00:00"))
        reported_time = datetime.fromisoformat(incident.reported_timestamp.replace("Z", "+00:00"))
        actual_response_time = int((reported_time - detection_time).total_seconds() / 60)
        
        test_passed = actual_response_time <= required_time
        
        return IncidentResponseTest(
            test_id=f"RT_{incident.incident_id}",
            test_name="Response Time Compliance",
            incident_scenario={"incident_id": incident.incident_id, "severity": severity.value},
            expected_response_time=required_time,
            expected_actions=["immediate_response_team_activation"],
            expected_notifications=["security_team", "incident_commander"],
            actual_response_time=actual_response_time,
            actual_actions=["response_team_activated"],
            actual_notifications=incident.stakeholders_notified,
            test_passed=test_passed,
            gaps_identified=["slow_response_time"] if not test_passed else [],
            recommendations=["Improve detection and alerting systems"] if not test_passed else []
        )
    
    def _test_containment_actions(self, incident: SecurityIncident) -> IncidentResponseTest:
        """Test incident containment actions."""
        required_actions = self.required_response_actions.get(incident.incident_type, [])
        
        # Check if required containment actions were taken
        actions_taken = set(incident.containment_actions)
        required_containment = set([action for action in required_actions if "containment" in action or "isolation" in action])
        
        missing_actions = required_containment - actions_taken
        test_passed = len(missing_actions) == 0
        
        return IncidentResponseTest(
            test_id=f"CT_{incident.incident_id}",
            test_name="Containment Actions",
            incident_scenario={"incident_type": incident.incident_type.value},
            expected_response_time=self.response_time_requirements[incident.severity]["initial_containment"],
            expected_actions=list(required_containment),
            expected_notifications=["affected_system_owners"],
            actual_response_time=None,  # Not time-based test
            actual_actions=incident.containment_actions,
            actual_notifications=incident.stakeholders_notified,
            test_passed=test_passed,
            gaps_identified=list(missing_actions),
            recommendations=["Implement missing containment procedures"] if missing_actions else []
        )
    
    def _test_stakeholder_notifications(self, incident: SecurityIncident) -> IncidentResponseTest:
        """Test stakeholder notification procedures."""
        required_stakeholders = ["incident_commander", "security_team", "legal_counsel"]
        if incident.severity in [IncidentSeverity.CRITICAL, IncidentSeverity.HIGH]:
            required_stakeholders.extend(["ciso", "executive_team"])
        
        notified_stakeholders = set(incident.stakeholders_notified)
        required_stakeholders_set = set(required_stakeholders)
        
        missing_notifications = required_stakeholders_set - notified_stakeholders
        test_passed = len(missing_notifications) == 0
        
        return IncidentResponseTest(
            test_id=f"SN_{incident.incident_id}",
            test_name="Stakeholder Notifications",
            incident_scenario={"severity": incident.severity.value},
            expected_response_time=self.response_time_requirements[incident.severity]["stakeholder_notification"],
            expected_actions=["stakeholder_notification"],
            expected_notifications=required_stakeholders,
            actual_response_time=None,
            actual_actions=["notifications_sent"],
            actual_notifications=incident.stakeholders_notified,
            test_passed=test_passed,
            gaps_identified=list(missing_notifications),
            recommendations=["Update stakeholder notification procedures"] if missing_notifications else []
        )
    
    def _test_evidence_collection(self, incident: SecurityIncident) -> IncidentResponseTest:
        """Test evidence collection procedures."""
        required_evidence_types = ["system_logs", "network_logs", "access_logs", "timeline_documentation"]
        
        collected_evidence_types = set([evidence.get("type", "") for evidence in incident.evidence_collected])
        required_evidence_set = set(required_evidence_types)
        
        missing_evidence = required_evidence_set - collected_evidence_types
        test_passed = len(missing_evidence) == 0
        
        return IncidentResponseTest(
            test_id=f"EC_{incident.incident_id}",
            test_name="Evidence Collection",
            incident_scenario={"incident_type": incident.incident_type.value},
            expected_response_time=60,  # Evidence collection should start within 1 hour
            expected_actions=["evidence_collection", "chain_of_custody"],
            expected_notifications=["forensics_team"],
            actual_response_time=None,
            actual_actions=["evidence_collected"],
            actual_notifications=incident.stakeholders_notified,
            test_passed=test_passed,
            gaps_identified=list(missing_evidence),
            recommendations=["Improve evidence collection procedures"] if missing_evidence else []
        )
    
    def _test_regulatory_compliance(self, incident: SecurityIncident) -> IncidentResponseTest:
        """Test regulatory compliance requirements."""
        test_passed = True
        gaps = []
        
        if incident.regulatory_notification_required:
            if not incident.regulatory_notifications_sent:
                test_passed = False
                gaps.append("regulatory_notifications_not_sent")
            
            # Check notification timing
            for notification in incident.regulatory_notifications_sent:
                notification_time = datetime.fromisoformat(notification.get("sent_timestamp", "").replace("Z", "+00:00"))
                incident_time = datetime.fromisoformat(incident.detection_timestamp.replace("Z", "+00:00"))
                hours_elapsed = (notification_time - incident_time).total_seconds() / 3600
                
                if hours_elapsed > 72:  # Most regulations require 72-hour notification
                    test_passed = False
                    gaps.append("late_regulatory_notification")
        
        return IncidentResponseTest(
            test_id=f"RC_{incident.incident_id}",
            test_name="Regulatory Compliance",
            incident_scenario={"regulatory_required": incident.regulatory_notification_required},
            expected_response_time=4320,  # 72 hours in minutes
            expected_actions=["regulatory_assessment", "notification_preparation", "notification_submission"],
            expected_notifications=["legal_counsel", "privacy_officer"],
            actual_response_time=None,
            actual_actions=["regulatory_notifications_sent"] if incident.regulatory_notifications_sent else [],
            actual_notifications=incident.stakeholders_notified,
            test_passed=test_passed,
            gaps_identified=gaps,
            recommendations=["Improve regulatory notification procedures"] if gaps else []
        )
    
    def _test_email_notifications(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Test email notification system."""
        return {
            "notification_type": "email",
            "test_id": f"EMAIL_{incident.incident_id}",
            "recipients_expected": 5,
            "recipients_notified": len(incident.stakeholders_notified),
            "delivery_success_rate": 0.9,  # Simulated
            "average_delivery_time_seconds": 30,
            "success": len(incident.stakeholders_notified) >= 3,
            "issues": [] if len(incident.stakeholders_notified) >= 3 else ["insufficient_recipients"]
        }
    
    def _test_sms_notifications(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Test SMS notification system."""
        critical_incident = incident.severity == IncidentSeverity.CRITICAL
        
        return {
            "notification_type": "sms",
            "test_id": f"SMS_{incident.incident_id}",
            "required_for_severity": critical_incident,
            "sms_sent": critical_incident,  # Simulate SMS for critical incidents
            "delivery_success_rate": 0.95,
            "average_delivery_time_seconds": 10,
            "success": True if not critical_incident else critical_incident,
            "issues": [] if not critical_incident else ([] if critical_incident else ["sms_not_sent_for_critical"])
        }
    
    def _test_dashboard_notifications(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Test dashboard notification system."""
        return {
            "notification_type": "dashboard",
            "test_id": f"DASH_{incident.incident_id}",
            "alert_displayed": True,
            "alert_visibility": "high",
            "auto_refresh_enabled": True,
            "escalation_indicators": incident.severity in [IncidentSeverity.CRITICAL, IncidentSeverity.HIGH],
            "success": True,
            "issues": []
        }
    
    def _test_escalation_procedures(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Test incident escalation procedures."""
        should_escalate = incident.severity in [IncidentSeverity.CRITICAL, IncidentSeverity.HIGH]
        escalated = "executive_team" in incident.stakeholders_notified or "ciso" in incident.stakeholders_notified
        
        return {
            "notification_type": "escalation",
            "test_id": f"ESC_{incident.incident_id}",
            "escalation_required": should_escalate,
            "escalation_performed": escalated,
            "escalation_timeframe_met": True,  # Simulated
            "success": not should_escalate or escalated,
            "issues": ["escalation_not_performed"] if should_escalate and not escalated else []
        }
    
    def _test_backup_activation(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Test backup system activation."""
        requires_backup = incident.incident_type in [IncidentType.RANSOMWARE, IncidentType.SYSTEM_COMPROMISE]
        
        return {
            "test_name": "backup_activation",
            "required": requires_backup,
            "success": requires_backup,  # Simulate successful backup activation
            "activation_time_minutes": 15 if requires_backup else 0,
            "criticality": "high" if requires_backup else "low",
            "issues": [] if requires_backup else []
        }
    
    def _test_failover_procedures(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Test system failover procedures."""
        requires_failover = incident.severity == IncidentSeverity.CRITICAL
        
        return {
            "test_name": "failover_procedures",
            "required": requires_failover,
            "success": requires_failover,  # Simulate successful failover
            "failover_time_minutes": 10 if requires_failover else 0,
            "criticality": "high" if requires_failover else "medium",
            "issues": [] if requires_failover else []
        }
    
    def _test_data_recovery(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Test data recovery capabilities."""
        requires_recovery = incident.incident_type in [IncidentType.DATA_BREACH, IncidentType.RANSOMWARE]
        
        return {
            "test_name": "data_recovery",
            "required": requires_recovery,
            "success": True,  # Simulate successful recovery
            "recovery_time_hours": 2 if requires_recovery else 0,
            "data_integrity_verified": True,
            "criticality": "high" if requires_recovery else "low",
            "issues": []
        }
    
    def _test_communication_continuity(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Test communication system continuity."""
        return {
            "test_name": "communication_continuity",
            "required": True,
            "success": True,
            "backup_channels_available": True,
            "stakeholder_reachability": 0.95,
            "criticality": "medium",
            "issues": []
        }
    
    def _simulate_post_incident_activities(self, incident: SecurityIncident) -> List[str]:
        """Simulate post-incident activities completion."""
        # Simulate based on incident characteristics
        completed = ["timeline_documentation", "impact_assessment"]
        
        if incident.root_cause:
            completed.append("root_cause_analysis")
        
        if len(incident.lessons_learned) > 0:
            completed.append("lessons_learned_documentation")
        
        if incident.status == IncidentStatus.CLOSED:
            completed.extend(["response_effectiveness_review", "documentation_archival"])
        
        return completed
    
    def _calculate_incident_response_metrics(self, detection_alerts: List[IncidentAlert],
                                           response_tests: List[IncidentResponseTest],
                                           notification_tests: List[Dict[str, Any]],
                                           continuity_tests: List[Dict[str, Any]],
                                           post_incident_tests: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate incident response metrics."""
        metrics = {}
        
        # Detection effectiveness
        total_alerts = len(detection_alerts)
        escalated_alerts = len([a for a in detection_alerts if a.escalated_to_incident])
        metrics["detection_effectiveness"] = escalated_alerts / total_alerts if total_alerts > 0 else 0.0
        
        # Response time compliance
        response_time_tests = [t for t in response_tests if t.test_name == "Response Time Compliance"]
        passed_response_tests = len([t for t in response_time_tests if t.test_passed])
        metrics["response_time_compliance"] = passed_response_tests / len(response_time_tests) if response_time_tests else 0.0
        
        # Overall response effectiveness
        total_response_tests = len(response_tests)
        passed_response_tests = len([t for t in response_tests if t.test_passed])
        metrics["response_effectiveness"] = passed_response_tests / total_response_tests if total_response_tests else 0.0
        
        # Notification success rate
        successful_notifications = len([t for t in notification_tests if t.get("success", False)])
        metrics["notification_success_rate"] = successful_notifications / len(notification_tests) if notification_tests else 0.0
        
        # Business continuity readiness
        successful_continuity = len([t for t in continuity_tests if t.get("overall_success", False)])
        metrics["business_continuity_readiness"] = successful_continuity / len(continuity_tests) if continuity_tests else 0.0
        
        # Post-incident compliance
        compliant_post_incident = len([t for t in post_incident_tests if t.get("compliance_met", False)])
        metrics["post_incident_compliance"] = compliant_post_incident / len(post_incident_tests) if post_incident_tests else 0.0
        
        return metrics
    
    def _assess_incident_response_maturity(self, metrics: Dict[str, float]) -> str:
        """Assess incident response maturity level."""
        overall_score = sum(metrics.values()) / len(metrics) if metrics else 0.0
        
        if overall_score >= 0.9:
            return "optimised"
        elif overall_score >= 0.8:
            return "managed"
        elif overall_score >= 0.6:
            return "defined"
        elif overall_score >= 0.4:
            return "repeatable"
        else:
            return "initial"
    
    def _generate_incident_response_recommendations(self, metrics: Dict[str, float],
                                                  response_tests: List[IncidentResponseTest],
                                                  continuity_tests: List[Dict[str, Any]]) -> List[str]:
        """Generate incident response improvement recommendations."""
        recommendations = []
        
        if metrics.get("detection_effectiveness", 0) < 0.8:
            recommendations.append("Improve security monitoring and detection capabilities")
        
        if metrics.get("response_time_compliance", 0) < 0.9:
            recommendations.append("Reduce incident response times through automation and training")
        
        if metrics.get("notification_success_rate", 0) < 0.95:
            recommendations.append("Enhance notification systems reliability and redundancy")
        
        if metrics.get("business_continuity_readiness", 0) < 0.8:
            recommendations.append("Strengthen business continuity and disaster recovery procedures")
        
        if metrics.get("post_incident_compliance", 0) < 0.9:
            recommendations.append("Improve post-incident analysis and documentation procedures")
        
        # Specific recommendations based on test failures
        failed_tests = [t for t in response_tests if not t.test_passed]
        if len(failed_tests) > len(response_tests) * 0.2:  # More than 20% failures
            recommendations.append("Conduct incident response training and tabletop exercises")
        
        critical_continuity_failures = [t for t in continuity_tests if t.get("critical_failures")]
        if critical_continuity_failures:
            recommendations.append("Address critical business continuity gaps immediately")
        
        return recommendations
    
    def _assess_incident_response_compliance(self, response_tests: List[IncidentResponseTest],
                                           post_incident_tests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess incident response regulatory compliance."""
        regulatory_tests = [t for t in response_tests if t.test_name == "Regulatory Compliance"]
        regulatory_compliance = len([t for t in regulatory_tests if t.test_passed]) / len(regulatory_tests) if regulatory_tests else 0.0
        
        post_incident_compliance = len([t for t in post_incident_tests if t.get("compliance_met", False)]) / len(post_incident_tests) if post_incident_tests else 0.0
        
        overall_compliance = (regulatory_compliance + post_incident_compliance) / 2
        
        return {
            "regulatory_compliance_rate": regulatory_compliance,
            "post_incident_compliance_rate": post_incident_compliance,
            "overall_compliance_rate": overall_compliance,
            "compliance_level": "compliant" if overall_compliance >= 0.9 else "partially_compliant" if overall_compliance >= 0.7 else "non_compliant"
        }


# Test suite
class TestSecurityIncidentResponse:
    """Test suite for security incident response."""
    
    @pytest.fixture
    def incident_response_tester(self):
        """Create incident response tester instance."""
        return SecurityIncidentResponseTester()
    
    @pytest.fixture
    def test_security_events(self):
        """Test security events for detection testing."""
        return [
            {
                "event_type": "failed_login_attempts",
                "timestamp": datetime.now().isoformat(),
                "source_ip": "192.168.1.100",
                "user_id": "test_user",
                "attempt_count": 8
            },
            {
                "event_type": "data_access_anomaly",
                "timestamp": datetime.now().isoformat(),
                "user_id": "analyst_user",
                "records_accessed": 5000,
                "access_pattern": "bulk_download"
            },
            {
                "event_type": "privilege_escalation",
                "timestamp": datetime.now().isoformat(),
                "user_id": "regular_user",
                "privileges_granted": ["admin", "data_export"],
                "granted_by": "unknown",
                "approved": False
            }
        ]
    
    @pytest.fixture
    def test_security_incident(self):
        """Test security incident."""
        now = datetime.now()
        return SecurityIncident(
            incident_id="INC_001",
            incident_type=IncidentType.DATA_BREACH,
            severity=IncidentSeverity.HIGH,
            status=IncidentStatus.RECOVERY,
            title="Unauthorised Data Access",
            description="Unauthorised access to patient health records detected",
            affected_systems=["patient_database", "web_application"],
            affected_data_types=["patient_records", "health_information"],
            detection_timestamp=now.isoformat(),
            reported_timestamp=(now + timedelta(minutes=20)).isoformat(),
            response_team=["incident_commander", "security_analyst", "legal_counsel"],
            stakeholders_notified=["incident_commander", "security_team", "legal_counsel", "privacy_officer"],
            containment_actions=["immediate_containment", "system_isolation", "access_revocation"],
            eradication_actions=["vulnerability_patching", "malware_removal"],
            recovery_actions=["system_restoration", "monitoring_enhancement"],
            lessons_learned=["Improve access controls", "Enhance monitoring"],
            estimated_impact={"records_affected": 1500, "financial_impact": 50000},
            regulatory_notification_required=True,
            regulatory_notifications_sent=[{
                "authority": "OAIC",
                "sent_timestamp": (now + timedelta(hours=48)).isoformat(),
                "notification_type": "data_breach"
            }],
            evidence_collected=[
                {"type": "system_logs", "collected_by": "forensics_team"},
                {"type": "access_logs", "collected_by": "security_analyst"}
            ],
            timeline=[
                {"timestamp": now.isoformat(), "event": "Incident detected"},
                {"timestamp": (now + timedelta(minutes=20)).isoformat(), "event": "Incident reported"}
            ],
            root_cause="Insufficient access controls",
            resolution_timestamp=(now + timedelta(days=3)).isoformat()
        )
    
    def test_incident_detection_capabilities(self, incident_response_tester, test_security_events):
        """Test security incident detection capabilities."""
        alerts = incident_response_tester.test_incident_detection_capabilities(test_security_events)
        
        # Should generate alerts for suspicious events
        assert len(alerts) > 0, "Should generate security alerts"
        
        # Check for specific alert types
        alert_types = [alert.alert_type for alert in alerts]
        expected_types = ["brute_force_attempt", "abnormal_data_access", "privilege_escalation"]
        
        detected_types = [atype for atype in expected_types if atype in alert_types]
        assert len(detected_types) > 0, "Should detect specific threat patterns"
        
        # Verify alert structure
        for alert in alerts:
            assert alert.alert_id is not None
            assert alert.severity in [s for s in IncidentSeverity]
            assert isinstance(alert.indicators, dict)
            assert alert.escalated_to_incident in [True, False]
            
        # Check escalation logic
        high_severity_alerts = [a for a in alerts if a.severity in [IncidentSeverity.HIGH, IncidentSeverity.CRITICAL]]
        escalated_alerts = [a for a in alerts if a.escalated_to_incident]
        
        # High severity alerts should be escalated to incidents
        for alert in high_severity_alerts:
            if alert.alert_type == "abnormal_data_access":  # Should always escalate
                assert alert.escalated_to_incident is True
    
    def test_incident_response_procedures(self, incident_response_tester, test_security_incident):
        """Test incident response procedures."""
        response_tests = incident_response_tester.test_incident_response_procedures(test_security_incident)
        
        # Should perform multiple response tests
        assert len(response_tests) > 0, "Should perform incident response tests"
        
        # Check for required test types
        test_names = [test.test_name for test in response_tests]
        required_tests = ["Response Time Compliance", "Containment Actions", "Stakeholder Notifications"]
        
        for required_test in required_tests:
            assert required_test in test_names, f"Should include {required_test} test"
        
        # Verify test structure
        for test in response_tests:
            assert test.test_id is not None
            assert test.test_passed in [True, False]
            assert isinstance(test.gaps_identified, list)
            assert isinstance(test.recommendations, list)
        
        # Response time test should pass for reasonable timing
        response_time_test = next((t for t in response_tests if t.test_name == "Response Time Compliance"), None)
        if response_time_test and response_time_test.actual_response_time and response_time_test.actual_response_time <= 30:
            assert response_time_test.test_passed is True
    
    def test_notification_systems(self, incident_response_tester, test_security_incident):
        """Test incident notification systems."""
        notification_tests = incident_response_tester.test_notification_systems(test_security_incident)
        
        # Should test multiple notification channels
        assert len(notification_tests) > 0, "Should test notification systems"
        
        # Check for different notification types
        notification_types = [test.get("notification_type") for test in notification_tests]
        expected_types = ["email", "sms", "dashboard", "escalation"]
        
        for expected_type in expected_types:
            assert expected_type in notification_types, f"Should test {expected_type} notifications"
        
        # Verify notification test structure
        for test in notification_tests:
            assert "success" in test
            assert "issues" in test
            assert test["success"] in [True, False]
            
        # Critical incidents should trigger SMS notifications
        sms_test = next((t for t in notification_tests if t.get("notification_type") == "sms"), None)
        if sms_test and test_security_incident.severity == IncidentSeverity.CRITICAL:
            assert sms_test["sms_sent"] is True
    
    def test_business_continuity_response(self, incident_response_tester, test_security_incident):
        """Test business continuity response procedures."""
        continuity_test = incident_response_tester.test_business_continuity_response(test_security_incident)
        
        # Verify test structure
        assert "test_id" in continuity_test
        assert "tests_performed" in continuity_test
        assert "overall_success" in continuity_test
        assert "critical_failures" in continuity_test
        assert "recommendations" in continuity_test
        
        # Should perform multiple continuity tests
        assert len(continuity_test["tests_performed"]) > 0, "Should perform business continuity tests"
        
        # Check for required test types
        test_names = [test["test_name"] for test in continuity_test["tests_performed"]]
        expected_tests = ["backup_activation", "failover_procedures", "data_recovery", "communication_continuity"]
        
        for expected_test in expected_tests:
            assert expected_test in test_names, f"Should include {expected_test} test"
        
        # Verify individual test structure
        for test in continuity_test["tests_performed"]:
            assert "test_name" in test
            assert "success" in test
            assert "criticality" in test
            assert test["success"] in [True, False]
            assert test["criticality"] in ["low", "medium", "high"]
    
    def test_post_incident_procedures(self, incident_response_tester, test_security_incident):
        """Test post-incident analysis procedures."""
        post_incident_test = incident_response_tester.test_post_incident_procedures(test_security_incident)
        
        # Verify test structure
        required_keys = [
            "test_id", "incident_id", "required_activities", "completed_activities",
            "gaps_identified", "lessons_learned", "improvement_actions", "compliance_met"
        ]
        
        for key in required_keys:
            assert key in post_incident_test, f"Post-incident test should include {key}"
        
        # Should have defined required activities
        assert len(post_incident_test["required_activities"]) > 0, "Should have required post-incident activities"
        
        # Should track completed activities
        assert isinstance(post_incident_test["completed_activities"], list)
        
        # Should identify gaps if activities are missing
        required_activities = set(post_incident_test["required_activities"])
        completed_activities = set(post_incident_test["completed_activities"])
        expected_gaps = required_activities - completed_activities
        actual_gaps = set(post_incident_test["gaps_identified"])
        
        # Gaps should be properly identified
        for gap in expected_gaps:
            assert gap in actual_gaps or gap.replace("_", "") in str(actual_gaps), f"Should identify gap: {gap}"
        
        # Compliance should be met if no gaps and lessons learned documented
        if len(post_incident_test["gaps_identified"]) == 0 and len(test_security_incident.lessons_learned) > 0:
            assert post_incident_test["compliance_met"] is True
    
    def test_comprehensive_incident_response_assessment(self, incident_response_tester, 
                                                      test_security_incident, test_security_events):
        """Test comprehensive incident response assessment."""
        incidents = [test_security_incident]
        assessment = incident_response_tester.conduct_comprehensive_incident_response_assessment(
            incidents, test_security_events
        )
        
        # Verify assessment structure
        required_keys = [
            "assessment_id", "assessment_date", "incidents_assessed", "security_events_tested",
            "detection_alerts_generated", "response_tests_performed", "notification_tests_performed",
            "continuity_tests_performed", "post_incident_tests_performed", "metrics",
            "maturity_level", "recommendations", "detailed_results", "compliance_assessment"
        ]
        
        for key in required_keys:
            assert key in assessment, f"Assessment should include {key}"
        
        # Should assess the correct number of incidents and events
        assert assessment["incidents_assessed"] == len(incidents)
        assert assessment["security_events_tested"] == len(test_security_events)
        
        # Should generate alerts and perform tests
        assert assessment["detection_alerts_generated"] > 0, "Should generate detection alerts"
        assert assessment["response_tests_performed"] > 0, "Should perform response tests"
        assert assessment["notification_tests_performed"] > 0, "Should perform notification tests"
        
        # Should calculate metrics
        metrics = assessment["metrics"]
        expected_metrics = [
            "detection_effectiveness", "response_time_compliance", "response_effectiveness",
            "notification_success_rate", "business_continuity_readiness", "post_incident_compliance"
        ]
        
        for metric in expected_metrics:
            assert metric in metrics, f"Should calculate {metric} metric"
            assert 0.0 <= metrics[metric] <= 1.0, f"Metric {metric} should be between 0 and 1"
        
        # Should determine maturity level
        maturity_levels = ["initial", "repeatable", "defined", "managed", "optimised"]
        assert assessment["maturity_level"] in maturity_levels, "Should determine valid maturity level"
        
        # Should provide recommendations
        assert len(assessment["recommendations"]) >= 0, "Should provide recommendations"
        
        # Should assess compliance
        compliance = assessment["compliance_assessment"]
        assert "overall_compliance_rate" in compliance
        assert "compliance_level" in compliance
        assert compliance["compliance_level"] in ["compliant", "partially_compliant", "non_compliant"]
    
    def test_incident_response_time_requirements(self, incident_response_tester):
        """Test incident response time requirements configuration."""
        requirements = incident_response_tester.response_time_requirements
        
        # Should have requirements for all severity levels
        for severity in IncidentSeverity:
            if severity != IncidentSeverity.INFO:  # INFO may not have requirements
                assert severity in requirements, f"Should have requirements for {severity}"
                
                severity_reqs = requirements[severity]
                required_timeframes = ["detection_to_response", "initial_containment", "stakeholder_notification"]
                
                for timeframe in required_timeframes:
                    assert timeframe in severity_reqs, f"Should have {timeframe} requirement for {severity}"
                    assert severity_reqs[timeframe] > 0, f"{timeframe} should be positive for {severity}"
        
        # Critical incidents should have the shortest response times
        critical_reqs = requirements[IncidentSeverity.CRITICAL]
        high_reqs = requirements[IncidentSeverity.HIGH]
        
        assert critical_reqs["detection_to_response"] <= high_reqs["detection_to_response"], "Critical should have faster response than high"
        assert critical_reqs["initial_containment"] <= high_reqs["initial_containment"], "Critical should have faster containment than high"
    
    def test_required_response_actions_configuration(self, incident_response_tester):
        """Test required response actions configuration."""
        actions = incident_response_tester.required_response_actions
        
        # Should have actions defined for key incident types
        key_incident_types = [
            IncidentType.DATA_BREACH,
            IncidentType.UNAUTHORISED_ACCESS,
            IncidentType.MALWARE_INFECTION,
            IncidentType.RANSOMWARE
        ]
        
        for incident_type in key_incident_types:
            assert incident_type in actions, f"Should have required actions for {incident_type}"
            assert len(actions[incident_type]) > 0, f"Should have actions defined for {incident_type}"
        
        # Data breach should include regulatory notification
        data_breach_actions = actions[IncidentType.DATA_BREACH]
        assert "regulatory_notification" in data_breach_actions, "Data breach should require regulatory notification"
        assert "stakeholder_notification" in data_breach_actions, "Data breach should require stakeholder notification"
        
        # Ransomware should include isolation and backup restoration
        ransomware_actions = actions[IncidentType.RANSOMWARE]
        assert any("isolation" in action for action in ransomware_actions), "Ransomware should require isolation"
        assert any("backup" in action for action in ransomware_actions), "Ransomware should require backup restoration"
    
    def test_regulatory_requirements_configuration(self, incident_response_tester):
        """Test regulatory requirements configuration."""
        regulations = incident_response_tester.regulatory_requirements
        
        # Should include Australian regulatory frameworks
        expected_regulations = [
            "notifiable_data_breaches_scheme",
            "health_records_act",
            "therapeutic_goods_administration"
        ]
        
        for regulation in expected_regulations:
            assert regulation in regulations, f"Should include {regulation}"
            
            reg_info = regulations[regulation]
            assert "notification_timeframe_hours" in reg_info
            assert "authority" in reg_info
            assert "triggers" in reg_info
            
            # Notification timeframe should be reasonable
            assert 0 < reg_info["notification_timeframe_hours"] <= 168, "Notification timeframe should be 1-168 hours"
            
            # Should have trigger conditions
            assert len(reg_info["triggers"]) > 0, f"Should have trigger conditions for {regulation}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])