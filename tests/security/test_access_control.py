"""
Access Control and Authentication Security Testing

Comprehensive testing suite for access control and authentication mechanisms including:
- Role-based access control (RBAC) validation
- Authentication mechanism testing
- Authorisation boundary testing
- Session management security testing
- Multi-factor authentication validation
- Privilege escalation prevention testing

This test suite ensures the platform implements robust access control mechanisms
that protect sensitive health data from unauthorised access.
"""

import json
import pytest
import hashlib
import secrets
import jwt
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


class UserRole(Enum):
    """User roles in the health analytics system."""
    SUPER_ADMIN = "super_admin"
    SYSTEM_ADMIN = "system_admin"
    DATA_ADMIN = "data_admin"
    HEALTH_ANALYST = "health_analyst"
    RESEARCHER = "researcher"
    VIEWER = "viewer"
    GUEST = "guest"


class Permission(Enum):
    """System permissions."""
    READ_ALL_DATA = "read_all_data"
    READ_AGGREGATED_DATA = "read_aggregated_data"
    READ_DEIDENTIFIED_DATA = "read_deidentified_data"
    WRITE_DATA = "write_data"
    DELETE_DATA = "delete_data"
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    EXPORT_DATA = "export_data"
    ACCESS_RAW_DATA = "access_raw_data"
    SYSTEM_ADMINISTRATION = "system_administration"
    AUDIT_ACCESS = "audit_access"
    PRIVACY_ADMINISTRATION = "privacy_administration"


class AuthenticationMethod(Enum):
    """Authentication methods."""
    PASSWORD = "password"
    MFA_TOTP = "mfa_totp"
    MFA_SMS = "mfa_sms"
    CERTIFICATE = "certificate"
    SSO = "sso"
    API_KEY = "api_key"


class SessionStatus(Enum):
    """Session status."""
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    LOCKED = "locked"


@dataclass
class User:
    """User account information."""
    user_id: str
    username: str
    email: str
    roles: List[UserRole]
    permissions: List[Permission]
    account_status: str
    created_date: str
    last_login: Optional[str] = None
    failed_login_attempts: int = 0
    account_locked: bool = False
    mfa_enabled: bool = False
    password_last_changed: Optional[str] = None


@dataclass
class Session:
    """User session information."""
    session_id: str
    user_id: str
    created_timestamp: str
    last_activity: str
    ip_address: str
    user_agent: str
    status: SessionStatus
    expires_at: str
    mfa_verified: bool = False
    privileged_actions: List[str] = field(default_factory=list)


@dataclass
class AccessAttempt:
    """Access attempt record."""
    attempt_id: str
    user_id: Optional[str]
    username: Optional[str]
    resource: str
    action: str
    timestamp: str
    ip_address: str
    user_agent: str
    authentication_method: AuthenticationMethod
    success: bool
    failure_reason: Optional[str] = None
    risk_score: float = 0.0


@dataclass
class AccessControlViolation:
    """Access control security violation."""
    violation_type: str
    severity: str  # "critical", "high", "medium", "low"
    description: str
    affected_resources: List[str]
    affected_users: List[str]
    evidence: List[Dict[str, Any]]
    remediation_required: bool
    remediation_timeframe: str
    security_impact: str
    compliance_impact: str
    details: Dict[str, Any]


class AccessControlTester:
    """Access control and authentication security tester."""
    
    def __init__(self):
        """Initialise access control tester."""
        self.logger = logger.bind(component="access_control_tester")
        
        # Role-based permissions matrix
        self.role_permissions = {
            UserRole.SUPER_ADMIN: [
                Permission.READ_ALL_DATA,
                Permission.WRITE_DATA,
                Permission.DELETE_DATA,
                Permission.MANAGE_USERS,
                Permission.MANAGE_ROLES,
                Permission.EXPORT_DATA,
                Permission.ACCESS_RAW_DATA,
                Permission.SYSTEM_ADMINISTRATION,
                Permission.AUDIT_ACCESS,
                Permission.PRIVACY_ADMINISTRATION
            ],
            UserRole.SYSTEM_ADMIN: [
                Permission.READ_ALL_DATA,
                Permission.WRITE_DATA,
                Permission.MANAGE_USERS,
                Permission.EXPORT_DATA,
                Permission.SYSTEM_ADMINISTRATION,
                Permission.AUDIT_ACCESS
            ],
            UserRole.DATA_ADMIN: [
                Permission.READ_ALL_DATA,
                Permission.WRITE_DATA,
                Permission.EXPORT_DATA,
                Permission.ACCESS_RAW_DATA,
                Permission.PRIVACY_ADMINISTRATION
            ],
            UserRole.HEALTH_ANALYST: [
                Permission.READ_AGGREGATED_DATA,
                Permission.READ_DEIDENTIFIED_DATA,
                Permission.EXPORT_DATA
            ],
            UserRole.RESEARCHER: [
                Permission.READ_AGGREGATED_DATA,
                Permission.READ_DEIDENTIFIED_DATA
            ],
            UserRole.VIEWER: [
                Permission.READ_AGGREGATED_DATA
            ],
            UserRole.GUEST: []
        }
        
        # Sensitive resources requiring elevated permissions
        self.sensitive_resources = {
            "raw_health_data": [Permission.ACCESS_RAW_DATA],
            "patient_identifiers": [Permission.ACCESS_RAW_DATA, Permission.PRIVACY_ADMINISTRATION],
            "user_management": [Permission.MANAGE_USERS],
            "system_configuration": [Permission.SYSTEM_ADMINISTRATION],
            "audit_logs": [Permission.AUDIT_ACCESS],
            "export_functions": [Permission.EXPORT_DATA],
            "data_deletion": [Permission.DELETE_DATA]
        }
        
        # Security thresholds
        self.security_thresholds = {
            "max_failed_login_attempts": 5,
            "account_lockout_duration_minutes": 30,
            "session_timeout_minutes": 30,
            "privileged_session_timeout_minutes": 15,
            "password_min_length": 12,
            "password_complexity_required": True,
            "mfa_required_for_privileged": True,
            "max_concurrent_sessions": 3
        }
    
    def test_role_based_access_control(self, users: List[User], access_attempts: List[AccessAttempt]) -> List[AccessControlViolation]:
        """
        Test role-based access control implementation.
        
        Args:
            users: List of user accounts
            access_attempts: List of access attempts to analyse
            
        Returns:
            List of RBAC violations
        """
        violations = []
        
        # Create user lookup
        user_lookup = {user.user_id: user for user in users}
        
        for attempt in access_attempts:
            if not attempt.success:
                continue  # Only analyse successful attempts for RBAC violations
            
            user = user_lookup.get(attempt.user_id)
            if not user:
                violations.append(AccessControlViolation(
                    violation_type="unauthorised_access_unknown_user",
                    severity="critical",
                    description=f"Successful access by unknown user: {attempt.user_id}",
                    affected_resources=[attempt.resource],
                    affected_users=[attempt.user_id or "unknown"],
                    evidence=[{"attempt": attempt.__dict__}],
                    remediation_required=True,
                    remediation_timeframe="immediate",
                    security_impact="high",
                    compliance_impact="high",
                    details={"unknown_user_id": attempt.user_id}
                ))
                continue
            
            # Check if user has required permissions for resource
            required_permissions = self.sensitive_resources.get(attempt.resource, [])
            if required_permissions:
                user_permissions = set(user.permissions)
                missing_permissions = [perm for perm in required_permissions if perm not in user_permissions]
                
                if missing_permissions:
                    violations.append(AccessControlViolation(
                        violation_type="insufficient_permissions",
                        severity="high",
                        description=f"User {user.username} accessed {attempt.resource} without required permissions",
                        affected_resources=[attempt.resource],
                        affected_users=[user.user_id],
                        evidence=[{
                            "user": user.username,
                            "resource": attempt.resource,
                            "missing_permissions": [perm.value for perm in missing_permissions],
                            "user_permissions": [perm.value for perm in user.permissions]
                        }],
                        remediation_required=True,
                        remediation_timeframe="24 hours",
                        security_impact="high",
                        compliance_impact="medium",
                        details={"required_permissions": required_permissions, "missing_permissions": missing_permissions}
                    ))
            
            # Check role-permission consistency
            expected_permissions = set()
            for role in user.roles:
                expected_permissions.update(self.role_permissions.get(role, []))
            
            actual_permissions = set(user.permissions)
            excessive_permissions = actual_permissions - expected_permissions
            
            if excessive_permissions:
                violations.append(AccessControlViolation(
                    violation_type="excessive_permissions",
                    severity="medium",
                    description=f"User {user.username} has permissions beyond their role",
                    affected_resources=["user_account"],
                    affected_users=[user.user_id],
                    evidence=[{
                        "user": user.username,
                        "roles": [role.value for role in user.roles],
                        "excessive_permissions": [perm.value for perm in excessive_permissions]
                    }],
                    remediation_required=True,
                    remediation_timeframe="7 days",
                    security_impact="medium",
                    compliance_impact="medium",
                    details={"excessive_permissions": excessive_permissions}
                ))
        
        return violations
    
    def test_authentication_security(self, users: List[User], access_attempts: List[AccessAttempt]) -> List[AccessControlViolation]:
        """
        Test authentication mechanism security.
        
        Args:
            users: List of user accounts
            access_attempts: List of access attempts
            
        Returns:
            List of authentication security violations
        """
        violations = []
        
        # Test password policy compliance
        for user in users:
            # Check account lockout policy
            if user.failed_login_attempts >= self.security_thresholds["max_failed_login_attempts"] and not user.account_locked:
                violations.append(AccessControlViolation(
                    violation_type="account_lockout_policy_violation",
                    severity="medium",
                    description=f"User {user.username} exceeded failed login threshold but account not locked",
                    affected_resources=["authentication_system"],
                    affected_users=[user.user_id],
                    evidence=[{"user": user.username, "failed_attempts": user.failed_login_attempts}],
                    remediation_required=True,
                    remediation_timeframe="immediate",
                    security_impact="medium",
                    compliance_impact="low",
                    details={"failed_attempts": user.failed_login_attempts, "threshold": self.security_thresholds["max_failed_login_attempts"]}
                ))
            
            # Check MFA requirement for privileged users
            privileged_roles = [UserRole.SUPER_ADMIN, UserRole.SYSTEM_ADMIN, UserRole.DATA_ADMIN]
            if any(role in user.roles for role in privileged_roles) and not user.mfa_enabled:
                violations.append(AccessControlViolation(
                    violation_type="mfa_requirement_violation",
                    severity="high",
                    description=f"Privileged user {user.username} does not have MFA enabled",
                    affected_resources=["authentication_system"],
                    affected_users=[user.user_id],
                    evidence=[{"user": user.username, "roles": [role.value for role in user.roles]}],
                    remediation_required=True,
                    remediation_timeframe="7 days",
                    security_impact="high",
                    compliance_impact="high",
                    details={"privileged_roles": [role.value for role in user.roles if role in privileged_roles]}
                ))
            
            # Check password age
            if user.password_last_changed:
                try:
                    password_age = datetime.now() - datetime.fromisoformat(user.password_last_changed.replace("Z", "+00:00"))
                    if password_age.days > 90:  # 90-day password policy
                        violations.append(AccessControlViolation(
                            violation_type="password_age_violation",
                            severity="low",
                            description=f"User {user.username} password not changed in {password_age.days} days",
                            affected_resources=["authentication_system"],
                            affected_users=[user.user_id],
                            evidence=[{"user": user.username, "password_age_days": password_age.days}],
                            remediation_required=True,
                            remediation_timeframe="14 days",
                            security_impact="low",
                            compliance_impact="low",
                            details={"password_age_days": password_age.days, "policy_max_days": 90}
                        ))
                except ValueError:
                    pass  # Invalid date format
        
        # Test for brute force attacks
        failed_attempts_by_user = {}
        for attempt in access_attempts:
            if not attempt.success and attempt.user_id:
                if attempt.user_id not in failed_attempts_by_user:
                    failed_attempts_by_user[attempt.user_id] = []
                failed_attempts_by_user[attempt.user_id].append(attempt)
        
        for user_id, failed_attempts in failed_attempts_by_user.items():
            # Check for rapid failed attempts (potential brute force)
            if len(failed_attempts) > 10:  # More than 10 failed attempts
                recent_attempts = [attempt for attempt in failed_attempts 
                                 if (datetime.now() - datetime.fromisoformat(attempt.timestamp.replace("Z", "+00:00"))).seconds < 3600]
                
                if len(recent_attempts) > 5:  # More than 5 in last hour
                    violations.append(AccessControlViolation(
                        violation_type="brute_force_attack_detected",
                        severity="high",
                        description=f"Potential brute force attack against user {user_id}",
                        affected_resources=["authentication_system"],
                        affected_users=[user_id],
                        evidence=[{"user_id": user_id, "failed_attempts_count": len(recent_attempts)}],
                        remediation_required=True,
                        remediation_timeframe="immediate",
                        security_impact="high",
                        compliance_impact="medium",
                        details={"total_failed_attempts": len(failed_attempts), "recent_attempts": len(recent_attempts)}
                    ))
        
        return violations
    
    def test_session_management_security(self, sessions: List[Session]) -> List[AccessControlViolation]:
        """
        Test session management security.
        
        Args:
            sessions: List of user sessions
            
        Returns:
            List of session management violations
        """
        violations = []
        
        # Group sessions by user
        sessions_by_user = {}
        for session in sessions:
            if session.user_id not in sessions_by_user:
                sessions_by_user[session.user_id] = []
            sessions_by_user[session.user_id].append(session)
        
        for user_id, user_sessions in sessions_by_user.items():
            active_sessions = [s for s in user_sessions if s.status == SessionStatus.ACTIVE]
            
            # Check concurrent session limits
            if len(active_sessions) > self.security_thresholds["max_concurrent_sessions"]:
                violations.append(AccessControlViolation(
                    violation_type="concurrent_session_limit_exceeded",
                    severity="medium",
                    description=f"User {user_id} has {len(active_sessions)} concurrent sessions (limit: {self.security_thresholds['max_concurrent_sessions']})",
                    affected_resources=["session_management"],
                    affected_users=[user_id],
                    evidence=[{"user_id": user_id, "active_sessions": len(active_sessions)}],
                    remediation_required=True,
                    remediation_timeframe="immediate",
                    security_impact="medium",
                    compliance_impact="low",
                    details={"active_sessions": len(active_sessions), "limit": self.security_thresholds["max_concurrent_sessions"]}
                ))
            
            # Check session timeout compliance
            for session in active_sessions:
                try:
                    last_activity = datetime.fromisoformat(session.last_activity.replace("Z", "+00:00"))
                    time_since_activity = datetime.now() - last_activity
                    
                    # Different timeout for privileged sessions
                    timeout_minutes = (self.security_thresholds["privileged_session_timeout_minutes"] 
                                     if session.privileged_actions 
                                     else self.security_thresholds["session_timeout_minutes"])
                    
                    if time_since_activity.total_seconds() > (timeout_minutes * 60):
                        violations.append(AccessControlViolation(
                            violation_type="session_timeout_violation",
                            severity="medium",
                            description=f"Session {session.session_id} exceeded timeout limit",
                            affected_resources=["session_management"],
                            affected_users=[user_id],
                            evidence=[{
                                "session_id": session.session_id,
                                "last_activity": session.last_activity,
                                "timeout_minutes": timeout_minutes
                            }],
                            remediation_required=True,
                            remediation_timeframe="immediate",
                            security_impact="medium",
                            compliance_impact="low",
                            details={"time_since_activity_minutes": time_since_activity.total_seconds() / 60, "timeout_minutes": timeout_minutes}
                        ))
                
                except ValueError:
                    pass  # Invalid date format
            
            # Check for session hijacking indicators
            ip_addresses = set(session.ip_address for session in active_sessions)
            if len(ip_addresses) > 2:  # Multiple IP addresses for concurrent sessions
                violations.append(AccessControlViolation(
                    violation_type="potential_session_hijacking",
                    severity="high",
                    description=f"User {user_id} has concurrent sessions from {len(ip_addresses)} different IP addresses",
                    affected_resources=["session_management"],
                    affected_users=[user_id],
                    evidence=[{"user_id": user_id, "ip_addresses": list(ip_addresses)}],
                    remediation_required=True,
                    remediation_timeframe="immediate",
                    security_impact="high",
                    compliance_impact="medium",
                    details={"ip_addresses": list(ip_addresses)}
                ))
        
        return violations
    
    def test_privilege_escalation_prevention(self, users: List[User], access_attempts: List[AccessAttempt]) -> List[AccessControlViolation]:
        """
        Test prevention of privilege escalation attacks.
        
        Args:
            users: List of user accounts
            access_attempts: List of access attempts
            
        Returns:
            List of privilege escalation violations
        """
        violations = []
        
        # Create user lookup
        user_lookup = {user.user_id: user for user in users}
        
        # Test for horizontal privilege escalation
        for attempt in access_attempts:
            if not attempt.success or not attempt.user_id:
                continue
            
            user = user_lookup.get(attempt.user_id)
            if not user:
                continue
            
            # Check if user is trying to access another user's data
            if "user_id=" in attempt.resource or "/users/" in attempt.resource:
                # Extract target user ID from resource path
                target_user_id = None
                if "user_id=" in attempt.resource:
                    target_user_id = attempt.resource.split("user_id=")[1].split("&")[0]
                elif "/users/" in attempt.resource:
                    target_user_id = attempt.resource.split("/users/")[1].split("/")[0]
                
                if target_user_id and target_user_id != user.user_id:
                    # Check if user has permission to access other users' data
                    if Permission.MANAGE_USERS not in user.permissions and Permission.AUDIT_ACCESS not in user.permissions:
                        violations.append(AccessControlViolation(
                            violation_type="horizontal_privilege_escalation",
                            severity="high",
                            description=f"User {user.username} accessed another user's data without permission",
                            affected_resources=[attempt.resource],
                            affected_users=[user.user_id, target_user_id],
                            evidence=[{
                                "accessing_user": user.username,
                                "target_user_id": target_user_id,
                                "resource": attempt.resource
                            }],
                            remediation_required=True,
                            remediation_timeframe="immediate",
                            security_impact="high",
                            compliance_impact="high",
                            details={"target_user_id": target_user_id, "user_permissions": [perm.value for perm in user.permissions]}
                        ))
            
            # Check for vertical privilege escalation (accessing admin functions)
            admin_resources = ["system_configuration", "user_management", "audit_logs"]
            if any(admin_resource in attempt.resource for admin_resource in admin_resources):
                non_admin_roles = [UserRole.HEALTH_ANALYST, UserRole.RESEARCHER, UserRole.VIEWER, UserRole.GUEST]
                if all(role in non_admin_roles for role in user.roles):
                    violations.append(AccessControlViolation(
                        violation_type="vertical_privilege_escalation",
                        severity="critical",
                        description=f"Non-admin user {user.username} accessed administrative resource",
                        affected_resources=[attempt.resource],
                        affected_users=[user.user_id],
                        evidence=[{
                            "user": user.username,
                            "roles": [role.value for role in user.roles],
                            "resource": attempt.resource
                        }],
                        remediation_required=True,
                        remediation_timeframe="immediate",
                        security_impact="critical",
                        compliance_impact="high",
                        details={"admin_resource_accessed": attempt.resource, "user_roles": [role.value for role in user.roles]}
                    ))
        
        return violations
    
    def test_authorisation_boundaries(self, users: List[User], access_attempts: List[AccessAttempt]) -> List[AccessControlViolation]:
        """
        Test authorisation boundary enforcement.
        
        Args:
            users: List of user accounts
            access_attempts: List of access attempts
            
        Returns:
            List of authorisation boundary violations
        """
        violations = []
        
        # Create user lookup
        user_lookup = {user.user_id: user for user in users}
        
        # Test data access boundaries based on roles
        for attempt in access_attempts:
            if not attempt.success or not attempt.user_id:
                continue
            
            user = user_lookup.get(attempt.user_id)
            if not user:
                continue
            
            # Test health data access boundaries
            if "raw_health_data" in attempt.resource:
                if Permission.ACCESS_RAW_DATA not in user.permissions:
                    violations.append(AccessControlViolation(
                        violation_type="unauthorised_raw_data_access",
                        severity="critical",
                        description=f"User {user.username} accessed raw health data without permission",
                        affected_resources=[attempt.resource],
                        affected_users=[user.user_id],
                        evidence=[{
                            "user": user.username,
                            "resource": attempt.resource,
                            "user_permissions": [perm.value for perm in user.permissions]
                        }],
                        remediation_required=True,
                        remediation_timeframe="immediate",
                        security_impact="critical",
                        compliance_impact="critical",
                        details={"resource": attempt.resource, "missing_permission": Permission.ACCESS_RAW_DATA.value}
                    ))
            
            # Test export functionality boundaries
            if "export" in attempt.resource.lower():
                if Permission.EXPORT_DATA not in user.permissions:
                    violations.append(AccessControlViolation(
                        violation_type="unauthorised_data_export",
                        severity="high",
                        description=f"User {user.username} attempted data export without permission",
                        affected_resources=[attempt.resource],
                        affected_users=[user.user_id],
                        evidence=[{
                            "user": user.username,
                            "resource": attempt.resource,
                            "user_permissions": [perm.value for perm in user.permissions]
                        }],
                        remediation_required=True,
                        remediation_timeframe="24 hours",
                        security_impact="high",
                        compliance_impact="high",
                        details={"resource": attempt.resource, "missing_permission": Permission.EXPORT_DATA.value}
                    ))
        
        return violations
    
    def conduct_comprehensive_access_control_assessment(self, 
                                                       users: List[User],
                                                       sessions: List[Session],
                                                       access_attempts: List[AccessAttempt]) -> Dict[str, Any]:
        """
        Conduct comprehensive access control security assessment.
        
        Args:
            users: List of user accounts
            sessions: List of user sessions
            access_attempts: List of access attempts
            
        Returns:
            Comprehensive access control assessment results
        """
        assessment_id = f"access_control_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        all_violations = []
        
        # Run all access control tests
        rbac_violations = self.test_role_based_access_control(users, access_attempts)
        all_violations.extend(rbac_violations)
        
        auth_violations = self.test_authentication_security(users, access_attempts)
        all_violations.extend(auth_violations)
        
        session_violations = self.test_session_management_security(sessions)
        all_violations.extend(session_violations)
        
        privilege_violations = self.test_privilege_escalation_prevention(users, access_attempts)
        all_violations.extend(privilege_violations)
        
        boundary_violations = self.test_authorisation_boundaries(users, access_attempts)
        all_violations.extend(boundary_violations)
        
        # Categorise violations by severity
        violations_by_severity = {
            "critical": [v for v in all_violations if v.severity == "critical"],
            "high": [v for v in all_violations if v.severity == "high"],
            "medium": [v for v in all_violations if v.severity == "medium"],
            "low": [v for v in all_violations if v.severity == "low"]
        }
        
        # Calculate security metrics
        total_users = len(users)
        active_sessions = len([s for s in sessions if s.status == SessionStatus.ACTIVE])
        successful_attempts = len([a for a in access_attempts if a.success])
        failed_attempts = len([a for a in access_attempts if not a.success])
        
        # Calculate risk score
        risk_score = self._calculate_access_control_risk_score(violations_by_severity, total_users, failed_attempts, successful_attempts)
        
        # Generate recommendations
        recommendations = self._generate_access_control_recommendations(all_violations, violations_by_severity)
        
        return {
            "assessment_id": assessment_id,
            "assessment_timestamp": datetime.now().isoformat(),
            "total_violations": len(all_violations),
            "violations_by_severity": {
                severity: len(violations) for severity, violations in violations_by_severity.items()
            },
            "violations_by_type": self._categorise_violations_by_type(all_violations),
            "security_metrics": {
                "total_users": total_users,
                "active_sessions": active_sessions,
                "successful_attempts": successful_attempts,
                "failed_attempts": failed_attempts,
                "failure_rate": failed_attempts / (successful_attempts + failed_attempts) if (successful_attempts + failed_attempts) > 0 else 0
            },
            "risk_assessment": {
                "overall_risk_score": risk_score,
                "risk_level": self._determine_risk_level(risk_score),
                "critical_vulnerabilities": len(violations_by_severity["critical"]),
                "high_vulnerabilities": len(violations_by_severity["high"])
            },
            "recommendations": recommendations,
            "detailed_violations": [violation.__dict__ for violation in all_violations],
            "compliance_impact": self._assess_compliance_impact(all_violations),
            "next_assessment_date": (datetime.now() + timedelta(days=30)).isoformat()
        }
    
    def _calculate_access_control_risk_score(self, violations_by_severity: Dict[str, List], 
                                           total_users: int, failed_attempts: int, successful_attempts: int) -> float:
        """Calculate overall access control risk score."""
        score = 0.0
        
        # Violation severity scoring
        score += len(violations_by_severity["critical"]) * 10.0
        score += len(violations_by_severity["high"]) * 5.0
        score += len(violations_by_severity["medium"]) * 2.0
        score += len(violations_by_severity["low"]) * 1.0
        
        # Failure rate impact
        if successful_attempts + failed_attempts > 0:
            failure_rate = failed_attempts / (successful_attempts + failed_attempts)
            score += failure_rate * 20.0  # High failure rate increases risk
        
        # User base impact (more users = more complexity)
        if total_users > 100:
            score += 5.0
        elif total_users > 50:
            score += 2.0
        
        # Normalise to 0-100 scale
        return min(score, 100.0)
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on score."""
        if risk_score >= 75:
            return "critical"
        elif risk_score >= 50:
            return "high"
        elif risk_score >= 25:
            return "medium"
        else:
            return "low"
    
    def _categorise_violations_by_type(self, violations: List[AccessControlViolation]) -> Dict[str, int]:
        """Categorise violations by type."""
        type_counts = {}
        for violation in violations:
            if violation.violation_type not in type_counts:
                type_counts[violation.violation_type] = 0
            type_counts[violation.violation_type] += 1
        return type_counts
    
    def _generate_access_control_recommendations(self, all_violations: List[AccessControlViolation], 
                                               violations_by_severity: Dict[str, List]) -> List[str]:
        """Generate access control recommendations."""
        recommendations = []
        
        if violations_by_severity["critical"]:
            recommendations.append("URGENT: Address critical access control violations immediately")
        
        violation_types = [v.violation_type for v in all_violations]
        
        if "unauthorised_access_unknown_user" in violation_types:
            recommendations.append("Implement stronger user authentication and session validation")
        
        if "insufficient_permissions" in violation_types:
            recommendations.append("Review and update role-based access control permissions")
        
        if "mfa_requirement_violation" in violation_types:
            recommendations.append("Enforce multi-factor authentication for all privileged users")
        
        if "brute_force_attack_detected" in violation_types:
            recommendations.append("Implement rate limiting and IP blocking for failed login attempts")
        
        if "privilege_escalation" in " ".join(violation_types):
            recommendations.append("Strengthen privilege escalation prevention mechanisms")
        
        if "session_timeout_violation" in violation_types:
            recommendations.append("Enforce session timeout policies and automatic logout")
        
        if len(all_violations) > 10:
            recommendations.append("Consider comprehensive access control system review and redesign")
        
        return recommendations
    
    def _assess_compliance_impact(self, violations: List[AccessControlViolation]) -> Dict[str, Any]:
        """Assess compliance impact of violations."""
        high_compliance_impact = len([v for v in violations if v.compliance_impact == "high"])
        medium_compliance_impact = len([v for v in violations if v.compliance_impact == "medium"])
        
        return {
            "high_compliance_impact_violations": high_compliance_impact,
            "medium_compliance_impact_violations": medium_compliance_impact,
            "overall_compliance_risk": "high" if high_compliance_impact > 0 else "medium" if medium_compliance_impact > 0 else "low"
        }


# Test suite
class TestAccessControl:
    """Test suite for access control and authentication security."""
    
    @pytest.fixture
    def access_control_tester(self):
        """Create access control tester instance."""
        return AccessControlTester()
    
    @pytest.fixture
    def test_users(self):
        """Test users with different roles."""
        return [
            User(
                user_id="admin001",
                username="admin",
                email="admin@example.com",
                roles=[UserRole.SUPER_ADMIN],
                permissions=[Permission.READ_ALL_DATA, Permission.MANAGE_USERS, Permission.SYSTEM_ADMINISTRATION],
                account_status="active",
                created_date="2023-01-01T00:00:00Z",
                mfa_enabled=True,
                password_last_changed="2023-06-01T00:00:00Z"
            ),
            User(
                user_id="analyst001",
                username="analyst",
                email="analyst@example.com",
                roles=[UserRole.HEALTH_ANALYST],
                permissions=[Permission.READ_AGGREGATED_DATA, Permission.EXPORT_DATA],
                account_status="active",
                created_date="2023-02-01T00:00:00Z",
                mfa_enabled=False,
                password_last_changed="2023-01-01T00:00:00Z"  # Old password
            ),
            User(
                user_id="viewer001",
                username="viewer",
                email="viewer@example.com",
                roles=[UserRole.VIEWER],
                permissions=[Permission.READ_AGGREGATED_DATA],
                account_status="active",
                created_date="2023-03-01T00:00:00Z",
                failed_login_attempts=6,  # Over threshold
                account_locked=False  # Should be locked
            )
        ]
    
    @pytest.fixture
    def test_sessions(self):
        """Test user sessions."""
        now = datetime.now()
        return [
            Session(
                session_id="sess001",
                user_id="admin001",
                created_timestamp=now.isoformat(),
                last_activity=(now - timedelta(minutes=10)).isoformat(),
                ip_address="192.168.1.100",
                user_agent="Mozilla/5.0",
                status=SessionStatus.ACTIVE,
                expires_at=(now + timedelta(hours=8)).isoformat(),
                mfa_verified=True
            ),
            Session(
                session_id="sess002",
                user_id="analyst001",
                created_timestamp=now.isoformat(),
                last_activity=(now - timedelta(minutes=45)).isoformat(),  # Exceeded timeout
                ip_address="192.168.1.101",
                user_agent="Mozilla/5.0",
                status=SessionStatus.ACTIVE,
                expires_at=(now + timedelta(hours=8)).isoformat()
            ),
            Session(
                session_id="sess003",
                user_id="analyst001",
                created_timestamp=now.isoformat(),
                last_activity=(now - timedelta(minutes=5)).isoformat(),
                ip_address="10.0.0.50",  # Different IP
                user_agent="Chrome/91.0",
                status=SessionStatus.ACTIVE,
                expires_at=(now + timedelta(hours=8)).isoformat()
            )
        ]
    
    @pytest.fixture
    def test_access_attempts(self):
        """Test access attempts."""
        now = datetime.now()
        return [
            AccessAttempt(
                attempt_id="att001",
                user_id="analyst001",
                username="analyst",
                resource="raw_health_data",
                action="read",
                timestamp=now.isoformat(),
                ip_address="192.168.1.101",
                user_agent="Mozilla/5.0",
                authentication_method=AuthenticationMethod.PASSWORD,
                success=True  # Should fail - analyst shouldn't access raw data
            ),
            AccessAttempt(
                attempt_id="att002",
                user_id="viewer001",
                username="viewer",
                resource="user_management",
                action="read",
                timestamp=now.isoformat(),
                ip_address="192.168.1.102",
                user_agent="Mozilla/5.0",
                authentication_method=AuthenticationMethod.PASSWORD,
                success=True  # Should fail - viewer shouldn't access user management
            ),
            AccessAttempt(
                attempt_id="att003",
                user_id=None,
                username="unknown",
                resource="system_configuration",
                action="read",
                timestamp=now.isoformat(),
                ip_address="192.168.1.200",
                user_agent="Mozilla/5.0",
                authentication_method=AuthenticationMethod.PASSWORD,
                success=False,
                failure_reason="invalid_credentials"
            )
        ]
    
    def test_role_based_access_control_violations(self, access_control_tester, test_users, test_access_attempts):
        """Test role-based access control violation detection."""
        violations = access_control_tester.test_role_based_access_control(test_users, test_access_attempts)
        
        # Should detect RBAC violations
        assert len(violations) > 0, "Should detect RBAC violations"
        
        # Check for specific violation types
        violation_types = [v.violation_type for v in violations]
        assert "insufficient_permissions" in violation_types, "Should detect insufficient permissions"
        
        # Verify violation details
        for violation in violations:
            assert violation.severity in ["critical", "high", "medium", "low"]
            assert violation.remediation_required in [True, False]
            assert len(violation.affected_resources) > 0
            assert len(violation.affected_users) > 0
    
    def test_authentication_security_violations(self, access_control_tester, test_users, test_access_attempts):
        """Test authentication security violation detection."""
        violations = access_control_tester.test_authentication_security(test_users, test_access_attempts)
        
        # Should detect authentication violations
        assert len(violations) > 0, "Should detect authentication security violations"
        
        # Check for specific violation types
        violation_types = [v.violation_type for v in violations]
        expected_types = ["account_lockout_policy_violation", "password_age_violation"]
        
        for expected_type in expected_types:
            if expected_type in violation_types:
                violation = next(v for v in violations if v.violation_type == expected_type)
                assert violation.severity in ["critical", "high", "medium", "low"]
                assert violation.remediation_required is True
    
    def test_session_management_security(self, access_control_tester, test_sessions):
        """Test session management security violation detection."""
        violations = access_control_tester.test_session_management_security(test_sessions)
        
        # Should detect session management violations
        assert len(violations) > 0, "Should detect session management violations"
        
        # Check for specific violation types
        violation_types = [v.violation_type for v in violations]
        expected_types = ["session_timeout_violation", "potential_session_hijacking"]
        
        for expected_type in expected_types:
            if expected_type in violation_types:
                violation = next(v for v in violations if v.violation_type == expected_type)
                assert violation.severity in ["high", "medium"]
                assert violation.remediation_required is True
    
    def test_privilege_escalation_prevention(self, access_control_tester, test_users, test_access_attempts):
        """Test privilege escalation prevention."""
        violations = access_control_tester.test_privilege_escalation_prevention(test_users, test_access_attempts)
        
        # Should detect privilege escalation attempts
        assert len(violations) > 0, "Should detect privilege escalation violations"
        
        # Check for escalation types
        violation_types = [v.violation_type for v in violations]
        escalation_types = ["horizontal_privilege_escalation", "vertical_privilege_escalation"]
        
        detected_escalations = [vtype for vtype in escalation_types if vtype in violation_types]
        assert len(detected_escalations) > 0, "Should detect at least one type of privilege escalation"
        
        for violation in violations:
            if "escalation" in violation.violation_type:
                assert violation.severity in ["critical", "high"]
                assert violation.remediation_timeframe == "immediate"
    
    def test_authorisation_boundaries(self, access_control_tester, test_users, test_access_attempts):
        """Test authorisation boundary enforcement."""
        violations = access_control_tester.test_authorisation_boundaries(test_users, test_access_attempts)
        
        # Should detect boundary violations
        assert len(violations) > 0, "Should detect authorisation boundary violations"
        
        # Check for boundary violation types
        violation_types = [v.violation_type for v in violations]
        boundary_types = ["unauthorised_raw_data_access", "unauthorised_data_export"]
        
        detected_boundaries = [vtype for vtype in boundary_types if vtype in violation_types]
        assert len(detected_boundaries) > 0, "Should detect authorisation boundary violations"
        
        for violation in violations:
            assert violation.severity in ["critical", "high"]
            assert violation.compliance_impact in ["critical", "high"]
    
    def test_comprehensive_access_control_assessment(self, access_control_tester, test_users, test_sessions, test_access_attempts):
        """Test comprehensive access control assessment."""
        assessment = access_control_tester.conduct_comprehensive_access_control_assessment(
            test_users, test_sessions, test_access_attempts
        )
        
        # Verify assessment structure
        assert "assessment_id" in assessment
        assert "assessment_timestamp" in assessment
        assert "total_violations" in assessment
        assert "violations_by_severity" in assessment
        assert "violations_by_type" in assessment
        assert "security_metrics" in assessment
        assert "risk_assessment" in assessment
        assert "recommendations" in assessment
        assert "detailed_violations" in assessment
        assert "compliance_impact" in assessment
        assert "next_assessment_date" in assessment
        
        # Verify assessment quality
        assert assessment["total_violations"] > 0, "Should detect violations in test data"
        assert assessment["risk_assessment"]["overall_risk_score"] > 0, "Should calculate risk score"
        assert len(assessment["recommendations"]) > 0, "Should provide recommendations"
        
        # Verify security metrics
        metrics = assessment["security_metrics"]
        assert metrics["total_users"] == len(test_users)
        assert metrics["active_sessions"] > 0
        assert 0.0 <= metrics["failure_rate"] <= 1.0
        
        # Verify risk assessment
        risk_assessment = assessment["risk_assessment"]
        assert risk_assessment["risk_level"] in ["low", "medium", "high", "critical"]
        assert risk_assessment["overall_risk_score"] >= 0
    
    def test_access_control_security_thresholds(self, access_control_tester):
        """Test access control security thresholds are properly configured."""
        thresholds = access_control_tester.security_thresholds
        
        # Verify critical thresholds
        assert thresholds["max_failed_login_attempts"] >= 3, "Failed login threshold should be reasonable"
        assert thresholds["session_timeout_minutes"] <= 60, "Session timeout should be secure"
        assert thresholds["privileged_session_timeout_minutes"] <= thresholds["session_timeout_minutes"], "Privileged sessions should have shorter timeout"
        assert thresholds["password_min_length"] >= 8, "Password minimum length should be secure"
        assert thresholds["mfa_required_for_privileged"] is True, "MFA should be required for privileged users"
    
    def test_role_permission_matrix_integrity(self, access_control_tester):
        """Test role-permission matrix integrity."""
        role_permissions = access_control_tester.role_permissions
        
        # Verify all roles have permissions defined
        for role in UserRole:
            assert role in role_permissions, f"Role {role} should have permissions defined"
        
        # Verify permission hierarchy
        super_admin_perms = set(role_permissions[UserRole.SUPER_ADMIN])
        system_admin_perms = set(role_permissions[UserRole.SYSTEM_ADMIN])
        viewer_perms = set(role_permissions[UserRole.VIEWER])
        
        # Super admin should have more permissions than system admin
        assert len(super_admin_perms) >= len(system_admin_perms), "Super admin should have at least as many permissions as system admin"
        
        # System admin should have more permissions than viewer
        assert len(system_admin_perms) > len(viewer_perms), "System admin should have more permissions than viewer"
        
        # Viewer should have minimal permissions
        assert len(viewer_perms) <= 3, "Viewer should have minimal permissions"
    
    def test_sensitive_resource_protection(self, access_control_tester):
        """Test sensitive resource protection configuration."""
        sensitive_resources = access_control_tester.sensitive_resources
        
        # Verify critical resources are protected
        critical_resources = ["raw_health_data", "patient_identifiers", "system_configuration"]
        for resource in critical_resources:
            assert resource in sensitive_resources, f"Critical resource {resource} should be protected"
            assert len(sensitive_resources[resource]) > 0, f"Resource {resource} should have required permissions"
        
        # Verify raw health data has strongest protection
        raw_data_perms = sensitive_resources["raw_health_data"]
        assert Permission.ACCESS_RAW_DATA in raw_data_perms, "Raw health data should require special permission"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])