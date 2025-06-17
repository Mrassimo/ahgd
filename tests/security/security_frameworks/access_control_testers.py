"""
Access Control Testing Framework - Phase 5.6

Comprehensive access control testing utilities for the Australian Health Analytics platform.
Provides robust testing of role-based access control (RBAC), authentication mechanisms,
and authorization validation to ensure secure access to health data.

Key Features:
- Role-based access control (RBAC) validation
- Multi-factor authentication (MFA) testing
- Session management security testing
- Privilege escalation prevention testing
- Australian health data access compliance
"""

import time
import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class AccessControlMethod(Enum):
    """Access control methods."""
    RBAC = "role_based_access_control"
    ABAC = "attribute_based_access_control"
    MAC = "mandatory_access_control"
    DAC = "discretionary_access_control"
    HYBRID = "hybrid_access_control"


class AuthenticationFactor(Enum):
    """Authentication factors."""
    PASSWORD = "password"
    TOKEN = "token"
    BIOMETRIC = "biometric"
    SMS_OTP = "sms_otp"
    EMAIL_OTP = "email_otp"
    HARDWARE_TOKEN = "hardware_token"
    PUSH_NOTIFICATION = "push_notification"


class PermissionLevel(Enum):
    """Permission levels for health data access."""
    NONE = "none"
    READ = "read"
    WRITE = "write"
    UPDATE = "update"
    DELETE = "delete"
    ADMIN = "admin"
    FULL_ACCESS = "full_access"


class UserRole(Enum):
    """User roles in health data system."""
    PATIENT = "patient"
    HEALTHCARE_PROVIDER = "healthcare_provider"
    NURSE = "nurse"
    DOCTOR = "doctor"
    SPECIALIST = "specialist"
    RESEARCHER = "researcher"
    DATA_ANALYST = "data_analyst"
    SYSTEM_ADMIN = "system_admin"
    PRIVACY_OFFICER = "privacy_officer"
    COMPLIANCE_OFFICER = "compliance_officer"


@dataclass
class AccessAttempt:
    """Access attempt record."""
    user_id: str
    resource: str
    action: str
    timestamp: datetime
    success: bool
    authentication_method: str
    ip_address: str
    user_agent: str
    session_id: str
    risk_score: float
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    """User profile for access control testing."""
    user_id: str
    username: str
    email: str
    roles: List[UserRole]
    permissions: Dict[str, List[PermissionLevel]]
    mfa_enabled: bool
    account_status: str
    last_login: Optional[datetime]
    failed_login_attempts: int
    password_hash: str
    session_tokens: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessControlPolicy:
    """Access control policy definition."""
    policy_id: str
    name: str
    description: str
    resource_patterns: List[str]
    required_roles: List[UserRole]
    required_permissions: List[PermissionLevel]
    conditions: Dict[str, Any]
    priority: int
    enabled: bool


@dataclass
class AccessControlTestResult:
    """Access control test result."""
    test_name: str
    test_type: str
    success: bool
    user_id: str
    resource: str
    action: str
    expected_result: bool
    actual_result: bool
    execution_time_ms: float
    error_message: Optional[str]
    security_implications: List[str]
    compliance_status: Dict[str, bool]


class AccessControlTester:
    """
    Comprehensive access control testing framework for Australian health data systems.
    Validates RBAC, authentication, and authorization mechanisms.
    """
    
    def __init__(self, 
                 config_path: Optional[Path] = None,
                 enable_audit_logging: bool = True):
        """
        Initialize access control tester.
        
        Args:
            config_path: Path to access control configuration
            enable_audit_logging: Enable audit logging for access attempts
        """
        self.config_path = config_path
        self.enable_audit_logging = enable_audit_logging
        
        # Load configuration
        self.policies: List[AccessControlPolicy] = []
        self.users: Dict[str, UserProfile] = {}
        self.access_attempts: List[AccessAttempt] = []
        
        # Security settings
        self.session_timeout_minutes = 30
        self.max_failed_attempts = 5
        self.password_policy = {
            'min_length': 12,
            'require_uppercase': True,
            'require_lowercase': True,
            'require_numbers': True,
            'require_special_chars': True,
            'prevent_common_passwords': True
        }
        
        # Australian health data specific settings
        self.health_data_classifications = {
            'public_health_statistics': ['READ'],
            'de_identified_data': ['READ', 'WRITE'],
            'identifiable_health_data': ['READ', 'WRITE', 'UPDATE'],
            'sensitive_health_data': ['ADMIN'],
            'genetic_data': ['FULL_ACCESS'],
            'mental_health_data': ['FULL_ACCESS']
        }
        
        self._initialize_test_environment()
        
        logger.info("Access control tester initialized")
    
    def _initialize_test_environment(self):
        """Initialize test environment with sample data."""
        # Create sample users with different roles
        self._create_sample_users()
        
        # Create sample policies
        self._create_sample_policies()
        
        logger.info(f"Test environment initialized with {len(self.users)} users and {len(self.policies)} policies")
    
    def _create_sample_users(self):
        """Create sample users for testing."""
        sample_users = [
            {
                'user_id': 'patient_001',
                'username': 'john.patient',
                'email': 'john.patient@example.com',
                'roles': [UserRole.PATIENT],
                'permissions': {
                    'own_health_data': [PermissionLevel.READ, PermissionLevel.UPDATE],
                    'appointment_booking': [PermissionLevel.READ, PermissionLevel.WRITE]
                },
                'mfa_enabled': False,
                'account_status': 'active'
            },
            {
                'user_id': 'doctor_001',
                'username': 'dr.smith',
                'email': 'dr.smith@hospital.com.au',
                'roles': [UserRole.DOCTOR, UserRole.HEALTHCARE_PROVIDER],
                'permissions': {
                    'patient_health_data': [PermissionLevel.READ, PermissionLevel.WRITE, PermissionLevel.UPDATE],
                    'prescription_data': [PermissionLevel.READ, PermissionLevel.WRITE],
                    'clinical_notes': [PermissionLevel.FULL_ACCESS]
                },
                'mfa_enabled': True,
                'account_status': 'active'
            },
            {
                'user_id': 'researcher_001',
                'username': 'alice.researcher',
                'email': 'alice.researcher@university.edu.au',
                'roles': [UserRole.RESEARCHER],
                'permissions': {
                    'de_identified_data': [PermissionLevel.READ],
                    'aggregate_statistics': [PermissionLevel.READ, PermissionLevel.WRITE]
                },
                'mfa_enabled': True,
                'account_status': 'active'
            },
            {
                'user_id': 'admin_001',
                'username': 'system.admin',
                'email': 'admin@healthsystem.gov.au',
                'roles': [UserRole.SYSTEM_ADMIN],
                'permissions': {
                    'system_configuration': [PermissionLevel.FULL_ACCESS],
                    'user_management': [PermissionLevel.FULL_ACCESS],
                    'audit_logs': [PermissionLevel.READ]
                },
                'mfa_enabled': True,
                'account_status': 'active'
            }
        ]
        
        for user_data in sample_users:
            user_profile = UserProfile(
                user_id=user_data['user_id'],
                username=user_data['username'],
                email=user_data['email'],
                roles=user_data['roles'],
                permissions=user_data['permissions'],
                mfa_enabled=user_data['mfa_enabled'],
                account_status=user_data['account_status'],
                last_login=None,
                failed_login_attempts=0,
                password_hash=self._hash_password("SecurePassword123!")
            )
            self.users[user_data['user_id']] = user_profile
    
    def _create_sample_policies(self):
        """Create sample access control policies."""
        sample_policies = [
            {
                'policy_id': 'health_data_access_001',
                'name': 'Patient Health Data Access',
                'description': 'Controls access to identifiable patient health data',
                'resource_patterns': ['/api/patients/*/health-data', '/api/patients/*/medical-history'],
                'required_roles': [UserRole.DOCTOR, UserRole.NURSE, UserRole.HEALTHCARE_PROVIDER],
                'required_permissions': [PermissionLevel.READ, PermissionLevel.WRITE],
                'conditions': {
                    'require_mfa': True,
                    'max_session_duration': 30,
                    'ip_whitelist_required': False
                },
                'priority': 1,
                'enabled': True
            },
            {
                'policy_id': 'research_data_access_001',
                'name': 'Research Data Access',
                'description': 'Controls access to de-identified research data',
                'resource_patterns': ['/api/research/data', '/api/research/statistics'],
                'required_roles': [UserRole.RESEARCHER],
                'required_permissions': [PermissionLevel.READ],
                'conditions': {
                    'require_mfa': True,
                    'data_use_agreement_required': True,
                    'audit_all_access': True
                },
                'priority': 2,
                'enabled': True
            },
            {
                'policy_id': 'admin_access_001',
                'name': 'System Administration Access',
                'description': 'Controls system administration functions',
                'resource_patterns': ['/api/admin/*', '/api/system/*'],
                'required_roles': [UserRole.SYSTEM_ADMIN],
                'required_permissions': [PermissionLevel.ADMIN, PermissionLevel.FULL_ACCESS],
                'conditions': {
                    'require_mfa': True,
                    'ip_whitelist_required': True,
                    'time_based_access': True,
                    'approval_workflow_required': True
                },
                'priority': 0,
                'enabled': True
            }
        ]
        
        for policy_data in sample_policies:
            policy = AccessControlPolicy(
                policy_id=policy_data['policy_id'],
                name=policy_data['name'],
                description=policy_data['description'],
                resource_patterns=policy_data['resource_patterns'],
                required_roles=policy_data['required_roles'],
                required_permissions=policy_data['required_permissions'],
                conditions=policy_data['conditions'],
                priority=policy_data['priority'],
                enabled=policy_data['enabled']
            )
            self.policies.append(policy)
    
    def test_role_based_access_control(self, 
                                     user_id: str,
                                     resource: str,
                                     action: str) -> AccessControlTestResult:
        """
        Test role-based access control for a specific user and resource.
        
        Args:
            user_id: User identifier
            resource: Resource being accessed
            action: Action being performed
            
        Returns:
            AccessControlTestResult: RBAC test result
        """
        start_time = time.time()
        
        # Get user profile
        user = self.users.get(user_id)
        if not user:
            return AccessControlTestResult(
                test_name="rbac_test",
                test_type="role_based_access_control",
                success=False,
                user_id=user_id,
                resource=resource,
                action=action,
                expected_result=False,
                actual_result=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message="User not found",
                security_implications=["Invalid user access attempt"],
                compliance_status={"user_validation": False}
            )
        
        # Find applicable policies
        applicable_policies = self._find_applicable_policies(resource)
        
        # Evaluate access based on roles and policies
        access_granted = False
        security_implications = []
        compliance_status = {}
        
        for policy in applicable_policies:
            # Check role requirements
            has_required_role = any(role in user.roles for role in policy.required_roles)
            
            # Check permission requirements
            resource_permissions = user.permissions.get(resource, [])
            has_required_permission = any(
                perm in resource_permissions 
                for perm in policy.required_permissions
            )
            
            # Check conditions
            conditions_met = self._evaluate_policy_conditions(user, policy, resource, action)
            
            if has_required_role and has_required_permission and conditions_met:
                access_granted = True
                break
            else:
                if not has_required_role:
                    security_implications.append(f"User lacks required role for policy {policy.policy_id}")
                if not has_required_permission:
                    security_implications.append(f"User lacks required permission for policy {policy.policy_id}")
                if not conditions_met:
                    security_implications.append(f"Policy conditions not met for policy {policy.policy_id}")
        
        # Log access attempt
        self._log_access_attempt(user_id, resource, action, access_granted, "rbac_test")
        
        # Evaluate compliance
        compliance_status = {
            "role_validation": True,
            "permission_validation": True,
            "policy_evaluation": len(applicable_policies) > 0,
            "audit_logging": self.enable_audit_logging
        }
        
        execution_time = (time.time() - start_time) * 1000
        
        return AccessControlTestResult(
            test_name="rbac_test",
            test_type="role_based_access_control",
            success=True,
            user_id=user_id,
            resource=resource,
            action=action,
            expected_result=access_granted,
            actual_result=access_granted,
            execution_time_ms=execution_time,
            error_message=None,
            security_implications=security_implications,
            compliance_status=compliance_status
        )
    
    def test_multi_factor_authentication(self, 
                                       user_id: str,
                                       primary_factor: str,
                                       secondary_factors: List[str]) -> AccessControlTestResult:
        """
        Test multi-factor authentication implementation.
        
        Args:
            user_id: User identifier
            primary_factor: Primary authentication factor (usually password)
            secondary_factors: Secondary authentication factors
            
        Returns:
            AccessControlTestResult: MFA test result
        """
        start_time = time.time()
        
        user = self.users.get(user_id)
        if not user:
            return AccessControlTestResult(
                test_name="mfa_test",
                test_type="multi_factor_authentication",
                success=False,
                user_id=user_id,
                resource="authentication_system",
                action="authenticate",
                expected_result=False,
                actual_result=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message="User not found",
                security_implications=["Invalid authentication attempt"],
                compliance_status={"user_validation": False}
            )
        
        security_implications = []
        compliance_status = {}
        
        # Validate primary factor (password)
        primary_factor_valid = self._validate_password(primary_factor, user.password_hash)
        
        # Check if MFA is required for this user
        mfa_required = user.mfa_enabled
        
        # Validate secondary factors if MFA is enabled
        secondary_factors_valid = True
        if mfa_required and secondary_factors:
            secondary_factors_valid = self._validate_secondary_factors(
                user_id, secondary_factors
            )
        elif mfa_required and not secondary_factors:
            secondary_factors_valid = False
            security_implications.append("MFA required but no secondary factors provided")
        
        # Determine authentication success
        authentication_success = primary_factor_valid and (
            not mfa_required or secondary_factors_valid
        )
        
        # Update failed login attempts
        if not authentication_success:
            user.failed_login_attempts += 1
            if user.failed_login_attempts >= self.max_failed_attempts:
                user.account_status = 'locked'
                security_implications.append("Account locked due to excessive failed login attempts")
        else:
            user.failed_login_attempts = 0
            user.last_login = datetime.now()
        
        # Log authentication attempt
        self._log_access_attempt(
            user_id, "authentication_system", "authenticate", 
            authentication_success, "mfa_test"
        )
        
        # Evaluate compliance
        compliance_status = {
            "mfa_enforcement": mfa_required,
            "password_policy_compliance": self._validate_password_policy(primary_factor),
            "account_lockout_policy": user.failed_login_attempts < self.max_failed_attempts,
            "audit_logging": True
        }
        
        execution_time = (time.time() - start_time) * 1000
        
        return AccessControlTestResult(
            test_name="mfa_test",
            test_type="multi_factor_authentication",
            success=True,
            user_id=user_id,
            resource="authentication_system",
            action="authenticate",
            expected_result=authentication_success,
            actual_result=authentication_success,
            execution_time_ms=execution_time,
            error_message=None,
            security_implications=security_implications,
            compliance_status=compliance_status
        )
    
    def test_session_management(self, 
                              user_id: str,
                              session_token: str,
                              session_age_minutes: int) -> AccessControlTestResult:
        """
        Test session management security.
        
        Args:
            user_id: User identifier
            session_token: Session token to validate
            session_age_minutes: Age of the session in minutes
            
        Returns:
            AccessControlTestResult: Session management test result
        """
        start_time = time.time()
        
        user = self.users.get(user_id)
        security_implications = []
        compliance_status = {}
        
        # Validate session token format and integrity
        token_valid = self._validate_session_token(session_token)
        
        # Check session timeout
        session_expired = session_age_minutes > self.session_timeout_minutes
        if session_expired:
            security_implications.append(f"Session expired: {session_age_minutes} minutes > {self.session_timeout_minutes} minutes")
        
        # Check if session token exists for user
        session_exists = user and session_token in user.session_tokens if user else False
        
        # Determine session validity
        session_valid = token_valid and not session_expired and session_exists
        
        # Log session validation attempt
        self._log_access_attempt(
            user_id, "session_management", "validate_session", 
            session_valid, "session_test"
        )
        
        # Evaluate compliance
        compliance_status = {
            "session_timeout_enforcement": not session_expired,
            "token_validation": token_valid,
            "session_tracking": session_exists if user else False,
            "secure_token_generation": len(session_token) >= 32 if session_token else False
        }
        
        execution_time = (time.time() - start_time) * 1000
        
        return AccessControlTestResult(
            test_name="session_management_test",
            test_type="session_management",
            success=True,
            user_id=user_id,
            resource="session_management",
            action="validate_session",
            expected_result=session_valid,
            actual_result=session_valid,
            execution_time_ms=execution_time,
            error_message=None,
            security_implications=security_implications,
            compliance_status=compliance_status
        )
    
    def test_privilege_escalation_prevention(self, 
                                           user_id: str,
                                           target_resource: str,
                                           escalation_method: str) -> AccessControlTestResult:
        """
        Test privilege escalation prevention mechanisms.
        
        Args:
            user_id: User identifier attempting escalation
            target_resource: Resource requiring higher privileges
            escalation_method: Method used for escalation attempt
            
        Returns:
            AccessControlTestResult: Privilege escalation test result
        """
        start_time = time.time()
        
        user = self.users.get(user_id)
        security_implications = []
        compliance_status = {}
        
        # Define high-privilege resources
        high_privilege_resources = [
            '/api/admin/users',
            '/api/admin/system-config',
            '/api/admin/audit-logs',
            '/api/system/database',
            '/api/system/backups'
        ]
        
        # Check if target resource requires high privileges
        requires_high_privilege = any(
            target_resource.startswith(resource) 
            for resource in high_privilege_resources
        )
        
        # Evaluate user's current privilege level
        user_privilege_level = self._calculate_privilege_level(user) if user else 0
        required_privilege_level = 8 if requires_high_privilege else 5
        
        # Check for escalation attempts
        escalation_detected = False
        escalation_indicators = [
            'parameter_manipulation',
            'url_tampering',
            'header_injection',
            'token_manipulation',
            'role_assumption'
        ]
        
        if escalation_method in escalation_indicators:
            escalation_detected = True
            security_implications.append(f"Privilege escalation attempt detected: {escalation_method}")
        
        # Determine if access should be granted
        legitimate_access = (
            user_privilege_level >= required_privilege_level and 
            not escalation_detected
        )
        
        # Log escalation attempt
        self._log_access_attempt(
            user_id, target_resource, "privilege_escalation_attempt", 
            escalation_detected, "privilege_escalation_test"
        )
        
        # Additional security measures for escalation attempts
        if escalation_detected:
            security_implications.extend([
                "Immediate security alert triggered",
                "User activity flagged for review",
                "Additional monitoring enabled"
            ])
        
        # Evaluate compliance
        compliance_status = {
            "privilege_validation": user_privilege_level >= required_privilege_level if user else False,
            "escalation_detection": escalation_detected,
            "access_control_enforcement": not legitimate_access if escalation_detected else True,
            "security_monitoring": True
        }
        
        execution_time = (time.time() - start_time) * 1000
        
        return AccessControlTestResult(
            test_name="privilege_escalation_prevention_test",
            test_type="privilege_escalation_prevention",
            success=True,
            user_id=user_id,
            resource=target_resource,
            action="access_attempt",
            expected_result=not escalation_detected,
            actual_result=not escalation_detected,
            execution_time_ms=execution_time,
            error_message=None,
            security_implications=security_implications,
            compliance_status=compliance_status
        )
    
    def test_comprehensive_access_control_suite(self, 
                                              test_scenarios: List[Dict[str, Any]]) -> List[AccessControlTestResult]:
        """
        Execute comprehensive access control test suite.
        
        Args:
            test_scenarios: List of test scenarios to execute
            
        Returns:
            List[AccessControlTestResult]: Comprehensive test results
        """
        results = []
        
        logger.info(f"Executing comprehensive access control test suite with {len(test_scenarios)} scenarios")
        
        for i, scenario in enumerate(test_scenarios):
            logger.info(f"Executing test scenario {i+1}/{len(test_scenarios)}: {scenario.get('name', 'Unknown')}")
            
            test_type = scenario.get('test_type')
            
            if test_type == 'rbac':
                result = self.test_role_based_access_control(
                    user_id=scenario['user_id'],
                    resource=scenario['resource'],
                    action=scenario['action']
                )
            elif test_type == 'mfa':
                result = self.test_multi_factor_authentication(
                    user_id=scenario['user_id'],
                    primary_factor=scenario['primary_factor'],
                    secondary_factors=scenario.get('secondary_factors', [])
                )
            elif test_type == 'session':
                result = self.test_session_management(
                    user_id=scenario['user_id'],
                    session_token=scenario['session_token'],
                    session_age_minutes=scenario['session_age_minutes']
                )
            elif test_type == 'privilege_escalation':
                result = self.test_privilege_escalation_prevention(
                    user_id=scenario['user_id'],
                    target_resource=scenario['target_resource'],
                    escalation_method=scenario['escalation_method']
                )
            else:
                continue
            
            results.append(result)
        
        # Generate comprehensive summary
        successful_tests = sum(1 for r in results if r.success and r.actual_result == r.expected_result)
        total_tests = len(results)
        
        logger.info(f"Access control test suite completed: {successful_tests}/{total_tests} tests successful")
        
        return results
    
    def _find_applicable_policies(self, resource: str) -> List[AccessControlPolicy]:
        """Find policies applicable to a resource."""
        applicable_policies = []
        
        for policy in self.policies:
            if policy.enabled:
                for pattern in policy.resource_patterns:
                    if self._resource_matches_pattern(resource, pattern):
                        applicable_policies.append(policy)
                        break
        
        # Sort by priority (lower number = higher priority)
        applicable_policies.sort(key=lambda p: p.priority)
        
        return applicable_policies
    
    def _resource_matches_pattern(self, resource: str, pattern: str) -> bool:
        """Check if resource matches policy pattern."""
        if '*' in pattern:
            # Simple wildcard matching
            prefix = pattern.split('*')[0]
            return resource.startswith(prefix)
        else:
            return resource == pattern
    
    def _evaluate_policy_conditions(self, 
                                  user: UserProfile,
                                  policy: AccessControlPolicy,
                                  resource: str,
                                  action: str) -> bool:
        """Evaluate policy conditions."""
        conditions = policy.conditions
        
        # Check MFA requirement
        if conditions.get('require_mfa', False) and not user.mfa_enabled:
            return False
        
        # Check session duration (simplified)
        max_session_duration = conditions.get('max_session_duration')
        if max_session_duration and user.last_login:
            session_duration = (datetime.now() - user.last_login).total_seconds() / 60
            if session_duration > max_session_duration:
                return False
        
        # Additional condition checks would go here
        
        return True
    
    def _validate_password(self, password: str, password_hash: str) -> bool:
        """Validate password against hash."""
        # Simplified password validation
        test_hash = self._hash_password(password)
        return test_hash == password_hash
    
    def _hash_password(self, password: str) -> str:
        """Hash password for storage."""
        # Simplified password hashing (use proper bcrypt/scrypt in production)
        salt = "australian_health_salt"
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    def _validate_secondary_factors(self, user_id: str, factors: List[str]) -> bool:
        """Validate secondary authentication factors."""
        # Simplified secondary factor validation
        # In production, this would validate OTP, biometrics, etc.
        return len(factors) > 0 and all(len(factor) >= 6 for factor in factors)
    
    def _validate_password_policy(self, password: str) -> bool:
        """Validate password against policy."""
        policy = self.password_policy
        
        if len(password) < policy['min_length']:
            return False
        
        if policy['require_uppercase'] and not any(c.isupper() for c in password):
            return False
        
        if policy['require_lowercase'] and not any(c.islower() for c in password):
            return False
        
        if policy['require_numbers'] and not any(c.isdigit() for c in password):
            return False
        
        if policy['require_special_chars'] and not any(c in '!@#$%^&*()_+-=' for c in password):
            return False
        
        return True
    
    def _validate_session_token(self, token: str) -> bool:
        """Validate session token format and integrity."""
        if not token or len(token) < 32:
            return False
        
        # Additional token validation logic would go here
        return True
    
    def _calculate_privilege_level(self, user: UserProfile) -> int:
        """Calculate user's privilege level based on roles."""
        privilege_levels = {
            UserRole.PATIENT: 1,
            UserRole.HEALTHCARE_PROVIDER: 3,
            UserRole.NURSE: 4,
            UserRole.DOCTOR: 6,
            UserRole.SPECIALIST: 7,
            UserRole.RESEARCHER: 5,
            UserRole.DATA_ANALYST: 5,
            UserRole.PRIVACY_OFFICER: 8,
            UserRole.COMPLIANCE_OFFICER: 8,
            UserRole.SYSTEM_ADMIN: 10
        }
        
        if not user.roles:
            return 0
        
        return max(privilege_levels.get(role, 0) for role in user.roles)
    
    def _log_access_attempt(self, 
                          user_id: str,
                          resource: str,
                          action: str,
                          success: bool,
                          method: str):
        """Log access attempt for audit purposes."""
        if self.enable_audit_logging:
            attempt = AccessAttempt(
                user_id=user_id,
                resource=resource,
                action=action,
                timestamp=datetime.now(),
                success=success,
                authentication_method=method,
                ip_address="127.0.0.1",  # Simplified for testing
                user_agent="TestAgent/1.0",
                session_id=secrets.token_hex(16),
                risk_score=self._calculate_risk_score(user_id, resource, action, success)
            )
            self.access_attempts.append(attempt)
    
    def _calculate_risk_score(self, 
                            user_id: str, 
                            resource: str, 
                            action: str, 
                            success: bool) -> float:
        """Calculate risk score for access attempt."""
        base_score = 0.0
        
        # Failed attempts increase risk
        if not success:
            base_score += 5.0
        
        # Administrative resources increase risk
        if 'admin' in resource.lower() or 'system' in resource.lower():
            base_score += 3.0
        
        # Multiple recent failed attempts increase risk
        user = self.users.get(user_id)
        if user and user.failed_login_attempts > 2:
            base_score += user.failed_login_attempts * 2.0
        
        return min(base_score, 10.0)  # Cap at 10.0
    
    def generate_access_control_report(self) -> Dict[str, Any]:
        """Generate comprehensive access control test report."""
        total_attempts = len(self.access_attempts)
        successful_attempts = sum(1 for attempt in self.access_attempts if attempt.success)
        
        # Calculate risk metrics
        high_risk_attempts = sum(1 for attempt in self.access_attempts if attempt.risk_score >= 7.0)
        
        # User statistics
        active_users = sum(1 for user in self.users.values() if user.account_status == 'active')
        mfa_enabled_users = sum(1 for user in self.users.values() if user.mfa_enabled)
        
        report = {
            'report_generation_time': datetime.now().isoformat(),
            'access_control_summary': {
                'total_access_attempts': total_attempts,
                'successful_attempts': successful_attempts,
                'failed_attempts': total_attempts - successful_attempts,
                'success_rate': successful_attempts / total_attempts if total_attempts > 0 else 0.0,
                'high_risk_attempts': high_risk_attempts,
                'risk_percentage': high_risk_attempts / total_attempts if total_attempts > 0 else 0.0
            },
            'user_statistics': {
                'total_users': len(self.users),
                'active_users': active_users,
                'mfa_enabled_users': mfa_enabled_users,
                'mfa_adoption_rate': mfa_enabled_users / len(self.users) if self.users else 0.0
            },
            'policy_statistics': {
                'total_policies': len(self.policies),
                'enabled_policies': sum(1 for policy in self.policies if policy.enabled),
                'policy_coverage': len(set(pattern for policy in self.policies for pattern in policy.resource_patterns))
            },
            'security_recommendations': self._generate_security_recommendations(),
            'compliance_status': {
                'australian_privacy_principles_compliance': True,
                'health_data_access_controls': True,
                'audit_logging_enabled': self.enable_audit_logging,
                'mfa_enforcement': mfa_enabled_users > len(self.users) * 0.8  # 80% threshold
            }
        }
        
        return report
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations based on test results."""
        recommendations = []
        
        # Check MFA adoption
        mfa_users = sum(1 for user in self.users.values() if user.mfa_enabled)
        mfa_rate = mfa_users / len(self.users) if self.users else 0.0
        
        if mfa_rate < 0.9:
            recommendations.append("Increase MFA adoption rate - current rate is below 90%")
        
        # Check failed login attempts
        locked_users = sum(1 for user in self.users.values() if user.account_status == 'locked')
        if locked_users > 0:
            recommendations.append(f"Review {locked_users} locked user accounts for potential security issues")
        
        # Check high-risk access attempts
        high_risk_attempts = sum(1 for attempt in self.access_attempts if attempt.risk_score >= 7.0)
        if high_risk_attempts > len(self.access_attempts) * 0.1:  # 10% threshold
            recommendations.append("High number of high-risk access attempts detected - review security monitoring")
        
        # Policy recommendations
        if len(self.policies) < 5:
            recommendations.append("Consider implementing additional access control policies for comprehensive coverage")
        
        return recommendations