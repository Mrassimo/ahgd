"""
Encryption Security Testing

Comprehensive testing suite for encryption security including:
- Data encryption at rest validation (AES-256)
- Data encryption in transit validation (TLS 1.3)
- Key management and rotation testing
- Certificate validation and PKI testing
- Cryptographic algorithm strength validation
- Secure communication protocol testing

This test suite ensures the platform implements robust encryption mechanisms
that protect sensitive health data according to enterprise security standards.
"""

import json
import pytest
import hashlib
import secrets
import ssl
import socket
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Union, Any
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass, field
from enum import Enum
import logging
import base64
import os

import polars as pl
import numpy as np
from loguru import logger


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    AES_128_GCM = "aes_128_gcm"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    ECDSA_P256 = "ecdsa_p256"
    ECDSA_P384 = "ecdsa_p384"


class TLSVersion(Enum):
    """TLS protocol versions."""
    TLS_1_0 = "tls_1.0"
    TLS_1_1 = "tls_1.1"
    TLS_1_2 = "tls_1.2"
    TLS_1_3 = "tls_1.3"


class CertificateType(Enum):
    """Certificate types."""
    ROOT_CA = "root_ca"
    INTERMEDIATE_CA = "intermediate_ca"
    SERVER_CERTIFICATE = "server_certificate"
    CLIENT_CERTIFICATE = "client_certificate"
    CODE_SIGNING = "code_signing"


class KeyUsage(Enum):
    """Certificate key usage."""
    DIGITAL_SIGNATURE = "digital_signature"
    KEY_ENCIPHERMENT = "key_encipherment"
    DATA_ENCIPHERMENT = "data_encipherment"
    KEY_AGREEMENT = "key_agreement"
    CERTIFICATE_SIGNING = "certificate_signing"
    CRL_SIGNING = "crl_signing"


@dataclass
class EncryptionConfiguration:
    """Encryption configuration."""
    algorithm: EncryptionAlgorithm
    key_size: int
    mode: str
    padding: Optional[str]
    iv_size: Optional[int]
    tag_size: Optional[int]
    key_derivation: str
    salt_size: Optional[int]
    iterations: Optional[int]


@dataclass
class Certificate:
    """Digital certificate information."""
    certificate_id: str
    subject: str
    issuer: str
    serial_number: str
    not_before: str
    not_after: str
    certificate_type: CertificateType
    key_algorithm: str
    key_size: int
    signature_algorithm: str
    key_usage: List[KeyUsage]
    extended_key_usage: List[str]
    san_domains: List[str]
    is_ca: bool
    path_length_constraint: Optional[int]
    fingerprint_sha256: str


@dataclass
class TLSConfiguration:
    """TLS configuration."""
    protocol_version: TLSVersion
    cipher_suites: List[str]
    supported_groups: List[str]
    signature_algorithms: List[str]
    certificate_chain: List[Certificate]
    hsts_enabled: bool
    hsts_max_age: Optional[int]
    ocsp_stapling: bool
    perfect_forward_secrecy: bool


@dataclass
class EncryptionViolation:
    """Encryption security violation."""
    violation_type: str
    severity: str  # "critical", "high", "medium", "low"
    description: str
    affected_components: List[str]
    weak_algorithms: List[str]
    compliance_impact: str
    remediation_required: bool
    remediation_timeframe: str
    evidence: List[Dict[str, Any]]
    security_impact: str
    details: Dict[str, Any]


class EncryptionSecurityTester:
    """Encryption security tester."""
    
    def __init__(self):
        """Initialise encryption security tester."""
        self.logger = logger.bind(component="encryption_security_tester")
        
        # Approved encryption algorithms and minimum key sizes
        self.approved_algorithms = {
            EncryptionAlgorithm.AES_256_GCM: {"min_key_size": 256, "status": "recommended"},
            EncryptionAlgorithm.AES_256_CBC: {"min_key_size": 256, "status": "acceptable"},
            EncryptionAlgorithm.AES_128_GCM: {"min_key_size": 128, "status": "acceptable"},
            EncryptionAlgorithm.CHACHA20_POLY1305: {"min_key_size": 256, "status": "recommended"},
            EncryptionAlgorithm.RSA_2048: {"min_key_size": 2048, "status": "minimum"},
            EncryptionAlgorithm.RSA_4096: {"min_key_size": 4096, "status": "recommended"},
            EncryptionAlgorithm.ECDSA_P256: {"min_key_size": 256, "status": "acceptable"},
            EncryptionAlgorithm.ECDSA_P384: {"min_key_size": 384, "status": "recommended"}
        }
        
        # Deprecated/weak algorithms
        self.deprecated_algorithms = [
            "des", "3des", "rc4", "md5", "sha1", "rsa_1024", "dh_1024"
        ]
        
        # Approved TLS configurations
        self.approved_tls_versions = [TLSVersion.TLS_1_2, TLSVersion.TLS_1_3]
        self.deprecated_tls_versions = [TLSVersion.TLS_1_0, TLSVersion.TLS_1_1]
        
        # Secure cipher suites (TLS 1.3)
        self.secure_cipher_suites = [
            "TLS_AES_256_GCM_SHA384",
            "TLS_CHACHA20_POLY1305_SHA256",
            "TLS_AES_128_GCM_SHA256"
        ]
        
        # Weak cipher suites to avoid
        self.weak_cipher_suites = [
            "TLS_RSA_WITH_3DES_EDE_CBC_SHA",
            "TLS_RSA_WITH_RC4_128_SHA",
            "TLS_RSA_WITH_RC4_128_MD5",
            "TLS_RSA_WITH_DES_CBC_SHA"
        ]
        
        # Key management requirements
        self.key_management_requirements = {
            "rotation_period_days": 365,
            "backup_encryption_required": True,
            "hardware_security_module": True,
            "key_escrow_required": True,
            "minimum_key_storage_encryption": EncryptionAlgorithm.AES_256_GCM
        }
    
    def test_data_encryption_at_rest(self, storage_configurations: List[Dict[str, Any]]) -> List[EncryptionViolation]:
        """
        Test data encryption at rest implementation.
        
        Args:
            storage_configurations: List of storage configuration details
            
        Returns:
            List of encryption at rest violations
        """
        violations = []
        
        for config in storage_configurations:
            storage_name = config.get("storage_name", "unknown")
            
            # Check if encryption is enabled
            encryption_enabled = config.get("encryption_enabled", False)
            if not encryption_enabled:
                violations.append(EncryptionViolation(
                    violation_type="encryption_at_rest_disabled",
                    severity="critical",
                    description=f"Data encryption at rest disabled for storage: {storage_name}",
                    affected_components=[storage_name],
                    weak_algorithms=[],
                    compliance_impact="critical",
                    remediation_required=True,
                    remediation_timeframe="immediate",
                    evidence=[{"storage_config": config}],
                    security_impact="critical",
                    details={"storage_name": storage_name, "encryption_status": "disabled"}
                ))
                continue
            
            # Check encryption algorithm strength
            algorithm = config.get("encryption_algorithm", "").lower()
            key_size = config.get("key_size", 0)
            
            if any(weak_alg in algorithm for weak_alg in self.deprecated_algorithms):
                violations.append(EncryptionViolation(
                    violation_type="weak_encryption_algorithm",
                    severity="high",
                    description=f"Weak encryption algorithm in use: {algorithm}",
                    affected_components=[storage_name],
                    weak_algorithms=[algorithm],
                    compliance_impact="high",
                    remediation_required=True,
                    remediation_timeframe="30 days",
                    evidence=[{"storage_config": config}],
                    security_impact="high",
                    details={"algorithm": algorithm, "recommended_algorithms": ["AES-256-GCM", "ChaCha20-Poly1305"]}
                ))
            
            # Check key size adequacy
            if "aes" in algorithm and key_size < 256:
                violations.append(EncryptionViolation(
                    violation_type="insufficient_key_size",
                    severity="high",
                    description=f"Insufficient key size for AES: {key_size} bits (minimum: 256)",
                    affected_components=[storage_name],
                    weak_algorithms=[f"{algorithm}_{key_size}"],
                    compliance_impact="high",
                    remediation_required=True,
                    remediation_timeframe="30 days",
                    evidence=[{"storage_config": config}],
                    security_impact="high",
                    details={"current_key_size": key_size, "minimum_key_size": 256}
                ))
            
            # Check key management
            key_management = config.get("key_management", {})
            if not key_management.get("automated_rotation", False):
                violations.append(EncryptionViolation(
                    violation_type="key_rotation_not_automated",
                    severity="medium",
                    description=f"Automated key rotation not configured for: {storage_name}",
                    affected_components=[storage_name],
                    weak_algorithms=[],
                    compliance_impact="medium",
                    remediation_required=True,
                    remediation_timeframe="60 days",
                    evidence=[{"key_management": key_management}],
                    security_impact="medium",
                    details={"storage_name": storage_name, "rotation_status": "manual"}
                ))
            
            # Check backup encryption
            backup_config = config.get("backup_configuration", {})
            if not backup_config.get("encrypted", False):
                violations.append(EncryptionViolation(
                    violation_type="unencrypted_backups",
                    severity="high",
                    description=f"Backups not encrypted for storage: {storage_name}",
                    affected_components=[storage_name, "backup_system"],
                    weak_algorithms=[],
                    compliance_impact="high",
                    remediation_required=True,
                    remediation_timeframe="14 days",
                    evidence=[{"backup_config": backup_config}],
                    security_impact="high",
                    details={"storage_name": storage_name, "backup_encryption": "disabled"}
                ))
        
        return violations
    
    def test_data_encryption_in_transit(self, tls_configurations: List[TLSConfiguration]) -> List[EncryptionViolation]:
        """
        Test data encryption in transit implementation.
        
        Args:
            tls_configurations: List of TLS configurations
            
        Returns:
            List of encryption in transit violations
        """
        violations = []
        
        for config in tls_configurations:
            endpoint = f"TLS_Config_{tls_configurations.index(config)}"
            
            # Check TLS version
            if config.protocol_version in self.deprecated_tls_versions:
                violations.append(EncryptionViolation(
                    violation_type="deprecated_tls_version",
                    severity="high",
                    description=f"Deprecated TLS version in use: {config.protocol_version.value}",
                    affected_components=[endpoint],
                    weak_algorithms=[config.protocol_version.value],
                    compliance_impact="high",
                    remediation_required=True,
                    remediation_timeframe="30 days",
                    evidence=[{"tls_config": config.__dict__}],
                    security_impact="high",
                    details={"current_version": config.protocol_version.value, "minimum_version": "TLS 1.2"}
                ))
            
            # Check cipher suites
            weak_ciphers_found = [cipher for cipher in config.cipher_suites if cipher in self.weak_cipher_suites]
            if weak_ciphers_found:
                violations.append(EncryptionViolation(
                    violation_type="weak_cipher_suites",
                    severity="high",
                    description=f"Weak cipher suites configured: {weak_ciphers_found}",
                    affected_components=[endpoint],
                    weak_algorithms=weak_ciphers_found,
                    compliance_impact="high",
                    remediation_required=True,
                    remediation_timeframe="14 days",
                    evidence=[{"weak_ciphers": weak_ciphers_found}],
                    security_impact="high",
                    details={"weak_ciphers": weak_ciphers_found, "secure_ciphers": self.secure_cipher_suites}
                ))
            
            # Check perfect forward secrecy
            if not config.perfect_forward_secrecy:
                violations.append(EncryptionViolation(
                    violation_type="perfect_forward_secrecy_disabled",
                    severity="medium",
                    description="Perfect Forward Secrecy not enabled",
                    affected_components=[endpoint],
                    weak_algorithms=[],
                    compliance_impact="medium",
                    remediation_required=True,
                    remediation_timeframe="30 days",
                    evidence=[{"pfs_status": config.perfect_forward_secrecy}],
                    security_impact="medium",
                    details={"pfs_enabled": config.perfect_forward_secrecy}
                ))
            
            # Check HSTS configuration
            if not config.hsts_enabled:
                violations.append(EncryptionViolation(
                    violation_type="hsts_not_enabled",
                    severity="medium",
                    description="HTTP Strict Transport Security (HSTS) not enabled",
                    affected_components=[endpoint],
                    weak_algorithms=[],
                    compliance_impact="medium",
                    remediation_required=True,
                    remediation_timeframe="14 days",
                    evidence=[{"hsts_status": config.hsts_enabled}],
                    security_impact="medium",
                    details={"hsts_enabled": config.hsts_enabled, "recommended_max_age": 31536000}
                ))
            elif config.hsts_max_age and config.hsts_max_age < 31536000:  # 1 year
                violations.append(EncryptionViolation(
                    violation_type="insufficient_hsts_max_age",
                    severity="low",
                    description=f"HSTS max-age too short: {config.hsts_max_age} seconds",
                    affected_components=[endpoint],
                    weak_algorithms=[],
                    compliance_impact="low",
                    remediation_required=True,
                    remediation_timeframe="30 days",
                    evidence=[{"hsts_max_age": config.hsts_max_age}],
                    security_impact="low",
                    details={"current_max_age": config.hsts_max_age, "recommended_max_age": 31536000}
                ))
        
        return violations
    
    def test_certificate_validation(self, certificates: List[Certificate]) -> List[EncryptionViolation]:
        """
        Test digital certificate validation.
        
        Args:
            certificates: List of certificates to validate
            
        Returns:
            List of certificate validation violations
        """
        violations = []
        
        for cert in certificates:
            # Check certificate expiry
            try:
                not_after = datetime.fromisoformat(cert.not_after.replace("Z", "+00:00"))
                days_until_expiry = (not_after - datetime.now()).days
                
                if days_until_expiry < 0:
                    violations.append(EncryptionViolation(
                        violation_type="expired_certificate",
                        severity="critical",
                        description=f"Certificate expired: {cert.subject}",
                        affected_components=[cert.subject],
                        weak_algorithms=[],
                        compliance_impact="critical",
                        remediation_required=True,
                        remediation_timeframe="immediate",
                        evidence=[{"certificate": cert.__dict__}],
                        security_impact="critical",
                        details={"subject": cert.subject, "expiry_date": cert.not_after, "days_overdue": abs(days_until_expiry)}
                    ))
                elif days_until_expiry < 30:
                    violations.append(EncryptionViolation(
                        violation_type="certificate_expiring_soon",
                        severity="high",
                        description=f"Certificate expiring in {days_until_expiry} days: {cert.subject}",
                        affected_components=[cert.subject],
                        weak_algorithms=[],
                        compliance_impact="medium",
                        remediation_required=True,
                        remediation_timeframe="immediate",
                        evidence=[{"certificate": cert.__dict__}],
                        security_impact="medium",
                        details={"subject": cert.subject, "expiry_date": cert.not_after, "days_remaining": days_until_expiry}
                    ))
            
            except ValueError:
                violations.append(EncryptionViolation(
                    violation_type="invalid_certificate_date",
                    severity="medium",
                    description=f"Invalid certificate date format: {cert.not_after}",
                    affected_components=[cert.subject],
                    weak_algorithms=[],
                    compliance_impact="medium",
                    remediation_required=True,
                    remediation_timeframe="7 days",
                    evidence=[{"certificate": cert.__dict__}],
                    security_impact="medium",
                    details={"subject": cert.subject, "invalid_date": cert.not_after}
                ))
            
            # Check key algorithm and size
            if "rsa" in cert.key_algorithm.lower():
                if cert.key_size < 2048:
                    violations.append(EncryptionViolation(
                        violation_type="weak_certificate_key_size",
                        severity="high",
                        description=f"RSA key size too small: {cert.key_size} bits (minimum: 2048)",
                        affected_components=[cert.subject],
                        weak_algorithms=[f"RSA-{cert.key_size}"],
                        compliance_impact="high",
                        remediation_required=True,
                        remediation_timeframe="60 days",
                        evidence=[{"certificate": cert.__dict__}],
                        security_impact="high",
                        details={"current_key_size": cert.key_size, "minimum_key_size": 2048}
                    ))
            
            # Check signature algorithm
            if any(weak_alg in cert.signature_algorithm.lower() for weak_alg in ["md5", "sha1"]):
                violations.append(EncryptionViolation(
                    violation_type="weak_signature_algorithm",
                    severity="high",
                    description=f"Weak signature algorithm: {cert.signature_algorithm}",
                    affected_components=[cert.subject],
                    weak_algorithms=[cert.signature_algorithm],
                    compliance_impact="high",
                    remediation_required=True,
                    remediation_timeframe="30 days",
                    evidence=[{"certificate": cert.__dict__}],
                    security_impact="high",
                    details={"current_algorithm": cert.signature_algorithm, "recommended_algorithms": ["SHA-256", "SHA-384", "SHA-512"]}
                ))
            
            # Check key usage restrictions
            if cert.certificate_type == CertificateType.SERVER_CERTIFICATE:
                required_key_usage = [KeyUsage.DIGITAL_SIGNATURE, KeyUsage.KEY_ENCIPHERMENT]
                missing_usage = [usage for usage in required_key_usage if usage not in cert.key_usage]
                
                if missing_usage:
                    violations.append(EncryptionViolation(
                        violation_type="insufficient_key_usage",
                        severity="medium",
                        description=f"Certificate missing required key usage: {missing_usage}",
                        affected_components=[cert.subject],
                        weak_algorithms=[],
                        compliance_impact="medium",
                        remediation_required=True,
                        remediation_timeframe="30 days",
                        evidence=[{"certificate": cert.__dict__, "missing_usage": [usage.value for usage in missing_usage]}],
                        security_impact="medium",
                        details={"missing_key_usage": [usage.value for usage in missing_usage]}
                    ))
        
        return violations
    
    def test_key_management_security(self, key_management_configs: List[Dict[str, Any]]) -> List[EncryptionViolation]:
        """
        Test key management security practices.
        
        Args:
            key_management_configs: List of key management configurations
            
        Returns:
            List of key management violations
        """
        violations = []
        
        for config in key_management_configs:
            system_name = config.get("system_name", "unknown")
            
            # Check key rotation policy
            rotation_period_days = config.get("rotation_period_days", 0)
            if rotation_period_days == 0:
                violations.append(EncryptionViolation(
                    violation_type="no_key_rotation_policy",
                    severity="high",
                    description=f"No key rotation policy configured for: {system_name}",
                    affected_components=[system_name],
                    weak_algorithms=[],
                    compliance_impact="high",
                    remediation_required=True,
                    remediation_timeframe="30 days",
                    evidence=[{"config": config}],
                    security_impact="high",
                    details={"system_name": system_name, "recommended_rotation_days": 365}
                ))
            elif rotation_period_days > self.key_management_requirements["rotation_period_days"]:
                violations.append(EncryptionViolation(
                    violation_type="excessive_key_rotation_period",
                    severity="medium",
                    description=f"Key rotation period too long: {rotation_period_days} days",
                    affected_components=[system_name],
                    weak_algorithms=[],
                    compliance_impact="medium",
                    remediation_required=True,
                    remediation_timeframe="60 days",
                    evidence=[{"config": config}],
                    security_impact="medium",
                    details={"current_period": rotation_period_days, "maximum_period": self.key_management_requirements["rotation_period_days"]}
                ))
            
            # Check key storage security
            key_storage = config.get("key_storage", {})
            if not key_storage.get("hardware_security_module", False):
                violations.append(EncryptionViolation(
                    violation_type="key_storage_not_in_hsm",
                    severity="medium",
                    description=f"Keys not stored in Hardware Security Module: {system_name}",
                    affected_components=[system_name],
                    weak_algorithms=[],
                    compliance_impact="medium",
                    remediation_required=True,
                    remediation_timeframe="90 days",
                    evidence=[{"key_storage": key_storage}],
                    security_impact="medium",
                    details={"system_name": system_name, "hsm_required": True}
                ))
            
            # Check key backup encryption
            backup_encryption = config.get("backup_encryption", {})
            if not backup_encryption.get("enabled", False):
                violations.append(EncryptionViolation(
                    violation_type="key_backup_not_encrypted",
                    severity="high",
                    description=f"Key backups not encrypted: {system_name}",
                    affected_components=[system_name],
                    weak_algorithms=[],
                    compliance_impact="high",
                    remediation_required=True,
                    remediation_timeframe="14 days",
                    evidence=[{"backup_encryption": backup_encryption}],
                    security_impact="high",
                    details={"system_name": system_name, "backup_encryption_required": True}
                ))
            
            # Check access controls
            access_controls = config.get("access_controls", {})
            if not access_controls.get("multi_person_control", False):
                violations.append(EncryptionViolation(
                    violation_type="insufficient_key_access_control",
                    severity="medium",
                    description=f"Multi-person control not implemented for key access: {system_name}",
                    affected_components=[system_name],
                    weak_algorithms=[],
                    compliance_impact="medium",
                    remediation_required=True,
                    remediation_timeframe="30 days",
                    evidence=[{"access_controls": access_controls}],
                    security_impact="medium",
                    details={"system_name": system_name, "multi_person_control_required": True}
                ))
        
        return violations
    
    def test_cryptographic_implementations(self, crypto_implementations: List[Dict[str, Any]]) -> List[EncryptionViolation]:
        """
        Test cryptographic implementation security.
        
        Args:
            crypto_implementations: List of cryptographic implementations
            
        Returns:
            List of cryptographic implementation violations
        """
        violations = []
        
        for impl in crypto_implementations:
            component_name = impl.get("component_name", "unknown")
            
            # Check for custom cryptographic implementations
            is_custom = impl.get("custom_implementation", False)
            if is_custom:
                violations.append(EncryptionViolation(
                    violation_type="custom_cryptographic_implementation",
                    severity="high",
                    description=f"Custom cryptographic implementation detected: {component_name}",
                    affected_components=[component_name],
                    weak_algorithms=[],
                    compliance_impact="high",
                    remediation_required=True,
                    remediation_timeframe="60 days",
                    evidence=[{"implementation": impl}],
                    security_impact="high",
                    details={"component_name": component_name, "recommendation": "Use standard libraries"}
                ))
            
            # Check random number generation
            rng_config = impl.get("random_number_generation", {})
            rng_source = rng_config.get("source", "").lower()
            
            if "pseudo" in rng_source or "deterministic" in rng_source:
                violations.append(EncryptionViolation(
                    violation_type="weak_random_number_generation",
                    severity="high",
                    description=f"Weak random number generation: {rng_source}",
                    affected_components=[component_name],
                    weak_algorithms=[rng_source],
                    compliance_impact="high",
                    remediation_required=True,
                    remediation_timeframe="30 days",
                    evidence=[{"rng_config": rng_config}],
                    security_impact="high",
                    details={"current_source": rng_source, "recommended_source": "Cryptographically secure random"}
                ))
            
            # Check for hardcoded keys or secrets
            has_hardcoded_secrets = impl.get("hardcoded_secrets", False)
            if has_hardcoded_secrets:
                violations.append(EncryptionViolation(
                    violation_type="hardcoded_cryptographic_secrets",
                    severity="critical",
                    description=f"Hardcoded cryptographic secrets detected: {component_name}",
                    affected_components=[component_name],
                    weak_algorithms=[],
                    compliance_impact="critical",
                    remediation_required=True,
                    remediation_timeframe="immediate",
                    evidence=[{"component": component_name}],
                    security_impact="critical",
                    details={"component_name": component_name, "remediation": "Use secure key management"}
                ))
            
            # Check initialization vector (IV) generation
            iv_config = impl.get("iv_generation", {})
            if iv_config.get("reused", False):
                violations.append(EncryptionViolation(
                    violation_type="iv_reuse_detected",
                    severity="high",
                    description=f"Initialization Vector reuse detected: {component_name}",
                    affected_components=[component_name],
                    weak_algorithms=[],
                    compliance_impact="high",
                    remediation_required=True,
                    remediation_timeframe="14 days",
                    evidence=[{"iv_config": iv_config}],
                    security_impact="high",
                    details={"component_name": component_name, "requirement": "Unique IV per encryption operation"}
                ))
        
        return violations
    
    def conduct_comprehensive_encryption_assessment(self, 
                                                  storage_configs: List[Dict[str, Any]],
                                                  tls_configs: List[TLSConfiguration],
                                                  certificates: List[Certificate],
                                                  key_management_configs: List[Dict[str, Any]],
                                                  crypto_implementations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Conduct comprehensive encryption security assessment.
        
        Args:
            storage_configs: Storage encryption configurations
            tls_configs: TLS configurations
            certificates: Digital certificates
            key_management_configs: Key management configurations
            crypto_implementations: Cryptographic implementations
            
        Returns:
            Comprehensive encryption assessment results
        """
        assessment_id = f"encryption_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        all_violations = []
        
        # Run all encryption security tests
        at_rest_violations = self.test_data_encryption_at_rest(storage_configs)
        all_violations.extend(at_rest_violations)
        
        in_transit_violations = self.test_data_encryption_in_transit(tls_configs)
        all_violations.extend(in_transit_violations)
        
        certificate_violations = self.test_certificate_validation(certificates)
        all_violations.extend(certificate_violations)
        
        key_mgmt_violations = self.test_key_management_security(key_management_configs)
        all_violations.extend(key_mgmt_violations)
        
        crypto_impl_violations = self.test_cryptographic_implementations(crypto_implementations)
        all_violations.extend(crypto_impl_violations)
        
        # Categorise violations by severity
        violations_by_severity = {
            "critical": [v for v in all_violations if v.severity == "critical"],
            "high": [v for v in all_violations if v.severity == "high"],
            "medium": [v for v in all_violations if v.severity == "medium"],
            "low": [v for v in all_violations if v.severity == "low"]
        }
        
        # Calculate encryption strength metrics
        encryption_metrics = self._calculate_encryption_metrics(
            storage_configs, tls_configs, certificates, key_management_configs
        )
        
        # Generate recommendations
        recommendations = self._generate_encryption_recommendations(all_violations, violations_by_severity)
        
        return {
            "assessment_id": assessment_id,
            "assessment_timestamp": datetime.now().isoformat(),
            "total_violations": len(all_violations),
            "violations_by_severity": {
                severity: len(violations) for severity, violations in violations_by_severity.items()
            },
            "violations_by_category": {
                "at_rest": len(at_rest_violations),
                "in_transit": len(in_transit_violations),
                "certificates": len(certificate_violations),
                "key_management": len(key_mgmt_violations),
                "crypto_implementation": len(crypto_impl_violations)
            },
            "encryption_metrics": encryption_metrics,
            "security_posture": {
                "overall_encryption_strength": self._calculate_overall_strength(encryption_metrics, all_violations),
                "compliance_status": self._assess_compliance_status(all_violations),
                "immediate_risks": len(violations_by_severity["critical"]) + len(violations_by_severity["high"])
            },
            "recommendations": recommendations,
            "detailed_violations": [violation.__dict__ for violation in all_violations],
            "next_assessment_date": (datetime.now() + timedelta(days=90)).isoformat()
        }
    
    def _calculate_encryption_metrics(self, storage_configs: List[Dict[str, Any]], 
                                    tls_configs: List[TLSConfiguration],
                                    certificates: List[Certificate],
                                    key_management_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate encryption strength metrics."""
        metrics = {
            "storage_encryption_coverage": 0.0,
            "strong_algorithm_usage": 0.0,
            "certificate_health": 0.0,
            "key_management_maturity": 0.0
        }
        
        # Storage encryption coverage
        if storage_configs:
            encrypted_storages = len([s for s in storage_configs if s.get("encryption_enabled", False)])
            metrics["storage_encryption_coverage"] = encrypted_storages / len(storage_configs)
        
        # Strong algorithm usage
        total_configs = len(storage_configs) + len(tls_configs)
        if total_configs > 0:
            strong_algorithms = 0
            
            for config in storage_configs:
                algorithm = config.get("encryption_algorithm", "").lower()
                if "aes-256" in algorithm or "chacha20" in algorithm:
                    strong_algorithms += 1
            
            for config in tls_configs:
                if config.protocol_version in self.approved_tls_versions:
                    strong_algorithms += 1
            
            metrics["strong_algorithm_usage"] = strong_algorithms / total_configs
        
        # Certificate health
        if certificates:
            healthy_certs = 0
            for cert in certificates:
                try:
                    not_after = datetime.fromisoformat(cert.not_after.replace("Z", "+00:00"))
                    days_until_expiry = (not_after - datetime.now()).days
                    
                    if days_until_expiry > 30 and cert.key_size >= 2048:
                        healthy_certs += 1
                except:
                    pass
            
            metrics["certificate_health"] = healthy_certs / len(certificates)
        
        # Key management maturity
        if key_management_configs:
            mature_configs = 0
            for config in key_management_configs:
                score = 0
                if config.get("rotation_period_days", 0) > 0:
                    score += 1
                if config.get("key_storage", {}).get("hardware_security_module", False):
                    score += 1
                if config.get("backup_encryption", {}).get("enabled", False):
                    score += 1
                if config.get("access_controls", {}).get("multi_person_control", False):
                    score += 1
                
                if score >= 3:  # Considered mature if 3+ requirements met
                    mature_configs += 1
            
            metrics["key_management_maturity"] = mature_configs / len(key_management_configs)
        
        return metrics
    
    def _calculate_overall_strength(self, metrics: Dict[str, Any], violations: List[EncryptionViolation]) -> str:
        """Calculate overall encryption strength."""
        # Calculate weighted score
        weights = {
            "storage_encryption_coverage": 0.3,
            "strong_algorithm_usage": 0.3,
            "certificate_health": 0.2,
            "key_management_maturity": 0.2
        }
        
        score = sum(metrics.get(metric, 0) * weight for metric, weight in weights.items())
        
        # Adjust for critical violations
        critical_violations = len([v for v in violations if v.severity == "critical"])
        if critical_violations > 0:
            score *= 0.5  # Significant penalty for critical violations
        
        high_violations = len([v for v in violations if v.severity == "high"])
        if high_violations > 3:
            score *= 0.8  # Moderate penalty for multiple high violations
        
        # Determine strength level
        if score >= 0.9:
            return "excellent"
        elif score >= 0.7:
            return "good"
        elif score >= 0.5:
            return "adequate"
        elif score >= 0.3:
            return "weak"
        else:
            return "poor"
    
    def _assess_compliance_status(self, violations: List[EncryptionViolation]) -> str:
        """Assess encryption compliance status."""
        critical_violations = len([v for v in violations if v.severity == "critical"])
        high_violations = len([v for v in violations if v.severity == "high"])
        
        if critical_violations > 0:
            return "non_compliant"
        elif high_violations > 5:
            return "partially_compliant"
        elif high_violations > 0:
            return "mostly_compliant"
        else:
            return "compliant"
    
    def _generate_encryption_recommendations(self, all_violations: List[EncryptionViolation], 
                                           violations_by_severity: Dict[str, List]) -> List[str]:
        """Generate encryption security recommendations."""
        recommendations = []
        
        if violations_by_severity["critical"]:
            recommendations.append("URGENT: Address critical encryption vulnerabilities immediately")
        
        violation_types = [v.violation_type for v in all_violations]
        
        if "encryption_at_rest_disabled" in violation_types:
            recommendations.append("Enable AES-256-GCM encryption for all data storage systems")
        
        if "deprecated_tls_version" in violation_types:
            recommendations.append("Upgrade to TLS 1.3 for all network communications")
        
        if "weak_cipher_suites" in violation_types:
            recommendations.append("Configure secure cipher suites and disable weak algorithms")
        
        if "expired_certificate" in violation_types or "certificate_expiring_soon" in violation_types:
            recommendations.append("Implement automated certificate lifecycle management")
        
        if "no_key_rotation_policy" in violation_types:
            recommendations.append("Establish automated key rotation policies")
        
        if "custom_cryptographic_implementation" in violation_types:
            recommendations.append("Replace custom crypto implementations with standard libraries")
        
        if len(all_violations) > 15:
            recommendations.append("Consider comprehensive encryption architecture review")
        
        return recommendations


# Test suite
class TestEncryptionSecurity:
    """Test suite for encryption security."""
    
    @pytest.fixture
    def encryption_tester(self):
        """Create encryption security tester instance."""
        return EncryptionSecurityTester()
    
    @pytest.fixture
    def test_storage_configs(self):
        """Test storage configurations."""
        return [
            {
                "storage_name": "primary_database",
                "encryption_enabled": True,
                "encryption_algorithm": "AES-256-GCM",
                "key_size": 256,
                "key_management": {
                    "automated_rotation": True,
                    "rotation_period_days": 365
                },
                "backup_configuration": {
                    "encrypted": True,
                    "encryption_algorithm": "AES-256-GCM"
                }
            },
            {
                "storage_name": "file_storage",
                "encryption_enabled": False,  # Violation
                "encryption_algorithm": "",
                "key_size": 0,
                "key_management": {},
                "backup_configuration": {"encrypted": False}
            },
            {
                "storage_name": "legacy_system",
                "encryption_enabled": True,
                "encryption_algorithm": "DES",  # Weak algorithm
                "key_size": 56,
                "key_management": {
                    "automated_rotation": False,  # Manual rotation
                    "rotation_period_days": 0
                },
                "backup_configuration": {"encrypted": False}
            }
        ]
    
    @pytest.fixture
    def test_tls_configs(self):
        """Test TLS configurations."""
        return [
            TLSConfiguration(
                protocol_version=TLSVersion.TLS_1_3,
                cipher_suites=["TLS_AES_256_GCM_SHA384", "TLS_CHACHA20_POLY1305_SHA256"],
                supported_groups=["x25519", "secp256r1"],
                signature_algorithms=["rsa_pss_rsae_sha256", "ecdsa_secp256r1_sha256"],
                certificate_chain=[],
                hsts_enabled=True,
                hsts_max_age=31536000,
                ocsp_stapling=True,
                perfect_forward_secrecy=True
            ),
            TLSConfiguration(
                protocol_version=TLSVersion.TLS_1_1,  # Deprecated
                cipher_suites=["TLS_RSA_WITH_RC4_128_SHA"],  # Weak cipher
                supported_groups=["secp256r1"],
                signature_algorithms=["rsa_pkcs1_sha1"],
                certificate_chain=[],
                hsts_enabled=False,  # Not enabled
                hsts_max_age=None,
                ocsp_stapling=False,
                perfect_forward_secrecy=False  # Not enabled
            )
        ]
    
    @pytest.fixture
    def test_certificates(self):
        """Test certificates."""
        now = datetime.now()
        return [
            Certificate(
                certificate_id="cert001",
                subject="CN=api.healthanalytics.com",
                issuer="CN=GlobalSign RSA OV SSL CA 2018",
                serial_number="123456789",
                not_before=now.isoformat(),
                not_after=(now + timedelta(days=365)).isoformat(),
                certificate_type=CertificateType.SERVER_CERTIFICATE,
                key_algorithm="RSA",
                key_size=2048,
                signature_algorithm="SHA256WithRSA",
                key_usage=[KeyUsage.DIGITAL_SIGNATURE, KeyUsage.KEY_ENCIPHERMENT],
                extended_key_usage=["serverAuth"],
                san_domains=["api.healthanalytics.com", "www.healthanalytics.com"],
                is_ca=False,
                path_length_constraint=None,
                fingerprint_sha256="abc123def456"
            ),
            Certificate(
                certificate_id="cert002",
                subject="CN=expired.example.com",
                issuer="CN=Test CA",
                serial_number="987654321",
                not_before=(now - timedelta(days=400)).isoformat(),
                not_after=(now - timedelta(days=1)).isoformat(),  # Expired
                certificate_type=CertificateType.SERVER_CERTIFICATE,
                key_algorithm="RSA",
                key_size=1024,  # Weak key size
                signature_algorithm="MD5WithRSA",  # Weak signature
                key_usage=[KeyUsage.DIGITAL_SIGNATURE],  # Missing key usage
                extended_key_usage=["serverAuth"],
                san_domains=["expired.example.com"],
                is_ca=False,
                path_length_constraint=None,
                fingerprint_sha256="def456ghi789"
            )
        ]
    
    @pytest.fixture
    def test_key_management_configs(self):
        """Test key management configurations."""
        return [
            {
                "system_name": "primary_kms",
                "rotation_period_days": 365,
                "key_storage": {
                    "hardware_security_module": True,
                    "vendor": "AWS CloudHSM"
                },
                "backup_encryption": {
                    "enabled": True,
                    "algorithm": "AES-256-GCM"
                },
                "access_controls": {
                    "multi_person_control": True,
                    "audit_logging": True
                }
            },
            {
                "system_name": "legacy_kms",
                "rotation_period_days": 0,  # No rotation policy
                "key_storage": {
                    "hardware_security_module": False  # Not using HSM
                },
                "backup_encryption": {
                    "enabled": False  # Not encrypted
                },
                "access_controls": {
                    "multi_person_control": False  # Insufficient controls
                }
            }
        ]
    
    @pytest.fixture
    def test_crypto_implementations(self):
        """Test cryptographic implementations."""
        return [
            {
                "component_name": "data_processor",
                "custom_implementation": False,
                "random_number_generation": {
                    "source": "cryptographically_secure_random",
                    "library": "secrets"
                },
                "hardcoded_secrets": False,
                "iv_generation": {
                    "unique_per_operation": True,
                    "reused": False
                }
            },
            {
                "component_name": "legacy_component",
                "custom_implementation": True,  # Custom crypto
                "random_number_generation": {
                    "source": "pseudo_random",  # Weak RNG
                    "library": "random"
                },
                "hardcoded_secrets": True,  # Hardcoded secrets
                "iv_generation": {
                    "unique_per_operation": False,
                    "reused": True  # IV reuse
                }
            }
        ]
    
    def test_data_encryption_at_rest(self, encryption_tester, test_storage_configs):
        """Test data encryption at rest validation."""
        violations = encryption_tester.test_data_encryption_at_rest(test_storage_configs)
        
        # Should detect multiple violations
        assert len(violations) > 0, "Should detect encryption at rest violations"
        
        violation_types = [v.violation_type for v in violations]
        expected_types = [
            "encryption_at_rest_disabled",
            "weak_encryption_algorithm", 
            "key_rotation_not_automated",
            "unencrypted_backups"
        ]
        
        detected_types = [vtype for vtype in expected_types if vtype in violation_types]
        assert len(detected_types) > 0, "Should detect specific encryption violations"
        
        # Check for critical violations
        critical_violations = [v for v in violations if v.severity == "critical"]
        assert len(critical_violations) > 0, "Should have critical violations for disabled encryption"
    
    def test_data_encryption_in_transit(self, encryption_tester, test_tls_configs):
        """Test data encryption in transit validation."""
        violations = encryption_tester.test_data_encryption_in_transit(test_tls_configs)
        
        # Should detect TLS violations
        assert len(violations) > 0, "Should detect TLS configuration violations"
        
        violation_types = [v.violation_type for v in violations]
        expected_types = [
            "deprecated_tls_version",
            "weak_cipher_suites",
            "perfect_forward_secrecy_disabled",
            "hsts_not_enabled"
        ]
        
        detected_types = [vtype for vtype in expected_types if vtype in violation_types]
        assert len(detected_types) > 0, "Should detect TLS configuration issues"
        
        # Check severity levels
        high_violations = [v for v in violations if v.severity == "high"]
        assert len(high_violations) > 0, "Should have high severity violations for weak TLS"
    
    def test_certificate_validation(self, encryption_tester, test_certificates):
        """Test certificate validation."""
        violations = encryption_tester.test_certificate_validation(test_certificates)
        
        # Should detect certificate issues
        assert len(violations) > 0, "Should detect certificate violations"
        
        violation_types = [v.violation_type for v in violations]
        expected_types = [
            "expired_certificate",
            "weak_certificate_key_size",
            "weak_signature_algorithm",
            "insufficient_key_usage"
        ]
        
        detected_types = [vtype for vtype in expected_types if vtype in violation_types]
        assert len(detected_types) > 0, "Should detect certificate validation issues"
        
        # Check for critical violations (expired certificates)
        critical_violations = [v for v in violations if v.severity == "critical"]
        if critical_violations:
            assert any("expired" in v.violation_type for v in critical_violations)
    
    def test_key_management_security(self, encryption_tester, test_key_management_configs):
        """Test key management security."""
        violations = encryption_tester.test_key_management_security(test_key_management_configs)
        
        # Should detect key management violations
        assert len(violations) > 0, "Should detect key management violations"
        
        violation_types = [v.violation_type for v in violations]
        expected_types = [
            "no_key_rotation_policy",
            "key_storage_not_in_hsm",
            "key_backup_not_encrypted",
            "insufficient_key_access_control"
        ]
        
        detected_types = [vtype for vtype in expected_types if vtype in violation_types]
        assert len(detected_types) > 0, "Should detect key management issues"
        
        # Verify violation details
        for violation in violations:
            assert violation.severity in ["critical", "high", "medium", "low"]
            assert violation.remediation_required in [True, False]
    
    def test_cryptographic_implementations(self, encryption_tester, test_crypto_implementations):
        """Test cryptographic implementation security."""
        violations = encryption_tester.test_cryptographic_implementations(test_crypto_implementations)
        
        # Should detect crypto implementation violations
        assert len(violations) > 0, "Should detect cryptographic implementation violations"
        
        violation_types = [v.violation_type for v in violations]
        expected_types = [
            "custom_cryptographic_implementation",
            "weak_random_number_generation",
            "hardcoded_cryptographic_secrets",
            "iv_reuse_detected"
        ]
        
        detected_types = [vtype for vtype in expected_types if vtype in violation_types]
        assert len(detected_types) > 0, "Should detect crypto implementation issues"
        
        # Check for critical violations (hardcoded secrets)
        critical_violations = [v for v in violations if v.severity == "critical"]
        assert len(critical_violations) > 0, "Should have critical violations for hardcoded secrets"
    
    def test_comprehensive_encryption_assessment(self, encryption_tester, test_storage_configs, 
                                                test_tls_configs, test_certificates, 
                                                test_key_management_configs, test_crypto_implementations):
        """Test comprehensive encryption assessment."""
        assessment = encryption_tester.conduct_comprehensive_encryption_assessment(
            test_storage_configs, test_tls_configs, test_certificates, 
            test_key_management_configs, test_crypto_implementations
        )
        
        # Verify assessment structure
        required_keys = [
            "assessment_id", "assessment_timestamp", "total_violations",
            "violations_by_severity", "violations_by_category", "encryption_metrics",
            "security_posture", "recommendations", "detailed_violations", "next_assessment_date"
        ]
        
        for key in required_keys:
            assert key in assessment, f"Assessment should include {key}"
        
        # Verify assessment quality
        assert assessment["total_violations"] > 0, "Should detect violations in test data"
        assert "overall_encryption_strength" in assessment["security_posture"]
        assert assessment["security_posture"]["overall_encryption_strength"] in ["excellent", "good", "adequate", "weak", "poor"]
        assert len(assessment["recommendations"]) > 0, "Should provide recommendations"
        
        # Verify encryption metrics
        metrics = assessment["encryption_metrics"]
        for metric_name, metric_value in metrics.items():
            assert 0.0 <= metric_value <= 1.0, f"Metric {metric_name} should be between 0 and 1"
    
    def test_approved_algorithms_configuration(self, encryption_tester):
        """Test approved algorithms configuration."""
        approved_algorithms = encryption_tester.approved_algorithms
        
        # Verify strong algorithms are approved
        strong_algorithms = [
            EncryptionAlgorithm.AES_256_GCM,
            EncryptionAlgorithm.CHACHA20_POLY1305,
            EncryptionAlgorithm.RSA_4096
        ]
        
        for algorithm in strong_algorithms:
            assert algorithm in approved_algorithms, f"Strong algorithm {algorithm} should be approved"
            assert approved_algorithms[algorithm]["min_key_size"] > 0, f"Algorithm {algorithm} should have minimum key size"
        
        # Verify deprecated algorithms are identified
        deprecated_algorithms = encryption_tester.deprecated_algorithms
        weak_algorithms = ["des", "3des", "rc4", "md5", "sha1"]
        
        for weak_alg in weak_algorithms:
            assert weak_alg in deprecated_algorithms, f"Weak algorithm {weak_alg} should be in deprecated list"
    
    def test_tls_configuration_validation(self, encryption_tester):
        """Test TLS configuration validation."""
        # Verify secure TLS versions
        approved_versions = encryption_tester.approved_tls_versions
        assert TLSVersion.TLS_1_2 in approved_versions, "TLS 1.2 should be approved"
        assert TLSVersion.TLS_1_3 in approved_versions, "TLS 1.3 should be approved"
        
        # Verify deprecated versions
        deprecated_versions = encryption_tester.deprecated_tls_versions
        assert TLSVersion.TLS_1_0 in deprecated_versions, "TLS 1.0 should be deprecated"
        assert TLSVersion.TLS_1_1 in deprecated_versions, "TLS 1.1 should be deprecated"
        
        # Verify secure cipher suites
        secure_ciphers = encryption_tester.secure_cipher_suites
        assert "TLS_AES_256_GCM_SHA384" in secure_ciphers, "Strong cipher should be in secure list"
        
        # Verify weak cipher suites
        weak_ciphers = encryption_tester.weak_cipher_suites
        assert "TLS_RSA_WITH_RC4_128_SHA" in weak_ciphers, "Weak cipher should be identified"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])