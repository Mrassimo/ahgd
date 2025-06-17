"""
Australian Privacy Principles (APP) Compliance Testing

Comprehensive testing suite for Australian Privacy Principles compliance including:
- APP 1-13 comprehensive compliance testing
- Privacy policy and consent management validation
- Data collection limitation testing
- Purpose specification and use limitation validation
- Cross-border data transfer restrictions testing
- Privacy impact assessment validation

This test suite ensures all Australian health data processing complies with
the Privacy Act 1988 and Australian Privacy Principles.
"""

import json
import pytest
import hashlib
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Union, Any
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass, field
from enum import Enum
import logging

import polars as pl
import numpy as np
from loguru import logger


class APPRequirement(Enum):
    """Australian Privacy Principles enumeration."""
    APP1 = "Open and transparent management of personal information"
    APP2 = "Anonymity and pseudonymity"
    APP3 = "Collection of solicited personal information"
    APP4 = "Dealing with unsolicited personal information"
    APP5 = "Notification of the collection of personal information"
    APP6 = "Use or disclosure of personal information"
    APP7 = "Direct marketing"
    APP8 = "Cross-border disclosure of personal information"
    APP9 = "Adoption, use or disclosure by individuals"
    APP10 = "Quality of personal information"
    APP11 = "Security of personal information"
    APP12 = "Access to personal information"
    APP13 = "Correction of personal information"


class ComplianceStatus(Enum):
    """Compliance status levels."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_APPLICABLE = "not_applicable"
    REQUIRES_REVIEW = "requires_review"


class DataHandlingPurpose(Enum):
    """Valid data handling purposes."""
    HEALTH_RESEARCH = "health_research"
    SERVICE_DELIVERY = "service_delivery"
    QUALITY_IMPROVEMENT = "quality_improvement"
    PUBLIC_HEALTH_MONITORING = "public_health_monitoring"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    STATISTICAL_ANALYSIS = "statistical_analysis"


@dataclass
class ConsentRecord:
    """Individual consent record."""
    individual_id: str
    consent_timestamp: str
    purposes_consented: List[DataHandlingPurpose]
    withdrawal_mechanism: str
    consent_evidence: str
    expiry_date: Optional[str] = None
    withdrawal_date: Optional[str] = None


@dataclass
class PrivacyNotice:
    """Privacy notice/policy record."""
    notice_id: str
    version: str
    effective_date: str
    content: Dict[str, Any]
    notification_method: str
    acknowledgment_required: bool
    last_updated: str


@dataclass
class CrossBorderTransfer:
    """Cross-border data transfer record."""
    transfer_id: str
    destination_country: str
    recipient_organisation: str
    transfer_purpose: DataHandlingPurpose
    adequacy_decision: bool
    safeguards_implemented: List[str]
    transfer_date: str
    approval_reference: Optional[str] = None


@dataclass
class APPComplianceViolation:
    """APP compliance violation."""
    app_principle: APPRequirement
    violation_type: str
    severity: str  # "critical", "high", "medium", "low"
    description: str
    affected_data_elements: List[str]
    remediation_required: bool
    remediation_timeframe: str
    regulatory_risk: str
    sample_evidence: List[Dict[str, Any]]
    compliance_standard: str
    details: Dict[str, Any]


@dataclass
class APPComplianceAssessment:
    """Comprehensive APP compliance assessment."""
    assessment_id: str
    assessment_date: str
    organisation_details: Dict[str, str]
    data_processing_activities: List[Dict[str, Any]]
    app_compliance_status: Dict[APPRequirement, ComplianceStatus]
    violations_found: List[APPComplianceViolation]
    recommendations: List[str]
    overall_compliance_score: float
    regulatory_risk_level: str
    next_review_date: str
    audit_trail: Dict[str, Any]


class APPComplianceTester:
    """Australian Privacy Principles compliance tester."""
    
    def __init__(self):
        """Initialise APP compliance tester."""
        self.logger = logger.bind(component="app_compliance_tester")
        
        # APP compliance criteria
        self.app_criteria = {
            APPRequirement.APP1: {
                "requirements": [
                    "Privacy policy publicly available",
                    "Privacy policy up to date",
                    "Clear contact information for privacy enquiries",
                    "Governance framework documented"
                ],
                "evidence_required": ["privacy_policy", "governance_framework", "contact_details"]
            },
            APPRequirement.APP2: {
                "requirements": [
                    "Option for anonymous interaction where practicable",
                    "Option for pseudonymous interaction where practicable",
                    "Clear explanation when anonymity not possible"
                ],
                "evidence_required": ["anonymity_options", "pseudonymity_options", "justification_documents"]
            },
            APPRequirement.APP3: {
                "requirements": [
                    "Collection only when necessary for functions",
                    "Collection by lawful and fair means",
                    "Collection from individual where reasonable and practicable",
                    "Sensitive information collection restrictions"
                ],
                "evidence_required": ["collection_policy", "necessity_assessment", "lawful_basis"]
            },
            APPRequirement.APP4: {
                "requirements": [
                    "Procedures for handling unsolicited information",
                    "Destruction if not permitted to collect",
                    "De-identification where destruction not practicable"
                ],
                "evidence_required": ["unsolicited_procedures", "destruction_records", "deidentification_processes"]
            },
            APPRequirement.APP5: {
                "requirements": [
                    "Notification at or before collection",
                    "Clear statement of purposes",
                    "Legal authority for collection",
                    "Contact details provided",
                    "Consequences of not providing information explained"
                ],
                "evidence_required": ["collection_notices", "privacy_notices", "consent_forms"]
            },
            APPRequirement.APP6: {
                "requirements": [
                    "Use only for primary purpose or related secondary purpose",
                    "Disclosure only with consent or legal authority",
                    "Purpose limitation respected",
                    "Use and disclosure register maintained"
                ],
                "evidence_required": ["use_disclosure_register", "purpose_documentation", "consent_records"]
            },
            APPRequirement.APP7: {
                "requirements": [
                    "Direct marketing restrictions observed",
                    "Opt-out mechanisms provided",
                    "Source disclosure for third-party lists"
                ],
                "evidence_required": ["marketing_procedures", "opt_out_mechanisms", "source_documentation"]
            },
            APPRequirement.APP8: {
                "requirements": [
                    "Cross-border disclosure restrictions",
                    "Adequacy assessments conducted",
                    "Appropriate safeguards implemented",
                    "Individual notification where required"
                ],
                "evidence_required": ["transfer_assessments", "safeguards_documentation", "adequacy_decisions"]
            },
            APPRequirement.APP9: {
                "requirements": [
                    "Individual adoption restrictions",
                    "Government identifier restrictions"
                ],
                "evidence_required": ["identifier_policies", "adoption_procedures"]
            },
            APPRequirement.APP10: {
                "requirements": [
                    "Information accuracy ensured",
                    "Information completeness verified",
                    "Information currency maintained",
                    "Reasonable steps for quality assurance"
                ],
                "evidence_required": ["quality_procedures", "accuracy_checks", "currency_processes"]
            },
            APPRequirement.APP11: {
                "requirements": [
                    "Reasonable security measures implemented",
                    "Unauthorised access prevention",
                    "Unauthorised disclosure prevention",
                    "Destruction/de-identification when no longer needed"
                ],
                "evidence_required": ["security_procedures", "access_controls", "destruction_schedules"]
            },
            APPRequirement.APP12: {
                "requirements": [
                    "Individual access rights respected",
                    "Reasonable steps to provide access",
                    "Access refusal justifications documented",
                    "Access fees reasonable"
                ],
                "evidence_required": ["access_procedures", "request_handling", "fee_schedule"]
            },
            APPRequirement.APP13: {
                "requirements": [
                    "Correction mechanisms available",
                    "Reasonable steps to correct information",
                    "Correction refusal justifications documented",
                    "Associated entities notified of corrections"
                ],
                "evidence_required": ["correction_procedures", "request_handling", "notification_records"]
            }
        }
        
        # Sensitive information categories (APP 3)
        self.sensitive_information_categories = [
            "health_information",
            "genetic_information", 
            "biometric_information",
            "racial_ethnic_origin",
            "political_opinions",
            "religious_beliefs",
            "philosophical_beliefs",
            "trade_union_membership",
            "sexual_orientation",
            "criminal_record"
        ]
        
        # Cross-border transfer adequacy decisions
        self.adequacy_jurisdictions = [
            "european_union",
            "united_kingdom",
            "new_zealand",
            "canada",  # Limited
            "switzerland"
        ]
    
    def assess_app1_compliance(self, privacy_policy: Dict[str, Any], governance_framework: Dict[str, Any]) -> APPComplianceViolation:
        """
        Assess APP 1 - Open and transparent management compliance.
        
        Args:
            privacy_policy: Privacy policy content and metadata
            governance_framework: Governance framework documentation
            
        Returns:
            APP compliance violation if any
        """
        violations = []
        
        # Check privacy policy availability and currency
        if not privacy_policy.get("publicly_available", False):
            violations.append({
                "issue": "Privacy policy not publicly available",
                "severity": "high",
                "evidence": privacy_policy.get("availability_status", "unknown")
            })
        
        # Check last update date
        last_updated = privacy_policy.get("last_updated")
        if last_updated:
            try:
                update_date = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
                if (datetime.now() - update_date).days > 365:
                    violations.append({
                        "issue": "Privacy policy not updated within 12 months",
                        "severity": "medium",
                        "evidence": f"Last updated: {last_updated}"
                    })
            except ValueError:
                violations.append({
                    "issue": "Invalid privacy policy update date format",
                    "severity": "low",
                    "evidence": f"Date provided: {last_updated}"
                })
        
        # Check contact information
        contact_info = privacy_policy.get("contact_information", {})
        required_contact_fields = ["privacy_officer", "email", "phone", "postal_address"]
        missing_contact_fields = [field for field in required_contact_fields if not contact_info.get(field)]
        
        if missing_contact_fields:
            violations.append({
                "issue": "Missing privacy contact information",
                "severity": "high",
                "evidence": f"Missing fields: {missing_contact_fields}"
            })
        
        # Check governance framework
        if not governance_framework.get("documented", False):
            violations.append({
                "issue": "Privacy governance framework not documented",
                "severity": "high",
                "evidence": "No governance framework documentation found"
            })
        
        if violations:
            return APPComplianceViolation(
                app_principle=APPRequirement.APP1,
                violation_type="transparency_governance_failure",
                severity="high" if any(v["severity"] == "high" for v in violations) else "medium",
                description="APP 1 transparency and governance requirements not met",
                affected_data_elements=["privacy_policy", "governance_framework"],
                remediation_required=True,
                remediation_timeframe="30 days",
                regulatory_risk="high",
                sample_evidence=violations[:3],
                compliance_standard="Privacy Act 1988 APP 1",
                details={"total_violations": len(violations), "violations": violations}
            )
        
        return None
    
    def assess_app3_compliance(self, collection_practices: Dict[str, Any], data_elements: List[str]) -> List[APPComplianceViolation]:
        """
        Assess APP 3 - Collection of solicited personal information compliance.
        
        Args:
            collection_practices: Data collection practices documentation
            data_elements: List of collected data elements
            
        Returns:
            List of APP 3 compliance violations
        """
        violations = []
        
        # Check necessity assessment
        if not collection_practices.get("necessity_assessment_conducted", False):
            violations.append(APPComplianceViolation(
                app_principle=APPRequirement.APP3,
                violation_type="necessity_assessment_missing",
                severity="high",
                description="No necessity assessment conducted for data collection",
                affected_data_elements=data_elements,
                remediation_required=True,
                remediation_timeframe="14 days",
                regulatory_risk="high",
                sample_evidence=[{"issue": "necessity_assessment_missing"}],
                compliance_standard="Privacy Act 1988 APP 3.1",
                details={"assessment_status": collection_practices.get("necessity_assessment_conducted")}
            ))
        
        # Check for sensitive information collection
        sensitive_elements = [element for element in data_elements 
                            if any(sensitive in element.lower() for sensitive in self.sensitive_information_categories)]
        
        if sensitive_elements:
            # Check if sensitive information collection is justified
            sensitive_justification = collection_practices.get("sensitive_information_justification", {})
            
            for sensitive_element in sensitive_elements:
                if sensitive_element not in sensitive_justification:
                    violations.append(APPComplianceViolation(
                        app_principle=APPRequirement.APP3,
                        violation_type="sensitive_information_unjustified",
                        severity="critical",
                        description=f"Sensitive information '{sensitive_element}' collected without justification",
                        affected_data_elements=[sensitive_element],
                        remediation_required=True,
                        remediation_timeframe="7 days",
                        regulatory_risk="critical",
                        sample_evidence=[{"sensitive_element": sensitive_element, "justification": "none"}],
                        compliance_standard="Privacy Act 1988 APP 3.3",
                        details={"sensitive_categories": sensitive_elements}
                    ))
        
        # Check lawful and fair collection
        collection_methods = collection_practices.get("collection_methods", [])
        unlawful_methods = ["deception", "coercion", "undisclosed_collection"]
        
        problematic_methods = [method for method in collection_methods if method in unlawful_methods]
        if problematic_methods:
            violations.append(APPComplianceViolation(
                app_principle=APPRequirement.APP3,
                violation_type="unlawful_collection_methods",
                severity="critical",
                description="Unlawful or unfair collection methods identified",
                affected_data_elements=data_elements,
                remediation_required=True,
                remediation_timeframe="immediate",
                regulatory_risk="critical",
                sample_evidence=[{"problematic_methods": problematic_methods}],
                compliance_standard="Privacy Act 1988 APP 3.2",
                details={"collection_methods": collection_methods}
            ))
        
        return violations
    
    def assess_app5_compliance(self, collection_notices: List[Dict[str, Any]]) -> List[APPComplianceViolation]:
        """
        Assess APP 5 - Notification of collection compliance.
        
        Args:
            collection_notices: List of collection notice records
            
        Returns:
            List of APP 5 compliance violations
        """
        violations = []
        
        required_elements = [
            "organisation_identity",
            "collection_purposes",
            "legal_authority",
            "consequences_of_not_providing",
            "disclosure_recipients",
            "privacy_policy_reference",
            "contact_details"
        ]
        
        for notice in collection_notices:
            missing_elements = [element for element in required_elements 
                              if not notice.get(element)]
            
            if missing_elements:
                violations.append(APPComplianceViolation(
                    app_principle=APPRequirement.APP5,
                    violation_type="incomplete_collection_notice",
                    severity="high",
                    description=f"Collection notice missing required elements: {missing_elements}",
                    affected_data_elements=[notice.get("notice_id", "unknown")],
                    remediation_required=True,
                    remediation_timeframe="14 days",
                    regulatory_risk="high",
                    sample_evidence=[{"missing_elements": missing_elements, "notice_id": notice.get("notice_id")}],
                    compliance_standard="Privacy Act 1988 APP 5.1",
                    details={"notice_content": notice, "missing_elements": missing_elements}
                ))
            
            # Check timing of notification
            notification_timing = notice.get("notification_timing", "")
            if notification_timing not in ["at_collection", "before_collection"]:
                violations.append(APPComplianceViolation(
                    app_principle=APPRequirement.APP5,
                    violation_type="improper_notification_timing",
                    severity="medium",
                    description="Collection notice not provided at or before collection",
                    affected_data_elements=[notice.get("notice_id", "unknown")],
                    remediation_required=True,
                    remediation_timeframe="immediate",
                    regulatory_risk="medium",
                    sample_evidence=[{"timing": notification_timing, "notice_id": notice.get("notice_id")}],
                    compliance_standard="Privacy Act 1988 APP 5.1",
                    details={"notification_timing": notification_timing}
                ))
        
        return violations
    
    def assess_app6_compliance(self, use_disclosure_register: List[Dict[str, Any]], purposes: List[str]) -> List[APPComplianceViolation]:
        """
        Assess APP 6 - Use or disclosure compliance.
        
        Args:
            use_disclosure_register: Register of use and disclosure activities
            purposes: List of declared purposes
            
        Returns:
            List of APP 6 compliance violations
        """
        violations = []
        
        for record in use_disclosure_register:
            record_purpose = record.get("purpose", "")
            primary_purposes = [p for p in purposes if record.get("purpose_type") == "primary"]
            
            # Check purpose limitation
            if record.get("purpose_type") == "secondary":
                if not record.get("consent_obtained", False) and not record.get("legal_authority", False):
                    violations.append(APPComplianceViolation(
                        app_principle=APPRequirement.APP6,
                        violation_type="unauthorised_secondary_use",
                        severity="high",
                        description=f"Secondary use/disclosure without consent or legal authority: {record_purpose}",
                        affected_data_elements=[record.get("data_elements", "unknown")],
                        remediation_required=True,
                        remediation_timeframe="immediate",
                        regulatory_risk="high",
                        sample_evidence=[{"record": record}],
                        compliance_standard="Privacy Act 1988 APP 6.1-6.2",
                        details={"unauthorised_purpose": record_purpose}
                    ))
            
            # Check for external disclosures
            if record.get("disclosure_type") == "external":
                if not record.get("recipient_notification", False):
                    violations.append(APPComplianceViolation(
                        app_principle=APPRequirement.APP6,
                        violation_type="recipient_notification_missing",
                        severity="medium",
                        description="External disclosure without proper recipient notification",
                        affected_data_elements=[record.get("data_elements", "unknown")],
                        remediation_required=True,
                        remediation_timeframe="7 days",
                        regulatory_risk="medium",
                        sample_evidence=[{"record": record}],
                        compliance_standard="Privacy Act 1988 APP 6",
                        details={"recipient": record.get("recipient")}
                    ))
        
        return violations
    
    def assess_app8_compliance(self, cross_border_transfers: List[CrossBorderTransfer]) -> List[APPComplianceViolation]:
        """
        Assess APP 8 - Cross-border disclosure compliance.
        
        Args:
            cross_border_transfers: List of cross-border transfer records
            
        Returns:
            List of APP 8 compliance violations
        """
        violations = []
        
        for transfer in cross_border_transfers:
            # Check adequacy decision
            if transfer.destination_country.lower() not in self.adequacy_jurisdictions:
                if not transfer.adequacy_decision and not transfer.safeguards_implemented:
                    violations.append(APPComplianceViolation(
                        app_principle=APPRequirement.APP8,
                        violation_type="inadequate_cross_border_protection",
                        severity="critical",
                        description=f"Cross-border transfer to {transfer.destination_country} without adequate protection",
                        affected_data_elements=["personal_information"],
                        remediation_required=True,
                        remediation_timeframe="immediate",
                        regulatory_risk="critical",
                        sample_evidence=[{"transfer": transfer.__dict__}],
                        compliance_standard="Privacy Act 1988 APP 8.1",
                        details={"destination": transfer.destination_country, "safeguards": transfer.safeguards_implemented}
                    ))
                
                # Check safeguards
                elif transfer.safeguards_implemented:
                    required_safeguards = ["contractual_clauses", "adequacy_assessment", "consent_obtained"]
                    missing_safeguards = [safeguard for safeguard in required_safeguards 
                                        if safeguard not in transfer.safeguards_implemented]
                    
                    if missing_safeguards:
                        violations.append(APPComplianceViolation(
                            app_principle=APPRequirement.APP8,
                            violation_type="insufficient_safeguards",
                            severity="high",
                            description=f"Insufficient safeguards for transfer to {transfer.destination_country}",
                            affected_data_elements=["personal_information"],
                            remediation_required=True,
                            remediation_timeframe="14 days",
                            regulatory_risk="high",
                            sample_evidence=[{"missing_safeguards": missing_safeguards}],
                            compliance_standard="Privacy Act 1988 APP 8.2",
                            details={"destination": transfer.destination_country, "missing_safeguards": missing_safeguards}
                        ))
        
        return violations
    
    def assess_app11_compliance(self, security_measures: Dict[str, Any], access_controls: Dict[str, Any]) -> List[APPComplianceViolation]:
        """
        Assess APP 11 - Security of personal information compliance.
        
        Args:
            security_measures: Security measures documentation
            access_controls: Access control implementation details
            
        Returns:
            List of APP 11 compliance violations
        """
        violations = []
        
        # Check encryption
        encryption_status = security_measures.get("encryption", {})
        required_encryption = ["data_at_rest", "data_in_transit", "backup_encryption"]
        
        missing_encryption = [enc_type for enc_type in required_encryption 
                            if not encryption_status.get(enc_type, False)]
        
        if missing_encryption:
            violations.append(APPComplianceViolation(
                app_principle=APPRequirement.APP11,
                violation_type="insufficient_encryption",
                severity="high",
                description=f"Missing encryption for: {missing_encryption}",
                affected_data_elements=["personal_information"],
                remediation_required=True,
                remediation_timeframe="30 days",
                regulatory_risk="high",
                sample_evidence=[{"missing_encryption": missing_encryption}],
                compliance_standard="Privacy Act 1988 APP 11.1",
                details={"encryption_status": encryption_status}
            ))
        
        # Check access controls
        access_control_measures = [
            "user_authentication",
            "role_based_access",
            "access_logging",
            "regular_access_review"
        ]
        
        missing_controls = [control for control in access_control_measures 
                          if not access_controls.get(control, False)]
        
        if missing_controls:
            violations.append(APPComplianceViolation(
                app_principle=APPRequirement.APP11,
                violation_type="inadequate_access_controls",
                severity="high",
                description=f"Missing access controls: {missing_controls}",
                affected_data_elements=["personal_information"],
                remediation_required=True,
                remediation_timeframe="14 days",
                regulatory_risk="high",
                sample_evidence=[{"missing_controls": missing_controls}],
                compliance_standard="Privacy Act 1988 APP 11.1",
                details={"access_controls": access_controls}
            ))
        
        # Check data retention and destruction
        retention_policy = security_measures.get("retention_policy", {})
        if not retention_policy.get("documented", False):
            violations.append(APPComplianceViolation(
                app_principle=APPRequirement.APP11,
                violation_type="missing_retention_policy",
                severity="medium",
                description="No documented data retention and destruction policy",
                affected_data_elements=["personal_information"],
                remediation_required=True,
                remediation_timeframe="30 days",
                regulatory_risk="medium",
                sample_evidence=[{"retention_policy_status": "not_documented"}],
                compliance_standard="Privacy Act 1988 APP 11.2",
                details={"retention_policy": retention_policy}
            ))
        
        return violations
    
    def conduct_comprehensive_app_assessment(self, 
                                           organisation_details: Dict[str, str],
                                           privacy_documentation: Dict[str, Any],
                                           data_processing_activities: List[Dict[str, Any]]) -> APPComplianceAssessment:
        """
        Conduct comprehensive Australian Privacy Principles compliance assessment.
        
        Args:
            organisation_details: Organisation information
            privacy_documentation: Privacy policies and documentation
            data_processing_activities: List of data processing activities
            
        Returns:
            Comprehensive APP compliance assessment
        """
        assessment_id = f"app_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        all_violations = []
        app_compliance_status = {}
        
        # APP 1 Assessment
        app1_violation = self.assess_app1_compliance(
            privacy_documentation.get("privacy_policy", {}),
            privacy_documentation.get("governance_framework", {})
        )
        if app1_violation:
            all_violations.append(app1_violation)
            app_compliance_status[APPRequirement.APP1] = ComplianceStatus.NON_COMPLIANT
        else:
            app_compliance_status[APPRequirement.APP1] = ComplianceStatus.COMPLIANT
        
        # APP 3 Assessment
        app3_violations = self.assess_app3_compliance(
            privacy_documentation.get("collection_practices", {}),
            privacy_documentation.get("data_elements", [])
        )
        all_violations.extend(app3_violations)
        app_compliance_status[APPRequirement.APP3] = (
            ComplianceStatus.NON_COMPLIANT if app3_violations else ComplianceStatus.COMPLIANT
        )
        
        # APP 5 Assessment
        app5_violations = self.assess_app5_compliance(
            privacy_documentation.get("collection_notices", [])
        )
        all_violations.extend(app5_violations)
        app_compliance_status[APPRequirement.APP5] = (
            ComplianceStatus.NON_COMPLIANT if app5_violations else ComplianceStatus.COMPLIANT
        )
        
        # APP 6 Assessment
        app6_violations = self.assess_app6_compliance(
            privacy_documentation.get("use_disclosure_register", []),
            privacy_documentation.get("purposes", [])
        )
        all_violations.extend(app6_violations)
        app_compliance_status[APPRequirement.APP6] = (
            ComplianceStatus.NON_COMPLIANT if app6_violations else ComplianceStatus.COMPLIANT
        )
        
        # APP 8 Assessment
        cross_border_transfers = [CrossBorderTransfer(**transfer) for transfer in 
                                privacy_documentation.get("cross_border_transfers", [])]
        app8_violations = self.assess_app8_compliance(cross_border_transfers)
        all_violations.extend(app8_violations)
        app_compliance_status[APPRequirement.APP8] = (
            ComplianceStatus.NON_COMPLIANT if app8_violations else ComplianceStatus.COMPLIANT
        )
        
        # APP 11 Assessment
        app11_violations = self.assess_app11_compliance(
            privacy_documentation.get("security_measures", {}),
            privacy_documentation.get("access_controls", {})
        )
        all_violations.extend(app11_violations)
        app_compliance_status[APPRequirement.APP11] = (
            ComplianceStatus.NON_COMPLIANT if app11_violations else ComplianceStatus.COMPLIANT
        )
        
        # Set remaining APPs as requires review (simplified assessment)
        remaining_apps = [app for app in APPRequirement if app not in app_compliance_status]
        for app in remaining_apps:
            app_compliance_status[app] = ComplianceStatus.REQUIRES_REVIEW
        
        # Calculate overall compliance score
        compliant_count = sum(1 for status in app_compliance_status.values() 
                            if status == ComplianceStatus.COMPLIANT)
        total_assessed = len([status for status in app_compliance_status.values() 
                            if status != ComplianceStatus.REQUIRES_REVIEW])
        overall_compliance_score = (compliant_count / total_assessed) if total_assessed > 0 else 0.0
        
        # Determine regulatory risk level
        critical_violations = [v for v in all_violations if v.severity == "critical"]
        high_violations = [v for v in all_violations if v.severity == "high"]
        
        if critical_violations:
            regulatory_risk_level = "critical"
        elif len(high_violations) > 3:
            regulatory_risk_level = "high"
        elif len(all_violations) > 5:
            regulatory_risk_level = "medium"
        else:
            regulatory_risk_level = "low"
        
        # Generate recommendations
        recommendations = self._generate_app_recommendations(all_violations, app_compliance_status)
        
        # Calculate next review date
        next_review_date = (datetime.now() + timedelta(days=365)).isoformat()
        if regulatory_risk_level in ["critical", "high"]:
            next_review_date = (datetime.now() + timedelta(days=90)).isoformat()
        
        return APPComplianceAssessment(
            assessment_id=assessment_id,
            assessment_date=datetime.now().isoformat(),
            organisation_details=organisation_details,
            data_processing_activities=data_processing_activities,
            app_compliance_status=app_compliance_status,
            violations_found=all_violations,
            recommendations=recommendations,
            overall_compliance_score=overall_compliance_score,
            regulatory_risk_level=regulatory_risk_level,
            next_review_date=next_review_date,
            audit_trail={
                "assessment_method": "comprehensive_app_compliance_assessment",
                "apps_assessed": list(app_compliance_status.keys()),
                "total_violations": len(all_violations),
                "violations_by_severity": {
                    "critical": len(critical_violations),
                    "high": len(high_violations),
                    "medium": len([v for v in all_violations if v.severity == "medium"]),
                    "low": len([v for v in all_violations if v.severity == "low"])
                }
            }
        )
    
    def _generate_app_recommendations(self, violations: List[APPComplianceViolation], 
                                    compliance_status: Dict[APPRequirement, ComplianceStatus]) -> List[str]:
        """Generate APP compliance recommendations."""
        recommendations = []
        
        # Priority recommendations based on violations
        critical_violations = [v for v in violations if v.severity == "critical"]
        if critical_violations:
            recommendations.append("URGENT: Address critical APP violations immediately to avoid regulatory action")
        
        # Specific APP recommendations
        non_compliant_apps = [app for app, status in compliance_status.items() 
                            if status == ComplianceStatus.NON_COMPLIANT]
        
        if APPRequirement.APP1 in non_compliant_apps:
            recommendations.append("Update privacy policy and establish privacy governance framework")
        
        if APPRequirement.APP3 in non_compliant_apps:
            recommendations.append("Conduct necessity assessment for all data collection activities")
        
        if APPRequirement.APP5 in non_compliant_apps:
            recommendations.append("Review and update collection notices to include all required elements")
        
        if APPRequirement.APP6 in non_compliant_apps:
            recommendations.append("Implement use and disclosure register with purpose limitation controls")
        
        if APPRequirement.APP8 in non_compliant_apps:
            recommendations.append("Review cross-border transfers and implement adequate safeguards")
        
        if APPRequirement.APP11 in non_compliant_apps:
            recommendations.append("Strengthen security measures including encryption and access controls")
        
        # General recommendations
        if len(violations) > 5:
            recommendations.append("Consider engaging privacy consultant for comprehensive compliance review")
        
        requires_review_count = sum(1 for status in compliance_status.values() 
                                  if status == ComplianceStatus.REQUIRES_REVIEW)
        if requires_review_count > 3:
            recommendations.append("Complete detailed assessment of remaining APP requirements")
        
        return recommendations


# Test suite
class TestAPPCompliance:
    """Test suite for Australian Privacy Principles compliance."""
    
    @pytest.fixture
    def app_tester(self):
        """Create APP compliance tester instance."""
        return APPComplianceTester()
    
    @pytest.fixture
    def compliant_privacy_policy(self):
        """Compliant privacy policy for testing."""
        return {
            "publicly_available": True,
            "last_updated": datetime.now().isoformat(),
            "contact_information": {
                "privacy_officer": "Privacy Officer",
                "email": "privacy@example.com",
                "phone": "1300 123 456",
                "postal_address": "123 Privacy St, Sydney NSW 2000"
            },
            "version": "2.1",
            "effective_date": datetime.now().isoformat()
        }
    
    @pytest.fixture
    def non_compliant_privacy_policy(self):
        """Non-compliant privacy policy for testing."""
        return {
            "publicly_available": False,
            "last_updated": "2022-01-01T00:00:00Z",  # Old date
            "contact_information": {
                "email": "info@example.com"  # Missing required fields
            },
            "version": "1.0"
        }
    
    @pytest.fixture
    def compliant_governance_framework(self):
        """Compliant governance framework for testing."""
        return {
            "documented": True,
            "privacy_impact_assessments": True,
            "staff_training": True,
            "incident_response_plan": True,
            "regular_reviews": True
        }
    
    @pytest.fixture
    def collection_practices_compliant(self):
        """Compliant collection practices for testing."""
        return {
            "necessity_assessment_conducted": True,
            "collection_methods": ["direct_collection", "website_forms", "surveys"],
            "sensitive_information_justification": {
                "health_information": "Required for health service delivery",
                "genetic_information": "Required for genetic counselling services"
            },
            "lawful_basis": "Consent and legitimate interest"
        }
    
    @pytest.fixture
    def collection_practices_non_compliant(self):
        """Non-compliant collection practices for testing."""
        return {
            "necessity_assessment_conducted": False,
            "collection_methods": ["deception", "undisclosed_collection"],
            "sensitive_information_justification": {},  # Empty
            "lawful_basis": "undefined"
        }
    
    @pytest.fixture
    def compliant_collection_notices(self):
        """Compliant collection notices for testing."""
        return [
            {
                "notice_id": "CN001",
                "organisation_identity": "Test Health Organisation",
                "collection_purposes": ["Health service delivery", "Quality improvement"],
                "legal_authority": "Health Records Act, Consent",
                "consequences_of_not_providing": "Unable to provide health services",
                "disclosure_recipients": ["Healthcare providers", "Government agencies"],
                "privacy_policy_reference": "Available at www.example.com/privacy",
                "contact_details": "privacy@example.com",
                "notification_timing": "at_collection"
            }
        ]
    
    @pytest.fixture
    def non_compliant_collection_notices(self):
        """Non-compliant collection notices for testing."""
        return [
            {
                "notice_id": "CN002",
                "organisation_identity": "Test Organisation",
                # Missing required elements
                "notification_timing": "after_collection"
            }
        ]
    
    def test_app1_compliance_assessment_compliant(self, app_tester, compliant_privacy_policy, compliant_governance_framework):
        """Test APP 1 compliance assessment with compliant data."""
        violation = app_tester.assess_app1_compliance(compliant_privacy_policy, compliant_governance_framework)
        
        # Should have no violations for compliant policy
        assert violation is None, "Compliant privacy policy should not generate APP 1 violations"
    
    def test_app1_compliance_assessment_non_compliant(self, app_tester, non_compliant_privacy_policy, compliant_governance_framework):
        """Test APP 1 compliance assessment with non-compliant data."""
        # Missing governance framework
        violation = app_tester.assess_app1_compliance(non_compliant_privacy_policy, {"documented": False})
        
        # Should detect violations
        assert violation is not None, "Non-compliant privacy policy should generate APP 1 violations"
        assert violation.app_principle == APPRequirement.APP1
        assert violation.severity in ["high", "critical"]
        assert violation.remediation_required is True
        assert len(violation.sample_evidence) > 0
    
    def test_app3_compliance_assessment_compliant(self, app_tester, collection_practices_compliant):
        """Test APP 3 compliance assessment with compliant practices."""
        data_elements = ["name", "address", "health_information"]
        violations = app_tester.assess_app3_compliance(collection_practices_compliant, data_elements)
        
        # Should have no violations for compliant practices
        assert len(violations) == 0, "Compliant collection practices should not generate APP 3 violations"
    
    def test_app3_compliance_assessment_non_compliant(self, app_tester, collection_practices_non_compliant):
        """Test APP 3 compliance assessment with non-compliant practices."""
        data_elements = ["name", "address", "health_information", "genetic_information"]
        violations = app_tester.assess_app3_compliance(collection_practices_non_compliant, data_elements)
        
        # Should detect multiple violations
        assert len(violations) > 0, "Non-compliant collection practices should generate APP 3 violations"
        
        violation_types = [v.violation_type for v in violations]
        assert "necessity_assessment_missing" in violation_types
        assert "sensitive_information_unjustified" in violation_types
        assert "unlawful_collection_methods" in violation_types
        
        # Check severity levels
        critical_violations = [v for v in violations if v.severity == "critical"]
        assert len(critical_violations) > 0, "Should have critical violations for unlawful collection"
    
    def test_app5_compliance_assessment_compliant(self, app_tester, compliant_collection_notices):
        """Test APP 5 compliance assessment with compliant notices."""
        violations = app_tester.assess_app5_compliance(compliant_collection_notices)
        
        # Should have no violations for compliant notices
        assert len(violations) == 0, "Compliant collection notices should not generate APP 5 violations"
    
    def test_app5_compliance_assessment_non_compliant(self, app_tester, non_compliant_collection_notices):
        """Test APP 5 compliance assessment with non-compliant notices."""
        violations = app_tester.assess_app5_compliance(non_compliant_collection_notices)
        
        # Should detect violations
        assert len(violations) > 0, "Non-compliant collection notices should generate APP 5 violations"
        
        violation_types = [v.violation_type for v in violations]
        assert "incomplete_collection_notice" in violation_types
        assert "improper_notification_timing" in violation_types
        
        for violation in violations:
            assert violation.app_principle == APPRequirement.APP5
            assert violation.remediation_required is True
    
    def test_app8_cross_border_compliance(self, app_tester):
        """Test APP 8 cross-border transfer compliance."""
        # Non-compliant transfer
        non_compliant_transfers = [
            CrossBorderTransfer(
                transfer_id="T001",
                destination_country="united_states",
                recipient_organisation="US Data Corp",
                transfer_purpose=DataHandlingPurpose.HEALTH_RESEARCH,
                adequacy_decision=False,
                safeguards_implemented=[],
                transfer_date=datetime.now().isoformat()
            )
        ]
        
        violations = app_tester.assess_app8_compliance(non_compliant_transfers)
        
        # Should detect inadequate protection
        assert len(violations) > 0, "Non-compliant cross-border transfer should generate APP 8 violations"
        
        violation = violations[0]
        assert violation.app_principle == APPRequirement.APP8
        assert violation.severity == "critical"
        assert "inadequate_cross_border_protection" in violation.violation_type
    
    def test_app11_security_compliance(self, app_tester):
        """Test APP 11 security compliance assessment."""
        # Non-compliant security measures
        non_compliant_security = {
            "encryption": {
                "data_at_rest": False,
                "data_in_transit": True,
                "backup_encryption": False
            },
            "retention_policy": {
                "documented": False
            }
        }
        
        non_compliant_access_controls = {
            "user_authentication": True,
            "role_based_access": False,
            "access_logging": False,
            "regular_access_review": False
        }
        
        violations = app_tester.assess_app11_compliance(non_compliant_security, non_compliant_access_controls)
        
        # Should detect security violations
        assert len(violations) > 0, "Non-compliant security measures should generate APP 11 violations"
        
        violation_types = [v.violation_type for v in violations]
        assert "insufficient_encryption" in violation_types
        assert "inadequate_access_controls" in violation_types
        assert "missing_retention_policy" in violation_types
        
        for violation in violations:
            assert violation.app_principle == APPRequirement.APP11
            assert violation.severity in ["high", "medium"]
    
    def test_comprehensive_app_assessment(self, app_tester, compliant_privacy_policy, compliant_governance_framework, 
                                        collection_practices_compliant, compliant_collection_notices):
        """Test comprehensive APP compliance assessment."""
        organisation_details = {
            "name": "Test Health Organisation",
            "abn": "12345678901",
            "contact": "info@example.com"
        }
        
        privacy_documentation = {
            "privacy_policy": compliant_privacy_policy,
            "governance_framework": compliant_governance_framework,
            "collection_practices": collection_practices_compliant,
            "collection_notices": compliant_collection_notices,
            "data_elements": ["name", "address", "health_information"],
            "purposes": ["health_service_delivery", "quality_improvement"],
            "use_disclosure_register": [],
            "cross_border_transfers": [],
            "security_measures": {
                "encryption": {
                    "data_at_rest": True,
                    "data_in_transit": True,
                    "backup_encryption": True
                },
                "retention_policy": {"documented": True}
            },
            "access_controls": {
                "user_authentication": True,
                "role_based_access": True,
                "access_logging": True,
                "regular_access_review": True
            }
        }
        
        data_processing_activities = [
            {
                "activity": "Patient record management",
                "purpose": "Health service delivery",
                "data_types": ["Health information", "Contact details"]
            }
        ]
        
        assessment = app_tester.conduct_comprehensive_app_assessment(
            organisation_details, privacy_documentation, data_processing_activities
        )
        
        # Verify assessment structure
        assert isinstance(assessment, APPComplianceAssessment)
        assert assessment.assessment_id is not None
        assert assessment.assessment_date is not None
        assert assessment.organisation_details == organisation_details
        assert isinstance(assessment.app_compliance_status, dict)
        assert isinstance(assessment.violations_found, list)
        assert isinstance(assessment.recommendations, list)
        assert 0.0 <= assessment.overall_compliance_score <= 1.0
        assert assessment.regulatory_risk_level in ["low", "medium", "high", "critical"]
        assert assessment.next_review_date is not None
        assert isinstance(assessment.audit_trail, dict)
        
        # Should have good compliance with compliant documentation
        compliant_apps = [app for app, status in assessment.app_compliance_status.items() 
                         if status == ComplianceStatus.COMPLIANT]
        assert len(compliant_apps) >= 3, "Should have multiple compliant APPs with good documentation"
        
        # Should have reasonable compliance score
        assert assessment.overall_compliance_score >= 0.5, "Should have reasonable compliance score with compliant documentation"
    
    def test_app_criteria_completeness(self, app_tester):
        """Test that APP criteria are complete and well-defined."""
        # Verify all APPs have criteria defined
        for app in APPRequirement:
            assert app in app_tester.app_criteria, f"APP {app} should have criteria defined"
            
            criteria = app_tester.app_criteria[app]
            assert "requirements" in criteria, f"APP {app} should have requirements"
            assert "evidence_required" in criteria, f"APP {app} should have evidence requirements"
            assert isinstance(criteria["requirements"], list), f"APP {app} requirements should be a list"
            assert isinstance(criteria["evidence_required"], list), f"APP {app} evidence should be a list"
    
    def test_sensitive_information_categories(self, app_tester):
        """Test sensitive information categories are comprehensive."""
        # Verify sensitive information categories include health information
        assert "health_information" in app_tester.sensitive_information_categories
        assert "genetic_information" in app_tester.sensitive_information_categories
        assert "biometric_information" in app_tester.sensitive_information_categories
        
        # Should have reasonable number of categories
        assert len(app_tester.sensitive_information_categories) >= 8
    
    def test_adequacy_jurisdictions(self, app_tester):
        """Test adequacy jurisdiction list is current."""
        # Verify key adequacy decisions are included
        assert "european_union" in app_tester.adequacy_jurisdictions
        assert "united_kingdom" in app_tester.adequacy_jurisdictions
        assert "new_zealand" in app_tester.adequacy_jurisdictions
        
        # Should have reasonable number of jurisdictions
        assert len(app_tester.adequacy_jurisdictions) >= 4


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])