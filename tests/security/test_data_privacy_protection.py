"""
Health Data Privacy Protection Testing

Comprehensive testing suite for health data privacy protection including:
- Data de-identification and anonymisation validation
- Statistical disclosure control testing 
- K-anonymity and L-diversity compliance validation
- Sensitive data handling and protection testing
- Data minimisation principle enforcement
- HIPAA-equivalent identifier removal for Australian context

This test suite ensures all Australian health data processing complies with
privacy regulations and maintains appropriate de-identification standards.
"""

import json
import pytest
import hashlib
import secrets
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Union, Any
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass, field
from enum import Enum
import logging

import polars as pl
import numpy as np
from loguru import logger

# Import existing privacy compliance infrastructure
from tests.data_quality.test_privacy_compliance import (
    PrivacyComplianceValidator, 
    DataClassification, 
    PrivacyRiskLevel,
    DeIdentificationTechnique,
    PrivacyViolation,
    DeIdentificationAssessment
)


class HealthDataSensitivity(Enum):
    """Health data sensitivity levels."""
    NON_SENSITIVE = "non_sensitive"
    QUASI_IDENTIFIER = "quasi_identifier"
    SENSITIVE = "sensitive"
    HIGHLY_SENSITIVE = "highly_sensitive"
    CRITICAL = "critical"


class DeIdentificationStandard(Enum):
    """De-identification standards."""
    SAFE_HARBOUR = "safe_harbour"  # HIPAA Safe Harbour equivalent
    EXPERT_DETERMINATION = "expert_determination"
    STATISTICAL_DISCLOSURE_CONTROL = "statistical_disclosure_control"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    SYNTHETIC_DATA = "synthetic_data"


@dataclass
class HealthIdentifier:
    """Health data identifier classification."""
    identifier_type: str
    sensitivity_level: HealthDataSensitivity
    removal_required: bool
    masking_technique: Optional[str] = None
    retention_period: Optional[str] = None
    compliance_standard: str = "APP11"


@dataclass
class PrivacyProtectionResult:
    """Privacy protection assessment result."""
    dataset_id: str
    assessment_timestamp: str
    identifiers_found: List[HealthIdentifier]
    deidentification_applied: List[DeIdentificationTechnique]
    disclosure_risk_score: float
    privacy_risk_level: PrivacyRiskLevel
    compliance_status: Dict[str, bool]
    recommendations: List[str]
    audit_trail: Dict[str, Any]


class HealthDataPrivacyTester:
    """Health data privacy protection tester."""
    
    def __init__(self):
        """Initialise health data privacy tester."""
        self.logger = logger.bind(component="health_data_privacy_tester")
        
        # Australian health data identifiers (HIPAA-equivalent for Australian context)
        self.health_identifiers = {
            "direct_identifiers": [
                "name", "full_name", "first_name", "last_name", "surname", "given_name",
                "medicare_number", "healthcare_identifier", "ihpa_identifier", "patient_id",
                "provider_number", "prescriber_number", "pharmacy_code", "hospital_id",
                "address", "street_address", "home_address", "postal_address",
                "email", "email_address", "phone", "telephone", "mobile", "contact_number",
                "date_of_birth", "birth_date", "dob", "birth_year"
            ],
            "quasi_identifiers": [
                "postcode", "suburb", "sa1_code", "sa2_code", "sa3_code", "sa4_code",
                "age", "exact_age", "birth_month", "gender", "sex",
                "occupation", "employment", "profession", "job_title",
                "admission_date", "discharge_date", "service_date", "consultation_date",
                "indigenous_status", "country_of_birth", "language", "marital_status"
            ],
            "sensitive_attributes": [
                "diagnosis", "icd_code", "health_condition", "medical_condition",
                "medication", "drug_name", "prescription", "treatment", "procedure",
                "mental_health", "substance_abuse", "reproductive_health",
                "genetic_information", "disability", "chronic_condition"
            ],
            "institutional_identifiers": [
                "hospital_name", "clinic_name", "provider_name", "organisation_name",
                "practice_name", "facility_id", "department", "ward", "unit"
            ]
        }
        
        # De-identification thresholds and rules
        self.deidentification_rules = {
            "minimum_cell_size": 5,  # Minimum count for statistical disclosure control
            "k_anonymity_k": 5,  # Minimum group size for k-anonymity
            "l_diversity_l": 3,  # Minimum diversity for sensitive attributes
            "age_grouping_years": 5,  # Age groups in 5-year intervals
            "date_precision_days": 30,  # Date precision to month level
            "geographic_precision": "sa2",  # Maximum geographic precision
            "suppression_threshold": 0.05,  # Maximum suppression rate (5%)
        }
        
        # Privacy risk scoring weights
        self.risk_weights = {
            "direct_identifier_weight": 10.0,
            "quasi_identifier_weight": 3.0,
            "sensitive_attribute_weight": 5.0,
            "institutional_identifier_weight": 2.0,
            "uniqueness_weight": 4.0,
            "dataset_size_weight": 2.0
        }
    
    def classify_health_data_sensitivity(self, df: pl.DataFrame) -> Dict[str, HealthDataSensitivity]:
        """
        Classify columns by health data sensitivity level.
        
        Args:
            df: DataFrame to classify
            
        Returns:
            Dictionary mapping column names to sensitivity levels
        """
        sensitivity_classification = {}
        
        for column in df.columns:
            column_lower = column.lower()
            
            # Check for direct identifiers
            if any(identifier in column_lower for identifier in self.health_identifiers["direct_identifiers"]):
                sensitivity_classification[column] = HealthDataSensitivity.CRITICAL
            
            # Check for sensitive attributes
            elif any(identifier in column_lower for identifier in self.health_identifiers["sensitive_attributes"]):
                sensitivity_classification[column] = HealthDataSensitivity.HIGHLY_SENSITIVE
            
            # Check for quasi-identifiers
            elif any(identifier in column_lower for identifier in self.health_identifiers["quasi_identifiers"]):
                sensitivity_classification[column] = HealthDataSensitivity.QUASI_IDENTIFIER
            
            # Check for institutional identifiers
            elif any(identifier in column_lower for identifier in self.health_identifiers["institutional_identifiers"]):
                sensitivity_classification[column] = HealthDataSensitivity.SENSITIVE
            
            else:
                sensitivity_classification[column] = HealthDataSensitivity.NON_SENSITIVE
        
        return sensitivity_classification
    
    def validate_deidentification_completeness(self, df: pl.DataFrame) -> List[HealthIdentifier]:
        """
        Validate completeness of de-identification process.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            List of remaining health identifiers that require attention
        """
        remaining_identifiers = []
        sensitivity_classification = self.classify_health_data_sensitivity(df)
        
        for column, sensitivity in sensitivity_classification.items():
            if sensitivity in [HealthDataSensitivity.CRITICAL, HealthDataSensitivity.HIGHLY_SENSITIVE]:
                # Check if direct identifiers are properly masked
                sample_values = df[column].drop_nulls().head(10).to_list()
                
                identifier_patterns = self._detect_identifier_patterns(sample_values, column)
                
                if identifier_patterns:
                    remaining_identifiers.append(HealthIdentifier(
                        identifier_type=column,
                        sensitivity_level=sensitivity,
                        removal_required=True,
                        masking_technique="suppression_or_generalisation",
                        retention_period="not_applicable",
                        compliance_standard="APP11"
                    ))
        
        return remaining_identifiers
    
    def test_statistical_disclosure_control(self, df: pl.DataFrame) -> List[PrivacyViolation]:
        """
        Test statistical disclosure control measures.
        
        Args:
            df: DataFrame to test
            
        Returns:
            List of statistical disclosure control violations
        """
        violations = []
        
        # Test minimum cell size rule
        count_columns = [col for col in df.columns if any(term in col.lower() for term in ["count", "total", "sum", "number"])]
        
        for count_col in count_columns:
            if count_col in df.columns and df[count_col].dtype.is_numeric():
                low_counts = df.filter(pl.col(count_col) < self.deidentification_rules["minimum_cell_size"])
                
                if len(low_counts) > 0:
                    violation_rate = len(low_counts) / len(df)
                    
                    violations.append(PrivacyViolation(
                        violation_type="statistical_disclosure_control_violation",
                        severity=PrivacyRiskLevel.HIGH if violation_rate > 0.1 else PrivacyRiskLevel.MEDIUM,
                        description=f"Cell counts below minimum threshold in {count_col}: {len(low_counts)} cells",
                        affected_columns=[count_col],
                        risk_score=8.0 if violation_rate > 0.1 else 6.0,
                        mitigation_required=True,
                        sample_violations=[{
                            "column": count_col,
                            "low_count_cells": len(low_counts),
                            "violation_rate": violation_rate,
                            "minimum_threshold": self.deidentification_rules["minimum_cell_size"]
                        }],
                        compliance_standard="Statistical Disclosure Control",
                        details={
                            "rule": "minimum_cell_size",
                            "threshold": self.deidentification_rules["minimum_cell_size"],
                            "violations": len(low_counts)
                        }
                    ))
        
        # Test for dominant contributions
        numeric_columns = [col for col in df.columns if df[col].dtype.is_numeric()]
        
        for col in numeric_columns:
            if len(df) > 0:
                total_sum = df[col].sum()
                if total_sum and total_sum > 0:
                    max_value = df[col].max()
                    dominance_ratio = max_value / total_sum
                    
                    if dominance_ratio > 0.8:  # 80% dominance threshold
                        violations.append(PrivacyViolation(
                            violation_type="dominance_disclosure_risk",
                            severity=PrivacyRiskLevel.MEDIUM,
                            description=f"High dominance ratio in {col}: {dominance_ratio:.1%}",
                            affected_columns=[col],
                            risk_score=6.5,
                            mitigation_required=True,
                            sample_violations=[{
                                "column": col,
                                "dominance_ratio": dominance_ratio,
                                "max_value": max_value,
                                "total_sum": total_sum
                            }],
                            compliance_standard="Statistical Disclosure Control",
                            details={
                                "rule": "dominance_control",
                                "threshold": 0.8,
                                "actual_ratio": dominance_ratio
                            }
                        ))
        
        return violations
    
    def test_k_anonymity_compliance(self, df: pl.DataFrame, quasi_identifier_columns: List[str], k: Optional[int] = None) -> List[PrivacyViolation]:
        """
        Test k-anonymity compliance for health data.
        
        Args:
            df: DataFrame to test
            quasi_identifier_columns: List of quasi-identifier columns
            k: Minimum group size (default from rules)
            
        Returns:
            List of k-anonymity violations
        """
        violations = []
        k_value = k or self.deidentification_rules["k_anonymity_k"]
        
        # Filter to existing columns
        existing_qi_columns = [col for col in quasi_identifier_columns if col in df.columns]
        
        if not existing_qi_columns:
            return violations
        
        try:
            # Group by quasi-identifiers and count occurrences
            grouped = df.group_by(existing_qi_columns).agg(pl.count().alias("group_count"))
            small_groups = grouped.filter(pl.col("group_count") < k_value)
            
            if len(small_groups) > 0:
                total_records_at_risk = small_groups["group_count"].sum()
                risk_percentage = (total_records_at_risk / len(df)) * 100
                
                severity = PrivacyRiskLevel.CRITICAL if risk_percentage > 20 else PrivacyRiskLevel.HIGH
                
                violations.append(PrivacyViolation(
                    violation_type="k_anonymity_violation",
                    severity=severity,
                    description=f"K-anonymity violation: {len(small_groups)} groups with size < {k_value}",
                    affected_columns=existing_qi_columns,
                    risk_score=9.0 if risk_percentage > 20 else 7.5,
                    mitigation_required=True,
                    sample_violations=[{
                        "k_value": k_value,
                        "small_groups_count": len(small_groups),
                        "records_at_risk": total_records_at_risk,
                        "risk_percentage": risk_percentage
                    }],
                    compliance_standard="K-Anonymity Privacy Standard",
                    details={
                        "quasi_identifiers": existing_qi_columns,
                        "minimum_k": k_value,
                        "actual_min_group_size": small_groups["group_count"].min()
                    }
                ))
        
        except Exception as e:
            self.logger.warning(f"K-anonymity validation failed: {str(e)}")
        
        return violations
    
    def test_l_diversity_compliance(self, df: pl.DataFrame, quasi_identifier_columns: List[str], sensitive_column: str, l: Optional[int] = None) -> List[PrivacyViolation]:
        """
        Test l-diversity compliance for health data.
        
        Args:
            df: DataFrame to test
            quasi_identifier_columns: List of quasi-identifier columns
            sensitive_column: Sensitive attribute column
            l: Minimum diversity (default from rules)
            
        Returns:
            List of l-diversity violations
        """
        violations = []
        l_value = l or self.deidentification_rules["l_diversity_l"]
        
        # Filter to existing columns
        existing_qi_columns = [col for col in quasi_identifier_columns if col in df.columns]
        
        if not existing_qi_columns or sensitive_column not in df.columns:
            return violations
        
        try:
            # Group by quasi-identifiers and check diversity of sensitive attribute
            grouped = df.group_by(existing_qi_columns).agg([
                pl.col(sensitive_column).n_unique().alias("sensitive_diversity"),
                pl.count().alias("group_count")
            ])
            
            low_diversity_groups = grouped.filter(pl.col("sensitive_diversity") < l_value)
            
            if len(low_diversity_groups) > 0:
                total_records_at_risk = low_diversity_groups["group_count"].sum()
                risk_percentage = (total_records_at_risk / len(df)) * 100
                
                severity = PrivacyRiskLevel.HIGH if risk_percentage > 15 else PrivacyRiskLevel.MEDIUM
                
                violations.append(PrivacyViolation(
                    violation_type="l_diversity_violation",
                    severity=severity,
                    description=f"L-diversity violation: {len(low_diversity_groups)} groups with diversity < {l_value}",
                    affected_columns=existing_qi_columns + [sensitive_column],
                    risk_score=8.0 if risk_percentage > 15 else 6.0,
                    mitigation_required=True,
                    sample_violations=[{
                        "l_value": l_value,
                        "low_diversity_groups": len(low_diversity_groups),
                        "records_at_risk": total_records_at_risk,
                        "risk_percentage": risk_percentage,
                        "sensitive_attribute": sensitive_column
                    }],
                    compliance_standard="L-Diversity Privacy Standard",
                    details={
                        "quasi_identifiers": existing_qi_columns,
                        "sensitive_attribute": sensitive_column,
                        "minimum_l": l_value,
                        "actual_min_diversity": low_diversity_groups["sensitive_diversity"].min()
                    }
                ))
        
        except Exception as e:
            self.logger.warning(f"L-diversity validation failed: {str(e)}")
        
        return violations
    
    def test_data_minimisation_compliance(self, df: pl.DataFrame, purpose_specification: str) -> List[PrivacyViolation]:
        """
        Test data minimisation principle compliance.
        
        Args:
            df: DataFrame to test
            purpose_specification: Declared purpose for data collection
            
        Returns:
            List of data minimisation violations
        """
        violations = []
        
        # Define purpose-specific necessary columns
        purpose_requirements = {
            "health_analytics": {
                "necessary": ["sa2_code", "age_group", "health_condition", "seifa_decile"],
                "optional": ["gender", "service_type", "provider_type"],
                "prohibited": ["name", "address", "phone", "email", "exact_age", "date_of_birth"]
            },
            "geographic_analysis": {
                "necessary": ["sa2_code", "sa3_code", "postcode"],
                "optional": ["age_group", "population_count"],
                "prohibited": ["name", "address", "phone", "email", "health_condition"]
            },
            "service_planning": {
                "necessary": ["sa2_code", "age_group", "service_type", "utilisation_rate"],
                "optional": ["provider_type", "distance_to_service"],
                "prohibited": ["name", "address", "phone", "email", "specific_diagnosis"]
            }
        }
        
        if purpose_specification.lower() not in purpose_requirements:
            # Generic assessment if purpose not recognised
            prohibited_columns = []
            for column in df.columns:
                column_lower = column.lower()
                if any(identifier in column_lower for identifier in self.health_identifiers["direct_identifiers"]):
                    prohibited_columns.append(column)
            
            if prohibited_columns:
                violations.append(PrivacyViolation(
                    violation_type="data_minimisation_violation",
                    severity=PrivacyRiskLevel.HIGH,
                    description=f"Direct identifiers present without clear purpose justification",
                    affected_columns=prohibited_columns,
                    risk_score=7.5,
                    mitigation_required=True,
                    sample_violations=[{"prohibited_columns": prohibited_columns}],
                    compliance_standard="APP3 - Collection of solicited personal information",
                    details={
                        "purpose": purpose_specification,
                        "recommendation": "Specify clear purpose or remove prohibited identifiers"
                    }
                ))
        else:
            purpose_reqs = purpose_requirements[purpose_specification.lower()]
            
            # Check for prohibited columns
            prohibited_found = []
            for column in df.columns:
                column_lower = column.lower()
                if any(prohibited in column_lower for prohibited in purpose_reqs["prohibited"]):
                    prohibited_found.append(column)
            
            if prohibited_found:
                violations.append(PrivacyViolation(
                    violation_type="data_minimisation_violation",
                    severity=PrivacyRiskLevel.HIGH,
                    description=f"Prohibited data elements for purpose '{purpose_specification}'",
                    affected_columns=prohibited_found,
                    risk_score=8.0,
                    mitigation_required=True,
                    sample_violations=[{"prohibited_columns": prohibited_found}],
                    compliance_standard="APP3 - Collection of solicited personal information",
                    details={
                        "purpose": purpose_specification,
                        "prohibited_elements": prohibited_found
                    }
                ))
            
            # Check for missing necessary columns
            necessary_missing = []
            for necessary_col in purpose_reqs["necessary"]:
                if not any(necessary_col in col.lower() for col in df.columns):
                    necessary_missing.append(necessary_col)
            
            if necessary_missing:
                violations.append(PrivacyViolation(
                    violation_type="inadequate_data_for_purpose",
                    severity=PrivacyRiskLevel.MEDIUM,
                    description=f"Missing necessary data elements for purpose '{purpose_specification}'",
                    affected_columns=[],
                    risk_score=5.0,
                    mitigation_required=False,
                    sample_violations=[{"missing_elements": necessary_missing}],
                    compliance_standard="Data Quality and Completeness",
                    details={
                        "purpose": purpose_specification,
                        "missing_elements": necessary_missing
                    }
                ))
        
        return violations
    
    def assess_privacy_protection(self, df: pl.DataFrame, purpose: str = "health_analytics") -> PrivacyProtectionResult:
        """
        Comprehensive privacy protection assessment.
        
        Args:
            df: DataFrame to assess
            purpose: Data processing purpose
            
        Returns:
            Privacy protection assessment result
        """
        assessment_id = f"privacy_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Classify data sensitivity
        sensitivity_classification = self.classify_health_data_sensitivity(df)
        
        # Validate de-identification completeness
        identifiers_found = self.validate_deidentification_completeness(df)
        
        # Test various compliance measures
        sdc_violations = self.test_statistical_disclosure_control(df)
        
        # Test k-anonymity with common quasi-identifiers
        quasi_identifiers = [col for col, sensitivity in sensitivity_classification.items() 
                           if sensitivity == HealthDataSensitivity.QUASI_IDENTIFIER]
        k_anonymity_violations = self.test_k_anonymity_compliance(df, quasi_identifiers)
        
        # Test l-diversity if sensitive attributes present
        sensitive_attributes = [col for col, sensitivity in sensitivity_classification.items() 
                              if sensitivity == HealthDataSensitivity.HIGHLY_SENSITIVE]
        l_diversity_violations = []
        if sensitive_attributes and quasi_identifiers:
            for sensitive_attr in sensitive_attributes[:1]:  # Test first sensitive attribute
                l_diversity_violations.extend(
                    self.test_l_diversity_compliance(df, quasi_identifiers, sensitive_attr)
                )
        
        # Test data minimisation
        minimisation_violations = self.test_data_minimisation_compliance(df, purpose)
        
        # Combine all violations
        all_violations = sdc_violations + k_anonymity_violations + l_diversity_violations + minimisation_violations
        
        # Calculate overall disclosure risk score
        disclosure_risk_score = self._calculate_disclosure_risk_score(df, sensitivity_classification, all_violations)
        
        # Determine privacy risk level
        if disclosure_risk_score < 0.2:
            privacy_risk_level = PrivacyRiskLevel.LOW
        elif disclosure_risk_score < 0.5:
            privacy_risk_level = PrivacyRiskLevel.MEDIUM
        elif disclosure_risk_score < 0.8:
            privacy_risk_level = PrivacyRiskLevel.HIGH
        else:
            privacy_risk_level = PrivacyRiskLevel.CRITICAL
        
        # Assess compliance status
        compliance_status = {
            "statistical_disclosure_control": len(sdc_violations) == 0,
            "k_anonymity": len(k_anonymity_violations) == 0,
            "l_diversity": len(l_diversity_violations) == 0,
            "data_minimisation": len(minimisation_violations) == 0,
            "overall_compliance": len(all_violations) == 0
        }
        
        # Generate recommendations
        recommendations = self._generate_privacy_recommendations(all_violations, sensitivity_classification)
        
        # Identify applied de-identification techniques
        deidentification_applied = self._identify_applied_techniques(df, sensitivity_classification)
        
        return PrivacyProtectionResult(
            dataset_id=assessment_id,
            assessment_timestamp=datetime.now().isoformat(),
            identifiers_found=identifiers_found,
            deidentification_applied=deidentification_applied,
            disclosure_risk_score=disclosure_risk_score,
            privacy_risk_level=privacy_risk_level,
            compliance_status=compliance_status,
            recommendations=recommendations,
            audit_trail={
                "assessment_method": "comprehensive_privacy_protection_assessment",
                "standards_tested": ["Statistical Disclosure Control", "K-Anonymity", "L-Diversity", "Data Minimisation"],
                "total_violations": len(all_violations),
                "violations_by_type": {
                    "sdc": len(sdc_violations),
                    "k_anonymity": len(k_anonymity_violations),
                    "l_diversity": len(l_diversity_violations),
                    "data_minimisation": len(minimisation_violations)
                }
            }
        )
    
    def _detect_identifier_patterns(self, sample_values: List[Any], column_name: str) -> List[Dict[str, Any]]:
        """Detect identifier patterns in sample values."""
        patterns = []
        
        for value in sample_values:
            if not isinstance(value, str):
                continue
            
            # Email pattern
            if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value):
                patterns.append({"type": "email", "column": column_name, "sample": value[:10] + "..."})
            
            # Phone pattern (Australian)
            elif re.match(r'^(\+61|0)[2-9]\d{8}$', re.sub(r'[\s\-\(\)]', '', value)):
                patterns.append({"type": "phone", "column": column_name, "sample": value[:5] + "..."})
            
            # Medicare number pattern
            elif re.match(r'^\d{10}$', value) or re.match(r'^\d{4}\s\d{5}\s\d{1}$', value):
                patterns.append({"type": "medicare_number", "column": column_name, "sample": "****"})
            
            # Name pattern (basic heuristic)
            elif len(value.split()) >= 2 and all(part.isalpha() and len(part) > 1 for part in value.split()):
                patterns.append({"type": "name", "column": column_name, "sample": value[:3] + "..."})
        
        return patterns
    
    def _calculate_disclosure_risk_score(self, df: pl.DataFrame, sensitivity_classification: Dict[str, HealthDataSensitivity], violations: List[PrivacyViolation]) -> float:
        """Calculate overall disclosure risk score."""
        risk_factors = []
        
        # Factor 1: Sensitive column density
        sensitive_columns = [col for col, sensitivity in sensitivity_classification.items() 
                           if sensitivity in [HealthDataSensitivity.CRITICAL, HealthDataSensitivity.HIGHLY_SENSITIVE]]
        sensitive_density = len(sensitive_columns) / len(df.columns) if df.columns else 0
        risk_factors.append(sensitive_density * self.risk_weights["direct_identifier_weight"])
        
        # Factor 2: Dataset size (smaller datasets have higher risk)
        size_risk = max(0, 1 - (len(df) / 10000))
        risk_factors.append(size_risk * self.risk_weights["dataset_size_weight"])
        
        # Factor 3: Violation severity
        violation_risk = 0
        for violation in violations:
            if violation.severity == PrivacyRiskLevel.CRITICAL:
                violation_risk += 0.4
            elif violation.severity == PrivacyRiskLevel.HIGH:
                violation_risk += 0.3
            elif violation.severity == PrivacyRiskLevel.MEDIUM:
                violation_risk += 0.1
        
        risk_factors.append(min(violation_risk, 1.0))
        
        # Combine risk factors (normalised)
        total_risk = sum(risk_factors) / len(risk_factors) if risk_factors else 0.5
        return min(total_risk, 1.0)
    
    def _generate_privacy_recommendations(self, violations: List[PrivacyViolation], sensitivity_classification: Dict[str, HealthDataSensitivity]) -> List[str]:
        """Generate privacy protection recommendations."""
        recommendations = []
        
        violation_types = [v.violation_type for v in violations]
        
        if "statistical_disclosure_control_violation" in violation_types:
            recommendations.append("Apply cell suppression or perturbation for counts below minimum threshold")
        
        if "k_anonymity_violation" in violation_types:
            recommendations.append("Increase generalisation of quasi-identifiers to achieve k-anonymity")
        
        if "l_diversity_violation" in violation_types:
            recommendations.append("Ensure diversity in sensitive attributes within each quasi-identifier group")
        
        if "data_minimisation_violation" in violation_types:
            recommendations.append("Remove unnecessary data elements that exceed purpose requirements")
        
        # Check for critical sensitive columns
        critical_columns = [col for col, sensitivity in sensitivity_classification.items() 
                          if sensitivity == HealthDataSensitivity.CRITICAL]
        if critical_columns:
            recommendations.append(f"Remove or mask critical identifiers: {', '.join(critical_columns)}")
        
        # General recommendations
        if len(violations) > 5:
            recommendations.append("Consider synthetic data generation as alternative to extensive de-identification")
        
        if not recommendations:
            recommendations.append("Continue monitoring privacy protection measures and conduct regular assessments")
        
        return recommendations
    
    def _identify_applied_techniques(self, df: pl.DataFrame, sensitivity_classification: Dict[str, HealthDataSensitivity]) -> List[DeIdentificationTechnique]:
        """Identify applied de-identification techniques based on data characteristics."""
        techniques = []
        
        # Check for generalisation (age groups, date ranges)
        age_columns = [col for col in df.columns if "age" in col.lower()]
        for col in age_columns:
            if "group" in col.lower() or "range" in col.lower():
                techniques.append(DeIdentificationTechnique.GENERALIZATION)
                break
        
        # Check for suppression (missing critical identifiers)
        critical_identifiers = ["name", "address", "phone", "email", "medicare_number"]
        has_critical = any(any(identifier in col.lower() for identifier in critical_identifiers) 
                          for col in df.columns)
        if not has_critical:
            techniques.append(DeIdentificationTechnique.SUPPRESSION)
        
        # Check for aggregation (statistical summaries)
        aggregation_indicators = ["count", "total", "average", "mean", "sum"]
        has_aggregation = any(any(indicator in col.lower() for indicator in aggregation_indicators) 
                            for col in df.columns)
        if has_aggregation:
            techniques.append(DeIdentificationTechnique.AGGREGATION)
        
        # Default to pseudonymisation if no specific technique detected
        if not techniques:
            techniques.append(DeIdentificationTechnique.PSEUDONYMIZATION)
        
        return techniques


# Test suite
class TestHealthDataPrivacyProtection:
    """Test suite for health data privacy protection."""
    
    @pytest.fixture
    def privacy_tester(self):
        """Create privacy tester instance."""
        return HealthDataPrivacyTester()
    
    @pytest.fixture
    def raw_health_dataset(self):
        """Raw health dataset with identifiers (for testing)."""
        return pl.DataFrame({
            "patient_id": ["P12345", "P67890", "P11111", "P22222", "P33333"],
            "name": ["John Smith", "Jane Doe", "Bob Johnson", "Alice Brown", "Charlie Davis"],
            "medicare_number": ["1234567890", "0987654321", "1111111111", "2222222222", "3333333333"],
            "date_of_birth": ["1980-03-15", "1975-07-22", "1992-11-08", "1988-05-12", "1995-01-30"],
            "address": ["123 Main St", "456 Oak Ave", "789 Pine Rd", "321 Elm St", "654 Maple Dr"],
            "phone": ["0412345678", "0498765432", "0411223344", "0422334455", "0433445566"],
            "email": ["john@example.com", "jane@example.com", "bob@example.com", "alice@example.com", "charlie@example.com"],
            "sa2_code_2021": ["101021007", "201011001", "301011002", "101021007", "201011001"],
            "age": [43, 48, 31, 35, 28],
            "gender": ["M", "F", "M", "F", "M"],
            "diagnosis": ["diabetes", "hypertension", "asthma", "diabetes", "hypertension"],
            "provider_id": ["PR001", "PR002", "PR003", "PR001", "PR002"]
        })
    
    @pytest.fixture
    def deidentified_health_dataset(self):
        """De-identified health dataset."""
        return pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001", "301011002", "101021007", "201011001"],
            "age_group": ["40-44", "45-49", "30-34", "35-39", "25-29"],
            "gender": ["M", "F", "M", "F", "M"],
            "diagnosis_category": ["endocrine", "cardiovascular", "respiratory", "endocrine", "cardiovascular"],
            "provider_type": ["GP", "specialist", "GP", "GP", "specialist"],
            "seifa_decile": [7, 5, 8, 7, 5]
        })
    
    @pytest.fixture
    def aggregated_health_dataset(self):
        """Aggregated health dataset for statistical disclosure control testing."""
        return pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001", "301011002", "401011003", "501011004"],
            "age_group": ["30-34", "35-39", "40-44", "45-49", "50-54"],
            "diagnosis_count": [25, 3, 18, 2, 35],  # Some low counts for testing
            "total_patients": [1500, 45, 800, 25, 2000],
            "average_cost": [285.50, 195.75, 340.25, 425.80, 310.60]
        })
    
    def test_health_data_sensitivity_classification(self, privacy_tester, raw_health_dataset):
        """Test health data sensitivity classification."""
        classification = privacy_tester.classify_health_data_sensitivity(raw_health_dataset)
        
        # Verify critical identifiers are correctly classified
        critical_columns = ["patient_id", "name", "medicare_number", "date_of_birth", "address", "phone", "email"]
        for col in critical_columns:
            assert classification[col] == HealthDataSensitivity.CRITICAL, f"Column {col} should be classified as CRITICAL"
        
        # Verify quasi-identifiers
        quasi_columns = ["sa2_code_2021", "age", "gender"]
        for col in quasi_columns:
            assert classification[col] == HealthDataSensitivity.QUASI_IDENTIFIER, f"Column {col} should be classified as QUASI_IDENTIFIER"
        
        # Verify sensitive attributes
        sensitive_columns = ["diagnosis"]
        for col in sensitive_columns:
            assert classification[col] == HealthDataSensitivity.HIGHLY_SENSITIVE, f"Column {col} should be classified as HIGHLY_SENSITIVE"
    
    def test_deidentification_completeness_validation(self, privacy_tester, raw_health_dataset):
        """Test de-identification completeness validation."""
        remaining_identifiers = privacy_tester.validate_deidentification_completeness(raw_health_dataset)
        
        # Should detect remaining identifiers in raw dataset
        assert len(remaining_identifiers) > 0, "Should detect remaining identifiers in raw dataset"
        
        # Verify critical identifiers are flagged
        flagged_types = [identifier.identifier_type for identifier in remaining_identifiers]
        critical_identifiers = ["patient_id", "name", "medicare_number", "date_of_birth", "address", "phone", "email"]
        
        for critical_id in critical_identifiers:
            assert critical_id in flagged_types, f"Critical identifier {critical_id} should be flagged"
        
        # All flagged identifiers should require removal
        for identifier in remaining_identifiers:
            assert identifier.removal_required is True
            assert identifier.compliance_standard == "APP11"
    
    def test_statistical_disclosure_control(self, privacy_tester, aggregated_health_dataset):
        """Test statistical disclosure control validation."""
        violations = privacy_tester.test_statistical_disclosure_control(aggregated_health_dataset)
        
        # Should detect low cell counts
        sdc_violations = [v for v in violations if v.violation_type == "statistical_disclosure_control_violation"]
        assert len(sdc_violations) > 0, "Should detect statistical disclosure control violations"
        
        for violation in sdc_violations:
            assert violation.severity in [PrivacyRiskLevel.HIGH, PrivacyRiskLevel.MEDIUM]
            assert violation.mitigation_required is True
            assert "Statistical Disclosure Control" in violation.compliance_standard
            assert violation.risk_score >= 6.0
    
    def test_k_anonymity_compliance(self, privacy_tester, raw_health_dataset):
        """Test k-anonymity compliance validation."""
        quasi_identifiers = ["sa2_code_2021", "age", "gender"]
        violations = privacy_tester.test_k_anonymity_compliance(raw_health_dataset, quasi_identifiers, k=5)
        
        # Small dataset should likely have k-anonymity violations
        k_violations = [v for v in violations if v.violation_type == "k_anonymity_violation"]
        
        if k_violations:  # May not always have violations depending on data distribution
            for violation in k_violations:
                assert violation.severity in [PrivacyRiskLevel.CRITICAL, PrivacyRiskLevel.HIGH]
                assert "K-Anonymity Privacy Standard" in violation.compliance_standard
                assert violation.mitigation_required is True
                assert len(violation.affected_columns) >= len([col for col in quasi_identifiers if col in raw_health_dataset.columns])
    
    def test_l_diversity_compliance(self, privacy_tester, raw_health_dataset):
        """Test l-diversity compliance validation."""
        quasi_identifiers = ["sa2_code_2021", "age"]
        sensitive_attribute = "diagnosis"
        violations = privacy_tester.test_l_diversity_compliance(
            raw_health_dataset, quasi_identifiers, sensitive_attribute, l=3
        )
        
        # May have l-diversity violations with small dataset
        l_violations = [v for v in violations if v.violation_type == "l_diversity_violation"]
        
        if l_violations:
            for violation in l_violations:
                assert violation.severity in [PrivacyRiskLevel.HIGH, PrivacyRiskLevel.MEDIUM]
                assert "L-Diversity Privacy Standard" in violation.compliance_standard
                assert sensitive_attribute in violation.affected_columns
                assert violation.mitigation_required is True
    
    def test_data_minimisation_compliance(self, privacy_tester, raw_health_dataset):
        """Test data minimisation compliance."""
        violations = privacy_tester.test_data_minimisation_compliance(raw_health_dataset, "health_analytics")
        
        # Should detect prohibited data elements
        minimisation_violations = [v for v in violations if v.violation_type == "data_minimisation_violation"]
        assert len(minimisation_violations) > 0, "Should detect data minimisation violations"
        
        for violation in minimisation_violations:
            assert violation.severity == PrivacyRiskLevel.HIGH
            assert "APP3" in violation.compliance_standard
            assert violation.mitigation_required is True
            assert len(violation.affected_columns) > 0
    
    def test_comprehensive_privacy_protection_assessment(self, privacy_tester, raw_health_dataset):
        """Test comprehensive privacy protection assessment."""
        assessment = privacy_tester.assess_privacy_protection(raw_health_dataset, "health_analytics")
        
        # Verify assessment structure
        assert isinstance(assessment, PrivacyProtectionResult)
        assert assessment.dataset_id is not None
        assert assessment.assessment_timestamp is not None
        assert len(assessment.identifiers_found) > 0  # Should find identifiers in raw data
        assert isinstance(assessment.deidentification_applied, list)
        assert 0.0 <= assessment.disclosure_risk_score <= 1.0
        assert isinstance(assessment.privacy_risk_level, PrivacyRiskLevel)
        assert isinstance(assessment.compliance_status, dict)
        assert isinstance(assessment.recommendations, list)
        assert isinstance(assessment.audit_trail, dict)
        
        # Raw health dataset should have high disclosure risk
        assert assessment.privacy_risk_level in [PrivacyRiskLevel.HIGH, PrivacyRiskLevel.CRITICAL]
        assert assessment.disclosure_risk_score > 0.5
        
        # Should have compliance failures
        assert assessment.compliance_status["overall_compliance"] is False
        
        # Should have recommendations
        assert len(assessment.recommendations) > 0
        
        # Audit trail should be complete
        assert "assessment_method" in assessment.audit_trail
        assert "total_violations" in assessment.audit_trail
        assert assessment.audit_trail["total_violations"] > 0
    
    def test_deidentified_dataset_assessment(self, privacy_tester, deidentified_health_dataset):
        """Test assessment of properly de-identified dataset."""
        assessment = privacy_tester.assess_privacy_protection(deidentified_health_dataset, "health_analytics")
        
        # De-identified dataset should have lower risk
        assert assessment.privacy_risk_level in [PrivacyRiskLevel.LOW, PrivacyRiskLevel.MEDIUM]
        assert assessment.disclosure_risk_score < 0.8
        
        # Should have fewer identifiers found
        assert len(assessment.identifiers_found) < 5  # Significantly fewer than raw dataset
        
        # Should have better compliance
        compliance_count = sum(assessment.compliance_status.values())
        assert compliance_count >= 2  # At least some compliance measures should pass
    
    def test_privacy_protection_edge_cases(self, privacy_tester):
        """Test privacy protection with edge cases."""
        # Empty dataset
        empty_df = pl.DataFrame()
        assessment = privacy_tester.assess_privacy_protection(empty_df, "health_analytics")
        assert assessment.privacy_risk_level == PrivacyRiskLevel.LOW  # No data, no risk
        
        # Single column dataset
        single_col_df = pl.DataFrame({"count": [1, 2, 3, 4, 5]})
        assessment = privacy_tester.assess_privacy_protection(single_col_df, "health_analytics")
        assert isinstance(assessment, PrivacyProtectionResult)
        
        # Dataset with only non-sensitive columns
        non_sensitive_df = pl.DataFrame({
            "total_population": [1000, 1500, 2000],
            "area_km2": [50.5, 75.3, 120.8],
            "category": ["A", "B", "C"]
        })
        assessment = privacy_tester.assess_privacy_protection(non_sensitive_df, "health_analytics")
        assert assessment.privacy_risk_level in [PrivacyRiskLevel.LOW, PrivacyRiskLevel.MEDIUM]
    
    def test_health_identifier_detection_patterns(self, privacy_tester):
        """Test health identifier pattern detection."""
        test_values = [
            "john.smith@example.com",  # Email
            "0412345678",              # Phone
            "1234567890",              # Medicare number
            "John Smith",              # Name
            "123 Main Street"          # Address
        ]
        
        patterns = privacy_tester._detect_identifier_patterns(test_values, "test_column")
        
        # Should detect multiple pattern types
        pattern_types = [p["type"] for p in patterns]
        expected_types = ["email", "phone", "medicare_number", "name"]
        
        for expected_type in expected_types:
            assert expected_type in pattern_types, f"Should detect {expected_type} pattern"
    
    def test_privacy_risk_scoring_accuracy(self, privacy_tester, raw_health_dataset, deidentified_health_dataset):
        """Test accuracy of privacy risk scoring."""
        # Raw dataset should have higher risk score than de-identified
        raw_assessment = privacy_tester.assess_privacy_protection(raw_health_dataset, "health_analytics")
        deidentified_assessment = privacy_tester.assess_privacy_protection(deidentified_health_dataset, "health_analytics")
        
        assert raw_assessment.disclosure_risk_score > deidentified_assessment.disclosure_risk_score
        assert raw_assessment.privacy_risk_level.value != "low"  # Raw data shouldn't be low risk
        
        # Risk scores should be within valid range
        assert 0.0 <= raw_assessment.disclosure_risk_score <= 1.0
        assert 0.0 <= deidentified_assessment.disclosure_risk_score <= 1.0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])