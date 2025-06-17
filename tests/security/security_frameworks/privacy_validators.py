"""
Privacy Validation Framework

Comprehensive privacy validation utilities and frameworks for health data
privacy protection, de-identification validation, and Australian Privacy
Principles (APP) compliance testing.
"""

import json
import re
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging

import polars as pl
import numpy as np
from loguru import logger


class PrivacyLevel(Enum):
    """Privacy protection levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    SECRET = "secret"


class DeidentificationMethod(Enum):
    """De-identification methods."""
    SUPPRESSION = "suppression"
    GENERALISATION = "generalisation"
    PERTURBATION = "perturbation"
    AGGREGATION = "aggregation"
    PSEUDONYMISATION = "pseudonymisation"
    ANONYMISATION = "anonymisation"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    SYNTHETIC_DATA = "synthetic_data"


@dataclass
class PrivacyValidationResult:
    """Privacy validation result."""
    validation_id: str
    timestamp: str
    dataset_id: str
    privacy_level: PrivacyLevel
    is_compliant: bool
    violations_found: List[Dict[str, Any]]
    recommendations: List[str]
    risk_score: float
    confidence_level: float
    details: Dict[str, Any]


@dataclass
class DeidentificationValidation:
    """De-identification validation result."""
    validation_id: str
    dataset_id: str
    original_record_count: int
    deidentified_record_count: int
    methods_applied: List[DeidentificationMethod]
    effectiveness_score: float
    utility_preservation: float
    disclosure_risk: float
    k_anonymity_compliance: bool
    l_diversity_compliance: bool
    statistical_disclosure_control: bool
    recommendations: List[str]


class PrivacyValidator:
    """Privacy validation framework for health data."""
    
    def __init__(self):
        """Initialise privacy validator."""
        self.logger = logger.bind(component="privacy_validator")
        
        # Australian health data identifiers (based on HIPAA Safe Harbor + Australian context)
        self.direct_identifiers = [
            "name", "full_name", "first_name", "last_name", "surname", "given_name",
            "medicare_number", "healthcare_identifier", "patient_id", "person_id",
            "address", "street_address", "home_address", "postal_address",
            "phone", "telephone", "mobile", "contact_number", "fax",
            "email", "email_address", "web_address", "url",
            "social_security_number", "tax_file_number", "driver_licence",
            "passport_number", "certificate_number", "licence_number",
            "account_number", "medical_record_number", "health_plan_number",
            "date_of_birth", "birth_date", "dob", "birth_year", "birth_month",
            "death_date", "date_of_death", "admission_date", "discharge_date"
        ]
        
        # Quasi-identifiers for Australian health data
        self.quasi_identifiers = [
            "postcode", "suburb", "sa1_code", "sa2_code", "sa3_code", "sa4_code",
            "age", "exact_age", "age_in_days", "age_group",
            "gender", "sex", "marital_status",
            "occupation", "employment", "profession", "job_title", "industry",
            "indigenous_status", "country_of_birth", "language", "religion",
            "education_level", "income_bracket", "household_composition",
            "veteran_status", "disability_status", "carer_status"
        ]
        
        # Sensitive health attributes
        self.sensitive_attributes = [
            "diagnosis", "icd_code", "health_condition", "medical_condition",
            "medication", "drug_name", "prescription", "treatment", "procedure",
            "mental_health", "psychiatric", "psychological", "substance_abuse",
            "reproductive_health", "pregnancy", "fertility", "contraception",
            "genetic_information", "dna", "genetic_test", "family_history",
            "disability", "chronic_condition", "terminal_illness",
            "laboratory_result", "test_result", "pathology", "radiology"
        ]
        
        # Privacy risk scoring weights
        self.risk_weights = {
            "direct_identifier_presence": 10.0,
            "quasi_identifier_density": 3.0,
            "sensitive_attribute_exposure": 5.0,
            "data_volume": 2.0,
            "uniqueness_risk": 4.0,
            "linkage_risk": 6.0
        }
        
        # Statistical disclosure control thresholds
        self.sdc_thresholds = {
            "minimum_cell_count": 5,
            "maximum_dominance": 0.85,
            "k_anonymity_k": 5,
            "l_diversity_l": 3,
            "suppression_threshold": 0.05
        }
    
    def validate_privacy_compliance(self, df: pl.DataFrame, expected_privacy_level: PrivacyLevel, 
                                  dataset_id: str = "unknown") -> PrivacyValidationResult:
        """
        Validate privacy compliance for a dataset.
        
        Args:
            df: DataFrame to validate
            expected_privacy_level: Expected privacy protection level
            dataset_id: Identifier for the dataset
            
        Returns:
            Privacy validation result
        """
        validation_id = f"privacy_val_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        violations = []
        recommendations = []
        
        # Check for direct identifiers
        direct_id_violations = self._check_direct_identifiers(df, expected_privacy_level)
        violations.extend(direct_id_violations)
        
        # Check quasi-identifier density
        qi_violations = self._check_quasi_identifier_density(df, expected_privacy_level)
        violations.extend(qi_violations)
        
        # Check sensitive attribute protection
        sensitive_violations = self._check_sensitive_attributes(df, expected_privacy_level)
        violations.extend(sensitive_violations)
        
        # Calculate risk score
        risk_score = self._calculate_privacy_risk_score(df, violations)
        
        # Determine compliance status
        is_compliant = len([v for v in violations if v["severity"] in ["critical", "high"]]) == 0
        
        # Generate recommendations
        recommendations = self._generate_privacy_recommendations(violations, expected_privacy_level)
        
        # Calculate confidence level
        confidence_level = self._calculate_confidence_level(df, violations)
        
        return PrivacyValidationResult(
            validation_id=validation_id,
            timestamp=datetime.now().isoformat(),
            dataset_id=dataset_id,
            privacy_level=expected_privacy_level,
            is_compliant=is_compliant,
            violations_found=violations,
            recommendations=recommendations,
            risk_score=risk_score,
            confidence_level=confidence_level,
            details={
                "total_columns": len(df.columns),
                "total_records": len(df),
                "direct_identifiers_found": len(direct_id_violations),
                "quasi_identifiers_found": len(qi_violations),
                "sensitive_attributes_found": len(sensitive_violations)
            }
        )
    
    def validate_deidentification(self, original_df: pl.DataFrame, deidentified_df: pl.DataFrame,
                                dataset_id: str = "unknown") -> DeidentificationValidation:
        """
        Validate de-identification effectiveness.
        
        Args:
            original_df: Original dataset
            deidentified_df: De-identified dataset
            dataset_id: Identifier for the dataset
            
        Returns:
            De-identification validation result
        """
        validation_id = f"deident_val_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Identify applied de-identification methods
        methods_applied = self._identify_deidentification_methods(original_df, deidentified_df)
        
        # Calculate effectiveness metrics
        effectiveness_score = self._calculate_deidentification_effectiveness(original_df, deidentified_df)
        utility_preservation = self._calculate_utility_preservation(original_df, deidentified_df)
        disclosure_risk = self._calculate_disclosure_risk(deidentified_df)
        
        # Test privacy protection standards
        k_anonymity_compliance = self._test_k_anonymity(deidentified_df)
        l_diversity_compliance = self._test_l_diversity(deidentified_df)
        sdc_compliance = self._test_statistical_disclosure_control(deidentified_df)
        
        # Generate recommendations
        recommendations = self._generate_deidentification_recommendations(
            effectiveness_score, utility_preservation, disclosure_risk,
            k_anonymity_compliance, l_diversity_compliance, sdc_compliance
        )
        
        return DeidentificationValidation(
            validation_id=validation_id,
            dataset_id=dataset_id,
            original_record_count=len(original_df),
            deidentified_record_count=len(deidentified_df),
            methods_applied=methods_applied,
            effectiveness_score=effectiveness_score,
            utility_preservation=utility_preservation,
            disclosure_risk=disclosure_risk,
            k_anonymity_compliance=k_anonymity_compliance,
            l_diversity_compliance=l_diversity_compliance,
            statistical_disclosure_control=sdc_compliance,
            recommendations=recommendations
        )
    
    def validate_k_anonymity(self, df: pl.DataFrame, quasi_identifier_columns: List[str], 
                           k: int = 5) -> Dict[str, Any]:
        """
        Validate k-anonymity compliance.
        
        Args:
            df: DataFrame to test
            quasi_identifier_columns: Quasi-identifier columns
            k: Minimum group size
            
        Returns:
            K-anonymity validation result
        """
        # Filter to existing columns
        existing_qi_columns = [col for col in quasi_identifier_columns if col in df.columns]
        
        if not existing_qi_columns:
            return {
                "compliant": True,
                "reason": "No quasi-identifiers found",
                "groups_below_k": 0,
                "records_at_risk": 0
            }
        
        try:
            # Group by quasi-identifiers and count occurrences
            grouped = df.group_by(existing_qi_columns).agg(pl.count().alias("group_count"))
            small_groups = grouped.filter(pl.col("group_count") < k)
            
            groups_below_k = len(small_groups)
            records_at_risk = small_groups["group_count"].sum() if groups_below_k > 0 else 0
            
            return {
                "compliant": groups_below_k == 0,
                "k_value": k,
                "quasi_identifiers": existing_qi_columns,
                "total_groups": len(grouped),
                "groups_below_k": groups_below_k,
                "records_at_risk": records_at_risk,
                "risk_percentage": (records_at_risk / len(df)) * 100 if len(df) > 0 else 0
            }
        
        except Exception as e:
            self.logger.warning(f"K-anonymity validation failed: {str(e)}")
            return {
                "compliant": False,
                "error": str(e),
                "groups_below_k": -1,
                "records_at_risk": -1
            }
    
    def validate_l_diversity(self, df: pl.DataFrame, quasi_identifier_columns: List[str], 
                           sensitive_column: str, l: int = 3) -> Dict[str, Any]:
        """
        Validate l-diversity compliance.
        
        Args:
            df: DataFrame to test
            quasi_identifier_columns: Quasi-identifier columns
            sensitive_column: Sensitive attribute column
            l: Minimum diversity
            
        Returns:
            L-diversity validation result
        """
        # Filter to existing columns
        existing_qi_columns = [col for col in quasi_identifier_columns if col in df.columns]
        
        if not existing_qi_columns or sensitive_column not in df.columns:
            return {
                "compliant": True,
                "reason": "Required columns not found",
                "groups_below_l": 0,
                "records_at_risk": 0
            }
        
        try:
            # Group by quasi-identifiers and check diversity of sensitive attribute
            grouped = df.group_by(existing_qi_columns).agg([
                pl.col(sensitive_column).n_unique().alias("sensitive_diversity"),
                pl.count().alias("group_count")
            ])
            
            low_diversity_groups = grouped.filter(pl.col("sensitive_diversity") < l)
            groups_below_l = len(low_diversity_groups)
            records_at_risk = low_diversity_groups["group_count"].sum() if groups_below_l > 0 else 0
            
            return {
                "compliant": groups_below_l == 0,
                "l_value": l,
                "quasi_identifiers": existing_qi_columns,
                "sensitive_attribute": sensitive_column,
                "total_groups": len(grouped),
                "groups_below_l": groups_below_l,
                "records_at_risk": records_at_risk,
                "risk_percentage": (records_at_risk / len(df)) * 100 if len(df) > 0 else 0
            }
        
        except Exception as e:
            self.logger.warning(f"L-diversity validation failed: {str(e)}")
            return {
                "compliant": False,
                "error": str(e),
                "groups_below_l": -1,
                "records_at_risk": -1
            }
    
    def _check_direct_identifiers(self, df: pl.DataFrame, privacy_level: PrivacyLevel) -> List[Dict[str, Any]]:
        """Check for presence of direct identifiers."""
        violations = []
        
        for column in df.columns:
            column_lower = column.lower()
            
            # Check if column name matches direct identifier patterns
            for identifier in self.direct_identifiers:
                if identifier in column_lower:
                    # Direct identifiers should only be in SECRET or RESTRICTED data
                    if privacy_level not in [PrivacyLevel.SECRET, PrivacyLevel.RESTRICTED]:
                        violations.append({
                            "violation_type": "direct_identifier_present",
                            "severity": "critical",
                            "column": column,
                            "identifier_type": identifier,
                            "description": f"Direct identifier '{identifier}' found in {privacy_level.value} data",
                            "recommendation": f"Remove or mask {identifier} for {privacy_level.value} classification"
                        })
                    break
        
        return violations
    
    def _check_quasi_identifier_density(self, df: pl.DataFrame, privacy_level: PrivacyLevel) -> List[Dict[str, Any]]:
        """Check quasi-identifier density."""
        violations = []
        
        qi_columns = []
        for column in df.columns:
            column_lower = column.lower()
            if any(qi in column_lower for qi in self.quasi_identifiers):
                qi_columns.append(column)
        
        qi_density = len(qi_columns) / len(df.columns) if df.columns else 0
        
        # Different thresholds for different privacy levels
        thresholds = {
            PrivacyLevel.PUBLIC: 0.2,
            PrivacyLevel.INTERNAL: 0.4,
            PrivacyLevel.CONFIDENTIAL: 0.6,
            PrivacyLevel.RESTRICTED: 0.8,
            PrivacyLevel.SECRET: 1.0
        }
        
        threshold = thresholds.get(privacy_level, 0.5)
        
        if qi_density > threshold:
            violations.append({
                "violation_type": "high_quasi_identifier_density",
                "severity": "high" if qi_density > threshold * 1.5 else "medium",
                "qi_density": qi_density,
                "threshold": threshold,
                "qi_columns": qi_columns,
                "description": f"High quasi-identifier density ({qi_density:.1%}) for {privacy_level.value} data",
                "recommendation": "Consider generalising or suppressing quasi-identifiers"
            })
        
        return violations
    
    def _check_sensitive_attributes(self, df: pl.DataFrame, privacy_level: PrivacyLevel) -> List[Dict[str, Any]]:
        """Check sensitive attribute protection."""
        violations = []
        
        sensitive_columns = []
        for column in df.columns:
            column_lower = column.lower()
            if any(sensitive in column_lower for sensitive in self.sensitive_attributes):
                sensitive_columns.append(column)
        
        # Sensitive attributes in public data should be protected
        if privacy_level == PrivacyLevel.PUBLIC and sensitive_columns:
            violations.append({
                "violation_type": "sensitive_attributes_in_public_data",
                "severity": "high",
                "sensitive_columns": sensitive_columns,
                "description": f"Sensitive health attributes found in {privacy_level.value} data",
                "recommendation": "Remove, aggregate, or further de-identify sensitive attributes"
            })
        
        return violations
    
    def _calculate_privacy_risk_score(self, df: pl.DataFrame, violations: List[Dict[str, Any]]) -> float:
        """Calculate overall privacy risk score (0-10, lower is better)."""
        score = 0.0
        
        # Base score from violations
        for violation in violations:
            if violation["severity"] == "critical":
                score += 3.0
            elif violation["severity"] == "high":
                score += 2.0
            elif violation["severity"] == "medium":
                score += 1.0
            else:
                score += 0.5
        
        # Dataset size factor (larger datasets have higher inherent risk)
        if len(df) > 100000:
            score += 1.0
        elif len(df) > 10000:
            score += 0.5
        
        # Column count factor
        if len(df.columns) > 50:
            score += 0.5
        
        return min(score, 10.0)
    
    def _generate_privacy_recommendations(self, violations: List[Dict[str, Any]], 
                                        privacy_level: PrivacyLevel) -> List[str]:
        """Generate privacy protection recommendations."""
        recommendations = []
        
        violation_types = [v["violation_type"] for v in violations]
        
        if "direct_identifier_present" in violation_types:
            recommendations.append("Remove or pseudonymise direct identifiers")
        
        if "high_quasi_identifier_density" in violation_types:
            recommendations.append("Apply generalisation to reduce quasi-identifier precision")
        
        if "sensitive_attributes_in_public_data" in violation_types:
            recommendations.append("Remove or aggregate sensitive health attributes")
        
        # General recommendations based on privacy level
        if privacy_level == PrivacyLevel.PUBLIC:
            recommendations.append("Ensure statistical disclosure control measures are applied")
            recommendations.append("Consider differential privacy for additional protection")
        
        return recommendations
    
    def _calculate_confidence_level(self, df: pl.DataFrame, violations: List[Dict[str, Any]]) -> float:
        """Calculate confidence level in privacy assessment (0-1)."""
        base_confidence = 0.8
        
        # Reduce confidence for critical violations
        critical_violations = len([v for v in violations if v["severity"] == "critical"])
        confidence = base_confidence - (critical_violations * 0.1)
        
        # Reduce confidence for very small datasets (harder to assess)
        if len(df) < 100:
            confidence -= 0.1
        
        # Reduce confidence for datasets with many columns (complexity)
        if len(df.columns) > 100:
            confidence -= 0.1
        
        return max(confidence, 0.1)
    
    def _identify_deidentification_methods(self, original_df: pl.DataFrame, 
                                         deidentified_df: pl.DataFrame) -> List[DeidentificationMethod]:
        """Identify de-identification methods applied."""
        methods = []
        
        # Check for suppression (missing columns)
        if len(deidentified_df.columns) < len(original_df.columns):
            methods.append(DeidentificationMethod.SUPPRESSION)
        
        # Check for aggregation (fewer rows)
        if len(deidentified_df) < len(original_df):
            methods.append(DeidentificationMethod.AGGREGATION)
        
        # Check for generalisation (age ranges, etc.)
        common_columns = set(original_df.columns) & set(deidentified_df.columns)
        for col in common_columns:
            if "group" in col.lower() or "range" in col.lower():
                methods.append(DeidentificationMethod.GENERALISATION)
                break
        
        # Check for pseudonymisation (different IDs but same count)
        if len(deidentified_df) == len(original_df) and len(deidentified_df.columns) == len(original_df.columns):
            methods.append(DeidentificationMethod.PSEUDONYMISATION)
        
        return list(set(methods))  # Remove duplicates
    
    def _calculate_deidentification_effectiveness(self, original_df: pl.DataFrame, 
                                                deidentified_df: pl.DataFrame) -> float:
        """Calculate de-identification effectiveness (0-1, higher is better)."""
        # Count direct identifiers in both datasets
        original_identifiers = self._count_identifiers(original_df)
        deidentified_identifiers = self._count_identifiers(deidentified_df)
        
        if original_identifiers == 0:
            return 1.0  # No identifiers to remove
        
        effectiveness = 1.0 - (deidentified_identifiers / original_identifiers)
        return max(effectiveness, 0.0)
    
    def _calculate_utility_preservation(self, original_df: pl.DataFrame, 
                                      deidentified_df: pl.DataFrame) -> float:
        """Calculate utility preservation (0-1, higher is better)."""
        if len(original_df.columns) == 0:
            return 0.0
        
        # Column preservation
        common_columns = set(original_df.columns) & set(deidentified_df.columns)
        column_preservation = len(common_columns) / len(original_df.columns)
        
        # Row preservation
        row_preservation = len(deidentified_df) / len(original_df) if len(original_df) > 0 else 0
        
        # Overall utility (weighted average)
        utility = (column_preservation * 0.6) + (min(row_preservation, 1.0) * 0.4)
        return utility
    
    def _calculate_disclosure_risk(self, df: pl.DataFrame) -> float:
        """Calculate disclosure risk (0-1, lower is better)."""
        if len(df) == 0:
            return 0.0
        
        # Risk factors
        risk_factors = []
        
        # Size factor (smaller datasets have higher risk)
        size_risk = max(0, 1 - (len(df) / 10000))
        risk_factors.append(size_risk)
        
        # Uniqueness factor
        if len(df.columns) > 0:
            # Estimate uniqueness by checking if all rows are unique
            try:
                unique_rows = df.unique().height
                uniqueness_risk = unique_rows / len(df)
                risk_factors.append(uniqueness_risk)
            except:
                risk_factors.append(0.5)  # Default moderate risk
        
        # Identifier presence
        identifier_count = self._count_identifiers(df)
        identifier_risk = min(identifier_count / 10, 1.0)  # Normalize to 0-1
        risk_factors.append(identifier_risk)
        
        return sum(risk_factors) / len(risk_factors)
    
    def _test_k_anonymity(self, df: pl.DataFrame) -> bool:
        """Test k-anonymity compliance (simplified)."""
        # Use common quasi-identifier column patterns
        qi_patterns = ["age", "gender", "postcode", "sa2", "occupation"]
        qi_columns = [col for col in df.columns if any(pattern in col.lower() for pattern in qi_patterns)]
        
        if not qi_columns:
            return True  # No quasi-identifiers found
        
        result = self.validate_k_anonymity(df, qi_columns[:3], k=self.sdc_thresholds["k_anonymity_k"])
        return result.get("compliant", False)
    
    def _test_l_diversity(self, df: pl.DataFrame) -> bool:
        """Test l-diversity compliance (simplified)."""
        # Find quasi-identifiers and sensitive attributes
        qi_patterns = ["age", "gender", "postcode"]
        sensitive_patterns = ["diagnosis", "condition", "medication"]
        
        qi_columns = [col for col in df.columns if any(pattern in col.lower() for pattern in qi_patterns)]
        sensitive_columns = [col for col in df.columns if any(pattern in col.lower() for pattern in sensitive_patterns)]
        
        if not qi_columns or not sensitive_columns:
            return True  # Required columns not found
        
        result = self.validate_l_diversity(df, qi_columns[:2], sensitive_columns[0], l=self.sdc_thresholds["l_diversity_l"])
        return result.get("compliant", False)
    
    def _test_statistical_disclosure_control(self, df: pl.DataFrame) -> bool:
        """Test statistical disclosure control compliance."""
        # Check for low cell counts
        count_columns = [col for col in df.columns if "count" in col.lower() or "total" in col.lower()]
        
        for count_col in count_columns:
            if count_col in df.columns and df[count_col].dtype.is_numeric():
                low_counts = df.filter(pl.col(count_col) < self.sdc_thresholds["minimum_cell_count"])
                if len(low_counts) > 0:
                    return False
        
        return True
    
    def _count_identifiers(self, df: pl.DataFrame) -> int:
        """Count potential identifiers in dataset."""
        identifier_count = 0
        
        for column in df.columns:
            column_lower = column.lower()
            
            # Check for direct identifiers
            if any(identifier in column_lower for identifier in self.direct_identifiers):
                identifier_count += 3  # High weight for direct identifiers
            
            # Check for quasi-identifiers
            elif any(qi in column_lower for qi in self.quasi_identifiers):
                identifier_count += 1
        
        return identifier_count
    
    def _generate_deidentification_recommendations(self, effectiveness: float, utility: float, 
                                                 disclosure_risk: float, k_anon: bool, 
                                                 l_div: bool, sdc: bool) -> List[str]:
        """Generate de-identification recommendations."""
        recommendations = []
        
        if effectiveness < 0.8:
            recommendations.append("Improve de-identification by removing more direct identifiers")
        
        if utility < 0.6:
            recommendations.append("Balance privacy and utility - consider less aggressive de-identification")
        
        if disclosure_risk > 0.3:
            recommendations.append("High disclosure risk - apply additional privacy protection measures")
        
        if not k_anon:
            recommendations.append("Improve k-anonymity by generalising quasi-identifiers")
        
        if not l_div:
            recommendations.append("Enhance l-diversity by increasing sensitive attribute variety")
        
        if not sdc:
            recommendations.append("Apply statistical disclosure control measures for small counts")
        
        if not recommendations:
            recommendations.append("De-identification appears adequate - continue monitoring")
        
        return recommendations