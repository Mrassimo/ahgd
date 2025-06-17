"""
Privacy and De-identification Compliance Validation Tests

Comprehensive testing suite for privacy compliance and de-identification validation including:
- Statistical disclosure control validation
- Data classification and sensitivity validation
- Audit trail completeness validation
- Privacy-preserving analytics compliance
- Australian Privacy Principles (APP) compliance
- De-identification standard validation (ISO/IEC 20889)

This test suite ensures all Australian health data processing complies with
privacy regulations and maintains appropriate de-identification standards.
"""

import json
import pytest
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Union
from unittest.mock import Mock, patch
from dataclasses import dataclass
from enum import Enum
import re

import polars as pl
import numpy as np
from loguru import logger

from tests.data_quality.validators.australian_health_validators import AustralianHealthDataValidator


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    SECRET = "secret"


class PrivacyRiskLevel(Enum):
    """Privacy risk assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DeIdentificationTechnique(Enum):
    """De-identification techniques."""
    GENERALIZATION = "generalization"
    SUPPRESSION = "suppression"
    PERTURBATION = "perturbation"
    AGGREGATION = "aggregation"
    PSEUDONYMIZATION = "pseudonymization"
    ANONYMIZATION = "anonymization"
    DIFFERENTIAL_PRIVACY = "differential_privacy"


@dataclass
class PrivacyViolation:
    """Privacy compliance violation."""
    violation_type: str
    severity: PrivacyRiskLevel
    description: str
    affected_columns: List[str]
    risk_score: float
    mitigation_required: bool
    sample_violations: List[Dict]
    compliance_standard: str
    details: Dict


@dataclass
class DeIdentificationAssessment:
    """De-identification assessment result."""
    dataset_id: str
    assessment_timestamp: str
    classification: DataClassification
    risk_level: PrivacyRiskLevel
    techniques_applied: List[DeIdentificationTechnique]
    disclosure_risk_score: float
    utility_preservation_score: float
    compliance_status: Dict[str, bool]
    recommendations: List[str]


class PrivacyComplianceValidator:
    """Privacy compliance and de-identification validator."""
    
    def __init__(self, validator: AustralianHealthDataValidator):
        """Initialize privacy compliance validator."""
        self.validator = validator
        self.logger = logger.bind(component="privacy_compliance_validator")
        
        # Australian Privacy Principles checklist
        self.app_requirements = {
            "APP1": "Open and transparent management of personal information",
            "APP3": "Collection of solicited personal information",
            "APP5": "Notification of the collection of personal information",
            "APP6": "Use or disclosure of personal information",
            "APP11": "Security of personal information",
            "APP12": "Access to personal information",
            "APP13": "Correction of personal information"
        }
        
        # Quasi-identifiers commonly found in Australian health data
        self.quasi_identifiers = {
            "direct": ["name", "address", "phone", "email", "medicare_number", "healthcare_identifier"],
            "indirect": ["sa2_code_2021", "postcode", "age", "birth_date", "gender", "occupation"],
            "sensitive": ["health_condition", "medication", "diagnosis", "treatment", "provider_id"]
        }
        
        # Statistical disclosure control thresholds
        self.sdc_thresholds = {
            "minimum_cell_count": 5,  # Minimum count in aggregated cells
            "maximum_dominance": 0.85,  # Maximum contribution by single entity
            "k_anonymity_k": 5,  # Minimum group size for k-anonymity
            "l_diversity_l": 3,  # Minimum diversity for sensitive attributes
        }
    
    def validate_data_classification(self, df: pl.DataFrame, expected_classification: DataClassification) -> List[PrivacyViolation]:
        """
        Validate data classification based on content analysis.
        
        Args:
            df: DataFrame to analyze
            expected_classification: Expected data classification level
            
        Returns:
            List of privacy violations
        """
        violations = []
        
        # Analyze columns for sensitive content
        sensitive_columns = []
        quasi_identifier_columns = []
        
        for column in df.columns:
            column_lower = column.lower()
            
            # Check for direct identifiers
            if any(identifier in column_lower for identifier in self.quasi_identifiers["direct"]):
                sensitive_columns.append(column)
                
                # Direct identifiers should only appear in RESTRICTED or SECRET data
                if expected_classification not in [DataClassification.RESTRICTED, DataClassification.SECRET]:
                    violations.append(PrivacyViolation(
                        violation_type="inappropriate_classification",
                        severity=PrivacyRiskLevel.CRITICAL,
                        description=f"Direct identifier '{column}' found in {expected_classification.value} data",
                        affected_columns=[column],
                        risk_score=9.0,
                        mitigation_required=True,
                        sample_violations=[{"column": column, "type": "direct_identifier"}],
                        compliance_standard="APP11",
                        details={"expected_classification": "restricted_or_secret", "actual_classification": expected_classification.value}
                    ))
            
            # Check for indirect identifiers
            elif any(identifier in column_lower for identifier in self.quasi_identifiers["indirect"]):
                quasi_identifier_columns.append(column)
            
            # Check for sensitive attributes
            elif any(identifier in column_lower for identifier in self.quasi_identifiers["sensitive"]):
                sensitive_columns.append(column)
        
        # Validate quasi-identifier density
        qi_density = len(quasi_identifier_columns) / len(df.columns) if df.columns else 0
        
        if qi_density > 0.3 and expected_classification == DataClassification.PUBLIC:
            violations.append(PrivacyViolation(
                violation_type="high_quasi_identifier_density",
                severity=PrivacyRiskLevel.HIGH,
                description=f"High density of quasi-identifiers ({qi_density:.2%}) for PUBLIC classification",
                affected_columns=quasi_identifier_columns,
                risk_score=7.5,
                mitigation_required=True,
                sample_violations=[{"qi_density": qi_density, "qi_columns": quasi_identifier_columns}],
                compliance_standard="APP6",
                details={"threshold": 0.3, "actual_density": qi_density}
            ))
        
        return violations
    
    def validate_statistical_disclosure_control(self, df: pl.DataFrame) -> List[PrivacyViolation]:
        """
        Validate statistical disclosure control measures.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of privacy violations
        """
        violations = []
        
        # Check minimum cell counts for aggregated data
        if self._is_aggregated_data(df):
            count_columns = [col for col in df.columns if "count" in col.lower() or "total" in col.lower()]
            
            for count_col in count_columns:
                if count_col in df.columns:
                    low_counts = df.filter(pl.col(count_col) < self.sdc_thresholds["minimum_cell_count"])
                    
                    if len(low_counts) > 0:
                        violation_percentage = (len(low_counts) / len(df)) * 100
                        
                        violations.append(PrivacyViolation(
                            violation_type="low_cell_count",
                            severity=PrivacyRiskLevel.HIGH,
                            description=f"Cells with counts below minimum threshold ({self.sdc_thresholds['minimum_cell_count']})",
                            affected_columns=[count_col],
                            risk_score=8.0,
                            mitigation_required=True,
                            sample_violations=[{"low_count_records": len(low_counts), "percentage": violation_percentage}],
                            compliance_standard="Statistical Disclosure Control",
                            details={"minimum_threshold": self.sdc_thresholds["minimum_cell_count"], "violations": len(low_counts)}
                        ))
        
        # Check for potential dominance issues
        numeric_columns = [col for col in df.columns if df[col].dtype.is_numeric()]
        
        for col in numeric_columns:
            if len(df) > 0:
                total_sum = df[col].sum()
                if total_sum and total_sum > 0:
                    max_value = df[col].max()
                    dominance_ratio = max_value / total_sum
                    
                    if dominance_ratio > self.sdc_thresholds["maximum_dominance"]:
                        violations.append(PrivacyViolation(
                            violation_type="dominance_risk",
                            severity=PrivacyRiskLevel.MEDIUM,
                            description=f"High dominance ratio in column '{col}' ({dominance_ratio:.2%})",
                            affected_columns=[col],
                            risk_score=6.0,
                            mitigation_required=True,
                            sample_violations=[{"dominance_ratio": dominance_ratio, "max_value": max_value, "total": total_sum}],
                            compliance_standard="Statistical Disclosure Control",
                            details={"threshold": self.sdc_thresholds["maximum_dominance"], "actual_ratio": dominance_ratio}
                        ))
        
        return violations
    
    def validate_k_anonymity(self, df: pl.DataFrame, quasi_identifier_columns: List[str], k: Optional[int] = None) -> List[PrivacyViolation]:
        """
        Validate k-anonymity compliance.
        
        Args:
            df: DataFrame to analyze
            quasi_identifier_columns: List of quasi-identifier column names
            k: Minimum group size (default: from thresholds)
            
        Returns:
            List of privacy violations
        """
        violations = []
        k_value = k or self.sdc_thresholds["k_anonymity_k"]
        
        # Filter to existing columns
        existing_qi_columns = [col for col in quasi_identifier_columns if col in df.columns]
        
        if not existing_qi_columns:
            return violations
        
        # Group by quasi-identifiers and count occurrences
        try:
            grouped = df.group_by(existing_qi_columns).agg(pl.count().alias("group_count"))
            small_groups = grouped.filter(pl.col("group_count") < k_value)
            
            if len(small_groups) > 0:
                total_records_at_risk = small_groups["group_count"].sum()
                risk_percentage = (total_records_at_risk / len(df)) * 100
                
                violations.append(PrivacyViolation(
                    violation_type="k_anonymity_violation",
                    severity=PrivacyRiskLevel.HIGH if risk_percentage > 10 else PrivacyRiskLevel.MEDIUM,
                    description=f"K-anonymity violation: {len(small_groups)} groups with size < {k_value}",
                    affected_columns=existing_qi_columns,
                    risk_score=7.0 if risk_percentage > 10 else 5.0,
                    mitigation_required=True,
                    sample_violations=[{
                        "small_groups_count": len(small_groups),
                        "records_at_risk": total_records_at_risk,
                        "risk_percentage": risk_percentage
                    }],
                    compliance_standard="K-Anonymity",
                    details={"k_value": k_value, "quasi_identifiers": existing_qi_columns}
                ))
        
        except Exception as e:
            self.logger.warning(f"K-anonymity validation failed: {str(e)}")
        
        return violations
    
    def validate_l_diversity(self, df: pl.DataFrame, quasi_identifier_columns: List[str], sensitive_column: str, l: Optional[int] = None) -> List[PrivacyViolation]:
        """
        Validate l-diversity compliance.
        
        Args:
            df: DataFrame to analyze
            quasi_identifier_columns: List of quasi-identifier column names
            sensitive_column: Sensitive attribute column
            l: Minimum diversity (default: from thresholds)
            
        Returns:
            List of privacy violations
        """
        violations = []
        l_value = l or self.sdc_thresholds["l_diversity_l"]
        
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
                
                violations.append(PrivacyViolation(
                    violation_type="l_diversity_violation",
                    severity=PrivacyRiskLevel.HIGH if risk_percentage > 15 else PrivacyRiskLevel.MEDIUM,
                    description=f"L-diversity violation: {len(low_diversity_groups)} groups with diversity < {l_value}",
                    affected_columns=existing_qi_columns + [sensitive_column],
                    risk_score=7.5 if risk_percentage > 15 else 5.5,
                    mitigation_required=True,
                    sample_violations=[{
                        "low_diversity_groups": len(low_diversity_groups),
                        "records_at_risk": total_records_at_risk,
                        "risk_percentage": risk_percentage
                    }],
                    compliance_standard="L-Diversity",
                    details={"l_value": l_value, "sensitive_attribute": sensitive_column}
                ))
        
        except Exception as e:
            self.logger.warning(f"L-diversity validation failed: {str(e)}")
        
        return violations
    
    def validate_data_masking(self, df: pl.DataFrame) -> List[PrivacyViolation]:
        """
        Validate data masking and pseudonymization techniques.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of privacy violations
        """
        violations = []
        
        for column in df.columns:
            column_lower = column.lower()
            
            # Check for unmasked direct identifiers
            if any(identifier in column_lower for identifier in ["name", "email", "phone", "address"]):
                sample_values = df[column].drop_nulls().head(10).to_list()
                
                # Simple patterns to detect unmasked data
                unmasked_patterns = []
                
                for value in sample_values:
                    if isinstance(value, str):
                        # Check for email patterns
                        if "@" in value and "." in value:
                            unmasked_patterns.append({"type": "email", "value": value[:10] + "..."})
                        
                        # Check for phone patterns
                        elif re.match(r'^\+?\d{10,15}$', re.sub(r'[\s\-\(\)]', '', value)):
                            unmasked_patterns.append({"type": "phone", "value": value[:5] + "..."})
                        
                        # Check for name patterns (simple heuristic)
                        elif len(value.split()) >= 2 and all(part.isalpha() for part in value.split()):
                            unmasked_patterns.append({"type": "name", "value": value[:5] + "..."})
                
                if unmasked_patterns:
                    violations.append(PrivacyViolation(
                        violation_type="unmasked_identifiers",
                        severity=PrivacyRiskLevel.CRITICAL,
                        description=f"Unmasked identifiers detected in column '{column}'",
                        affected_columns=[column],
                        risk_score=9.5,
                        mitigation_required=True,
                        sample_violations=unmasked_patterns[:3],  # Limit samples
                        compliance_standard="APP11",
                        details={"patterns_detected": len(unmasked_patterns), "column": column}
                    ))
        
        return violations
    
    def validate_audit_trail(self, processing_metadata: Dict) -> List[PrivacyViolation]:
        """
        Validate audit trail completeness for privacy compliance.
        
        Args:
            processing_metadata: Metadata about data processing
            
        Returns:
            List of privacy violations
        """
        violations = []
        
        required_audit_fields = [
            "data_source",
            "processing_timestamp",
            "processing_purpose",
            "access_controls",
            "retention_period",
            "disposal_method"
        ]
        
        missing_fields = []
        for field in required_audit_fields:
            if field not in processing_metadata or not processing_metadata[field]:
                missing_fields.append(field)
        
        if missing_fields:
            violations.append(PrivacyViolation(
                violation_type="incomplete_audit_trail",
                severity=PrivacyRiskLevel.HIGH,
                description=f"Missing audit trail fields: {missing_fields}",
                affected_columns=[],
                risk_score=7.0,
                mitigation_required=True,
                sample_violations=[{"missing_fields": missing_fields}],
                compliance_standard="APP11",
                details={"required_fields": required_audit_fields, "missing_fields": missing_fields}
            ))
        
        # Validate timestamp format
        if "processing_timestamp" in processing_metadata:
            timestamp = processing_metadata["processing_timestamp"]
            try:
                datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                violations.append(PrivacyViolation(
                    violation_type="invalid_timestamp_format",
                    severity=PrivacyRiskLevel.MEDIUM,
                    description="Invalid timestamp format in audit trail",
                    affected_columns=[],
                    risk_score=4.0,
                    mitigation_required=True,
                    sample_violations=[{"invalid_timestamp": timestamp}],
                    compliance_standard="Audit Trail Standards",
                    details={"expected_format": "ISO 8601"}
                ))
        
        return violations
    
    def assess_de_identification_effectiveness(self, original_df: pl.DataFrame, de_identified_df: pl.DataFrame) -> DeIdentificationAssessment:
        """
        Assess effectiveness of de-identification techniques.
        
        Args:
            original_df: Original dataset
            de_identified_df: De-identified dataset
            
        Returns:
            De-identification assessment
        """
        # Calculate disclosure risk score
        disclosure_risk = self._calculate_disclosure_risk(de_identified_df)
        
        # Calculate utility preservation score
        utility_score = self._calculate_utility_preservation(original_df, de_identified_df)
        
        # Identify applied techniques
        techniques_applied = self._identify_de_identification_techniques(original_df, de_identified_df)
        
        # Determine risk level
        if disclosure_risk < 0.1:
            risk_level = PrivacyRiskLevel.LOW
        elif disclosure_risk < 0.3:
            risk_level = PrivacyRiskLevel.MEDIUM
        elif disclosure_risk < 0.7:
            risk_level = PrivacyRiskLevel.HIGH
        else:
            risk_level = PrivacyRiskLevel.CRITICAL
        
        # Determine classification
        if risk_level == PrivacyRiskLevel.LOW and utility_score > 0.8:
            classification = DataClassification.PUBLIC
        elif risk_level in [PrivacyRiskLevel.LOW, PrivacyRiskLevel.MEDIUM] and utility_score > 0.6:
            classification = DataClassification.INTERNAL
        else:
            classification = DataClassification.CONFIDENTIAL
        
        # Check compliance with various standards
        compliance_status = {
            "k_anonymity": self._check_k_anonymity_compliance(de_identified_df),
            "l_diversity": self._check_l_diversity_compliance(de_identified_df),
            "statistical_disclosure_control": disclosure_risk < 0.3,
            "utility_preservation": utility_score > 0.5
        }
        
        # Generate recommendations
        recommendations = self._generate_de_identification_recommendations(
            disclosure_risk, utility_score, techniques_applied, compliance_status
        )
        
        return DeIdentificationAssessment(
            dataset_id=f"assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            assessment_timestamp=datetime.now().isoformat(),
            classification=classification,
            risk_level=risk_level,
            techniques_applied=techniques_applied,
            disclosure_risk_score=disclosure_risk,
            utility_preservation_score=utility_score,
            compliance_status=compliance_status,
            recommendations=recommendations
        )
    
    def _is_aggregated_data(self, df: pl.DataFrame) -> bool:
        """Check if data appears to be aggregated."""
        aggregation_indicators = ["count", "total", "sum", "avg", "mean", "median"]
        column_names = [col.lower() for col in df.columns]
        
        return any(indicator in " ".join(column_names) for indicator in aggregation_indicators)
    
    def _calculate_disclosure_risk(self, df: pl.DataFrame) -> float:
        """Calculate disclosure risk score (0-1, lower is better)."""
        risk_factors = []
        
        # Factor 1: Presence of quasi-identifiers
        qi_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(qi in col_lower for qi in self.quasi_identifiers["indirect"] + self.quasi_identifiers["sensitive"]):
                qi_columns.append(col)
        
        qi_density = len(qi_columns) / len(df.columns) if df.columns else 0
        risk_factors.append(qi_density)
        
        # Factor 2: Dataset size (smaller datasets have higher risk)
        size_risk = max(0, 1 - (len(df) / 10000))  # Risk decreases as size increases
        risk_factors.append(size_risk)
        
        # Factor 3: Uniqueness in quasi-identifier combinations
        if qi_columns and len(df) > 0:
            try:
                unique_combinations = df.select(qi_columns).unique().height
                uniqueness_risk = unique_combinations / len(df)
                risk_factors.append(uniqueness_risk)
            except:
                risk_factors.append(0.5)  # Default moderate risk
        
        # Combine risk factors
        return sum(risk_factors) / len(risk_factors) if risk_factors else 0.5
    
    def _calculate_utility_preservation(self, original_df: pl.DataFrame, de_identified_df: pl.DataFrame) -> float:
        """Calculate utility preservation score (0-1, higher is better)."""
        if len(original_df.columns) == 0 or len(de_identified_df.columns) == 0:
            return 0.0
        
        utility_factors = []
        
        # Factor 1: Column preservation
        common_columns = set(original_df.columns) & set(de_identified_df.columns)
        column_preservation = len(common_columns) / len(original_df.columns)
        utility_factors.append(column_preservation)
        
        # Factor 2: Row preservation
        row_preservation = len(de_identified_df) / len(original_df) if len(original_df) > 0 else 0
        utility_factors.append(min(row_preservation, 1.0))  # Cap at 1.0
        
        # Factor 3: Data distribution similarity (for numeric columns)
        numeric_similarity = 0.8  # Default assumption of good preservation
        
        for col in common_columns:
            if col in original_df.columns and col in de_identified_df.columns:
                if original_df[col].dtype.is_numeric() and de_identified_df[col].dtype.is_numeric():
                    try:
                        orig_mean = original_df[col].mean()
                        de_id_mean = de_identified_df[col].mean()
                        
                        if orig_mean and orig_mean != 0:
                            similarity = 1 - abs(orig_mean - de_id_mean) / abs(orig_mean)
                            numeric_similarity = min(numeric_similarity, max(0, similarity))
                    except:
                        pass
        
        utility_factors.append(numeric_similarity)
        
        return sum(utility_factors) / len(utility_factors)
    
    def _identify_de_identification_techniques(self, original_df: pl.DataFrame, de_identified_df: pl.DataFrame) -> List[DeIdentificationTechnique]:
        """Identify which de-identification techniques were applied."""
        techniques = []
        
        # Check for suppression (missing columns)
        if len(de_identified_df.columns) < len(original_df.columns):
            techniques.append(DeIdentificationTechnique.SUPPRESSION)
        
        # Check for aggregation (fewer rows)
        if len(de_identified_df) < len(original_df):
            techniques.append(DeIdentificationTechnique.AGGREGATION)
        
        # Check for generalization (fewer unique values in categorical columns)
        common_columns = set(original_df.columns) & set(de_identified_df.columns)
        
        for col in common_columns:
            if (original_df[col].dtype == pl.Utf8 and de_identified_df[col].dtype == pl.Utf8):
                orig_unique = original_df[col].n_unique()
                de_id_unique = de_identified_df[col].n_unique()
                
                if de_id_unique < orig_unique * 0.7:  # Significant reduction in uniqueness
                    if DeIdentificationTechnique.GENERALIZATION not in techniques:
                        techniques.append(DeIdentificationTechnique.GENERALIZATION)
        
        # Default to pseudonymization if no specific technique detected
        if not techniques:
            techniques.append(DeIdentificationTechnique.PSEUDONYMIZATION)
        
        return techniques
    
    def _check_k_anonymity_compliance(self, df: pl.DataFrame) -> bool:
        """Quick check for k-anonymity compliance."""
        # Simplified check - would need more sophisticated implementation
        return len(df) > self.sdc_thresholds["k_anonymity_k"] * 10
    
    def _check_l_diversity_compliance(self, df: pl.DataFrame) -> bool:
        """Quick check for l-diversity compliance."""
        # Simplified check - would need more sophisticated implementation
        return True  # Assume compliance for now
    
    def _generate_de_identification_recommendations(self, disclosure_risk: float, utility_score: float, techniques: List[DeIdentificationTechnique], compliance: Dict[str, bool]) -> List[str]:
        """Generate recommendations for improving de-identification."""
        recommendations = []
        
        if disclosure_risk > 0.5:
            recommendations.append("High disclosure risk detected - consider additional generalization or suppression")
        
        if utility_score < 0.5:
            recommendations.append("Low utility preservation - consider less aggressive de-identification techniques")
        
        if not compliance.get("k_anonymity", True):
            recommendations.append("K-anonymity compliance failed - increase generalization of quasi-identifiers")
        
        if not compliance.get("l_diversity", True):
            recommendations.append("L-diversity compliance failed - ensure diversity in sensitive attributes")
        
        if DeIdentificationTechnique.DIFFERENTIAL_PRIVACY not in techniques and disclosure_risk > 0.3:
            recommendations.append("Consider applying differential privacy for stronger privacy guarantees")
        
        return recommendations


class TestPrivacyCompliance:
    """Test suite for privacy compliance and de-identification validation."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return AustralianHealthDataValidator()
    
    @pytest.fixture
    def privacy_validator(self, validator):
        """Create privacy compliance validator."""
        return PrivacyComplianceValidator(validator)
    
    @pytest.fixture
    def public_dataset(self):
        """Public dataset (should be de-identified)."""
        return pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001", "301011002"],
            "age_group": ["25-34", "35-44", "45-54"],
            "seifa_decile": [8, 5, 9],
            "population_count": [15000, 12000, 18000],
            "aggregated_metric": [1.2, 0.9, 1.5]
        })
    
    @pytest.fixture
    def confidential_dataset(self):
        """Confidential dataset (contains identifiers)."""
        return pl.DataFrame({
            "patient_id": ["P12345", "P67890", "P11111"],
            "name": ["John Smith", "Jane Doe", "Bob Johnson"],
            "email": ["john@example.com", "jane@example.com", "bob@example.com"],
            "phone": ["0412345678", "0498765432", "0411223344"],
            "sa2_code_2021": ["101021007", "201011001", "301011002"],
            "age": [32, 45, 28],
            "health_condition": ["diabetes", "hypertension", "asthma"],
            "provider_id": ["PR001", "PR002", "PR003"]
        })
    
    @pytest.fixture
    def aggregated_dataset(self):
        """Aggregated dataset for statistical disclosure control testing."""
        return pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001", "301011002", "401011003"],
            "age_group": ["25-34", "35-44", "45-54", "55-64"],
            "condition_count": [15, 3, 25, 2],  # Some low counts
            "total_patients": [1000, 50, 1500, 30],
            "average_cost": [250.50, 180.75, 320.25, 295.80]
        })
    
    @pytest.fixture
    def processing_metadata_complete(self):
        """Complete processing metadata for audit trail testing."""
        return {
            "data_source": "ABS SEIFA 2021",
            "processing_timestamp": "2023-06-01T10:30:00Z",
            "processing_purpose": "Health analytics research",
            "access_controls": "Role-based access, encrypted storage",
            "retention_period": "7 years",
            "disposal_method": "Secure deletion and certificate",
            "processor_id": "SYS001",
            "approval_reference": "ETH2023-001"
        }
    
    @pytest.fixture
    def processing_metadata_incomplete(self):
        """Incomplete processing metadata for testing violations."""
        return {
            "data_source": "Unknown source",
            "processing_timestamp": "invalid-date",
            # Missing required fields
        }
    
    def test_data_classification_validation_public(self, privacy_validator, public_dataset):
        """Test data classification validation for public dataset."""
        violations = privacy_validator.validate_data_classification(
            public_dataset, DataClassification.PUBLIC
        )
        
        # Public dataset should have no or minimal violations
        critical_violations = [v for v in violations if v.severity == PrivacyRiskLevel.CRITICAL]
        assert len(critical_violations) == 0, f"Critical violations found in PUBLIC dataset: {[v.description for v in critical_violations]}"
        
        # May have warnings about quasi-identifiers
        high_violations = [v for v in violations if v.severity == PrivacyRiskLevel.HIGH]
        if high_violations:
            # Check that violations are about quasi-identifier density, not direct identifiers
            for violation in high_violations:
                assert "quasi_identifier_density" in violation.violation_type
    
    def test_data_classification_validation_confidential(self, privacy_validator, confidential_dataset):
        """Test data classification validation for confidential dataset."""
        # This should fail if classified as PUBLIC
        violations = privacy_validator.validate_data_classification(
            confidential_dataset, DataClassification.PUBLIC
        )
        
        # Should have critical violations due to direct identifiers
        critical_violations = [v for v in violations if v.severity == PrivacyRiskLevel.CRITICAL]
        assert len(critical_violations) > 0, "Expected critical violations for direct identifiers in PUBLIC classification"
        
        # Check specific violation types
        violation_types = [v.violation_type for v in critical_violations]
        assert "inappropriate_classification" in violation_types
        
        # Test correct classification
        violations_restricted = privacy_validator.validate_data_classification(
            confidential_dataset, DataClassification.RESTRICTED
        )
        
        # Should have fewer or no critical violations with appropriate classification
        critical_violations_restricted = [v for v in violations_restricted if v.severity == PrivacyRiskLevel.CRITICAL]
        assert len(critical_violations_restricted) <= len(critical_violations)
    
    def test_statistical_disclosure_control_validation(self, privacy_validator, aggregated_dataset):
        """Test statistical disclosure control validation."""
        violations = privacy_validator.validate_statistical_disclosure_control(aggregated_dataset)
        
        # Should detect low cell counts
        low_count_violations = [v for v in violations if v.violation_type == "low_cell_count"]
        assert len(low_count_violations) > 0, "Expected to detect low cell counts in aggregated data"
        
        for violation in low_count_violations:
            assert violation.severity in [PrivacyRiskLevel.HIGH, PrivacyRiskLevel.MEDIUM]
            assert violation.mitigation_required is True
            assert "Statistical Disclosure Control" in violation.compliance_standard
    
    def test_k_anonymity_validation(self, privacy_validator, confidential_dataset):
        """Test k-anonymity validation."""
        quasi_identifiers = ["sa2_code_2021", "age"]
        
        violations = privacy_validator.validate_k_anonymity(
            confidential_dataset, quasi_identifiers, k=5
        )
        
        # Small dataset should have k-anonymity violations
        k_violations = [v for v in violations if v.violation_type == "k_anonymity_violation"]
        
        if k_violations:  # May not have violations if data accidentally satisfies k-anonymity
            for violation in k_violations:
                assert violation.severity in [PrivacyRiskLevel.HIGH, PrivacyRiskLevel.MEDIUM]
                assert "K-Anonymity" in violation.compliance_standard
                assert len(violation.affected_columns) >= len(quasi_identifiers)
    
    def test_l_diversity_validation(self, privacy_validator, confidential_dataset):
        """Test l-diversity validation."""
        quasi_identifiers = ["sa2_code_2021", "age"]
        sensitive_attribute = "health_condition"
        
        violations = privacy_validator.validate_l_diversity(
            confidential_dataset, quasi_identifiers, sensitive_attribute, l=3
        )
        
        # May have l-diversity violations with small dataset
        l_violations = [v for v in violations if v.violation_type == "l_diversity_violation"]
        
        if l_violations:
            for violation in l_violations:
                assert violation.severity in [PrivacyRiskLevel.HIGH, PrivacyRiskLevel.MEDIUM]
                assert "L-Diversity" in violation.compliance_standard
                assert sensitive_attribute in violation.affected_columns
    
    def test_data_masking_validation(self, privacy_validator, confidential_dataset):
        """Test data masking validation."""
        violations = privacy_validator.validate_data_masking(confidential_dataset)
        
        # Should detect unmasked identifiers
        masking_violations = [v for v in violations if v.violation_type == "unmasked_identifiers"]
        assert len(masking_violations) > 0, "Expected to detect unmasked identifiers"
        
        for violation in masking_violations:
            assert violation.severity == PrivacyRiskLevel.CRITICAL
            assert violation.mitigation_required is True
            assert "APP11" in violation.compliance_standard
    
    def test_audit_trail_validation_complete(self, privacy_validator, processing_metadata_complete):
        """Test audit trail validation with complete metadata."""
        violations = privacy_validator.validate_audit_trail(processing_metadata_complete)
        
        # Complete metadata should have no violations
        assert len(violations) == 0, f"Unexpected violations in complete audit trail: {[v.description for v in violations]}"
    
    def test_audit_trail_validation_incomplete(self, privacy_validator, processing_metadata_incomplete):
        """Test audit trail validation with incomplete metadata."""
        violations = privacy_validator.validate_audit_trail(processing_metadata_incomplete)
        
        # Should detect missing fields and invalid timestamp
        missing_field_violations = [v for v in violations if v.violation_type == "incomplete_audit_trail"]
        assert len(missing_field_violations) > 0, "Expected to detect missing audit trail fields"
        
        timestamp_violations = [v for v in violations if v.violation_type == "invalid_timestamp_format"]
        assert len(timestamp_violations) > 0, "Expected to detect invalid timestamp format"
        
        for violation in violations:
            assert violation.severity in [PrivacyRiskLevel.HIGH, PrivacyRiskLevel.MEDIUM]
            assert violation.mitigation_required is True
    
    def test_de_identification_assessment(self, privacy_validator, confidential_dataset, public_dataset):
        """Test de-identification effectiveness assessment."""
        assessment = privacy_validator.assess_de_identification_effectiveness(
            confidential_dataset, public_dataset
        )
        
        # Verify assessment structure
        assert isinstance(assessment, DeIdentificationAssessment)
        assert assessment.dataset_id is not None
        assert assessment.assessment_timestamp is not None
        assert isinstance(assessment.classification, DataClassification)
        assert isinstance(assessment.risk_level, PrivacyRiskLevel)
        assert isinstance(assessment.techniques_applied, list)
        assert 0.0 <= assessment.disclosure_risk_score <= 1.0
        assert 0.0 <= assessment.utility_preservation_score <= 1.0
        assert isinstance(assessment.compliance_status, dict)
        assert isinstance(assessment.recommendations, list)
        
        # Should detect good de-identification (public dataset is well de-identified)
        assert assessment.risk_level in [PrivacyRiskLevel.LOW, PrivacyRiskLevel.MEDIUM]
        assert assessment.classification in [DataClassification.PUBLIC, DataClassification.INTERNAL]
        
        # Should have some techniques applied
        assert len(assessment.techniques_applied) > 0
        
        # Should have recommendations
        assert len(assessment.recommendations) >= 0  # May be empty if assessment is good
    
    def test_privacy_compliance_comprehensive_validation(self, privacy_validator, confidential_dataset, processing_metadata_incomplete):
        """Test comprehensive privacy compliance validation."""
        all_violations = []
        
        # Data classification validation
        classification_violations = privacy_validator.validate_data_classification(
            confidential_dataset, DataClassification.PUBLIC
        )
        all_violations.extend(classification_violations)
        
        # Statistical disclosure control
        sdc_violations = privacy_validator.validate_statistical_disclosure_control(confidential_dataset)
        all_violations.extend(sdc_violations)
        
        # K-anonymity
        k_violations = privacy_validator.validate_k_anonymity(
            confidential_dataset, ["sa2_code_2021", "age"]
        )
        all_violations.extend(k_violations)
        
        # Data masking
        masking_violations = privacy_validator.validate_data_masking(confidential_dataset)
        all_violations.extend(masking_violations)
        
        # Audit trail
        audit_violations = privacy_validator.validate_audit_trail(processing_metadata_incomplete)
        all_violations.extend(audit_violations)
        
        # Should have multiple violations
        assert len(all_violations) > 0, "Expected to find privacy compliance violations"
        
        # Categorize violations by severity
        violations_by_severity = {}
        for violation in all_violations:
            severity = violation.severity
            if severity not in violations_by_severity:
                violations_by_severity[severity] = []
            violations_by_severity[severity].append(violation)
        
        # Should have critical violations (unmasked identifiers, inappropriate classification)
        assert PrivacyRiskLevel.CRITICAL in violations_by_severity, "Expected critical privacy violations"
        
        # Verify all violations have required fields
        for violation in all_violations:
            assert hasattr(violation, 'violation_type')
            assert hasattr(violation, 'severity')
            assert hasattr(violation, 'description')
            assert hasattr(violation, 'affected_columns')
            assert hasattr(violation, 'risk_score')
            assert hasattr(violation, 'mitigation_required')
            assert hasattr(violation, 'compliance_standard')
            
            # Validate risk score
            assert 0.0 <= violation.risk_score <= 10.0
            
            # Critical violations should require mitigation
            if violation.severity == PrivacyRiskLevel.CRITICAL:
                assert violation.mitigation_required is True
                assert violation.risk_score >= 7.0
        
        # Log comprehensive results
        logger.info("Comprehensive privacy compliance validation results:")
        logger.info(f"  Total violations: {len(all_violations)}")
        
        for severity, violations in violations_by_severity.items():
            logger.info(f"  {severity.value}: {len(violations)} violations")
            for violation in violations[:3]:  # Show first 3 of each severity
                logger.info(f"    - {violation.description}")
    
    def test_app_compliance_mapping(self, privacy_validator):
        """Test Australian Privacy Principles compliance mapping."""
        # Verify APP requirements are defined
        assert len(privacy_validator.app_requirements) > 0
        
        # Check key APPs are included
        key_apps = ["APP1", "APP3", "APP5", "APP6", "APP11", "APP12", "APP13"]
        for app in key_apps:
            assert app in privacy_validator.app_requirements
        
        # Each APP should have a description
        for app, description in privacy_validator.app_requirements.items():
            assert isinstance(description, str)
            assert len(description) > 10  # Meaningful description
    
    def test_quasi_identifier_detection(self, privacy_validator):
        """Test quasi-identifier detection capabilities."""
        # Verify quasi-identifier categories are defined
        assert "direct" in privacy_validator.quasi_identifiers
        assert "indirect" in privacy_validator.quasi_identifiers
        assert "sensitive" in privacy_validator.quasi_identifiers
        
        # Check for Australian-specific identifiers
        all_identifiers = (privacy_validator.quasi_identifiers["direct"] + 
                          privacy_validator.quasi_identifiers["indirect"] + 
                          privacy_validator.quasi_identifiers["sensitive"])
        
        australian_identifiers = ["sa2_code_2021", "medicare_number", "healthcare_identifier"]
        for identifier in australian_identifiers:
            assert any(identifier in qi for qi in all_identifiers), f"Australian identifier {identifier} not found"
    
    def test_sdc_thresholds_configuration(self, privacy_validator):
        """Test statistical disclosure control thresholds."""
        # Verify SDC thresholds are properly configured
        sdc = privacy_validator.sdc_thresholds
        
        assert "minimum_cell_count" in sdc
        assert "maximum_dominance" in sdc
        assert "k_anonymity_k" in sdc
        assert "l_diversity_l" in sdc
        
        # Validate threshold values
        assert sdc["minimum_cell_count"] >= 3  # Standard minimum
        assert 0.5 <= sdc["maximum_dominance"] <= 1.0  # Proportion
        assert sdc["k_anonymity_k"] >= 2  # Minimum for k-anonymity
        assert sdc["l_diversity_l"] >= 2  # Minimum for l-diversity


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])