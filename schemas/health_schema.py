"""
Health indicator data schemas for AHGD.

This module defines schemas for various health-related data including
disease prevalence, mortality rates, healthcare utilisation, and health outcomes.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import Field, field_validator, model_validator

from .base_schema import (
    VersionedSchema,
    TemporalData,
    DataSource,
    SchemaVersion,
    DataQualityLevel
)


class HealthIndicatorType(str, Enum):
    """Types of health indicators."""
    PREVALENCE = "prevalence"
    INCIDENCE = "incidence"
    MORTALITY = "mortality"
    MORBIDITY = "morbidity"
    UTILISATION = "utilisation"
    SCREENING = "screening"
    IMMUNISATION = "immunisation"
    RISK_FACTOR = "risk_factor"
    HEALTH_SERVICE = "health_service"
    MENTAL_HEALTH = "mental_health"
    DISABILITY = "disability"


class AgeGroupType(str, Enum):
    """Standard age group classifications."""
    ALL_AGES = "all_ages"
    AGE_0_4 = "0-4"
    AGE_5_9 = "5-9"
    AGE_10_14 = "10-14"
    AGE_15_19 = "15-19"
    AGE_20_24 = "20-24"
    AGE_25_34 = "25-34"
    AGE_35_44 = "35-44"
    AGE_45_54 = "45-54"
    AGE_55_64 = "55-64"
    AGE_65_74 = "65-74"
    AGE_75_84 = "75-84"
    AGE_85_PLUS = "85+"
    WORKING_AGE = "15-64"
    YOUTH = "15-24"
    ELDERLY = "65+"


class HealthIndicator(VersionedSchema, TemporalData):
    """Base schema for health indicator data."""
    
    # Geographic identifier
    geographic_id: str = Field(..., description="Geographic area identifier (SA2, SA3, etc)")
    geographic_level: str = Field(..., description="Geographic level (SA2, SA3, SA4, STATE)")
    
    # Indicator details
    indicator_name: str = Field(..., min_length=1, description="Name of health indicator")
    indicator_code: str = Field(..., description="Unique indicator code")
    indicator_type: HealthIndicatorType = Field(..., description="Type of health indicator")
    
    # Measurement
    value: float = Field(..., description="Indicator value")
    unit: str = Field(..., description="Unit of measurement")
    
    # Statistical properties
    confidence_interval_lower: Optional[float] = Field(None, description="Lower CI bound")
    confidence_interval_upper: Optional[float] = Field(None, description="Upper CI bound")
    standard_error: Optional[float] = Field(None, ge=0, description="Standard error")
    sample_size: Optional[int] = Field(None, ge=0, description="Sample size if applicable")
    
    # Demographics
    age_group: AgeGroupType = Field(
        default=AgeGroupType.ALL_AGES,
        description="Age group for this indicator"
    )
    sex: Optional[str] = Field(None, description="Sex (Male/Female/Persons)")
    
    # Data quality
    suppressed: bool = Field(False, description="Whether value is suppressed for privacy")
    reliability: Optional[str] = Field(None, description="Statistical reliability rating")
    
    # Source
    data_source: DataSource = Field(..., description="Source of the health data")
    
    @field_validator('sex')
    @classmethod
    def validate_sex_category(cls, v: Optional[str]) -> Optional[str]:
        """Validate sex categories."""
        if v is not None:
            valid_sex = {'Male', 'Female', 'Persons', 'M', 'F', 'P'}
            if v not in valid_sex:
                raise ValueError(f"Invalid sex category: {v}")
        return v
    
    @model_validator(mode='after')
    def validate_value_and_ci_consistency(self) -> 'HealthIndicator':
        """Validate percentage bounds and CI order."""
        # Validate percentage bounds
        unit = self.unit.lower()
        if unit in ['%', 'percent', 'percentage']:
            if self.value < 0 or self.value > 100:
                raise ValueError(f"Percentage value must be between 0 and 100, got {self.value}")
        
        # Validate CI order
        if (self.confidence_interval_lower is not None and 
            self.confidence_interval_upper is not None):
            if self.confidence_interval_upper < self.confidence_interval_lower:
                raise ValueError("Upper CI must be greater than lower CI")
        
        # Validate suppression consistency
        if self.suppressed:
            # Suppressed values might be set to specific values like -999 or None
            if self.value is not None and self.value >= 0:
                # Log warning - suppressed but has real value
                pass
                
        return self
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "HealthIndicator"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate health indicator data integrity."""
        errors = []
        
        # Check CI consistency with value
        if self.confidence_interval_lower and self.confidence_interval_upper:
            if not (self.confidence_interval_lower <= self.value <= self.confidence_interval_upper):
                errors.append("Value should be within confidence interval")
                
        # Check rate bounds
        if self.unit.lower() in ['rate per 100,000', 'per 100,000']:
            if self.value < 0:
                errors.append("Rate cannot be negative")
            # Sanity check - rate per 100k shouldn't exceed 100k
            if self.value > 100000:
                errors.append("Rate per 100,000 exceeds 100,000")
                
        return errors


class MortalityData(HealthIndicator):
    """Schema for mortality-specific data."""
    
    # Mortality-specific fields
    cause_of_death: str = Field(..., description="ICD-10 cause of death category")
    icd10_code: Optional[str] = Field(None, description="Specific ICD-10 code")
    
    # Mortality metrics
    deaths_count: Optional[int] = Field(None, ge=0, description="Number of deaths")
    years_of_life_lost: Optional[float] = Field(None, ge=0, description="Years of life lost")
    age_standardised_rate: Optional[float] = Field(None, ge=0, description="Age-standardised rate")
    crude_rate: Optional[float] = Field(None, ge=0, description="Crude mortality rate")
    
    # Premature mortality
    is_premature: bool = Field(False, description="Whether classified as premature death")
    preventable: bool = Field(False, description="Whether death is preventable")
    
    @field_validator('icd10_code')
    @classmethod
    def validate_icd10_format(cls, v: Optional[str]) -> Optional[str]:
        """Basic ICD-10 code format validation."""
        if v is not None:
            # ICD-10 codes start with letter followed by numbers and optional decimal
            import re
            if not re.match(r'^[A-Z]\d{2}(\.\d{1,2})?$', v.upper()):
                raise ValueError(f"Invalid ICD-10 code format: {v}")
            return v.upper()
        return v
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "MortalityData"


class DiseasePrevalence(HealthIndicator):
    """Schema for disease prevalence data."""
    
    # Disease information
    disease_name: str = Field(..., description="Name of disease or condition")
    disease_category: str = Field(..., description="Disease category")
    icd10_codes: List[str] = Field(default_factory=list, description="Related ICD-10 codes")
    
    # Prevalence metrics
    prevalence_count: Optional[int] = Field(None, ge=0, description="Number of cases")
    prevalence_rate: float = Field(..., ge=0, le=100, description="Prevalence rate %")
    
    # Severity and impact
    severity_level: Optional[str] = Field(None, description="Severity classification")
    hospitalisations: Optional[int] = Field(None, ge=0, description="Related hospitalisations")
    
    # Comorbidities
    common_comorbidities: List[str] = Field(
        default_factory=list,
        description="Common co-occurring conditions"
    )
    
    @field_validator('severity_level')
    @classmethod
    def validate_severity(cls, v: Optional[str]) -> Optional[str]:
        """Validate severity level."""
        if v is not None:
            valid_levels = {'mild', 'moderate', 'severe', 'critical'}
            if v.lower() not in valid_levels:
                raise ValueError(f"Invalid severity level: {v}")
        return v
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "DiseasePrevalence"


class HealthcareUtilisation(HealthIndicator):
    """Schema for healthcare utilisation data."""
    
    # Service type
    service_type: str = Field(..., description="Type of healthcare service")
    service_category: str = Field(..., description="Category of service")
    
    # Utilisation metrics
    visits_count: Optional[int] = Field(None, ge=0, description="Number of visits/services")
    utilisation_rate: float = Field(..., ge=0, description="Utilisation rate")
    
    # Cost information
    total_cost: Optional[float] = Field(None, ge=0, description="Total cost if available")
    average_cost_per_service: Optional[float] = Field(None, ge=0, description="Average cost")
    bulk_billed_percentage: Optional[float] = Field(
        None, 
        ge=0, 
        le=100,
        description="Percentage bulk billed"
    )
    
    # Provider information
    provider_type: Optional[str] = Field(None, description="Type of healthcare provider")
    
    # Wait times and access
    average_wait_days: Optional[float] = Field(None, ge=0, description="Average wait time")
    
    @field_validator('service_category')
    @classmethod
    def validate_service_category(cls, v: str) -> str:
        """Validate healthcare service categories."""
        valid_categories = {
            'primary_care', 'specialist', 'emergency', 'hospital',
            'mental_health', 'allied_health', 'diagnostic', 'preventive',
            'pharmaceutical', 'telehealth'
        }
        if v.lower() not in valid_categories:
            # Allow but log warning
            pass
        return v
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "HealthcareUtilisation"


class RiskFactorData(HealthIndicator):
    """Schema for health risk factor data."""
    
    # Risk factor details
    risk_factor_name: str = Field(..., description="Name of risk factor")
    risk_category: str = Field(..., description="Category of risk factor")
    
    # Exposure metrics
    exposed_percentage: float = Field(..., ge=0, le=100, description="% of population exposed")
    high_risk_percentage: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="% at high risk"
    )
    
    # Impact metrics
    attributable_burden: Optional[float] = Field(
        None,
        ge=0,
        description="Disease burden attributable to risk factor"
    )
    relative_risk: Optional[float] = Field(None, gt=0, description="Relative risk ratio")
    
    # Modifiability
    modifiable: bool = Field(..., description="Whether risk factor is modifiable")
    intervention_available: bool = Field(False, description="Whether interventions exist")
    
    @field_validator('risk_category')
    @classmethod
    def validate_risk_category(cls, v: str) -> str:
        """Validate risk factor categories."""
        valid_categories = {
            'behavioural', 'biomedical', 'environmental', 
            'social', 'genetic', 'occupational'
        }
        if v.lower() not in valid_categories:
            # Allow custom categories but standardise known ones
            return v.lower() if v.lower() in valid_categories else v
        return v.lower()
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "RiskFactorData"


class MentalHealthIndicator(HealthIndicator):
    """Schema for mental health specific indicators."""
    
    # Mental health condition
    condition_name: str = Field(..., description="Mental health condition name")
    condition_category: str = Field(..., description="Category of mental health condition")
    
    # Severity and impact
    severity_distribution: Optional[Dict[str, float]] = Field(
        None,
        description="Distribution across severity levels"
    )
    functional_impact_score: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Functional impact score"
    )
    
    # Treatment
    treatment_rate: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="% receiving treatment"
    )
    medication_rate: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="% on medication"
    )
    therapy_rate: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="% receiving therapy"
    )
    
    # Outcomes
    recovery_rate: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Recovery rate %"
    )
    relapse_rate: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Relapse rate %"
    )
    
    @field_validator('condition_category')
    @classmethod
    def validate_condition_category(cls, v: str) -> str:
        """Validate mental health condition categories."""
        valid_categories = {
            'anxiety_disorders', 'mood_disorders', 'psychotic_disorders',
            'personality_disorders', 'substance_use_disorders',
            'neurodevelopmental_disorders', 'trauma_related'
        }
        if v.lower() not in valid_categories:
            # Allow custom categories
            pass
        return v
    
    @field_validator('severity_distribution')
    @classmethod
    def validate_severity_distribution(cls, v: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        """Validate severity distribution sums to 100%."""
        if v is not None:
            total = sum(v.values())
            if abs(total - 100.0) > 0.1:  # Allow small rounding errors
                raise ValueError(f"Severity distribution must sum to 100%, got {total}")
        return v
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "MentalHealthIndicator"


# Aggregated health data schema
class HealthDataAggregate(VersionedSchema):
    """Schema for aggregated health data across multiple indicators."""
    
    # Geographic and temporal identifiers
    geographic_id: str = Field(..., description="Geographic area identifier")
    geographic_level: str = Field(..., description="Geographic level")
    reporting_period: TemporalData = Field(..., description="Reporting period")
    
    # Aggregated indicators
    total_indicators: int = Field(..., ge=0, description="Total number of indicators")
    indicators_by_type: Dict[str, int] = Field(
        ...,
        description="Count of indicators by type"
    )
    
    # Summary statistics
    population_health_score: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Overall population health score"
    )
    
    # Top health issues
    top_mortality_causes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top causes of mortality"
    )
    top_morbidity_conditions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top morbidity conditions"
    )
    key_risk_factors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Key risk factors"
    )
    
    # Data completeness
    data_completeness_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Data completeness percentage"
    )
    missing_indicators: List[str] = Field(
        default_factory=list,
        description="List of missing key indicators"
    )
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "HealthDataAggregate"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate aggregated data integrity."""
        errors = []
        
        # Check indicator counts
        type_total = sum(self.indicators_by_type.values())
        if type_total != self.total_indicators:
            errors.append(f"Indicator type sum ({type_total}) doesn't match total ({self.total_indicators})")
            
        # Validate top lists
        for cause in self.top_mortality_causes:
            if 'rank' not in cause or 'cause' not in cause:
                errors.append("Top mortality causes must have rank and cause fields")
                
        return errors