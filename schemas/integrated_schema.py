"""
Integrated data schema for AHGD project target outputs.

This module defines the target schema structures that combine all data sources
into unified records suitable for final consumption and analysis.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import Field, field_validator, model_validator

from .base_schema import (
    VersionedSchema,
    GeographicBoundary,
    TemporalData,
    DataSource,
    SchemaVersion,
    DataQualityLevel
)
from .health_schema import HealthIndicatorType, AgeGroupType
from .seifa_schema import SEIFAIndexType


class DataIntegrationLevel(str, Enum):
    """Level of data integration completeness."""
    MINIMAL = "minimal"  # Basic geographic and demographic only
    STANDARD = "standard"  # Health and socioeconomic included  
    COMPREHENSIVE = "comprehensive"  # All available data sources
    ENHANCED = "enhanced"  # Includes derived indicators and analysis


class HealthOutcome(str, Enum):
    """Standardised health outcome categories."""
    EXCELLENT = "excellent"
    VERY_GOOD = "very_good"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class UrbanRuralClassification(str, Enum):
    """Urban/rural classification based on ABS standards."""
    MAJOR_URBAN = "major_urban"
    OTHER_URBAN = "other_urban"
    BOUNDED_LOCALITY = "bounded_locality"
    RURAL_BALANCE = "rural_balance"
    MIGRATORY_OFFSHORE = "migratory_offshore"


class MasterHealthRecord(VersionedSchema):
    """
    Master integrated health record combining all data sources.
    
    This is the primary target schema representing a complete health and
    demographic profile for a Statistical Area Level 2 (SA2).
    """
    
    # === PRIMARY IDENTIFICATION ===
    sa2_code: str = Field(
        ..., 
        pattern=r'^\d{9}$', 
        description="9-digit Statistical Area Level 2 code (primary key)"
    )
    sa2_name: str = Field(
        ..., 
        min_length=1, 
        max_length=100, 
        description="Official SA2 name"
    )
    
    # === GEOGRAPHIC DIMENSIONS ===
    geographic_hierarchy: Dict[str, str] = Field(
        ...,
        description="Complete geographic hierarchy (SA1s, SA3, SA4, State, Postcode)"
    )
    boundary_data: GeographicBoundary = Field(
        ...,
        description="Complete boundary geometry and metrics"
    )
    urbanisation: UrbanRuralClassification = Field(
        ...,
        description="Urban/rural classification"
    )
    remoteness_category: str = Field(
        ...,
        description="ABS Remoteness Area classification"
    )
    
    # === DEMOGRAPHIC PROFILE ===
    demographic_profile: Dict[str, Any] = Field(
        ...,
        description="Complete demographic breakdown"
    )
    total_population: int = Field(
        ..., 
        ge=0, 
        description="Total usual resident population"
    )
    population_density_per_sq_km: float = Field(
        ..., 
        ge=0, 
        description="Population density per square kilometre"
    )
    median_age: Optional[float] = Field(
        None, 
        ge=0, 
        le=120, 
        description="Median age of residents"
    )
    
    # === SOCIOECONOMIC INDICATORS ===
    seifa_scores: Dict[SEIFAIndexType, float] = Field(
        ...,
        description="All SEIFA index scores (IRSD, IRSAD, IER, IEO)"
    )
    seifa_deciles: Dict[SEIFAIndexType, int] = Field(
        ...,
        description="National deciles for all SEIFA indexes"
    )
    disadvantage_category: str = Field(
        ...,
        description="Overall disadvantage classification"
    )
    median_household_income: Optional[float] = Field(
        None, 
        ge=0, 
        description="Median weekly household income"
    )
    unemployment_rate: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Unemployment rate percentage"
    )
    
    # === HEALTH INDICATORS SUMMARY ===
    health_outcomes_summary: Dict[str, float] = Field(
        ...,
        description="Summary of key health outcome indicators"
    )
    life_expectancy: Optional[Dict[str, float]] = Field(
        None,
        description="Life expectancy by sex (years)"
    )
    self_assessed_health: Optional[Dict[HealthOutcome, float]] = Field(
        None,
        description="Distribution of self-assessed health outcomes (%)"
    )
    
    # Mortality indicators
    mortality_indicators: Dict[str, float] = Field(
        default_factory=dict,
        description="Age-standardised mortality rates by major causes"
    )
    avoidable_mortality_rate: Optional[float] = Field(
        None, 
        ge=0, 
        description="Avoidable mortality rate per 100,000"
    )
    infant_mortality_rate: Optional[float] = Field(
        None, 
        ge=0, 
        description="Infant mortality rate per 1,000 live births"
    )
    
    # Disease prevalence
    chronic_disease_prevalence: Dict[str, float] = Field(
        default_factory=dict,
        description="Prevalence rates for major chronic diseases (%)"
    )
    
    # Mental health
    mental_health_indicators: Dict[str, float] = Field(
        default_factory=dict,
        description="Mental health prevalence and service utilisation rates"
    )
    psychological_distress_high: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="High psychological distress prevalence (%)"
    )
    
    # === HEALTHCARE ACCESS AND UTILISATION ===
    healthcare_access: Dict[str, Any] = Field(
        default_factory=dict,
        description="Healthcare access and availability metrics"
    )
    gp_services_per_1000: Optional[float] = Field(
        None, 
        ge=0, 
        description="GP services per 1,000 population"
    )
    specialist_services_per_1000: Optional[float] = Field(
        None, 
        ge=0, 
        description="Specialist services per 1,000 population"
    )
    bulk_billing_rate: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Bulk billing rate percentage"
    )
    emergency_dept_presentations_per_1000: Optional[float] = Field(
        None, 
        ge=0, 
        description="Emergency department presentations per 1,000 population"
    )
    
    # === PHARMACEUTICAL UTILISATION ===
    pharmaceutical_utilisation: Dict[str, float] = Field(
        default_factory=dict,
        description="PBS pharmaceutical utilisation rates"
    )
    
    # === RISK FACTORS ===
    risk_factors: Dict[str, float] = Field(
        default_factory=dict,
        description="Prevalence of major modifiable risk factors (%)"
    )
    smoking_prevalence: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Current smoking prevalence (%)"
    )
    obesity_prevalence: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Obesity prevalence (%)"
    )
    physical_inactivity_prevalence: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Physical inactivity prevalence (%)"
    )
    harmful_alcohol_use_prevalence: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Harmful alcohol use prevalence (%)"
    )
    
    # === ENVIRONMENTAL FACTORS ===
    environmental_indicators: Dict[str, Any] = Field(
        default_factory=dict,
        description="Environmental health indicators"
    )
    air_quality_index: Optional[float] = Field(
        None, 
        ge=0, 
        description="Average air quality index"
    )
    green_space_access: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Percentage with access to green space"
    )
    
    # === DATA INTEGRATION METADATA ===
    integration_level: DataIntegrationLevel = Field(
        ...,
        description="Level of data integration achieved"
    )
    data_completeness_score: float = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Overall data completeness percentage"
    )
    integration_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this integrated record was created"
    )
    source_datasets: List[str] = Field(
        ...,
        description="List of source datasets included in integration"
    )
    missing_indicators: List[str] = Field(
        default_factory=list,
        description="List of indicators that could not be integrated"
    )
    
    # === DERIVED INDICATORS ===
    composite_health_index: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Composite health index score (0-100)"
    )
    health_inequality_index: Optional[float] = Field(
        None, 
        ge=0, 
        description="Health inequality index relative to national average"
    )
    
    @field_validator('geographic_hierarchy')
    @classmethod
    def validate_geographic_hierarchy(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate geographic hierarchy completeness."""
        required_levels = ['sa3_code', 'sa4_code', 'state_code']
        for level in required_levels:
            if level not in v:
                raise ValueError(f"Missing required geographic level: {level}")
        return v
    
    @field_validator('demographic_profile')
    @classmethod
    def validate_demographic_profile(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate demographic profile structure."""
        required_fields = ['age_groups', 'sex_distribution']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required demographic field: {field}")
        return v
    
    @model_validator(mode='after')
    def validate_health_record_consistency(self) -> 'MasterHealthRecord':
        """Validate overall record consistency."""
        # Validate population density calculation
        if self.boundary_data.area_sq_km and self.boundary_data.area_sq_km > 0:
            calculated_density = self.total_population / self.boundary_data.area_sq_km
            if abs(calculated_density - self.population_density_per_sq_km) > 1:
                raise ValueError("Population density calculation inconsistent")
        
        # Validate SEIFA scores and deciles alignment
        for index_type in self.seifa_scores:
            if index_type not in self.seifa_deciles:
                raise ValueError(f"Missing decile for SEIFA index {index_type}")
        
        # Validate health outcomes sum to 100% if present
        if self.self_assessed_health:
            total_pct = sum(self.self_assessed_health.values())
            if abs(total_pct - 100.0) > 0.1:
                raise ValueError(f"Self-assessed health percentages sum to {total_pct}, not 100%")
        
        return self
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "MasterHealthRecord"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate integrated health record data integrity."""
        errors = []
        
        # Check completeness score consistency
        total_possible_indicators = 50  # Define based on comprehensive list
        available_indicators = len([
            v for v in [
                self.life_expectancy, self.self_assessed_health,
                self.avoidable_mortality_rate, self.infant_mortality_rate,
                self.psychological_distress_high, self.gp_services_per_1000,
                self.bulk_billing_rate, self.smoking_prevalence,
                self.obesity_prevalence
            ] if v is not None
        ])
        
        calculated_completeness = (available_indicators / total_possible_indicators) * 100
        if abs(self.data_completeness_score - calculated_completeness) > 5:
            errors.append("Data completeness score inconsistent with available indicators")
        
        # Validate percentage bounds for risk factors
        percentage_fields = [
            self.smoking_prevalence, self.obesity_prevalence,
            self.physical_inactivity_prevalence, self.harmful_alcohol_use_prevalence,
            self.psychological_distress_high, self.bulk_billing_rate
        ]
        
        for i, field in enumerate(percentage_fields):
            if field is not None and (field < 0 or field > 100):
                errors.append(f"Percentage field {i} outside valid range: {field}")
        
        # Validate geographic consistency
        sa2_first_digit = int(self.sa2_code[0])
        state_mapping = {1: 'NSW', 2: 'VIC', 3: 'QLD', 4: 'SA', 5: 'WA', 6: 'TAS', 7: 'NT', 8: 'ACT'}
        expected_state = state_mapping.get(sa2_first_digit)
        if expected_state and self.geographic_hierarchy.get('state_code') != expected_state:
            errors.append(f"SA2 state digit inconsistent with state code")
        
        return errors


class SA2HealthProfile(VersionedSchema):
    """
    Complete SA2-level health and demographic profile.
    
    A focused view of health indicators and outcomes for a single SA2,
    suitable for health analysis and reporting.
    """
    
    # === IDENTIFICATION ===
    sa2_code: str = Field(..., pattern=r'^\d{9}$', description="SA2 code")
    sa2_name: str = Field(..., description="SA2 name")
    reference_period: TemporalData = Field(..., description="Data reference period")
    
    # === POPULATION CONTEXT ===
    total_population: int = Field(..., ge=0, description="Total population")
    population_by_age_sex: Dict[str, Dict[str, int]] = Field(
        ...,
        description="Population breakdown by age group and sex"
    )
    indigenous_population_percentage: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Aboriginal and Torres Strait Islander population %"
    )
    
    # === SOCIOECONOMIC CONTEXT ===
    seifa_disadvantage_decile: int = Field(
        ..., 
        ge=1, 
        le=10, 
        description="SEIFA disadvantage decile (1=most disadvantaged)"
    )
    socioeconomic_category: str = Field(
        ...,
        description="Socioeconomic classification (Most Disadvantaged to Least Disadvantaged)"
    )
    
    # === HEALTH OUTCOMES ===
    # Life expectancy
    life_expectancy_male: Optional[float] = Field(
        None, 
        ge=0, 
        le=120, 
        description="Male life expectancy (years)"
    )
    life_expectancy_female: Optional[float] = Field(
        None, 
        ge=0, 
        le=120, 
        description="Female life expectancy (years)"
    )
    
    # Self-assessed health
    excellent_very_good_health_percentage: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="% reporting excellent/very good health"
    )
    
    # === MORTALITY ===
    # Age-standardised mortality rates per 100,000
    all_cause_mortality_rate: Optional[float] = Field(
        None, 
        ge=0, 
        description="All-cause age-standardised mortality rate"
    )
    cardiovascular_mortality_rate: Optional[float] = Field(
        None, 
        ge=0, 
        description="Cardiovascular disease mortality rate"
    )
    cancer_mortality_rate: Optional[float] = Field(
        None, 
        ge=0, 
        description="Cancer mortality rate"
    )
    respiratory_mortality_rate: Optional[float] = Field(
        None, 
        ge=0, 
        description="Respiratory disease mortality rate"
    )
    diabetes_mortality_rate: Optional[float] = Field(
        None, 
        ge=0, 
        description="Diabetes mortality rate"
    )
    suicide_mortality_rate: Optional[float] = Field(
        None, 
        ge=0, 
        description="Suicide mortality rate"
    )
    
    # === MORBIDITY ===
    # Chronic disease prevalence rates (%)
    diabetes_prevalence: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Diabetes prevalence %"
    )
    hypertension_prevalence: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Hypertension prevalence %"
    )
    heart_disease_prevalence: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Heart disease prevalence %"
    )
    asthma_prevalence: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Asthma prevalence %"
    )
    copd_prevalence: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="COPD prevalence %"
    )
    mental_health_condition_prevalence: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Mental health condition prevalence %"
    )
    
    # === HEALTHCARE UTILISATION ===
    gp_visits_per_capita: Optional[float] = Field(
        None, 
        ge=0, 
        description="GP visits per capita per year"
    )
    specialist_visits_per_capita: Optional[float] = Field(
        None, 
        ge=0, 
        description="Specialist visits per capita per year"
    )
    hospital_admissions_per_1000: Optional[float] = Field(
        None, 
        ge=0, 
        description="Hospital admissions per 1,000 population"
    )
    emergency_presentations_per_1000: Optional[float] = Field(
        None, 
        ge=0, 
        description="Emergency department presentations per 1,000"
    )
    
    # === RISK FACTORS ===
    current_smoking_percentage: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Current smoking prevalence %"
    )
    obesity_percentage: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Obesity prevalence %"
    )
    overweight_obesity_percentage: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Overweight and obesity prevalence %"
    )
    physical_inactivity_percentage: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Physical inactivity prevalence %"
    )
    risky_alcohol_consumption_percentage: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Risky alcohol consumption prevalence %"
    )
    high_psychological_distress_percentage: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="High psychological distress prevalence %"
    )
    
    # === ACCESS AND QUALITY ===
    bulk_billing_percentage: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Bulk billing rate %"
    )
    distance_to_nearest_hospital_km: Optional[float] = Field(
        None, 
        ge=0, 
        description="Distance to nearest hospital (km)"
    )
    gp_workforce_per_1000: Optional[float] = Field(
        None, 
        ge=0, 
        description="GP workforce per 1,000 population"
    )
    
    # === MATERNAL AND CHILD HEALTH ===
    infant_mortality_rate: Optional[float] = Field(
        None, 
        ge=0, 
        description="Infant mortality rate per 1,000 live births"
    )
    low_birth_weight_percentage: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Low birth weight percentage"
    )
    
    # === HEALTH QUALITY METRICS ===
    profile_completeness_score: float = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Health profile data completeness %"
    )
    data_quality_flags: List[str] = Field(
        default_factory=list,
        description="Any data quality issues identified"
    )
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "SA2HealthProfile"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate SA2 health profile data integrity."""
        errors = []
        
        # Validate population age/sex breakdown consistency
        if self.population_by_age_sex:
            total_from_breakdown = sum(
                sum(sex_data.values()) for sex_data in self.population_by_age_sex.values()
            )
            if abs(total_from_breakdown - self.total_population) > 10:
                errors.append("Population breakdown doesn't match total population")
        
        # Check percentage bounds
        percentage_fields = [
            'excellent_very_good_health_percentage', 'diabetes_prevalence',
            'current_smoking_percentage', 'obesity_percentage',
            'bulk_billing_percentage', 'low_birth_weight_percentage'
        ]
        
        for field_name in percentage_fields:
            value = getattr(self, field_name)
            if value is not None and (value < 0 or value > 100):
                errors.append(f"{field_name} outside valid percentage range: {value}")
        
        return errors


class HealthIndicatorSummary(VersionedSchema):
    """
    Standardised health indicator aggregations for analysis and reporting.
    
    Provides summary statistics and aggregated indicators suitable for
    comparative analysis across areas or time periods.
    """
    
    # === IDENTIFICATION ===
    geographic_level: str = Field(..., description="Geographic aggregation level")
    geographic_id: str = Field(..., description="Geographic area identifier")
    geographic_name: str = Field(..., description="Geographic area name")
    reporting_period: TemporalData = Field(..., description="Reporting period")
    
    # === POPULATION SUMMARY ===
    population_covered: int = Field(..., ge=0, description="Total population covered")
    population_density: float = Field(..., ge=0, description="Population density per sq km")
    
    # === HEALTH OUTCOME INDICATORS ===
    health_outcome_score: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Composite health outcome score (0-100)"
    )
    
    # Mortality summary
    mortality_indicators: Dict[str, float] = Field(
        default_factory=dict,
        description="Key mortality indicators (age-standardised rates)"
    )
    avoidable_deaths_rate: Optional[float] = Field(
        None, 
        ge=0, 
        description="Avoidable deaths rate per 100,000"
    )
    
    # Morbidity summary  
    chronic_disease_burden: Dict[str, float] = Field(
        default_factory=dict,
        description="Chronic disease prevalence summary"
    )
    disability_adjusted_life_years: Optional[float] = Field(
        None, 
        ge=0, 
        description="Disability-adjusted life years per 1,000"
    )
    
    # Mental health summary
    mental_health_indicators: Dict[str, float] = Field(
        default_factory=dict,
        description="Mental health indicator summary"
    )
    
    # === HEALTHCARE SYSTEM INDICATORS ===
    healthcare_access_score: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Healthcare access composite score"
    )
    healthcare_utilisation_indicators: Dict[str, float] = Field(
        default_factory=dict,
        description="Healthcare utilisation summary"
    )
    healthcare_quality_indicators: Dict[str, float] = Field(
        default_factory=dict,
        description="Healthcare quality measures"
    )
    
    # === PREVENTION INDICATORS ===
    prevention_indicators: Dict[str, float] = Field(
        default_factory=dict,
        description="Prevention and screening indicators"
    )
    immunisation_coverage: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Overall immunisation coverage %"
    )
    
    # === RISK FACTOR INDICATORS ===
    risk_factor_burden: Dict[str, float] = Field(
        default_factory=dict,
        description="Risk factor prevalence summary"
    )
    modifiable_risk_score: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Modifiable risk factor burden score"
    )
    
    # === EQUITY INDICATORS ===
    health_equity_indicators: Dict[str, float] = Field(
        default_factory=dict,
        description="Health equity and disparity measures"
    )
    socioeconomic_health_gradient: Optional[float] = Field(
        None,
        description="Health gradient across socioeconomic groups"
    )
    
    # === COMPARATIVE METRICS ===
    national_comparison: Dict[str, float] = Field(
        default_factory=dict,
        description="Comparison to national averages (ratio)"
    )
    state_comparison: Dict[str, float] = Field(
        default_factory=dict,
        description="Comparison to state averages (ratio)"
    )
    peer_group_comparison: Dict[str, float] = Field(
        default_factory=dict,
        description="Comparison to similar areas"
    )
    
    # === TREND INDICATORS ===
    trend_indicators: Dict[str, float] = Field(
        default_factory=dict,
        description="Trend changes (annual % change)"
    )
    improvement_indicators: List[str] = Field(
        default_factory=list,
        description="Indicators showing improvement"
    )
    decline_indicators: List[str] = Field(
        default_factory=list,
        description="Indicators showing decline"
    )
    
    # === DATA QUALITY ===
    indicator_completeness: Dict[str, float] = Field(
        default_factory=dict,
        description="Completeness % for each indicator category"
    )
    overall_completeness: float = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Overall indicator completeness %"
    )
    quality_score: float = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Overall data quality score"
    )
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "HealthIndicatorSummary"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate health indicator summary data integrity."""
        errors = []
        
        # Validate comparison ratios
        for comparison_type in [self.national_comparison, self.state_comparison]:
            for indicator, ratio in comparison_type.items():
                if ratio < 0:
                    errors.append(f"Negative comparison ratio for {indicator}: {ratio}")
        
        # Validate completeness consistency
        if self.indicator_completeness:
            calculated_overall = sum(self.indicator_completeness.values()) / len(self.indicator_completeness)
            if abs(calculated_overall - self.overall_completeness) > 5:
                errors.append("Overall completeness inconsistent with category completeness")
        
        return errors


class GeographicHealthMapping(VersionedSchema):
    """
    Geographic relationships and health data linkages.
    
    Defines spatial relationships and geographic patterns in health data
    for spatial analysis and mapping applications.
    """
    
    # === GEOGRAPHIC IDENTIFICATION ===
    primary_area_id: str = Field(..., description="Primary geographic area identifier")
    primary_area_type: str = Field(..., description="Primary area type (SA2, SA3, etc)")
    primary_area_name: str = Field(..., description="Primary area name")
    
    # === SPATIAL RELATIONSHIPS ===
    containing_areas: Dict[str, str] = Field(
        ...,
        description="Higher-level areas containing this area"
    )
    contained_areas: List[str] = Field(
        default_factory=list,
        description="Lower-level areas contained within this area"
    )
    adjacent_areas: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Adjacent areas with relationship metrics"
    )
    
    # === GEOMETRIC PROPERTIES ===
    centroid_coordinates: Dict[str, float] = Field(
        ...,
        description="Centroid latitude and longitude"
    )
    area_square_km: float = Field(..., ge=0, description="Area in square kilometres")
    perimeter_km: float = Field(..., ge=0, description="Perimeter in kilometres")
    compactness_ratio: Optional[float] = Field(
        None, 
        ge=0, 
        le=1, 
        description="Shape compactness ratio"
    )
    
    # === ACCESS AND DISTANCE METRICS ===
    distance_to_services: Dict[str, float] = Field(
        default_factory=dict,
        description="Distance to key health services (km)"
    )
    travel_time_to_services: Dict[str, float] = Field(
        default_factory=dict,
        description="Travel time to key services (minutes)"
    )
    service_catchment_populations: Dict[str, int] = Field(
        default_factory=dict,
        description="Population within service catchments"
    )
    
    # === HEALTH SERVICE DENSITY ===
    health_services_within_area: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of health services within area"
    )
    health_workforce_density: Dict[str, float] = Field(
        default_factory=dict,
        description="Health workforce per 1,000 population"
    )
    
    # === ENVIRONMENTAL HEALTH FACTORS ===
    environmental_exposures: Dict[str, float] = Field(
        default_factory=dict,
        description="Environmental health exposure metrics"
    )
    green_space_percentage: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Green space coverage %"
    )
    air_quality_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Air quality measurements"
    )
    
    # === HEALTH PATTERNS ===
    spatial_health_clusters: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Identified spatial health clusters"
    )
    health_hotspots: List[str] = Field(
        default_factory=list,
        description="Health condition hotspot classifications"
    )
    health_coldspots: List[str] = Field(
        default_factory=list,
        description="Areas with better than expected health"
    )
    
    # === ACCESSIBILITY CLASSIFICATIONS ===
    remoteness_area: str = Field(..., description="ABS Remoteness Area classification")
    accessibility_category: str = Field(
        ...,
        description="Health service accessibility classification"
    )
    transport_access_score: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Public transport access score"
    )
    
    # === NEIGHBOURHOOD EFFECTS ===
    spatial_autocorrelation: Dict[str, float] = Field(
        default_factory=dict,
        description="Spatial autocorrelation measures for health indicators"
    )
    spillover_effects: Dict[str, float] = Field(
        default_factory=dict,
        description="Spillover effects from neighbouring areas"
    )
    
    # === MAPPING METADATA ===
    coordinate_system: str = Field(
        default="GDA2020",
        description="Coordinate reference system"
    )
    geometry_source: str = Field(..., description="Source of geometric data")
    geometry_date: datetime = Field(..., description="Date of geometry data")
    simplification_tolerance: Optional[float] = Field(
        None,
        description="Geometry simplification tolerance (metres)"
    )
    
    @field_validator('centroid_coordinates')
    @classmethod
    def validate_centroid_format(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate centroid coordinates format."""
        required_keys = ['latitude', 'longitude']
        for key in required_keys:
            if key not in v:
                raise ValueError(f"Missing required coordinate: {key}")
        
        lat = v['latitude']
        lon = v['longitude']
        
        # Validate Australian coordinate bounds
        if not (-44 <= lat <= -10):
            raise ValueError(f"Latitude {lat} outside Australian bounds")
        if not (112 <= lon <= 154):
            raise ValueError(f"Longitude {lon} outside Australian bounds")
        
        return v
    
    @field_validator('distance_to_services')
    @classmethod
    def validate_service_distances(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate service distance values."""
        for service, distance in v.items():
            if distance < 0:
                raise ValueError(f"Negative distance to {service}: {distance}")
            if distance > 1000:  # More than 1000km seems unrealistic
                raise ValueError(f"Unrealistic distance to {service}: {distance}km")
        return v
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "GeographicHealthMapping"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate geographic health mapping data integrity."""
        errors = []
        
        # Validate area calculations
        if self.area_square_km and self.perimeter_km:
            # Calculate theoretical compactness
            theoretical_compactness = (4 * 3.14159 * self.area_square_km) / (self.perimeter_km ** 2)
            if self.compactness_ratio and abs(self.compactness_ratio - theoretical_compactness) > 0.1:
                errors.append("Compactness ratio inconsistent with area and perimeter")
        
        # Validate containment hierarchy
        for level, area_id in self.containing_areas.items():
            if level == 'sa3' and len(area_id) != 5:
                errors.append(f"Invalid SA3 code length: {area_id}")
            elif level == 'sa4' and len(area_id) != 3:
                errors.append(f"Invalid SA4 code length: {area_id}")
        
        # Check service density reasonableness
        for service, density in self.health_workforce_density.items():
            if density < 0:
                errors.append(f"Negative workforce density for {service}")
            elif density > 100:  # More than 100 per 1000 seems high
                errors.append(f"Unusually high workforce density for {service}: {density}")
        
        return errors


# Migration functions for integrated schemas

def migrate_master_health_record_v1_to_v2(old_data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate MasterHealthRecord from v1.0.0 to v2.0.0."""
    new_data = old_data.copy()
    
    # Restructure geographic hierarchy if needed
    if 'sa3_code' in old_data:
        if 'geographic_hierarchy' not in new_data:
            new_data['geographic_hierarchy'] = {}
        new_data['geographic_hierarchy']['sa3_code'] = old_data.pop('sa3_code')
        
    if 'sa4_code' in old_data:
        if 'geographic_hierarchy' not in new_data:
            new_data['geographic_hierarchy'] = {}
        new_data['geographic_hierarchy']['sa4_code'] = old_data.pop('sa4_code')
    
    # Update schema version
    new_data['schema_version'] = SchemaVersion.V2_0_0.value
    
    return new_data


def validate_integrated_schema_set(schemas: List[VersionedSchema]) -> List[str]:
    """
    Validate a set of integrated schemas for consistency.
    
    Args:
        schemas: List of schema instances to validate together
        
    Returns:
        List of validation errors across the schema set
    """
    errors = []
    
    # Extract SA2 codes and check for consistency
    sa2_codes = []
    for schema in schemas:
        if hasattr(schema, 'sa2_code'):
            sa2_codes.append(schema.sa2_code)
        elif hasattr(schema, 'primary_area_id'):
            sa2_codes.append(schema.primary_area_id)
    
    # Check for duplicate SA2 codes
    if len(sa2_codes) != len(set(sa2_codes)):
        errors.append("Duplicate SA2 codes found in schema set")
    
    # Validate cross-schema consistency
    master_records = [s for s in schemas if isinstance(s, MasterHealthRecord)]
    health_profiles = [s for s in schemas if isinstance(s, SA2HealthProfile)]
    
    for master in master_records:
        # Find corresponding health profile
        matching_profiles = [hp for hp in health_profiles if hp.sa2_code == master.sa2_code]
        if matching_profiles:
            profile = matching_profiles[0]
            # Check population consistency
            if abs(master.total_population - profile.total_population) > 10:
                errors.append(f"Population mismatch for SA2 {master.sa2_code}")
    
    return errors