"""
Health Data Models for Australian Health Analytics

Pydantic models for health service data (MBS/PBS), mortality data (AIHW), 
chronic disease data (PHIDU), and healthcare variation data.
"""

from typing import Optional, List, Union
from decimal import Decimal
from datetime import date, datetime
from enum import Enum

from pydantic import Field, validator
from pydantic.types import constr, confloat, conint

from .base import GeographicModel, DataQualityMixin, TimestampedModel, PopulationMixin


class ServiceType(str, Enum):
    """Health service types."""
    MEDICAL = "MEDICAL"           # Medical services
    DIAGNOSTIC = "DIAGNOSTIC"     # Diagnostic procedures  
    PATHOLOGY = "PATHOLOGY"       # Pathology tests
    ALLIED_HEALTH = "ALLIED_HEALTH"  # Allied health services
    SPECIALIST = "SPECIALIST"     # Specialist consultations
    SURGICAL = "SURGICAL"         # Surgical procedures
    EMERGENCY = "EMERGENCY"       # Emergency services
    MENTAL_HEALTH = "MENTAL_HEALTH"  # Mental health services


class AgeGroup(str, Enum):
    """Standard age groupings for health data."""
    INFANT = "0-1"
    CHILD = "2-12" 
    ADOLESCENT = "13-17"
    YOUNG_ADULT = "18-24"
    ADULT = "25-44"
    MIDDLE_AGE = "45-64"
    OLDER_ADULT = "65-74"
    ELDERLY = "75+"
    ALL_AGES = "ALL"


class Gender(str, Enum):
    """Gender categories."""
    MALE = "MALE"
    FEMALE = "FEMALE" 
    OTHER = "OTHER"
    ALL = "ALL"


class MBSRecord(GeographicModel, DataQualityMixin, TimestampedModel):
    """
    Medicare Benefits Schedule (MBS) service utilisation record.
    
    Captures healthcare service usage patterns by geographic area,
    age group, and service type.
    """
    
    # Service identification
    mbs_item_number: constr(pattern=r"^[0-9]{1,6}$") = Field(
        ...,
        description="MBS item number",
        examples=["23", "721", "36"]
    )
    
    mbs_item_description: constr(min_length=1, max_length=500) = Field(
        ...,
        description="Description of MBS service"
    )
    
    service_type: ServiceType = Field(
        ...,
        description="Categorised service type"
    )
    
    # Demographics
    age_group: AgeGroup = Field(
        ...,
        description="Age group of service recipients"
    )
    
    gender: Gender = Field(
        ...,
        description="Gender of service recipients"
    )
    
    # Service utilisation metrics
    service_count: conint(ge=0) = Field(
        ...,
        description="Number of services provided"
    )
    
    patient_count: Optional[conint(ge=0)] = Field(
        None,
        description="Number of unique patients (if available)"
    )
    
    benefit_paid: confloat(ge=0.0) = Field(
        ...,
        description="Total Medicare benefit paid (AUD)"
    )
    
    # Rates per population
    services_per_1000_population: Optional[confloat(ge=0.0)] = Field(
        None,
        description="Service rate per 1,000 population"
    )
    
    patients_per_1000_population: Optional[confloat(ge=0.0)] = Field(
        None,
        description="Patient rate per 1,000 population"
    )
    
    average_benefit_per_service: Optional[confloat(ge=0.0)] = Field(
        None,
        description="Average benefit amount per service (AUD)"
    )
    
    # Time period
    financial_year: constr(pattern=r"^20[0-9]{2}-[0-9]{2}$") = Field(
        ...,
        description="Financial year (e.g., '2021-22')",
        examples=["2021-22", "2020-21"]
    )
    
    quarter: Optional[constr(pattern=r"^Q[1-4]$")] = Field(
        None,
        description="Quarter within financial year",
        examples=["Q1", "Q2", "Q3", "Q4"]
    )


class PBSRecord(GeographicModel, DataQualityMixin, TimestampedModel):
    """
    Pharmaceutical Benefits Scheme (PBS) prescription data.
    
    Tracks pharmaceutical usage patterns and costs by geographic area.
    """
    
    # Medicine identification
    pbs_item_code: constr(pattern=r"^[0-9]{4}[A-Z]?$") = Field(
        ...,
        description="PBS item code",
        examples=["8254K", "2622B", "1215Y"]
    )
    
    medicine_name: constr(min_length=1, max_length=200) = Field(
        ...,
        description="Generic medicine name"
    )
    
    brand_name: Optional[str] = Field(
        None,
        description="Brand/trade name"
    )
    
    atc_code: Optional[constr(pattern=r"^[A-Z][0-9]{2}[A-Z]{2}[0-9]{2}$")] = Field(
        None,
        description="Anatomical Therapeutic Chemical (ATC) classification code",
        examples=["C09AA02", "N06AB03"]
    )
    
    therapeutic_group: Optional[str] = Field(
        None,
        description="Therapeutic group classification"
    )
    
    # Demographics
    age_group: AgeGroup = Field(
        ...,
        description="Age group of patients"
    )
    
    gender: Gender = Field(
        ...,
        description="Gender of patients"
    )
    
    # Prescription metrics
    prescription_count: conint(ge=0) = Field(
        ...,
        description="Number of prescriptions dispensed"
    )
    
    patient_count: Optional[conint(ge=0)] = Field(
        None,
        description="Number of unique patients"
    )
    
    ddd_per_1000_population_per_day: Optional[confloat(ge=0.0)] = Field(
        None,
        description="Defined Daily Doses per 1000 population per day"
    )
    
    # Costs
    government_benefit: confloat(ge=0.0) = Field(
        ...,
        description="Government benefit paid (AUD)"
    )
    
    patient_contribution: Optional[confloat(ge=0.0)] = Field(
        None,
        description="Patient co-payment (AUD)"
    )
    
    total_cost: Optional[confloat(ge=0.0)] = Field(
        None,
        description="Total cost of medicines (AUD)"
    )
    
    # Time period
    financial_year: constr(pattern=r"^20[0-9]{2}-[0-9]{2}$") = Field(
        ...,
        description="Financial year"
    )
    
    month: Optional[constr(pattern=r"^(0[1-9]|1[0-2])$")] = Field(
        None,
        description="Month (01-12)"
    )


class CauseOfDeath(str, Enum):
    """Standard cause of death categories."""
    ALL_CAUSES = "ALL_CAUSES"
    CANCER = "CANCER"
    CARDIOVASCULAR = "CARDIOVASCULAR"  
    RESPIRATORY = "RESPIRATORY"
    DIABETES = "DIABETES"
    MENTAL_HEALTH = "MENTAL_HEALTH"
    SUICIDE = "SUICIDE"
    ACCIDENT = "ACCIDENT"
    DEMENTIA = "DEMENTIA"
    KIDNEY_DISEASE = "KIDNEY_DISEASE"
    LIVER_DISEASE = "LIVER_DISEASE"
    COPD = "COPD"
    OTHER = "OTHER"


class AIHWMortalityRecord(GeographicModel, DataQualityMixin, TimestampedModel):
    """
    AIHW mortality data from MORT and GRIM datasets.
    
    Provides death counts, rates, and mortality indicators by geographic area
    and cause of death.
    """
    
    # Cause classification
    cause_of_death: CauseOfDeath = Field(
        ...,
        description="Primary cause of death category"
    )
    
    icd_10_code: Optional[constr(pattern=r"^[A-Z][0-9]{2}(\.[0-9])?$")] = Field(
        None,
        description="ICD-10 disease classification code",
        examples=["C78.0", "I21.9", "F32.2"]
    )
    
    cause_description: Optional[str] = Field(
        None,
        description="Detailed description of cause of death"
    )
    
    # Demographics
    age_group: AgeGroup = Field(
        ...,
        description="Age group of deaths"
    )
    
    gender: Gender = Field(
        ...,
        description="Gender of deaths"
    )
    
    # Mortality indicators
    death_count: conint(ge=0) = Field(
        ...,
        description="Number of deaths"
    )
    
    crude_death_rate: Optional[confloat(ge=0.0)] = Field(
        None,
        description="Crude death rate per 100,000 population"
    )
    
    age_standardised_rate: Optional[confloat(ge=0.0)] = Field(
        None,
        description="Age-standardised death rate per 100,000 population"
    )
    
    # Premature mortality
    premature_death_count: Optional[conint(ge=0)] = Field(
        None,
        description="Deaths before age 75"
    )
    
    years_of_life_lost: Optional[confloat(ge=0.0)] = Field(
        None,
        description="Potential years of life lost"
    )
    
    avoidable_death_count: Optional[conint(ge=0)] = Field(
        None,
        description="Potentially avoidable deaths"
    )
    
    # Time period
    calendar_year: conint(ge=1900, le=2030) = Field(
        ...,
        description="Calendar year of death"
    )
    
    # Data quality
    suppression_flag: Optional[bool] = Field(
        None,
        description="Whether data is suppressed for privacy (<5 deaths)"
    )
    
    data_source: str = Field(
        ...,
        pattern=r"^(MORT|GRIM|NMD)$",
        description="Source dataset (MORT/GRIM/National Mortality Database)"
    )


class ChronicDiseaseType(str, Enum):
    """Chronic disease categories."""
    DIABETES = "DIABETES"
    CARDIOVASCULAR = "CARDIOVASCULAR"
    CANCER = "CANCER" 
    MENTAL_HEALTH = "MENTAL_HEALTH"
    RESPIRATORY = "RESPIRATORY"
    ARTHRITIS = "ARTHRITIS"
    KIDNEY_DISEASE = "KIDNEY_DISEASE"
    DEMENTIA = "DEMENTIA"
    STROKE = "STROKE"
    OSTEOPOROSIS = "OSTEOPOROSIS"


class PHIDUChronicDiseaseRecord(GeographicModel, DataQualityMixin, PopulationMixin):
    """
    PHIDU chronic disease prevalence data.
    
    Population Health Information Development Unit data on chronic disease
    prevalence and health service utilisation.
    """
    
    # Disease classification
    disease_type: ChronicDiseaseType = Field(
        ...,
        description="Type of chronic disease"
    )
    
    disease_description: Optional[str] = Field(
        None,
        description="Detailed disease description"
    )
    
    # Prevalence indicators
    prevalence_rate: confloat(ge=0.0, le=100.0) = Field(
        ...,
        description="Disease prevalence rate (%)"
    )
    
    prevalence_count: Optional[conint(ge=0)] = Field(
        None,
        description="Estimated number of people with disease"
    )
    
    age_standardised_prevalence: Optional[confloat(ge=0.0, le=100.0)] = Field(
        None,
        description="Age-standardised prevalence rate (%)"
    )
    
    # Demographics  
    age_group: AgeGroup = Field(
        ...,
        description="Age group for prevalence data"
    )
    
    gender: Gender = Field(
        ...,
        description="Gender for prevalence data"
    )
    
    # Service utilisation
    gp_visits_per_person: Optional[confloat(ge=0.0)] = Field(
        None,
        description="Average GP visits per person per year"
    )
    
    specialist_visits_per_person: Optional[confloat(ge=0.0)] = Field(
        None,
        description="Average specialist visits per person per year"
    )
    
    hospitalisation_rate: Optional[confloat(ge=0.0)] = Field(
        None,
        description="Hospitalisation rate per 1000 population"
    )
    
    # Risk factors
    risk_factor_score: Optional[confloat(ge=0.0, le=1.0)] = Field(
        None,
        description="Composite risk factor score"
    )
    
    modifiable_risk_factors: Optional[List[str]] = Field(
        None,
        description="List of relevant modifiable risk factors"
    )
    
    # Geographic mapping (PHAs to SA2s)
    pha_code: Optional[str] = Field(
        None,
        description="Population Health Area code"
    )
    
    pha_name: Optional[str] = Field(
        None,
        description="Population Health Area name"  
    )
    
    sa2_mapping_percentage: Optional[confloat(ge=0.0, le=100.0)] = Field(
        None,
        description="Percentage of PHA mapped to this SA2"
    )


class HealthcareVariationType(str, Enum):
    """Healthcare variation indicator types."""
    HOSPITALISATION = "HOSPITALISATION"
    SURGERY = "SURGERY" 
    INVESTIGATION = "INVESTIGATION"
    MEDICATION_USE = "MEDICATION_USE"
    SCREENING = "SCREENING"
    EMERGENCY_ADMISSION = "EMERGENCY_ADMISSION"
    PLANNED_ADMISSION = "PLANNED_ADMISSION"


class HealthcareVariationRecord(GeographicModel, DataQualityMixin, TimestampedModel):
    """
    Australian Atlas of Healthcare Variation data.
    
    Captures variation in healthcare delivery and outcomes across
    geographic areas and healthcare providers.
    """
    
    # Indicator identification
    variation_type: HealthcareVariationType = Field(
        ...,
        description="Type of healthcare variation indicator"
    )
    
    indicator_name: str = Field(
        ...,
        description="Specific healthcare indicator name"
    )
    
    indicator_description: Optional[str] = Field(
        None,
        description="Detailed description of the indicator"
    )
    
    # Clinical condition
    primary_condition: Optional[str] = Field(
        None,
        description="Primary clinical condition or procedure"
    )
    
    procedure_code: Optional[str] = Field(
        None,
        description="Clinical procedure or diagnosis code"
    )
    
    # Variation metrics
    rate_per_population: confloat(ge=0.0) = Field(
        ...,
        description="Rate per population (various denominators)"
    )
    
    population_denominator: conint(ge=1000) = Field(
        ...,
        description="Population denominator for rate calculation"
    )
    
    # Comparative measures
    national_average: Optional[confloat(ge=0.0)] = Field(
        None,
        description="National average rate for comparison"
    )
    
    variation_ratio: Optional[confloat(ge=0.0)] = Field(
        None,
        description="Ratio compared to national average"
    )
    
    percentile_rank: Optional[conint(ge=1, le=100)] = Field(
        None,
        description="Percentile ranking compared to all areas"
    )
    
    # Statistical measures
    confidence_interval_lower: Optional[confloat(ge=0.0)] = Field(
        None,
        description="Lower 95% confidence interval"
    )
    
    confidence_interval_upper: Optional[confloat(ge=0.0)] = Field(
        None,
        description="Upper 95% confidence interval"
    )
    
    # Demographics
    age_group: AgeGroup = Field(
        AgeGroup.ALL_AGES,
        description="Age group for this indicator"
    )
    
    gender: Gender = Field(
        Gender.ALL,
        description="Gender for this indicator"
    )
    
    # Provider information  
    primary_health_network: Optional[str] = Field(
        None,
        description="Primary Health Network code"
    )
    
    provider_type: Optional[str] = Field(
        None,
        pattern=r"^(PUBLIC|PRIVATE|MIXED)$",
        description="Type of healthcare provider"
    )
    
    # Time period
    financial_year_start: constr(pattern=r"^20[0-9]{2}$") = Field(
        ...,
        description="Start year of reporting period",
        examples=["2017", "2018"]
    )
    
    financial_year_end: constr(pattern=r"^20[0-9]{2}$") = Field(
        ...,
        description="End year of reporting period",
        examples=["2018", "2019"]
    )
    
    @validator('financial_year_end')
    def validate_year_sequence(cls, v, values):
        """Ensure end year is after start year."""
        start_year = values.get('financial_year_start')
        if start_year and int(v) <= int(start_year):
            raise ValueError("End year must be after start year")
        return v