"""
AIHW Mortality data schemas for AHGD.

This module defines schemas for Australian Institute of Health and Welfare (AIHW)
mortality data including death statistics, causes of death, and health outcomes.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
from enum import Enum
from pydantic import Field, field_validator, model_validator
import re

from .base_schema import (
    VersionedSchema,
    DataSource,
    SchemaVersion,
    DataQualityLevel,
    TemporalData
)


class CauseOfDeathType(str, Enum):
    """Types of cause of death classifications."""
    UNDERLYING = "underlying"  # Underlying cause of death
    IMMEDIATE = "immediate"  # Immediate cause of death
    CONTRIBUTING = "contributing"  # Contributing cause
    MULTIPLE = "multiple"  # Multiple causes


class DeathRegistrationType(str, Enum):
    """Types of death registration."""
    DOCTOR_CERTIFIED = "doctor_certified"
    CORONER_CERTIFIED = "coroner_certified"
    UNKNOWN = "unknown"


class PlaceOfDeathType(str, Enum):
    """Place where death occurred."""
    HOSPITAL = "hospital"
    HOME = "home"
    AGED_CARE = "aged_care"
    OTHER_INSTITUTION = "other_institution"
    PUBLIC_PLACE = "public_place"
    OTHER = "other"
    NOT_STATED = "not_stated"


class MortalityRecord(VersionedSchema, TemporalData):
    """Schema for individual mortality records."""
    
    # Record identification
    record_id: str = Field(..., description="Unique mortality record identifier")
    registration_year: int = Field(..., ge=1900, description="Year of death registration")
    registration_state: str = Field(..., description="State of death registration")
    
    # Geographic information
    usual_residence_sa2: Optional[str] = Field(None, description="SA2 of usual residence")
    usual_residence_sa3: Optional[str] = Field(None, description="SA3 of usual residence")
    usual_residence_sa4: Optional[str] = Field(None, description="SA4 of usual residence")
    usual_residence_state: Optional[str] = Field(None, description="State of usual residence")
    place_of_death_sa2: Optional[str] = Field(None, description="SA2 where death occurred")
    place_of_death_type: PlaceOfDeathType = Field(..., description="Type of place where death occurred")
    
    # Demographics
    age_at_death: int = Field(..., ge=0, le=150, description="Age at death in years")
    sex: str = Field(..., description="Sex of deceased")
    indigenous_status: Optional[str] = Field(None, description="Indigenous status")
    country_of_birth: Optional[str] = Field(None, description="Country of birth")
    
    # Cause of death
    underlying_cause_icd10: str = Field(..., description="Underlying cause of death (ICD-10)")
    underlying_cause_description: str = Field(..., description="Description of underlying cause")
    immediate_cause_icd10: Optional[str] = Field(None, description="Immediate cause of death (ICD-10)")
    contributing_causes: List[str] = Field(default_factory=list, description="Contributing causes (ICD-10)")
    
    # Death circumstances
    death_date: date = Field(..., description="Date of death")
    registration_type: DeathRegistrationType = Field(..., description="Type of death certification")
    autopsy_performed: Optional[bool] = Field(None, description="Whether autopsy was performed")
    
    # Classification flags
    is_external_cause: bool = Field(False, description="Whether death was due to external causes")
    is_injury_death: bool = Field(False, description="Whether death was injury-related")
    is_suicide: bool = Field(False, description="Whether death was suicide")
    is_accident: bool = Field(False, description="Whether death was accidental")
    is_assault: bool = Field(False, description="Whether death was due to assault")
    
    # Data processing
    years_of_life_lost: Optional[float] = Field(None, ge=0, description="Years of potential life lost")
    age_standardised_flag: bool = Field(False, description="Whether record is included in age-standardised rates")
    
    # Data source
    data_source: DataSource = Field(..., description="Source of mortality data")
    
    @field_validator('sex')
    @classmethod
    def validate_sex(cls, v: str) -> str:
        """Validate sex categories."""
        valid_sex = {'Male', 'Female', 'M', 'F', '1', '2'}
        if v not in valid_sex:
            raise ValueError(f"Invalid sex value: {v}")
        # Standardise to full names
        if v in {'M', '1'}:
            return 'Male'
        elif v in {'F', '2'}:
            return 'Female'
        return v
    
    @field_validator('underlying_cause_icd10', 'immediate_cause_icd10')
    @classmethod
    def validate_icd10_code(cls, v: Optional[str]) -> Optional[str]:
        """Validate ICD-10 code format."""
        if v is None:
            return v
        
        # ICD-10 codes: Letter followed by 2 digits, optional decimal point and 1-2 more digits
        if not re.match(r'^[A-Z]\d{2}(\.\d{1,2})?$', v.upper()):
            raise ValueError(f"Invalid ICD-10 code format: {v}")
        
        return v.upper()
    
    @field_validator('contributing_causes')
    @classmethod
    def validate_contributing_causes(cls, v: List[str]) -> List[str]:
        """Validate contributing cause ICD-10 codes."""
        validated = []
        for cause in v:
            if not re.match(r'^[A-Z]\d{2}(\.\d{1,2})?$', cause.upper()):
                raise ValueError(f"Invalid ICD-10 code in contributing causes: {cause}")
            validated.append(cause.upper())
        return validated
    
    @field_validator('registration_state', 'usual_residence_state')
    @classmethod
    def validate_state(cls, v: Optional[str]) -> Optional[str]:
        """Validate Australian state codes."""
        if v is None:
            return v
        
        valid_states = {'NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT', 'OS'}
        if v.upper() not in valid_states:
            raise ValueError(f"Invalid state code: {v}")
        return v.upper()
    
    @model_validator(mode='after')
    def validate_mortality_consistency(self) -> 'MortalityRecord':
        """Validate mortality record consistency."""
        # Check age against years of life lost
        if self.years_of_life_lost is not None:
            # Standard life expectancy assumption (could be more sophisticated)
            expected_life_expectancy = 85
            expected_yll = max(0, expected_life_expectancy - self.age_at_death)
            
            if abs(self.years_of_life_lost - expected_yll) > 10:
                # Allow some variation but flag major discrepancies
                pass
        
        # Check external cause flags are consistent
        external_causes = ['V', 'W', 'X', 'Y']  # ICD-10 chapters for external causes
        is_external_by_code = any(self.underlying_cause_icd10.startswith(prefix) 
                                 for prefix in external_causes)
        
        if is_external_by_code != self.is_external_cause:
            raise ValueError("External cause flag inconsistent with ICD-10 code")
        
        # Check suicide flag consistency
        suicide_codes = ['X60', 'X61', 'X62', 'X63', 'X64', 'X65', 'X66', 'X67', 
                        'X68', 'X69', 'X70', 'X71', 'X72', 'X73', 'X74', 'X75', 
                        'X76', 'X77', 'X78', 'X79', 'X80', 'X81', 'X82', 'X83', 'X84']
        is_suicide_by_code = any(self.underlying_cause_icd10.startswith(code) 
                               for code in suicide_codes)
        
        if is_suicide_by_code != self.is_suicide:
            raise ValueError("Suicide flag inconsistent with ICD-10 code")
        
        return self
    
    def get_age_group_5year(self) -> str:
        """Get 5-year age group classification."""
        if self.age_at_death < 1:
            return "0"
        elif self.age_at_death < 5:
            return "1-4"
        elif self.age_at_death >= 85:
            return "85+"
        else:
            lower = (self.age_at_death // 5) * 5
            upper = lower + 4
            return f"{lower}-{upper}"
    
    def get_broad_cause_category(self) -> str:
        """Get broad cause of death category based on ICD-10."""
        code = self.underlying_cause_icd10
        
        if code.startswith(('A', 'B')):
            return "Infectious and parasitic diseases"
        elif code.startswith('C') or code.startswith('D') and code <= 'D48':
            return "Neoplasms"
        elif code.startswith('D5') or code.startswith('D6') or code.startswith('D7') or code.startswith('D8'):
            return "Blood and immune disorders"
        elif code.startswith('E'):
            return "Endocrine, nutritional and metabolic diseases"
        elif code.startswith('F'):
            return "Mental and behavioural disorders"
        elif code.startswith('G'):
            return "Nervous system diseases"
        elif code.startswith('I'):
            return "Circulatory system diseases"
        elif code.startswith('J'):
            return "Respiratory system diseases"
        elif code.startswith('K'):
            return "Digestive system diseases"
        elif code.startswith(('V', 'W', 'X', 'Y')):
            return "External causes"
        else:
            return "Other causes"
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "MortalityRecord"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate mortality data integrity."""
        errors = []
        
        # Check reasonable age at death
        if self.age_at_death > 120:
            errors.append(f"Age at death unusually high: {self.age_at_death}")
        
        # Check date consistency
        if self.death_date.year != self.registration_year:
            # Death and registration can be in different years, but flag large gaps
            if abs(self.death_date.year - self.registration_year) > 3:
                errors.append("Large gap between death date and registration year")
        
        # Check geographic consistency
        if (self.usual_residence_state and self.registration_state and
            self.usual_residence_state != self.registration_state):
            # Cross-border deaths are possible but worth noting
            pass
        
        return errors


class MortalityStatistics(VersionedSchema, TemporalData):
    """Schema for aggregated mortality statistics."""
    
    # Geographic and temporal identifiers
    geographic_id: str = Field(..., description="Geographic area identifier")
    geographic_level: str = Field(..., description="Geographic level")
    geographic_name: str = Field(..., description="Geographic area name")
    reference_year: int = Field(..., ge=1900, description="Reference year")
    
    # Population denominator
    population_base: int = Field(..., ge=0, description="Population base for rate calculation")
    
    # Death counts by demographics
    total_deaths: int = Field(..., ge=0, description="Total number of deaths")
    male_deaths: int = Field(0, ge=0, description="Deaths among males")
    female_deaths: int = Field(0, ge=0, description="Deaths among females")
    
    # Deaths by age group
    deaths_0_4: int = Field(0, ge=0, description="Deaths aged 0-4 years")
    deaths_5_14: int = Field(0, ge=0, description="Deaths aged 5-14 years")
    deaths_15_24: int = Field(0, ge=0, description="Deaths aged 15-24 years")
    deaths_25_34: int = Field(0, ge=0, description="Deaths aged 25-34 years")
    deaths_35_44: int = Field(0, ge=0, description="Deaths aged 35-44 years")
    deaths_45_54: int = Field(0, ge=0, description="Deaths aged 45-54 years")
    deaths_55_64: int = Field(0, ge=0, description="Deaths aged 55-64 years")
    deaths_65_74: int = Field(0, ge=0, description="Deaths aged 65-74 years")
    deaths_75_84: int = Field(0, ge=0, description="Deaths aged 75-84 years")
    deaths_85_plus: int = Field(0, ge=0, description="Deaths aged 85+ years")
    
    # Major cause categories
    cardiovascular_deaths: int = Field(0, ge=0, description="Deaths from cardiovascular disease")
    cancer_deaths: int = Field(0, ge=0, description="Deaths from cancer")
    respiratory_deaths: int = Field(0, ge=0, description="Deaths from respiratory disease")
    external_deaths: int = Field(0, ge=0, description="Deaths from external causes")
    dementia_deaths: int = Field(0, ge=0, description="Deaths from dementia")
    diabetes_deaths: int = Field(0, ge=0, description="Deaths from diabetes")
    kidney_disease_deaths: int = Field(0, ge=0, description="Deaths from kidney disease")
    suicide_deaths: int = Field(0, ge=0, description="Deaths from suicide")
    
    # Rates per 100,000 population
    crude_death_rate: float = Field(..., ge=0, description="Crude death rate per 100,000")
    age_standardised_rate: Optional[float] = Field(None, ge=0, description="Age-standardised death rate")
    infant_mortality_rate: Optional[float] = Field(None, ge=0, description="Infant mortality rate per 1,000 births")
    
    # Life expectancy measures
    life_expectancy_male: Optional[float] = Field(None, ge=0, le=120, description="Male life expectancy")
    life_expectancy_female: Optional[float] = Field(None, ge=0, le=120, description="Female life expectancy")
    
    # Years of life lost
    total_yll: Optional[float] = Field(None, ge=0, description="Total years of life lost")
    yll_rate: Optional[float] = Field(None, ge=0, description="YLL rate per 100,000")
    
    # Data quality indicators
    completeness_score: float = Field(..., ge=0, le=100, description="Data completeness percentage")
    timeliness_score: Optional[float] = Field(None, ge=0, le=100, description="Data timeliness score")
    
    # Data source
    data_source: DataSource = Field(..., description="Source of mortality statistics")
    
    @model_validator(mode='after')
    def validate_statistics_consistency(self) -> 'MortalityStatistics':
        """Validate mortality statistics consistency."""
        # Check gender totals
        if self.male_deaths + self.female_deaths != self.total_deaths:
            raise ValueError("Male + female deaths must equal total deaths")
        
        # Check age group totals
        age_total = (self.deaths_0_4 + self.deaths_5_14 + self.deaths_15_24 +
                    self.deaths_25_34 + self.deaths_35_44 + self.deaths_45_54 +
                    self.deaths_55_64 + self.deaths_65_74 + self.deaths_75_84 +
                    self.deaths_85_plus)
        
        if abs(age_total - self.total_deaths) > 5:  # Allow small discrepancies
            raise ValueError("Age group deaths don't sum to total deaths")
        
        # Check crude death rate calculation
        if self.population_base > 0:
            calculated_rate = (self.total_deaths / self.population_base) * 100000
            if abs(calculated_rate - self.crude_death_rate) > 1:
                raise ValueError("Crude death rate doesn't match calculated value")
        
        # Check life expectancy gender difference
        if (self.life_expectancy_male and self.life_expectancy_female):
            if self.life_expectancy_male > self.life_expectancy_female + 5:
                raise ValueError("Male life expectancy unusually higher than female")
        
        return self
    
    def get_cause_specific_rate(self, cause_deaths: int) -> float:
        """Calculate cause-specific mortality rate per 100,000."""
        if self.population_base > 0:
            return (cause_deaths / self.population_base) * 100000
        return 0.0
    
    def get_proportional_mortality(self, cause_deaths: int) -> float:
        """Calculate proportional mortality ratio as percentage."""
        if self.total_deaths > 0:
            return (cause_deaths / self.total_deaths) * 100
        return 0.0
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "MortalityStatistics"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate mortality statistics integrity."""
        errors = []
        
        # Check for unusually high death rates
        if self.crude_death_rate > 3000:  # More than 3% of population
            errors.append(f"Crude death rate unusually high: {self.crude_death_rate}")
        
        # Check life expectancy bounds
        if self.life_expectancy_male and self.life_expectancy_male < 50:
            errors.append(f"Male life expectancy unusually low: {self.life_expectancy_male}")
        
        if self.life_expectancy_female and self.life_expectancy_female < 50:
            errors.append(f"Female life expectancy unusually low: {self.life_expectancy_female}")
        
        # Check cause-specific proportions
        major_causes = (self.cardiovascular_deaths + self.cancer_deaths + 
                       self.respiratory_deaths + self.external_deaths)
        if self.total_deaths > 0:
            major_cause_pct = (major_causes / self.total_deaths) * 100
            if major_cause_pct > 90:
                errors.append("Major causes account for unusually high proportion of deaths")
        
        return errors


class MortalityTrend(VersionedSchema):
    """Schema for mortality trend analysis over time."""
    
    # Geographic identification
    geographic_id: str = Field(..., description="Geographic area identifier")
    geographic_level: str = Field(..., description="Geographic level")
    cause_of_death: str = Field(..., description="Cause of death (ICD-10 or category)")
    
    # Time series data
    start_year: int = Field(..., ge=1900, description="Start year of trend")
    end_year: int = Field(..., ge=1900, description="End year of trend")
    data_points: List[Dict[str, Any]] = Field(..., description="Yearly data points")
    
    # Trend analysis
    trend_direction: str = Field(..., description="Overall trend direction (increasing/decreasing/stable)")
    annual_change_rate: float = Field(..., description="Average annual change rate (%)")
    r_squared: Optional[float] = Field(None, ge=0, le=1, description="R-squared value for trend line")
    statistical_significance: Optional[bool] = Field(None, description="Whether trend is statistically significant")
    
    # Breakpoint analysis
    has_breakpoints: bool = Field(False, description="Whether trend has significant breakpoints")
    breakpoint_years: List[int] = Field(default_factory=list, description="Years where trend changes")
    
    @field_validator('data_points')
    @classmethod
    def validate_data_points(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate data points structure."""
        for point in v:
            if 'year' not in point or 'rate' not in point:
                raise ValueError("Each data point must have 'year' and 'rate' fields")
            if not isinstance(point['year'], int) or not isinstance(point['rate'], (int, float)):
                raise ValueError("Year must be integer, rate must be numeric")
        return v
    
    @model_validator(mode='after')
    def validate_trend_consistency(self) -> 'MortalityTrend':
        """Validate trend analysis consistency."""
        if self.end_year <= self.start_year:
            raise ValueError("End year must be after start year")
        
        # Check data points span the expected period
        years = [point['year'] for point in self.data_points]
        if years:
            if min(years) > self.start_year or max(years) < self.end_year:
                raise ValueError("Data points don't cover the specified time period")
        
        return self
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "MortalityTrend"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate trend data integrity."""
        errors = []
        
        # Check for reasonable change rates
        if abs(self.annual_change_rate) > 50:
            errors.append(f"Annual change rate unusually high: {self.annual_change_rate}%")
        
        # Check R-squared value
        if self.r_squared is not None and self.r_squared < 0.1:
            errors.append("Very low R-squared suggests poor trend fit")
        
        return errors


# Migration functions for mortality schemas

def migrate_mortality_v1_to_v2(old_data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate mortality data from v1.0.0 to v2.0.0."""
    new_data = old_data.copy()
    
    # Example migration: standardise cause codes
    if 'cause_code' in old_data:
        new_data['underlying_cause_icd10'] = old_data.pop('cause_code')
    
    # Convert old age categories
    if 'infant_deaths' in old_data and 'child_deaths' in old_data:
        infant = old_data.pop('infant_deaths', 0)
        child = old_data.pop('child_deaths', 0)
        new_data['deaths_0_4'] = infant + child
    
    # Update geographic fields
    if 'area_code' in old_data:
        new_data['geographic_id'] = old_data.pop('area_code')
    
    # Update schema version
    new_data['schema_version'] = SchemaVersion.V2_0_0.value
    
    return new_data