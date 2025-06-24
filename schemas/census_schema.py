"""
ABS Census data schemas for AHGD.

This module defines schemas for Australian Bureau of Statistics (ABS) census data
including population demographics, housing characteristics, and socio-economic indicators.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import Field, field_validator, model_validator

from .base_schema import (
    VersionedSchema,
    DataSource,
    SchemaVersion,
    DataQualityLevel,
    GeographicBoundary,
    TemporalData
)


class CensusTableType(str, Enum):
    """Types of ABS census tables."""
    BASIC_DEMOGRAPHICS = "basic_demographics"  # G01
    AGE_SEX = "age_sex"  # G02, G17A
    ANCESTRY = "ancestry"  # G09
    COUNTRY_OF_BIRTH = "country_of_birth"  # G10
    LANGUAGE = "language"  # G11
    RELIGION = "religion"  # G14
    MARITAL_STATUS = "marital_status"  # G13
    EDUCATION = "education"  # G16, G18
    EMPLOYMENT = "employment"  # G17, G43-51
    INCOME = "income"  # G17A-D
    HOUSING = "housing"  # G31-G42
    FAMILY_COMPOSITION = "family_composition"  # G24-G30
    DWELLING_STRUCTURE = "dwelling_structure"  # G32
    TENURE_TYPE = "tenure_type"  # G34
    MORTGAGE_RENT = "mortgage_rent"  # G35-G36


class CensusDemographics(VersionedSchema, TemporalData):
    """Schema for basic census demographic data (Table G01)."""
    
    # Geographic identification
    geographic_id: str = Field(..., description="Geographic area identifier")
    geographic_level: str = Field(..., description="Geographic level (SA1, SA2, etc)")
    geographic_name: str = Field(..., description="Geographic area name")
    state_territory: str = Field(..., description="State or territory")
    
    # Census details
    census_year: int = Field(..., ge=1911, description="Census year")
    table_code: str = Field(default="G01", description="Census table code")
    table_name: str = Field(default="Basic Demographic Profile", description="Table name")
    
    # Basic demographics
    total_population: int = Field(..., ge=0, description="Total usual resident population")
    males: int = Field(..., ge=0, description="Number of males")
    females: int = Field(..., ge=0, description="Number of females")
    
    # Age groups (5-year age groups)
    age_0_4: int = Field(0, ge=0, description="Population aged 0-4 years")
    age_5_9: int = Field(0, ge=0, description="Population aged 5-9 years")
    age_10_14: int = Field(0, ge=0, description="Population aged 10-14 years")
    age_15_19: int = Field(0, ge=0, description="Population aged 15-19 years")
    age_20_24: int = Field(0, ge=0, description="Population aged 20-24 years")
    age_25_29: int = Field(0, ge=0, description="Population aged 25-29 years")
    age_30_34: int = Field(0, ge=0, description="Population aged 30-34 years")
    age_35_39: int = Field(0, ge=0, description="Population aged 35-39 years")
    age_40_44: int = Field(0, ge=0, description="Population aged 40-44 years")
    age_45_49: int = Field(0, ge=0, description="Population aged 45-49 years")
    age_50_54: int = Field(0, ge=0, description="Population aged 50-54 years")
    age_55_59: int = Field(0, ge=0, description="Population aged 55-59 years")
    age_60_64: int = Field(0, ge=0, description="Population aged 60-64 years")
    age_65_69: int = Field(0, ge=0, description="Population aged 65-69 years")
    age_70_74: int = Field(0, ge=0, description="Population aged 70-74 years")
    age_75_79: int = Field(0, ge=0, description="Population aged 75-79 years")
    age_80_84: int = Field(0, ge=0, description="Population aged 80-84 years")
    age_85_plus: int = Field(0, ge=0, description="Population aged 85+ years")
    
    # Indigenous status
    indigenous: int = Field(0, ge=0, description="Aboriginal and Torres Strait Islander population")
    non_indigenous: int = Field(0, ge=0, description="Non-Indigenous population")
    indigenous_not_stated: int = Field(0, ge=0, description="Indigenous status not stated")
    
    # Household and dwelling counts
    total_private_dwellings: int = Field(0, ge=0, description="Total private dwellings")
    occupied_private_dwellings: int = Field(0, ge=0, description="Occupied private dwellings")
    unoccupied_private_dwellings: int = Field(0, ge=0, description="Unoccupied private dwellings")
    total_families: int = Field(0, ge=0, description="Total number of families")
    
    # Data source
    data_source: DataSource = Field(..., description="Source of census data")
    
    @field_validator('census_year')
    @classmethod
    def validate_census_year(cls, v: int) -> int:
        """Validate census year is a valid ABS census year."""
        valid_years = {1911, 1921, 1933, 1947, 1954, 1961, 1966, 1971, 1976, 
                      1981, 1986, 1991, 1996, 2001, 2006, 2011, 2016, 2021, 2026, 2031}
        if v not in valid_years:
            raise ValueError(f"Invalid census year: {v}")
        return v
    
    @field_validator('state_territory')
    @classmethod
    def validate_state_territory(cls, v: str) -> str:
        """Validate Australian state/territory codes."""
        valid_states = {'NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT', 'OT'}
        if v.upper() not in valid_states:
            raise ValueError(f"Invalid state/territory: {v}")
        return v.upper()
    
    @model_validator(mode='after')
    def validate_demographic_consistency(self) -> 'CensusDemographics':
        """Validate demographic data consistency."""
        # Check total population equals males + females
        if self.males + self.females != self.total_population:
            raise ValueError("Males + Females must equal total population")
        
        # Check age groups sum to total (allowing for small discrepancies due to rounding)
        age_total = (self.age_0_4 + self.age_5_9 + self.age_10_14 + self.age_15_19 +
                    self.age_20_24 + self.age_25_29 + self.age_30_34 + self.age_35_39 +
                    self.age_40_44 + self.age_45_49 + self.age_50_54 + self.age_55_59 +
                    self.age_60_64 + self.age_65_69 + self.age_70_74 + self.age_75_79 +
                    self.age_80_84 + self.age_85_plus)
        
        if abs(age_total - self.total_population) > 5:  # Allow small discrepancies
            raise ValueError(f"Age groups sum ({age_total}) doesn't match total population ({self.total_population})")
        
        # Check Indigenous status consistency
        indigenous_total = self.indigenous + self.non_indigenous + self.indigenous_not_stated
        if abs(indigenous_total - self.total_population) > 5:
            raise ValueError("Indigenous status categories don't sum to total population")
        
        # Check dwelling occupancy
        total_calc = self.occupied_private_dwellings + self.unoccupied_private_dwellings
        if self.total_private_dwellings > 0 and abs(total_calc - self.total_private_dwellings) > 2:
            raise ValueError("Occupied + unoccupied dwellings don't match total")
        
        return self
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "CensusDemographics"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate census data integrity."""
        errors = []
        
        # Check for reasonable population density if area is available
        # This would require geographic boundary data
        
        # Check age distribution is realistic
        working_age = (self.age_15_19 + self.age_20_24 + self.age_25_29 + self.age_30_34 +
                      self.age_35_39 + self.age_40_44 + self.age_45_49 + self.age_50_54 +
                      self.age_55_59 + self.age_60_64)
        
        if self.total_population > 0:
            working_age_pct = (working_age / self.total_population) * 100
            if working_age_pct < 30:
                errors.append(f"Working age population unusually low: {working_age_pct:.1f}%")
            elif working_age_pct > 85:
                errors.append(f"Working age population unusually high: {working_age_pct:.1f}%")
        
        # Check gender balance
        if self.total_population > 0:
            male_pct = (self.males / self.total_population) * 100
            if male_pct < 40 or male_pct > 60:
                errors.append(f"Gender balance unusual: {male_pct:.1f}% male")
        
        return errors


class CensusEducation(VersionedSchema, TemporalData):
    """Schema for census education data (Tables G16, G18)."""
    
    # Geographic identification
    geographic_id: str = Field(..., description="Geographic area identifier")
    geographic_level: str = Field(..., description="Geographic level")
    census_year: int = Field(..., description="Census year")
    
    # Education population base
    education_pop_base: int = Field(..., ge=0, description="Population base for education data")
    
    # Highest level of schooling (for population aged 15+)
    year_12_or_equivalent: int = Field(0, ge=0, description="Completed Year 12 or equivalent")
    year_11_or_equivalent: int = Field(0, ge=0, description="Completed Year 11 or equivalent")
    year_10_or_equivalent: int = Field(0, ge=0, description="Completed Year 10 or equivalent")
    year_9_or_equivalent: int = Field(0, ge=0, description="Completed Year 9 or equivalent")
    year_8_or_below: int = Field(0, ge=0, description="Year 8 or below")
    did_not_go_to_school: int = Field(0, ge=0, description="Did not go to school")
    schooling_not_stated: int = Field(0, ge=0, description="Schooling level not stated")
    
    # Non-school qualifications
    postgraduate_degree: int = Field(0, ge=0, description="Postgraduate degree level")
    graduate_diploma: int = Field(0, ge=0, description="Graduate diploma and graduate certificate")
    bachelor_degree: int = Field(0, ge=0, description="Bachelor degree level")
    advanced_diploma: int = Field(0, ge=0, description="Advanced diploma and diploma level")
    certificate_iii_iv: int = Field(0, ge=0, description="Certificate III & IV level")
    certificate_i_ii: int = Field(0, ge=0, description="Certificate I & II level")
    no_qualification: int = Field(0, ge=0, description="No non-school qualification")
    qualification_not_stated: int = Field(0, ge=0, description="Qualification not stated")
    
    # Field of study (top level categories)
    natural_physical_sciences: int = Field(0, ge=0, description="Natural and physical sciences")
    information_technology: int = Field(0, ge=0, description="Information technology")
    engineering: int = Field(0, ge=0, description="Engineering and related technologies")
    architecture_building: int = Field(0, ge=0, description="Architecture and building")
    agriculture: int = Field(0, ge=0, description="Agriculture, environmental and related")
    health: int = Field(0, ge=0, description="Health")
    education: int = Field(0, ge=0, description="Education")
    management_commerce: int = Field(0, ge=0, description="Management and commerce")
    society_culture: int = Field(0, ge=0, description="Society and culture")
    creative_arts: int = Field(0, ge=0, description="Creative arts")
    food_hospitality: int = Field(0, ge=0, description="Food, hospitality and personal services")
    mixed_field: int = Field(0, ge=0, description="Mixed field programmes")
    field_not_stated: int = Field(0, ge=0, description="Field of study not stated")
    
    # Data source
    data_source: DataSource = Field(..., description="Source of education data")
    
    @model_validator(mode='after')
    def validate_education_totals(self) -> 'CensusEducation':
        """Validate education data totals."""
        # Check schooling levels sum approximately to population base
        schooling_total = (self.year_12_or_equivalent + self.year_11_or_equivalent +
                          self.year_10_or_equivalent + self.year_9_or_equivalent +
                          self.year_8_or_below + self.did_not_go_to_school +
                          self.schooling_not_stated)
        
        if abs(schooling_total - self.education_pop_base) > 10:
            raise ValueError("Schooling categories don't sum to population base")
        
        return self
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "CensusEducation"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate education data integrity."""
        errors = []
        
        if self.education_pop_base > 0:
            # Check university qualification rates
            uni_qualified = (self.postgraduate_degree + self.graduate_diploma + 
                           self.bachelor_degree)
            uni_rate = (uni_qualified / self.education_pop_base) * 100
            
            if uni_rate > 80:
                errors.append(f"University qualification rate unusually high: {uni_rate:.1f}%")
            
            # Check Year 12 completion rate
            year12_rate = (self.year_12_or_equivalent / self.education_pop_base) * 100
            if year12_rate > 95:
                errors.append(f"Year 12 completion rate unusually high: {year12_rate:.1f}%")
        
        return errors


class CensusEmployment(VersionedSchema, TemporalData):
    """Schema for census employment data (Tables G17, G43-G51)."""
    
    # Geographic identification
    geographic_id: str = Field(..., description="Geographic area identifier")
    geographic_level: str = Field(..., description="Geographic level")
    census_year: int = Field(..., description="Census year")
    
    # Labour force status (for population aged 15+)
    labour_force_pop: int = Field(..., ge=0, description="Population aged 15+ for labour force")
    employed_full_time: int = Field(0, ge=0, description="Employed full-time")
    employed_part_time: int = Field(0, ge=0, description="Employed part-time")
    unemployed: int = Field(0, ge=0, description="Unemployed")
    not_in_labour_force: int = Field(0, ge=0, description="Not in labour force")
    labour_force_not_stated: int = Field(0, ge=0, description="Labour force status not stated")
    
    # Industry of employment (ANZSIC divisions)
    agriculture_forestry_fishing: int = Field(0, ge=0, description="Agriculture, forestry and fishing")
    mining: int = Field(0, ge=0, description="Mining")
    manufacturing: int = Field(0, ge=0, description="Manufacturing")
    electricity_gas_water: int = Field(0, ge=0, description="Electricity, gas, water and waste services")
    construction: int = Field(0, ge=0, description="Construction")
    wholesale_trade: int = Field(0, ge=0, description="Wholesale trade")
    retail_trade: int = Field(0, ge=0, description="Retail trade")
    accommodation_food: int = Field(0, ge=0, description="Accommodation and food services")
    transport_postal: int = Field(0, ge=0, description="Transport, postal and warehousing")
    information_media: int = Field(0, ge=0, description="Information media and telecommunications")
    financial_insurance: int = Field(0, ge=0, description="Financial and insurance services")
    rental_real_estate: int = Field(0, ge=0, description="Rental, hiring and real estate services")
    professional_services: int = Field(0, ge=0, description="Professional, scientific and technical services")
    administrative_support: int = Field(0, ge=0, description="Administrative and support services")
    public_administration: int = Field(0, ge=0, description="Public administration and safety")
    education_training: int = Field(0, ge=0, description="Education and training")
    health_social_assistance: int = Field(0, ge=0, description="Health care and social assistance")
    arts_recreation: int = Field(0, ge=0, description="Arts and recreation services")
    other_services: int = Field(0, ge=0, description="Other services")
    industry_not_stated: int = Field(0, ge=0, description="Industry not stated")
    
    # Occupation (ANZSCO major groups)
    managers: int = Field(0, ge=0, description="Managers")
    professionals: int = Field(0, ge=0, description="Professionals")
    technicians_trades: int = Field(0, ge=0, description="Technicians and trades workers")
    community_personal_service: int = Field(0, ge=0, description="Community and personal service workers")
    clerical_administrative: int = Field(0, ge=0, description="Clerical and administrative workers")
    sales_workers: int = Field(0, ge=0, description="Sales workers")
    machinery_operators: int = Field(0, ge=0, description="Machinery operators and drivers")
    labourers: int = Field(0, ge=0, description="Labourers")
    occupation_not_stated: int = Field(0, ge=0, description="Occupation not stated")
    
    # Data source
    data_source: DataSource = Field(..., description="Source of employment data")
    
    @model_validator(mode='after')
    def validate_employment_totals(self) -> 'CensusEmployment':
        """Validate employment data consistency."""
        # Check labour force status totals
        lf_total = (self.employed_full_time + self.employed_part_time + self.unemployed +
                   self.not_in_labour_force + self.labour_force_not_stated)
        
        if abs(lf_total - self.labour_force_pop) > 10:
            raise ValueError("Labour force categories don't sum to population base")
        
        return self
    
    def get_unemployment_rate(self) -> Optional[float]:
        """Calculate unemployment rate."""
        total_employed = self.employed_full_time + self.employed_part_time
        total_labour_force = total_employed + self.unemployed
        
        if total_labour_force > 0:
            return (self.unemployed / total_labour_force) * 100
        return None
    
    def get_participation_rate(self) -> Optional[float]:
        """Calculate labour force participation rate."""
        total_labour_force = (self.employed_full_time + self.employed_part_time + 
                             self.unemployed)
        
        if self.labour_force_pop > 0:
            return (total_labour_force / self.labour_force_pop) * 100
        return None
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "CensusEmployment"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate employment data integrity."""
        errors = []
        
        # Check unemployment rate
        unemployment_rate = self.get_unemployment_rate()
        if unemployment_rate and unemployment_rate > 50:
            errors.append(f"Unemployment rate unusually high: {unemployment_rate:.1f}%")
        
        # Check participation rate
        participation_rate = self.get_participation_rate()
        if participation_rate and participation_rate > 95:
            errors.append(f"Participation rate unusually high: {participation_rate:.1f}%")
        
        return errors


class CensusHousing(VersionedSchema, TemporalData):
    """Schema for census housing and dwelling data (Tables G31-G42)."""
    
    # Geographic identification
    geographic_id: str = Field(..., description="Geographic area identifier")
    geographic_level: str = Field(..., description="Geographic level")
    census_year: int = Field(..., description="Census year")
    
    # Dwelling structure
    separate_house: int = Field(0, ge=0, description="Separate house")
    semi_detached: int = Field(0, ge=0, description="Semi-detached, row or terrace house")
    flat_apartment: int = Field(0, ge=0, description="Flat or apartment")
    other_dwelling: int = Field(0, ge=0, description="Other dwelling")
    dwelling_structure_not_stated: int = Field(0, ge=0, description="Dwelling structure not stated")
    
    # Tenure type
    owned_outright: int = Field(0, ge=0, description="Owned outright")
    owned_with_mortgage: int = Field(0, ge=0, description="Owned with a mortgage")
    rented: int = Field(0, ge=0, description="Rented")
    other_tenure: int = Field(0, ge=0, description="Other tenure type")
    tenure_not_stated: int = Field(0, ge=0, description="Tenure type not stated")
    
    # Landlord type (for rented dwellings)
    state_territory_housing: int = Field(0, ge=0, description="State or territory housing authority")
    private_landlord: int = Field(0, ge=0, description="Private landlord")
    real_estate_agent: int = Field(0, ge=0, description="Real estate agent")
    other_landlord: int = Field(0, ge=0, description="Other landlord type")
    landlord_not_stated: int = Field(0, ge=0, description="Landlord type not stated")
    
    # Number of bedrooms
    no_bedrooms: int = Field(0, ge=0, description="No bedrooms (bed-sitters etc)")
    one_bedroom: int = Field(0, ge=0, description="1 bedroom")
    two_bedrooms: int = Field(0, ge=0, description="2 bedrooms")
    three_bedrooms: int = Field(0, ge=0, description="3 bedrooms")
    four_bedrooms: int = Field(0, ge=0, description="4 bedrooms")
    five_plus_bedrooms: int = Field(0, ge=0, description="5 or more bedrooms")
    bedrooms_not_stated: int = Field(0, ge=0, description="Number of bedrooms not stated")
    
    # Mortgage and rent ranges (median values)
    median_mortgage_monthly: Optional[int] = Field(None, ge=0, description="Median monthly mortgage payment")
    median_rent_weekly: Optional[int] = Field(None, ge=0, description="Median weekly rent")
    
    # Internet connection
    internet_connection: int = Field(0, ge=0, description="Dwellings with internet connection")
    no_internet: int = Field(0, ge=0, description="Dwellings without internet")
    internet_not_stated: int = Field(0, ge=0, description="Internet connection not stated")
    
    # Motor vehicles
    no_motor_vehicles: int = Field(0, ge=0, description="No motor vehicles")
    one_motor_vehicle: int = Field(0, ge=0, description="1 motor vehicle")
    two_motor_vehicles: int = Field(0, ge=0, description="2 motor vehicles")
    three_plus_vehicles: int = Field(0, ge=0, description="3 or more motor vehicles")
    vehicles_not_stated: int = Field(0, ge=0, description="Number of vehicles not stated")
    
    # Data source
    data_source: DataSource = Field(..., description="Source of housing data")
    
    def get_home_ownership_rate(self) -> Optional[float]:
        """Calculate home ownership rate."""
        owned_total = self.owned_outright + self.owned_with_mortgage
        total_with_known_tenure = (owned_total + self.rented + self.other_tenure)
        
        if total_with_known_tenure > 0:
            return (owned_total / total_with_known_tenure) * 100
        return None
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "CensusHousing"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate housing data integrity."""
        errors = []
        
        # Check home ownership rate
        ownership_rate = self.get_home_ownership_rate()
        if ownership_rate and ownership_rate > 95:
            errors.append(f"Home ownership rate unusually high: {ownership_rate:.1f}%")
        
        # Check median rent/mortgage consistency
        if (self.median_rent_weekly and self.median_mortgage_monthly and
            self.median_rent_weekly * 4.33 > self.median_mortgage_monthly * 2):
            errors.append("Median rent appears unusually high compared to mortgage")
        
        return errors


class CensusSEIFA(VersionedSchema, TemporalData):
    """Schema for SEIFA socio-economic index data (ABS SEIFA 2021)."""
    
    # Geographic identification
    geographic_id: str = Field(..., description="Geographic area identifier")
    geographic_level: str = Field(..., description="Geographic level (SA1, SA2, etc)")
    geographic_name: str = Field(..., description="Geographic area name")
    state_territory: str = Field(..., description="State or territory")
    census_year: int = Field(..., description="Census year for SEIFA data")
    
    # SEIFA index scores (standard range 600-1400, but allow broader for edge cases)
    irsad_score: Optional[int] = Field(None, ge=1, le=2000, description="Index of Relative Socio-economic Advantage and Disadvantage score")
    irsd_score: Optional[int] = Field(None, ge=1, le=2000, description="Index of Relative Socio-economic Disadvantage score") 
    ier_score: Optional[int] = Field(None, ge=1, le=2000, description="Index of Education and Occupation score")
    ieo_score: Optional[int] = Field(None, ge=1, le=2000, description="Index of Economic Resources score")
    
    # Rankings (1 = most disadvantaged/lowest score)
    irsad_rank: Optional[int] = Field(None, ge=1, description="IRSAD national ranking")
    irsd_rank: Optional[int] = Field(None, ge=1, description="IRSD national ranking")
    ier_rank: Optional[int] = Field(None, ge=1, description="IER national ranking")
    ieo_rank: Optional[int] = Field(None, ge=1, description="IEO national ranking")
    
    # Decile groupings (1=most disadvantaged, 10=most advantaged)
    irsad_decile: Optional[int] = Field(None, ge=1, le=10, description="IRSAD decile")
    irsd_decile: Optional[int] = Field(None, ge=1, le=10, description="IRSD decile")
    ier_decile: Optional[int] = Field(None, ge=1, le=10, description="IER decile")
    ieo_decile: Optional[int] = Field(None, ge=1, le=10, description="IEO decile")
    
    # State-level rankings and deciles
    irsad_state_rank: Optional[int] = Field(None, ge=1, description="IRSAD state ranking")
    irsd_state_rank: Optional[int] = Field(None, ge=1, description="IRSD state ranking")
    irsad_state_decile: Optional[int] = Field(None, ge=1, le=10, description="IRSAD state decile")
    irsd_state_decile: Optional[int] = Field(None, ge=1, le=10, description="IRSD state decile")
    
    # Composite indicators (derived from multiple indices)
    overall_advantage_score: Optional[float] = Field(None, description="Composite advantage score")
    disadvantage_severity: Optional[str] = Field(None, description="Categorical disadvantage level")
    
    # Population base for the geographic area
    population_base: Optional[int] = Field(None, ge=0, description="Total population for SEIFA calculation")
    
    # Data source
    data_source: DataSource = Field(..., description="Source of SEIFA data")
    
    @field_validator('census_year')
    @classmethod
    def validate_seifa_year(cls, v: int) -> int:
        """Validate SEIFA year is a valid ABS SEIFA release year."""
        valid_years = {2001, 2006, 2011, 2016, 2021, 2026}  # SEIFA release years
        if v not in valid_years:
            raise ValueError(f"Invalid SEIFA year: {v}")
        return v
    
    @field_validator('state_territory')
    @classmethod
    def validate_state_territory(cls, v: str) -> str:
        """Validate Australian state/territory codes."""
        valid_states = {'NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT', 'OT'}
        if v.upper() not in valid_states:
            raise ValueError(f"Invalid state/territory: {v}")
        return v.upper()
    
    @field_validator('disadvantage_severity')
    @classmethod
    def validate_disadvantage_severity(cls, v: Optional[str]) -> Optional[str]:
        """Validate disadvantage severity categories."""
        if v is None:
            return v
        valid_categories = {'very_high', 'high', 'moderate', 'low', 'very_low'}
        if v.lower() not in valid_categories:
            raise ValueError(f"Invalid disadvantage severity: {v}")
        return v.lower()
    
    @model_validator(mode='after')
    def validate_seifa_consistency(self) -> 'CensusSEIFA':
        """Validate SEIFA data consistency."""
        # Check that scores and ranks are consistent (lower score = higher rank number)
        if self.irsd_score and self.irsd_rank:
            # IRSD: lower score = more disadvantaged = higher rank number
            # This is a basic sanity check - detailed validation would require full dataset
            if self.irsd_score < 700 and self.irsd_rank < 1000:
                # Very low score should have high rank number
                pass  # This would need full dataset context for proper validation
        
        # Check decile consistency with scores where possible
        if self.irsd_score and self.irsd_decile:
            # Rough validation: very low scores should be in low deciles
            if self.irsd_score < 800 and self.irsd_decile > 5:
                raise ValueError("IRSD score and decile appear inconsistent")
            if self.irsd_score > 1200 and self.irsd_decile < 5:
                raise ValueError("IRSD score and decile appear inconsistent")
        
        # Ensure at least one SEIFA index is present
        indices_present = [self.irsad_score, self.irsd_score, self.ier_score, self.ieo_score]
        if not any(indices_present):
            raise ValueError("At least one SEIFA index score must be provided")
        
        return self
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "CensusSEIFA"
    
    def get_overall_disadvantage_level(self) -> Optional[str]:
        """Calculate overall disadvantage level from available indices."""
        if not self.irsd_decile:
            return None
        
        # Use IRSD as primary disadvantage indicator
        if self.irsd_decile <= 2:
            return "very_high"
        elif self.irsd_decile <= 4:
            return "high"
        elif self.irsd_decile <= 6:
            return "moderate"
        elif self.irsd_decile <= 8:
            return "low"
        else:
            return "very_low"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate SEIFA data integrity."""
        errors = []
        
        # Check for reasonable score ranges
        score_fields = [
            ('irsad_score', self.irsad_score),
            ('irsd_score', self.irsd_score),
            ('ier_score', self.ier_score),
            ('ieo_score', self.ieo_score)
        ]
        
        for field_name, score in score_fields:
            if score and (score < 600 or score > 1400):
                errors.append(f"{field_name} outside typical range (600-1400): {score}")
        
        # Check rank consistency
        rank_fields = [
            ('irsad_rank', self.irsad_rank),
            ('irsd_rank', self.irsd_rank),
            ('ier_rank', self.ier_rank),
            ('ieo_rank', self.ieo_rank)
        ]
        
        for field_name, rank in rank_fields:
            if rank and rank > 50000:  # Rough upper bound for Australian geographic areas
                errors.append(f"{field_name} appears unusually high: {rank}")
        
        return errors


class IntegratedCensusData(VersionedSchema, TemporalData):
    """Unified master census dataset schema integrating all census domains."""
    
    # Universal identifiers (required)
    geographic_id: str = Field(..., description="Geographic area identifier")
    geographic_level: str = Field(..., description="Geographic level (SA1, SA2, etc)")
    geographic_name: str = Field(..., description="Geographic area name") 
    state_territory: str = Field(..., description="State or territory")
    census_year: int = Field(..., description="Census year")
    
    # Geographic coordinates (WGS84)
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0, description="Centroid latitude in decimal degrees (WGS84)")
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0, description="Centroid longitude in decimal degrees (WGS84)")
    
    # Population metrics
    total_population: int = Field(..., ge=0, description="Total resident population")
    working_age_population: Optional[int] = Field(None, ge=0, description="Population aged 15+")
    population_consistency_flag: Optional[bool] = Field(None, description="Population data consistency check")
    
    # Demographics domain (core indicators)
    median_age: Optional[float] = Field(None, ge=0, le=120, description="Median age of population")
    sex_ratio: Optional[float] = Field(None, ge=0, description="Males per 100 females")
    indigenous_percentage: Optional[float] = Field(None, ge=0, le=100, description="Indigenous population %")
    age_dependency_ratio: Optional[float] = Field(None, ge=0, description="Age dependency ratio")
    
    # Housing domain (core indicators) 
    home_ownership_rate: Optional[float] = Field(None, ge=0, le=100, description="Home ownership rate %")
    median_rent_weekly: Optional[int] = Field(None, ge=0, description="Median weekly rent")
    median_mortgage_monthly: Optional[int] = Field(None, ge=0, description="Median monthly mortgage")
    housing_stress_rate: Optional[float] = Field(None, ge=0, le=100, description="Housing stress rate %")
    average_household_size: Optional[float] = Field(None, ge=0, description="Average persons per household")
    
    # Employment domain (core indicators)
    unemployment_rate: Optional[float] = Field(None, ge=0, le=100, description="Unemployment rate %")
    participation_rate: Optional[float] = Field(None, ge=0, le=100, description="Labour force participation %")
    professional_employment_rate: Optional[float] = Field(None, ge=0, le=100, description="Professional occupation %")
    median_personal_income: Optional[int] = Field(None, ge=0, description="Median personal income")
    industry_diversity_index: Optional[float] = Field(None, ge=0, le=1, description="Industry diversity score")
    
    # Education domain (core indicators)
    university_qualification_rate: Optional[float] = Field(None, ge=0, le=100, description="University qualification %")
    year12_completion_rate: Optional[float] = Field(None, ge=0, le=100, description="Year 12 completion %")
    vocational_qualification_rate: Optional[float] = Field(None, ge=0, le=100, description="Vocational qualification %")
    
    # SEIFA domain (core indicators)
    irsad_score: Optional[int] = Field(None, ge=1, le=2000, description="IRSAD advantage/disadvantage score")
    irsd_score: Optional[int] = Field(None, ge=1, le=2000, description="IRSD disadvantage score") 
    irsad_decile: Optional[int] = Field(None, ge=1, le=10, description="IRSAD decile")
    irsd_decile: Optional[int] = Field(None, ge=1, le=10, description="IRSD decile")
    seifa_advantage_category: Optional[str] = Field(None, description="SEIFA advantage category")
    
    # Derived cross-domain insights
    socioeconomic_profile: str = Field(..., description="Overall socioeconomic classification")
    livability_index: Optional[float] = Field(None, ge=0, le=100, description="Composite livability score")
    economic_opportunity_score: Optional[float] = Field(None, ge=0, le=100, description="Economic opportunity rating")
    social_cohesion_index: Optional[float] = Field(None, ge=0, le=100, description="Social cohesion measure")
    housing_market_pressure: Optional[float] = Field(None, ge=0, le=100, description="Housing market pressure indicator")
    
    # Data quality metrics
    data_completeness_score: float = Field(..., ge=0, le=1, description="Data completeness 0-1")
    demographics_completeness: Optional[float] = Field(None, ge=0, le=1, description="Demographics data completeness")
    housing_completeness: Optional[float] = Field(None, ge=0, le=1, description="Housing data completeness")
    employment_completeness: Optional[float] = Field(None, ge=0, le=1, description="Employment data completeness")
    education_completeness: Optional[float] = Field(None, ge=0, le=1, description="Education data completeness")
    seifa_completeness: Optional[float] = Field(None, ge=0, le=1, description="SEIFA data completeness")
    temporal_quality_flag: bool = Field(..., description="Temporal alignment quality")
    consistency_score: float = Field(..., ge=0, le=1, description="Cross-domain consistency")
    
    # Integration metadata
    integration_sk: Optional[int] = Field(None, description="Integration surrogate key")
    source_datasets: List[str] = Field(..., description="Contributing datasets")
    integration_timestamp: Optional[datetime] = Field(None, description="Integration processing time")
    
    # Data source
    data_source: DataSource = Field(..., description="Source of integrated data")
    
    @field_validator('census_year')
    @classmethod
    def validate_census_year(cls, v: int) -> int:
        """Validate census year is a valid ABS census year."""
        valid_years = {1911, 1921, 1933, 1947, 1954, 1961, 1966, 1971, 1976, 
                      1981, 1986, 1991, 1996, 2001, 2006, 2011, 2016, 2021, 2026, 2031}
        if v not in valid_years:
            raise ValueError(f"Invalid census year: {v}")
        return v
    
    @field_validator('state_territory')
    @classmethod
    def validate_state_territory(cls, v: str) -> str:
        """Validate Australian state/territory codes."""
        valid_states = {'NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT', 'OT'}
        if v.upper() not in valid_states:
            raise ValueError(f"Invalid state/territory: {v}")
        return v.upper()
    
    @field_validator('socioeconomic_profile')
    @classmethod
    def validate_socioeconomic_profile(cls, v: str) -> str:
        """Validate socioeconomic profile categories."""
        valid_profiles = {'high', 'medium-high', 'medium-low', 'low'}
        if v.lower() not in valid_profiles:
            raise ValueError(f"Invalid socioeconomic profile: {v}")
        return v.lower()
    
    @field_validator('seifa_advantage_category')
    @classmethod
    def validate_seifa_category(cls, v: Optional[str]) -> Optional[str]:
        """Validate SEIFA advantage categories."""
        if v is None:
            return v
        valid_categories = {'very_high', 'high', 'moderate', 'low', 'very_low'}
        if v.lower() not in valid_categories:
            raise ValueError(f"Invalid SEIFA advantage category: {v}")
        return v.lower()
    
    @field_validator('latitude', 'longitude')
    @classmethod
    def validate_australian_coordinates(cls, v: Optional[float], info) -> Optional[float]:
        """Validate coordinates are within plausible range for Australia."""
        if v is None:
            return v
        
        field_name = info.field_name
        if field_name == 'latitude':
            # Australia's latitude range: approximately -44 to -10 degrees
            if v < -44.0 or v > -10.0:
                raise ValueError(f"Latitude {v} outside Australia's range (-44.0 to -10.0)")
        elif field_name == 'longitude':
            # Australia's longitude range: approximately 113 to 154 degrees  
            if v < 113.0 or v > 154.0:
                raise ValueError(f"Longitude {v} outside Australia's range (113.0 to 154.0)")
        
        return v
    
    @model_validator(mode='after')
    def validate_integrated_consistency(self) -> 'IntegratedCensusData':
        """Validate integrated data consistency."""
        # Check working age population consistency
        if (self.working_age_population and self.total_population and 
            self.working_age_population > self.total_population):
            raise ValueError("Working age population cannot exceed total population")
        
        # Check percentage fields are valid
        percentage_fields = [
            ('indigenous_percentage', self.indigenous_percentage),
            ('home_ownership_rate', self.home_ownership_rate),
            ('unemployment_rate', self.unemployment_rate),
            ('participation_rate', self.participation_rate),
            ('university_qualification_rate', self.university_qualification_rate)
        ]
        
        for field_name, value in percentage_fields:
            if value is not None and (value < 0 or value > 100):
                raise ValueError(f"{field_name} must be between 0 and 100")
        
        # Validate completeness scores sum
        if self.data_completeness_score > 0:
            domain_scores = [
                self.demographics_completeness or 0,
                self.housing_completeness or 0,
                self.employment_completeness or 0,
                self.education_completeness or 0,
                self.seifa_completeness or 0
            ]
            # Allow some tolerance for rounding
            calculated_avg = sum(domain_scores) / len(domain_scores)
            if abs(calculated_avg - self.data_completeness_score) > 0.01:
                raise ValueError("Data completeness score inconsistent with domain scores")
        
        return self
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "IntegratedCensusData"
    
    def get_data_quality_assessment(self) -> str:
        """Assess overall data quality level."""
        if self.data_completeness_score >= 0.9 and self.consistency_score >= 0.9:
            return "high"
        elif self.data_completeness_score >= 0.7 and self.consistency_score >= 0.7:
            return "medium"
        else:
            return "low"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate integrated data integrity."""
        errors = []
        
        # Check critical fields
        if self.total_population == 0:
            errors.append("Total population is zero")
        
        # Check data completeness
        if self.data_completeness_score < 0.5:
            errors.append(f"Data completeness below threshold: {self.data_completeness_score:.2f}")
        
        # Check temporal quality
        if not self.temporal_quality_flag:
            errors.append("Temporal alignment issues detected")
        
        # Check consistency
        if self.consistency_score < 0.7:
            errors.append(f"Cross-domain consistency below threshold: {self.consistency_score:.2f}")
        
        # Check derived indicators
        if self.livability_index and self.livability_index < 0:
            errors.append("Invalid livability index calculation")
        
        return errors


# Migration functions for census schemas

def migrate_census_v1_to_v2(old_data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate census data from v1.0.0 to v2.0.0."""
    new_data = old_data.copy()
    
    # Example migration: standardise field names
    if 'area_id' in old_data:
        new_data['geographic_id'] = old_data.pop('area_id')
    
    if 'area_name' in old_data:
        new_data['geographic_name'] = old_data.pop('area_name')
    
    # Convert old age group structure if needed
    if 'children' in old_data and 'adults' in old_data:
        # Split into proper age groups
        children = old_data.pop('children', 0)
        adults = old_data.pop('adults', 0)
        
        # Rough distribution (would need better logic in practice)
        new_data['age_0_4'] = int(children * 0.25)
        new_data['age_5_9'] = int(children * 0.25)
        new_data['age_10_14'] = int(children * 0.25)
        new_data['age_15_19'] = int(children * 0.25)
        new_data['age_20_24'] = int(adults * 0.15)
        # ... etc
    
    # Update schema version
    new_data['schema_version'] = SchemaVersion.V2_0_0.value
    
    return new_data