"""
SEIFA (Socio-Economic Indexes for Areas) data schema for AHGD.

This module defines schemas for SEIFA socio-economic data including
index scores, rankings, and decile/quintile classifications.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from pydantic import Field, field_validator, model_validator

from .base_schema import (
    VersionedSchema,
    DataSource,
    SchemaVersion,
    DataQualityLevel
)


class SEIFAIndexType(str, Enum):
    """Types of SEIFA indexes."""
    IRSD = "IRSD"  # Index of Relative Socio-economic Disadvantage
    IRSAD = "IRSAD"  # Index of Relative Socio-economic Advantage and Disadvantage
    IER = "IER"  # Index of Economic Resources
    IEO = "IEO"  # Index of Education and Occupation


class SEIFAScore(VersionedSchema):
    """Schema for SEIFA index scores and rankings."""
    
    # Geographic identifier
    geographic_id: str = Field(..., description="Geographic area identifier (SA1, SA2, etc)")
    geographic_level: str = Field(..., description="Geographic level")
    geographic_name: str = Field(..., description="Geographic area name")
    
    # Index details
    index_type: SEIFAIndexType = Field(..., description="Type of SEIFA index")
    reference_year: int = Field(..., ge=1986, description="Census year for SEIFA data")
    
    # Score and rankings
    score: float = Field(..., description="SEIFA score (standardised to mean 1000)")
    
    # National rankings
    national_rank: int = Field(..., ge=1, description="National rank (1 = most disadvantaged)")
    national_decile: int = Field(..., ge=1, le=10, description="National decile")
    national_quintile: int = Field(..., ge=1, le=5, description="National quintile")
    national_percentile: float = Field(..., ge=0, le=100, description="National percentile")
    
    # State rankings
    state_rank: int = Field(..., ge=1, description="State rank")
    state_decile: int = Field(..., ge=1, le=10, description="State decile")
    state_code: str = Field(..., description="State/territory code")
    
    # Population and validity
    usual_resident_population: Optional[int] = Field(
        None,
        ge=0,
        description="Usual resident population"
    )
    score_reliability: Optional[str] = Field(None, description="Score reliability indicator")
    excluded: bool = Field(False, description="Whether area is excluded from rankings")
    exclusion_reason: Optional[str] = Field(None, description="Reason for exclusion if applicable")
    
    # Data source
    data_source: DataSource = Field(..., description="Source of SEIFA data")
    
    @field_validator('score')
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Validate SEIFA score is within expected range."""
        # SEIFA scores typically range from 600-1200 (standardised to mean 1000)
        if v < 400 or v > 1400:
            raise ValueError(f"SEIFA score {v} outside expected range (400-1400)")
        return v
    
    @field_validator('reference_year')
    @classmethod
    def validate_census_year(cls, v: int) -> int:
        """Validate reference year is a census year."""
        census_years = {1986, 1991, 1996, 2001, 2006, 2011, 2016, 2021, 2026}
        if v not in census_years:
            raise ValueError(f"Reference year {v} is not a census year")
        return v
    
    @model_validator(mode='after')
    def validate_seifa_consistency(self) -> 'SEIFAScore':
        """Validate SEIFA data consistency."""
        # Validate percentile consistency with decile
        decile = self.national_decile
        percentile = self.national_percentile
        expected_min = (decile - 1) * 10
        expected_max = decile * 10
        if not (expected_min < percentile <= expected_max):
            raise ValueError(
                f"Percentile {percentile} inconsistent with decile {decile} "
                f"(expected {expected_min}-{expected_max})"
            )
        
        # Validate exclusion status
        if self.excluded:
            if not self.exclusion_reason:
                raise ValueError("Excluded areas must have an exclusion reason")
            # Excluded areas might have null rankings
            if self.national_rank is not None:
                # Log warning - excluded but has rank
                pass
        
        # Validate quintile/decile consistency
        national_quintile = self.national_quintile
        if decile and national_quintile:
            # Decile 1-2 = Quintile 1, 3-4 = Quintile 2, etc.
            expected_quintile = ((decile - 1) // 2) + 1
            if national_quintile != expected_quintile:
                raise ValueError(
                    f"Quintile {national_quintile} inconsistent with "
                    f"decile {decile} (expected {expected_quintile})"
                )
        
        return self
    
    def get_disadvantage_category(self) -> str:
        """
        Get human-readable disadvantage category based on decile.
        
        Returns:
            Category string (e.g., "Most Disadvantaged", "Least Disadvantaged")
        """
        if self.excluded:
            return "Excluded"
            
        if self.index_type == SEIFAIndexType.IRSD:
            # For IRSD, lower scores = more disadvantaged
            if self.national_decile <= 2:
                return "Most Disadvantaged"
            elif self.national_decile <= 4:
                return "Disadvantaged"
            elif self.national_decile <= 6:
                return "Middle"
            elif self.national_decile <= 8:
                return "Advantaged"
            else:
                return "Least Disadvantaged"
        else:
            # For other indexes, interpret differently
            if self.national_decile <= 2:
                return "Lowest"
            elif self.national_decile <= 4:
                return "Low"
            elif self.national_decile <= 6:
                return "Middle"
            elif self.national_decile <= 8:
                return "High"
            else:
                return "Highest"
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "SEIFAScore"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate SEIFA data integrity."""
        errors = []
        
        # Check score standardisation
        if self.index_type == SEIFAIndexType.IRSD:
            # IRSD typically has mean ~1000, SD ~100
            if abs(self.score - 1000) > 400:
                errors.append(f"IRSD score {self.score} unusually far from mean")
                
        # Validate population consistency
        if self.usual_resident_population == 0 and not self.excluded:
            errors.append("Non-excluded area has zero population")
            
        # Check ranking bounds
        if self.state_rank > self.national_rank:
            # State rank can't be worse than national rank
            errors.append("State rank cannot be higher than national rank")
            
        return errors
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "geographic_id": "101021007",
                "geographic_level": "SA2",
                "geographic_name": "Sydney - Haymarket - The Rocks",
                "index_type": "IRSAD",
                "reference_year": 2021,
                "score": 1124.5,
                "national_rank": 2145,
                "national_decile": 9,
                "national_quintile": 5,
                "national_percentile": 87.3,
                "state_rank": 412,
                "state_decile": 8,
                "state_code": "NSW",
                "usual_resident_population": 15234
            }
        }
    }


class SEIFAComponent(VersionedSchema):
    """Schema for SEIFA index component variables."""
    
    # Geographic and index identification
    geographic_id: str = Field(..., description="Geographic area identifier")
    index_type: SEIFAIndexType = Field(..., description="SEIFA index type")
    reference_year: int = Field(..., description="Census year")
    
    # Component details
    variable_name: str = Field(..., description="Name of component variable")
    variable_code: str = Field(..., description="Census variable code")
    variable_description: str = Field(..., description="Description of what variable measures")
    
    # Values and weights
    raw_value: float = Field(..., description="Raw variable value")
    standardised_value: float = Field(..., description="Standardised value")
    weight: float = Field(..., description="Weight in index calculation")
    contribution: float = Field(..., description="Contribution to final score")
    
    # Direction
    positive_indicator: bool = Field(
        ...,
        description="Whether higher values indicate advantage"
    )
    
    @field_validator('weight')
    @classmethod
    def validate_weight_range(cls, v: float) -> float:
        """Validate component weight is reasonable."""
        if abs(v) > 1:
            raise ValueError(f"Component weight {v} unusually large")
        return v
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "SEIFAComponent"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate component data."""
        errors = []
        
        # Check contribution calculation
        expected_contribution = self.standardised_value * self.weight
        if abs(self.contribution - expected_contribution) > 0.01:
            errors.append("Contribution doesn't match standardised value * weight")
            
        return errors


class SEIFAComparison(VersionedSchema):
    """Schema for SEIFA comparisons between areas or time periods."""
    
    # Comparison identifiers
    comparison_type: str = Field(..., description="Type of comparison (temporal, spatial)")
    
    # Areas being compared
    area_1_id: str = Field(..., description="First area identifier")
    area_1_name: str = Field(..., description="First area name")
    area_2_id: str = Field(..., description="Second area identifier")
    area_2_name: str = Field(..., description="Second area name")
    
    # Index details
    index_type: SEIFAIndexType = Field(..., description="SEIFA index type")
    
    # Comparison values
    area_1_score: float = Field(..., description="First area score")
    area_2_score: float = Field(..., description="Second area score")
    score_difference: float = Field(..., description="Score difference (area_2 - area_1)")
    
    area_1_decile: int = Field(..., ge=1, le=10, description="First area decile")
    area_2_decile: int = Field(..., ge=1, le=10, description="Second area decile")
    decile_change: int = Field(..., description="Decile change")
    
    # Statistical significance
    significant_change: Optional[bool] = Field(
        None,
        description="Whether change is statistically significant"
    )
    confidence_level: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Confidence level %"
    )
    
    # Temporal comparison fields
    time_period_1: Optional[int] = Field(None, description="First time period (year)")
    time_period_2: Optional[int] = Field(None, description="Second time period (year)")
    
    @field_validator('comparison_type')
    @classmethod
    def validate_comparison_type(cls, v: str) -> str:
        """Validate comparison type."""
        valid_types = {'temporal', 'spatial', 'cohort'}
        if v.lower() not in valid_types:
            raise ValueError(f"Invalid comparison type: {v}")
        return v.lower()
    
    @model_validator(mode='after')
    def calculate_differences(self) -> 'SEIFAComparison':
        """Calculate and validate difference fields."""
        # Calculate and validate score difference
        calc_diff = self.area_2_score - self.area_1_score
        if abs(self.score_difference - calc_diff) > 0.01:
            raise ValueError("Score difference doesn't match calculated value")
                
        # Calculate and validate decile change
        calc_change = self.area_2_decile - self.area_1_decile
        if self.decile_change != calc_change:
            raise ValueError("Decile change doesn't match calculated value")
                
        return self
    
    def get_improvement_direction(self) -> str:
        """
        Determine direction of change (improvement/decline).
        
        Returns:
            Direction string based on index type and change
        """
        if self.score_difference == 0:
            return "No Change"
            
        if self.index_type == SEIFAIndexType.IRSD:
            # For IRSD, higher scores = less disadvantaged = improvement
            return "Improvement" if self.score_difference > 0 else "Decline"
        else:
            # For other indexes, higher = better
            return "Improvement" if self.score_difference > 0 else "Decline"
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "SEIFAComparison"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate comparison data."""
        errors = []
        
        # Check temporal consistency
        if self.comparison_type == 'temporal':
            if not (self.time_period_1 and self.time_period_2):
                errors.append("Temporal comparison missing time periods")
            elif self.time_period_1 >= self.time_period_2:
                errors.append("Time period 2 must be after time period 1")
                
        # Validate significance claim
        if self.significant_change and not self.confidence_level:
            errors.append("Significant change claimed but no confidence level provided")
            
        return errors


class SEIFAAggregate(VersionedSchema):
    """Schema for aggregated SEIFA statistics."""
    
    # Aggregation level
    aggregation_level: str = Field(..., description="Level of aggregation (state, national)")
    aggregation_id: str = Field(..., description="Identifier for aggregation area")
    aggregation_name: str = Field(..., description="Name of aggregation area")
    
    # Index and time
    index_type: SEIFAIndexType = Field(..., description="SEIFA index type")
    reference_year: int = Field(..., description="Census year")
    
    # Summary statistics
    area_count: int = Field(..., ge=0, description="Number of areas included")
    population_total: int = Field(..., ge=0, description="Total population covered")
    
    # Score statistics
    mean_score: float = Field(..., description="Mean SEIFA score")
    median_score: float = Field(..., description="Median SEIFA score")
    std_dev_score: float = Field(..., ge=0, description="Standard deviation of scores")
    min_score: float = Field(..., description="Minimum score")
    max_score: float = Field(..., description="Maximum score")
    
    # Distribution
    decile_distribution: Dict[int, int] = Field(
        ...,
        description="Count of areas in each decile"
    )
    quintile_distribution: Dict[int, int] = Field(
        ...,
        description="Count of areas in each quintile"
    )
    
    # Inequality measures
    gini_coefficient: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Gini coefficient of inequality"
    )
    percentile_ratio_90_10: Optional[float] = Field(
        None,
        gt=0,
        description="90th/10th percentile ratio"
    )
    
    @field_validator('decile_distribution')
    @classmethod
    def validate_decile_distribution(cls, v: Dict[int, int]) -> Dict[int, int]:
        """Validate decile distribution completeness."""
        for decile in range(1, 11):
            if decile not in v:
                raise ValueError(f"Missing decile {decile} in distribution")
        return v
    
    @model_validator(mode='after')
    def validate_distribution_totals(self) -> 'SEIFAAggregate':
        """Validate distribution totals match area count."""
        decile_dist = self.decile_distribution
        
        if decile_dist:
            total_in_deciles = sum(decile_dist.values())
            if total_in_deciles != self.area_count:
                raise ValueError(
                    f"Decile distribution total ({total_in_deciles}) "
                    f"doesn't match area count ({self.area_count})"
                )
                
        return self
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "SEIFAAggregate"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate aggregate data integrity."""
        errors = []
        
        # Check score bounds
        if self.min_score > self.mean_score or self.max_score < self.mean_score:
            errors.append("Mean score outside min/max bounds")
            
        if self.min_score > self.median_score or self.max_score < self.median_score:
            errors.append("Median score outside min/max bounds")
            
        # Validate statistical consistency
        if self.std_dev_score == 0 and self.min_score != self.max_score:
            errors.append("Zero standard deviation but different min/max scores")
            
        return errors


# Migration functions for SEIFA schemas

def migrate_seifa_v1_to_v2(old_data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate SEIFA data from v1.0.0 to v2.0.0."""
    new_data = old_data.copy()
    
    # Example migration: rename fields
    if 'area_id' in old_data:
        new_data['geographic_id'] = old_data.pop('area_id')
        
    # Convert old ranking system
    if 'rank' in old_data and 'total_areas' in old_data:
        # Calculate percentile from rank
        rank = old_data['rank']
        total = old_data.pop('total_areas')
        new_data['national_percentile'] = ((total - rank + 1) / total) * 100
        
    # Update schema version
    new_data['schema_version'] = SchemaVersion.V2_0_0.value
    
    return new_data