"""
SEIFA Socio-Economic Data Models

Pydantic models for the Australian Bureau of Statistics Socio-Economic 
Indexes for Areas (SEIFA) data, supporting all four indexes at SA1 and SA2 levels.
"""

from typing import Optional, Union
from decimal import Decimal
from enum import Enum

from pydantic import Field, validator
from pydantic.types import constr, confloat, conint

from .base import GeographicModel, DataQualityMixin, PopulationMixin


class SEIFAIndexType(str, Enum):
    """SEIFA index types."""
    IRSAD = "IRSAD"  # Index of Relative Socio-economic Advantage and Disadvantage  
    IRSD = "IRSD"    # Index of Relative Socio-economic Disadvantage
    IER = "IER"      # Index of Education and Occupation
    IEO = "IEO"      # Index of Economic Resources


class GeographicLevel(str, Enum):
    """Geographic aggregation levels for SEIFA data."""
    SA1 = "SA1"      # Statistical Area Level 1 (~61,845 areas)
    SA2 = "SA2"      # Statistical Area Level 2 (~2,400 areas) 
    SA3 = "SA3"      # Statistical Area Level 3 (~358 areas)
    SA4 = "SA4"      # Statistical Area Level 4 (~107 areas)
    LGA = "LGA"      # Local Government Areas
    STATE = "STATE"  # States and Territories


class SEIFAIndex(GeographicModel, PopulationMixin, DataQualityMixin):
    """
    Individual SEIFA index score for a specific geographic area.
    
    Represents one of the four SEIFA indexes (IRSAD, IRSD, IER, IEO) 
    calculated for a particular geographic area.
    """
    
    # Index identification
    index_type: SEIFAIndexType = Field(
        ...,
        description="Type of SEIFA index"
    )
    
    geographic_level: GeographicLevel = Field(
        ...,
        description="Geographic aggregation level"
    )
    
    # Index values  
    index_score: confloat(ge=0.0) = Field(
        ...,
        description="SEIFA index score (higher = more advantaged, except IRSD where higher = more disadvantaged)"
    )
    
    # Rankings (lower rank = more disadvantaged)
    rank_australia: conint(ge=1) = Field(
        ...,
        description="Rank within Australia (1 = most disadvantaged)"
    )
    
    rank_state: Optional[conint(ge=1)] = Field(
        None,
        description="Rank within state/territory"
    )
    
    # Percentiles (0-100, higher = more advantaged except IRSD)
    percentile_australia: confloat(ge=0.0, le=100.0) = Field(
        ...,
        description="Percentile ranking within Australia"
    )
    
    percentile_state: Optional[confloat(ge=0.0, le=100.0)] = Field(
        None,
        description="Percentile ranking within state/territory"
    )
    
    # Deciles (1-10, higher = more advantaged except IRSD)
    decile_australia: conint(ge=1, le=10) = Field(
        ...,
        description="Decile ranking within Australia (1-10)"
    )
    
    decile_state: Optional[conint(ge=1, le=10)] = Field(
        None,
        description="Decile ranking within state/territory"
    )
    
    # Statistical measures
    standard_error: Optional[confloat(ge=0.0)] = Field(
        None,
        description="Standard error of the index score"
    )
    
    confidence_interval_lower: Optional[float] = Field(
        None,
        description="Lower bound of 95% confidence interval"
    )
    
    confidence_interval_upper: Optional[float] = Field(
        None, 
        description="Upper bound of 95% confidence interval"
    )
    
    # Index composition (for transparency)
    variable_count: Optional[conint(ge=1)] = Field(
        None,
        description="Number of variables used to calculate this index"
    )
    
    missing_variables: Optional[conint(ge=0)] = Field(
        None,
        description="Number of variables with missing data"
    )
    
    @validator('rank_australia')
    def validate_rank_bounds(cls, v, values):
        """Validate rank is within expected bounds for geographic level."""
        geographic_level = values.get('geographic_level')
        
        # Approximate maximum ranks by geographic level (2021 data)
        max_ranks = {
            GeographicLevel.SA1: 62000,
            GeographicLevel.SA2: 2500,
            GeographicLevel.SA3: 360, 
            GeographicLevel.SA4: 110,
            GeographicLevel.LGA: 600,
            GeographicLevel.STATE: 8
        }
        
        if geographic_level and geographic_level in max_ranks:
            max_rank = max_ranks[geographic_level]
            if v > max_rank:
                raise ValueError(f"Rank {v} exceeds maximum expected for {geographic_level.value} (~{max_rank})")
        
        return v
    
    @validator('decile_australia', 'decile_state')
    def validate_decile_range(cls, v):
        """Ensure decile is 1-10."""
        if v < 1 or v > 10:
            raise ValueError("Decile must be between 1 and 10")
        return v
    
    @validator('percentile_australia', 'percentile_state')
    def validate_percentile_range(cls, v):
        """Ensure percentile is 0-100.""" 
        if v is not None and (v < 0 or v > 100):
            raise ValueError("Percentile must be between 0 and 100")
        return v


class SEIFARecord(GeographicModel, PopulationMixin, DataQualityMixin):
    """
    Complete SEIFA record with all four indexes for a geographic area.
    
    Consolidates IRSAD, IRSD, IER, and IEO indexes into a single record
    for efficient storage and analysis.
    """
    
    geographic_level: GeographicLevel = Field(
        ...,
        description="Geographic aggregation level"
    )
    
    # IRSAD - Index of Relative Socio-economic Advantage and Disadvantage
    irsad_score: Optional[confloat(ge=0.0)] = Field(
        None,
        description="IRSAD score (higher = more advantaged)"
    )
    
    irsad_rank_australia: Optional[conint(ge=1)] = Field(
        None,
        description="IRSAD national rank"
    )
    
    irsad_decile_australia: Optional[conint(ge=1, le=10)] = Field(
        None,
        description="IRSAD national decile"
    )
    
    irsad_percentile_australia: Optional[confloat(ge=0.0, le=100.0)] = Field(
        None,
        description="IRSAD national percentile"
    )
    
    # IRSD - Index of Relative Socio-economic Disadvantage  
    irsd_score: Optional[confloat(ge=0.0)] = Field(
        None,
        description="IRSD score (higher = more disadvantaged)"
    )
    
    irsd_rank_australia: Optional[conint(ge=1)] = Field(
        None,
        description="IRSD national rank"
    )
    
    irsd_decile_australia: Optional[conint(ge=1, le=10)] = Field(
        None,
        description="IRSD national decile"
    )
    
    irsd_percentile_australia: Optional[confloat(ge=0.0, le=100.0)] = Field(
        None,
        description="IRSD national percentile"
    )
    
    # IER - Index of Education and Occupation
    ier_score: Optional[confloat(ge=0.0)] = Field(
        None,
        description="IER score (higher = more advantaged)"
    )
    
    ier_rank_australia: Optional[conint(ge=1)] = Field(
        None,
        description="IER national rank"
    )
    
    ier_decile_australia: Optional[conint(ge=1, le=10)] = Field(
        None,
        description="IER national decile"
    )
    
    ier_percentile_australia: Optional[confloat(ge=0.0, le=100.0)] = Field(
        None,
        description="IER national percentile"
    )
    
    # IEO - Index of Economic Resources
    ieo_score: Optional[confloat(ge=0.0)] = Field(
        None,
        description="IEO score (higher = more advantaged)"
    )
    
    ieo_rank_australia: Optional[conint(ge=1)] = Field(
        None,
        description="IEO national rank"
    )
    
    ieo_decile_australia: Optional[conint(ge=1, le=10)] = Field(
        None,
        description="IEO national decile"
    )
    
    ieo_percentile_australia: Optional[confloat(ge=0.0, le=100.0)] = Field(
        None,
        description="IEO national percentile"
    )
    
    # Composite indicators
    overall_advantage_score: Optional[confloat(ge=0.0, le=1.0)] = Field(
        None,
        description="Composite advantage score derived from all indexes"
    )
    
    disadvantage_category: Optional[str] = Field(
        None,
        pattern=r"^(very_high|high|moderate|low|very_low)$",
        description="Overall disadvantage category"
    )
    
    # Data quality indicators
    complete_indexes_count: conint(ge=0, le=4) = Field(
        0,
        description="Number of SEIFA indexes available for this area"
    )
    
    primary_index_used: Optional[SEIFAIndexType] = Field(
        None,
        description="Primary index used for analysis when not all are available"
    )
    
    @validator('complete_indexes_count')
    def validate_index_completeness(cls, v, values):
        """Validate that complete_indexes_count matches available data."""
        # Count non-None index scores
        score_fields = ['irsad_score', 'irsd_score', 'ier_score', 'ieo_score']
        actual_count = sum(1 for field in score_fields if values.get(field) is not None)
        
        if v != actual_count:
            raise ValueError(f"complete_indexes_count ({v}) doesn't match actual available indexes ({actual_count})")
        
        return v
    
    def get_primary_disadvantage_indicator(self) -> Optional[float]:
        """
        Get the primary disadvantage indicator (IRSD score) for analysis.
        
        Returns the IRSD score as the standard disadvantage measure,
        or None if not available.
        """
        return self.irsd_score
    
    def get_advantage_indicators(self) -> dict[str, Optional[float]]:
        """
        Get all advantage indicators as a dictionary.
        
        Returns all available SEIFA scores with their index types.
        """
        return {
            'irsad': self.irsad_score,
            'irsd': self.irsd_score,
            'ier': self.ier_score, 
            'ieo': self.ieo_score
        }
    
    def calculate_composite_disadvantage(self) -> Optional[float]:
        """
        Calculate a composite disadvantage score from available indexes.
        
        Uses weighted average of standardised index scores where available.
        """
        scores = []
        weights = {'irsad': 0.3, 'irsd': 0.4, 'ier': 0.2, 'ieo': 0.1}
        
        if self.irsad_percentile_australia:
            scores.append((self.irsad_percentile_australia, weights['irsad']))
        if self.irsd_percentile_australia:
            # IRSD is inverted (lower percentile = more disadvantaged)
            scores.append((100 - self.irsd_percentile_australia, weights['irsd']))
        if self.ier_percentile_australia:
            scores.append((self.ier_percentile_australia, weights['ier']))
        if self.ieo_percentile_australia:
            scores.append((self.ieo_percentile_australia, weights['ieo']))
        
        if not scores:
            return None
        
        # Calculate weighted average
        total_score = sum(score * weight for score, weight in scores)
        total_weight = sum(weight for _, weight in scores)
        
        return total_score / total_weight if total_weight > 0 else None