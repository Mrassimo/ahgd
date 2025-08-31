"""
Geographic Data Models for Australian Statistical Areas

Provides Pydantic models for SA1 and SA2 boundary data with full validation
and support for the Australian Statistical Geography Standard (ASGS).
"""

from typing import Optional, Union, Any, Dict, List
from decimal import Decimal
from enum import Enum

from pydantic import Field, validator
from pydantic.types import constr

from .base import GeographicModel, DataQualityMixin, PopulationMixin


class CoordinateSystem(str, Enum):
    """Supported Australian coordinate systems."""
    GDA2020 = "GDA2020"  # Modern Australian standard
    GDA94 = "GDA94"      # Legacy Australian standard
    WGS84 = "WGS84"      # Global standard


class ChangeType(str, Enum):
    """ABS change types for statistical areas."""
    NO_CHANGE = "0"
    NEW_AREA = "1"
    BOUNDARY_CHANGE = "2"
    CODE_CHANGE = "3"
    NAME_CHANGE = "4"
    SPLIT = "5"
    MERGE = "6"
    ABOLISHED = "7"


class SA1Boundary(GeographicModel, PopulationMixin, DataQualityMixin):
    """
    Statistical Area Level 1 (SA1) boundary model.
    
    SA1s are the smallest geographic unit in the ASGS, with populations
    of 200-800 people. There are ~61,845 SA1s across Australia.
    """
    
    # SA1-specific identifiers (11 digit codes)
    sa1_code: constr(
        pattern=r"^[1-8][0-9]{10}$", 
        min_length=11, 
        max_length=11
    ) = Field(
        ...,
        description="11-digit SA1 code",
        examples=["10102100701"]
    )
    
    sa1_name: constr(min_length=1, max_length=100) = Field(
        ...,
        description="SA1 name (often numeric or descriptive)"
    )
    
    # Hierarchical relationships
    sa2_code: constr(
        pattern=r"^[1-8][0-9]{8}$", 
        min_length=9, 
        max_length=9
    ) = Field(
        ...,
        description="Parent SA2 code (9 digits)",
        examples=["101021007"]
    )
    
    sa3_code: constr(
        pattern=r"^[1-8][0-9]{4}$", 
        min_length=5, 
        max_length=5
    ) = Field(
        ...,
        description="SA3 code (5 digits)",
        examples=["10102"]
    )
    
    sa3_name: Optional[str] = Field(
        None,
        description="SA3 name"
    )
    
    sa4_code: constr(
        pattern=r"^[1-8][0-9]{2}$", 
        min_length=3, 
        max_length=3
    ) = Field(
        ...,
        description="SA4 code (3 digits)",
        examples=["101"]
    )
    
    sa4_name: Optional[str] = Field(
        None,
        description="SA4 name"
    )
    
    # Change tracking
    change_flag: ChangeType = Field(
        ...,
        description="ABS change flag indicating modifications from previous census"
    )
    
    change_label: Optional[str] = Field(
        None,
        description="Description of changes made to this area"
    )
    
    # Coordinate system
    coordinate_system: CoordinateSystem = Field(
        CoordinateSystem.GDA2020,
        description="Coordinate reference system used for geometry"
    )
    
    # Geometry (stored as WKT or WKB)
    geometry_wkt: Optional[str] = Field(
        None,
        description="Well-Known Text representation of boundary polygon"
    )
    
    geometry_wkb: Optional[bytes] = Field(
        None,
        description="Well-Known Binary representation of boundary polygon"  
    )
    
    # Centroid coordinates
    centroid_longitude: Optional[Decimal] = Field(
        None,
        ge=-180,
        le=180,
        description="Longitude of area centroid"
    )
    
    centroid_latitude: Optional[Decimal] = Field(
        None,
        ge=-90, 
        le=90,
        description="Latitude of area centroid"
    )
    
    @validator('geographic_code')
    def sync_geographic_code_with_sa1(cls, v, values):
        """Ensure geographic_code matches sa1_code."""
        sa1_code = values.get('sa1_code')
        if sa1_code and v != sa1_code:
            raise ValueError("geographic_code must match sa1_code for SA1 boundaries")
        return v


class SA2Boundary(GeographicModel, PopulationMixin, DataQualityMixin):
    """
    Statistical Area Level 2 (SA2) boundary model.
    
    SA2s represent communities of 3,000-25,000 people. There are ~2,400 SA2s 
    across Australia, each containing multiple SA1s.
    """
    
    # SA2-specific identifiers (9 digit codes)
    sa2_code: constr(
        pattern=r"^[1-8][0-9]{8}$", 
        min_length=9, 
        max_length=9
    ) = Field(
        ...,
        description="9-digit SA2 code",
        examples=["101021007"]
    )
    
    sa2_name: constr(min_length=1, max_length=100) = Field(
        ...,
        description="SA2 name (suburb or locality based)"
    )
    
    # Hierarchical relationships
    sa3_code: constr(
        pattern=r"^[1-8][0-9]{4}$", 
        min_length=5, 
        max_length=5
    ) = Field(
        ...,
        description="Parent SA3 code",
        examples=["10102"]
    )
    
    sa3_name: Optional[str] = Field(
        None,
        description="SA3 name"
    )
    
    sa4_code: constr(
        pattern=r"^[1-8][0-9]{2}$", 
        min_length=3, 
        max_length=3
    ) = Field(
        ...,
        description="Parent SA4 code",
        examples=["101"]
    )
    
    sa4_name: Optional[str] = Field(
        None,
        description="SA4 name"
    )
    
    # Greater Capital City Statistical Area
    gcc_code: Optional[constr(pattern=r"^[1-8](GCCSA|REST)$")] = Field(
        None,
        description="Greater Capital City Statistical Area code",
        examples=["1GCCSA", "1REST"]
    )
    
    gcc_name: Optional[str] = Field(
        None,
        description="Greater Capital City Statistical Area name"
    )
    
    # Change tracking
    change_flag: ChangeType = Field(
        ...,
        description="ABS change flag"
    )
    
    change_label: Optional[str] = Field(
        None,
        description="Description of changes"
    )
    
    # Child SA1 tracking
    sa1_count: Optional[int] = Field(
        None,
        ge=1,
        description="Number of SA1s contained in this SA2"
    )
    
    # Coordinate system and geometry
    coordinate_system: CoordinateSystem = Field(
        CoordinateSystem.GDA2020,
        description="Coordinate reference system"
    )
    
    geometry_wkt: Optional[str] = Field(
        None,
        description="Boundary polygon as Well-Known Text"
    )
    
    geometry_wkb: Optional[bytes] = Field(
        None,
        description="Boundary polygon as Well-Known Binary"
    )
    
    centroid_longitude: Optional[Decimal] = Field(
        None,
        ge=-180,
        le=180,
        description="Longitude of centroid"
    )
    
    centroid_latitude: Optional[Decimal] = Field(
        None,
        ge=-90,
        le=90, 
        description="Latitude of centroid"
    )
    
    @validator('geographic_code')
    def sync_geographic_code_with_sa2(cls, v, values):
        """Ensure geographic_code matches sa2_code."""
        sa2_code = values.get('sa2_code')
        if sa2_code and v != sa2_code:
            raise ValueError("geographic_code must match sa2_code for SA2 boundaries")
        return v


class GeographicRelationship(GeographicModel):
    """
    Model for relationships between different geographic levels.
    
    Enables mapping between SA1s, SA2s, and other geographic classifications
    like LGAs, postcodes, etc.
    """
    
    # Source geographic area
    source_type: str = Field(
        ...,
        pattern=r"^(SA1|SA2|SA3|SA4|LGA|POA|CED|SED|SUA|UCL|SOS|SOSR|RA)$",
        description="Type of source geographic area"
    )
    
    source_code: str = Field(
        ...,
        description="Code of source geographic area"
    )
    
    source_name: Optional[str] = Field(
        None,
        description="Name of source geographic area"
    )
    
    # Target geographic area
    target_type: str = Field(
        ...,
        pattern=r"^(SA1|SA2|SA3|SA4|LGA|POA|CED|SED|SUA|UCL|SOS|SOSR|RA)$",
        description="Type of target geographic area"
    )
    
    target_code: str = Field(
        ...,
        description="Code of target geographic area"
    )
    
    target_name: Optional[str] = Field(
        None,
        description="Name of target geographic area"
    )
    
    # Relationship strength
    allocation_percentage: Optional[Decimal] = Field(
        None,
        ge=0,
        le=100,
        description="Percentage allocation for partial overlaps"
    )
    
    population_allocation: Optional[int] = Field(
        None,
        ge=0,
        description="Population count allocated to this relationship"
    )
    
    area_allocation_sqkm: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Area allocated to this relationship in square kilometres"
    )
    
    # Relationship metadata
    relationship_type: str = Field(
        ...,
        pattern=r"^(exact|partial|majority|approximation)$",
        description="Type of geographic relationship"
    )
    
    @validator('allocation_percentage')
    def validate_percentage_range(cls, v):
        """Ensure percentage is between 0 and 100."""
        if v is not None and (v < 0 or v > 100):
            raise ValueError("Allocation percentage must be between 0 and 100")
        return v