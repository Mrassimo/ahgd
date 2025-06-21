"""
SA2 (Statistical Area Level 2) geographic data schema for AHGD.

This module defines schemas for SA2 boundary data including validation
for coordinates, geometry, and spatial relationships.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import Field, field_validator, model_validator
import math

from .base_schema import (
    VersionedSchema, 
    GeographicBoundary, 
    DataSource,
    SchemaVersion,
    DataQualityLevel
)


class SA2Coordinates(VersionedSchema):
    """Schema for SA2 coordinate data with validation."""
    
    sa2_code: str = Field(..., regex=r'^\d{9}$', description="9-digit SA2 code")
    sa2_name: str = Field(..., min_length=1, max_length=100, description="SA2 name")
    
    # Extend GeographicBoundary fields
    boundary_data: GeographicBoundary = Field(..., description="Geographic boundary information")
    
    # Additional SA2-specific fields
    population: Optional[int] = Field(None, ge=0, description="Population count")
    dwellings: Optional[int] = Field(None, ge=0, description="Number of dwellings")
    
    # Neighbouring SA2s
    neighbours: List[str] = Field(
        default_factory=list,
        description="List of neighbouring SA2 codes"
    )
    
    # Hierarchical relationships
    sa3_code: str = Field(..., regex=r'^\d{5}$', description="Parent SA3 code")
    sa4_code: str = Field(..., regex=r'^\d{3}$', description="Parent SA4 code")
    state_code: str = Field(..., description="State/territory code")
    
    # Data source information
    data_source: DataSource = Field(..., description="Source of the SA2 data")
    
    @field_validator('sa2_code')
    @classmethod
    def validate_sa2_code_structure(cls, v: str) -> str:
        """Validate SA2 code structure and checksum."""
        if not v.isdigit() or len(v) != 9:
            raise ValueError("SA2 code must be exactly 9 digits")
            
        # First digit should be state code (1-8)
        state_digit = int(v[0])
        if state_digit < 1 or state_digit > 8:
            raise ValueError(f"Invalid state code in SA2: {state_digit}")
            
        return v
    
    @field_validator('neighbours')
    @classmethod
    def validate_neighbour_codes(cls, v: List[str]) -> List[str]:
        """Validate all neighbour codes are valid SA2 codes."""
        for code in v:
            if not code.isdigit() or len(code) != 9:
                raise ValueError(f"Invalid neighbour SA2 code: {code}")
        return v
    
    @model_validator(mode='after')
    def validate_hierarchical_consistency(self) -> 'SA2Coordinates':
        """Ensure SA2 code is consistent with SA3 and SA4 codes."""
        sa2_code = self.sa2_code
        sa3_code = self.sa3_code
        sa4_code = self.sa4_code
        
        if sa2_code and sa3_code:
            # SA2 code should start with SA3 code
            if not sa2_code.startswith(sa3_code):
                raise ValueError(f"SA2 code {sa2_code} inconsistent with SA3 code {sa3_code}")
                
        if sa3_code and sa4_code:
            # SA3 code should start with SA4 code
            if not sa3_code.startswith(sa4_code):
                raise ValueError(f"SA3 code {sa3_code} inconsistent with SA4 code {sa4_code}")
                
        return self
    
    @model_validator(mode='after')
    def validate_coordinate_bounds(self) -> 'SA2Coordinates':
        """Validate coordinates are within Australian bounds."""
        boundary = self.boundary_data
        if boundary:
            lat = boundary.centroid_lat
            lon = boundary.centroid_lon
            
            if lat and lon:
                # Australian mainland bounds (approximate)
                if not (-44 <= lat <= -10 and 112 <= lon <= 154):
                    # Could be external territory, log warning but allow
                    pass
                    
        return self
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "SA2Coordinates"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate SA2 data integrity."""
        errors = []
        
        # Check boundary geometry
        if self.boundary_data.geometry:
            geom_type = self.boundary_data.geometry.get('type')
            if geom_type not in ['Polygon', 'MultiPolygon']:
                errors.append(f"SA2 geometry should be Polygon or MultiPolygon, got {geom_type}")
                
        # Check area consistency
        if self.boundary_data.area_sq_km:
            # SA2s typically range from 0.1 to 10,000 sq km
            if self.boundary_data.area_sq_km < 0.01:
                errors.append("SA2 area suspiciously small")
            elif self.boundary_data.area_sq_km > 50000:
                errors.append("SA2 area suspiciously large")
                
        # Population density check
        if self.population and self.boundary_data.area_sq_km:
            density = self.population / self.boundary_data.area_sq_km
            if density > 50000:  # More than 50k per sq km is unusual
                errors.append(f"Population density unusually high: {density:.0f} per sq km")
                
        return errors
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "sa2_code": "101021007",
                "sa2_name": "Sydney - Haymarket - The Rocks",
                "boundary_data": {
                    "boundary_id": "101021007",
                    "boundary_type": "SA2",
                    "name": "Sydney - Haymarket - The Rocks",
                    "state": "NSW",
                    "area_sq_km": 2.45,
                    "centroid_lat": -33.8688,
                    "centroid_lon": 151.2093
                },
                "sa3_code": "10102",
                "sa4_code": "101",
                "state_code": "NSW"
            }
        }
    }


class SA2GeometryValidation(VersionedSchema):
    """Extended schema for detailed SA2 geometry validation."""
    
    sa2_code: str = Field(..., regex=r'^\d{9}$', description="9-digit SA2 code")
    
    # Geometry validation results
    is_valid_geometry: bool = Field(..., description="Whether geometry is valid")
    geometry_errors: List[str] = Field(
        default_factory=list,
        description="List of geometry validation errors"
    )
    
    # Topology checks
    is_simple: bool = Field(..., description="Whether geometry is simple (no self-intersections)")
    is_closed: bool = Field(..., description="Whether all rings are properly closed")
    has_holes: bool = Field(False, description="Whether polygon has interior holes")
    
    # Spatial metrics
    compactness_ratio: Optional[float] = Field(
        None, 
        ge=0, 
        le=1,
        description="Polsby-Popper compactness ratio"
    )
    
    # Coordinate precision
    coordinate_precision: int = Field(
        ..., 
        ge=1, 
        le=15,
        description="Decimal places in coordinates"
    )
    
    @field_validator('compactness_ratio')
    @classmethod
    def validate_compactness(cls, v: Optional[float]) -> Optional[float]:
        """Validate compactness ratio calculation."""
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Compactness ratio must be between 0 and 1")
        return v
    
    def calculate_compactness(self, area: float, perimeter: float) -> float:
        """
        Calculate Polsby-Popper compactness ratio.
        
        Ratio = (4 * π * Area) / (Perimeter²)
        """
        if perimeter <= 0:
            return 0.0
        return (4 * math.pi * area) / (perimeter ** 2)
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "SA2GeometryValidation"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate geometry validation data."""
        errors = []
        
        if not self.is_valid_geometry and not self.geometry_errors:
            errors.append("Invalid geometry but no errors specified")
            
        if self.is_simple and self.geometry_errors:
            for error in self.geometry_errors:
                if "intersection" in error.lower():
                    errors.append("Geometry marked as simple but has intersection errors")
                    break
                    
        return errors


class SA2BoundaryRelationship(VersionedSchema):
    """Schema for SA2 spatial relationships and adjacency."""
    
    sa2_code: str = Field(..., regex=r'^\d{9}$', description="Primary SA2 code")
    
    # Adjacent boundaries
    adjacent_sa2s: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of adjacent SA2s with shared boundary info"
    )
    
    # Containment relationships
    contains_sa1s: List[str] = Field(
        default_factory=list,
        description="List of SA1 codes contained within this SA2"
    )
    
    # Distance metrics
    nearest_coast_km: Optional[float] = Field(
        None,
        ge=0,
        description="Distance to nearest coastline in km"
    )
    nearest_capital_km: Optional[float] = Field(
        None,
        ge=0,
        description="Distance to nearest capital city in km"
    )
    
    # Remoteness classification
    remoteness_category: Optional[str] = Field(
        None,
        description="ABS remoteness structure category"
    )
    
    @field_validator('adjacent_sa2s')
    @classmethod
    def validate_adjacency_data(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate adjacency information structure."""
        for adj in v:
            if 'sa2_code' not in adj:
                raise ValueError("Adjacent SA2 must have sa2_code")
            if 'shared_boundary_length' in adj:
                if adj['shared_boundary_length'] < 0:
                    raise ValueError("Shared boundary length cannot be negative")
        return v
    
    @field_validator('remoteness_category')
    @classmethod
    def validate_remoteness(cls, v: Optional[str]) -> Optional[str]:
        """Validate remoteness category."""
        if v is not None:
            valid_categories = {
                'Major Cities', 
                'Inner Regional', 
                'Outer Regional',
                'Remote', 
                'Very Remote'
            }
            if v not in valid_categories:
                raise ValueError(f"Invalid remoteness category: {v}")
        return v
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "SA2BoundaryRelationship"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate relationship data integrity."""
        errors = []
        
        # Check for self-adjacency
        for adj in self.adjacent_sa2s:
            if adj.get('sa2_code') == self.sa2_code:
                errors.append("SA2 cannot be adjacent to itself")
                
        # Validate SA1 containment
        sa1_set = set(self.contains_sa1s)
        if len(sa1_set) != len(self.contains_sa1s):
            errors.append("Duplicate SA1 codes in containment list")
            
        return errors


# Migration functions for SA2 schemas

def migrate_sa2_v1_to_v2(old_data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate SA2 data from v1.0.0 to v2.0.0."""
    new_data = old_data.copy()
    
    # Example migration: restructure boundary data
    if 'lat' in old_data and 'lon' in old_data:
        # Move coordinates into boundary_data structure
        if 'boundary_data' not in new_data:
            new_data['boundary_data'] = {}
        new_data['boundary_data']['centroid_lat'] = old_data.pop('lat')
        new_data['boundary_data']['centroid_lon'] = old_data.pop('lon')
        
    # Update schema version
    new_data['schema_version'] = SchemaVersion.V2_0_0.value
    
    return new_data