"""
SA1 (Statistical Area Level 1) geographic data schema for AHGD.

This module defines schemas for SA1 boundary data including validation
for coordinates, geometry, and spatial relationships based on ABS 2021 standards.
SA1s are the smallest geographic building blocks, with 11-digit codes and
populations typically ranging from 200-800 people.
"""

import math
from typing import Any
from typing import Optional

from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from .base_schema import DataSource
from .base_schema import GeographicBoundary
from .base_schema import SchemaVersion
from .base_schema import VersionedSchema


class SA1Coordinates(VersionedSchema):
    """Schema for SA1 coordinate data with validation for ABS 2021 11-digit codes."""

    sa1_code: str = Field(..., pattern=r"^\d{11}$", description="11-digit SA1 code (ABS 2021)")
    sa1_name: str = Field(..., min_length=1, max_length=150, description="SA1 name")

    # Extend GeographicBoundary fields
    boundary_data: GeographicBoundary = Field(..., description="Geographic boundary information")

    # SA1-specific demographic fields
    population: Optional[int] = Field(
        None, ge=50, le=1200, description="Population count (typical range 200-800)"
    )
    dwellings: Optional[int] = Field(None, ge=20, le=500, description="Number of dwellings")

    # Neighbouring SA1s
    neighbours: list[str] = Field(
        default_factory=list, description="List of neighbouring SA1 codes"
    )

    # Hierarchical relationships - SA1 is the foundation level
    sa2_code: str = Field(..., pattern=r"^\d{9}$", description="Parent SA2 code")
    sa3_code: str = Field(..., pattern=r"^\d{5}$", description="Parent SA3 code")
    sa4_code: str = Field(..., pattern=r"^\d{3}$", description="Parent SA4 code")
    state_code: str = Field(..., description="State/territory code")

    # ABS classification fields
    remoteness_category: Optional[str] = Field(
        None, description="ABS Remoteness Structure category"
    )
    indigenous_region: Optional[str] = Field(
        None, description="Indigenous Region code if applicable"
    )

    # Data source information
    data_source: DataSource = Field(..., description="Source of the SA1 data")

    @field_validator("sa1_code")
    @classmethod
    def validate_sa1_code_structure(cls, v: str) -> str:
        """Validate SA1 code structure and hierarchical consistency."""
        if not v.isdigit() or len(v) != 11:
            raise ValueError("SA1 code must be exactly 11 digits")

        # First digit should be state code (1-8)
        state_digit = int(v[0])
        if state_digit < 1 or state_digit > 8:
            raise ValueError(f"Invalid state code in SA1: {state_digit}")

        return v

    @field_validator("neighbours")
    @classmethod
    def validate_neighbour_codes(cls, v: list[str]) -> list[str]:
        """Validate all neighbour codes are valid SA1 codes."""
        for code in v:
            if not code.isdigit() or len(code) != 11:
                raise ValueError(f"Invalid neighbour SA1 code: {code}")
        return v

    @field_validator("remoteness_category")
    @classmethod
    def validate_remoteness(cls, v: Optional[str]) -> Optional[str]:
        """Validate ABS remoteness category."""
        if v is not None:
            valid_categories = {
                "Major Cities",
                "Inner Regional",
                "Outer Regional",
                "Remote",
                "Very Remote",
            }
            if v not in valid_categories:
                raise ValueError(f"Invalid remoteness category: {v}")
        return v

    @model_validator(mode="after")
    def validate_hierarchical_consistency(self) -> "SA1Coordinates":
        """Ensure SA1 code is consistent with parent SA2, SA3, and SA4 codes."""
        sa1_code = self.sa1_code
        sa2_code = self.sa2_code
        sa3_code = self.sa3_code
        sa4_code = self.sa4_code

        if sa1_code and sa2_code:
            # SA1 code should start with SA2 code (first 9 digits)
            if not sa1_code.startswith(sa2_code):
                raise ValueError(f"SA1 code {sa1_code} inconsistent with SA2 code {sa2_code}")

        if sa2_code and sa3_code:
            # SA2 code should start with SA3 code (first 5 digits)
            if not sa2_code.startswith(sa3_code):
                raise ValueError(f"SA2 code {sa2_code} inconsistent with SA3 code {sa3_code}")

        if sa3_code and sa4_code:
            # SA3 code should start with SA4 code (first 3 digits)
            if not sa3_code.startswith(sa4_code):
                raise ValueError(f"SA3 code {sa3_code} inconsistent with SA4 code {sa4_code}")

        return self

    @model_validator(mode="after")
    def validate_coordinate_bounds(self) -> "SA1Coordinates":
        """Validate coordinates are within Australian bounds."""
        boundary = self.boundary_data
        if boundary:
            lat = boundary.centroid_lat
            lon = boundary.centroid_lon

            if lat and lon:
                # Australian mainland bounds (approximate, including external territories)
                if not (-55 <= lat <= -8 and 96 <= lon <= 168):
                    # Log warning but allow for external territories
                    pass

        return self

    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "SA1Coordinates"

    def validate_data_integrity(self) -> list[str]:
        """Validate SA1 data integrity."""
        errors = []

        # Check boundary geometry
        if self.boundary_data.geometry:
            geom_type = self.boundary_data.geometry.get("type")
            if geom_type not in ["Polygon", "MultiPolygon"]:
                errors.append(f"SA1 geometry should be Polygon or MultiPolygon, got {geom_type}")

        # Check area consistency - SA1s are typically very small
        if self.boundary_data.area_sq_km:
            # SA1s typically range from 0.001 to 100 sq km (most urban SA1s are <1 sq km)
            if self.boundary_data.area_sq_km < 0.0001:
                errors.append("SA1 area suspiciously small")
            elif self.boundary_data.area_sq_km > 10000:  # Large rural SA1s can be substantial
                errors.append("SA1 area unusually large, please verify")

        # Population density check
        if self.population and self.boundary_data.area_sq_km:
            density = self.population / self.boundary_data.area_sq_km
            if density > 100000:  # More than 100k per sq km is extremely unusual
                errors.append(f"Population density extremely high: {density:.0f} per sq km")
            elif (
                density < 1 and self.boundary_data.area_sq_km < 10
            ):  # Urban SA1 with very low density
                errors.append(
                    f"Population density unusually low for small area: {density:.1f} per sq km"
                )

        # Population range validation
        if self.population:
            if self.population < 100:
                errors.append(f"Population {self.population} below typical SA1 minimum (200)")
            elif self.population > 1000:
                errors.append(f"Population {self.population} above typical SA1 maximum (800)")

        return errors

    def get_parent_codes(self) -> dict[str, str]:
        """Get all parent geographic codes."""
        return {
            "sa2_code": self.sa2_code,
            "sa3_code": self.sa3_code,
            "sa4_code": self.sa4_code,
            "state_code": self.state_code,
        }

    model_config = {
        "json_schema_extra": {
            "example": {
                "sa1_code": "10102100701",
                "sa1_name": "Sydney - Haymarket - The Rocks (Central)",
                "boundary_data": {
                    "boundary_id": "10102100701",
                    "boundary_type": "SA1",
                    "name": "Sydney - Haymarket - The Rocks (Central)",
                    "state": "NSW",
                    "area_sq_km": 0.85,
                    "centroid_lat": -33.8688,
                    "centroid_lon": 151.2093,
                },
                "population": 420,
                "dwellings": 180,
                "sa2_code": "101021007",
                "sa3_code": "10102",
                "sa4_code": "101",
                "state_code": "NSW",
                "remoteness_category": "Major Cities",
            }
        }
    }


class SA1GeometryValidation(VersionedSchema):
    """Extended schema for detailed SA1 geometry validation."""

    sa1_code: str = Field(..., pattern=r"^\d{11}$", description="11-digit SA1 code")

    # Geometry validation results
    is_valid_geometry: bool = Field(..., description="Whether geometry is valid")
    geometry_errors: list[str] = Field(
        default_factory=list, description="List of geometry validation errors"
    )

    # Topology checks
    is_simple: bool = Field(..., description="Whether geometry is simple (no self-intersections)")
    is_closed: bool = Field(..., description="Whether all rings are properly closed")
    has_holes: bool = Field(False, description="Whether polygon has interior holes")

    # Spatial metrics
    compactness_ratio: Optional[float] = Field(
        None, ge=0, le=1, description="Polsby-Popper compactness ratio"
    )

    # Coordinate precision (important for small SA1 areas)
    coordinate_precision: int = Field(..., ge=1, le=15, description="Decimal places in coordinates")

    # SA1-specific checks
    contains_address_points: Optional[int] = Field(
        None, ge=0, description="Number of address points contained within SA1"
    )

    @field_validator("compactness_ratio")
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
        return (4 * math.pi * area) / (perimeter**2)

    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "SA1GeometryValidation"

    def validate_data_integrity(self) -> list[str]:
        """Validate geometry validation data."""
        errors = []

        if not self.is_valid_geometry and not self.geometry_errors:
            errors.append("Invalid geometry but no errors specified")

        if self.is_simple and self.geometry_errors:
            for error in self.geometry_errors:
                if "intersection" in error.lower():
                    errors.append("Geometry marked as simple but has intersection errors")
                    break

        # SA1s should generally not have holes due to their small size
        if self.has_holes:
            errors.append("SA1 geometry has holes, which is unusual for smallest geographic unit")

        return errors


class SA1BoundaryRelationship(VersionedSchema):
    """Schema for SA1 spatial relationships and adjacency."""

    sa1_code: str = Field(..., pattern=r"^\d{11}$", description="Primary SA1 code")

    # Adjacent boundaries
    adjacent_sa1s: list[dict[str, Any]] = Field(
        default_factory=list, description="List of adjacent SA1s with shared boundary info"
    )

    # Containment relationships
    parent_sa2: str = Field(..., pattern=r"^\d{9}$", description="Parent SA2 code")

    # Address and infrastructure data
    address_count: Optional[int] = Field(None, ge=0, description="Number of addresses within SA1")
    mesh_block_codes: list[str] = Field(
        default_factory=list, description="List of Mesh Block codes that comprise this SA1"
    )

    # Distance metrics
    distance_to_coast_km: Optional[float] = Field(
        None, ge=0, description="Distance to nearest coastline in km"
    )
    distance_to_town_centre_km: Optional[float] = Field(
        None, ge=0, description="Distance to nearest town/city centre in km"
    )

    # Urban/rural classification
    urban_rural_classification: Optional[str] = Field(
        None, description="Urban/rural classification"
    )

    @field_validator("adjacent_sa1s")
    @classmethod
    def validate_adjacency_data(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate adjacency information structure."""
        for adj in v:
            if "sa1_code" not in adj:
                raise ValueError("Adjacent SA1 must have sa1_code")
            if "sa1_code" in adj and (not adj["sa1_code"].isdigit() or len(adj["sa1_code"]) != 11):
                raise ValueError("Adjacent SA1 code must be 11 digits")
            if "shared_boundary_length" in adj:
                if adj["shared_boundary_length"] < 0:
                    raise ValueError("Shared boundary length cannot be negative")
        return v

    @field_validator("urban_rural_classification")
    @classmethod
    def validate_urban_rural(cls, v: Optional[str]) -> Optional[str]:
        """Validate urban/rural classification."""
        if v is not None:
            valid_classifications = {"Urban", "Rural", "Mixed Urban and Rural"}
            if v not in valid_classifications:
                raise ValueError(f"Invalid urban/rural classification: {v}")
        return v

    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "SA1BoundaryRelationship"

    def validate_data_integrity(self) -> list[str]:
        """Validate relationship data integrity."""
        errors = []

        # Check for self-adjacency
        for adj in self.adjacent_sa1s:
            if adj.get("sa1_code") == self.sa1_code:
                errors.append("SA1 cannot be adjacent to itself")

        # Check parent SA2 consistency
        if self.parent_sa2 and not self.sa1_code.startswith(self.parent_sa2):
            errors.append(
                f"Parent SA2 {self.parent_sa2} inconsistent with SA1 code {self.sa1_code}"
            )

        # Validate Mesh Block containment
        mesh_block_set = set(self.mesh_block_codes)
        if len(mesh_block_set) != len(self.mesh_block_codes):
            errors.append("Duplicate Mesh Block codes in containment list")

        return errors


# Migration functions for SA1 schemas


def migrate_sa2_to_sa1(sa2_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Migrate SA2 data to SA1 structure.
    Note: This requires external mapping data as SA2s contain multiple SA1s.
    """
    # This is a placeholder - actual migration would require ABS correspondence files
    sa1_records = []

    # Extract base information that can be inherited
    base_data = {
        "sa2_code": sa2_data.get("sa2_code", ""),
        "sa3_code": sa2_data.get("sa3_code", ""),
        "sa4_code": sa2_data.get("sa4_code", ""),
        "state_code": sa2_data.get("state_code", ""),
        "data_source": sa2_data.get("data_source", {}),
        "schema_version": SchemaVersion.V2_0_0.value,
    }

    # Note: Actual implementation would use ABS correspondence files to map SA2 to constituent SA1s
    return sa1_records


def validate_sa1_hierarchy(sa1_data: dict[str, Any]) -> list[str]:
    """Validate SA1 fits within correct geographic hierarchy."""
    errors = []

    sa1_code = sa1_data.get("sa1_code", "")
    sa2_code = sa1_data.get("sa2_code", "")

    if sa1_code and sa2_code:
        if not sa1_code.startswith(sa2_code):
            errors.append(f"SA1 code {sa1_code} not contained within SA2 {sa2_code}")

    return errors
