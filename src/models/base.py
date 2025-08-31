"""
Base Pydantic Models for AHGD Data Pipeline

Provides foundational model classes with common validation patterns,
geographic utilities, and data quality constraints.
"""

import re
from datetime import date
from datetime import datetime
from decimal import Decimal
from typing import Optional
from typing import Union

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import validator
from pydantic.types import constr


class BaseModel(PydanticBaseModel):
    """
    Base model with common configuration and utilities for all AHGD data models.

    Features:
    - Strict validation by default
    - Forbid extra fields to prevent data drift
    - Use enum values for serialisation
    - Validate assignment on field updates
    """

    model_config = ConfigDict(
        # Strict validation - no coercion unless explicitly allowed
        strict=True,
        # Forbid extra fields to catch data schema changes
        extra="forbid",
        # Use enum values instead of names in serialisation
        use_enum_values=True,
        # Validate on assignment updates
        validate_assignment=True,
        # Allow population by field name or alias
        populate_by_name=True,
        # Use JSON serialisable types by default
        arbitrary_types_allowed=False,
    )


class TimestampedModel(BaseModel):
    """
    Base model for data with temporal tracking.

    Includes standard timestamp fields for data lineage and versioning.
    """

    # Data reference date (when the data represents)
    reference_date: Optional[date] = Field(
        None, description="Date this data record represents (e.g., census collection date)"
    )

    # Data processing timestamps
    extracted_at: Optional[datetime] = Field(
        None, description="When this record was extracted from source"
    )

    processed_at: Optional[datetime] = Field(
        None, description="When this record was processed and validated"
    )

    # Data version tracking
    source_version: Optional[str] = Field(None, description="Version identifier of the source data")

    pipeline_version: Optional[str] = Field(
        None, description="Version of the processing pipeline used"
    )


class GeographicModel(TimestampedModel):
    """
    Base model for geographic/spatial data with Australian statistical geography.

    Provides common geographic identifiers and validation patterns.
    """

    # Primary geographic identifier
    geographic_code: constr(pattern=r"^[0-9]{9,11}$", min_length=9, max_length=11) = Field(
        ...,
        description="ABS statistical area code (SA1: 11 digits, SA2: 9 digits)",
        examples=["10102100701", "101021007"],
    )

    # Human-readable name
    geographic_name: constr(min_length=1, max_length=100) = Field(
        ..., description="Official name of the statistical area"
    )

    # State/territory classification
    state_code: constr(pattern=r"^[1-8]$", min_length=1, max_length=1) = Field(
        ..., description="ABS state/territory code (1-8)", examples=["1", "2", "3"]
    )

    state_name: constr(min_length=2, max_length=50) = Field(
        ..., description="State or territory name", examples=["NSW", "VIC", "QLD"]
    )

    # Area measurements
    area_sqkm: Optional[Union[float, Decimal]] = Field(
        None, ge=0, description="Area in square kilometres"
    )

    @validator("geographic_code")
    def validate_geographic_code_format(cls, v):
        """Validate Australian statistical area codes."""
        if len(v) == 11:
            # SA1 code format: SSCCCSSSSSS (state + SA4 + SA3 + SA2 + SA1)
            if not re.match(r"^[1-8][0-9]{10}$", v):
                raise ValueError("SA1 code must start with state digit 1-8 followed by 10 digits")
        elif len(v) == 9:
            # SA2 code format: SSCCCSSSS (state + SA4 + SA3 + SA2)
            if not re.match(r"^[1-8][0-9]{8}$", v):
                raise ValueError("SA2 code must start with state digit 1-8 followed by 8 digits")
        else:
            raise ValueError("Geographic code must be 9 digits (SA2) or 11 digits (SA1)")
        return v

    @validator("state_code")
    def validate_state_code(cls, v):
        """Validate ABS state/territory codes."""
        valid_codes = {"1", "2", "3", "4", "5", "6", "7", "8"}
        if v not in valid_codes:
            raise ValueError(f"State code must be one of {valid_codes}")
        return v

    @validator("state_name")
    def validate_state_name(cls, v):
        """Validate and standardise state/territory names."""
        # Mapping of variations to standard abbreviations
        state_mapping = {
            # Standard abbreviations
            "NSW": "NSW",
            "VIC": "VIC",
            "QLD": "QLD",
            "WA": "WA",
            "SA": "SA",
            "TAS": "TAS",
            "ACT": "ACT",
            "NT": "NT",
            # Full names
            "New South Wales": "NSW",
            "Victoria": "VIC",
            "Queensland": "QLD",
            "Western Australia": "WA",
            "South Australia": "SA",
            "Tasmania": "TAS",
            "Australian Capital Territory": "ACT",
            "Northern Territory": "NT",
            # Alternative forms
            "Other Territories": "OT",
            "OT": "OT",
        }

        standardised = state_mapping.get(v.strip())
        if not standardised:
            raise ValueError(
                f"Invalid state name: {v}. Must be one of {list(state_mapping.keys())}"
            )
        return standardised


class DataQualityMixin(BaseModel):
    """
    Mixin for models requiring data quality tracking and validation.
    """

    # Data quality flags
    has_missing_data: bool = Field(
        False, description="Whether this record has missing required fields"
    )

    quality_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Data quality score from 0.0 (poor) to 1.0 (excellent)"
    )

    validation_errors: Optional[list[str]] = Field(
        None, description="List of validation warnings or non-fatal errors"
    )

    # Source reliability
    source_reliability: Optional[str] = Field(
        None, pattern=r"^(high|medium|low)$", description="Reliability rating of the data source"
    )


class PopulationMixin(BaseModel):
    """
    Mixin for models with population data.
    """

    population_total: Optional[int] = Field(None, ge=0, description="Total population count")

    population_male: Optional[int] = Field(None, ge=0, description="Male population count")

    population_female: Optional[int] = Field(None, ge=0, description="Female population count")

    population_density_per_sqkm: Optional[float] = Field(
        None, ge=0, description="Population density per square kilometre"
    )

    @validator("population_male", "population_female")
    def validate_gender_population_sum(cls, v, values):
        """Validate that gender populations don't exceed total population."""
        if v is not None and "population_total" in values:
            total = values.get("population_total")
            if total is not None and v > total:
                raise ValueError(
                    f"Gender population ({v}) cannot exceed total population ({total})"
                )
        return v
