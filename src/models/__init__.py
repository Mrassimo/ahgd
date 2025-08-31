"""
Pydantic Data Models for Australian Health Data Analytics

This module provides type-safe, validated data models for all data sources
used in the AHGD project, ensuring data quality and consistency across
the modern data engineering pipeline.
"""

from .base import BaseModel
from .base import GeographicModel
from .base import TimestampedModel
from .climate import AirQualityRecord
from .climate import ClimateRecord
from .geographic import GeographicRelationship
from .geographic import SA1Boundary
from .geographic import SA2Boundary
from .health import AIHWMortalityRecord
from .health import HealthcareVariationRecord
from .health import MBSRecord
from .health import PBSRecord
from .health import PHIDUChronicDiseaseRecord
from .seifa import SEIFAIndex
from .seifa import SEIFARecord

__all__ = [
    # Base models
    "BaseModel",
    "TimestampedModel",
    "GeographicModel",
    # Geographic models
    "SA1Boundary",
    "SA2Boundary",
    "GeographicRelationship",
    # Socio-economic models
    "SEIFARecord",
    "SEIFAIndex",
    # Health data models
    "MBSRecord",
    "PBSRecord",
    "AIHWMortalityRecord",
    "PHIDUChronicDiseaseRecord",
    "HealthcareVariationRecord",
    # Environmental models
    "ClimateRecord",
    "AirQualityRecord",
]
