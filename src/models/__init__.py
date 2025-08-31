"""
Pydantic Data Models for Australian Health Data Analytics

This module provides type-safe, validated data models for all data sources
used in the AHGD project, ensuring data quality and consistency across
the modern data engineering pipeline.
"""

from .base import BaseModel, TimestampedModel, GeographicModel
from .geographic import SA1Boundary, SA2Boundary, GeographicRelationship
from .seifa import SEIFARecord, SEIFAIndex
from .health import (
    MBSRecord, 
    PBSRecord, 
    AIHWMortalityRecord, 
    PHIDUChronicDiseaseRecord,
    HealthcareVariationRecord
)
from .climate import ClimateRecord, AirQualityRecord

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