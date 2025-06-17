"""
Data Quality Validators Package

Comprehensive data quality validation utilities for Australian health data including:
- Australian health data standards validation
- Schema evolution and validation framework  
- Data quality metrics calculation and monitoring
- Cross-dataset consistency validation
- Data lineage and provenance tracking
- Privacy compliance and de-identification validation

This package provides the core validation infrastructure for ensuring
data quality throughout the Australian health analytics platform.
"""

from .australian_health_validators import (
    AustralianHealthDataValidator,
    DataQualityMetricsCalculator
)
from .schema_validators import (
    SchemaValidator,
    DataLineageTracker,
    SchemaCompatibility,
    SchemaChangeType
)
from .quality_metrics import (
    AustralianHealthQualityMetrics,
    QualityDimension,
    QualityThreshold,
    QualityMetric,
    QualityReport
)

__all__ = [
    # Australian Health Data Validators
    "AustralianHealthDataValidator",
    "DataQualityMetricsCalculator",
    
    # Schema Validators
    "SchemaValidator", 
    "DataLineageTracker",
    "SchemaCompatibility",
    "SchemaChangeType",
    
    # Quality Metrics
    "AustralianHealthQualityMetrics",
    "QualityDimension",
    "QualityThreshold", 
    "QualityMetric",
    "QualityReport"
]

__version__ = "1.0.0"
__author__ = "Australian Health Analytics Team"
__description__ = "Data quality validation framework for Australian health data"