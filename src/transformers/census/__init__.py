"""
Census data transformers for AHGD ETL pipeline.

This module contains transformers for processing Australian Bureau of Statistics
census data including demographics, housing, employment, and socioeconomic indicators.
"""

from .demographic_transformer import DemographicTransformer

__all__ = [
    "DemographicTransformer",
]

__version__ = "1.0.0"