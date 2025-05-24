"""
Model definitions for AHGD ETL pipeline.

This package contains the data model classes for dimension and fact tables
in the AHGD data warehouse.
"""

from .time_dimension import TimeDimensionModel, create_time_dimension, generate_time_dimension

__all__ = ['TimeDimensionModel', 'create_time_dimension', 'generate_time_dimension']