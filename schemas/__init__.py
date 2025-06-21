"""
AHGD Schema Definitions

This package contains schema definitions for all data types used in the
Australian Health and Geographic Data (AHGD) ETL pipeline.
"""

# Import all schema modules to make them available at package level
from . import base_schema
from . import census_schema  
from . import environmental_schema
from . import health_schema
from . import integrated_schema
from . import mortality_schema
from . import quality_standards
from . import sa2_schema
from . import seifa_schema
from . import target_outputs

# Make key classes available at package level
try:
    from .base_schema import BaseSchemaV1, BaseSchemaV2
    from .health_schema import HealthIndicatorSchema
    from .census_schema import CensusDataSchema
    from .sa2_schema import SA2GeographicSchema
    from .seifa_schema import SEIFASchema
    from .quality_standards import QualityStandards
    from .target_outputs import TargetSchema
except ImportError:
    # Handle cases where some schema modules might not have expected classes
    pass

__all__ = [
    'base_schema',
    'census_schema',
    'environmental_schema', 
    'health_schema',
    'integrated_schema',
    'mortality_schema',
    'quality_standards',
    'sa2_schema',
    'seifa_schema',
    'target_outputs',
]