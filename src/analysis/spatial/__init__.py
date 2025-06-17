"""
Spatial Health Analysis Module

Provides geographic analysis and spatial health pattern identification for Australian SA2 areas.

Key Classes:
- SA2HealthMapper: Geographic health outcome mapping
- SpatialClustering: Geographic health pattern identification
- CatchmentAnalyzer: Population catchment area analysis
"""

from .sa2_health_mapper import SA2HealthMapper

__all__ = [
    "SA2HealthMapper",
]