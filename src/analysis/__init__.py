"""
Australian Health Analytics - Analysis Module

This module provides comprehensive health analytics for Australian population data,
leveraging processed SEIFA socio-economic indices, geographic boundaries, and health utilisation data.

Modules:
- risk: Health risk assessment and scoring algorithms
- spatial: Geographic analysis and spatial health patterns
- health: Health service utilisation and access analysis
- demographic: Population health profiling and demographics

Data Foundation (Phase 2):
- 2,293 SA2 areas with complete SEIFA socio-economic profiles
- 2,454 geographic boundaries with state/territory metadata
- 492,434 PBS health records for pharmaceutical utilisation analysis
- 92.9% integration success rate between datasets
"""

from .risk import HealthRiskCalculator
from .spatial import SA2HealthMapper
from .health import MedicareUtilisationAnalyzer, PharmaceuticalAnalyzer

__all__ = [
    "HealthRiskCalculator",
    "SA2HealthMapper", 
    "MedicareUtilisationAnalyzer",
    "PharmaceuticalAnalyzer",
]