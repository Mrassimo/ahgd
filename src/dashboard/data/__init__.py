"""
Data Layer for Australian Health Analytics Dashboard

This module contains all data loading and processing functionality,
extracted from the monolithic dashboard for better modularity and testability.

Modules:
    loaders: Core data loading functions with caching
    processors: Data transformation and processing utilities
"""

from .loaders import load_data, calculate_correlations
from .processors import (
    filter_data_by_states,
    validate_health_data,
    calculate_health_risk_score,
    identify_health_hotspots,
    prepare_correlation_data,
    generate_health_indicators,
    calculate_data_quality_metrics,
    apply_scenario_analysis
)

__all__ = [
    'load_data',
    'calculate_correlations', 
    'filter_data_by_states',
    'validate_health_data',
    'calculate_health_risk_score',
    'identify_health_hotspots',
    'prepare_correlation_data',
    'generate_health_indicators',
    'calculate_data_quality_metrics',
    'apply_scenario_analysis'
]