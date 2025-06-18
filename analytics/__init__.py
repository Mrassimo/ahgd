"""
Australian Health Geography Data - Advanced Analytics Showcase

This module provides sophisticated data science capabilities for health analytics,
including machine learning, statistical modeling, and advanced visualizations.
"""

__version__ = "1.0.0"
__author__ = "AHGD Analytics Team"

from .modules.clustering import HealthClusterAnalyzer
from .modules.predictive import HealthPredictiveModels
from .modules.spatial import SpatialHealthAnalyzer
from .modules.statistical import AdvancedStatistics
from .modules.visualization import AdvancedVisualizations
from .modules.causal import CausalInferenceAnalyzer
from .modules.network import HealthNetworkAnalyzer
from .modules.bayesian import BayesianHealthModels

__all__ = [
    "HealthClusterAnalyzer",
    "HealthPredictiveModels", 
    "SpatialHealthAnalyzer",
    "AdvancedStatistics",
    "AdvancedVisualizations",
    "CausalInferenceAnalyzer",
    "HealthNetworkAnalyzer",
    "BayesianHealthModels",
]