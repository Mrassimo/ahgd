"""
Health Risk Assessment Module

Provides algorithms for calculating composite health risk scores using:
- SEIFA socio-economic disadvantage indices (IRSD, IRSAD, IER, IEO)
- Geographic accessibility factors
- Health service utilisation patterns
- Population demographics

Key Classes:
- HealthRiskCalculator: Multi-factor composite risk scoring
- HealthcareAccessScorer: Geographic accessibility analysis
- SocialDeterminantsAnalyzer: SEIFA-health correlation analysis
"""

from .health_risk_calculator import HealthRiskCalculator
from .healthcare_access_scorer import HealthcareAccessScorer

__all__ = [
    "HealthRiskCalculator",
    "HealthcareAccessScorer",
]