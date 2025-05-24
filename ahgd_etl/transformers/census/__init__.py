"""
Census data transformers for the AHGD ETL pipeline.

This package contains transformer classes for processing Australian Bureau of Statistics
Census data tables (G01, G17, G18, G19, G20, G21, G25).
"""

from .g01_population import G01PopulationTransformer
from .g17_income import G17IncomeTransformer
from .g18_assistance_needed import G18AssistanceNeededTransformer
from .g19_health_conditions import G19HealthConditionsTransformer
from .g20_selected_conditions import G20SelectedConditionsTransformer
from .g21_conditions_by_characteristics import G21ConditionsByCharacteristicsTransformer
from .g25_unpaid_assistance import G25UnpaidAssistanceTransformer
from .base import BaseCensusTransformer

# Export transformers
__all__ = [
    'BaseCensusTransformer',
    'G01PopulationTransformer',
    'G17IncomeTransformer',
    'G18AssistanceNeededTransformer',
    'G19HealthConditionsTransformer',
    'G20SelectedConditionsTransformer',
    'G21ConditionsByCharacteristicsTransformer',
    'G25UnpaidAssistanceTransformer'
]