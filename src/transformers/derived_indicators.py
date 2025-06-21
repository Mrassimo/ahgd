"""
Derived health indicators calculation for AHGD project.

This module calculates standardised health rates, composite indices, and
derived indicators from integrated health data following Australian standards.
"""

import logging
import math
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .base import BaseTransformer
from schemas.integrated_schema import MasterHealthRecord, HealthOutcome
from ..utils.interfaces import DataBatch, TransformationError
from ..utils.logging import get_logger


@dataclass
class StandardPopulation:
    """Standard population structure for age standardisation."""
    
    age_groups: Dict[str, int]
    total_population: int
    reference_year: int = 2021
    source: str = "ABS_Standard_Population"


@dataclass
class IndicatorDefinition:
    """Definition of a derived health indicator."""
    
    indicator_name: str
    calculation_method: str
    required_fields: List[str]
    unit: str
    interpretation: str
    higher_is_better: bool = True
    australian_average: Optional[float] = None
    data_source_priority: List[str] = None


class HealthIndicatorDeriver(BaseTransformer):
    """
    Calculates standardised health rates and indices from integrated data.
    
    Implements Australian health indicator calculation methodologies including
    age standardisation, rate calculations, and composite indices.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialise the health indicator deriver.
        
        Args:
            config: Configuration including indicator definitions and standards
            logger: Optional logger instance
        """
        super().__init__("health_indicator_deriver", config, logger)
        
        self.logger = get_logger(__name__) if logger is None else logger
        
        # Standard populations for age standardisation
        self.standard_populations = self._load_standard_populations(config)
        
        # Indicator definitions
        self.indicator_definitions = self._load_indicator_definitions(config)
        
        # Calculation parameters
        self.confidence_level = config.get('confidence_level', 95.0)
        self.minimum_sample_size = config.get('minimum_sample_size', 30)
        self.smoothing_enabled = config.get('smoothing_enabled', True)
        
        # Australian averages for comparison
        self.national_averages = config.get('national_averages', {})
        
    def transform(self, data: DataBatch, **kwargs) -> DataBatch:
        """
        Calculate derived health indicators for all records.
        
        Args:
            data: Batch of integrated health records
            **kwargs: Additional calculation parameters
            
        Returns:
            DataBatch: Records with calculated derived indicators
        """
        try:
            enhanced_records = []
            
            for record in data:
                enhanced_record = self.calculate_derived_indicators(record)
                enhanced_records.append(enhanced_record)
            
            self.logger.info(f"Calculated derived indicators for {len(enhanced_records)} records")
            return enhanced_records
            
        except Exception as e:
            self.logger.error(f"Derived indicator calculation failed: {e}")
            raise TransformationError(f"Derived indicator calculation failed: {e}") from e
    
    def calculate_derived_indicators(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate all derived health indicators for a single record.
        
        Args:
            record: Integrated health record
            
        Returns:
            Enhanced record with calculated derived indicators
        """
        enhanced_record = record.copy()
        
        try:
            # Calculate age-standardised rates
            enhanced_record.update(self._calculate_age_standardised_rates(record))
            
            # Calculate composite health indices
            enhanced_record.update(self._calculate_composite_indices(record))
            
            # Calculate inequality measures
            enhanced_record.update(self._calculate_inequality_measures(record))
            
            # Calculate accessibility indices
            enhanced_record.update(self._calculate_accessibility_indices(record))
            
            # Calculate environmental health indices
            enhanced_record.update(self._calculate_environmental_indices(record))
            
            # Calculate quality of life indices
            enhanced_record.update(self._calculate_quality_of_life_indices(record))
            
            # Add metadata about calculations
            enhanced_record['derived_indicators_calculated'] = datetime.utcnow().isoformat()
            enhanced_record['calculation_version'] = '1.0.0'
            
        except Exception as e:
            self.logger.error(f"Failed to calculate derived indicators for SA2 {record.get('sa2_code')}: {e}")
            enhanced_record['calculation_errors'] = [str(e)]
        
        return enhanced_record
    
    def get_schema(self) -> Dict[str, str]:
        """Get the expected output schema."""
        return {
            'sa2_code': 'string',
            'age_standardised_mortality_rate': 'float',
            'composite_health_index': 'float',
            'health_inequality_index': 'float',
            'healthcare_accessibility_index': 'float',
            'environmental_health_index': 'float'
        }
    
    def _calculate_age_standardised_rates(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate age-standardised mortality and morbidity rates."""
        standardised_rates = {}
        
        # Get population data
        population_data = record.get('demographic_profile', {})
        age_groups = population_data.get('age_groups', {})
        
        if not age_groups:
            return standardised_rates
        
        # Calculate age-standardised mortality rate
        mortality_data = record.get('mortality_indicators', {})
        if mortality_data:
            asmr = self._age_standardise_rate(
                age_specific_rates=self._extract_age_specific_mortality(mortality_data, age_groups),
                population_data=age_groups,
                indicator_name='mortality'
            )
            if asmr is not None:
                standardised_rates['age_standardised_mortality_rate'] = asmr
        
        # Calculate age-standardised disease prevalence rates
        disease_data = record.get('chronic_disease_prevalence', {})
        for disease, prevalence in disease_data.items():
            if isinstance(prevalence, (int, float)):
                # Simple age standardisation (would be more complex with age-specific data)
                standardised_rates[f'age_standardised_{disease}_rate'] = prevalence
        
        return standardised_rates
    
    def _calculate_composite_indices(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate composite health indices combining multiple indicators."""
        indices = {}
        
        # Overall Health Index (0-100 scale)
        health_components = []
        
        # Life expectancy component (25% weight)
        life_expectancy = record.get('life_expectancy')
        if life_expectancy:
            # Normalise against Australian average (~83 years)
            le_score = min(100, (float(life_expectancy) / 83.0) * 100)
            health_components.append(('life_expectancy', le_score, 0.25))
        
        # Mortality component (25% weight)
        mortality_rate = record.get('age_standardised_mortality_rate')
        if mortality_rate:
            # Lower mortality is better (invert and normalise)
            mort_score = max(0, 100 - (float(mortality_rate) / 10))  # Assuming rate per 1000
            health_components.append(('mortality', mort_score, 0.25))
        
        # Healthcare access component (25% weight)
        gp_services = record.get('gp_services_per_1000')
        if gp_services:
            # Normalise against national average
            gp_score = min(100, (float(gp_services) / 2.0) * 100)  # Assuming 2.0 is average
            health_components.append(('healthcare_access', gp_score, 0.25))
        
        # Risk factors component (25% weight)
        risk_factors = record.get('risk_factors', {})
        if risk_factors:
            risk_score = self._calculate_risk_factor_score(risk_factors)
            health_components.append(('risk_factors', risk_score, 0.25))
        
        # Calculate weighted composite score
        if health_components:
            total_weight = sum(weight for _, _, weight in health_components)
            weighted_score = sum(score * weight for _, score, weight in health_components)
            indices['composite_health_index'] = weighted_score / total_weight if total_weight > 0 else 0
        
        # Prevention Index
        indices['prevention_index'] = self._calculate_prevention_index(record)
        
        # Wellbeing Index
        indices['wellbeing_index'] = self._calculate_wellbeing_index(record)
        
        return indices
    
    def _calculate_inequality_measures(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate health inequality and equity measures."""
        inequality_measures = {}
        
        # Health Inequality Index relative to national average
        composite_health = record.get('composite_health_index')
        if composite_health is not None:
            national_avg = self.national_averages.get('composite_health_index', 75.0)
            inequality_measures['health_inequality_index'] = abs(composite_health - national_avg) / national_avg
        
        # Socioeconomic gradient in health
        seifa_score = record.get('seifa_scores', {}).get('IRSD')
        if seifa_score and composite_health:
            # Calculate correlation with socioeconomic status
            inequality_measures['socioeconomic_health_gradient'] = self._calculate_ses_gradient(
                ses_score=seifa_score,
                health_score=composite_health
            )
        
        # Access inequality
        gp_services = record.get('gp_services_per_1000')
        population_density = record.get('population_density_per_sq_km')
        if gp_services is not None and population_density is not None:
            inequality_measures['healthcare_access_inequality'] = self._calculate_access_inequality(
                gp_services, population_density
            )
        
        return inequality_measures
    
    def _calculate_accessibility_indices(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate healthcare accessibility indices."""
        accessibility = {}
        
        # Geographic accessibility
        geographic_data = record.get('boundary_data', {})
        area = geographic_data.get('area_sq_km', 0)
        population = record.get('total_population', 0)
        
        if area > 0 and population > 0:
            density = population / area
            accessibility['population_density_factor'] = min(1.0, density / 1000.0)  # Normalise to urban density
        
        # Service accessibility
        gp_services = record.get('gp_services_per_1000', 0)
        specialist_services = record.get('specialist_services_per_1000', 0)
        
        # Calculate service accessibility index (0-100)
        service_components = []
        if gp_services > 0:
            service_components.append(min(100, (gp_services / 2.0) * 100))  # Normalise against target of 2 per 1000
        if specialist_services > 0:
            service_components.append(min(100, (specialist_services / 0.5) * 100))  # Target 0.5 per 1000
        
        if service_components:
            accessibility['healthcare_accessibility_index'] = sum(service_components) / len(service_components)
        
        # Economic accessibility
        bulk_billing_rate = record.get('bulk_billing_rate')
        if bulk_billing_rate is not None:
            accessibility['economic_accessibility_index'] = float(bulk_billing_rate)
        
        return accessibility
    
    def _calculate_environmental_indices(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate environmental health indices."""
        environmental = {}
        
        env_indicators = record.get('environmental_indicators', {})
        
        # Air quality index
        air_quality = record.get('air_quality_index')
        if air_quality is not None:
            # Convert to health impact score (lower AQI is better)
            environmental['air_quality_health_score'] = max(0, 100 - float(air_quality))
        
        # Green space access
        green_space = record.get('green_space_access')
        if green_space is not None:
            environmental['green_space_health_score'] = float(green_space)
        
        # Composite environmental health index
        env_components = [
            environmental.get('air_quality_health_score'),
            environmental.get('green_space_health_score')
        ]
        
        valid_components = [comp for comp in env_components if comp is not None]
        if valid_components:
            environmental['environmental_health_index'] = sum(valid_components) / len(valid_components)
        
        return environmental
    
    def _calculate_quality_of_life_indices(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality of life and wellbeing indices."""
        qol_indices = {}
        
        # Mental health component
        mental_health = record.get('mental_health_indicators', {})
        psychological_distress = record.get('psychological_distress_high')
        
        if psychological_distress is not None:
            # Lower distress is better
            qol_indices['mental_wellbeing_score'] = max(0, 100 - float(psychological_distress))
        
        # Social determinants component
        seifa_scores = record.get('seifa_scores', {})
        if seifa_scores:
            # Use IRSAD (education and occupation) as proxy for social wellbeing
            irsad_score = seifa_scores.get('IRSAD')
            if irsad_score:
                # Normalise SEIFA score to 0-100
                qol_indices['social_determinants_score'] = ((irsad_score - 500) / 500) * 100
        
        # Overall quality of life index
        qol_components = [
            qol_indices.get('mental_wellbeing_score'),
            qol_indices.get('social_determinants_score'),
            record.get('composite_health_index')
        ]
        
        valid_qol_components = [comp for comp in qol_components if comp is not None]
        if valid_qol_components:
            qol_indices['quality_of_life_index'] = sum(valid_qol_components) / len(valid_qol_components)
        
        return qol_indices
    
    def _age_standardise_rate(
        self, 
        age_specific_rates: Dict[str, float], 
        population_data: Dict[str, int],
        indicator_name: str
    ) -> Optional[float]:
        """
        Calculate age-standardised rate using direct standardisation.
        
        Args:
            age_specific_rates: Rates by age group
            population_data: Population counts by age group
            indicator_name: Name of indicator being standardised
            
        Returns:
            Age-standardised rate or None if insufficient data
        """
        if not age_specific_rates or not population_data:
            return None
        
        # Get standard population
        standard_pop = self.standard_populations.get('australia_2021')
        if not standard_pop:
            return None
        
        # Calculate weighted rate
        numerator = 0
        denominator = 0
        
        for age_group in age_specific_rates:
            if age_group in standard_pop.age_groups:
                rate = age_specific_rates[age_group]
                std_pop = standard_pop.age_groups[age_group]
                
                numerator += rate * std_pop
                denominator += std_pop
        
        return numerator / denominator if denominator > 0 else None
    
    def _extract_age_specific_mortality(
        self, 
        mortality_data: Dict[str, Any], 
        age_groups: Dict[str, int]
    ) -> Dict[str, float]:
        """Extract age-specific mortality rates from data."""
        # This would extract actual age-specific rates from detailed mortality data
        # For now, return a simplified version
        age_specific_rates = {}
        
        overall_rate = mortality_data.get('all_cause_mortality_rate')
        if overall_rate:
            # Distribute overall rate across age groups with age-adjusted weights
            for age_group, population in age_groups.items():
                if population > 0:
                    # Apply age-specific multiplier (would be from actuarial tables)
                    age_multiplier = self._get_age_mortality_multiplier(age_group)
                    age_specific_rates[age_group] = overall_rate * age_multiplier
        
        return age_specific_rates
    
    def _get_age_mortality_multiplier(self, age_group: str) -> float:
        """Get age-specific mortality multiplier."""
        # Simplified age multipliers (would use actual life tables)
        multipliers = {
            'age_0_4': 0.2,
            'age_5_14': 0.1,
            'age_15_24': 0.3,
            'age_25_44': 0.5,
            'age_45_64': 1.0,
            'age_65_plus': 3.0
        }
        return multipliers.get(age_group, 1.0)
    
    def _calculate_risk_factor_score(self, risk_factors: Dict[str, Any]) -> float:
        """Calculate composite risk factor score."""
        risk_components = []
        
        # Major modifiable risk factors
        smoking = risk_factors.get('smoking_prevalence')
        if smoking is not None:
            risk_components.append(100 - float(smoking))  # Lower smoking is better
        
        obesity = risk_factors.get('obesity_prevalence')
        if obesity is not None:
            risk_components.append(100 - float(obesity))  # Lower obesity is better
        
        physical_inactivity = risk_factors.get('physical_inactivity_prevalence')
        if physical_inactivity is not None:
            risk_components.append(100 - float(physical_inactivity))  # Lower inactivity is better
        
        # Calculate average
        return sum(risk_components) / len(risk_components) if risk_components else 50.0
    
    def _calculate_prevention_index(self, record: Dict[str, Any]) -> float:
        """Calculate prevention and early intervention index."""
        # This would include screening rates, vaccination coverage, etc.
        # For now, return a placeholder based on healthcare access
        gp_services = record.get('gp_services_per_1000', 0)
        return min(100, (gp_services / 2.0) * 100)
    
    def _calculate_wellbeing_index(self, record: Dict[str, Any]) -> float:
        """Calculate overall wellbeing index."""
        # Combine multiple wellbeing indicators
        components = []
        
        # Health component
        health_index = record.get('composite_health_index')
        if health_index is not None:
            components.append(health_index)
        
        # Social component (SEIFA)
        seifa_scores = record.get('seifa_scores', {})
        irsad = seifa_scores.get('IRSAD')
        if irsad:
            components.append(((irsad - 500) / 500) * 100)
        
        # Environmental component
        env_index = record.get('environmental_health_index')
        if env_index is not None:
            components.append(env_index)
        
        return sum(components) / len(components) if components else 50.0
    
    def _calculate_ses_gradient(self, ses_score: float, health_score: float) -> float:
        """Calculate socioeconomic gradient in health."""
        # Simplified gradient calculation
        # Would typically use regression analysis across multiple areas
        national_ses_avg = 1000.0
        national_health_avg = 75.0
        
        ses_deviation = (ses_score - national_ses_avg) / national_ses_avg
        health_deviation = (health_score - national_health_avg) / national_health_avg
        
        # Return correlation-like measure
        return health_deviation / ses_deviation if ses_deviation != 0 else 0.0
    
    def _calculate_access_inequality(self, gp_services: float, population_density: float) -> float:
        """Calculate healthcare access inequality measure."""
        # Areas with high population density should have proportionally higher services
        expected_services = max(1.0, population_density / 500.0)  # 1 per 500 people per sq km
        actual_ratio = gp_services / expected_services if expected_services > 0 else 0
        
        # Return inequality measure (1.0 = perfect equality)
        return abs(1.0 - actual_ratio)
    
    def _load_standard_populations(self, config: Dict[str, Any]) -> Dict[str, StandardPopulation]:
        """Load standard population structures for age standardisation."""
        # Load Australian standard population (2021 Census)
        australia_2021 = StandardPopulation(
            age_groups={
                'age_0_4': 1565078,
                'age_5_14': 3114845,
                'age_15_24': 3320552,
                'age_25_44': 6810434,
                'age_45_64': 5373771,
                'age_65_plus': 4315013
            },
            total_population=25499881,
            reference_year=2021,
            source="ABS_Census_2021"
        )
        
        return {
            'australia_2021': australia_2021
        }
    
    def _load_indicator_definitions(self, config: Dict[str, Any]) -> Dict[str, IndicatorDefinition]:
        """Load indicator calculation definitions."""
        definitions = {}
        
        # Age-standardised mortality rate
        definitions['asmr'] = IndicatorDefinition(
            indicator_name="Age-Standardised Mortality Rate",
            calculation_method="direct_standardisation",
            required_fields=['mortality_indicators', 'demographic_profile'],
            unit="per 100,000 population",
            interpretation="Lower values indicate better health outcomes",
            higher_is_better=False,
            australian_average=550.0
        )
        
        # Composite health index
        definitions['chi'] = IndicatorDefinition(
            indicator_name="Composite Health Index",
            calculation_method="weighted_average",
            required_fields=['life_expectancy', 'mortality_indicators', 'healthcare_access'],
            unit="index (0-100)",
            interpretation="Higher values indicate better overall health",
            higher_is_better=True,
            australian_average=75.0
        )
        
        return definitions


class SocioeconomicCalculator(BaseTransformer):
    """Derives composite socioeconomic indicators beyond standard SEIFA."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialise the socioeconomic calculator."""
        super().__init__("socioeconomic_calculator", config, logger)
        
        self.logger = get_logger(__name__) if logger is None else logger
        
    def transform(self, data: DataBatch, **kwargs) -> DataBatch:
        """Calculate derived socioeconomic indicators."""
        try:
            enhanced_records = []
            
            for record in data:
                enhanced_record = self.calculate_socioeconomic_indicators(record)
                enhanced_records.append(enhanced_record)
            
            return enhanced_records
            
        except Exception as e:
            self.logger.error(f"Socioeconomic calculation failed: {e}")
            raise TransformationError(f"Socioeconomic calculation failed: {e}") from e
    
    def calculate_socioeconomic_indicators(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive socioeconomic indicators."""
        enhanced_record = record.copy()
        
        # Composite disadvantage index
        enhanced_record['composite_disadvantage_index'] = self._calculate_composite_disadvantage(record)
        
        # Educational opportunity index
        enhanced_record['educational_opportunity_index'] = self._calculate_educational_opportunity(record)
        
        # Economic opportunity index
        enhanced_record['economic_opportunity_index'] = self._calculate_economic_opportunity(record)
        
        return enhanced_record
    
    def get_schema(self) -> Dict[str, str]:
        """Get the expected output schema."""
        return {
            'sa2_code': 'string',
            'composite_disadvantage_index': 'float',
            'educational_opportunity_index': 'float',
            'economic_opportunity_index': 'float'
        }
    
    def _calculate_composite_disadvantage(self, record: Dict[str, Any]) -> Optional[float]:
        """Calculate composite disadvantage index from all SEIFA scores."""
        seifa_scores = record.get('seifa_scores', {})
        if not seifa_scores:
            return None
        
        # Weight different SEIFA indices
        weights = {
            'IRSD': 0.4,  # Socioeconomic disadvantage (most comprehensive)
            'IRSAD': 0.3,  # Education and occupation access
            'IER': 0.2,   # Economic resources
            'IEO': 0.1    # Education and occupation
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for index_type, weight in weights.items():
            if index_type in seifa_scores:
                weighted_sum += seifa_scores[index_type] * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else None
    
    def _calculate_educational_opportunity(self, record: Dict[str, Any]) -> Optional[float]:
        """Calculate educational opportunity index."""
        # Based on IEO and other educational factors
        seifa_scores = record.get('seifa_scores', {})
        ieo_score = seifa_scores.get('IEO')
        
        if ieo_score:
            # Normalise to 0-100 scale
            return ((ieo_score - 500) / 500) * 100
        
        return None
    
    def _calculate_economic_opportunity(self, record: Dict[str, Any]) -> Optional[float]:
        """Calculate economic opportunity index."""
        # Based on IER and economic indicators
        seifa_scores = record.get('seifa_scores', {})
        ier_score = seifa_scores.get('IER')
        
        if ier_score:
            # Normalise to 0-100 scale
            return ((ier_score - 500) / 500) * 100
        
        return None


class EnvironmentalHealthCalculator(BaseTransformer):
    """Calculates environmental health indices and exposure measures."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialise the environmental health calculator."""
        super().__init__("environmental_health_calculator", config, logger)
        
        self.logger = get_logger(__name__) if logger is None else logger
        
    def transform(self, data: DataBatch, **kwargs) -> DataBatch:
        """Calculate environmental health indicators."""
        try:
            enhanced_records = []
            
            for record in data:
                enhanced_record = self.calculate_environmental_indicators(record)
                enhanced_records.append(enhanced_record)
            
            return enhanced_records
            
        except Exception as e:
            self.logger.error(f"Environmental health calculation failed: {e}")
            raise TransformationError(f"Environmental health calculation failed: {e}") from e
    
    def calculate_environmental_indicators(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive environmental health indicators."""
        enhanced_record = record.copy()
        
        # Environmental exposure index
        enhanced_record['environmental_exposure_index'] = self._calculate_exposure_index(record)
        
        # Natural environment access index
        enhanced_record['natural_environment_access_index'] = self._calculate_natural_access(record)
        
        # Built environment quality index
        enhanced_record['built_environment_quality_index'] = self._calculate_built_environment(record)
        
        return enhanced_record
    
    def get_schema(self) -> Dict[str, str]:
        """Get the expected output schema."""
        return {
            'sa2_code': 'string',
            'environmental_exposure_index': 'float',
            'natural_environment_access_index': 'float',
            'built_environment_quality_index': 'float'
        }
    
    def _calculate_exposure_index(self, record: Dict[str, Any]) -> Optional[float]:
        """Calculate environmental exposure risk index."""
        air_quality = record.get('air_quality_index')
        if air_quality is not None:
            # Convert AQI to health risk (lower is better)
            return max(0, 100 - float(air_quality))
        return None
    
    def _calculate_natural_access(self, record: Dict[str, Any]) -> Optional[float]:
        """Calculate natural environment access index."""
        green_space = record.get('green_space_access')
        return float(green_space) if green_space is not None else None
    
    def _calculate_built_environment(self, record: Dict[str, Any]) -> Optional[float]:
        """Calculate built environment quality index."""
        # This would incorporate walkability, public transport, etc.
        # For now, use population density as a proxy
        density = record.get('population_density_per_sq_km', 0)
        
        if density > 0:
            # Optimal density for health (moderate density)
            optimal_density = 2000  # People per sq km
            density_score = 100 - abs(density - optimal_density) / optimal_density * 100
            return max(0, min(100, density_score))
        
        return None


class AccessibilityCalculator(BaseTransformer):
    """Calculates healthcare accessibility measures and travel burden."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialise the accessibility calculator."""
        super().__init__("accessibility_calculator", config, logger)
        
        self.logger = get_logger(__name__) if logger is None else logger
        
    def transform(self, data: DataBatch, **kwargs) -> DataBatch:
        """Calculate accessibility indicators."""
        try:
            enhanced_records = []
            
            for record in data:
                enhanced_record = self.calculate_accessibility_indicators(record)
                enhanced_records.append(enhanced_record)
            
            return enhanced_records
            
        except Exception as e:
            self.logger.error(f"Accessibility calculation failed: {e}")
            raise TransformationError(f"Accessibility calculation failed: {e}") from e
    
    def calculate_accessibility_indicators(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive accessibility indicators."""
        enhanced_record = record.copy()
        
        # Geographic accessibility
        enhanced_record['geographic_accessibility_index'] = self._calculate_geographic_accessibility(record)
        
        # Economic accessibility
        enhanced_record['economic_accessibility_index'] = self._calculate_economic_accessibility(record)
        
        # Service availability index
        enhanced_record['service_availability_index'] = self._calculate_service_availability(record)
        
        return enhanced_record
    
    def get_schema(self) -> Dict[str, str]:
        """Get the expected output schema."""
        return {
            'sa2_code': 'string',
            'geographic_accessibility_index': 'float',
            'economic_accessibility_index': 'float',
            'service_availability_index': 'float'
        }
    
    def _calculate_geographic_accessibility(self, record: Dict[str, Any]) -> Optional[float]:
        """Calculate geographic accessibility to healthcare."""
        # This would use actual distance/travel time data
        # For now, use service density as a proxy
        gp_services = record.get('gp_services_per_1000', 0)
        specialist_services = record.get('specialist_services_per_1000', 0)
        
        total_services = gp_services + specialist_services
        
        # Normalise against national targets
        target_services = 2.5  # Services per 1000 population
        accessibility_score = min(100, (total_services / target_services) * 100)
        
        return accessibility_score
    
    def _calculate_economic_accessibility(self, record: Dict[str, Any]) -> Optional[float]:
        """Calculate economic accessibility to healthcare."""
        bulk_billing_rate = record.get('bulk_billing_rate')
        return float(bulk_billing_rate) if bulk_billing_rate is not None else None
    
    def _calculate_service_availability(self, record: Dict[str, Any]) -> Optional[float]:
        """Calculate healthcare service availability index."""
        gp_services = record.get('gp_services_per_1000', 0)
        specialist_services = record.get('specialist_services_per_1000', 0)
        hospital_beds = record.get('hospital_beds_per_1000', 0)
        
        # Weight different service types
        weighted_availability = (
            gp_services * 0.5 +           # Primary care (50%)
            specialist_services * 0.3 +    # Specialist care (30%)
            hospital_beds * 0.2            # Hospital care (20%)
        )
        
        # Normalise against national average
        national_average = 3.0  # Weighted services per 1000
        return min(100, (weighted_availability / national_average) * 100)