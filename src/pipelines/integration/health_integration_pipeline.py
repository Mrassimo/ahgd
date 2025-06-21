"""
Health data integration pipeline for AHGD project.

This module implements the health data integration workflow that combines
health indicators, mortality data, morbidity data, and healthcare utilisation
into comprehensive health profiles.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from ..base_pipeline import BasePipeline
from ..stage import Stage
from ...transformers.data_integrator import MasterDataIntegrator
from ...transformers.derived_indicators import HealthIndicatorDeriver
from ...utils.integration_rules import DataIntegrationRules, ConflictResolver
from ...utils.interfaces import DataBatch, PipelineError
from ...utils.logging import get_logger, track_lineage
from ...schemas.integrated_schema import MasterHealthRecord, DataIntegrationLevel


class HealthDataValidationStage(Stage):
    """Validates health data before integration."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("health_data_validation", config)
        self.validation_rules = config.get('validation_rules', {})
        
    def execute(self, data: DataBatch, **kwargs) -> DataBatch:
        """Validate health data inputs."""
        validated_data = []
        validation_errors = []
        
        for record in data:
            sa2_code = record.get('sa2_code')
            
            # Check mandatory health fields
            required_fields = ['total_population', 'demographic_profile']
            missing_fields = []
            
            for field in required_fields:
                if field not in record or record[field] is None:
                    missing_fields.append(field)
            
            if missing_fields:
                validation_errors.append({
                    'sa2_code': sa2_code,
                    'error': 'missing_required_fields',
                    'fields': missing_fields
                })
                continue
            
            # Validate data ranges
            population = record.get('total_population', 0)
            if population < 0 or population > 1000000:  # Reasonable bounds for SA2
                validation_errors.append({
                    'sa2_code': sa2_code,
                    'error': 'population_out_of_range',
                    'value': population
                })
                continue
            
            # Validate mortality rates
            mortality_indicators = record.get('mortality_indicators', {})
            for indicator, rate in mortality_indicators.items():
                if isinstance(rate, (int, float)) and (rate < 0 or rate > 10000):  # per 100,000
                    validation_errors.append({
                        'sa2_code': sa2_code,
                        'error': 'mortality_rate_out_of_range',
                        'indicator': indicator,
                        'value': rate
                    })
            
            validated_data.append(record)
        
        # Log validation results
        if validation_errors:
            self.logger.warning(f"Health data validation found {len(validation_errors)} errors")
            for error in validation_errors[:5]:  # Log first 5 errors
                self.logger.warning(f"Validation error: {error}")
        
        self.logger.info(f"Health data validation: {len(validated_data)} records validated")
        
        return validated_data


class HealthIndicatorStandardisationStage(Stage):
    """Standardises health indicators from different sources."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("health_indicator_standardisation", config)
        self.standardisation_mapping = config.get('standardisation_mapping', {})
        
    def execute(self, data: DataBatch, **kwargs) -> DataBatch:
        """Standardise health indicators."""
        standardised_data = []
        
        for record in data:
            standardised_record = record.copy()
            
            # Standardise field names
            for source_field, target_field in self.standardisation_mapping.items():
                if source_field in record:
                    standardised_record[target_field] = record[source_field]
                    if source_field != target_field:
                        del standardised_record[source_field]
            
            # Standardise units (rates to per 100,000)
            mortality_indicators = standardised_record.get('mortality_indicators', {})
            for indicator, rate in mortality_indicators.items():
                if isinstance(rate, (int, float)):
                    # Assume rates are per 1,000 and convert to per 100,000
                    if 'per_1000' in indicator:
                        mortality_indicators[indicator.replace('per_1000', 'per_100000')] = rate * 100
                        del mortality_indicators[indicator]
            
            # Standardise prevalence rates (ensure they're percentages)
            disease_prevalence = standardised_record.get('chronic_disease_prevalence', {})
            for disease, prevalence in disease_prevalence.items():
                if isinstance(prevalence, (int, float)):
                    # Ensure prevalence is between 0-100 (percentage)
                    if prevalence > 1.0:
                        disease_prevalence[disease] = min(100.0, prevalence)
                    else:
                        disease_prevalence[disease] = prevalence * 100  # Convert from proportion
            
            standardised_data.append(standardised_record)
        
        self.logger.info(f"Standardised {len(standardised_data)} health records")
        return standardised_data


class HealthDataIntegrationStage(Stage):
    """Integrates health data from multiple sources."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("health_data_integration", config)
        
        # Initialise integration components
        self.integration_rules = DataIntegrationRules(config.get('integration_rules', {}))
        self.conflict_resolver = ConflictResolver(config.get('conflict_resolution', {}))
        
        # Field priorities for health data
        self.health_field_priorities = config.get('health_field_priorities', {
            'life_expectancy': ['health_indicators', 'aihw', 'census'],
            'mortality_rates': ['health_indicators', 'abs', 'aihw'],
            'morbidity_rates': ['health_indicators', 'medicare_pbs'],
            'healthcare_utilisation': ['medicare_pbs', 'health_indicators']
        })
        
    def execute(self, data: DataBatch, **kwargs) -> DataBatch:
        """Integrate health data from multiple sources."""
        integrated_data = []
        integration_conflicts = []
        
        for record in data:
            sa2_code = record.get('sa2_code')
            
            # Extract health data by source
            health_sources = self._extract_health_sources(record)
            
            # Integrate each health indicator
            integrated_health = {}
            
            # Life expectancy integration
            life_expectancy = self._integrate_life_expectancy(sa2_code, health_sources)
            if life_expectancy:
                integrated_health.update(life_expectancy)
            
            # Mortality indicators integration
            mortality = self._integrate_mortality_indicators(sa2_code, health_sources)
            if mortality:
                integrated_health['mortality_indicators'] = mortality
            
            # Morbidity indicators integration
            morbidity = self._integrate_morbidity_indicators(sa2_code, health_sources)
            if morbidity:
                integrated_health['chronic_disease_prevalence'] = morbidity
            
            # Healthcare utilisation integration
            utilisation = self._integrate_healthcare_utilisation(sa2_code, health_sources)
            if utilisation:
                integrated_health.update(utilisation)
            
            # Mental health indicators integration
            mental_health = self._integrate_mental_health_indicators(sa2_code, health_sources)
            if mental_health:
                integrated_health['mental_health_indicators'] = mental_health
            
            # Risk factors integration
            risk_factors = self._integrate_risk_factors(sa2_code, health_sources)
            if risk_factors:
                integrated_health['risk_factors'] = risk_factors
            
            # Create integrated record
            integrated_record = record.copy()
            integrated_record.update(integrated_health)
            
            # Add integration metadata
            integrated_record['health_integration_timestamp'] = datetime.utcnow()
            integrated_record['health_sources_count'] = len(health_sources)
            
            integrated_data.append(integrated_record)
        
        self.logger.info(f"Integrated health data for {len(integrated_data)} SA2 areas")
        if integration_conflicts:
            self.logger.info(f"Resolved {len(integration_conflicts)} health data conflicts")
        
        return integrated_data
    
    def _extract_health_sources(self, record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract health data organised by source."""
        sources = {}
        
        # Expected health data sources
        health_source_types = [
            'health_indicators', 'aihw', 'medicare_pbs', 'abs_health', 'nhmd'
        ]
        
        for source_type in health_source_types:
            if source_type in record:
                sources[source_type] = record[source_type]
        
        return sources
    
    def _integrate_life_expectancy(
        self, 
        sa2_code: str, 
        sources: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Integrate life expectancy data."""
        life_expectancy_data = {}
        
        # Look for life expectancy in sources
        candidates = {}
        for source_name, source_data in sources.items():
            for field in ['life_expectancy', 'life_expectancy_male', 'life_expectancy_female']:
                if field in source_data and source_data[field] is not None:
                    if field not in candidates:
                        candidates[field] = {}
                    candidates[field][source_name] = source_data[field]
        
        # Select best values
        for field, source_values in candidates.items():
            if len(source_values) == 1:
                # Only one source, use it
                life_expectancy_data[field] = list(source_values.values())[0]
            elif len(source_values) > 1:
                # Multiple sources, need conflict resolution
                # Use priority order for life expectancy
                priority_order = self.health_field_priorities.get('life_expectancy', [])
                
                selected_value = None
                for priority_source in priority_order:
                    if priority_source in source_values:
                        selected_value = source_values[priority_source]
                        break
                
                if selected_value is None:
                    # No priority source found, use highest quality
                    selected_value = list(source_values.values())[0]
                
                life_expectancy_data[field] = selected_value
        
        return life_expectancy_data
    
    def _integrate_mortality_indicators(
        self, 
        sa2_code: str, 
        sources: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Integrate mortality indicators."""
        mortality_indicators = {}
        
        # Common mortality indicators
        mortality_fields = [
            'all_cause_mortality_rate',
            'cardiovascular_mortality_rate',
            'cancer_mortality_rate',
            'respiratory_mortality_rate',
            'diabetes_mortality_rate',
            'suicide_mortality_rate',
            'infant_mortality_rate',
            'avoidable_mortality_rate'
        ]
        
        for field in mortality_fields:
            candidates = {}
            for source_name, source_data in sources.items():
                if field in source_data and source_data[field] is not None:
                    candidates[source_name] = source_data[field]
            
            if candidates:
                # Use priority order for mortality rates
                priority_order = self.health_field_priorities.get('mortality_rates', [])
                
                selected_value = None
                for priority_source in priority_order:
                    if priority_source in candidates:
                        selected_value = candidates[priority_source]
                        break
                
                if selected_value is None:
                    selected_value = list(candidates.values())[0]
                
                mortality_indicators[field] = float(selected_value)
        
        return mortality_indicators
    
    def _integrate_morbidity_indicators(
        self, 
        sa2_code: str, 
        sources: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Integrate morbidity and disease prevalence indicators."""
        morbidity_indicators = {}
        
        # Common disease prevalence indicators
        disease_fields = [
            'diabetes_prevalence',
            'hypertension_prevalence',
            'heart_disease_prevalence',
            'asthma_prevalence',
            'copd_prevalence',
            'cancer_prevalence',
            'mental_health_condition_prevalence',
            'obesity_prevalence',
            'arthritis_prevalence'
        ]
        
        for field in disease_fields:
            candidates = {}
            for source_name, source_data in sources.items():
                # Look in both main level and chronic_disease_prevalence sub-dict
                if field in source_data and source_data[field] is not None:
                    candidates[source_name] = source_data[field]
                elif 'chronic_disease_prevalence' in source_data:
                    chronic_diseases = source_data['chronic_disease_prevalence']
                    if isinstance(chronic_diseases, dict) and field in chronic_diseases:
                        candidates[source_name] = chronic_diseases[field]
            
            if candidates:
                # Use priority order for morbidity rates
                priority_order = self.health_field_priorities.get('morbidity_rates', [])
                
                selected_value = None
                for priority_source in priority_order:
                    if priority_source in candidates:
                        selected_value = candidates[priority_source]
                        break
                
                if selected_value is None:
                    selected_value = list(candidates.values())[0]
                
                # Ensure prevalence is a valid percentage
                prevalence_value = float(selected_value)
                if 0 <= prevalence_value <= 100:
                    morbidity_indicators[field] = prevalence_value
        
        return morbidity_indicators
    
    def _integrate_healthcare_utilisation(
        self, 
        sa2_code: str, 
        sources: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Integrate healthcare utilisation indicators."""
        utilisation_data = {}
        
        # Healthcare utilisation indicators
        utilisation_fields = [
            'gp_services_per_1000',
            'specialist_services_per_1000',
            'bulk_billing_rate',
            'emergency_dept_presentations_per_1000',
            'hospital_admissions_per_1000',
            'pharmaceutical_prescriptions_per_1000'
        ]
        
        for field in utilisation_fields:
            candidates = {}
            for source_name, source_data in sources.items():
                if field in source_data and source_data[field] is not None:
                    candidates[source_name] = source_data[field]
                elif 'healthcare_utilisation' in source_data:
                    utilisation = source_data['healthcare_utilisation']
                    if isinstance(utilisation, dict) and field in utilisation:
                        candidates[source_name] = utilisation[field]
            
            if candidates:
                # Use priority order for utilisation data
                priority_order = self.health_field_priorities.get('healthcare_utilisation', [])
                
                selected_value = None
                for priority_source in priority_order:
                    if priority_source in candidates:
                        selected_value = candidates[priority_source]
                        break
                
                if selected_value is None:
                    selected_value = list(candidates.values())[0]
                
                utilisation_data[field] = float(selected_value)
        
        return utilisation_data
    
    def _integrate_mental_health_indicators(
        self, 
        sa2_code: str, 
        sources: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Integrate mental health indicators."""
        mental_health_data = {}
        
        # Mental health indicators
        mental_health_fields = [
            'psychological_distress_high',
            'anxiety_prevalence',
            'depression_prevalence',
            'suicide_ideation_prevalence',
            'mental_health_service_utilisation',
            'psychologist_services_per_1000',
            'psychiatrist_services_per_1000'
        ]
        
        for field in mental_health_fields:
            candidates = {}
            for source_name, source_data in sources.items():
                if field in source_data and source_data[field] is not None:
                    candidates[source_name] = source_data[field]
                elif 'mental_health_indicators' in source_data:
                    mental_health = source_data['mental_health_indicators']
                    if isinstance(mental_health, dict) and field in mental_health:
                        candidates[source_name] = mental_health[field]
            
            if candidates:
                selected_value = list(candidates.values())[0]  # Take first available
                mental_health_data[field] = float(selected_value)
        
        return mental_health_data
    
    def _integrate_risk_factors(
        self, 
        sa2_code: str, 
        sources: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Integrate risk factor prevalence data."""
        risk_factors_data = {}
        
        # Risk factor indicators
        risk_factor_fields = [
            'smoking_prevalence',
            'physical_inactivity_prevalence',
            'harmful_alcohol_use_prevalence',
            'overweight_prevalence',
            'obesity_prevalence',
            'high_blood_pressure_prevalence',
            'high_cholesterol_prevalence'
        ]
        
        for field in risk_factor_fields:
            candidates = {}
            for source_name, source_data in sources.items():
                if field in source_data and source_data[field] is not None:
                    candidates[source_name] = source_data[field]
                elif 'risk_factors' in source_data:
                    risk_factors = source_data['risk_factors']
                    if isinstance(risk_factors, dict) and field in risk_factors:
                        candidates[source_name] = risk_factors[field]
            
            if candidates:
                selected_value = list(candidates.values())[0]  # Take first available
                
                # Ensure risk factor is a valid percentage
                risk_value = float(selected_value)
                if 0 <= risk_value <= 100:
                    risk_factors_data[field] = risk_value
        
        return risk_factors_data


class HealthQualityAssessmentStage(Stage):
    """Assesses health data quality and completeness."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("health_quality_assessment", config)
        self.quality_indicators = config.get('quality_indicators', [])
        
    def execute(self, data: DataBatch, **kwargs) -> DataBatch:
        """Assess health data quality."""
        quality_assessed_data = []
        
        for record in data:
            assessed_record = record.copy()
            
            # Calculate health data completeness
            health_completeness = self._calculate_health_completeness(record)
            assessed_record['health_data_completeness'] = health_completeness
            
            # Assess health indicator quality
            quality_scores = self._assess_health_quality(record)
            assessed_record['health_quality_scores'] = quality_scores
            
            # Overall health data quality score
            overall_quality = sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0
            assessed_record['health_overall_quality_score'] = overall_quality
            
            quality_assessed_data.append(assessed_record)
        
        self.logger.info(f"Assessed health data quality for {len(quality_assessed_data)} records")
        return quality_assessed_data
    
    def _calculate_health_completeness(self, record: Dict[str, Any]) -> float:
        """Calculate health data completeness score."""
        essential_health_fields = [
            'life_expectancy',
            'mortality_indicators',
            'chronic_disease_prevalence',
            'gp_services_per_1000',
            'bulk_billing_rate'
        ]
        
        available_fields = 0
        for field in essential_health_fields:
            if field in record and record[field] is not None:
                if isinstance(record[field], dict):
                    # For nested dictionaries, check if they have content
                    if record[field]:
                        available_fields += 1
                else:
                    available_fields += 1
        
        return (available_fields / len(essential_health_fields)) * 100
    
    def _assess_health_quality(self, record: Dict[str, Any]) -> Dict[str, float]:
        """Assess quality of specific health indicators."""
        quality_scores = {}
        
        # Life expectancy quality
        life_expectancy = record.get('life_expectancy')
        if life_expectancy:
            # Check if within reasonable bounds (70-90 years)
            if 70 <= life_expectancy <= 90:
                quality_scores['life_expectancy'] = 1.0
            else:
                quality_scores['life_expectancy'] = 0.5
        
        # Mortality rates quality
        mortality_indicators = record.get('mortality_indicators', {})
        if mortality_indicators:
            valid_rates = 0
            total_rates = len(mortality_indicators)
            
            for rate in mortality_indicators.values():
                if isinstance(rate, (int, float)) and 0 <= rate <= 5000:  # per 100,000
                    valid_rates += 1
            
            quality_scores['mortality_indicators'] = valid_rates / total_rates if total_rates > 0 else 0
        
        # Disease prevalence quality
        disease_prevalence = record.get('chronic_disease_prevalence', {})
        if disease_prevalence:
            valid_prevalences = 0
            total_prevalences = len(disease_prevalence)
            
            for prevalence in disease_prevalence.values():
                if isinstance(prevalence, (int, float)) and 0 <= prevalence <= 100:
                    valid_prevalences += 1
            
            quality_scores['chronic_disease_prevalence'] = valid_prevalences / total_prevalences if total_prevalences > 0 else 0
        
        return quality_scores


class HealthIntegrationPipeline(BasePipeline):
    """
    Complete health data integration pipeline.
    
    Orchestrates the end-to-end health data integration process from
    validation through quality assessment.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialise the health integration pipeline.
        
        Args:
            config: Pipeline configuration
            logger: Optional logger instance
        """
        super().__init__("health_integration_pipeline", config, logger)
        
        # Create pipeline stages
        self.stages = [
            HealthDataValidationStage(config.get('validation', {})),
            HealthIndicatorStandardisationStage(config.get('standardisation', {})),
            HealthDataIntegrationStage(config.get('integration', {})),
            HealthQualityAssessmentStage(config.get('quality_assessment', {}))
        ]
        
        # Integration tracking
        self.integration_metrics = {
            'records_processed': 0,
            'validation_errors': 0,
            'integration_conflicts': 0,
            'quality_scores': []
        }
    
    def execute(self, data: DataBatch, **kwargs) -> DataBatch:
        """Execute the complete health integration pipeline."""
        self.logger.info("Starting health data integration pipeline")
        
        # Track lineage
        track_lineage(
            input_data="raw_health_data",
            output_data="integrated_health_data",
            transformation="health_integration_pipeline"
        )
        
        # Execute pipeline stages
        current_data = data
        
        for stage in self.stages:
            try:
                self.logger.info(f"Executing stage: {stage.name}")
                current_data = stage.execute(current_data, **kwargs)
                self.logger.info(f"Stage {stage.name} completed: {len(current_data)} records")
                
            except Exception as e:
                self.logger.error(f"Stage {stage.name} failed: {e}")
                raise PipelineError(f"Health integration pipeline failed at stage {stage.name}: {e}")
        
        # Update metrics
        self.integration_metrics['records_processed'] = len(current_data)
        
        # Calculate average quality scores
        quality_scores = []
        for record in current_data:
            if 'health_overall_quality_score' in record:
                quality_scores.append(record['health_overall_quality_score'])
        
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            self.integration_metrics['quality_scores'].append(avg_quality)
            self.logger.info(f"Average health data quality score: {avg_quality:.2f}")
        
        self.logger.info(f"Health integration pipeline completed: {len(current_data)} records integrated")
        
        return current_data
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get integration pipeline metrics."""
        return self.integration_metrics.copy()
    
    def validate_pipeline_config(self) -> List[str]:
        """Validate pipeline configuration."""
        errors = []
        
        # Check required configuration sections
        required_sections = ['validation', 'standardisation', 'integration', 'quality_assessment']
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Missing required configuration section: {section}")
        
        return errors