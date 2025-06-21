"""
Demographic data integration pipeline for AHGD project.

This module implements the demographic data integration workflow that combines
Census data, population statistics, age-sex distributions, and socioeconomic
indicators into comprehensive demographic profiles.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from ..base_pipeline import BasePipeline
from ..stage import PipelineStage
from ...transformers.data_integrator import DemographicProfileBuilder
from ...utils.integration_rules import DataIntegrationRules, ConflictResolver
from ...utils.interfaces import DataBatch
from ..base_pipeline import PipelineError
from ...utils.logging import get_logger, track_lineage
from schemas.seifa_schema import SEIFAIndexType


class DemographicDataValidationStage(PipelineStage):
    """Validates demographic data before integration."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("demographic_data_validation", config)
        self.validation_thresholds = config.get('validation_thresholds', {})
        
    def execute(self, data: DataBatch, **kwargs) -> DataBatch:
        """Validate demographic data inputs."""
        validated_data = []
        validation_errors = []
        
        for record in data:
            sa2_code = record.get('sa2_code')
            
            # Validate population totals
            validation_result = self._validate_population_totals(record)
            if validation_result['errors']:
                validation_errors.extend([
                    {'sa2_code': sa2_code, **error} for error in validation_result['errors']
                ])
            
            # Validate age distributions
            age_validation = self._validate_age_distributions(record)
            if age_validation['errors']:
                validation_errors.extend([
                    {'sa2_code': sa2_code, **error} for error in age_validation['errors']
                ])
            
            # Validate SEIFA data
            seifa_validation = self._validate_seifa_data(record)
            if seifa_validation['errors']:
                validation_errors.extend([
                    {'sa2_code': sa2_code, **error} for error in seifa_validation['errors']
                ])
            
            # Only include records that pass validation
            if not validation_result['errors'] and not age_validation['errors']:
                validated_data.append(record)
        
        # Log validation results
        if validation_errors:
            self.logger.warning(f"Demographic validation found {len(validation_errors)} errors")
            # Log sample of errors
            for error in validation_errors[:3]:
                self.logger.warning(f"Validation error: {error}")
        
        self.logger.info(f"Demographic validation: {len(validated_data)} records validated")
        
        return validated_data
    
    def _validate_population_totals(self, record: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Validate population total consistency."""
        errors = []
        
        total_population = record.get('total_population')
        male_population = record.get('male_population')
        female_population = record.get('female_population')
        
        # Check if populations are non-negative
        for pop_type, population in [
            ('total', total_population),
            ('male', male_population),
            ('female', female_population)
        ]:
            if population is not None and population < 0:
                errors.append({
                    'error': 'negative_population',
                    'population_type': pop_type,
                    'value': population
                })
        
        # Check if male + female equals total (within tolerance)
        if all(p is not None for p in [total_population, male_population, female_population]):
            sex_sum = male_population + female_population
            difference = abs(total_population - sex_sum)
            tolerance = max(10, total_population * 0.01)  # 1% or minimum 10 people
            
            if difference > tolerance:
                errors.append({
                    'error': 'population_mismatch',
                    'total_population': total_population,
                    'sex_sum': sex_sum,
                    'difference': difference
                })
        
        # Check reasonable population bounds for SA2
        if total_population is not None:
            min_pop = self.validation_thresholds.get('min_sa2_population', 0)
            max_pop = self.validation_thresholds.get('max_sa2_population', 50000)
            
            if total_population < min_pop or total_population > max_pop:
                errors.append({
                    'error': 'population_out_of_bounds',
                    'population': total_population,
                    'min_bound': min_pop,
                    'max_bound': max_pop
                })
        
        return {'errors': errors}
    
    def _validate_age_distributions(self, record: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Validate age distribution data."""
        errors = []
        
        demographic_profile = record.get('demographic_profile', {})
        age_groups = demographic_profile.get('age_groups', {})
        
        if age_groups:
            # Check if age group totals are reasonable
            total_from_ages = sum(age_groups.values())
            total_population = record.get('total_population', 0)
            
            if total_population > 0:
                age_difference = abs(total_from_ages - total_population)
                tolerance = max(10, total_population * 0.05)  # 5% tolerance
                
                if age_difference > tolerance:
                    errors.append({
                        'error': 'age_distribution_mismatch',
                        'total_from_ages': total_from_ages,
                        'total_population': total_population,
                        'difference': age_difference
                    })
            
            # Check for negative age group values
            for age_group, count in age_groups.items():
                if count < 0:
                    errors.append({
                        'error': 'negative_age_group',
                        'age_group': age_group,
                        'count': count
                    })
        
        return {'errors': errors}
    
    def _validate_seifa_data(self, record: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Validate SEIFA index data."""
        errors = []
        
        seifa_scores = record.get('seifa_scores', {})
        seifa_deciles = record.get('seifa_deciles', {})
        
        # Validate SEIFA scores (should be around 500-1500 range)
        for index_type, score in seifa_scores.items():
            if isinstance(score, (int, float)):
                if not (200 <= score <= 1800):  # Reasonable bounds
                    errors.append({
                        'error': 'seifa_score_out_of_bounds',
                        'index_type': index_type,
                        'score': score
                    })
        
        # Validate SEIFA deciles (should be 1-10)
        for index_type, decile in seifa_deciles.items():
            if isinstance(decile, int):
                if not (1 <= decile <= 10):
                    errors.append({
                        'error': 'seifa_decile_out_of_bounds',
                        'index_type': index_type,
                        'decile': decile
                    })
        
        # Check consistency between scores and deciles
        for index_type in seifa_scores:
            if index_type in seifa_deciles:
                score = seifa_scores[index_type]
                decile = seifa_deciles[index_type]
                
                # Basic consistency check (this would be more sophisticated with actual decile thresholds)
                if isinstance(score, (int, float)) and isinstance(decile, int):
                    if score < 800 and decile > 7:  # Low score should not have high decile
                        errors.append({
                            'error': 'seifa_score_decile_inconsistency',
                            'index_type': index_type,
                            'score': score,
                            'decile': decile
                        })
        
        return {'errors': errors}


class PopulationStandardisationStage(PipelineStage):
    """Standardises population data from different sources."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("population_standardisation", config)
        self.age_group_mappings = config.get('age_group_mappings', {})
        
    def execute(self, data: DataBatch, **kwargs) -> DataBatch:
        """Standardise population data."""
        standardised_data = []
        
        for record in data:
            standardised_record = record.copy()
            
            # Standardise age group classifications
            standardised_record = self._standardise_age_groups(standardised_record)
            
            # Standardise population density calculation
            standardised_record = self._standardise_population_density(standardised_record)
            
            # Standardise Indigenous population reporting
            standardised_record = self._standardise_indigenous_data(standardised_record)
            
            standardised_data.append(standardised_record)
        
        self.logger.info(f"Standardised {len(standardised_data)} demographic records")
        return standardised_data
    
    def _standardise_age_groups(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Standardise age group classifications to ABS standards."""
        demographic_profile = record.get('demographic_profile', {})
        age_groups = demographic_profile.get('age_groups', {})
        
        if not age_groups:
            return record
        
        # Standard ABS age groups
        standard_age_groups = {
            'age_0_4': 0,
            'age_5_14': 0,
            'age_15_24': 0,
            'age_25_44': 0,
            'age_45_64': 0,
            'age_65_plus': 0
        }
        
        # Map existing age groups to standard groups
        for source_group, count in age_groups.items():
            if isinstance(count, (int, float)):
                # Map to standard groups based on age range
                if any(age in source_group.lower() for age in ['0_4', '0-4', 'under_5']):
                    standard_age_groups['age_0_4'] += count
                elif any(age in source_group.lower() for age in ['5_14', '5-14', '5_9', '10_14']):
                    standard_age_groups['age_5_14'] += count
                elif any(age in source_group.lower() for age in ['15_24', '15-24', '15_19', '20_24']):
                    standard_age_groups['age_15_24'] += count
                elif any(age in source_group.lower() for age in ['25_44', '25-44', '25_34', '35_44']):
                    standard_age_groups['age_25_44'] += count
                elif any(age in source_group.lower() for age in ['45_64', '45-64', '45_54', '55_64']):
                    standard_age_groups['age_45_64'] += count
                elif any(age in source_group.lower() for age in ['65', 'over_65', '65_plus']):
                    standard_age_groups['age_65_plus'] += count
        
        # Update record with standardised age groups
        if 'demographic_profile' not in record:
            record['demographic_profile'] = {}
        record['demographic_profile']['age_groups'] = standard_age_groups
        
        return record
    
    def _standardise_population_density(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure consistent population density calculation."""
        total_population = record.get('total_population', 0)
        
        # Try to get area from different possible sources
        area_sq_km = None
        if 'boundary_data' in record:
            boundary_data = record['boundary_data']
            if isinstance(boundary_data, dict):
                area_sq_km = boundary_data.get('area_sq_km')
        
        # Fall back to direct area field
        if area_sq_km is None:
            area_sq_km = record.get('area_sq_km')
        
        # Calculate population density
        if area_sq_km and area_sq_km > 0:
            population_density = total_population / area_sq_km
            record['population_density_per_sq_km'] = population_density
        else:
            record['population_density_per_sq_km'] = 0.0
        
        return record
    
    def _standardise_indigenous_data(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Standardise Indigenous population data."""
        # Look for Indigenous population data in various forms
        indigenous_count = None
        indigenous_sources = [
            'indigenous_population',
            'aboriginal_torres_strait_islander_population',
            'atsi_population'
        ]
        
        for source_field in indigenous_sources:
            if source_field in record and record[source_field] is not None:
                indigenous_count = record[source_field]
                break
        
        # Calculate percentage if we have both count and total
        total_population = record.get('total_population', 0)
        if indigenous_count is not None and total_population > 0:
            indigenous_percentage = (indigenous_count / total_population) * 100
            record['indigenous_population_percentage'] = indigenous_percentage
            
            # Also store the count if not already present
            if 'indigenous_population' not in record:
                record['indigenous_population'] = indigenous_count
        
        return record


class SEIFAIntegrationStage(PipelineStage):
    """Integrates SEIFA socioeconomic data."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("seifa_integration", config)
        self.seifa_priorities = config.get('seifa_priorities', ['seifa', 'abs'])
        
    def execute(self, data: DataBatch, **kwargs) -> DataBatch:
        """Integrate SEIFA data from multiple sources."""
        integrated_data = []
        
        for record in data:
            integrated_record = record.copy()
            
            # Integrate SEIFA scores and deciles
            seifa_data = self._integrate_seifa_indices(record)
            integrated_record.update(seifa_data)
            
            # Calculate derived socioeconomic indicators
            derived_indicators = self._calculate_derived_socioeconomic(integrated_record)
            integrated_record.update(derived_indicators)
            
            integrated_data.append(integrated_record)
        
        self.logger.info(f"Integrated SEIFA data for {len(integrated_data)} records")
        return integrated_data
    
    def _integrate_seifa_indices(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate SEIFA indices from multiple sources."""
        seifa_scores = {}
        seifa_deciles = {}
        
        # Look for SEIFA data in different source formats
        seifa_sources = []
        
        # Direct SEIFA fields
        if 'seifa_scores' in record:
            seifa_sources.append(('direct', record))
        
        # Source-specific SEIFA data
        for source in ['seifa', 'abs', 'census']:
            if source in record and isinstance(record[source], dict):
                seifa_sources.append((source, record[source]))
        
        # Extract SEIFA indices
        for index_type in SEIFAIndexType:
            score_field = f"seifa_{index_type.value}_score"
            decile_field = f"seifa_{index_type.value}_decile"
            
            # Find best source for this index
            for source_priority in self.seifa_priorities:
                for source_name, source_data in seifa_sources:
                    if source_name == source_priority or source_name == 'direct':
                        if score_field in source_data and source_data[score_field] is not None:
                            seifa_scores[index_type] = float(source_data[score_field])
                        if decile_field in source_data and source_data[decile_field] is not None:
                            seifa_deciles[index_type] = int(source_data[decile_field])
                        
                        # Break if we found this index in this source
                        if index_type in seifa_scores or index_type in seifa_deciles:
                            break
                
                # Break if we found this index
                if index_type in seifa_scores or index_type in seifa_deciles:
                    break
        
        return {
            'seifa_scores': seifa_scores,
            'seifa_deciles': seifa_deciles
        }
    
    def _calculate_derived_socioeconomic(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate derived socioeconomic indicators."""
        derived = {}
        
        seifa_deciles = record.get('seifa_deciles', {})
        
        # Overall disadvantage category based on IRSD
        if SEIFAIndexType.IRSD in seifa_deciles:
            irsd_decile = seifa_deciles[SEIFAIndexType.IRSD]
            
            if irsd_decile <= 2:
                disadvantage_category = "Most Disadvantaged"
            elif irsd_decile <= 4:
                disadvantage_category = "Disadvantaged"
            elif irsd_decile <= 6:
                disadvantage_category = "Average"
            elif irsd_decile <= 8:
                disadvantage_category = "Advantaged"
            else:
                disadvantage_category = "Most Advantaged"
            
            derived['disadvantage_category'] = disadvantage_category
        
        # Socioeconomic classification
        if SEIFAIndexType.IRSAD in seifa_deciles:
            irsad_decile = seifa_deciles[SEIFAIndexType.IRSAD]
            
            if irsad_decile <= 3:
                socioeconomic_category = "Low"
            elif irsad_decile <= 7:
                socioeconomic_category = "Medium"
            else:
                socioeconomic_category = "High"
            
            derived['socioeconomic_category'] = socioeconomic_category
        
        return derived


class DemographicQualityAssessmentStage(PipelineStage):
    """Assesses demographic data quality and completeness."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("demographic_quality_assessment", config)
        self.completeness_weights = config.get('completeness_weights', {})
        
    def execute(self, data: DataBatch, **kwargs) -> DataBatch:
        """Assess demographic data quality."""
        quality_assessed_data = []
        
        for record in data:
            assessed_record = record.copy()
            
            # Calculate demographic completeness
            completeness_scores = self._calculate_demographic_completeness(record)
            assessed_record['demographic_completeness_scores'] = completeness_scores
            
            # Overall demographic quality
            overall_completeness = sum(completeness_scores.values()) / len(completeness_scores)
            assessed_record['demographic_overall_completeness'] = overall_completeness
            
            # Data quality flags
            quality_flags = self._generate_quality_flags(record)
            assessed_record['demographic_quality_flags'] = quality_flags
            
            quality_assessed_data.append(assessed_record)
        
        self.logger.info(f"Assessed demographic quality for {len(quality_assessed_data)} records")
        return quality_assessed_data
    
    def _calculate_demographic_completeness(self, record: Dict[str, Any]) -> Dict[str, float]:
        """Calculate completeness scores for demographic components."""
        completeness_scores = {}
        
        # Population data completeness
        population_fields = ['total_population', 'male_population', 'female_population']
        population_available = sum(1 for field in population_fields if record.get(field) is not None)
        completeness_scores['population'] = population_available / len(population_fields)
        
        # Age distribution completeness
        age_groups = record.get('demographic_profile', {}).get('age_groups', {})
        expected_age_groups = 6  # Standard ABS age groups
        age_available = sum(1 for count in age_groups.values() if count is not None and count >= 0)
        completeness_scores['age_distribution'] = min(1.0, age_available / expected_age_groups)
        
        # SEIFA completeness
        seifa_scores = record.get('seifa_scores', {})
        seifa_deciles = record.get('seifa_deciles', {})
        expected_seifa_indices = len(SEIFAIndexType)
        
        seifa_scores_available = len([s for s in seifa_scores.values() if s is not None])
        seifa_deciles_available = len([d for d in seifa_deciles.values() if d is not None])
        
        seifa_completeness = (seifa_scores_available + seifa_deciles_available) / (expected_seifa_indices * 2)
        completeness_scores['seifa'] = min(1.0, seifa_completeness)
        
        # Geographic completeness
        geographic_fields = ['sa2_code', 'sa2_name', 'boundary_data']
        geographic_available = sum(1 for field in geographic_fields if record.get(field) is not None)
        completeness_scores['geographic'] = geographic_available / len(geographic_fields)
        
        return completeness_scores
    
    def _generate_quality_flags(self, record: Dict[str, Any]) -> List[str]:
        """Generate quality flags for demographic data."""
        flags = []
        
        # High quality population data
        total_pop = record.get('total_population', 0)
        male_pop = record.get('male_population')
        female_pop = record.get('female_population')
        
        if all(p is not None for p in [total_pop, male_pop, female_pop]):
            if abs(total_pop - (male_pop + female_pop)) <= max(5, total_pop * 0.005):
                flags.append('high_quality_population')
        
        # Complete SEIFA data
        seifa_scores = record.get('seifa_scores', {})
        if len(seifa_scores) == len(SEIFAIndexType):
            flags.append('complete_seifa')
        
        # Complete age distribution
        age_groups = record.get('demographic_profile', {}).get('age_groups', {})
        if len(age_groups) >= 6:
            flags.append('complete_age_distribution')
        
        # Indigenous data available
        if 'indigenous_population_percentage' in record:
            flags.append('indigenous_data_available')
        
        return flags


class DemographicIntegrationPipeline(BasePipeline):
    """
    Complete demographic data integration pipeline.
    
    Orchestrates the end-to-end demographic data integration process from
    validation through quality assessment.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialise the demographic integration pipeline.
        
        Args:
            config: Pipeline configuration
            logger: Optional logger instance
        """
        super().__init__("demographic_integration_pipeline", config, logger)
        
        # Create pipeline stages
        self.stages = [
            DemographicDataValidationStage(config.get('validation', {})),
            PopulationStandardisationStage(config.get('standardisation', {})),
            SEIFAIntegrationStage(config.get('seifa_integration', {})),
            DemographicQualityAssessmentStage(config.get('quality_assessment', {}))
        ]
        
        # Integration tracking
        self.integration_metrics = {
            'records_processed': 0,
            'validation_errors': 0,
            'seifa_completeness': [],
            'population_accuracy': []
        }
    
    def execute(self, data: DataBatch, **kwargs) -> DataBatch:
        """Execute the complete demographic integration pipeline."""
        self.logger.info("Starting demographic data integration pipeline")
        
        # Track lineage
        track_lineage(
            input_data="raw_demographic_data",
            output_data="integrated_demographic_data",
            transformation="demographic_integration_pipeline"
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
                raise PipelineError(f"Demographic integration pipeline failed at stage {stage.name}: {e}")
        
        # Update metrics
        self.integration_metrics['records_processed'] = len(current_data)
        
        # Calculate quality metrics
        seifa_completeness = []
        for record in current_data:
            completeness_scores = record.get('demographic_completeness_scores', {})
            if 'seifa' in completeness_scores:
                seifa_completeness.append(completeness_scores['seifa'])
        
        if seifa_completeness:
            avg_seifa_completeness = sum(seifa_completeness) / len(seifa_completeness)
            self.integration_metrics['seifa_completeness'].append(avg_seifa_completeness)
            self.logger.info(f"Average SEIFA completeness: {avg_seifa_completeness:.2f}")
        
        self.logger.info(f"Demographic integration pipeline completed: {len(current_data)} records integrated")
        
        return current_data
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get integration pipeline metrics."""
        return self.integration_metrics.copy()
    
    def validate_pipeline_config(self) -> List[str]:
        """Validate pipeline configuration."""
        errors = []
        
        # Check required configuration sections
        required_sections = ['validation', 'standardisation', 'seifa_integration', 'quality_assessment']
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Missing required configuration section: {section}")
        
        return errors