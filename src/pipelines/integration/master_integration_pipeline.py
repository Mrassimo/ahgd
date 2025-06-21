"""
Master integration pipeline for AHGD project.

This module implements the final master record creation pipeline that combines
all data sources into comprehensive MasterHealthRecord instances, ensuring
compliance with target schema requirements.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import asyncio

from ..base_pipeline import BasePipeline
from ..stage import Stage
from ...transformers.data_integrator import MasterDataIntegrator
from ...transformers.denormaliser import HealthDataDenormaliser, MetadataDenormaliser
from ...transformers.derived_indicators import HealthIndicatorDeriver
from ...utils.integration_rules import DataIntegrationRules, ConflictResolver, QualityBasedSelector
from ...utils.interfaces import DataBatch, PipelineError
from ...utils.logging import get_logger, track_lineage
from ...schemas.integrated_schema import MasterHealthRecord, DataIntegrationLevel
from ...schemas.quality_standards import validate_against_standards
from .health_integration_pipeline import HealthIntegrationPipeline
from .demographic_integration_pipeline import DemographicIntegrationPipeline
from .geographic_integration_pipeline import GeographicIntegrationPipeline


class DataSourceOrchestrationStage(Stage):
    """Orchestrates integration of multiple data source pipelines."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("data_source_orchestration", config)
        
        # Initialize sub-pipelines
        self.health_pipeline = HealthIntegrationPipeline(
            config.get('health_integration', {}),
            self.logger
        )
        self.demographic_pipeline = DemographicIntegrationPipeline(
            config.get('demographic_integration', {}),
            self.logger
        )
        self.geographic_pipeline = GeographicIntegrationPipeline(
            config.get('geographic_integration', {}),
            self.logger
        )
        
        # Execution configuration
        self.parallel_execution = config.get('parallel_execution', True)
        self.pipeline_timeout = config.get('pipeline_timeout_minutes', 30)
        
    def execute(self, data: DataBatch, **kwargs) -> DataBatch:
        """Execute all source-specific integration pipelines."""
        self.logger.info("Starting data source orchestration")
        
        # Prepare data for each pipeline
        prepared_data = self._prepare_pipeline_data(data)
        
        if self.parallel_execution:
            # Execute pipelines in parallel
            integrated_results = self._execute_pipelines_parallel(prepared_data)
        else:
            # Execute pipelines sequentially
            integrated_results = self._execute_pipelines_sequential(prepared_data)
        
        # Merge results from all pipelines
        merged_data = self._merge_pipeline_results(integrated_results)
        
        self.logger.info(f"Data source orchestration completed: {len(merged_data)} integrated records")
        
        return merged_data
    
    def _prepare_pipeline_data(self, data: DataBatch) -> Dict[str, DataBatch]:
        """Prepare data for each specialized pipeline."""
        prepared_data = {
            'health': [],
            'demographic': [],
            'geographic': []
        }
        
        for record in data:
            sa2_code = record.get('sa2_code')
            
            # Prepare health data
            health_record = {
                'sa2_code': sa2_code,
                'total_population': record.get('total_population'),
                'demographic_profile': record.get('demographic_profile'),
            }
            
            # Add health-specific data sources
            for source in ['health_indicators', 'aihw', 'medicare_pbs', 'nhmd']:
                if source in record:
                    health_record[source] = record[source]
            
            prepared_data['health'].append(health_record)
            
            # Prepare demographic data
            demographic_record = {
                'sa2_code': sa2_code,
                'total_population': record.get('total_population'),
                'male_population': record.get('male_population'),
                'female_population': record.get('female_population'),
                'demographic_profile': record.get('demographic_profile'),
                'seifa_scores': record.get('seifa_scores'),
                'seifa_deciles': record.get('seifa_deciles'),
            }
            
            # Add demographic-specific sources
            for source in ['census', 'seifa', 'abs']:
                if source in record:
                    demographic_record[source] = record[source]
            
            prepared_data['demographic'].append(demographic_record)
            
            # Prepare geographic data
            geographic_record = {
                'sa2_code': sa2_code,
                'sa2_name': record.get('sa2_name'),
                'geographic_hierarchy': record.get('geographic_hierarchy'),
                'boundary_data': record.get('boundary_data'),
                'total_population': record.get('total_population'),
            }
            
            # Add geographic-specific sources
            for source in ['geographic_boundaries', 'abs_boundaries']:
                if source in record:
                    geographic_record[source] = record[source]
            
            prepared_data['geographic'].append(geographic_record)
        
        return prepared_data
    
    def _execute_pipelines_parallel(self, prepared_data: Dict[str, DataBatch]) -> Dict[str, DataBatch]:
        """Execute integration pipelines in parallel."""
        async def run_pipeline_async(pipeline, data, pipeline_name):
            """Run a pipeline asynchronously."""
            try:
                self.logger.info(f"Starting {pipeline_name} pipeline")
                result = pipeline.execute(data)
                self.logger.info(f"Completed {pipeline_name} pipeline: {len(result)} records")
                return result
            except Exception as e:
                self.logger.error(f"{pipeline_name} pipeline failed: {e}")
                raise
        
        async def run_all_pipelines():
            """Run all pipelines concurrently."""
            tasks = [
                run_pipeline_async(self.health_pipeline, prepared_data['health'], 'health'),
                run_pipeline_async(self.demographic_pipeline, prepared_data['demographic'], 'demographic'),
                run_pipeline_async(self.geographic_pipeline, prepared_data['geographic'], 'geographic')
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check for exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    pipeline_names = ['health', 'demographic', 'geographic']
                    raise PipelineError(f"{pipeline_names[i]} pipeline failed: {result}")
            
            return {
                'health': results[0],
                'demographic': results[1],
                'geographic': results[2]
            }
        
        # Run the async pipelines
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(run_all_pipelines())
    
    def _execute_pipelines_sequential(self, prepared_data: Dict[str, DataBatch]) -> Dict[str, DataBatch]:
        """Execute integration pipelines sequentially."""
        results = {}
        
        # Execute health pipeline
        self.logger.info("Executing health integration pipeline")
        results['health'] = self.health_pipeline.execute(prepared_data['health'])
        
        # Execute demographic pipeline
        self.logger.info("Executing demographic integration pipeline")
        results['demographic'] = self.demographic_pipeline.execute(prepared_data['demographic'])
        
        # Execute geographic pipeline
        self.logger.info("Executing geographic integration pipeline")
        results['geographic'] = self.geographic_pipeline.execute(prepared_data['geographic'])
        
        return results
    
    def _merge_pipeline_results(self, results: Dict[str, DataBatch]) -> DataBatch:
        """Merge results from all integration pipelines."""
        merged_data = []
        
        # Create lookup dictionaries for efficient merging
        health_lookup = {record.get('sa2_code'): record for record in results.get('health', [])}
        demographic_lookup = {record.get('sa2_code'): record for record in results.get('demographic', [])}
        geographic_lookup = {record.get('sa2_code'): record for record in results.get('geographic', [])}
        
        # Get all SA2 codes
        all_sa2_codes = set()
        all_sa2_codes.update(health_lookup.keys())
        all_sa2_codes.update(demographic_lookup.keys())
        all_sa2_codes.update(geographic_lookup.keys())
        
        # Merge data for each SA2
        for sa2_code in all_sa2_codes:
            if not sa2_code:  # Skip invalid codes
                continue
            
            merged_record = {'sa2_code': sa2_code}
            
            # Merge health data
            health_data = health_lookup.get(sa2_code, {})
            merged_record.update(health_data)
            
            # Merge demographic data
            demographic_data = demographic_lookup.get(sa2_code, {})
            merged_record.update(demographic_data)
            
            # Merge geographic data
            geographic_data = geographic_lookup.get(sa2_code, {})
            merged_record.update(geographic_data)
            
            # Add integration metadata
            merged_record['pipeline_integration_sources'] = []
            if sa2_code in health_lookup:
                merged_record['pipeline_integration_sources'].append('health')
            if sa2_code in demographic_lookup:
                merged_record['pipeline_integration_sources'].append('demographic')
            if sa2_code in geographic_lookup:
                merged_record['pipeline_integration_sources'].append('geographic')
            
            merged_data.append(merged_record)
        
        return merged_data


class MasterRecordCreationStage(Stage):
    """Creates final MasterHealthRecord instances."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("master_record_creation", config)
        
        # Initialize master data integrator
        self.master_integrator = MasterDataIntegrator(
            config.get('master_integration', {}),
            self.logger
        )
        
        # Integration rules and conflict resolution
        self.integration_rules = DataIntegrationRules(config.get('integration_rules', {}))
        self.conflict_resolver = ConflictResolver(config.get('conflict_resolution', {}))
        self.quality_selector = QualityBasedSelector(config.get('quality_selection', {}))
        
        # Missing data handling
        self.missing_data_strategies = config.get('missing_data_strategies', {})
        self.interpolation_enabled = config.get('interpolation_enabled', True)
        
    def execute(self, data: DataBatch, **kwargs) -> DataBatch:
        """Create MasterHealthRecord instances."""
        master_records = []
        creation_errors = []
        
        for record in data:
            try:
                # Create master health record
                master_record = self._create_master_health_record(record)
                
                if master_record:
                    master_records.append(master_record)
                else:
                    creation_errors.append({
                        'sa2_code': record.get('sa2_code'),
                        'error': 'failed_to_create_master_record'
                    })
                    
            except Exception as e:
                creation_errors.append({
                    'sa2_code': record.get('sa2_code'),
                    'error': str(e)
                })
                self.logger.error(f"Failed to create master record for {record.get('sa2_code')}: {e}")
        
        if creation_errors:
            self.logger.warning(f"Master record creation had {len(creation_errors)} errors")
        
        self.logger.info(f"Created {len(master_records)} master health records")
        
        return master_records
    
    def _create_master_health_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a single MasterHealthRecord from integrated data."""
        sa2_code = record.get('sa2_code')
        
        if not sa2_code:
            return None
        
        try:
            # Extract and validate required components
            geographic_data = self._extract_geographic_components(record)
            demographic_data = self._extract_demographic_components(record)
            health_data = self._extract_health_components(record)
            socioeconomic_data = self._extract_socioeconomic_components(record)
            
            # Handle missing data
            filled_data = self._handle_missing_data(
                sa2_code, geographic_data, demographic_data, health_data, socioeconomic_data
            )
            
            # Calculate data quality metrics
            quality_metrics = self._calculate_data_quality_metrics(filled_data)
            
            # Create MasterHealthRecord
            master_record = {
                # Primary identification
                'sa2_code': sa2_code,
                'sa2_name': record.get('sa2_name', f"SA2 {sa2_code}"),
                
                # Geographic dimensions
                'geographic_hierarchy': filled_data['geographic']['hierarchy'],
                'boundary_data': filled_data['geographic']['boundary'],
                'urbanisation': filled_data['geographic'].get('urbanisation'),
                'remoteness_category': filled_data['geographic'].get('remoteness_category'),
                
                # Demographic profile
                'demographic_profile': filled_data['demographic']['profile'],
                'total_population': filled_data['demographic']['total_population'],
                'population_density_per_sq_km': filled_data['demographic']['population_density'],
                'median_age': filled_data['demographic'].get('median_age'),
                
                # Socioeconomic indicators
                'seifa_scores': filled_data['socioeconomic']['seifa_scores'],
                'seifa_deciles': filled_data['socioeconomic']['seifa_deciles'],
                'disadvantage_category': filled_data['socioeconomic'].get('disadvantage_category'),
                'median_household_income': filled_data['health'].get('median_household_income'),
                'unemployment_rate': filled_data['health'].get('unemployment_rate'),
                
                # Health indicators summary
                'health_outcomes_summary': filled_data['health'].get('outcomes_summary', {}),
                'life_expectancy': filled_data['health'].get('life_expectancy'),
                'self_assessed_health': filled_data['health'].get('self_assessed_health'),
                
                # Mortality indicators
                'mortality_indicators': filled_data['health'].get('mortality_indicators', {}),
                'avoidable_mortality_rate': filled_data['health'].get('avoidable_mortality_rate'),
                'infant_mortality_rate': filled_data['health'].get('infant_mortality_rate'),
                
                # Disease prevalence
                'chronic_disease_prevalence': filled_data['health'].get('chronic_disease_prevalence', {}),
                
                # Mental health
                'mental_health_indicators': filled_data['health'].get('mental_health_indicators', {}),
                'psychological_distress_high': filled_data['health'].get('psychological_distress_high'),
                
                # Healthcare access and utilisation
                'healthcare_access': filled_data['health'].get('healthcare_access', {}),
                'gp_services_per_1000': filled_data['health'].get('gp_services_per_1000'),
                'specialist_services_per_1000': filled_data['health'].get('specialist_services_per_1000'),
                'bulk_billing_rate': filled_data['health'].get('bulk_billing_rate'),
                'emergency_dept_presentations_per_1000': filled_data['health'].get('emergency_dept_presentations_per_1000'),
                
                # Pharmaceutical utilisation
                'pharmaceutical_utilisation': filled_data['health'].get('pharmaceutical_utilisation', {}),
                
                # Risk factors
                'risk_factors': filled_data['health'].get('risk_factors', {}),
                'smoking_prevalence': filled_data['health'].get('smoking_prevalence'),
                'obesity_prevalence': filled_data['health'].get('obesity_prevalence'),
                'physical_inactivity_prevalence': filled_data['health'].get('physical_inactivity_prevalence'),
                'harmful_alcohol_use_prevalence': filled_data['health'].get('harmful_alcohol_use_prevalence'),
                
                # Environmental factors
                'environmental_indicators': filled_data['health'].get('environmental_indicators', {}),
                'air_quality_index': filled_data['health'].get('air_quality_index'),
                'green_space_access': filled_data['health'].get('green_space_access'),
                
                # Data integration metadata
                'integration_level': quality_metrics['integration_level'],
                'data_completeness_score': quality_metrics['completeness_score'],
                'integration_timestamp': datetime.utcnow(),
                'source_datasets': quality_metrics['source_datasets'],
                'missing_indicators': quality_metrics['missing_indicators'],
                
                # Derived indicators (will be calculated in next stage)
                'composite_health_index': None,
                'health_inequality_index': None,
                
                # Schema version
                'schema_version': "2.0.0"
            }
            
            return master_record
            
        except Exception as e:
            self.logger.error(f"Error creating master record for {sa2_code}: {e}")
            return None
    
    def _extract_geographic_components(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract geographic components from integrated record."""
        return {
            'hierarchy': record.get('geographic_hierarchy', {}),
            'boundary': record.get('boundary_data', {}),
            'urbanisation': record.get('urbanisation'),
            'remoteness_category': record.get('remoteness_category')
        }
    
    def _extract_demographic_components(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract demographic components from integrated record."""
        return {
            'profile': record.get('demographic_profile', {}),
            'total_population': record.get('total_population', 0),
            'population_density': record.get('population_density_per_sq_km', 0.0),
            'median_age': record.get('median_age')
        }
    
    def _extract_health_components(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract health components from integrated record."""
        return {
            'outcomes_summary': record.get('health_outcomes_summary'),
            'life_expectancy': record.get('life_expectancy'),
            'self_assessed_health': record.get('self_assessed_health'),
            'mortality_indicators': record.get('mortality_indicators'),
            'chronic_disease_prevalence': record.get('chronic_disease_prevalence'),
            'mental_health_indicators': record.get('mental_health_indicators'),
            'healthcare_access': record.get('healthcare_access'),
            'pharmaceutical_utilisation': record.get('pharmaceutical_utilisation'),
            'risk_factors': record.get('risk_factors'),
            'environmental_indicators': record.get('environmental_indicators'),
            # Individual indicators
            'avoidable_mortality_rate': record.get('avoidable_mortality_rate'),
            'infant_mortality_rate': record.get('infant_mortality_rate'),
            'psychological_distress_high': record.get('psychological_distress_high'),
            'gp_services_per_1000': record.get('gp_services_per_1000'),
            'specialist_services_per_1000': record.get('specialist_services_per_1000'),
            'bulk_billing_rate': record.get('bulk_billing_rate'),
            'emergency_dept_presentations_per_1000': record.get('emergency_dept_presentations_per_1000'),
            'smoking_prevalence': record.get('smoking_prevalence'),
            'obesity_prevalence': record.get('obesity_prevalence'),
            'physical_inactivity_prevalence': record.get('physical_inactivity_prevalence'),
            'harmful_alcohol_use_prevalence': record.get('harmful_alcohol_use_prevalence'),
            'air_quality_index': record.get('air_quality_index'),
            'green_space_access': record.get('green_space_access'),
            'median_household_income': record.get('median_household_income'),
            'unemployment_rate': record.get('unemployment_rate')
        }
    
    def _extract_socioeconomic_components(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract socioeconomic components from integrated record."""
        return {
            'seifa_scores': record.get('seifa_scores', {}),
            'seifa_deciles': record.get('seifa_deciles', {}),
            'disadvantage_category': record.get('disadvantage_category')
        }
    
    def _handle_missing_data(
        self, 
        sa2_code: str, 
        geographic_data: Dict[str, Any],
        demographic_data: Dict[str, Any],
        health_data: Dict[str, Any],
        socioeconomic_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Handle missing data using configured strategies."""
        filled_data = {
            'geographic': geographic_data.copy(),
            'demographic': demographic_data.copy(),
            'health': health_data.copy(),
            'socioeconomic': socioeconomic_data.copy()
        }
        
        # Apply missing data strategies
        for category, data in filled_data.items():
            strategy = self.missing_data_strategies.get(category, 'flag')
            
            if strategy == 'interpolate' and self.interpolation_enabled:
                filled_data[category] = self._interpolate_missing_values(sa2_code, data, category)
            elif strategy == 'default':
                filled_data[category] = self._apply_default_values(data, category)
            elif strategy == 'flag':
                filled_data[category] = self._flag_missing_values(data)
        
        return filled_data
    
    def _interpolate_missing_values(self, sa2_code: str, data: Dict[str, Any], category: str) -> Dict[str, Any]:
        """Interpolate missing values using spatial or demographic approaches."""
        # This is a simplified implementation
        # In practice, would use sophisticated spatial interpolation
        
        interpolated_data = data.copy()
        
        # For numeric fields, use category-specific defaults
        defaults = {
            'health': {
                'gp_services_per_1000': 1.5,
                'bulk_billing_rate': 85.0,
                'smoking_prevalence': 15.0,
                'obesity_prevalence': 25.0
            },
            'demographic': {
                'median_age': 38.0,
                'population_density': 100.0
            }
        }
        
        category_defaults = defaults.get(category, {})
        
        for field, default_value in category_defaults.items():
            if field in interpolated_data and interpolated_data[field] is None:
                interpolated_data[field] = default_value
                interpolated_data[f'{field}_interpolated'] = True
        
        return interpolated_data
    
    def _apply_default_values(self, data: Dict[str, Any], category: str) -> Dict[str, Any]:
        """Apply default values for missing data."""
        data_with_defaults = data.copy()
        
        # Apply conservative defaults
        if category == 'health':
            health_defaults = {
                'gp_services_per_1000': 0.0,
                'bulk_billing_rate': 0.0,
                'smoking_prevalence': 0.0,
                'obesity_prevalence': 0.0
            }
            
            for field, default_value in health_defaults.items():
                if field in data_with_defaults and data_with_defaults[field] is None:
                    data_with_defaults[field] = default_value
        
        return data_with_defaults
    
    def _flag_missing_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Flag missing values without imputation."""
        # Data remains unchanged, missing values are preserved as None
        return data
    
    def _calculate_data_quality_metrics(self, filled_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate data quality metrics for the integrated record."""
        # Count available indicators
        total_indicators = 0
        available_indicators = 0
        source_datasets = set()
        missing_indicators = []
        
        # Key indicators to check
        key_indicators = [
            ('health', 'life_expectancy'),
            ('health', 'gp_services_per_1000'),
            ('health', 'bulk_billing_rate'),
            ('demographic', 'total_population'),
            ('demographic', 'median_age'),
            ('socioeconomic', 'seifa_scores'),
            ('geographic', 'boundary')
        ]
        
        for category, indicator in key_indicators:
            total_indicators += 1
            category_data = filled_data.get(category, {})
            
            if indicator in category_data and category_data[indicator] is not None:
                available_indicators += 1
                source_datasets.add(category)
            else:
                missing_indicators.append(f"{category}_{indicator}")
        
        completeness_score = (available_indicators / total_indicators * 100) if total_indicators > 0 else 0
        
        # Determine integration level
        if len(source_datasets) >= 4:
            integration_level = DataIntegrationLevel.COMPREHENSIVE
        elif len(source_datasets) >= 3:
            integration_level = DataIntegrationLevel.STANDARD
        elif len(source_datasets) >= 2:
            integration_level = DataIntegrationLevel.MINIMAL
        else:
            integration_level = DataIntegrationLevel.MINIMAL
        
        return {
            'completeness_score': completeness_score,
            'integration_level': integration_level,
            'source_datasets': list(source_datasets),
            'missing_indicators': missing_indicators
        }


class DerivedIndicatorCalculationStage(Stage):
    """Calculates derived health indicators and composite indices."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("derived_indicator_calculation", config)
        
        self.indicator_deriver = HealthIndicatorDeriver(
            config.get('derived_indicators', {}),
            self.logger
        )
        
    def execute(self, data: DataBatch, **kwargs) -> DataBatch:
        """Calculate derived indicators for master records."""
        enhanced_records = []
        
        for record in data:
            try:
                # Calculate derived indicators
                enhanced_record = self.indicator_deriver.calculate_derived_indicators(record)
                enhanced_records.append(enhanced_record)
                
            except Exception as e:
                self.logger.error(f"Failed to calculate derived indicators for {record.get('sa2_code')}: {e}")
                # Keep original record if calculation fails
                enhanced_records.append(record)
        
        self.logger.info(f"Calculated derived indicators for {len(enhanced_records)} records")
        
        return enhanced_records


class TargetSchemaValidationStage(Stage):
    """Validates master records against target schema requirements."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("target_schema_validation", config)
        self.validation_standards = config.get('validation_standards', [])
        self.strict_validation = config.get('strict_validation', True)
        
    def execute(self, data: DataBatch, **kwargs) -> DataBatch:
        """Validate records against target schema."""
        validated_records = []
        validation_errors = []
        
        for record in data:
            try:
                # Validate against MasterHealthRecord schema
                validation_result = self._validate_master_health_record(record)
                
                if validation_result['valid'] or not self.strict_validation:
                    # Add validation metadata
                    record['schema_validation_passed'] = validation_result['valid']
                    record['schema_validation_errors'] = validation_result['errors']
                    record['schema_validation_timestamp'] = datetime.utcnow()
                    
                    validated_records.append(record)
                else:
                    validation_errors.append({
                        'sa2_code': record.get('sa2_code'),
                        'errors': validation_result['errors']
                    })
                    
            except Exception as e:
                self.logger.error(f"Validation failed for {record.get('sa2_code')}: {e}")
                if not self.strict_validation:
                    validated_records.append(record)
        
        if validation_errors:
            self.logger.warning(f"Schema validation found {len(validation_errors)} invalid records")
        
        self.logger.info(f"Schema validation completed: {len(validated_records)} valid records")
        
        return validated_records
    
    def _validate_master_health_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a record against MasterHealthRecord schema."""
        errors = []
        
        # Required fields validation
        required_fields = [
            'sa2_code', 'sa2_name', 'geographic_hierarchy', 'boundary_data',
            'demographic_profile', 'total_population', 'seifa_scores', 'seifa_deciles'
        ]
        
        for field in required_fields:
            if field not in record or record[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Data type validation
        if 'total_population' in record and record['total_population'] is not None:
            if not isinstance(record['total_population'], int) or record['total_population'] < 0:
                errors.append("total_population must be a non-negative integer")
        
        # Range validation for percentages
        percentage_fields = [
            'smoking_prevalence', 'obesity_prevalence', 'bulk_billing_rate',
            'physical_inactivity_prevalence', 'harmful_alcohol_use_prevalence'
        ]
        
        for field in percentage_fields:
            if field in record and record[field] is not None:
                value = record[field]
                if not isinstance(value, (int, float)) or not (0 <= value <= 100):
                    errors.append(f"{field} must be between 0 and 100")
        
        # SEIFA validation
        seifa_scores = record.get('seifa_scores', {})
        seifa_deciles = record.get('seifa_deciles', {})
        
        for index_type in seifa_scores:
            if index_type not in seifa_deciles:
                errors.append(f"Missing SEIFA decile for index {index_type}")
        
        # Schema version validation
        if 'schema_version' not in record:
            errors.append("Missing schema_version")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }


class MasterIntegrationPipeline(BasePipeline):
    """
    Master integration pipeline that creates final MasterHealthRecord instances.
    
    This is the main pipeline that orchestrates all integration processes and
    produces the final target schema-compliant health records.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialise the master integration pipeline.
        
        Args:
            config: Pipeline configuration
            logger: Optional logger instance
        """
        super().__init__("master_integration_pipeline", config, logger)
        
        # Create pipeline stages
        self.stages = [
            DataSourceOrchestrationStage(config.get('data_source_orchestration', {})),
            MasterRecordCreationStage(config.get('master_record_creation', {})),
            DerivedIndicatorCalculationStage(config.get('derived_indicator_calculation', {})),
            TargetSchemaValidationStage(config.get('target_schema_validation', {}))
        ]
        
        # Integration tracking
        self.integration_metrics = {
            'total_input_records': 0,
            'total_output_records': 0,
            'validation_pass_rate': 0.0,
            'average_completeness_score': 0.0,
            'integration_levels': {},
            'processing_time_seconds': 0.0
        }
        
        # Performance optimisation
        self.enable_performance_monitoring = config.get('enable_performance_monitoring', True)
        self.batch_size = config.get('batch_size', 1000)
        
    def execute(self, data: DataBatch, **kwargs) -> DataBatch:
        """Execute the complete master integration pipeline."""
        start_time = datetime.utcnow()
        
        self.logger.info("Starting master integration pipeline")
        self.logger.info(f"Processing {len(data)} input records")
        
        self.integration_metrics['total_input_records'] = len(data)
        
        # Track lineage
        track_lineage(
            input_data="multi_source_integrated_data",
            output_data="master_health_records",
            transformation="master_integration_pipeline"
        )
        
        # Execute pipeline stages
        current_data = data
        
        for stage in self.stages:
            try:
                stage_start = datetime.utcnow()
                self.logger.info(f"Executing stage: {stage.name}")
                
                # Process in batches if data is large
                if len(current_data) > self.batch_size:
                    current_data = self._process_in_batches(stage, current_data)
                else:
                    current_data = stage.execute(current_data, **kwargs)
                
                stage_duration = (datetime.utcnow() - stage_start).total_seconds()
                self.logger.info(f"Stage {stage.name} completed: {len(current_data)} records in {stage_duration:.1f}s")
                
            except Exception as e:
                self.logger.error(f"Stage {stage.name} failed: {e}")
                raise PipelineError(f"Master integration pipeline failed at stage {stage.name}: {e}")
        
        # Calculate final metrics
        self._calculate_final_metrics(current_data, start_time)
        
        self.logger.info(f"Master integration pipeline completed: {len(current_data)} master health records created")
        
        return current_data
    
    def _process_in_batches(self, stage: Stage, data: DataBatch) -> DataBatch:
        """Process large datasets in batches."""
        self.logger.info(f"Processing {len(data)} records in batches of {self.batch_size}")
        
        all_results = []
        
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batch_results = stage.execute(batch)
            all_results.extend(batch_results)
            
            self.logger.debug(f"Processed batch {i//self.batch_size + 1}: {len(batch_results)} records")
        
        return all_results
    
    def _calculate_final_metrics(self, final_data: DataBatch, start_time: datetime) -> None:
        """Calculate final integration metrics."""
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        self.integration_metrics['total_output_records'] = len(final_data)
        self.integration_metrics['processing_time_seconds'] = processing_time
        
        # Calculate validation pass rate
        valid_records = sum(
            1 for record in final_data 
            if record.get('schema_validation_passed', False)
        )
        self.integration_metrics['validation_pass_rate'] = (
            valid_records / len(final_data) * 100 if final_data else 0
        )
        
        # Calculate average completeness
        completeness_scores = [
            record.get('data_completeness_score', 0) for record in final_data
            if record.get('data_completeness_score') is not None
        ]
        self.integration_metrics['average_completeness_score'] = (
            sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0
        )
        
        # Count integration levels
        integration_levels = {}
        for record in final_data:
            level = record.get('integration_level', 'unknown')
            if isinstance(level, DataIntegrationLevel):
                level = level.value
            integration_levels[level] = integration_levels.get(level, 0) + 1
        
        self.integration_metrics['integration_levels'] = integration_levels
        
        # Log metrics
        self.logger.info(f"Integration metrics:")
        self.logger.info(f"  - Input records: {self.integration_metrics['total_input_records']}")
        self.logger.info(f"  - Output records: {self.integration_metrics['total_output_records']}")
        self.logger.info(f"  - Validation pass rate: {self.integration_metrics['validation_pass_rate']:.1f}%")
        self.logger.info(f"  - Average completeness: {self.integration_metrics['average_completeness_score']:.1f}%")
        self.logger.info(f"  - Processing time: {processing_time:.1f} seconds")
        self.logger.info(f"  - Integration levels: {integration_levels}")
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get comprehensive integration metrics."""
        return self.integration_metrics.copy()
    
    def validate_pipeline_config(self) -> List[str]:
        """Validate pipeline configuration."""
        errors = []
        
        # Check required configuration sections
        required_sections = [
            'data_source_orchestration',
            'master_record_creation', 
            'derived_indicator_calculation',
            'target_schema_validation'
        ]
        
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Missing required configuration section: {section}")
        
        # Validate sub-pipeline configurations
        for stage in self.stages:
            if hasattr(stage, 'validate_pipeline_config'):
                stage_errors = stage.validate_pipeline_config()
                errors.extend([f"{stage.name}: {error}" for error in stage_errors])
        
        return errors
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and health."""
        return {
            'pipeline_name': self.name,
            'total_stages': len(self.stages),
            'last_execution_metrics': self.integration_metrics,
            'configuration_valid': len(self.validate_pipeline_config()) == 0,
            'performance_monitoring_enabled': self.enable_performance_monitoring
        }