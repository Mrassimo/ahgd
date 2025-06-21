"""
Schema version management and validation orchestration for AHGD.

This module provides utilities for managing schema versions, performing
migrations, and orchestrating validation across different data types.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, Callable
from enum import Enum

from pydantic import BaseModel, ValidationError
import pandas as pd

# Import all schema modules
from schemas.base_schema import (
    VersionedSchema,
    SchemaVersion,
    MigrationRecord,
    DataQualityLevel,
    validate_schema_compatibility,
    get_schema_version,
    upgrade_schema,
    SchemaVersionManager,
    ValidationMetrics,
    CompatibilityChecker
)
from schemas.sa2_schema import (
    SA2Coordinates,
    SA2GeometryValidation,
    SA2BoundaryRelationship,
    migrate_sa2_v1_to_v2
)
from schemas.health_schema import (
    HealthIndicator,
    MortalityData,
    DiseasePrevalence,
    HealthcareUtilisation,
    RiskFactorData,
    MentalHealthIndicator,
    HealthDataAggregate
)
from schemas.seifa_schema import (
    SEIFAScore,
    SEIFAComponent,
    SEIFAComparison,
    SEIFAAggregate,
    migrate_seifa_v1_to_v2
)
from schemas.census_schema import (
    CensusDemographics,
    CensusEducation,
    CensusEmployment,
    CensusHousing,
    migrate_census_v1_to_v2
)
from schemas.mortality_schema import (
    MortalityRecord,
    MortalityStatistics,
    MortalityTrend,
    migrate_mortality_v1_to_v2
)
from schemas.environmental_schema import (
    WeatherObservation,
    ClimateStatistics,
    EnvironmentalHealthIndex,
    migrate_environmental_v1_to_v2
)


logger = logging.getLogger(__name__)


class SchemaType(str, Enum):
    """Enumeration of available schema types."""
    # Geographic schemas
    SA2_COORDINATES = "SA2Coordinates"
    SA2_GEOMETRY = "SA2GeometryValidation"
    SA2_RELATIONSHIP = "SA2BoundaryRelationship"
    
    # Health schemas
    HEALTH_INDICATOR = "HealthIndicator"
    MORTALITY_DATA = "MortalityData"
    DISEASE_PREVALENCE = "DiseasePrevalence"
    HEALTHCARE_UTILISATION = "HealthcareUtilisation"
    RISK_FACTOR = "RiskFactorData"
    MENTAL_HEALTH = "MentalHealthIndicator"
    HEALTH_AGGREGATE = "HealthDataAggregate"
    
    # SEIFA schemas
    SEIFA_SCORE = "SEIFAScore"
    SEIFA_COMPONENT = "SEIFAComponent"
    SEIFA_COMPARISON = "SEIFAComparison"
    SEIFA_AGGREGATE = "SEIFAAggregate"
    
    # Census schemas
    CENSUS_DEMOGRAPHICS = "CensusDemographics"
    CENSUS_EDUCATION = "CensusEducation"
    CENSUS_EMPLOYMENT = "CensusEmployment"
    CENSUS_HOUSING = "CensusHousing"
    
    # Mortality schemas
    MORTALITY_RECORD = "MortalityRecord"
    MORTALITY_STATISTICS = "MortalityStatistics"
    MORTALITY_TREND = "MortalityTrend"
    
    # Environmental schemas
    WEATHER_OBSERVATION = "WeatherObservation"
    CLIMATE_STATISTICS = "ClimateStatistics"
    ENVIRONMENTAL_HEALTH = "EnvironmentalHealthIndex"


class SchemaRegistry:
    """Registry for all available schemas and their versions."""
    
    def __init__(self):
        """Initialise schema registry."""
        self._schemas: Dict[SchemaType, Type[VersionedSchema]] = {
            # Geographic schemas
            SchemaType.SA2_COORDINATES: SA2Coordinates,
            SchemaType.SA2_GEOMETRY: SA2GeometryValidation,
            SchemaType.SA2_RELATIONSHIP: SA2BoundaryRelationship,
            
            # Health schemas
            SchemaType.HEALTH_INDICATOR: HealthIndicator,
            SchemaType.MORTALITY_DATA: MortalityData,
            SchemaType.DISEASE_PREVALENCE: DiseasePrevalence,
            SchemaType.HEALTHCARE_UTILISATION: HealthcareUtilisation,
            SchemaType.RISK_FACTOR: RiskFactorData,
            SchemaType.MENTAL_HEALTH: MentalHealthIndicator,
            SchemaType.HEALTH_AGGREGATE: HealthDataAggregate,
            
            # SEIFA schemas
            SchemaType.SEIFA_SCORE: SEIFAScore,
            SchemaType.SEIFA_COMPONENT: SEIFAComponent,
            SchemaType.SEIFA_COMPARISON: SEIFAComparison,
            SchemaType.SEIFA_AGGREGATE: SEIFAAggregate,
            
            # Census schemas
            SchemaType.CENSUS_DEMOGRAPHICS: CensusDemographics,
            SchemaType.CENSUS_EDUCATION: CensusEducation,
            SchemaType.CENSUS_EMPLOYMENT: CensusEmployment,
            SchemaType.CENSUS_HOUSING: CensusHousing,
            
            # Mortality schemas
            SchemaType.MORTALITY_RECORD: MortalityRecord,
            SchemaType.MORTALITY_STATISTICS: MortalityStatistics,
            SchemaType.MORTALITY_TREND: MortalityTrend,
            
            # Environmental schemas
            SchemaType.WEATHER_OBSERVATION: WeatherObservation,
            SchemaType.CLIMATE_STATISTICS: ClimateStatistics,
            SchemaType.ENVIRONMENTAL_HEALTH: EnvironmentalHealthIndex
        }
        
        # Migration functions registry
        self._migrations: Dict[str, Callable] = {
            "SA2Coordinates_v1.0.0_to_v2.0.0": migrate_sa2_v1_to_v2,
            "SEIFAScore_v1.0.0_to_v2.0.0": migrate_seifa_v1_to_v2,
            "CensusDemographics_v1.0.0_to_v2.0.0": migrate_census_v1_to_v2,
            "CensusEducation_v1.0.0_to_v2.0.0": migrate_census_v1_to_v2,
            "CensusEmployment_v1.0.0_to_v2.0.0": migrate_census_v1_to_v2,
            "CensusHousing_v1.0.0_to_v2.0.0": migrate_census_v1_to_v2,
            "MortalityRecord_v1.0.0_to_v2.0.0": migrate_mortality_v1_to_v2,
            "MortalityStatistics_v1.0.0_to_v2.0.0": migrate_mortality_v1_to_v2,
            "MortalityTrend_v1.0.0_to_v2.0.0": migrate_mortality_v1_to_v2,
            "WeatherObservation_v1.0.0_to_v2.0.0": migrate_environmental_v1_to_v2,
            "ClimateStatistics_v1.0.0_to_v2.0.0": migrate_environmental_v1_to_v2,
            "EnvironmentalHealthIndex_v1.0.0_to_v2.0.0": migrate_environmental_v1_to_v2
        }
        
        # Schema validation metrics
        self._validation_metrics = ValidationMetrics()
        
        # Schema version manager
        self._version_manager = SchemaVersionManager()
        
        # Register all current schema versions
        for schema_type, schema_class in self._schemas.items():
            self._version_manager.register_schema_version(
                schema_type.value, SchemaVersion.V2_0_0
            )
        
    def get_schema(self, schema_type: Union[SchemaType, str]) -> Type[VersionedSchema]:
        """
        Get schema class by type.
        
        Args:
            schema_type: Schema type enum or string
            
        Returns:
            Schema class
            
        Raises:
            ValueError: If schema type not found
        """
        if isinstance(schema_type, str):
            try:
                schema_type = SchemaType(schema_type)
            except ValueError:
                raise ValueError(f"Unknown schema type: {schema_type}")
                
        return self._schemas[schema_type]
    
    def list_schemas(self) -> List[str]:
        """List all available schema types."""
        return [schema.value for schema in SchemaType]
    
    def get_migration(self, schema_type: str, from_version: str, to_version: str) -> Optional[Callable]:
        """
        Get migration function for schema version upgrade.
        
        Args:
            schema_type: Type of schema
            from_version: Source version
            to_version: Target version
            
        Returns:
            Migration function if available, None otherwise
        """
        migration_key = f"{schema_type}_{from_version}_to_{to_version}"
        return self._migrations.get(migration_key)


class SchemaValidator:
    """Orchestrates validation across different schema types."""
    
    def __init__(self, registry: Optional[SchemaRegistry] = None):
        """
        Initialise schema validator.
        
        Args:
            registry: Schema registry instance (creates new if None)
        """
        self.registry = registry or SchemaRegistry()
        self.validation_stats = {
            'total_validated': 0,
            'validation_errors': 0,
            'integrity_errors': 0
        }
        self.validation_metrics = ValidationMetrics()
        self.compatibility_checker = CompatibilityChecker()
        
    def validate_single(self, data: Dict[str, Any], schema_type: Union[SchemaType, str]) -> Dict[str, Any]:
        """
        Validate a single data record against a schema.
        
        Args:
            data: Data dictionary to validate
            schema_type: Type of schema to validate against
            
        Returns:
            Dictionary with validation results
        """
        start_time = datetime.utcnow()
        schema_class = self.registry.get_schema(schema_type)
        result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'data_quality': DataQualityLevel.UNKNOWN,
            'schema_type': str(schema_type),
            'timestamp': start_time.isoformat(),
            'validation_duration_ms': 0
        }
        
        error_type = None
        
        try:
            # Create instance and validate
            instance = schema_class(**data)
            result['valid'] = True
            
            # Check data integrity
            integrity_errors = instance.validate_data_integrity()
            if integrity_errors:
                result['warnings'].extend(integrity_errors)
                self.validation_stats['integrity_errors'] += len(integrity_errors)
                
            # Assess data quality
            result['data_quality'] = self._assess_data_quality(instance, integrity_errors)
            
            # Add schema version information
            result['schema_version'] = instance.schema_version.value if hasattr(instance, 'schema_version') else None
            
        except ValidationError as e:
            error_type = 'ValidationError'
            result['errors'] = [
                {
                    'field': '.'.join(str(loc) for loc in err['loc']),
                    'message': err['msg'],
                    'type': err['type']
                }
                for err in e.errors()
            ]
            self.validation_stats['validation_errors'] += 1
            
        except Exception as e:
            error_type = type(e).__name__
            result['errors'] = [{'message': str(e), 'type': 'UnexpectedError'}]
            self.validation_stats['validation_errors'] += 1
            
        # Calculate validation duration
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds() * 1000
        result['validation_duration_ms'] = duration
        
        # Record metrics
        self.validation_metrics.record_validation(result['valid'], duration, error_type)
        self.validation_stats['total_validated'] += 1
        
        return result
    
    def validate_batch(self, data_list: List[Dict[str, Any]], schema_type: Union[SchemaType, str],
                      continue_on_error: bool = True) -> Dict[str, Any]:
        """
        Validate a batch of records.
        
        Args:
            data_list: List of data dictionaries
            schema_type: Schema type to validate against
            continue_on_error: Whether to continue validating after errors
            
        Returns:
            Batch validation results
        """
        results = {
            'total': len(data_list),
            'valid': 0,
            'invalid': 0,
            'warnings': 0,
            'records': [],
            'summary': {}
        }
        
        for idx, data in enumerate(data_list):
            try:
                record_result = self.validate_single(data, schema_type)
                record_result['index'] = idx
                
                if record_result['valid']:
                    results['valid'] += 1
                    if record_result['warnings']:
                        results['warnings'] += 1
                else:
                    results['invalid'] += 1
                    
                results['records'].append(record_result)
                
            except Exception as e:
                logger.error(f"Error validating record {idx}: {e}")
                if not continue_on_error:
                    raise
                    
        # Generate summary
        results['summary'] = {
            'success_rate': results['valid'] / results['total'] * 100 if results['total'] > 0 else 0,
            'warning_rate': results['warnings'] / results['total'] * 100 if results['total'] > 0 else 0,
            'common_errors': self._summarise_errors(results['records'])
        }
        
        return results
    
    def validate_dataframe(self, df: pd.DataFrame, schema_type: Union[SchemaType, str],
                         sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Validate a pandas DataFrame against a schema.
        
        Args:
            df: DataFrame to validate
            schema_type: Schema type to validate against
            sample_size: Number of rows to sample (None for all)
            
        Returns:
            Validation results
        """
        # Sample if requested
        if sample_size and len(df) > sample_size:
            sample_df = df.sample(n=sample_size)
            logger.info(f"Validating sample of {sample_size} rows from {len(df)} total")
        else:
            sample_df = df
            
        # Convert to list of dicts
        records = sample_df.to_dict('records')
        
        # Validate batch
        results = self.validate_batch(records, schema_type)
        results['dataframe_info'] = {
            'total_rows': len(df),
            'validated_rows': len(sample_df),
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict()
        }
        
        return results
    
    def _assess_data_quality(self, instance: VersionedSchema, integrity_errors: List[str]) -> DataQualityLevel:
        """
        Assess overall data quality based on validation results.
        
        Args:
            instance: Validated schema instance
            integrity_errors: List of integrity errors
            
        Returns:
            Data quality level
        """
        # Start with high quality
        quality = DataQualityLevel.HIGH
        
        # Check for missing optional fields
        missing_fields = 0
        for field_name, field_info in instance.__fields__.items():
            if getattr(instance, field_name) is None and not field_info.required:
                missing_fields += 1
                
        # Downgrade based on missing data
        missing_ratio = missing_fields / len(instance.__fields__)
        if missing_ratio > 0.5:
            quality = DataQualityLevel.LOW
        elif missing_ratio > 0.2:
            quality = DataQualityLevel.MEDIUM
            
        # Further downgrade based on integrity errors
        if len(integrity_errors) > 3:
            quality = DataQualityLevel.LOW
        elif len(integrity_errors) > 0 and quality == DataQualityLevel.HIGH:
            quality = DataQualityLevel.MEDIUM
            
        return quality
    
    def _summarise_errors(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        """Summarise common validation errors."""
        error_counts = {}
        
        for record in records:
            for error in record.get('errors', []):
                error_key = f"{error['field']}: {error['type']}"
                error_counts[error_key] = error_counts.get(error_key, 0) + 1
                
        # Sort by frequency
        return dict(sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10])


class SchemaMigrationManager:
    """Manages schema version migrations."""
    
    def __init__(self, registry: Optional[SchemaRegistry] = None):
        """
        Initialise migration manager.
        
        Args:
            registry: Schema registry instance
        """
        self.registry = registry or SchemaRegistry()
        self.migration_history: List[MigrationRecord] = []
        
    def migrate_record(self, data: Dict[str, Any], schema_type: str,
                      target_version: SchemaVersion) -> Dict[str, Any]:
        """
        Migrate a single record to target schema version.
        
        Args:
            data: Data to migrate
            schema_type: Type of schema
            target_version: Target schema version
            
        Returns:
            Migrated data
        """
        current_version = get_schema_version(data)
        if not current_version:
            raise ValueError("Cannot determine current schema version")
            
        if current_version == target_version:
            return data  # Already at target version
            
        # Get migration function
        migration_func = self.registry.get_migration(
            schema_type,
            current_version.value,
            target_version.value
        )
        
        if not migration_func:
            # Try to use default migration from schema class
            schema_class = self.registry.get_schema(schema_type)
            return upgrade_schema(data, schema_class, target_version)
            
        # Apply migration
        migrated_data = migration_func(data)
        
        # Validate migrated data
        schema_class = self.registry.get_schema(schema_type)
        try:
            schema_class(**migrated_data)
        except ValidationError as e:
            raise ValueError(f"Migration produced invalid data: {e}")
            
        return migrated_data
    
    def migrate_batch(self, data_list: List[Dict[str, Any]], schema_type: str,
                     target_version: SchemaVersion, track_history: bool = True) -> Dict[str, Any]:
        """
        Migrate a batch of records.
        
        Args:
            data_list: List of records to migrate
            schema_type: Schema type
            target_version: Target version
            track_history: Whether to track migration history
            
        Returns:
            Migration results
        """
        start_time = datetime.utcnow()
        results = {
            'total': len(data_list),
            'migrated': 0,
            'failed': 0,
            'errors': [],
            'records': []
        }
        
        for idx, data in enumerate(data_list):
            try:
                migrated = self.migrate_record(data, schema_type, target_version)
                results['records'].append(migrated)
                results['migrated'] += 1
                
            except Exception as e:
                logger.error(f"Failed to migrate record {idx}: {e}")
                results['failed'] += 1
                results['errors'].append({
                    'index': idx,
                    'error': str(e)
                })
                
        # Track migration history
        if track_history and results['migrated'] > 0:
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Get first record to determine versions
            first_record = data_list[0] if data_list else {}
            from_version = get_schema_version(first_record) or SchemaVersion.V1_0_0
            
            migration_record = MigrationRecord(
                from_version=from_version,
                to_version=target_version,
                record_count=results['migrated'],
                success=results['failed'] == 0,
                errors=[str(e) for e in results['errors'][:10]],  # Keep first 10 errors
                duration_seconds=duration
            )
            self.migration_history.append(migration_record)
            
        return results
    
    def export_migration_history(self, filepath: Path) -> None:
        """Export migration history to JSON file."""
        history_data = [
            record.dict() for record in self.migration_history
        ]
        
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2, default=str)
            
        logger.info(f"Exported {len(history_data)} migration records to {filepath}")


class SchemaDocumentationGenerator:
    """Generates documentation for schemas."""
    
    def __init__(self, registry: Optional[SchemaRegistry] = None):
        """Initialise documentation generator."""
        self.registry = registry or SchemaRegistry()
        
    def generate_schema_docs(self, schema_type: Union[SchemaType, str]) -> Dict[str, Any]:
        """
        Generate documentation for a schema.
        
        Args:
            schema_type: Schema type to document
            
        Returns:
            Schema documentation dictionary
        """
        schema_class = self.registry.get_schema(schema_type)
        
        docs = {
            'schema_name': schema_class.__name__,
            'description': schema_class.__doc__,
            'fields': {},
            'validators': [],
            'examples': [],
            'version_info': {
                'current_version': SchemaVersion.V2_0_0.value,
                'compatible_versions': [v.value for v in SchemaVersion]
            }
        }
        
        # Document fields
        for field_name, field_info in schema_class.__fields__.items():
            docs['fields'][field_name] = {
                'type': str(field_info.type_),
                'required': field_info.required,
                'description': field_info.field_info.description,
                'default': field_info.default if not field_info.required else None,
                'constraints': self._extract_constraints(field_info)
            }
            
        # Extract validators
        for validator_name in dir(schema_class):
            if validator_name.startswith('validate_'):
                validator_func = getattr(schema_class, validator_name)
                if hasattr(validator_func, '__doc__'):
                    docs['validators'].append({
                        'name': validator_name,
                        'description': validator_func.__doc__
                    })
                    
        # Add examples from schema config
        if hasattr(schema_class.Config, 'schema_extra'):
            example = schema_class.Config.schema_extra.get('example')
            if example:
                docs['examples'].append(example)
                
        return docs
    
    def _extract_constraints(self, field_info) -> Dict[str, Any]:
        """Extract field constraints from field info."""
        constraints = {}
        
        # Numeric constraints
        for constraint in ['gt', 'ge', 'lt', 'le']:
            if hasattr(field_info.field_info, constraint):
                value = getattr(field_info.field_info, constraint)
                if value is not None:
                    constraints[constraint] = value
                    
        # String constraints
        if hasattr(field_info.field_info, 'min_length'):
            constraints['min_length'] = field_info.field_info.min_length
        if hasattr(field_info.field_info, 'max_length'):
            constraints['max_length'] = field_info.field_info.max_length
        if hasattr(field_info.field_info, 'regex'):
            constraints['regex'] = field_info.field_info.regex
            
        return constraints
    
    def generate_all_docs(self, output_dir: Path) -> None:
        """Generate documentation for all schemas."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_docs = {}
        for schema_type in SchemaType:
            docs = self.generate_schema_docs(schema_type)
            all_docs[schema_type.value] = docs
            
            # Write individual schema docs
            schema_file = output_dir / f"{schema_type.value.lower()}_schema.json"
            with open(schema_file, 'w') as f:
                json.dump(docs, f, indent=2)
                
        # Write combined documentation
        combined_file = output_dir / "all_schemas.json"
        with open(combined_file, 'w') as f:
            json.dump(all_docs, f, indent=2)
            
        logger.info(f"Generated documentation for {len(all_docs)} schemas in {output_dir}")


# Convenience functions

def create_schema_manager() -> 'SchemaManager':
    """Create a configured schema manager instance."""
    return SchemaManager()


class SchemaManager:
    """Main interface for schema management operations."""
    
    def __init__(self):
        """Initialise schema manager with all components."""
        self.registry = SchemaRegistry()
        self.validator = SchemaValidator(self.registry)
        self.migration_manager = SchemaMigrationManager(self.registry)
        self.doc_generator = SchemaDocumentationGenerator(self.registry)
        self.compatibility_checker = CompatibilityChecker()
        self.version_manager = SchemaVersionManager()
        
    def validate(self, data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
                schema_type: Union[SchemaType, str]) -> Dict[str, Any]:
        """
        Validate data against a schema.
        
        Automatically detects input type and calls appropriate validator.
        """
        if isinstance(data, dict):
            return self.validator.validate_single(data, schema_type)
        elif isinstance(data, list):
            return self.validator.validate_batch(data, schema_type)
        elif isinstance(data, pd.DataFrame):
            return self.validator.validate_dataframe(data, schema_type)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
            
    def migrate(self, data: Union[Dict[str, Any], List[Dict[str, Any]]],
               schema_type: str, target_version: SchemaVersion) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Migrate data to target schema version.
        
        Returns migrated data in same format as input.
        """
        if isinstance(data, dict):
            return self.migration_manager.migrate_record(data, schema_type, target_version)
        elif isinstance(data, list):
            result = self.migration_manager.migrate_batch(data, schema_type, target_version)
            return result['records']
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
            
    def get_validation_stats(self) -> Dict[str, int]:
        """Get validation statistics."""
        return self.validator.validation_stats.copy()
    
    def get_migration_history(self) -> List[MigrationRecord]:
        """Get migration history."""
        return self.migration_manager.migration_history.copy()
    
    def generate_docs(self, output_dir: Optional[Path] = None) -> None:
        """Generate schema documentation."""
        if output_dir is None:
            output_dir = Path("docs/schemas")
        self.doc_generator.generate_all_docs(output_dir)
    
    def check_compatibility(self, schema_type: Union[SchemaType, str], 
                          old_version: SchemaVersion, new_version: SchemaVersion) -> Dict[str, Any]:
        """
        Check compatibility between schema versions.
        
        Args:
            schema_type: Schema type to check
            old_version: Source version
            new_version: Target version
            
        Returns:
            Compatibility report
        """
        schema_class = self.registry.get_schema(schema_type)
        
        # For now, we'll use basic compatibility checking
        # In a full implementation, you'd need version-specific schema classes
        forward_compatible = self.compatibility_checker.check_forward_compatibility(
            old_version, new_version
        )
        
        return {
            'schema_type': str(schema_type),
            'from_version': old_version.value,
            'to_version': new_version.value,
            'forward_compatible': forward_compatible,
            'migration_available': self.registry.get_migration(
                str(schema_type), old_version.value, new_version.value
            ) is not None,
            'migration_path': self.version_manager.get_migration_path(old_version, new_version)
        }
    
    def validate_with_version_check(self, data: Dict[str, Any], 
                                  schema_type: Union[SchemaType, str],
                                  target_version: Optional[SchemaVersion] = None) -> Dict[str, Any]:
        """
        Validate data with automatic version detection and migration.
        
        Args:
            data: Data to validate
            schema_type: Schema type
            target_version: Target version (latest if None)
            
        Returns:
            Validation results with migration information
        """
        # Detect current version
        current_version = get_schema_version(data)
        if target_version is None:
            target_version = self.version_manager.get_latest_version(str(schema_type))
        
        result = {
            'original_version': current_version.value if current_version else None,
            'target_version': target_version.value,
            'migration_performed': False,
            'migration_errors': []
        }
        
        # Migrate if necessary
        if current_version and current_version != target_version:
            try:
                migrated_data = self.migrate(data, str(schema_type), target_version)
                result['migration_performed'] = True
                data = migrated_data
            except Exception as e:
                result['migration_errors'].append(str(e))
                # Continue with original data
        
        # Validate
        validation_result = self.validate(data, schema_type)
        result.update(validation_result)
        
        return result
    
    def bulk_validate_with_reporting(self, data_sources: Dict[str, Any],
                                   output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Validate multiple data sources and generate comprehensive reports.
        
        Args:
            data_sources: Dictionary mapping source names to (data, schema_type) tuples
            output_dir: Directory for reports (default: current directory)
            
        Returns:
            Summary of all validation results
        """
        if output_dir is None:
            output_dir = Path("validation_reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'total_sources': len(data_sources),
            'successful_validations': 0,
            'failed_validations': 0,
            'sources': {},
            'overall_metrics': {},
            'report_timestamp': datetime.utcnow().isoformat()
        }
        
        for source_name, (data, schema_type) in data_sources.items():
            logger.info(f"Validating {source_name} against {schema_type}")
            
            try:
                result = self.validate(data, schema_type)
                summary['sources'][source_name] = result
                
                if isinstance(result, dict) and result.get('valid', False):
                    summary['successful_validations'] += 1
                elif isinstance(result, dict) and result.get('success_rate', 0) > 0.8:
                    summary['successful_validations'] += 1
                else:
                    summary['failed_validations'] += 1
                
                # Write individual report
                report_file = output_dir / f"{source_name}_validation_report.json"
                with open(report_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                    
            except Exception as e:
                logger.error(f"Error validating {source_name}: {e}")
                summary['sources'][source_name] = {'error': str(e)}
                summary['failed_validations'] += 1
        
        # Add overall metrics
        summary['overall_metrics'] = self.get_validation_metrics()
        
        # Write summary report
        summary_file = output_dir / "validation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Validation complete. Reports written to {output_dir}")
        return summary
    
    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get comprehensive validation metrics."""
        base_stats = self.get_validation_stats()
        detailed_metrics = self.validator.validation_metrics.get_metrics()
        
        return {
            'basic_stats': base_stats,
            'detailed_metrics': detailed_metrics,
            'schema_registry_stats': {
                'total_schemas': len(self.registry.list_schemas()),
                'available_migrations': len(self.registry._migrations)
            }
        }
    
    def reset_metrics(self) -> None:
        """Reset all validation metrics."""
        self.validator.validation_stats = {
            'total_validated': 0,
            'validation_errors': 0,
            'integrity_errors': 0
        }
        self.validator.validation_metrics.reset()
    
    def get_schema_info(self, schema_type: Union[SchemaType, str]) -> Dict[str, Any]:
        """
        Get comprehensive information about a schema.
        
        Args:
            schema_type: Schema type to inspect
            
        Returns:
            Schema information including fields, validators, examples
        """
        schema_class = self.registry.get_schema(schema_type)
        
        # Get basic documentation
        docs = self.doc_generator.generate_schema_docs(schema_type)
        
        # Add runtime information
        docs['runtime_info'] = {
            'latest_version': self.version_manager.get_latest_version(str(schema_type)).value,
            'available_migrations': [
                key for key in self.registry._migrations.keys() 
                if key.startswith(f"{schema_type}_")
            ],
            'inheritance_chain': [cls.__name__ for cls in schema_class.__mro__],
            'total_fields': len(schema_class.model_fields) if hasattr(schema_class, 'model_fields') else 0
        }
        
        return docs