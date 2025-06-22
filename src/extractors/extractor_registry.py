"""
Extractor registry and factory for AHGD data extractors.

This module provides centralized registry, factory, validator, and monitor
for all data source extractors working backwards from target schema requirements.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Type, Union
import logging

from .base import BaseExtractor
from .aihw_extractor import (
    AIHWMortalityExtractor,
    AIHWHospitalisationExtractor,
    AIHWHealthIndicatorExtractor,
    AIHWMedicareExtractor,
)
from .abs_extractor import (
    ABSGeographicExtractor,
    ABSCensusExtractor,
    ABSSEIFAExtractor,
    ABSPostcodeExtractor,
)
from .bom_extractor import (
    BOMClimateExtractor,
    BOMWeatherStationExtractor,
    BOMEnvironmentalExtractor,
)
from .medicare_pbs_extractor import (
    MedicareUtilisationExtractor,
    PBSPrescriptionExtractor,
    HealthcareServicesExtractor,
)
from ..utils.interfaces import (
    DataBatch,
    ExtractionError,
    ProcessingStatus,
    SourceMetadata,
)
from ..utils.logging import get_logger
from ..utils.config import get_config


logger = get_logger(__name__)


class ExtractorType(str, Enum):
    """Types of data extractors."""
    AIHW_MORTALITY = "aihw_mortality"
    AIHW_HOSPITALISATION = "aihw_hospitalisation"
    AIHW_HEALTH_INDICATORS = "aihw_health_indicators"
    AIHW_MEDICARE = "aihw_medicare"
    
    ABS_GEOGRAPHIC = "abs_geographic"
    ABS_CENSUS = "abs_census"
    ABS_SEIFA = "abs_seifa"
    ABS_POSTCODE = "abs_postcode"
    
    BOM_CLIMATE = "bom_climate"
    BOM_WEATHER_STATIONS = "bom_weather_stations"
    BOM_ENVIRONMENTAL = "bom_environmental"
    
    MEDICARE_UTILISATION = "medicare_utilisation"
    PBS_PRESCRIPTIONS = "pbs_prescriptions"
    HEALTHCARE_SERVICES = "healthcare_services"


class DataCategory(str, Enum):
    """Categories of data for organization."""
    HEALTH = "health"
    GEOGRAPHIC = "geographic"
    DEMOGRAPHIC = "demographic"
    ENVIRONMENTAL = "environmental"
    HEALTHCARE_UTILISATION = "healthcare_utilisation"
    SOCIOECONOMIC = "socioeconomic"


@dataclass
class ExtractorMetadata:
    """Metadata about a registered extractor."""
    extractor_type: ExtractorType
    extractor_class: Type[BaseExtractor]
    data_category: DataCategory
    description: str
    source_organization: str
    update_frequency: str
    geographic_coverage: str
    target_schemas: List[str]
    dependencies: List[ExtractorType] = field(default_factory=list)
    priority: int = 50  # 1-100, higher is more important
    enabled: bool = True
    config_path: Optional[Path] = None
    last_successful_run: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate metadata after initialization."""
        if self.priority < 1 or self.priority > 100:
            raise ValueError(f"Priority must be between 1-100, got {self.priority}")


@dataclass
class ExtractionJob:
    """Represents an extraction job."""
    job_id: str
    extractor_type: ExtractorType
    source: Union[str, Path, Dict[str, Any]]
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    records_extracted: int = 0
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate job duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class ExtractorRegistry:
    """
    Central registry for all data source extractors.
    
    Provides discovery, registration, and metadata management for extractors
    working backwards from target schema requirements.
    """
    
    def __init__(self):
        self._extractors: Dict[ExtractorType, ExtractorMetadata] = {}
        self._register_built_in_extractors()
    
    def _register_built_in_extractors(self) -> None:
        """Register all built-in extractors."""
        
        # AIHW extractors
        self.register(
            ExtractorType.AIHW_MORTALITY,
            AIHWMortalityExtractor,
            DataCategory.HEALTH,
            description="AIHW GRIM (General Record of Incidence of Mortality) data",
            source_organization="Australian Institute of Health and Welfare",
            update_frequency="annual",
            geographic_coverage="National (SA2 level)",
            target_schemas=["MortalityData", "SA2HealthProfile", "HealthIndicator"],
            priority=90,
        )
        
        self.register(
            ExtractorType.AIHW_HOSPITALISATION,
            AIHWHospitalisationExtractor,
            DataCategory.HEALTHCARE_UTILISATION,
            description="AIHW hospital separation and utilisation data",
            source_organization="Australian Institute of Health and Welfare",
            update_frequency="annual",
            geographic_coverage="National (SA2 level)",
            target_schemas=["HealthcareUtilisation", "SA2HealthProfile"],
            priority=85,
        )
        
        self.register(
            ExtractorType.AIHW_HEALTH_INDICATORS,
            AIHWHealthIndicatorExtractor,
            DataCategory.HEALTH,
            description="AIHW health performance indicators and outcomes",
            source_organization="Australian Institute of Health and Welfare",
            update_frequency="biannual",
            geographic_coverage="National (SA2 level)",
            target_schemas=["HealthIndicator", "SA2HealthProfile", "HealthIndicatorSummary"],
            priority=88,
        )
        
        self.register(
            ExtractorType.AIHW_MEDICARE,
            AIHWMedicareExtractor,
            DataCategory.HEALTHCARE_UTILISATION,
            description="AIHW Medicare utilisation and access data",
            source_organization="Australian Institute of Health and Welfare",
            update_frequency="quarterly",
            geographic_coverage="National (SA2 level)",
            target_schemas=["HealthcareUtilisation", "MasterHealthRecord"],
            priority=82,
        )
        
        # ABS extractors
        self.register(
            ExtractorType.ABS_GEOGRAPHIC,
            ABSGeographicExtractor,
            DataCategory.GEOGRAPHIC,
            description="ABS Statistical Area boundaries and geographic hierarchies",
            source_organization="Australian Bureau of Statistics",
            update_frequency="5-yearly",
            geographic_coverage="National (all levels)",
            target_schemas=["GeographicBoundary", "SA2BoundaryData", "GeographicHealthMapping"],
            priority=95,  # Critical for all other data
        )
        
        self.register(
            ExtractorType.ABS_CENSUS,
            ABSCensusExtractor,
            DataCategory.DEMOGRAPHIC,
            description="ABS Census 2021 population and demographic data",
            source_organization="Australian Bureau of Statistics",
            update_frequency="5-yearly",
            geographic_coverage="National (SA2 level)",
            target_schemas=["CensusData", "MasterHealthRecord", "SA2HealthProfile"],
            dependencies=[ExtractorType.ABS_GEOGRAPHIC],
            priority=92,
        )
        
        self.register(
            ExtractorType.ABS_SEIFA,
            ABSSEIFAExtractor,
            DataCategory.SOCIOECONOMIC,
            description="ABS SEIFA 2021 socioeconomic indices",
            source_organization="Australian Bureau of Statistics",
            update_frequency="5-yearly",
            geographic_coverage="National (SA2 level)",
            target_schemas=["SEIFAIndex", "MasterHealthRecord", "SA2HealthProfile"],
            dependencies=[ExtractorType.ABS_GEOGRAPHIC],
            priority=90,
        )
        
        self.register(
            ExtractorType.ABS_POSTCODE,
            ABSPostcodeExtractor,
            DataCategory.GEOGRAPHIC,
            description="ABS postcode to SA2 correspondence tables",
            source_organization="Australian Bureau of Statistics",
            update_frequency="5-yearly",
            geographic_coverage="National",
            target_schemas=["PostcodeCorrespondence"],
            dependencies=[ExtractorType.ABS_GEOGRAPHIC],
            priority=75,
        )
        
        # BOM extractors
        self.register(
            ExtractorType.BOM_CLIMATE,
            BOMClimateExtractor,
            DataCategory.ENVIRONMENTAL,
            description="Bureau of Meteorology climate and weather data",
            source_organization="Bureau of Meteorology",
            update_frequency="daily",
            geographic_coverage="National (station-based)",
            target_schemas=["ClimateData", "EnvironmentalIndicator", "MasterHealthRecord"],
            priority=78,
        )
        
        self.register(
            ExtractorType.BOM_WEATHER_STATIONS,
            BOMWeatherStationExtractor,
            DataCategory.ENVIRONMENTAL,
            description="Bureau of Meteorology weather station metadata and SA2 mappings",
            source_organization="Bureau of Meteorology",
            update_frequency="annual",
            geographic_coverage="National",
            target_schemas=["WeatherStationData", "GeographicHealthMapping"],
            dependencies=[ExtractorType.ABS_GEOGRAPHIC],
            priority=70,
        )
        
        self.register(
            ExtractorType.BOM_ENVIRONMENTAL,
            BOMEnvironmentalExtractor,
            DataCategory.ENVIRONMENTAL,
            description="Bureau of Meteorology air quality and environmental indicators",
            source_organization="Bureau of Meteorology",
            update_frequency="daily",
            geographic_coverage="Major cities",
            target_schemas=["AirQualityData", "EnvironmentalIndicator"],
            priority=72,
        )
        
        # Medicare and PBS extractors
        self.register(
            ExtractorType.MEDICARE_UTILISATION,
            MedicareUtilisationExtractor,
            DataCategory.HEALTHCARE_UTILISATION,
            description="Medicare Benefits Schedule utilisation data with privacy protection",
            source_organization="Department of Health",
            update_frequency="quarterly",
            geographic_coverage="National (SA2 level)",
            target_schemas=["HealthcareUtilisation", "MasterHealthRecord", "SA2HealthProfile"],
            dependencies=[ExtractorType.ABS_GEOGRAPHIC],
            priority=85,
        )
        
        self.register(
            ExtractorType.PBS_PRESCRIPTIONS,
            PBSPrescriptionExtractor,
            DataCategory.HEALTHCARE_UTILISATION,
            description="Pharmaceutical Benefits Scheme prescription data",
            source_organization="Department of Health",
            update_frequency="monthly",
            geographic_coverage="National (SA2 level)",
            target_schemas=["PharmaceuticalUtilisation", "MasterHealthRecord"],
            dependencies=[ExtractorType.ABS_GEOGRAPHIC],
            priority=80,
        )
        
        self.register(
            ExtractorType.HEALTHCARE_SERVICES,
            HealthcareServicesExtractor,
            DataCategory.HEALTHCARE_UTILISATION,
            description="Healthcare service locations, capacity, and access metrics",
            source_organization="Department of Health",
            update_frequency="annual",
            geographic_coverage="National (SA2 level)",
            target_schemas=["HealthcareAccess", "GeographicHealthMapping"],
            dependencies=[ExtractorType.ABS_GEOGRAPHIC],
            priority=76,
        )
    
    def register(
        self,
        extractor_type: ExtractorType,
        extractor_class: Type[BaseExtractor],
        data_category: DataCategory,
        description: str,
        source_organization: str,
        update_frequency: str,
        geographic_coverage: str,
        target_schemas: List[str],
        dependencies: List[ExtractorType] = None,
        priority: int = 50,
        enabled: bool = True,
        config_path: Optional[Path] = None,
    ) -> None:
        """Register an extractor with metadata."""
        metadata = ExtractorMetadata(
            extractor_type=extractor_type,
            extractor_class=extractor_class,
            data_category=data_category,
            description=description,
            source_organization=source_organization,
            update_frequency=update_frequency,
            geographic_coverage=geographic_coverage,
            target_schemas=target_schemas,
            dependencies=dependencies or [],
            priority=priority,
            enabled=enabled,
            config_path=config_path,
        )
        
        self._extractors[extractor_type] = metadata
        logger.info(f"Registered extractor: {extractor_type.value}")
    
    def get_extractor_metadata(self, extractor_type: ExtractorType) -> Optional[ExtractorMetadata]:
        """Get metadata for an extractor."""
        return self._extractors.get(extractor_type)
    
    def list_extractors(
        self,
        data_category: Optional[DataCategory] = None,
        enabled_only: bool = True,
    ) -> List[ExtractorMetadata]:
        """List all registered extractors."""
        extractors = list(self._extractors.values())
        
        if data_category:
            extractors = [e for e in extractors if e.data_category == data_category]
        
        if enabled_only:
            extractors = [e for e in extractors if e.enabled]
        
        # Sort by priority (highest first)
        extractors.sort(key=lambda x: x.priority, reverse=True)
        
        return extractors
    
    def get_extractors_by_target_schema(self, schema_name: str) -> List[ExtractorMetadata]:
        """Get extractors that produce data for a specific target schema."""
        return [
            metadata for metadata in self._extractors.values()
            if schema_name in metadata.target_schemas and metadata.enabled
        ]
    
    def get_extraction_order(
        self,
        include_extractors: Optional[List[ExtractorType]] = None
    ) -> List[ExtractorType]:
        """
        Get the optimal order for running extractors based on dependencies.
        
        Args:
            include_extractors: Optional list to limit extractors to include
            
        Returns:
            List of extractor types in dependency order
        """
        if include_extractors is None:
            include_extractors = list(self._extractors.keys())
        
        # Filter to enabled extractors
        available_extractors = [
            ext_type for ext_type in include_extractors
            if self._extractors[ext_type].enabled
        ]
        
        # Topological sort based on dependencies
        ordered = []
        remaining = set(available_extractors)
        
        while remaining:
            # Find extractors with no unsatisfied dependencies
            ready = []
            for ext_type in remaining:
                metadata = self._extractors[ext_type]
                unsatisfied_deps = [
                    dep for dep in metadata.dependencies
                    if dep in remaining  # Only check dependencies we haven't processed
                ]
                if not unsatisfied_deps:
                    ready.append(ext_type)
            
            if not ready:
                # Circular dependency or missing dependency
                logger.warning(f"Circular dependency detected in extractors: {remaining}")
                # Add remaining by priority
                ready = sorted(remaining, key=lambda x: self._extractors[x].priority, reverse=True)
            
            # Sort ready extractors by priority (highest first)
            ready.sort(key=lambda x: self._extractors[x].priority, reverse=True)
            
            # Add to ordered list and remove from remaining
            for ext_type in ready:
                ordered.append(ext_type)
                remaining.remove(ext_type)
        
        return ordered
    
    def validate_dependencies(self) -> List[str]:
        """Validate that all extractor dependencies are available."""
        errors = []
        
        for extractor_type, metadata in self._extractors.items():
            for dependency in metadata.dependencies:
                if dependency not in self._extractors:
                    errors.append(
                        f"Extractor {extractor_type.value} depends on unregistered "
                        f"extractor {dependency.value}"
                    )
                elif not self._extractors[dependency].enabled:
                    errors.append(
                        f"Extractor {extractor_type.value} depends on disabled "
                        f"extractor {dependency.value}"
                    )
        
        return errors
    
    def get_extractor(
        self,
        extractor_id: str,
        config_override: Optional[Dict[str, Any]] = None,
    ) -> Optional[BaseExtractor]:
        """
        Get an extractor instance by string ID.
        
        Args:
            extractor_id: String ID of the extractor (e.g., 'abs_geographic')
            config_override: Optional configuration overrides
            
        Returns:
            Configured extractor instance or None if not found
            
        Raises:
            ExtractionError: If extractor creation fails
        """
        try:
            # Convert string ID to ExtractorType enum
            extractor_type = None
            for ext_type in ExtractorType:
                if ext_type.value == extractor_id:
                    extractor_type = ext_type
                    break
            
            if not extractor_type:
                logger.error(f"Unknown extractor ID: {extractor_id}")
                return None
            
            # Use factory to create instance
            factory = ExtractorFactory(self)
            extractor = factory.create_extractor(extractor_type, config_override)
            
            logger.info(f"Successfully created extractor instance: {extractor_id}")
            return extractor
            
        except Exception as e:
            logger.error(f"Failed to create extractor {extractor_id}: {e}")
            raise ExtractionError(f"Failed to create extractor {extractor_id}: {e}")


class ExtractorFactory:
    """
    Factory for creating extractor instances.
    
    Creates extractors with appropriate configuration working backwards
    from target schema requirements.
    """
    
    def __init__(self, registry: ExtractorRegistry):
        self.registry = registry
        self._config_cache: Dict[str, Dict[str, Any]] = {}
    
    def create_extractor(
        self,
        extractor_type: ExtractorType,
        config_override: Optional[Dict[str, Any]] = None,
    ) -> BaseExtractor:
        """
        Create an extractor instance.
        
        Args:
            extractor_type: Type of extractor to create
            config_override: Optional configuration overrides
            
        Returns:
            Configured extractor instance
            
        Raises:
            ExtractionError: If extractor type not found or creation fails
        """
        metadata = self.registry.get_extractor_metadata(extractor_type)
        if not metadata:
            raise ExtractionError(f"Unknown extractor type: {extractor_type.value}")
        
        if not metadata.enabled:
            raise ExtractionError(f"Extractor {extractor_type.value} is disabled")
        
        # Load configuration
        config = self._load_extractor_config(extractor_type)
        
        # Apply overrides
        if config_override:
            config.update(config_override)
        
        # Create extractor instance
        try:
            extractor = metadata.extractor_class(config)
            logger.info(f"Created extractor: {extractor_type.value}")
            return extractor
        except Exception as e:
            raise ExtractionError(f"Failed to create extractor {extractor_type.value}: {e}")
    
    def _load_extractor_config(self, extractor_type: ExtractorType) -> Dict[str, Any]:
        """Load configuration for an extractor type."""
        config_key = extractor_type.value
        
        # Check cache
        if config_key in self._config_cache:
            return self._config_cache[config_key].copy()
        
        # Load from configuration system
        try:
            # Try extractor-specific config first
            config = get_config(f"extractors.{config_key}", {})
            
            # Fall back to general extractor config
            if not config:
                config = get_config("extractors.default", {})
            
            # Add global settings
            config.update({
                'max_retries': get_config('extractors.max_retries', 3),
                'retry_delay': get_config('extractors.retry_delay', 1.0),
                'batch_size': get_config('extractors.batch_size', 1000),
                'timeout_seconds': get_config('extractors.timeout_seconds', 60),
            })
            
            # Cache the config
            self._config_cache[config_key] = config.copy()
            
            return config
            
        except Exception as e:
            logger.warning(f"Failed to load config for {config_key}: {e}")
            return {}


class ExtractorValidator:
    """
    Validator for extractor outputs against target schema requirements.
    
    Validates that extractor outputs are compatible with target schemas
    for successful data integration.
    """
    
    def __init__(self, registry: ExtractorRegistry):
        self.registry = registry
    
    def validate_extractor_output(
        self,
        extractor_type: ExtractorType,
        sample_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Validate extractor output against target schema requirements.
        
        Args:
            extractor_type: Type of extractor being validated
            sample_records: Sample records from the extractor
            
        Returns:
            Validation report with compatibility status
        """
        metadata = self.registry.get_extractor_metadata(extractor_type)
        if not metadata:
            return {'valid': False, 'error': f'Unknown extractor type: {extractor_type.value}'}
        
        validation_report = {
            'extractor_type': extractor_type.value,
            'target_schemas': metadata.target_schemas,
            'sample_size': len(sample_records),
            'validation_timestamp': datetime.now().isoformat(),
            'schema_compatibility': {},
            'field_coverage': {},
            'data_quality': {},
            'overall_valid': True,
            'errors': [],
            'warnings': [],
        }
        
        if not sample_records:
            validation_report['overall_valid'] = False
            validation_report['errors'].append('No sample records provided')
            return validation_report
        
        # Check each target schema
        for schema_name in metadata.target_schemas:
            schema_validation = self._validate_against_schema(schema_name, sample_records)
            validation_report['schema_compatibility'][schema_name] = schema_validation
            
            if not schema_validation['compatible']:
                validation_report['overall_valid'] = False
        
        # Field coverage analysis
        field_coverage = self._analyze_field_coverage(sample_records)
        validation_report['field_coverage'] = field_coverage
        
        # Data quality checks
        quality_checks = self._perform_quality_checks(sample_records)
        validation_report['data_quality'] = quality_checks
        
        return validation_report
    
    def _validate_against_schema(
        self,
        schema_name: str,
        sample_records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate records against a specific target schema."""
        schema_validation = {
            'schema_name': schema_name,
            'compatible': True,
            'required_fields_present': [],
            'missing_required_fields': [],
            'field_type_matches': {},
            'sample_record_valid': False,
        }
        
        # Get expected fields for schema (simplified mapping)
        expected_fields = self._get_expected_schema_fields(schema_name)
        
        if sample_records:
            sample_record = sample_records[0]
            
            # Check required fields
            for field in expected_fields.get('required', []):
                if field in sample_record:
                    schema_validation['required_fields_present'].append(field)
                else:
                    schema_validation['missing_required_fields'].append(field)
                    schema_validation['compatible'] = False
            
            # Check field types
            for field, expected_type in expected_fields.get('types', {}).items():
                if field in sample_record:
                    actual_value = sample_record[field]
                    type_match = self._check_field_type(actual_value, expected_type)
                    schema_validation['field_type_matches'][field] = type_match
                    if not type_match:
                        schema_validation['compatible'] = False
            
            # Overall sample record validation
            schema_validation['sample_record_valid'] = (
                len(schema_validation['missing_required_fields']) == 0
            )
        
        return schema_validation
    
    def _get_expected_schema_fields(self, schema_name: str) -> Dict[str, Any]:
        """Get expected fields for a target schema."""
        # Simplified schema field definitions for validation
        schema_fields = {
            'HealthIndicator': {
                'required': ['geographic_id', 'indicator_name', 'value', 'unit'],
                'types': {
                    'geographic_id': 'str',
                    'value': 'float',
                    'reference_year': 'int',
                }
            },
            'MortalityData': {
                'required': ['geographic_id', 'cause_of_death', 'mortality_rate'],
                'types': {
                    'deaths_count': 'int',
                    'mortality_rate': 'float',
                }
            },
            'HealthcareUtilisation': {
                'required': ['geographic_id', 'service_type', 'utilisation_rate'],
                'types': {
                    'visits_count': 'int',
                    'utilisation_rate': 'float',
                }
            },
            'GeographicBoundary': {
                'required': ['geographic_id', 'geographic_name', 'area_square_km'],
                'types': {
                    'area_square_km': 'float',
                }
            },
            'CensusData': {
                'required': ['geographic_id', 'census_year', 'total_population'],
                'types': {
                    'total_population': 'int',
                    'census_year': 'int',
                }
            },
        }
        
        return schema_fields.get(schema_name, {'required': [], 'types': {}})
    
    def _check_field_type(self, value: Any, expected_type: str) -> bool:
        """Check if a field value matches the expected type."""
        if value is None:
            return True  # None is acceptable for optional fields
        
        type_checks = {
            'str': lambda x: isinstance(x, str),
            'int': lambda x: isinstance(x, (int, float)) and float(x).is_integer(),
            'float': lambda x: isinstance(x, (int, float)),
            'bool': lambda x: isinstance(x, bool),
            'list': lambda x: isinstance(x, list),
            'dict': lambda x: isinstance(x, dict),
        }
        
        check_func = type_checks.get(expected_type)
        if check_func:
            return check_func(value)
        
        return True  # Unknown type, assume valid
    
    def _analyze_field_coverage(self, sample_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze field coverage across sample records."""
        if not sample_records:
            return {}
        
        all_fields = set()
        field_counts = {}
        
        for record in sample_records:
            all_fields.update(record.keys())
            for field, value in record.items():
                if field not in field_counts:
                    field_counts[field] = {'total': 0, 'non_null': 0}
                field_counts[field]['total'] += 1
                if value is not None:
                    field_counts[field]['non_null'] += 1
        
        coverage_analysis = {
            'total_fields': len(all_fields),
            'field_completeness': {},
        }
        
        for field, counts in field_counts.items():
            completeness = counts['non_null'] / counts['total'] * 100
            coverage_analysis['field_completeness'][field] = {
                'completeness_percent': round(completeness, 2),
                'non_null_count': counts['non_null'],
                'total_count': counts['total'],
            }
        
        return coverage_analysis
    
    def _perform_quality_checks(self, sample_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform basic data quality checks."""
        quality_report = {
            'duplicate_records': 0,
            'records_with_nulls': 0,
            'potential_data_issues': [],
        }
        
        # Check for duplicates (simplified)
        seen_records = set()
        for i, record in enumerate(sample_records):
            record_key = str(sorted(record.items()))
            if record_key in seen_records:
                quality_report['duplicate_records'] += 1
            else:
                seen_records.add(record_key)
        
        # Check for records with many null values
        for record in sample_records:
            null_count = sum(1 for value in record.values() if value is None)
            if null_count > len(record) * 0.5:  # More than 50% null
                quality_report['records_with_nulls'] += 1
        
        return quality_report


class ExtractorMonitor:
    """
    Monitor for tracking extractor performance and quality.
    
    Monitors extraction performance, data quality, and system health
    for continuous improvement of the extraction process.
    """
    
    def __init__(self, registry: ExtractorRegistry):
        self.registry = registry
        self._active_jobs: Dict[str, ExtractionJob] = {}
        self._job_history: List[ExtractionJob] = []
        self._performance_metrics: Dict[str, List[Dict[str, Any]]] = {}
    
    def start_job(
        self,
        extractor_type: ExtractorType,
        source: Union[str, Path, Dict[str, Any]],
        parameters: Dict[str, Any] = None,
    ) -> str:
        """Start monitoring an extraction job."""
        job_id = f"{extractor_type.value}_{int(time.time())}"
        
        job = ExtractionJob(
            job_id=job_id,
            extractor_type=extractor_type,
            source=source,
            parameters=parameters or {},
            status=ProcessingStatus.RUNNING,
            started_at=datetime.now(),
        )
        
        self._active_jobs[job_id] = job
        logger.info(f"Started monitoring extraction job: {job_id}")
        
        return job_id
    
    def update_job_progress(
        self,
        job_id: str,
        records_extracted: int,
        status: Optional[ProcessingStatus] = None,
    ) -> None:
        """Update job progress."""
        if job_id in self._active_jobs:
            job = self._active_jobs[job_id]
            job.records_extracted = records_extracted
            if status:
                job.status = status
    
    def complete_job(
        self,
        job_id: str,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> None:
        """Complete an extraction job."""
        if job_id not in self._active_jobs:
            logger.warning(f"Unknown job ID: {job_id}")
            return
        
        job = self._active_jobs[job_id]
        job.completed_at = datetime.now()
        job.status = ProcessingStatus.COMPLETED if success else ProcessingStatus.FAILED
        if error_message:
            job.error_message = error_message
        
        # Move to history
        self._job_history.append(job)
        del self._active_jobs[job_id]
        
        # Update extractor metadata
        if success:
            metadata = self.registry.get_extractor_metadata(job.extractor_type)
            if metadata:
                metadata.last_successful_run = job.completed_at
        
        logger.info(f"Completed extraction job: {job_id} ({'success' if success else 'failed'})")
    
    def get_job_status(self, job_id: str) -> Optional[ExtractionJob]:
        """Get status of an extraction job."""
        return self._active_jobs.get(job_id)
    
    def get_performance_summary(
        self,
        extractor_type: Optional[ExtractorType] = None,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """Get performance summary for extractors."""
        cutoff_time = datetime.now().replace(
            hour=datetime.now().hour - hours,
            minute=0,
            second=0,
            microsecond=0
        )
        
        # Filter jobs
        recent_jobs = [
            job for job in self._job_history
            if job.created_at >= cutoff_time
        ]
        
        if extractor_type:
            recent_jobs = [job for job in recent_jobs if job.extractor_type == extractor_type]
        
        # Calculate metrics
        total_jobs = len(recent_jobs)
        successful_jobs = len([job for job in recent_jobs if job.status == ProcessingStatus.COMPLETED])
        failed_jobs = len([job for job in recent_jobs if job.status == ProcessingStatus.FAILED])
        
        total_records = sum(job.records_extracted for job in recent_jobs)
        avg_duration = None
        if recent_jobs:
            durations = [job.duration_seconds for job in recent_jobs if job.duration_seconds]
            if durations:
                avg_duration = sum(durations) / len(durations)
        
        return {
            'time_period_hours': hours,
            'total_jobs': total_jobs,
            'successful_jobs': successful_jobs,
            'failed_jobs': failed_jobs,
            'success_rate_percent': (successful_jobs / total_jobs * 100) if total_jobs > 0 else 0,
            'total_records_extracted': total_records,
            'average_duration_seconds': avg_duration,
            'active_jobs_count': len(self._active_jobs),
        }
    
    def get_extractor_health_status(self) -> Dict[str, Any]:
        """Get overall health status of all extractors."""
        health_status = {
            'overall_status': 'healthy',
            'total_extractors': len(self.registry._extractors),
            'enabled_extractors': len([e for e in self.registry._extractors.values() if e.enabled]),
            'extractor_statuses': {},
            'dependency_issues': [],
            'last_successful_runs': {},
        }
        
        # Check dependency issues
        dependency_errors = self.registry.validate_dependencies()
        if dependency_errors:
            health_status['overall_status'] = 'degraded'
            health_status['dependency_issues'] = dependency_errors
        
        # Check extractor-specific health
        for extractor_type, metadata in self.registry._extractors.items():
            extractor_health = {
                'enabled': metadata.enabled,
                'last_successful_run': metadata.last_successful_run,
                'status': 'healthy' if metadata.enabled else 'disabled',
            }
            
            # Check recent performance
            recent_jobs = [
                job for job in self._job_history[-10:]  # Last 10 jobs
                if job.extractor_type == extractor_type
            ]
            
            if recent_jobs:
                recent_failures = [job for job in recent_jobs if job.status == ProcessingStatus.FAILED]
                if len(recent_failures) > len(recent_jobs) * 0.5:  # More than 50% failed
                    extractor_health['status'] = 'unhealthy'
                    health_status['overall_status'] = 'degraded'
            
            health_status['extractor_statuses'][extractor_type.value] = extractor_health
            
            if metadata.last_successful_run:
                health_status['last_successful_runs'][extractor_type.value] = metadata.last_successful_run.isoformat()
        
        return health_status


# Global registry instance
_global_registry = ExtractorRegistry()


def get_extractor_registry() -> ExtractorRegistry:
    """Get the global extractor registry."""
    return _global_registry


def get_extractor_factory() -> ExtractorFactory:
    """Get an extractor factory with the global registry."""
    return ExtractorFactory(_global_registry)


def get_extractor_validator() -> ExtractorValidator:
    """Get an extractor validator with the global registry."""
    return ExtractorValidator(_global_registry)


def get_extractor_monitor() -> ExtractorMonitor:
    """Get an extractor monitor with the global registry."""
    return ExtractorMonitor(_global_registry)