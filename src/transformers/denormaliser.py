"""
Denormalisation classes for AHGD project.

This module implements denormalisation processes to create wide-format tables
from normalised health data inputs, suitable for analysis and reporting.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass

from .base import BaseTransformer
from schemas.integrated_schema import MasterHealthRecord, SA2HealthProfile
from ..utils.interfaces import DataBatch, TransformationError
from ..utils.logging import get_logger


@dataclass
class DenormalisationStrategy:
    """Configuration for denormalisation approach."""
    
    strategy_name: str
    flatten_nested: bool = True
    include_metadata: bool = True
    prefix_sources: bool = True
    handle_nulls: str = "preserve"  # preserve, drop, fill_zero
    array_handling: str = "expand"  # expand, concatenate, json


class HealthDataDenormaliser(BaseTransformer):
    """
    Creates wide-format tables from normalised health data inputs.
    
    Transforms structured health records into flat, denormalised tables
    suitable for analytical tools and reporting systems.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialise the health data denormaliser.
        
        Args:
            config: Configuration including denormalisation strategies
            logger: Optional logger instance
        """
        super().__init__("health_data_denormaliser", config, logger)
        
        self.logger = get_logger(__name__) if logger is None else logger
        
        # Denormalisation configuration
        self.strategy = DenormalisationStrategy(
            strategy_name=config.get('strategy_name', 'comprehensive'),
            flatten_nested=config.get('flatten_nested', True),
            include_metadata=config.get('include_metadata', True),
            prefix_sources=config.get('prefix_sources', True),
            handle_nulls=config.get('handle_nulls', 'preserve'),
            array_handling=config.get('array_handling', 'expand')
        )
        
        # Field configuration
        self.field_mappings = config.get('field_mappings', {})
        self.excluded_fields = config.get('excluded_fields', [])
        self.priority_fields = config.get('priority_fields', [])
        
        # Flattening configuration
        self.max_nesting_depth = config.get('max_nesting_depth', 3)
        self.separator = config.get('separator', '_')
        
    def transform(self, data: DataBatch, **kwargs) -> DataBatch:
        """
        Transform normalised health records into denormalised wide format.
        
        Args:
            data: Batch of MasterHealthRecord or similar structured records
            **kwargs: Additional transformation parameters
            
        Returns:
            DataBatch: Denormalised wide-format records
        """
        try:
            denormalised_records = []
            
            for record in data:
                # Denormalise each record
                flat_record = self.denormalise_health_record(record)
                if flat_record:
                    denormalised_records.append(flat_record)
            
            self.logger.info(f"Denormalised {len(denormalised_records)} health records")
            return denormalised_records
            
        except Exception as e:
            self.logger.error(f"Health data denormalisation failed: {e}")
            raise TransformationError(f"Health data denormalisation failed: {e}") from e
    
    def denormalise_health_record(self, record: Union[Dict[str, Any], MasterHealthRecord]) -> Dict[str, Any]:
        """
        Denormalise a single health record into wide format.
        
        Args:
            record: Health record to denormalise
            
        Returns:
            Flattened dictionary with all fields in wide format
        """
        try:
            # Convert record to dictionary if it's a schema object
            if hasattr(record, '__dict__'):
                record_dict = record.__dict__
            else:
                record_dict = record
            
            # Start denormalisation
            denormalised = {}
            
            # Process core identification fields first
            denormalised.update(self._extract_core_fields(record_dict))
            
            # Flatten nested structures
            if self.strategy.flatten_nested:
                denormalised.update(self._flatten_nested_structures(record_dict))
            
            # Process arrays and lists
            denormalised.update(self._process_arrays(record_dict))
            
            # Add metadata if configured
            if self.strategy.include_metadata:
                denormalised.update(self._add_metadata_fields(record_dict))
            
            # Apply field mappings
            denormalised = self._apply_field_mappings(denormalised)
            
            # Handle null values
            denormalised = self._handle_null_values(denormalised)
            
            # Remove excluded fields
            denormalised = self._remove_excluded_fields(denormalised)
            
            return denormalised
            
        except Exception as e:
            self.logger.error(f"Failed to denormalise health record: {e}")
            return {}
    
    def get_schema(self) -> Dict[str, str]:
        """Get the expected output schema."""
        return {
            'sa2_code': 'string',
            'sa2_name': 'string',
            'total_population': 'integer',
            'life_expectancy': 'float',
            'seifa_irsd_score': 'integer',
            'seifa_irsd_decile': 'integer',
            'gp_services_per_1000': 'float',
            'data_completeness_score': 'float'
        }
    
    def _extract_core_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract core identification and primary fields."""
        core_fields = {}
        
        # Always include primary identification
        priority_keys = [
            'sa2_code', 'sa2_name', 'sa3_code', 'sa4_code', 'state_code',
            'total_population', 'population_density_per_sq_km', 'median_age'
        ]
        
        for key in priority_keys:
            if key in record:
                core_fields[key] = record[key]
        
        return core_fields
    
    def _flatten_nested_structures(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested dictionaries and objects."""
        flattened = {}
        
        def _flatten_dict(obj: Dict[str, Any], prefix: str = '', depth: int = 0) -> None:
            """Recursively flatten nested dictionary."""
            if depth >= self.max_nesting_depth:
                return
            
            for key, value in obj.items():
                new_key = f"{prefix}{self.separator}{key}" if prefix else key
                
                if isinstance(value, dict):
                    _flatten_dict(value, new_key, depth + 1)
                elif isinstance(value, list) and self.strategy.array_handling == "expand":
                    # Handle arrays by expanding to multiple columns
                    self._expand_array_to_columns(value, new_key, flattened)
                else:
                    flattened[new_key] = value
        
        # Process nested structures
        nested_fields = [
            'geographic_hierarchy', 'demographic_profile', 'seifa_scores', 
            'seifa_deciles', 'health_outcomes_summary', 'mortality_indicators',
            'chronic_disease_prevalence', 'mental_health_indicators',
            'healthcare_access', 'pharmaceutical_utilisation', 'risk_factors',
            'environmental_indicators'
        ]
        
        for field in nested_fields:
            if field in record and isinstance(record[field], dict):
                prefix = field if self.strategy.prefix_sources else ''
                _flatten_dict(record[field], prefix)
        
        return flattened
    
    def _process_arrays(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Process array and list fields."""
        array_data = {}
        
        array_fields = ['source_datasets', 'missing_indicators', 'quality_flags']
        
        for field in array_fields:
            if field in record and isinstance(record[field], list):
                if self.strategy.array_handling == "expand":
                    # Create separate columns for each array element
                    for i, item in enumerate(record[field]):
                        array_data[f"{field}{self.separator}{i}"] = item
                elif self.strategy.array_handling == "concatenate":
                    # Join array elements into single string
                    array_data[field] = ', '.join(str(item) for item in record[field])
                elif self.strategy.array_handling == "json":
                    # Store as JSON string
                    import json
                    array_data[field] = json.dumps(record[field])
                else:
                    # Keep as is
                    array_data[field] = record[field]
        
        return array_data
    
    def _expand_array_to_columns(self, array: List[Any], base_key: str, target_dict: Dict[str, Any]) -> None:
        """Expand array elements to separate columns."""
        for i, item in enumerate(array):
            column_key = f"{base_key}{self.separator}{i}"
            target_dict[column_key] = item
    
    def _add_metadata_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Add metadata fields for data lineage and quality."""
        metadata = {}
        
        metadata_fields = [
            'integration_level', 'data_completeness_score', 'integration_timestamp',
            'composite_health_index', 'health_inequality_index', 'schema_version'
        ]
        
        for field in metadata_fields:
            if field in record:
                metadata[field] = record[field]
        
        # Add computed metadata
        metadata['record_processed_timestamp'] = datetime.utcnow().isoformat()
        metadata['denormalisation_strategy'] = self.strategy.strategy_name
        
        return metadata
    
    def _apply_field_mappings(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Apply configured field name mappings."""
        if not self.field_mappings:
            return record
        
        mapped_record = {}
        
        for original_key, value in record.items():
            # Check if there's a mapping for this key
            mapped_key = self.field_mappings.get(original_key, original_key)
            mapped_record[mapped_key] = value
        
        return mapped_record
    
    def _handle_null_values(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Handle null values according to strategy."""
        if self.strategy.handle_nulls == "preserve":
            return record
        elif self.strategy.handle_nulls == "drop":
            return {k: v for k, v in record.items() if v is not None}
        elif self.strategy.handle_nulls == "fill_zero":
            return {k: (v if v is not None else 0) for k, v in record.items()}
        else:
            return record
    
    def _remove_excluded_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Remove fields specified in exclusion list."""
        if not self.excluded_fields:
            return record
        
        return {k: v for k, v in record.items() if k not in self.excluded_fields}


class GeographicDenormaliser(BaseTransformer):
    """
    Flattens geographic hierarchy into single records.
    
    Combines SA1, SA2, SA3, SA4, and State information into wide-format
    records with complete geographic context.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialise the geographic denormaliser."""
        super().__init__("geographic_denormaliser", config, logger)
        
        self.logger = get_logger(__name__) if logger is None else logger
        self.include_geometry = config.get('include_geometry', False)
        self.geometry_format = config.get('geometry_format', 'wkt')  # wkt, geojson, coordinates
        
    def transform(self, data: DataBatch, **kwargs) -> DataBatch:
        """Transform geographic data into denormalised format."""
        try:
            denormalised_records = []
            
            for record in data:
                flat_record = self.denormalise_geographic_record(record)
                if flat_record:
                    denormalised_records.append(flat_record)
            
            return denormalised_records
            
        except Exception as e:
            self.logger.error(f"Geographic denormalisation failed: {e}")
            raise TransformationError(f"Geographic denormalisation failed: {e}") from e
    
    def denormalise_geographic_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Denormalise geographic hierarchy into flat structure."""
        denormalised = {}
        
        # Extract geographic hierarchy
        hierarchy = record.get('geographic_hierarchy', {})
        
        # Add all hierarchy levels
        geographic_fields = {
            'sa2_code': record.get('sa2_code'),
            'sa2_name': record.get('sa2_name'),
            'sa3_code': hierarchy.get('sa3_code'),
            'sa3_name': hierarchy.get('sa3_name'),
            'sa4_code': hierarchy.get('sa4_code'),
            'sa4_name': hierarchy.get('sa4_name'),
            'state_code': hierarchy.get('state_code'),
            'state_name': hierarchy.get('state_name'),
            'postcode': hierarchy.get('postcode')
        }
        
        denormalised.update({k: v for k, v in geographic_fields.items() if v is not None})
        
        # Add boundary information
        boundary_data = record.get('boundary_data', {})
        if isinstance(boundary_data, dict):
            denormalised.update({
                'centroid_latitude': boundary_data.get('centroid_latitude'),
                'centroid_longitude': boundary_data.get('centroid_longitude'),
                'area_sq_km': boundary_data.get('area_sq_km'),
                'boundary_source': boundary_data.get('boundary_source')
            })
        
        # Add classification information
        denormalised.update({
            'urbanisation': record.get('urbanisation'),
            'remoteness_category': record.get('remoteness_category')
        })
        
        # Handle geometry if requested
        if self.include_geometry and 'boundary_data' in record:
            geometry = record['boundary_data'].get('geometry')
            if geometry:
                denormalised['geometry'] = self._format_geometry(geometry)
        
        return denormalised
    
    def get_schema(self) -> Dict[str, str]:
        """Get the expected output schema."""
        return {
            'sa2_code': 'string',
            'sa2_name': 'string',
            'sa3_code': 'string',
            'sa4_code': 'string',
            'state_code': 'string',
            'centroid_latitude': 'float',
            'centroid_longitude': 'float',
            'area_sq_km': 'float'
        }
    
    def _format_geometry(self, geometry: Dict[str, Any]) -> str:
        """Format geometry according to specified format."""
        if self.geometry_format == 'geojson':
            import json
            return json.dumps(geometry)
        elif self.geometry_format == 'wkt':
            # Convert to WKT format (simplified)
            return self._geojson_to_wkt(geometry)
        elif self.geometry_format == 'coordinates':
            # Return coordinate string
            return str(geometry.get('coordinates', []))
        else:
            return str(geometry)
    
    def _geojson_to_wkt(self, geojson: Dict[str, Any]) -> str:
        """Convert GeoJSON to WKT format (simplified implementation)."""
        geom_type = geojson.get('type', '').upper()
        coordinates = geojson.get('coordinates', [])
        
        if geom_type == 'POLYGON':
            # Simple polygon WKT
            if coordinates:
                coord_string = ', '.join(f"{lon} {lat}" for lon, lat in coordinates[0])
                return f"POLYGON(({coord_string}))"
        
        return f"{geom_type}({coordinates})"


class TemporalDenormaliser(BaseTransformer):
    """
    Handles time-series data aggregation into wide format.
    
    Transforms temporal health data into columns representing different
    time periods for trend analysis.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialise the temporal denormaliser."""
        super().__init__("temporal_denormaliser", config, logger)
        
        self.logger = get_logger(__name__) if logger is None else logger
        self.time_periods = config.get('time_periods', ['current', 'previous_year', 'trend_5yr'])
        self.aggregation_method = config.get('aggregation_method', 'latest')
        
    def transform(self, data: DataBatch, **kwargs) -> DataBatch:
        """Transform temporal data into denormalised format."""
        try:
            denormalised_records = []
            
            for record in data:
                flat_record = self.denormalise_temporal_record(record)
                if flat_record:
                    denormalised_records.append(flat_record)
            
            return denormalised_records
            
        except Exception as e:
            self.logger.error(f"Temporal denormalisation failed: {e}")
            raise TransformationError(f"Temporal denormalisation failed: {e}") from e
    
    def denormalise_temporal_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Denormalise temporal aspects of a health record."""
        denormalised = {}
        
        # Extract base record
        denormalised['sa2_code'] = record.get('sa2_code')
        
        # Process temporal fields
        for period in self.time_periods:
            period_data = record.get(f'data_{period}', {})
            
            for field, value in period_data.items():
                denormalised[f"{field}_{period}"] = value
        
        # Add temporal metadata
        denormalised['reference_period'] = record.get('reference_period')
        denormalised['data_currency'] = record.get('last_updated')
        
        return denormalised
    
    def get_schema(self) -> Dict[str, str]:
        """Get the expected output schema."""
        return {
            'sa2_code': 'string',
            'reference_period': 'string',
            'data_currency': 'datetime'
        }


class MetadataDenormaliser(BaseTransformer):
    """
    Embeds data lineage and quality metadata into denormalised records.
    
    Adds comprehensive metadata about data sources, processing history,
    and quality assessments to support data governance.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialise the metadata denormaliser."""
        super().__init__("metadata_denormaliser", config, logger)
        
        self.logger = get_logger(__name__) if logger is None else logger
        self.include_lineage = config.get('include_lineage', True)
        self.include_quality_metrics = config.get('include_quality_metrics', True)
        self.include_processing_history = config.get('include_processing_history', False)
        
    def transform(self, data: DataBatch, **kwargs) -> DataBatch:
        """Transform records to include comprehensive metadata."""
        try:
            enhanced_records = []
            
            for record in data:
                enhanced_record = self.add_metadata_to_record(record)
                enhanced_records.append(enhanced_record)
            
            return enhanced_records
            
        except Exception as e:
            self.logger.error(f"Metadata denormalisation failed: {e}")
            raise TransformationError(f"Metadata denormalisation failed: {e}") from e
    
    def add_metadata_to_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Add comprehensive metadata to a record."""
        enhanced = record.copy()
        
        # Add data lineage information
        if self.include_lineage:
            enhanced.update(self._extract_lineage_metadata(record))
        
        # Add quality metrics
        if self.include_quality_metrics:
            enhanced.update(self._extract_quality_metadata(record))
        
        # Add processing history
        if self.include_processing_history:
            enhanced.update(self._extract_processing_metadata(record))
        
        # Add standardised metadata
        enhanced.update(self._add_standard_metadata())
        
        return enhanced
    
    def get_schema(self) -> Dict[str, str]:
        """Get the expected output schema."""
        return {
            'sa2_code': 'string',
            'source_count': 'integer',
            'data_quality_score': 'float',
            'last_processed': 'datetime'
        }
    
    def _extract_lineage_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data lineage metadata."""
        lineage = {}
        
        # Source datasets
        source_datasets = record.get('source_datasets', [])
        lineage['source_count'] = len(source_datasets)
        
        # Create columns for each potential source
        expected_sources = ['census', 'seifa', 'health_indicators', 'geographic_boundaries', 'medicare_pbs']
        for source in expected_sources:
            lineage[f'has_{source}_data'] = source in source_datasets
        
        # Data integration level
        lineage['integration_level'] = record.get('integration_level')
        
        return lineage
    
    def _extract_quality_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data quality metadata."""
        quality = {}
        
        # Quality scores
        quality['data_completeness_score'] = record.get('data_completeness_score')
        quality['composite_health_index'] = record.get('composite_health_index')
        quality['health_inequality_index'] = record.get('health_inequality_index')
        
        # Missing indicators
        missing_indicators = record.get('missing_indicators', [])
        quality['missing_indicator_count'] = len(missing_indicators)
        
        # Quality flags
        quality_flags = record.get('quality_flags', [])
        for flag in ['high_confidence', 'complete_seifa', 'validated_geography']:
            quality[f'quality_flag_{flag}'] = flag in quality_flags
        
        return quality
    
    def _extract_processing_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract processing history metadata."""
        processing = {}
        
        processing['integration_timestamp'] = record.get('integration_timestamp')
        processing['schema_version'] = record.get('schema_version')
        
        return processing
    
    def _add_standard_metadata(self) -> Dict[str, Any]:
        """Add standard processing metadata."""
        return {
            'denormalised_timestamp': datetime.utcnow().isoformat(),
            'denormalisation_version': '1.0.0',
            'processor': 'AHGD_ETL_Pipeline'
        }