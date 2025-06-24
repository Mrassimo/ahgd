"""
BOM (Bureau of Meteorology) data extractors.

This module provides extractors for BOM data sources including climate data,
weather station data, and environmental indicators.
All extractors work backwards from target schema requirements.
"""

import csv
import io
import json
import re
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
from urllib.parse import urljoin, urlparse

import pandas as pd

from .base import BaseExtractor
from ..utils.interfaces import (
    DataBatch,
    DataRecord,
    ExtractionError,
    DataExtractionError,
    SourceMetadata,
)
from ..utils.logging import get_logger
from schemas.environmental_schema import (
    WeatherObservation,
    ClimateStatistics,
    EnvironmentalHealthIndex,
)
from schemas.integrated_schema import (
    MasterHealthRecord,
    GeographicHealthMapping,
)


logger = get_logger(__name__)


class BOMClimateExtractor(BaseExtractor):
    """
    Extractor for BOM climate data for SA2 areas.
    
    Extracts climate data compatible with ClimateData schema and 
    environmental indicators for health analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("bom_climate", config)
        self.base_url = config.get('bom_base_url', 'http://www.bom.gov.au/climate/data/')
        self.api_key = config.get('api_key')
        
        # Target schema field mappings
        self.climate_field_mappings = {
            'station_id': ['STATION_ID', 'station_number', 'station_id'],
            'station_name': ['STATION_NAME', 'station_name'],
            'latitude': ['LATITUDE', 'lat', 'latitude'],
            'longitude': ['LONGITUDE', 'lon', 'longitude'],
            'date': ['DATE', 'YYYY-MM-DD', 'date'],
            'temperature_max': ['TMAX', 'maximum_temperature', 'temp_max'],
            'temperature_min': ['TMIN', 'minimum_temperature', 'temp_min'],
            'rainfall': ['RAINFALL', 'precipitation', 'rainfall_mm'],
            'humidity_9am': ['HUMIDITY_9AM', 'relative_humidity_9am'],
            'humidity_3pm': ['HUMIDITY_3PM', 'relative_humidity_3pm'],
            'wind_speed': ['WIND_SPEED', 'wind_speed_kmh'],
            'solar_exposure': ['SOLAR_EXPOSURE', 'daily_global_solar_exposure'],
        }
    
    def extract(
        self,
        source: Union[str, Path, Dict[str, Any]],
        **kwargs
    ) -> Iterator[DataBatch]:
        """
        Extract BOM climate data.
        
        Args:
            source: BOM climate data source specification
            **kwargs: Additional parameters
            
        Yields:
            DataBatch: Batches of climate records
        """
        try:
            # Handle different source types
            if isinstance(source, dict):
                station_ids = source.get('station_ids', [])
                start_date = source.get('start_date')
                end_date = source.get('end_date')
                data_type = source.get('data_type', 'daily')
            else:
                station_ids = kwargs.get('station_ids', [])
                start_date = kwargs.get('start_date')
                end_date = kwargs.get('end_date')
                data_type = kwargs.get('data_type', 'daily')
            
            # Validate source
            if not self.validate_source(source):
                raise ExtractionError(f"Invalid BOM climate source: {source}")
            
            # Extract data based on parameters - NO FALLBACKS to demo data
            if station_ids:
                try:
                    yield from self._extract_station_data(station_ids, start_date, end_date, data_type)
                except Exception as extraction_error:
                    raise DataExtractionError(
                        f"BOM climate data extraction failed for stations",
                        source=str(station_ids),
                        source_type="api"
                    ) from extraction_error
            else:
                # No station IDs provided - this is a configuration error, not a fallback case
                raise DataExtractionError(
                    "No station data available for BOM climate extraction. "
                    "Station IDs must be provided for real data extraction.",
                    source=str(source),
                    source_type="config"
                )
                
        except DataExtractionError:
            # Re-raise DataExtractionError as-is
            raise
        except Exception as e:
            logger.error(f"BOM climate extraction failed: {e}")
            raise ExtractionError(f"BOM climate extraction failed: {e}")
    
    def _extract_station_data(
        self,
        station_ids: List[str],
        start_date: Optional[str],
        end_date: Optional[str],
        data_type: str
    ) -> Iterator[DataBatch]:
        """Extract climate data from specific stations."""
        for station_id in station_ids:
            logger.info(f"Extracting climate data from station {station_id}")
            
            try:
                # Build BOM data URL
                url = self._build_station_url(station_id, data_type)
                
                # Download and parse data
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Parse CSV data
                yield from self._parse_climate_csv(response.text, station_id, start_date, end_date)
                
            except requests.RequestException as e:
                logger.error(f"Failed to extract data from station {station_id}: {e}")
                raise DataExtractionError(
                    f"BOM climate data extraction failed for station {station_id}",
                    source=station_id,
                    source_type="api"
                ) from e
    
    def _build_station_url(self, station_id: str, data_type: str) -> str:
        """Build BOM data URL for station."""
        # BOM URL format for daily weather data
        return f"{self.base_url}observations/daily/{station_id}.csv"
    
    def _parse_climate_csv(
        self,
        csv_data: str,
        station_id: str,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> Iterator[DataBatch]:
        """Parse BOM climate CSV data."""
        lines = csv_data.strip().split('\n')
        
        # Skip header lines (BOM files often have metadata at top)
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith('Date') or 'YYYY-MM-DD' in line:
                data_start = i
                break
        
        if data_start == 0:
            logger.warning(f"Could not find data header in station {station_id}")
            return
        
        # Parse CSV data
        reader = csv.DictReader(lines[data_start:])
        
        batch = []
        for row in reader:
            # Filter by date range if specified
            if start_date or end_date:
                record_date = row.get('Date') or row.get('YYYY-MM-DD')
                if record_date:
                    if start_date and record_date < start_date:
                        continue
                    if end_date and record_date > end_date:
                        continue
            
            # Map fields to target schema
            mapped_record = self._map_climate_fields(row, station_id)
            
            # Validate and transform
            validated_record = self._validate_climate_record(mapped_record)
            
            if validated_record:
                batch.append(validated_record)
                
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
        
        # Yield remaining records
        if batch:
            yield batch
    
    def _map_climate_fields(self, record: Dict[str, Any], station_id: str) -> Dict[str, Any]:
        """Map source fields to target schema fields."""
        mapped = {}
        
        # Map each target field
        for target_field, source_fields in self.climate_field_mappings.items():
            for source_field in source_fields:
                if source_field in record and record[source_field] not in [None, '', 'NA', '-']:
                    mapped[target_field] = record[source_field]
                    break
        
        # Add station ID if not mapped
        if 'station_id' not in mapped:
            mapped['station_id'] = station_id
        
        # Add metadata
        mapped['data_source'] = 'BOM'
        mapped['extraction_timestamp'] = datetime.now().isoformat()
        mapped['source_record'] = record  # Keep original for debugging
        
        return mapped
    
    def _validate_climate_record(self, record: Dict[str, Any]) -> Optional[DataRecord]:
        """Validate climate record against target schema."""
        try:
            # Ensure required fields are present
            required_fields = ['station_id', 'date']
            for field in required_fields:
                if field not in record or record[field] is None:
                    return None
            
            # Validate date format
            date_str = record['date']
            try:
                parsed_date = datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                try:
                    parsed_date = datetime.strptime(date_str, '%d/%m/%Y')
                    date_str = parsed_date.strftime('%Y-%m-%d')
                except ValueError:
                    logger.warning(f"Invalid date format: {date_str}")
                    return None
            
            # Convert numeric fields
            numeric_fields = [
                'latitude', 'longitude', 'temperature_max', 'temperature_min',
                'rainfall', 'humidity_9am', 'humidity_3pm', 'wind_speed', 'solar_exposure'
            ]
            
            for field in numeric_fields:
                if field in record and record[field] is not None:
                    try:
                        record[field] = float(record[field])
                    except (ValueError, TypeError):
                        record[field] = None
            
            # Build target schema compatible record
            target_record = {
                'station_id': record['station_id'],
                'station_name': record.get('station_name', ''),
                'measurement_date': date_str,
                'latitude': record.get('latitude'),
                'longitude': record.get('longitude'),
                'temperature_max_celsius': record.get('temperature_max'),
                'temperature_min_celsius': record.get('temperature_min'),
                'rainfall_mm': record.get('rainfall'),
                'relative_humidity_9am_percent': record.get('humidity_9am'),
                'relative_humidity_3pm_percent': record.get('humidity_3pm'),
                'wind_speed_kmh': record.get('wind_speed'),
                'solar_exposure_mj_per_m2': record.get('solar_exposure'),
                'data_source_id': 'BOM_CLIMATE',
                'data_source_name': 'Bureau of Meteorology Climate Data',
                'extraction_timestamp': record['extraction_timestamp'],
            }
            
            # Calculate derived indicators for health analysis
            if target_record['temperature_max_celsius'] is not None:
                # Heat stress indicator
                if target_record['temperature_max_celsius'] > 35:
                    target_record['heat_stress_indicator'] = 'high'
                elif target_record['temperature_max_celsius'] > 30:
                    target_record['heat_stress_indicator'] = 'moderate'
                else:
                    target_record['heat_stress_indicator'] = 'low'
            
            return target_record
            
        except Exception as e:
            logger.error(f"Climate record validation failed: {e}")
            return None
    
    
    def get_source_metadata(
        self,
        source: Union[str, Path, Dict[str, Any]]
    ) -> SourceMetadata:
        """Get metadata about the BOM climate source."""
        if isinstance(source, dict):
            source_id = 'bom_climate'
            station_ids = source.get('station_ids', [])
        else:
            source_id = 'bom_climate'
            station_ids = []
        
        metadata = SourceMetadata(
            source_id=source_id,
            source_type='api',
            source_url=self.base_url,
            schema_version='1.0.0',
        )
        
        # Add station-specific metadata
        if station_ids:
            metadata.parameters = {'station_ids': station_ids}
            
        return metadata
    
    def validate_source(
        self,
        source: Union[str, Path, Dict[str, Any]]
    ) -> bool:
        """Validate BOM climate source."""
        try:
            if isinstance(source, dict):
                return True  # Basic validation for demo
            elif isinstance(source, str):
                return source.startswith('http') or source == 'demo'
            return False
        except Exception:
            return False


class BOMWeatherStationExtractor(BaseExtractor):
    """
    Extractor for BOM weather station data with SA2 mapping.
    
    Extracts weather station metadata and creates mappings to SA2 areas.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("bom_weather_stations", config)
        
    def extract(self, source, **kwargs) -> Iterator[DataBatch]:
        """Extract BOM weather station data."""
        logger.info("Extracting BOM weather station mappings")
        
        # Weather station real data extraction not yet implemented
        # In production, this would attempt to connect to BOM weather station metadata APIs
        raise DataExtractionError(
            "Real data extraction not implemented for BOM Weather Station extractor. "
            "Production systems must implement actual weather station metadata source connections.",
            source=str(source),
            source_type="station_api"
        )
    
    
    def get_source_metadata(self, source) -> SourceMetadata:
        """Get weather station source metadata."""
        return SourceMetadata(
            source_id='bom_weather_stations',
            source_type='station_api',
            schema_version='1.0.0',
        )
    
    def validate_source(self, source) -> bool:
        """Validate weather station source."""
        return True


class BOMEnvironmentalExtractor(BaseExtractor):
    """
    Extractor for BOM air quality and environmental indicators.
    
    Extracts environmental data compatible with environmental health analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("bom_environmental", config)
        
    def extract(self, source, **kwargs) -> Iterator[DataBatch]:
        """Extract BOM environmental data."""
        logger.info("Extracting BOM environmental indicators")
        
        # Environmental data real extraction not yet implemented
        # In production, this would attempt to connect to BOM air quality and environmental APIs
        raise DataExtractionError(
            "Real data extraction not implemented for BOM Environmental extractor. "
            "Production systems must implement actual environmental data source connections.",
            source=str(source),
            source_type="environmental_api"
        )
    
    
    def get_source_metadata(self, source) -> SourceMetadata:
        """Get environmental source metadata."""
        return SourceMetadata(
            source_id='bom_environmental',
            source_type='environmental_api',
            schema_version='1.0.0',
        )
    
    def validate_source(self, source) -> bool:
        """Validate environmental source."""
        return True