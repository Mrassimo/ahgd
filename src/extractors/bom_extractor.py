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
    SourceMetadata,
)
from ..utils.logging import get_logger
from ..schemas.environmental_schema import (
    EnvironmentalIndicator,
    ClimateData,
    AirQualityData,
    WeatherStationData,
)
from ..schemas.integrated_schema import (
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
            
            # Extract data based on parameters
            if station_ids:
                yield from self._extract_station_data(station_ids, start_date, end_date, data_type)
            else:
                # Fallback to demo data for development
                yield from self._extract_demo_climate_data()
                
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
                logger.warning(f"Failed to extract data from station {station_id}: {e}")
                continue
    
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
    
    def _extract_demo_climate_data(self) -> Iterator[DataBatch]:
        """Generate demo climate data for development."""
        logger.info("Generating demo BOM climate data")
        
        demo_records = []
        stations = [
            ('066062', 'Sydney Observatory Hill', -33.8607, 151.2050),
            ('086071', 'Melbourne Regional Office', -37.8103, 144.9633),
            ('040913', 'Brisbane Aero', -27.3917, 153.1292),
        ]
        
        # Generate data for last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        for station_id, station_name, lat, lon in stations:
            current_date = start_date
            while current_date <= end_date:
                demo_record = {
                    'station_id': station_id,
                    'station_name': station_name,
                    'measurement_date': current_date.strftime('%Y-%m-%d'),
                    'latitude': lat,
                    'longitude': lon,
                    'temperature_max_celsius': 25.5,
                    'temperature_min_celsius': 15.2,
                    'rainfall_mm': 2.4,
                    'relative_humidity_9am_percent': 65.0,
                    'relative_humidity_3pm_percent': 55.0,
                    'wind_speed_kmh': 12.8,
                    'solar_exposure_mj_per_m2': 18.5,
                    'heat_stress_indicator': 'low',
                    'data_source_id': 'BOM_CLIMATE_DEMO',
                    'data_source_name': 'BOM Climate Demo Data',
                    'extraction_timestamp': datetime.now().isoformat(),
                }
                demo_records.append(demo_record)
                current_date += timedelta(days=1)
        
        yield demo_records
    
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
        yield from self._extract_demo_station_data()
    
    def _extract_demo_station_data(self) -> Iterator[DataBatch]:
        """Generate demo weather station data."""
        demo_records = []
        
        # Demo weather stations with SA2 mappings
        stations = [
            {
                'station_id': '066062',
                'station_name': 'Sydney Observatory Hill',
                'latitude': -33.8607,
                'longitude': 151.2050,
                'elevation_m': 39,
                'state': 'NSW',
                'nearest_sa2_code': '101021001',
                'nearest_sa2_name': 'Sydney - Haymarket - The Rocks',
                'distance_to_sa2_km': 1.2,
            },
            {
                'station_id': '086071',
                'station_name': 'Melbourne Regional Office',
                'latitude': -37.8103,
                'longitude': 144.9633,
                'elevation_m': 31,
                'state': 'VIC',
                'nearest_sa2_code': '201011001',
                'nearest_sa2_name': 'Melbourne - CBD',
                'distance_to_sa2_km': 0.8,
            },
        ]
        
        for station_data in stations:
            demo_record = {
                'station_id': station_data['station_id'],
                'station_name': station_data['station_name'],
                'latitude': station_data['latitude'],
                'longitude': station_data['longitude'],
                'elevation_metres': station_data['elevation_m'],
                'state': station_data['state'],
                'nearest_sa2_code': station_data['nearest_sa2_code'],
                'nearest_sa2_name': station_data['nearest_sa2_name'],
                'distance_to_sa2_km': station_data['distance_to_sa2_km'],
                'operational_status': 'active',
                'data_source_id': 'BOM_STATIONS_DEMO',
                'data_source_name': 'BOM Weather Stations Demo',
                'extraction_timestamp': datetime.now().isoformat(),
            }
            demo_records.append(demo_record)
        
        yield demo_records
    
    def get_source_metadata(self, source) -> SourceMetadata:
        """Get weather station source metadata."""
        return SourceMetadata(
            source_id='bom_weather_stations',
            source_type='demo',
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
        yield from self._extract_demo_environmental_data()
    
    def _extract_demo_environmental_data(self) -> Iterator[DataBatch]:
        """Generate demo environmental data."""
        demo_records = []
        sa2_codes = ['101021001', '101021002', '201011001']
        
        for sa2_code in sa2_codes:
            # Air quality indicators
            air_quality_record = {
                'geographic_id': sa2_code,
                'geographic_level': 'SA2',
                'indicator_type': 'air_quality',
                'measurement_date': datetime.now().strftime('%Y-%m-%d'),
                'pm25_concentration_ug_m3': 8.5,
                'pm10_concentration_ug_m3': 15.2,
                'no2_concentration_ug_m3': 25.8,
                'o3_concentration_ug_m3': 45.2,
                'air_quality_index': 42,  # Good
                'air_quality_category': 'Good',
                'data_source_id': 'BOM_AIR_QUALITY_DEMO',
                'data_source_name': 'BOM Air Quality Demo',
                'extraction_timestamp': datetime.now().isoformat(),
            }
            demo_records.append(air_quality_record)
            
            # UV index
            uv_record = {
                'geographic_id': sa2_code,
                'geographic_level': 'SA2',
                'indicator_type': 'uv_index',
                'measurement_date': datetime.now().strftime('%Y-%m-%d'),
                'uv_index_max': 8,
                'uv_category': 'Very High',
                'sun_protection_required': True,
                'data_source_id': 'BOM_UV_DEMO',
                'data_source_name': 'BOM UV Index Demo',
                'extraction_timestamp': datetime.now().isoformat(),
            }
            demo_records.append(uv_record)
        
        yield demo_records
    
    def get_source_metadata(self, source) -> SourceMetadata:
        """Get environmental source metadata."""
        return SourceMetadata(
            source_id='bom_environmental',
            source_type='demo',
            schema_version='1.0.0',
        )
    
    def validate_source(self, source) -> bool:
        """Validate environmental source."""
        return True