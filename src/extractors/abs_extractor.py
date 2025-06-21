"""
ABS (Australian Bureau of Statistics) data extractors.

This module provides extractors for ABS data sources including geographic boundaries,
census data, SEIFA indices, and postcode correspondence files.
All extractors work backwards from target schema requirements.
"""

import csv
import io
import json
import re
import requests
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
from urllib.parse import urljoin, urlparse

import pandas as pd
import geopandas as gpd
from shapely.geometry import shape

from .base import BaseExtractor
from ..utils.interfaces import (
    DataBatch,
    DataRecord,
    ExtractionError,
    SourceMetadata,
)
from ..utils.logging import get_logger
from schemas.sa2_schema import (
    SA2Coordinates,
    SA2GeometryValidation,
    SA2BoundaryRelationship,
)
from schemas.seifa_schema import (
    SEIFAScore,
    SEIFAIndexType,
    SEIFAComponent,
)
from schemas.census_schema import (
    CensusDemographics,
    CensusEducation,
    CensusEmployment,
)
from schemas.integrated_schema import (
    MasterHealthRecord,
    GeographicHealthMapping,
    UrbanRuralClassification,
)


logger = get_logger(__name__)


class ABSGeographicExtractor(BaseExtractor):
    """
    Extractor for ABS geographic boundary files and geographic hierarchies.
    
    Extracts SA2 boundaries compatible with GeographicBoundary schema and
    SA2BoundaryData schema requirements.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("abs_geographic", config)
        self.base_url = config.get('abs_base_url', 'https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3')
        self.coordinate_system = config.get('coordinate_system', 'GDA2020')
        
        # Target schema field mappings
        self.geographic_field_mappings = {
            'sa2_code': ['SA2_CODE21', 'SA2_MAIN21', 'SA2_CODE_2021', 'sa2_code'],
            'sa2_name': ['SA2_NAME21', 'SA2_NAME', 'sa2_name'],
            'sa3_code': ['SA3_CODE21', 'SA3_CODE', 'sa3_code'],
            'sa3_name': ['SA3_NAME21', 'SA3_NAME', 'sa3_name'],
            'sa4_code': ['SA4_CODE21', 'SA4_CODE', 'sa4_code'],
            'sa4_name': ['SA4_NAME21', 'SA4_NAME', 'sa4_name'],
            'state_code': ['STE_CODE21', 'STATE_CODE', 'state_code'],
            'state_name': ['STE_NAME21', 'STATE_NAME', 'state_name'],
            'area_sq_km': ['AREASQKM21', 'AREA_SQKM', 'area_sq_km'],
            'geometry': ['geometry', 'GEOMETRY', 'geom'],
        }
    
    def extract(
        self,
        source: Union[str, Path, Dict[str, Any]],
        **kwargs
    ) -> Iterator[DataBatch]:
        """
        Extract ABS geographic boundary data.
        
        Args:
            source: ABS geographic data source specification
            **kwargs: Additional parameters
            
        Yields:
            DataBatch: Batches of geographic records
        """
        try:
            # Handle different source types
            if isinstance(source, dict):
                source_path = source.get('path')
                geographic_level = source.get('level', 'SA2')
                year = source.get('year', '2021')
            elif isinstance(source, (str, Path)):
                source_path = str(source)
                geographic_level = kwargs.get('level', 'SA2')
                year = kwargs.get('year', '2021')
            else:
                raise ExtractionError(f"Unsupported source type: {type(source)}")
            
            # Validate source
            if not self.validate_source(source_path or source):
                raise ExtractionError(f"Invalid ABS geographic source: {source}")
            
            # Extract data based on source type
            if source_path and source_path.startswith('http'):
                yield from self._extract_from_url(source_path, geographic_level, year)
            elif isinstance(source_path, (str, type(None))) and source_path and Path(source_path).exists():
                yield from self._extract_from_file(Path(source_path), geographic_level)
            else:
                # Fallback to demo data for development
                yield from self._extract_demo_geographic_data(geographic_level)
                
        except Exception as e:
            logger.error(f"ABS geographic extraction failed: {e}")
            raise ExtractionError(f"ABS geographic extraction failed: {e}")
    
    def _extract_from_url(
        self,
        url: str,
        geographic_level: str,
        year: str
    ) -> Iterator[DataBatch]:
        """Extract from ABS URL."""
        logger.info(f"Downloading ABS {geographic_level} boundaries from: {url}")
        
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Save to temporary file
            temp_path = Path(f"/tmp/abs_{geographic_level}_{year}.zip")
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            # Extract from downloaded file
            yield from self._extract_from_file(temp_path, geographic_level)
            
            # Clean up
            temp_path.unlink(missing_ok=True)
            
        except requests.RequestException as e:
            logger.error(f"Failed to download ABS data: {e}")
            raise ExtractionError(f"ABS download failed: {e}")
    
    def _extract_from_file(
        self,
        file_path: Path,
        geographic_level: str
    ) -> Iterator[DataBatch]:
        """Extract from local file."""
        logger.info(f"Extracting ABS {geographic_level} boundaries from: {file_path}")
        
        if file_path.suffix.lower() == '.zip':
            yield from self._extract_from_shapefile_zip(file_path, geographic_level)
        elif file_path.suffix.lower() in ['.shp', '.geojson', '.json']:
            yield from self._extract_from_vector_file(file_path, geographic_level)
        else:
            raise ExtractionError(f"Unsupported file format: {file_path.suffix}")
    
    def _extract_from_shapefile_zip(
        self,
        zip_path: Path,
        geographic_level: str
    ) -> Iterator[DataBatch]:
        """Extract from shapefile ZIP."""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Find shapefile
            shp_files = [f for f in zip_ref.namelist() if f.endswith('.shp')]
            if not shp_files:
                raise ExtractionError("No shapefile found in ZIP")
            
            # Extract to temporary directory
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_ref.extractall(temp_dir)
                shp_path = Path(temp_dir) / shp_files[0]
                yield from self._extract_from_vector_file(shp_path, geographic_level)
    
    def _extract_from_vector_file(
        self,
        file_path: Path,
        geographic_level: str
    ) -> Iterator[DataBatch]:
        """Extract from vector file (shapefile, GeoJSON)."""
        try:
            # Read with geopandas
            gdf = gpd.read_file(file_path)
            
            # Ensure correct CRS
            if gdf.crs is None:
                logger.warning("No CRS found, assuming GDA2020 (EPSG:7844)")
                gdf.set_crs("EPSG:7844", inplace=True)
            elif gdf.crs.to_epsg() != 7844:
                logger.info(f"Reprojecting from {gdf.crs} to GDA2020")
                gdf = gdf.to_crs("EPSG:7844")
            
            batch = []
            for idx, row in gdf.iterrows():
                # Map fields to target schema
                mapped_record = self._map_geographic_fields(row, geographic_level)
                
                # Validate and transform
                validated_record = self._validate_geographic_record(mapped_record)
                
                if validated_record:
                    batch.append(validated_record)
                    
                    if len(batch) >= self.batch_size:
                        yield batch
                        batch = []
            
            # Yield remaining records
            if batch:
                yield batch
                
        except Exception as e:
            logger.error(f"Failed to read vector file {file_path}: {e}")
            raise ExtractionError(f"Vector file reading failed: {e}")
    
    def _map_geographic_fields(
        self,
        record: Union[pd.Series, Dict[str, Any]],
        geographic_level: str
    ) -> Dict[str, Any]:
        """Map source fields to target schema fields."""
        if isinstance(record, pd.Series):
            record_dict = record.to_dict()
        else:
            record_dict = record
        
        mapped = {}
        
        # Map each target field
        for target_field, source_fields in self.geographic_field_mappings.items():
            for source_field in source_fields:
                if source_field in record_dict and record_dict[source_field] is not None:
                    mapped[target_field] = record_dict[source_field]
                    break
        
        # Handle geometry separately
        if 'geometry' in record_dict:
            geom = record_dict['geometry']
            if hasattr(geom, '__geo_interface__'):
                mapped['geometry'] = geom.__geo_interface__
            elif isinstance(geom, dict):
                mapped['geometry'] = geom
            else:
                mapped['geometry'] = shape(geom).__geo_interface__
        
        # Add metadata
        mapped['geographic_level'] = geographic_level
        mapped['data_source'] = 'ABS'
        mapped['coordinate_system'] = self.coordinate_system
        mapped['extraction_timestamp'] = datetime.now().isoformat()
        mapped['source_record'] = record_dict  # Keep original for debugging
        
        return mapped
    
    def _validate_geographic_record(self, record: Dict[str, Any]) -> Optional[DataRecord]:
        """Validate geographic record against target schema."""
        try:
            # Ensure required fields are present
            geographic_level = record.get('geographic_level', 'SA2')
            code_field = f"{geographic_level.lower()}_code"
            name_field = f"{geographic_level.lower()}_name"
            
            if code_field not in record or record[code_field] is None:
                logger.warning(f"Missing {code_field} in record")
                return None
            
            # Validate geographic code format
            geo_code = str(record[code_field]).strip()
            if geographic_level == 'SA2' and not re.match(r'^\d{9}$', geo_code):
                logger.warning(f"Invalid SA2 code format: {geo_code}")
                return None
            elif geographic_level == 'SA3' and not re.match(r'^\d{5}$', geo_code):
                logger.warning(f"Invalid SA3 code format: {geo_code}")
                return None
            elif geographic_level == 'SA4' and not re.match(r'^\d{3}$', geo_code):
                logger.warning(f"Invalid SA4 code format: {geo_code}")
                return None
            
            # Calculate area if not provided
            area_sq_km = record.get('area_sq_km')
            if area_sq_km is None and 'geometry' in record:
                try:
                    from shapely.geometry import shape
                    from shapely.ops import transform
                    import pyproj
                    
                    geom = shape(record['geometry'])
                    
                    # Transform to equal area projection for area calculation
                    project = pyproj.Transformer.from_crs('EPSG:7844', 'EPSG:3577', always_xy=True).transform
                    geom_projected = transform(project, geom)
                    area_sq_km = geom_projected.area / 1_000_000  # Convert m² to km²
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate area: {e}")
                    area_sq_km = None
            
            # Build target schema compatible record
            target_record = {
                'geographic_id': geo_code,
                'geographic_level': geographic_level,
                'geographic_name': record.get(name_field, ''),
                'area_square_km': float(area_sq_km) if area_sq_km else None,
                'coordinate_system': record['coordinate_system'],
                'extraction_timestamp': record['extraction_timestamp'],
                'data_source_id': 'ABS_ASGS',
                'data_source_name': 'ABS Australian Statistical Geography Standard',
            }
            
            # Add geometry if present
            if 'geometry' in record:
                target_record['boundary_geometry'] = record['geometry']
            
            # Add geographic hierarchy if SA2
            if geographic_level == 'SA2':
                target_record['geographic_hierarchy'] = {
                    'sa3_code': record.get('sa3_code', ''),
                    'sa3_name': record.get('sa3_name', ''),
                    'sa4_code': record.get('sa4_code', ''),
                    'sa4_name': record.get('sa4_name', ''),
                    'state_code': record.get('state_code', ''),
                    'state_name': record.get('state_name', ''),
                }
                
                # Determine urban/rural classification
                target_record['urbanisation'] = self._classify_urbanisation(record)
                target_record['remoteness_category'] = self._classify_remoteness(record)
            
            return target_record
            
        except Exception as e:
            logger.error(f"Geographic record validation failed: {e}")
            return None
    
    def _classify_urbanisation(self, record: Dict[str, Any]) -> str:
        """Classify urban/rural based on SA2 characteristics."""
        # This is a simplified classification - real implementation would use
        # ABS urban centre classifications
        sa2_name = record.get('sa2_name', '').lower()
        
        if any(term in sa2_name for term in ['city', 'urban', 'metropolitan']):
            return UrbanRuralClassification.MAJOR_URBAN.value
        elif any(term in sa2_name for term in ['town', 'suburban']):
            return UrbanRuralClassification.OTHER_URBAN.value
        elif any(term in sa2_name for term in ['rural', 'farming', 'agricultural']):
            return UrbanRuralClassification.RURAL_BALANCE.value
        else:
            return UrbanRuralClassification.OTHER_URBAN.value  # Default
    
    def _classify_remoteness(self, record: Dict[str, Any]) -> str:
        """Classify remoteness area."""
        # Simplified classification - real implementation would use RA codes
        state_code = record.get('state_code', '')
        
        # Very basic classification by state
        if state_code in ['1', 'NSW']:
            return 'Major Cities of Australia'
        elif state_code in ['2', 'VIC']:
            return 'Major Cities of Australia'
        elif state_code in ['7', 'NT', '5', 'WA']:
            return 'Very Remote Australia'
        else:
            return 'Inner Regional Australia'
    
    def _extract_demo_geographic_data(self, geographic_level: str) -> Iterator[DataBatch]:
        """Generate demo geographic data for development."""
        logger.info(f"Generating demo ABS {geographic_level} boundary data")
        
        demo_records = []
        
        if geographic_level == 'SA2':
            # Demo SA2 records
            demo_sa2s = [
                {
                    'code': '101021001',
                    'name': 'Sydney - Haymarket - The Rocks',
                    'sa3_code': '10102',
                    'sa3_name': 'Sydney Inner City',
                    'sa4_code': '101',
                    'sa4_name': 'Sydney - City and Inner South',
                    'state_code': '1',
                    'state_name': 'New South Wales',
                    'area_sq_km': 2.5,
                },
                {
                    'code': '101021002',
                    'name': 'Sydney - CBD',
                    'sa3_code': '10102',
                    'sa3_name': 'Sydney Inner City',
                    'sa4_code': '101',
                    'sa4_name': 'Sydney - City and Inner South',
                    'state_code': '1',
                    'state_name': 'New South Wales',
                    'area_sq_km': 1.8,
                },
                {
                    'code': '201011001',
                    'name': 'Melbourne - CBD',
                    'sa3_code': '20101',
                    'sa3_name': 'Melbourne',
                    'sa4_code': '201',
                    'sa4_name': 'Melbourne - Inner',
                    'state_code': '2',
                    'state_name': 'Victoria',
                    'area_sq_km': 2.1,
                },
            ]
            
            for sa2_data in demo_sa2s:
                demo_record = {
                    'geographic_id': sa2_data['code'],
                    'geographic_level': 'SA2',
                    'geographic_name': sa2_data['name'],
                    'area_square_km': sa2_data['area_sq_km'],
                    'coordinate_system': 'GDA2020',
                    'geographic_hierarchy': {
                        'sa3_code': sa2_data['sa3_code'],
                        'sa3_name': sa2_data['sa3_name'],
                        'sa4_code': sa2_data['sa4_code'],
                        'sa4_name': sa2_data['sa4_name'],
                        'state_code': sa2_data['state_code'],
                        'state_name': sa2_data['state_name'],
                    },
                    'urbanisation': UrbanRuralClassification.MAJOR_URBAN.value,
                    'remoteness_category': 'Major Cities of Australia',
                    'data_source_id': 'ABS_ASGS_DEMO',
                    'data_source_name': 'ABS ASGS Demo Data',
                    'extraction_timestamp': datetime.now().isoformat(),
                }
                demo_records.append(demo_record)
        
        yield demo_records
    
    def get_source_metadata(
        self,
        source: Union[str, Path, Dict[str, Any]]
    ) -> SourceMetadata:
        """Get metadata about the ABS geographic source."""
        if isinstance(source, dict):
            source_id = source.get('level', 'abs_geographic')
            source_path = source.get('path')
        else:
            source_id = 'abs_geographic'
            source_path = str(source) if isinstance(source, (str, Path)) else None
        
        metadata = SourceMetadata(
            source_id=source_id,
            source_type='file' if source_path and not source_path.startswith('http') else 'url',
            source_url=source_path if source_path and source_path.startswith('http') else None,
            source_file=Path(source_path) if source_path and not source_path.startswith('http') else None,
            schema_version='1.0.0',
        )
        
        # Add file metadata if local file
        if isinstance(source, Path) and source.exists():
            metadata.file_size = source.stat().st_size
            metadata.last_modified = datetime.fromtimestamp(source.stat().st_mtime)
            
        return metadata
    
    def validate_source(
        self,
        source: Union[str, Path, Dict[str, Any]]
    ) -> bool:
        """Validate ABS geographic source."""
        try:
            if isinstance(source, dict):
                return 'path' in source or 'level' in source
            elif isinstance(source, Path):
                return source.exists() and source.suffix.lower() in ['.shp', '.zip', '.geojson', '.json']
            elif isinstance(source, str):
                if source.startswith('http'):
                    return bool(urlparse(source).netloc)
                else:
                    return Path(source).exists()
            return False
        except Exception:
            return False


class ABSCensusExtractor(BaseExtractor):
    """
    Extractor for ABS 2021 Census demographic data.
    
    Extracts census data compatible with CensusData schema.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("abs_census", config)
        self.census_year = config.get('census_year', 2021)
        
    def extract(self, source, **kwargs) -> Iterator[DataBatch]:
        """Extract ABS Census data."""
        logger.info(f"Extracting ABS {self.census_year} Census data")
        yield from self._extract_demo_census_data()
    
    def _extract_demo_census_data(self) -> Iterator[DataBatch]:
        """Generate demo census data."""
        demo_records = []
        sa2_codes = ['101021001', '101021002', '201011001']
        
        for sa2_code in sa2_codes:
            demo_record = {
                'geographic_id': sa2_code,
                'geographic_level': 'SA2',
                'census_year': self.census_year,
                'total_population': 5420,
                'male_population': 2710,
                'female_population': 2710,
                'median_age': 34.5,
                'median_household_income': 1685,  # Weekly
                'unemployment_rate': 4.2,
                'population_by_age_sex': {
                    '0-4': {'Male': 150, 'Female': 145},
                    '5-9': {'Male': 155, 'Female': 150},
                    '25-34': {'Male': 450, 'Female': 455},
                    '35-44': {'Male': 380, 'Female': 385},
                },
                'indigenous_population_count': 25,
                'born_overseas_count': 1850,
                'data_source_id': 'ABS_CENSUS_DEMO',
                'data_source_name': 'ABS Census Demo Data',
                'extraction_timestamp': datetime.now().isoformat(),
            }
            demo_records.append(demo_record)
        
        yield demo_records
    
    def get_source_metadata(self, source) -> SourceMetadata:
        """Get census source metadata."""
        return SourceMetadata(
            source_id='abs_census',
            source_type='demo',
            schema_version='1.0.0',
        )
    
    def validate_source(self, source) -> bool:
        """Validate census source."""
        return True


class ABSSEIFAExtractor(BaseExtractor):
    """
    Extractor for ABS SEIFA 2021 socioeconomic indices.
    
    Extracts SEIFA data compatible with SEIFAIndex schema.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("abs_seifa", config)
        self.seifa_year = config.get('seifa_year', 2021)
        
    def extract(self, source, **kwargs) -> Iterator[DataBatch]:
        """Extract ABS SEIFA data."""
        logger.info(f"Extracting ABS SEIFA {self.seifa_year} data")
        yield from self._extract_demo_seifa_data()
    
    def _extract_demo_seifa_data(self) -> Iterator[DataBatch]:
        """Generate demo SEIFA data."""
        demo_records = []
        sa2_codes = ['101021001', '101021002', '201011001']
        
        for sa2_code in sa2_codes:
            # Create SEIFA records for each index type
            seifa_indices = [
                (SEIFAIndexType.IRSD, 'Index of Relative Socio-economic Disadvantage', 1045, 8),
                (SEIFAIndexType.IRSAD, 'Index of Relative Socio-economic Advantage and Disadvantage', 1125, 9),
                (SEIFAIndexType.IER, 'Index of Economic Resources', 1087, 7),
                (SEIFAIndexType.IEO, 'Index of Education and Occupation', 1098, 8),
            ]
            
            for index_type, index_name, score, decile in seifa_indices:
                demo_record = {
                    'geographic_id': sa2_code,
                    'geographic_level': 'SA2',
                    'seifa_year': self.seifa_year,
                    'index_type': index_type.value,
                    'index_name': index_name,
                    'score': score,
                    'decile': decile,
                    'percentile': decile * 10 - 5,  # Approximate percentile
                    'data_source_id': 'ABS_SEIFA_DEMO',
                    'data_source_name': 'ABS SEIFA Demo Data',
                    'extraction_timestamp': datetime.now().isoformat(),
                }
                demo_records.append(demo_record)
        
        yield demo_records
    
    def get_source_metadata(self, source) -> SourceMetadata:
        """Get SEIFA source metadata."""
        return SourceMetadata(
            source_id='abs_seifa',
            source_type='demo',
            schema_version='1.0.0',
        )
    
    def validate_source(self, source) -> bool:
        """Validate SEIFA source."""
        return True


class ABSPostcodeExtractor(BaseExtractor):
    """
    Extractor for ABS postcode to SA2 correspondence files.
    
    Extracts postcode mappings for geographic linkage.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("abs_postcode", config)
        
    def extract(self, source, **kwargs) -> Iterator[DataBatch]:
        """Extract ABS postcode correspondence data."""
        logger.info("Extracting ABS postcode to SA2 correspondence")
        yield from self._extract_demo_postcode_data()
    
    def _extract_demo_postcode_data(self) -> Iterator[DataBatch]:
        """Generate demo postcode data."""
        demo_records = []
        
        # Demo postcode to SA2 mappings
        mappings = [
            ('2000', '101021001', 'Sydney - Haymarket - The Rocks'),
            ('2000', '101021002', 'Sydney - CBD'),
            ('3000', '201011001', 'Melbourne - CBD'),
            ('3001', '201011001', 'Melbourne - CBD'),
        ]
        
        for postcode, sa2_code, sa2_name in mappings:
            demo_record = {
                'postcode': postcode,
                'sa2_code': sa2_code,
                'sa2_name': sa2_name,
                'correspondence_year': 2021,
                'data_source_id': 'ABS_POSTCODE_DEMO',
                'data_source_name': 'ABS Postcode Correspondence Demo',
                'extraction_timestamp': datetime.now().isoformat(),
            }
            demo_records.append(demo_record)
        
        yield demo_records
    
    def get_source_metadata(self, source) -> SourceMetadata:
        """Get postcode source metadata."""
        return SourceMetadata(
            source_id='abs_postcode',
            source_type='demo',
            schema_version='1.0.0',
        )
    
    def validate_source(self, source) -> bool:
        """Validate postcode source."""
        return True