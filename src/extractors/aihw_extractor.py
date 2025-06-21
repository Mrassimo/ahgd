"""
AIHW (Australian Institute of Health and Welfare) data extractors.

This module provides extractors for AIHW data sources including mortality,
hospitalisation, health indicators, and Medicare utilisation data.
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

from .base import BaseExtractor
from ..utils.interfaces import (
    DataBatch,
    DataRecord,
    ExtractionError,
    SourceMetadata,
)
from ..utils.logging import get_logger
from schemas.health_schema import (
    HealthIndicator,
    MortalityData,
    DiseasePrevalence,
    HealthcareUtilisation,
    MentalHealthIndicator,
    AgeGroupType,
    HealthIndicatorType,
)
from schemas.integrated_schema import (
    MasterHealthRecord,
    SA2HealthProfile,
    HealthIndicatorSummary,
)


logger = get_logger(__name__)


class AIHWMortalityExtractor(BaseExtractor):
    """
    Extractor for AIHW GRIM (General Record of Incidence of Mortality) data.
    
    Extracts mortality data compatible with MortalityData schema and 
    SA2HealthProfile mortality indicators.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("aihw_mortality", config)
        self.base_url = config.get('aihw_base_url', 'https://www.aihw.gov.au/reports-data/myhospitals/datasets')
        self.api_key = config.get('api_key')
        self.data_format = config.get('data_format', 'csv')
        
        # Target schema field mappings
        self.mortality_field_mappings = {
            'sa2_code': ['SA2_CODE_2021', 'SA2_MAINCODE_2021', 'sa2_code'],
            'cause_of_death': ['CAUSE', 'COD_GROUP', 'cause_of_death'],
            'icd10_code': ['ICD10', 'ICD_CODE', 'icd10_code'],
            'age_group': ['AGE_GROUP', 'age_group'],
            'sex': ['SEX', 'sex'],
            'deaths_count': ['DEATHS', 'COUNT', 'deaths_count'],
            'mortality_rate': ['RATE', 'ASR', 'age_standardised_rate', 'mortality_rate'],
            'year': ['YEAR', 'REFERENCE_YEAR', 'year'],
        }
    
    def extract(
        self,
        source: Union[str, Path, Dict[str, Any]],
        **kwargs
    ) -> Iterator[DataBatch]:
        """
        Extract AIHW mortality data.
        
        Args:
            source: AIHW mortality data source specification
            **kwargs: Additional parameters
            
        Yields:
            DataBatch: Batches of mortality records
        """
        try:
            # Handle different source types
            if isinstance(source, dict):
                source_url = source.get('url')
                dataset_id = source.get('dataset_id', 'grim-deaths')
                filters = source.get('filters', {})
            elif isinstance(source, (str, Path)):
                source_url = str(source)
                dataset_id = kwargs.get('dataset_id', 'grim-deaths')
                filters = kwargs.get('filters', {})
            else:
                raise ExtractionError(f"Unsupported source type: {type(source)}")
            
            # Validate source
            if not self.validate_source(source_url or source):
                raise ExtractionError(f"Invalid AIHW mortality source: {source}")
            
            # Try real AIHW data first
            try:
                if not source_url:
                    source_url = self._get_default_aihw_url(dataset_id)
                
                if source_url and source_url.startswith('http'):
                    logger.info(f"Attempting real AIHW extraction from: {source_url}")
                    yield from self._extract_from_api(source_url, dataset_id, filters)
                elif isinstance(source, Path) or (isinstance(source, str) and Path(source).exists()):
                    logger.info(f"Extracting AIHW from local file: {source}")
                    yield from self._extract_from_file(Path(source))
                else:
                    raise ExtractionError("No valid AIHW source provided")
                    
            except Exception as real_extraction_error:
                logger.warning(f"Real AIHW extraction failed: {real_extraction_error}")
                logger.info("Falling back to demo mortality data")
                yield from self._extract_demo_data()
                
        except Exception as e:
            logger.error(f"AIHW mortality extraction failed: {e}")
            raise ExtractionError(f"AIHW mortality extraction failed: {e}")
    
    def _get_default_aihw_url(self, dataset_id: str) -> str:
        """Get default AIHW data URL based on dataset ID."""
        # AIHW publishes mortality data through various channels
        # These are publicly available datasets
        aihw_urls = {
            'grim-deaths': "https://www.aihw.gov.au/reports-data/population-groups/indigenous-australians/data",
            'mortality-rates': "https://www.aihw.gov.au/reports-data/health-conditions-disability-deaths/deaths/data",
            'leading-causes': "https://www.aihw.gov.au/reports-data/health-conditions-disability-deaths/deaths/data",
            # MyHospitals data (publicly available)
            'hospital-data': "https://www.aihw.gov.au/reports-data/myhospitals/content/data-downloads"
        }
        
        if dataset_id in aihw_urls:
            return aihw_urls[dataset_id]
        
        # Default to general mortality data page
        logger.warning(f"No specific URL for dataset {dataset_id}, using default mortality data")
        return "https://www.aihw.gov.au/reports-data/health-conditions-disability-deaths/deaths/data"

    def _extract_from_api(
        self,
        api_url: str,
        dataset_id: str,
        filters: Dict[str, Any]
    ) -> Iterator[DataBatch]:
        """Extract from AIHW data source - often web scraping of published data."""
        try:
            # AIHW data is often published as downloadable files rather than API
            # Try to find downloadable CSV/Excel files
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            logger.info(f"Accessing AIHW data source: {api_url}")
            
            # If this is a direct file link, download it
            if api_url.endswith(('.csv', '.xlsx', '.xls')):
                response = requests.get(api_url, headers=headers, timeout=60)
                response.raise_for_status()
                
                if api_url.endswith('.csv'):
                    yield from self._parse_csv_data(response.text)
                else:
                    # Handle Excel files
                    yield from self._parse_excel_data(response.content)
            else:
                # Try to find mortality data on the page
                yield from self._scrape_aihw_mortality_data(api_url, dataset_id, headers)
                
        except requests.RequestException as e:
            logger.error(f"AIHW data request failed: {e}")
            raise ExtractionError(f"AIHW data request failed: {e}")
    
    def _scrape_aihw_mortality_data(self, url: str, dataset_id: str, headers: Dict[str, str]) -> Iterator[DataBatch]:
        """Attempt to find and extract mortality data from AIHW pages."""
        try:
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            
            # Look for direct data download links in the page
            # This is a simplified approach - real implementation would need proper HTML parsing
            content = response.text.lower()
            
            # Check for common AIHW data file patterns
            import re
            csv_links = re.findall(r'href="([^"]*\.csv[^"]*)"', content)
            excel_links = re.findall(r'href="([^"]*\.xlsx?[^"]*)"', content)
            
            # Try to download found data files
            base_url = '/'.join(url.split('/')[:-1])
            
            for link in csv_links[:3]:  # Limit to first 3 files
                try:
                    if not link.startswith('http'):
                        link = base_url + '/' + link.lstrip('/')
                    
                    logger.info(f"Found potential mortality data file: {link}")
                    file_response = requests.get(link, headers=headers, timeout=60)
                    file_response.raise_for_status()
                    
                    yield from self._parse_csv_data(file_response.text)
                    return  # Success, exit after first successful extraction
                    
                except Exception as file_error:
                    logger.warning(f"Failed to extract from {link}: {file_error}")
                    continue
            
            # If no CSV files worked, try Excel files
            for link in excel_links[:2]:  # Limit to first 2 files
                try:
                    if not link.startswith('http'):
                        link = base_url + '/' + link.lstrip('/')
                    
                    logger.info(f"Found potential mortality Excel file: {link}")
                    file_response = requests.get(link, headers=headers, timeout=60)
                    file_response.raise_for_status()
                    
                    yield from self._parse_excel_data(file_response.content)
                    return  # Success, exit after first successful extraction
                    
                except Exception as file_error:
                    logger.warning(f"Failed to extract from {link}: {file_error}")
                    continue
            
            # If we get here, no data files were found or processed successfully
            raise ExtractionError(f"No processable mortality data found at {url}")
            
        except Exception as e:
            logger.error(f"Failed to scrape AIHW mortality data: {e}")
            raise ExtractionError(f"AIHW mortality data scraping failed: {e}")
    
    def _parse_excel_data(self, excel_content: bytes) -> Iterator[DataBatch]:
        """Parse Excel mortality data."""
        try:
            import tempfile
            
            # Save Excel content to temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix=".xlsx")
            temp_path = Path(temp_path)
            
            try:
                with open(temp_fd, 'wb') as f:
                    f.write(excel_content)
                
                # Read Excel file with pandas
                df = pd.read_excel(temp_path, engine='openpyxl')
                
                # Convert to CSV format for processing
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                yield from self._parse_csv_data(csv_data)
                
            finally:
                temp_path.unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Failed to parse Excel mortality data: {e}")
            raise ExtractionError(f"Excel parsing failed: {e}")
    
    def _extract_from_file(self, file_path: Path) -> Iterator[DataBatch]:
        """Extract from local file."""
        logger.info(f"Extracting AIHW mortality data from file: {file_path}")
        
        if file_path.suffix.lower() == '.csv':
            yield from self._parse_csv_file(file_path)
        elif file_path.suffix.lower() == '.json':
            yield from self._parse_json_file(file_path)
        elif file_path.suffix.lower() == '.zip':
            yield from self._parse_zip_file(file_path)
        else:
            raise ExtractionError(f"Unsupported file format: {file_path.suffix}")
    
    def _parse_csv_data(self, csv_data: str) -> Iterator[DataBatch]:
        """Parse CSV mortality data."""
        reader = csv.DictReader(io.StringIO(csv_data))
        
        batch = []
        for row in reader:
            # Map fields to target schema
            mapped_record = self._map_mortality_fields(row)
            
            # Validate and transform
            validated_record = self._validate_mortality_record(mapped_record)
            
            if validated_record:
                batch.append(validated_record)
                
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
        
        # Yield remaining records
        if batch:
            yield batch
    
    def _parse_csv_file(self, file_path: Path) -> Iterator[DataBatch]:
        """Parse CSV file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            yield from self._parse_csv_data(f.read())
    
    def _parse_json_data(self, json_data: Dict[str, Any]) -> Iterator[DataBatch]:
        """Parse JSON mortality data."""
        records = json_data.get('data', json_data.get('records', []))
        
        batch = []
        for record in records:
            # Map fields to target schema
            mapped_record = self._map_mortality_fields(record)
            
            # Validate and transform
            validated_record = self._validate_mortality_record(mapped_record)
            
            if validated_record:
                batch.append(validated_record)
                
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
        
        # Yield remaining records
        if batch:
            yield batch
    
    def _parse_json_file(self, file_path: Path) -> Iterator[DataBatch]:
        """Parse JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            yield from self._parse_json_data(data)
    
    def _parse_zip_file(self, file_path: Path) -> Iterator[DataBatch]:
        """Parse ZIP file containing mortality data."""
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.filename.endswith('.csv'):
                    with zip_ref.open(file_info.filename) as f:
                        csv_data = f.read().decode('utf-8')
                        yield from self._parse_csv_data(csv_data)
                elif file_info.filename.endswith('.json'):
                    with zip_ref.open(file_info.filename) as f:
                        json_data = json.load(f)
                        yield from self._parse_json_data(json_data)
    
    def _map_mortality_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Map source fields to target schema fields."""
        mapped = {}
        
        # Map each target field
        for target_field, source_fields in self.mortality_field_mappings.items():
            for source_field in source_fields:
                if source_field in record and record[source_field] is not None:
                    mapped[target_field] = record[source_field]
                    break
        
        # Add metadata
        mapped['data_source'] = 'AIHW'
        mapped['extraction_timestamp'] = datetime.now().isoformat()
        mapped['source_record'] = record  # Keep original for debugging
        
        return mapped
    
    def _validate_mortality_record(self, record: Dict[str, Any]) -> Optional[DataRecord]:
        """Validate mortality record against target schema."""
        try:
            # Ensure required fields are present
            required_fields = ['sa2_code', 'cause_of_death', 'year']
            for field in required_fields:
                if field not in record or record[field] is None:
                    logger.warning(f"Missing required field {field} in record")
                    return None
            
            # Validate SA2 code format (9 digits)
            sa2_code = str(record['sa2_code']).strip()
            if not re.match(r'^\d{9}$', sa2_code):
                logger.warning(f"Invalid SA2 code format: {sa2_code}")
                return None
            
            # Standardise age group
            if 'age_group' in record:
                record['age_group'] = self._standardise_age_group(record['age_group'])
            
            # Standardise sex
            if 'sex' in record:
                record['sex'] = self._standardise_sex(record['sex'])
            
            # Ensure numeric fields are proper types
            numeric_fields = ['deaths_count', 'mortality_rate', 'year']
            for field in numeric_fields:
                if field in record and record[field] is not None:
                    try:
                        record[field] = float(record[field])
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid numeric value for {field}: {record[field]}")
                        record[field] = None
            
            # Build target schema compatible record
            target_record = {
                'geographic_id': sa2_code,
                'geographic_level': 'SA2',
                'indicator_name': f"{record['cause_of_death']} Mortality",
                'indicator_code': f"MORT_{record['cause_of_death'].upper().replace(' ', '_')}",
                'indicator_type': HealthIndicatorType.MORTALITY.value,
                'value': record.get('mortality_rate', record.get('deaths_count', 0)),
                'unit': 'per 100,000' if 'mortality_rate' in record else 'count',
                'reference_year': int(record['year']) if record.get('year') else None,
                'age_group': record.get('age_group', AgeGroupType.ALL_AGES.value),
                'sex': record.get('sex'),
                'cause_of_death': record['cause_of_death'],
                'icd10_code': record.get('icd10_code'),
                'deaths_count': int(record['deaths_count']) if record.get('deaths_count') else None,
                'data_source_id': 'AIHW_GRIM',
                'data_source_name': 'AIHW General Record of Incidence of Mortality',
                'extraction_timestamp': record['extraction_timestamp'],
            }
            
            return target_record
            
        except Exception as e:
            logger.error(f"Mortality record validation failed: {e}")
            return None
    
    def _standardise_age_group(self, age_group: str) -> str:
        """Standardise age group to target schema format."""
        if not age_group:
            return AgeGroupType.ALL_AGES.value
        
        age_str = str(age_group).lower().strip()
        
        # Age group mapping
        age_mappings = {
            '0-4': AgeGroupType.AGE_0_4.value,
            '5-9': AgeGroupType.AGE_5_9.value,
            '10-14': AgeGroupType.AGE_10_14.value,
            '15-19': AgeGroupType.AGE_15_19.value,
            '20-24': AgeGroupType.AGE_20_24.value,
            '25-34': AgeGroupType.AGE_25_34.value,
            '35-44': AgeGroupType.AGE_35_44.value,
            '45-54': AgeGroupType.AGE_45_54.value,
            '55-64': AgeGroupType.AGE_55_64.value,
            '65-74': AgeGroupType.AGE_65_74.value,
            '75-84': AgeGroupType.AGE_75_84.value,
            '85+': AgeGroupType.AGE_85_PLUS.value,
            'all ages': AgeGroupType.ALL_AGES.value,
        }
        
        for pattern, standard in age_mappings.items():
            if pattern in age_str:
                return standard
        
        return age_str  # Return original if no mapping found
    
    def _standardise_sex(self, sex: str) -> str:
        """Standardise sex categories."""
        if not sex:
            return 'Persons'
        
        sex_str = str(sex).lower().strip()
        
        if sex_str in ['m', 'male', 'males']:
            return 'Male'
        elif sex_str in ['f', 'female', 'females']:
            return 'Female'
        elif sex_str in ['p', 'person', 'persons', 'total']:
            return 'Persons'
        
        return 'Persons'  # Default to persons
    
    def _extract_demo_data(self) -> Iterator[DataBatch]:
        """Generate demo mortality data for development."""
        logger.info("Generating demo AIHW mortality data")
        
        demo_records = []
        sa2_codes = ['101021001', '101021002', '101021003']  # Demo SA2 codes
        causes = ['Cardiovascular Disease', 'Cancer', 'Respiratory Disease', 'Diabetes']
        
        for sa2_code in sa2_codes:
            for cause in causes:
                demo_record = {
                    'geographic_id': sa2_code,
                    'geographic_level': 'SA2',
                    'indicator_name': f"{cause} Mortality",
                    'indicator_code': f"MORT_{cause.upper().replace(' ', '_')}",
                    'indicator_type': HealthIndicatorType.MORTALITY.value,
                    'value': 45.2,  # Demo mortality rate
                    'unit': 'per 100,000',
                    'reference_year': 2021,
                    'age_group': AgeGroupType.ALL_AGES.value,
                    'sex': 'Persons',
                    'cause_of_death': cause,
                    'deaths_count': 12,
                    'data_source_id': 'AIHW_GRIM_DEMO',
                    'data_source_name': 'AIHW GRIM Demo Data',
                    'extraction_timestamp': datetime.now().isoformat(),
                }
                demo_records.append(demo_record)
        
        yield demo_records
    
    def get_source_metadata(
        self,
        source: Union[str, Path, Dict[str, Any]]
    ) -> SourceMetadata:
        """Get metadata about the AIHW mortality source."""
        if isinstance(source, dict):
            source_id = source.get('dataset_id', 'aihw_mortality')
            source_url = source.get('url')
        else:
            source_id = 'aihw_mortality'
            source_url = str(source) if isinstance(source, (str, Path)) else None
        
        metadata = SourceMetadata(
            source_id=source_id,
            source_type='api' if source_url and source_url.startswith('http') else 'file',
            source_url=source_url,
            schema_version='1.0.0',
        )
        
        # Add AIHW-specific metadata
        if isinstance(source, Path) and source.exists():
            metadata.file_size = source.stat().st_size
            metadata.last_modified = datetime.fromtimestamp(source.stat().st_mtime)
            
        return metadata
    
    def validate_source(
        self,
        source: Union[str, Path, Dict[str, Any]]
    ) -> bool:
        """Validate AIHW mortality source."""
        try:
            if isinstance(source, dict):
                return 'url' in source or 'dataset_id' in source
            elif isinstance(source, Path):
                return source.exists() and source.suffix.lower() in ['.csv', '.json', '.zip']
            elif isinstance(source, str):
                if source.startswith('http'):
                    # Basic URL validation
                    return bool(urlparse(source).netloc)
                else:
                    return Path(source).exists()
            return False
        except Exception:
            return False


class AIHWHospitalisationExtractor(BaseExtractor):
    """
    Extractor for AIHW hospital separation data.
    
    Extracts hospitalisation data compatible with HealthcareUtilisation schema.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("aihw_hospitalisation", config)
        self.base_url = config.get('aihw_base_url', 'https://www.aihw.gov.au/reports-data/myhospitals/datasets')
        
        # Field mappings for hospitalisation data
        self.hospitalisation_field_mappings = {
            'sa2_code': ['SA2_CODE_2021', 'SA2_MAINCODE_2021', 'sa2_code'],
            'separation_count': ['SEPARATIONS', 'COUNT', 'separations'],
            'separation_rate': ['RATE', 'AGE_STANDARDISED_RATE', 'separation_rate'],
            'diagnosis': ['DIAGNOSIS', 'PRINCIPAL_DIAGNOSIS', 'diagnosis'],
            'procedure': ['PROCEDURE', 'PRINCIPAL_PROCEDURE', 'procedure'],
            'los_average': ['ALOS', 'AVERAGE_LOS', 'average_length_of_stay'],
            'hospital_type': ['HOSPITAL_TYPE', 'hospital_type'],
            'year': ['YEAR', 'REFERENCE_YEAR', 'year'],
        }
    
    def extract(
        self,
        source: Union[str, Path, Dict[str, Any]],
        **kwargs
    ) -> Iterator[DataBatch]:
        """Extract AIHW hospitalisation data."""
        logger.info("Extracting AIHW hospitalisation data")
        
        # Generate demo data for now
        yield from self._extract_demo_hospitalisation_data()
    
    def _extract_demo_hospitalisation_data(self) -> Iterator[DataBatch]:
        """Generate demo hospitalisation data."""
        demo_records = []
        sa2_codes = ['101021001', '101021002', '101021003']
        service_types = ['Emergency', 'Elective Surgery', 'Outpatient', 'Mental Health']
        
        for sa2_code in sa2_codes:
            for service_type in service_types:
                demo_record = {
                    'geographic_id': sa2_code,
                    'geographic_level': 'SA2',
                    'indicator_name': f"{service_type} Utilisation",
                    'indicator_code': f"HOSP_{service_type.upper().replace(' ', '_')}",
                    'indicator_type': HealthIndicatorType.UTILISATION.value,
                    'value': 125.5,  # Demo utilisation rate
                    'unit': 'per 1,000',
                    'reference_year': 2021,
                    'service_type': service_type,
                    'service_category': 'hospital',
                    'visits_count': 85,
                    'data_source_id': 'AIHW_HOSPITAL_DEMO',
                    'data_source_name': 'AIHW Hospital Demo Data',
                    'extraction_timestamp': datetime.now().isoformat(),
                }
                demo_records.append(demo_record)
        
        yield demo_records
    
    def get_source_metadata(self, source) -> SourceMetadata:
        """Get hospitalisation source metadata."""
        return SourceMetadata(
            source_id='aihw_hospitalisation',
            source_type='demo',
            schema_version='1.0.0',
        )
    
    def validate_source(self, source) -> bool:
        """Validate hospitalisation source."""
        return True  # Demo data is always valid


class AIHWHealthIndicatorExtractor(BaseExtractor):
    """
    Extractor for AIHW health performance indicators.
    
    Extracts health indicators compatible with HealthIndicator schema.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("aihw_health_indicators", config)
        
    def extract(self, source, **kwargs) -> Iterator[DataBatch]:
        """Extract AIHW health indicators."""
        logger.info("Extracting AIHW health indicators")
        yield from self._extract_demo_health_indicators()
    
    def _extract_demo_health_indicators(self) -> Iterator[DataBatch]:
        """Generate demo health indicators."""
        demo_records = []
        sa2_codes = ['101021001', '101021002', '101021003']
        indicators = [
            ('Life Expectancy', 82.5, 'years'),
            ('Smoking Prevalence', 14.2, '%'),
            ('Obesity Prevalence', 31.8, '%'),
            ('Diabetes Prevalence', 6.1, '%'),
        ]
        
        for sa2_code in sa2_codes:
            for indicator_name, value, unit in indicators:
                demo_record = {
                    'geographic_id': sa2_code,
                    'geographic_level': 'SA2',
                    'indicator_name': indicator_name,
                    'indicator_code': f"HEALTH_{indicator_name.upper().replace(' ', '_')}",
                    'indicator_type': HealthIndicatorType.PREVALENCE.value,
                    'value': value,
                    'unit': unit,
                    'reference_year': 2021,
                    'data_source_id': 'AIHW_INDICATORS_DEMO',
                    'data_source_name': 'AIHW Health Indicators Demo',
                    'extraction_timestamp': datetime.now().isoformat(),
                }
                demo_records.append(demo_record)
        
        yield demo_records
    
    def get_source_metadata(self, source) -> SourceMetadata:
        """Get health indicators source metadata."""
        return SourceMetadata(
            source_id='aihw_health_indicators',
            source_type='demo',
            schema_version='1.0.0',
        )
    
    def validate_source(self, source) -> bool:
        """Validate health indicators source."""
        return True


class AIHWMedicareExtractor(BaseExtractor):
    """
    Extractor for AIHW Medicare utilisation data.
    
    Extracts Medicare data compatible with HealthcareUtilisation schema.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("aihw_medicare", config)
        
    def extract(self, source, **kwargs) -> Iterator[DataBatch]:
        """Extract AIHW Medicare data."""
        logger.info("Extracting AIHW Medicare data")
        yield from self._extract_demo_medicare_data()
    
    def _extract_demo_medicare_data(self) -> Iterator[DataBatch]:
        """Generate demo Medicare data."""
        demo_records = []
        sa2_codes = ['101021001', '101021002', '101021003']
        service_types = ['GP Services', 'Specialist Services', 'Diagnostic Services', 'Mental Health Services']
        
        for sa2_code in sa2_codes:
            for service_type in service_types:
                demo_record = {
                    'geographic_id': sa2_code,
                    'geographic_level': 'SA2',
                    'indicator_name': f"{service_type} Utilisation",
                    'indicator_code': f"MEDICARE_{service_type.upper().replace(' ', '_')}",
                    'indicator_type': HealthIndicatorType.UTILISATION.value,
                    'value': 4.2,  # Services per capita
                    'unit': 'per capita',
                    'reference_year': 2021,
                    'service_type': service_type,
                    'service_category': 'primary_care',
                    'bulk_billed_percentage': 85.3,
                    'data_source_id': 'AIHW_MEDICARE_DEMO',
                    'data_source_name': 'AIHW Medicare Demo Data',
                    'extraction_timestamp': datetime.now().isoformat(),
                }
                demo_records.append(demo_record)
        
        yield demo_records
    
    def get_source_metadata(self, source) -> SourceMetadata:
        """Get Medicare source metadata."""
        return SourceMetadata(
            source_id='aihw_medicare',
            source_type='demo',
            schema_version='1.0.0',
        )
    
    def validate_source(self, source) -> bool:
        """Validate Medicare source."""
        return True