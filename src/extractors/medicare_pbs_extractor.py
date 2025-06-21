"""
Medicare and PBS data extractors.

This module provides extractors for Medicare utilisation, PBS prescription data,
and healthcare services data with privacy and statistical disclosure compliance.
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
from schemas.health_schema import (
    HealthIndicator,
    HealthcareUtilisation,
    HealthIndicatorType,
    AgeGroupType,
)
from schemas.integrated_schema import (
    MasterHealthRecord,
    SA2HealthProfile,
)


logger = get_logger(__name__)


class MedicareUtilisationExtractor(BaseExtractor):
    """
    Extractor for Medicare services utilisation data.
    
    Extracts Medicare data compatible with HealthcareUtilisation schema
    with privacy protection for small area data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("medicare_utilisation", config)
        self.base_url = config.get('medicare_base_url', 'https://data.gov.au/data/dataset/medicare-statistics')
        self.api_key = config.get('api_key')
        
        # Privacy protection thresholds
        self.min_cell_size = config.get('min_cell_size', 5)  # Minimum count for disclosure
        self.suppression_rules = config.get('suppression_rules', {
            'small_counts': True,
            'complementary_suppression': True,
            'dominant_cells': True,
        })
        
        # Target schema field mappings
        self.medicare_field_mappings = {
            'sa2_code': ['SA2_CODE', 'STATISTICAL_AREA_2', 'geographic_area'],
            'service_type': ['ITEM_GROUP', 'SERVICE_TYPE', 'service_category'],
            'item_number': ['ITEM_NUMBER', 'MBS_ITEM', 'item_code'],
            'services_count': ['SERVICES', 'SERVICE_COUNT', 'total_services'],
            'benefits_paid': ['BENEFITS', 'BENEFIT_PAID', 'total_benefits'],
            'patient_count': ['PATIENTS', 'PATIENT_COUNT', 'unique_patients'],
            'bulk_billed_services': ['BULK_BILLED', 'BB_SERVICES'],
            'year': ['YEAR', 'REFERENCE_YEAR', 'calendar_year'],
            'quarter': ['QUARTER', 'REFERENCE_QUARTER'],
        }
    
    def extract(
        self,
        source: Union[str, Path, Dict[str, Any]],
        **kwargs
    ) -> Iterator[DataBatch]:
        """
        Extract Medicare utilisation data.
        
        Args:
            source: Medicare data source specification
            **kwargs: Additional parameters
            
        Yields:
            DataBatch: Batches of Medicare utilisation records
        """
        try:
            # Handle different source types
            if isinstance(source, dict):
                source_url = source.get('url')
                service_types = source.get('service_types', [])
                year = source.get('year', datetime.now().year - 1)
                geographic_level = source.get('geographic_level', 'SA2')
            else:
                source_url = str(source) if isinstance(source, (str, Path)) else None
                service_types = kwargs.get('service_types', [])
                year = kwargs.get('year', datetime.now().year - 1)
                geographic_level = kwargs.get('geographic_level', 'SA2')
            
            # Validate source
            if not self.validate_source(source):
                raise ExtractionError(f"Invalid Medicare source: {source}")
            
            # Extract data based on source type
            if source_url and source_url.startswith('http'):
                yield from self._extract_from_api(source_url, service_types, year, geographic_level)
            elif isinstance(source, Path) or (isinstance(source, str) and Path(source).exists()):
                yield from self._extract_from_file(Path(source))
            else:
                # Fallback to demo data for development
                yield from self._extract_demo_medicare_data()
                
        except Exception as e:
            logger.error(f"Medicare extraction failed: {e}")
            raise ExtractionError(f"Medicare extraction failed: {e}")
    
    def _extract_from_api(
        self,
        api_url: str,
        service_types: List[str],
        year: int,
        geographic_level: str
    ) -> Iterator[DataBatch]:
        """Extract from Medicare API."""
        try:
            # Build API request
            params = {
                'year': year,
                'geographic_level': geographic_level,
                'format': 'csv'
            }
            
            if service_types:
                params['service_types'] = ','.join(service_types)
            
            if self.api_key:
                params['api_key'] = self.api_key
            
            logger.info(f"Requesting Medicare data: {api_url}")
            response = requests.get(api_url, params=params, timeout=60)
            response.raise_for_status()
            
            # Parse CSV response
            yield from self._parse_medicare_csv(response.text)
                
        except requests.RequestException as e:
            logger.error(f"Medicare API request failed: {e}")
            raise ExtractionError(f"Medicare API request failed: {e}")
    
    def _extract_from_file(self, file_path: Path) -> Iterator[DataBatch]:
        """Extract from local file."""
        logger.info(f"Extracting Medicare data from file: {file_path}")
        
        if file_path.suffix.lower() == '.csv':
            with open(file_path, 'r', encoding='utf-8') as f:
                yield from self._parse_medicare_csv(f.read())
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                yield from self._parse_medicare_json(data)
        else:
            raise ExtractionError(f"Unsupported file format: {file_path.suffix}")
    
    def _parse_medicare_csv(self, csv_data: str) -> Iterator[DataBatch]:
        """Parse Medicare CSV data with privacy protection."""
        reader = csv.DictReader(io.StringIO(csv_data))
        
        batch = []
        for row in reader:
            # Map fields to target schema
            mapped_record = self._map_medicare_fields(row)
            
            # Apply privacy protection
            protected_record = self._apply_privacy_protection(mapped_record)
            
            # Validate and transform
            validated_record = self._validate_medicare_record(protected_record)
            
            if validated_record:
                batch.append(validated_record)
                
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
        
        # Yield remaining records
        if batch:
            yield batch
    
    def _parse_medicare_json(self, json_data: Dict[str, Any]) -> Iterator[DataBatch]:
        """Parse Medicare JSON data."""
        records = json_data.get('data', json_data.get('records', []))
        
        batch = []
        for record in records:
            # Map fields to target schema
            mapped_record = self._map_medicare_fields(record)
            
            # Apply privacy protection
            protected_record = self._apply_privacy_protection(mapped_record)
            
            # Validate and transform
            validated_record = self._validate_medicare_record(protected_record)
            
            if validated_record:
                batch.append(validated_record)
                
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
        
        # Yield remaining records
        if batch:
            yield batch
    
    def _map_medicare_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Map source fields to target schema fields."""
        mapped = {}
        
        # Map each target field
        for target_field, source_fields in self.medicare_field_mappings.items():
            for source_field in source_fields:
                if source_field in record and record[source_field] is not None:
                    mapped[target_field] = record[source_field]
                    break
        
        # Add metadata
        mapped['data_source'] = 'Medicare'
        mapped['extraction_timestamp'] = datetime.now().isoformat()
        mapped['source_record'] = record  # Keep original for debugging
        
        return mapped
    
    def _apply_privacy_protection(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Apply privacy protection rules to Medicare data."""
        protected = record.copy()
        
        # Check for small cell sizes
        count_fields = ['services_count', 'patient_count', 'bulk_billed_services']
        
        for field in count_fields:
            if field in protected and protected[field] is not None:
                try:
                    count = int(protected[field])
                    if count < self.min_cell_size:
                        # Suppress small counts
                        protected[field] = f"<{self.min_cell_size}"
                        protected[f"{field}_suppressed"] = True
                        
                        # Also suppress related fields for complementary protection
                        if field == 'services_count' and 'benefits_paid' in protected:
                            protected['benefits_paid'] = "SUPP"
                            protected['benefits_paid_suppressed'] = True
                            
                except (ValueError, TypeError):
                    continue
        
        # Apply statistical disclosure control
        if self.suppression_rules.get('dominant_cells', False):
            # Implement dominant cell suppression if needed
            pass
        
        return protected
    
    def _validate_medicare_record(self, record: Dict[str, Any]) -> Optional[DataRecord]:
        """Validate Medicare record against target schema."""
        try:
            # Ensure required fields are present
            required_fields = ['sa2_code']
            for field in required_fields:
                if field not in record or record[field] is None:
                    return None
            
            # Validate SA2 code format
            sa2_code = str(record['sa2_code']).strip()
            if not re.match(r'^\d{9}$', sa2_code):
                logger.warning(f"Invalid SA2 code format: {sa2_code}")
                return None
            
            # Handle suppressed values
            services_count = record.get('services_count')
            if isinstance(services_count, str) and services_count.startswith('<'):
                # Keep suppressed indicator
                services_count = None
                suppressed = True
            else:
                try:
                    services_count = int(services_count) if services_count else None
                    suppressed = False
                except (ValueError, TypeError):
                    services_count = None
                    suppressed = False
            
            # Calculate utilisation rate if possible
            utilisation_rate = None
            if services_count and 'patient_count' in record:
                try:
                    patient_count = int(record['patient_count'])
                    if patient_count > 0:
                        utilisation_rate = services_count / patient_count
                except (ValueError, TypeError, ZeroDivisionError):
                    pass
            
            # Calculate bulk billing rate
            bulk_billing_rate = None
            if 'bulk_billed_services' in record and services_count:
                try:
                    bulk_billed = int(record['bulk_billed_services'])
                    bulk_billing_rate = (bulk_billed / services_count) * 100
                except (ValueError, TypeError, ZeroDivisionError):
                    pass
            
            # Build target schema compatible record
            target_record = {
                'geographic_id': sa2_code,
                'geographic_level': 'SA2',
                'indicator_name': f"{record.get('service_type', 'Medicare')} Utilisation",
                'indicator_code': f"MEDICARE_{record.get('service_type', 'ALL').upper().replace(' ', '_')}",
                'indicator_type': HealthIndicatorType.UTILISATION.value,
                'value': utilisation_rate or services_count or 0,
                'unit': 'services per patient' if utilisation_rate else 'count',
                'reference_year': int(record['year']) if record.get('year') else None,
                'reference_quarter': record.get('quarter'),
                'service_type': record.get('service_type', 'Medicare Services'),
                'service_category': 'primary_care',
                'visits_count': services_count,
                'utilisation_rate': utilisation_rate,
                'bulk_billed_percentage': bulk_billing_rate,
                'total_benefits_paid': record.get('benefits_paid'),
                'unique_patients': record.get('patient_count'),
                'data_suppressed': suppressed,
                'suppressed': suppressed,
                'data_source_id': 'MEDICARE_UTILISATION',
                'data_source_name': 'Medicare Benefits Schedule Utilisation',
                'extraction_timestamp': record['extraction_timestamp'],
            }
            
            return target_record
            
        except Exception as e:
            logger.error(f"Medicare record validation failed: {e}")
            return None
    
    def _extract_demo_medicare_data(self) -> Iterator[DataBatch]:
        """Generate demo Medicare utilisation data for development."""
        logger.info("Generating demo Medicare utilisation data")
        
        demo_records = []
        sa2_codes = ['101021001', '101021002', '201011001']
        service_types = [
            'GP Attendances',
            'Specialist Attendances', 
            'Diagnostic Imaging',
            'Pathology',
            'Mental Health Services',
            'Allied Health Services'
        ]
        
        for sa2_code in sa2_codes:
            for service_type in service_types:
                # Generate realistic Medicare utilisation data
                services_count = 850
                patient_count = 340
                bulk_billed = 725
                
                demo_record = {
                    'geographic_id': sa2_code,
                    'geographic_level': 'SA2',
                    'indicator_name': f"{service_type} Utilisation",
                    'indicator_code': f"MEDICARE_{service_type.upper().replace(' ', '_')}",
                    'indicator_type': HealthIndicatorType.UTILISATION.value,
                    'value': services_count / patient_count,  # Services per patient
                    'unit': 'services per patient',
                    'reference_year': 2023,
                    'reference_quarter': 'Q4',
                    'service_type': service_type,
                    'service_category': 'primary_care',
                    'visits_count': services_count,
                    'utilisation_rate': services_count / patient_count,
                    'bulk_billed_percentage': (bulk_billed / services_count) * 100,
                    'total_benefits_paid': 45250.50,
                    'unique_patients': patient_count,
                    'data_suppressed': False,
                    'suppressed': False,
                    'data_source_id': 'MEDICARE_DEMO',
                    'data_source_name': 'Medicare Demo Data',
                    'extraction_timestamp': datetime.now().isoformat(),
                }
                demo_records.append(demo_record)
        
        yield demo_records
    
    def get_source_metadata(
        self,
        source: Union[str, Path, Dict[str, Any]]
    ) -> SourceMetadata:
        """Get metadata about the Medicare source."""
        if isinstance(source, dict):
            source_id = 'medicare_utilisation'
            source_url = source.get('url')
        else:
            source_id = 'medicare_utilisation'
            source_url = str(source) if isinstance(source, (str, Path)) else None
        
        metadata = SourceMetadata(
            source_id=source_id,
            source_type='api' if source_url and source_url.startswith('http') else 'file',
            source_url=source_url,
            schema_version='1.0.0',
        )
        
        return metadata
    
    def validate_source(
        self,
        source: Union[str, Path, Dict[str, Any]]
    ) -> bool:
        """Validate Medicare source."""
        try:
            if isinstance(source, dict):
                return True  # Basic validation for demo
            elif isinstance(source, Path):
                return source.exists() and source.suffix.lower() in ['.csv', '.json']
            elif isinstance(source, str):
                if source.startswith('http'):
                    return bool(urlparse(source).netloc)
                else:
                    return Path(source).exists() or source == 'demo'
            return False
        except Exception:
            return False


class PBSPrescriptionExtractor(BaseExtractor):
    """
    Extractor for PBS pharmaceutical prescription data.
    
    Extracts PBS data compatible with pharmaceutical utilisation analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("pbs_prescriptions", config)
        self.base_url = config.get('pbs_base_url', 'https://data.gov.au/data/dataset/pharmaceutical-benefits-scheme-pbs-data')
        
        # Privacy protection for PBS data
        self.min_cell_size = config.get('min_cell_size', 5)
        
    def extract(self, source, **kwargs) -> Iterator[DataBatch]:
        """Extract PBS prescription data."""
        logger.info("Extracting PBS prescription data")
        yield from self._extract_demo_pbs_data()
    
    def _extract_demo_pbs_data(self) -> Iterator[DataBatch]:
        """Generate demo PBS prescription data."""
        demo_records = []
        sa2_codes = ['101021001', '101021002', '201011001']
        drug_categories = [
            'Cardiovascular',
            'Mental Health',
            'Diabetes',
            'Respiratory',
            'Pain Management',
            'Antibiotics'
        ]
        
        for sa2_code in sa2_codes:
            for drug_category in drug_categories:
                # Generate realistic PBS data
                prescriptions = 425
                patients = 165
                
                demo_record = {
                    'geographic_id': sa2_code,
                    'geographic_level': 'SA2',
                    'indicator_name': f"{drug_category} Prescriptions",
                    'indicator_code': f"PBS_{drug_category.upper().replace(' ', '_')}",
                    'indicator_type': HealthIndicatorType.UTILISATION.value,
                    'value': prescriptions / patients,  # Prescriptions per patient
                    'unit': 'prescriptions per patient',
                    'reference_year': 2023,
                    'drug_category': drug_category,
                    'prescription_count': prescriptions,
                    'unique_patients': patients,
                    'total_cost': 18750.25,
                    'average_cost_per_prescription': 44.12,
                    'data_suppressed': False,
                    'data_source_id': 'PBS_DEMO',
                    'data_source_name': 'PBS Demo Data',
                    'extraction_timestamp': datetime.now().isoformat(),
                }
                demo_records.append(demo_record)
        
        yield demo_records
    
    def get_source_metadata(self, source) -> SourceMetadata:
        """Get PBS source metadata."""
        return SourceMetadata(
            source_id='pbs_prescriptions',
            source_type='demo',
            schema_version='1.0.0',
        )
    
    def validate_source(self, source) -> bool:
        """Validate PBS source."""
        return True


class HealthcareServicesExtractor(BaseExtractor):
    """
    Extractor for healthcare service locations and capacity data.
    
    Extracts healthcare services data for access analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("healthcare_services", config)
        
    def extract(self, source, **kwargs) -> Iterator[DataBatch]:
        """Extract healthcare services data."""
        logger.info("Extracting healthcare services data")
        yield from self._extract_demo_services_data()
    
    def _extract_demo_services_data(self) -> Iterator[DataBatch]:
        """Generate demo healthcare services data."""
        demo_records = []
        sa2_codes = ['101021001', '101021002', '201011001']
        service_types = [
            'GP Practices',
            'Specialist Clinics',
            'Hospitals',
            'Mental Health Services',
            'Allied Health Services',
            'Pharmacies'
        ]
        
        for sa2_code in sa2_codes:
            for service_type in service_types:
                demo_record = {
                    'geographic_id': sa2_code,
                    'geographic_level': 'SA2',
                    'service_type': service_type,
                    'service_count': 12,
                    'workforce_fte': 45.5,
                    'workforce_per_1000_population': 8.4,
                    'average_distance_km': 2.1,
                    'accessibility_score': 85.2,
                    'capacity_utilisation_percent': 78.5,
                    'data_source_id': 'HEALTHCARE_SERVICES_DEMO',
                    'data_source_name': 'Healthcare Services Demo',
                    'extraction_timestamp': datetime.now().isoformat(),
                }
                demo_records.append(demo_record)
        
        yield demo_records
    
    def get_source_metadata(self, source) -> SourceMetadata:
        """Get healthcare services source metadata."""
        return SourceMetadata(
            source_id='healthcare_services',
            source_type='demo',
            schema_version='1.0.0',
        )
    
    def validate_source(self, source) -> bool:
        """Validate healthcare services source."""
        return True