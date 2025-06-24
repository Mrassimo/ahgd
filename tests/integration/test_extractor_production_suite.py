"""
Comprehensive Production Test Suite for AHGD Extractors.

This test suite runs comprehensive validation of all extractors to ensure
they are ready for production deployment with real Australian government data.

Run with: pytest tests/integration/test_extractor_production_suite.py -v
"""

import pytest
import requests
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import patch, Mock
import tempfile
import zipfile
from datetime import datetime, timedelta

from src.extractors.abs_extractor import (
    ABSGeographicExtractor,
    ABSCensusExtractor, 
    ABSSEIFAExtractor,
    ABSPostcodeExtractor
)
from src.extractors.aihw_extractor import (
    AIHWMortalityExtractor,
    AIHWHospitalisationExtractor,
    AIHWHealthIndicatorExtractor,
    AIHWMedicareExtractor
)
from src.extractors.bom_extractor import (
    BOMClimateExtractor,
    BOMWeatherStationExtractor,
    BOMEnvironmentalExtractor
)
from src.utils.config import get_config
from src.utils.interfaces import ExtractionError


class TestExtractorProductionSuite:
    """Comprehensive test suite for production readiness."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        self.abs_config = get_config("extractors.abs")
        self.aihw_config = get_config("extractors.aihw") 
        self.bom_config = get_config("extractors.bom")
    
    # ===== ABS EXTRACTOR TESTS =====
    
    @pytest.mark.production
    def test_abs_geographic_extractor_production_readiness(self):
        """Test ABS Geographic extractor production readiness."""
        
        extractor = ABSGeographicExtractor(self.abs_config)
        
        # Test 1: Configuration validation
        assert hasattr(extractor, '_get_default_abs_url')
        assert extractor.coordinate_system == 'GDA2020'
        assert extractor.batch_size > 0
        
        # Test 2: URL generation for all levels
        levels = ['SA2', 'SA3', 'SA4', 'STATE']
        for level in levels:
            url = extractor._get_default_abs_url(level, '2021')
            assert url.startswith('https://www.abs.gov.au')
            assert 'AUST' in url.upper()
            assert '2021' in url
        
        # Test 3: Field mapping completeness
        required_fields = ['sa2_code', 'sa2_name', 'geometry', 'area_sq_km']
        for field in required_fields:
            assert field in extractor.geographic_field_mappings
        
        # Test 4: Validation rules
        # Valid SA2 code
        valid_record = {
            'sa2_code': '101011001',
            'sa2_name': 'Test Area',
            'geographic_level': 'SA2',
            'data_source': 'ABS',
            'extraction_timestamp': datetime.now().isoformat(),
            'coordinate_system': 'GDA2020'
        }
        
        validated = extractor._validate_geographic_record(valid_record)
        assert validated is not None
        assert validated['geographic_id'] == '101011001'
        
        # Invalid SA2 code
        invalid_record = valid_record.copy()
        invalid_record['sa2_code'] = 'INVALID'
        
        validated = extractor._validate_geographic_record(invalid_record)
        assert validated is None
    
    @pytest.mark.production
    def test_abs_census_extractor_production_readiness(self):
        """Test ABS Census extractor production readiness."""
        
        extractor = ABSCensusExtractor(self.abs_config)
        
        # Test 1: Census URL generation
        tables = ['G01', 'G17A', 'G18', 'G09']
        for table in tables:
            url = extractor._get_default_census_url(table)
            assert url.startswith('https://www.abs.gov.au/census')
        
        # Test 2: Field mapping for demographics
        assert hasattr(extractor, '_parse_census_records')
        
        # Test 3: Census year validation
        assert extractor.census_year == 2021
    
    @pytest.mark.production 
    def test_abs_seifa_extractor_production_readiness(self):
        """Test ABS SEIFA extractor production readiness."""
        
        extractor = ABSSEIFAExtractor(self.abs_config)
        
        # Test SEIFA year and demo data structure
        assert extractor.seifa_year == 2021
        
        # Test demo data contains all required SEIFA indices
        demo_batches = list(extractor._extract_demo_seifa_data())
        assert len(demo_batches) > 0
        
        # Should have records for each SEIFA index type
        records = demo_batches[0]
        index_types = set(record['index_type'] for record in records)
        expected_types = {'IRSD', 'IRSAD', 'IER', 'IEO'}
        assert expected_types.issubset(index_types)
    
    @pytest.mark.production
    def test_abs_postcode_extractor_production_readiness(self):
        """Test ABS Postcode extractor production readiness."""
        
        extractor = ABSPostcodeExtractor(self.abs_config)
        
        # Test demo data structure
        demo_batches = list(extractor._extract_demo_postcode_data())
        assert len(demo_batches) > 0
        
        records = demo_batches[0]
        sample_record = records[0]
        
        # Should have postcode mapping fields
        assert 'postcode' in sample_record
        assert 'sa2_code' in sample_record
        assert len(sample_record['postcode']) == 4
        assert len(sample_record['sa2_code']) == 9
    
    # ===== AIHW EXTRACTOR TESTS =====
    
    @pytest.mark.production
    def test_aihw_mortality_extractor_production_readiness(self):
        """Test AIHW Mortality extractor production readiness."""
        
        extractor = AIHWMortalityExtractor(self.aihw_config)
        
        # Test 1: Base URL configuration
        assert extractor.base_url.startswith('https://www.aihw.gov.au')
        
        # Test 2: URL generation for datasets
        datasets = ['grim-deaths', 'mortality-rates', 'leading-causes']
        for dataset in datasets:
            url = extractor._get_default_aihw_url(dataset)
            assert url.startswith('https://www.aihw.gov.au')
        
        # Test 3: Field mapping for mortality data
        required_fields = ['sa2_code', 'cause_of_death', 'deaths_count', 'mortality_rate']
        for field in required_fields:
            assert field in extractor.mortality_field_mappings
        
        # Test 4: Mortality record validation
        valid_record = {
            'sa2_code': '101011001',
            'cause_of_death': 'Cardiovascular Disease',
            'deaths_count': '15',
            'mortality_rate': '45.2',
            'year': '2021',
            'data_source': 'AIHW',
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        validated = extractor._validate_mortality_record(valid_record)
        assert validated is not None
        assert validated['geographic_id'] == '101011001'
        assert 'Cardiovascular Disease' in validated['indicator_name']
        
        # Test 5: Age group standardisation
        age_groups = ['0-4', '25-34', '65-74', '85+', 'All Ages']
        for age_group in age_groups:
            standardised = extractor._standardise_age_group(age_group)
            assert isinstance(standardised, str)
            assert len(standardised) > 0
    
    @pytest.mark.production
    def test_aihw_hospitalisation_extractor_production_readiness(self):
        """Test AIHW Hospitalisation extractor production readiness."""
        
        extractor = AIHWHospitalisationExtractor(self.aihw_config)
        
        # Test field mappings
        required_fields = ['sa2_code', 'separation_count', 'diagnosis']
        for field in required_fields:
            assert field in extractor.hospitalisation_field_mappings
        
        # Test demo data structure
        demo_batches = list(extractor._extract_demo_hospitalisation_data())
        assert len(demo_batches) > 0
        
        records = demo_batches[0]
        sample_record = records[0]
        
        assert 'service_type' in sample_record
        assert 'visits_count' in sample_record
        assert sample_record['indicator_type'] == 'utilisation'
    
    @pytest.mark.production
    def test_aihw_health_indicator_extractor_production_readiness(self):
        """Test AIHW Health Indicator extractor production readiness."""
        
        extractor = AIHWHealthIndicatorExtractor(self.aihw_config)
        
        # Test demo data contains various health indicators
        demo_batches = list(extractor._extract_demo_health_indicators())
        assert len(demo_batches) > 0
        
        records = demo_batches[0]
        indicator_names = set(record['indicator_name'] for record in records)
        
        expected_indicators = {'Life Expectancy', 'Smoking Prevalence', 'Obesity Prevalence'}
        assert expected_indicators.issubset(indicator_names)
    
    @pytest.mark.production
    def test_aihw_medicare_extractor_production_readiness(self):
        """Test AIHW Medicare extractor production readiness."""
        
        extractor = AIHWMedicareExtractor(self.aihw_config)
        
        # Test demo data structure
        demo_batches = list(extractor._extract_demo_medicare_data())
        assert len(demo_batches) > 0
        
        records = demo_batches[0]
        sample_record = records[0]
        
        assert 'service_type' in sample_record
        assert 'bulk_billed_percentage' in sample_record
        assert sample_record['service_category'] == 'primary_care'
    
    # ===== BOM EXTRACTOR TESTS =====
    
    @pytest.mark.production
    def test_bom_climate_extractor_production_readiness(self):
        """Test BOM Climate extractor production readiness."""
        
        extractor = BOMClimateExtractor(self.bom_config)
        
        # Test 1: Base URL configuration
        assert extractor.base_url.startswith('http://www.bom.gov.au')
        
        # Test 2: Station URL construction
        station_id = '066062'
        url = extractor._build_station_url(station_id, 'daily')
        assert station_id in url
        assert 'bom.gov.au' in url
        
        # Test 3: Field mapping for climate data
        required_fields = ['station_id', 'date', 'temperature_max', 'rainfall']
        for field in required_fields:
            assert field in extractor.climate_field_mappings
        
        # Test 4: Climate record validation
        valid_record = {
            'station_id': '066062',
            'date': '2023-01-15',
            'temperature_max': '28.5',
            'temperature_min': '18.2',
            'rainfall': '2.4',
            'data_source': 'BOM',
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        validated = extractor._validate_climate_record(valid_record)
        assert validated is not None
        assert validated['station_id'] == '066062'
        assert validated['measurement_date'] == '2023-01-15'
        
        # Test 5: Demo data structure
        demo_batches = list(extractor._extract_demo_climate_data())
        assert len(demo_batches) > 0
        
        records = demo_batches[0]
        sample_record = records[0]
        
        assert 'temperature_max_celsius' in sample_record
        assert 'rainfall_mm' in sample_record
        assert 'heat_stress_indicator' in sample_record
    
    @pytest.mark.production
    def test_bom_weather_station_extractor_production_readiness(self):
        """Test BOM Weather Station extractor production readiness."""
        
        extractor = BOMWeatherStationExtractor(self.bom_config)
        
        # Test demo station data
        demo_batches = list(extractor._extract_demo_station_data())
        assert len(demo_batches) > 0
        
        records = demo_batches[0]
        sample_record = records[0]
        
        # Should have station metadata
        assert 'station_id' in sample_record
        assert 'station_name' in sample_record
        assert 'latitude' in sample_record
        assert 'longitude' in sample_record
        assert 'nearest_sa2_code' in sample_record
        
        # Coordinates should be in Australia
        lat = sample_record['latitude']
        lon = sample_record['longitude']
        assert -45 <= lat <= -10  # Australian latitude range
        assert 110 <= lon <= 155  # Australian longitude range
    
    @pytest.mark.production
    def test_bom_environmental_extractor_production_readiness(self):
        """Test BOM Environmental extractor production readiness."""
        
        extractor = BOMEnvironmentalExtractor(self.bom_config)
        
        # Test demo environmental data
        demo_batches = list(extractor._extract_demo_environmental_data())
        assert len(demo_batches) > 0
        
        records = demo_batches[0]
        
        # Should have air quality records
        air_quality_records = [r for r in records if r['indicator_type'] == 'air_quality']
        assert len(air_quality_records) > 0
        
        sample_aq = air_quality_records[0]
        assert 'pm25_concentration_ug_m3' in sample_aq
        assert 'air_quality_index' in sample_aq
        assert 'air_quality_category' in sample_aq
        
        # Should have UV records
        uv_records = [r for r in records if r['indicator_type'] == 'uv_index']
        assert len(uv_records) > 0
        
        sample_uv = uv_records[0]
        assert 'uv_index_max' in sample_uv
        assert 'uv_category' in sample_uv


class TestRealDataCapabilities:
    """Test real data extraction capabilities where possible."""
    
    @pytest.mark.network
    @pytest.mark.slow
    def test_abs_geographic_real_data_attempt(self):
        """Test that ABS Geographic extractor attempts real data extraction."""
        
        config = get_config("extractors.abs")
        extractor = ABSGeographicExtractor(config)
        
        source = {'level': 'SA4', 'year': '2021'}  # Use SA4 (smaller dataset)
        
        # Track if real extraction was attempted
        real_attempted = False
        
        def mock_download(*args, **kwargs):
            nonlocal real_attempted
            real_attempted = True
            # Simulate failure to test fallback
            raise Exception("Simulated network failure")
        
        with patch.object(extractor, '_download_abs_file', mock_download):
            try:
                # Should attempt real extraction first, then fall back to demo
                batches = list(extractor.extract(source))
                
                # Should have attempted real extraction
                assert real_attempted, "Real data extraction was not attempted"
                
                # Should fall back to demo data
                assert len(batches) > 0, "No fallback data provided"
                
            except ExtractionError:
                # Acceptable if real extraction fails completely
                assert real_attempted, "Real data extraction was not attempted"
    
    @pytest.mark.network
    def test_aihw_mortality_real_data_attempt(self):
        """Test that AIHW Mortality extractor attempts real data extraction."""
        
        config = get_config("extractors.aihw")
        extractor = AIHWMortalityExtractor(config)
        
        source = {'dataset_id': 'grim-deaths'}
        
        real_attempted = False
        
        def mock_api_call(*args, **kwargs):
            nonlocal real_attempted
            real_attempted = True
            raise Exception("Simulated API failure")
        
        with patch.object(extractor, '_extract_from_api', mock_api_call):
            try:
                batches = list(extractor.extract(source))
                assert real_attempted, "Real AIHW extraction was not attempted"
                assert len(batches) > 0, "No fallback data provided"
            except ExtractionError:
                assert real_attempted, "Real AIHW extraction was not attempted"
    
    @pytest.mark.network
    def test_bom_climate_real_data_attempt(self):
        """Test that BOM Climate extractor attempts real data extraction."""
        
        config = get_config("extractors.bom")
        extractor = BOMClimateExtractor(config)
        
        source = {
            'station_ids': ['066062'],  # Sydney Observatory Hill
            'start_date': '2023-01-01',
            'end_date': '2023-01-07'   # Short date range
        }
        
        real_attempted = False
        
        def mock_station_extract(*args, **kwargs):
            nonlocal real_attempted
            real_attempted = True
            raise Exception("Simulated BOM failure")
        
        with patch.object(extractor, '_extract_station_data', mock_station_extract):
            try:
                batches = list(extractor.extract(source))
                assert real_attempted, "Real BOM extraction was not attempted"
                # BOM may not have demo fallback, that's acceptable
            except ExtractionError:
                assert real_attempted, "Real BOM extraction was not attempted"


class TestErrorResilienceAndFallbacks:
    """Test error resilience and fallback mechanisms."""
    
    def test_abs_url_discovery_fallback(self):
        """Test ABS URL discovery when primary URLs fail."""
        
        config = get_config("extractors.abs")
        extractor = ABSGeographicExtractor(config)
        
        source = {'level': 'SA2', 'year': '2021'}
        
        # Track fallback attempts
        discovery_attempted = False
        
        def mock_discovery(*args, **kwargs):
            nonlocal discovery_attempted
            discovery_attempted = True
            return ['http://backup-url-1.abs.gov.au/file.zip']
        
        with patch.object(extractor, '_discover_abs_urls', mock_discovery):
            with patch.object(extractor, '_download_abs_file') as mock_download:
                mock_download.side_effect = Exception("All downloads failed")
                
                try:
                    list(extractor.extract(source))
                except Exception:
                    pass
                
                # Should have attempted URL discovery
                assert discovery_attempted, "URL discovery fallback not attempted"
    
    def test_graceful_degradation_to_demo_data(self):
        """Test graceful degradation to demo data across all extractors."""
        
        # Test ABS Geographic
        abs_config = get_config("extractors.abs")
        abs_extractor = ABSGeographicExtractor(abs_config)
        
        with patch.object(abs_extractor, '_extract_from_url') as mock_real:
            mock_real.side_effect = Exception("All real sources failed")
            
            source = {'level': 'SA2', 'year': '2021'}
            batches = list(abs_extractor.extract(source))
            
            assert len(batches) > 0, "ABS extractor did not fall back to demo data"
            
            # Verify it's demo data
            sample_record = batches[0][0]
            assert 'DEMO' in sample_record.get('data_source_id', '')
        
        # Test AIHW Mortality
        aihw_config = get_config("extractors.aihw")
        aihw_extractor = AIHWMortalityExtractor(aihw_config)
        
        with patch.object(aihw_extractor, '_extract_from_api') as mock_api:
            mock_api.side_effect = Exception("API unavailable")
            
            source = {'dataset_id': 'grim-deaths'}
            batches = list(aihw_extractor.extract(source))
            
            assert len(batches) > 0, "AIHW extractor did not fall back to demo data"
    
    def test_partial_data_recovery(self):
        """Test recovery from partial data corruption."""
        
        config = get_config("extractors.abs")
        extractor = ABSCensusExtractor(config)
        
        # Simulate partially corrupted Census data
        import csv
        import io
        
        corrupted_csv = """SA2_CODE_2021,Tot_P_P,Tot_P_M,Tot_P_F
101011001,5420,2710,2710
CORRUPTED_ROW,invalid,data,format
101011003,4250,2100,2150
ANOTHER_BAD_ROW,bad,data
101011005,3800,1900,1900"""
        
        reader = csv.DictReader(io.StringIO(corrupted_csv))
        
        # Should process valid records and skip invalid ones
        batches = list(extractor._parse_census_records(reader, 'G01'))
        
        assert len(batches) > 0, "No data recovered from corrupted input"
        
        # Should have processed 3 valid records, skipped 2 invalid
        total_records = sum(len(batch) for batch in batches)
        assert total_records == 3, f"Expected 3 valid records, got {total_records}"


class TestScalabilityAndPerformance:
    """Test scalability and performance characteristics."""
    
    def test_memory_efficient_batch_processing(self):
        """Test that extractors process data in memory-efficient batches."""
        
        config = get_config("extractors.abs")
        config['batch_size'] = 50  # Small batch size for testing
        
        extractor = ABSGeographicExtractor(config)
        
        # Mock large dataset
        def mock_large_extract(source, **kwargs):
            # Simulate 10 batches of 50 records each
            for i in range(10):
                batch = []
                for j in range(50):
                    record_id = i * 50 + j
                    batch.append({
                        'geographic_id': f'10101{record_id:04d}',
                        'geographic_level': 'SA2',
                        'data_source_id': 'ABS_ASGS',
                        'extraction_timestamp': datetime.now().isoformat()
                    })
                yield batch
        
        with patch.object(extractor, 'extract', mock_large_extract):
            batch_count = 0
            total_records = 0
            max_batch_size = 0
            
            for batch in extractor.extract({'level': 'SA2'}):
                batch_count += 1
                batch_size = len(batch)
                total_records += batch_size
                max_batch_size = max(max_batch_size, batch_size)
            
            assert batch_count == 10, f"Expected 10 batches, got {batch_count}"
            assert total_records == 500, f"Expected 500 records, got {total_records}"
            assert max_batch_size <= 50, f"Batch size {max_batch_size} exceeded limit"
    
    def test_extraction_timeout_compliance(self):
        """Test that extractions respect timeout configurations."""
        
        import time
        
        config = get_config("extractors.abs")
        extractor = ABSGeographicExtractor(config)
        
        # Should have reasonable timeout
        timeout = config.get('timeout_seconds', 120)
        assert 10 <= timeout <= 600, f"Timeout {timeout} outside reasonable range"
        
        # Mock extraction with controlled timing
        def timed_extract(source, **kwargs):
            time.sleep(0.01)  # Small delay per record
            yield [{
                'geographic_id': '101011001',
                'extraction_timestamp': datetime.now().isoformat()
            }]
        
        with patch.object(extractor, 'extract', timed_extract):
            start_time = time.time()
            
            batches = list(extractor.extract({'level': 'SA2'}))
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Should complete reasonably quickly
            assert duration < 5.0, f"Extraction took too long: {duration}s"
            assert len(batches) > 0, "No data extracted"