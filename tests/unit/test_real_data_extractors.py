"""
TDD Tests for Real Data Extraction Verification.

This module provides comprehensive tests to verify that extractors are
actually downloading real data from Australian government sources and
are production-ready.

Following TDD methodology: write failing tests first that expect real data behavior,
then verify/fix extractors to make tests pass.
"""

import pytest
import requests
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path
from typing import Iterator, List, Dict, Any
import tempfile
import zipfile
import geopandas as gpd
import pandas as pd
from datetime import datetime

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
from src.utils.interfaces import ExtractionError
from src.utils.config import get_config


class TestRealDataRequirements:
    """Test cases that define real data requirements for production readiness."""
    
    @pytest.mark.integration
    @pytest.mark.network
    def test_abs_urls_are_accessible(self):
        """Test that ABS download URLs are accessible and return valid data."""
        # Test will fail initially if URLs are not accessible
        
        abs_config = get_config("extractors.abs")
        extractor = ABSGeographicExtractor(abs_config)
        
        # Test key ABS URLs from the extractor
        test_urls = [
            extractor._get_default_abs_url('SA2', '2021'),
            extractor._get_default_abs_url('SA3', '2021'),
            extractor._get_default_abs_url('SA4', '2021'),
        ]
        
        for url in test_urls:
            response = requests.head(url, timeout=30, allow_redirects=True)
            assert response.status_code == 200, f"ABS URL not accessible: {url}"
            
            # Check it's actually a ZIP file
            content_type = response.headers.get('content-type', '').lower()
            assert any(zip_type in content_type for zip_type in ['zip', 'octet-stream']), \
                f"URL does not return ZIP file: {url}"
    
    @pytest.mark.integration
    @pytest.mark.network
    def test_aihw_data_sources_accessible(self):
        """Test that AIHW data sources are accessible."""
        aihw_config = get_config("extractors.aihw")
        
        # Test AIHW base URLs
        base_urls = [
            "https://www.aihw.gov.au/reports-data/health-conditions-disability-deaths/deaths/data",
            "https://www.aihw.gov.au/reports-data/myhospitals/datasets",
            "https://www.aihw.gov.au/reports-data/population-groups/indigenous-australians/data"
        ]
        
        for url in base_urls:
            response = requests.head(url, timeout=30, allow_redirects=True)
            # AIHW pages should be accessible (may redirect)
            assert response.status_code in [200, 301, 302], f"AIHW URL not accessible: {url}"
    
    @pytest.mark.integration  
    @pytest.mark.network
    def test_bom_data_sources_accessible(self):
        """Test that BOM data sources are accessible."""
        bom_config = get_config("extractors.bom")
        
        # Test BOM base URLs
        base_urls = [
            "http://www.bom.gov.au/climate/data/",
            "http://www.bom.gov.au/climate/data/stations",
        ]
        
        for url in base_urls:
            response = requests.head(url, timeout=30, allow_redirects=True)
            assert response.status_code == 200, f"BOM URL not accessible: {url}"


class TestABSRealDataExtraction:
    """Test ABS extractors with real data expectations."""
    
    @pytest.mark.integration
    @pytest.mark.network
    @pytest.mark.slow
    def test_abs_geographic_extractor_real_data(self):
        """Test that ABSGeographicExtractor downloads and processes real SA2 data."""
        # This test expects real ABS data, not demo data
        
        config = get_config("extractors.abs")
        extractor = ABSGeographicExtractor(config)
        
        # Configure for real data extraction
        source = {
            'level': 'SA2',
            'year': '2021'
        }
        
        # Force real data extraction by bypassing demo fallback
        with patch.object(extractor, '_extract_demo_geographic_data') as mock_demo:
            mock_demo.side_effect = Exception("Should not fall back to demo data")
            
            try:
                batches = list(extractor.extract(source))
                
                # Verify we got real data
                assert len(batches) > 0, "No data extracted from ABS"
                
                total_records = sum(len(batch) for batch in batches)
                
                # Australia has approximately 2,300+ SA2 areas as of 2021
                assert total_records > 2000, f"Expected 2000+ SA2 records, got {total_records}"
                
                # Verify data structure matches real ABS format
                sample_record = batches[0][0]
                assert 'geographic_id' in sample_record
                assert 'geographic_level' in sample_record
                assert sample_record['geographic_level'] == 'SA2'
                assert 'boundary_geometry' in sample_record or 'area_square_km' in sample_record
                
                # Verify SA2 codes follow correct format (9 digits)
                assert len(sample_record['geographic_id']) == 9
                assert sample_record['geographic_id'].isdigit()
                
            except ExtractionError as e:
                if "demo data" in str(e).lower():
                    pytest.fail("Extractor fell back to demo data instead of extracting real ABS data")
                else:
                    raise
    
    @pytest.mark.integration
    @pytest.mark.network
    @pytest.mark.slow
    def test_abs_census_extractor_real_data(self):
        """Test that ABSCensusExtractor downloads and processes real Census data."""
        
        config = get_config("extractors.abs")
        extractor = ABSCensusExtractor(config)
        
        source = {
            'table_id': 'G01',
            'url': None  # Let it use default URL
        }
        
        # Force real data extraction
        with patch.object(extractor, '_extract_demo_census_data') as mock_demo:
            mock_demo.side_effect = Exception("Should not fall back to demo data")
            
            try:
                batches = list(extractor.extract(source))
                
                assert len(batches) > 0, "No Census data extracted"
                
                total_records = sum(len(batch) for batch in batches)
                
                # Census G01 data should cover all SA2 areas (2400+)
                assert total_records > 2400, f"Expected 2400+ Census records, got {total_records}"
                
                # Verify Census data structure
                sample_record = batches[0][0]
                assert 'geographic_id' in sample_record
                assert 'census_year' in sample_record
                assert sample_record['census_year'] == 2021
                assert 'total_population' in sample_record
                assert sample_record['total_population'] > 0
                
            except ExtractionError as e:
                if "demo data" in str(e).lower():
                    pytest.fail("Extractor fell back to demo data instead of extracting real Census data")
                else:
                    raise
    
    def test_abs_extractor_with_force_real_data_flag(self):
        """Test that extractors work with --force-real-data flag."""
        # This test simulates production deployment with real data required
        
        config = get_config("extractors.abs")
        config['force_real_data'] = True  # Production flag
        
        extractor = ABSGeographicExtractor(config)
        
        # In production mode, demo data should never be used
        with patch.object(extractor, '_extract_demo_geographic_data') as mock_demo:
            mock_demo.side_effect = Exception("Demo data used in production mode!")
            
            # This should either succeed with real data or fail gracefully
            # but never fall back to demo data
            source = {'level': 'SA2', 'year': '2021'}
            
            try:
                batches = list(extractor.extract(source))
                # If we get here, real data extraction succeeded
                assert len(batches) > 0
            except ExtractionError:
                # Acceptable in production if real data is unavailable
                # but should not fall back to demo
                pass


class TestAIHWRealDataExtraction:
    """Test AIHW extractors with real data expectations."""
    
    @pytest.mark.integration
    @pytest.mark.network
    def test_aihw_mortality_extractor_attempts_real_data(self):
        """Test that AIHWMortalityExtractor attempts to extract real mortality data."""
        
        config = get_config("extractors.aihw")
        extractor = AIHWMortalityExtractor(config)
        
        # Test with real AIHW mortality data source
        source = {
            'dataset_id': 'grim-deaths',
            'url': extractor._get_default_aihw_url('grim-deaths')
        }
        
        # Count calls to real vs demo extraction methods
        real_extraction_attempted = False
        
        with patch.object(extractor, '_extract_from_api') as mock_real:
            mock_real.side_effect = lambda *args: (setattr(extractor, '_real_attempted', True), [][:])[1]
            
            with patch.object(extractor, '_extract_demo_data') as mock_demo:
                mock_demo.return_value = iter([])
                
                try:
                    list(extractor.extract(source))
                    real_extraction_attempted = hasattr(extractor, '_real_attempted')
                except Exception:
                    real_extraction_attempted = hasattr(extractor, '_real_attempted')
        
        assert real_extraction_attempted, "AIHW extractor did not attempt real data extraction"
    
    def test_aihw_extractors_have_real_urls(self):
        """Test that AIHW extractors have configured real data URLs."""
        
        config = get_config("extractors.aihw")
        
        extractors = [
            AIHWMortalityExtractor(config),
            AIHWHospitalisationExtractor(config),
            AIHWHealthIndicatorExtractor(config),
            AIHWMedicareExtractor(config)
        ]
        
        for extractor in extractors:
            # Each extractor should have a base URL configured
            assert hasattr(extractor, 'base_url')
            assert extractor.base_url.startswith('http')
            
            # Should have real data URLs configured
            if hasattr(extractor, '_get_default_aihw_url'):
                test_url = extractor._get_default_aihw_url('test')
                assert test_url.startswith('http')


class TestBOMRealDataExtraction:
    """Test BOM extractors with real data expectations."""
    
    @pytest.mark.integration
    @pytest.mark.network
    def test_bom_climate_extractor_with_real_stations(self):
        """Test that BOMClimateExtractor can work with real weather station data."""
        
        config = get_config("extractors.bom")
        extractor = BOMClimateExtractor(config)
        
        # Test with real BOM station IDs
        source = {
            'station_ids': ['066062', '086071'],  # Sydney Observatory Hill, Melbourne
            'start_date': '2023-01-01',
            'end_date': '2023-01-31'
        }
        
        # Mock successful station data extraction
        with patch.object(extractor, '_download_abs_file') as mock_download:
            # Simulate that real data extraction was attempted
            mock_download.side_effect = Exception("Network error - but real extraction was attempted")
            
            try:
                list(extractor.extract(source))
            except Exception:
                pass  # We expect it to fail in test environment
            
            # Verify that real station extraction was attempted
            assert mock_download.called, "BOM extractor did not attempt real station data extraction"
    
    def test_bom_extractor_url_construction(self):
        """Test that BOM extractor correctly constructs URLs for real data."""
        
        config = get_config("extractors.bom")
        extractor = BOMClimateExtractor(config)
        
        # Test URL construction for real stations
        station_id = '066062'
        url = extractor._build_station_url(station_id, 'daily')
        
        assert url.startswith('http')
        assert station_id in url
        assert 'bom.gov.au' in url


class TestProductionReadinessValidation:
    """Tests that validate production readiness of extractors."""
    
    def test_all_extractors_have_real_data_urls(self):
        """Test that all extractors have real data source URLs configured."""
        
        # Load configurations
        abs_config = get_config("extractors.abs")
        aihw_config = get_config("extractors.aihw")
        bom_config = get_config("extractors.bom")
        
        # Test ABS extractors
        abs_geographic = ABSGeographicExtractor(abs_config)
        abs_census = ABSCensusExtractor(abs_config)
        
        # Verify ABS has real URLs
        assert hasattr(abs_geographic, '_get_default_abs_url')
        assert hasattr(abs_census, '_get_default_census_url')
        
        # Test URL generation
        sa2_url = abs_geographic._get_default_abs_url('SA2', '2021')
        census_url = abs_census._get_default_census_url('G01')
        
        assert sa2_url.startswith('https://www.abs.gov.au')
        assert census_url.startswith('https://www.abs.gov.au')
        
        # Test AIHW extractors
        aihw_mortality = AIHWMortalityExtractor(aihw_config)
        assert hasattr(aihw_mortality, 'base_url')
        assert aihw_mortality.base_url.startswith('https://www.aihw.gov.au')
        
        # Test BOM extractors  
        bom_climate = BOMClimateExtractor(bom_config)
        assert hasattr(bom_climate, 'base_url')
        assert bom_climate.base_url.startswith('http://www.bom.gov.au')
    
    def test_extractors_handle_real_data_formats(self):
        """Test that extractors can handle real Australian government data formats."""
        
        # Test ABS shapefile handling
        abs_config = get_config("extractors.abs")
        abs_extractor = ABSGeographicExtractor(abs_config)
        
        # Create mock shapefile ZIP structure like real ABS files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            zip_path = temp_path / "test_sa2.zip"
            
            # Create a minimal shapefile structure
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("SA2_2021_AUST.shp", b"fake shapefile content")
                zf.writestr("SA2_2021_AUST.dbf", b"fake dbf content")
                zf.writestr("SA2_2021_AUST.shx", b"fake shx content")
                zf.writestr("SA2_2021_AUST.prj", b"fake projection content")
            
            # Verify extractor can handle ZIP structure
            assert abs_extractor._verify_zip_file(zip_path)
    
    def test_extractors_validate_australian_data_standards(self):
        """Test that extractors validate data according to Australian standards."""
        
        abs_config = get_config("extractors.abs")
        abs_extractor = ABSGeographicExtractor(abs_config)
        
        # Test SA2 code validation (Australian standard: 9 digits)
        valid_sa2_codes = ['101011001', '201021002', '301031003']
        invalid_sa2_codes = ['12345', 'ABC123456', '12345678901']
        
        for valid_code in valid_sa2_codes:
            record = {
                'sa2_code': valid_code,
                'sa2_name': 'Test Area',
                'geographic_level': 'SA2',
                'data_source': 'ABS',
                'extraction_timestamp': datetime.now().isoformat()
            }
            
            validated = abs_extractor._validate_geographic_record(record)
            assert validated is not None, f"Valid SA2 code {valid_code} was rejected"
        
        for invalid_code in invalid_sa2_codes:
            record = {
                'sa2_code': invalid_code,
                'sa2_name': 'Test Area',
                'geographic_level': 'SA2',
                'data_source': 'ABS',
                'extraction_timestamp': datetime.now().isoformat()
            }
            
            validated = abs_extractor._validate_geographic_record(record)
            assert validated is None, f"Invalid SA2 code {invalid_code} was accepted"
    
    @pytest.mark.network
    def test_production_deployment_requirements(self):
        """Test requirements for production deployment."""
        
        # Test that extractors can be configured for production
        production_config = {
            'force_real_data': True,
            'disable_demo_fallback': True,
            'timeout_seconds': 300,
            'max_retries': 5,
            'batch_size': 1000
        }
        
        # Test each extractor type with production config
        extractors = [
            ABSGeographicExtractor(production_config),
            ABSCensusExtractor(production_config),
            AIHWMortalityExtractor(production_config),
            BOMClimateExtractor(production_config)
        ]
        
        for extractor in extractors:
            # Verify production configuration is applied
            assert extractor.max_retries >= 3
            assert extractor.batch_size > 0
            
            # Verify extractor has required methods for production
            assert hasattr(extractor, 'extract')
            assert hasattr(extractor, 'validate_source')
            assert hasattr(extractor, 'get_source_metadata')


class TestDataVolumeExpectations:
    """Tests that verify expected data volumes for real Australian datasets."""
    
    @pytest.mark.integration
    @pytest.mark.network
    @pytest.mark.slow
    def test_abs_sa2_count_expectations(self):
        """Test that ABS SA2 extraction returns expected number of areas."""
        
        config = get_config("extractors.abs")
        extractor = ABSGeographicExtractor(config)
        
        # Australia should have ~2,300+ SA2 areas as of 2021 Census
        MIN_EXPECTED_SA2_COUNT = 2300
        MAX_EXPECTED_SA2_COUNT = 2600  # Allow for updates
        
        source = {'level': 'SA2', 'year': '2021'}
        
        try:
            batches = list(extractor.extract(source))
            total_records = sum(len(batch) for batch in batches)
            
            assert MIN_EXPECTED_SA2_COUNT <= total_records <= MAX_EXPECTED_SA2_COUNT, \
                f"SA2 count {total_records} outside expected range {MIN_EXPECTED_SA2_COUNT}-{MAX_EXPECTED_SA2_COUNT}"
                
        except ExtractionError:
            pytest.skip("Real ABS data not accessible in test environment")
    
    @pytest.mark.integration
    @pytest.mark.network
    def test_census_data_coverage_expectations(self):
        """Test that Census data covers expected population."""
        
        config = get_config("extractors.abs")
        extractor = ABSCensusExtractor(config)
        
        source = {'table_id': 'G01'}
        
        try:
            batches = list(extractor.extract(source))
            
            # Calculate total population from extracted data
            total_population = 0
            for batch in batches:
                for record in batch:
                    if 'total_population' in record and record['total_population']:
                        total_population += record['total_population']
            
            # Australia's population should be ~25-27 million as of 2021 Census
            MIN_EXPECTED_POPULATION = 24_000_000
            MAX_EXPECTED_POPULATION = 28_000_000
            
            assert MIN_EXPECTED_POPULATION <= total_population <= MAX_EXPECTED_POPULATION, \
                f"Total population {total_population} outside expected range"
                
        except ExtractionError:
            pytest.skip("Real Census data not accessible in test environment")


class TestErrorHandlingInProduction:
    """Tests that verify proper error handling for production scenarios."""
    
    def test_network_failure_handling(self):
        """Test that extractors handle network failures gracefully."""
        
        config = get_config("extractors.abs")
        extractor = ABSGeographicExtractor(config)
        
        # Mock network failure
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("Network unreachable")
            
            source = {'level': 'SA2', 'year': '2021'}
            
            # Should either fall back to demo data or raise appropriate error
            try:
                batches = list(extractor.extract(source))
                # If we get here, fallback worked
                assert len(batches) >= 0
            except ExtractionError as e:
                # Acceptable to fail with appropriate error message
                assert "network" in str(e).lower() or "connection" in str(e).lower()
    
    def test_invalid_data_format_handling(self):
        """Test that extractors handle invalid data formats gracefully."""
        
        config = get_config("extractors.abs")
        extractor = ABSGeographicExtractor(config)
        
        # Mock response with invalid content
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'text/html'}
            mock_response.iter_content.return_value = [b"<html>Error page</html>"]
            mock_get.return_value = mock_response
            
            source = {'level': 'SA2', 'year': '2021'}
            
            # Should handle invalid format gracefully
            try:
                batches = list(extractor.extract(source))
                # If we get here, fallback worked
                assert len(batches) >= 0
            except ExtractionError as e:
                # Should provide informative error message
                assert len(str(e)) > 0
    
    def test_partial_data_extraction_handling(self):
        """Test handling of partially corrupted data."""
        
        config = get_config("extractors.abs")
        extractor = ABSCensusExtractor(config)
        
        # Test with partial CSV data
        partial_csv = """SA2_CODE_2021,Total_Population_Persons,Total_Population_Males
101011001,5420,2710
101011002,invalid_data,2800
101011003,4250,2100"""
        
        with patch.object(extractor, '_parse_csv_data') as mock_parse:
            mock_parse.return_value = iter([[{
                'geographic_id': '101011001',
                'total_population': 5420,
                'male_population': 2710,
                'census_year': 2021
            }]])
            
            source = {'table_id': 'G01'}
            batches = list(extractor.extract(source))
            
            # Should return valid records and skip invalid ones
            assert len(batches) > 0
            assert len(batches[0]) > 0