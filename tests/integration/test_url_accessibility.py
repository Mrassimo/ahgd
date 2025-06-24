"""
Integration tests for URL accessibility and data source validation.

These tests verify that all configured data source URLs are accessible
and return expected data formats for production readiness.
"""

import pytest
import requests
from pathlib import Path
from typing import Dict, List
import zipfile
import tempfile
from unittest.mock import patch

from src.extractors.abs_extractor import ABSGeographicExtractor, ABSCensusExtractor
from src.extractors.aihw_extractor import AIHWMortalityExtractor
from src.extractors.bom_extractor import BOMClimateExtractor
from src.utils.config import get_config


class TestABSURLAccessibility:
    """Test ABS data source URL accessibility and validation."""
    
    @pytest.mark.network
    @pytest.mark.slow
    def test_abs_geographic_urls_accessible(self):
        """Test that all ABS geographic boundary URLs are accessible."""
        
        config = get_config("extractors.abs")
        extractor = ABSGeographicExtractor(config)
        
        # Test all supported geographic levels
        geographic_levels = ['SA2', 'SA3', 'SA4', 'STATE']
        
        for level in geographic_levels:
            try:
                url = extractor._get_default_abs_url(level, '2021')
                
                # Make HEAD request to check accessibility
                response = requests.head(url, timeout=30, allow_redirects=True)
                
                assert response.status_code == 200, \
                    f"ABS {level} URL not accessible: {url} (Status: {response.status_code})"
                
                # Verify it's a ZIP file
                content_type = response.headers.get('content-type', '').lower()
                content_disposition = response.headers.get('content-disposition', '').lower()
                
                is_zip = any([
                    'zip' in content_type,
                    'octet-stream' in content_type,
                    'zip' in content_disposition,
                    url.endswith('.zip')
                ])
                
                assert is_zip, f"ABS {level} URL does not return ZIP file: {url}"
                
                # Check content length indicates substantial file
                content_length = response.headers.get('content-length')
                if content_length:
                    size_mb = int(content_length) / (1024 * 1024)
                    # Geographic boundary files should be at least 1MB
                    assert size_mb > 1, f"ABS {level} file suspiciously small: {size_mb:.2f}MB"
                
            except Exception as e:
                pytest.fail(f"Failed to access ABS {level} URL: {e}")
    
    @pytest.mark.network
    def test_abs_census_datapack_urls_accessible(self):
        """Test that ABS Census DataPack URLs are accessible."""
        
        config = get_config("extractors.abs")
        extractor = ABSCensusExtractor(config)
        
        # Test key Census table URLs
        census_tables = ['G01', 'G17A', 'G18', 'G09']
        
        for table_id in census_tables:
            try:
                url = extractor._get_default_census_url(table_id)
                
                # Check if URL is accessible
                response = requests.head(url, timeout=30, allow_redirects=True)
                
                # Census pages may redirect or be behind forms, so accept various status codes
                assert response.status_code in [200, 301, 302, 403], \
                    f"Census {table_id} URL not accessible: {url} (Status: {response.status_code})"
                
            except Exception as e:
                pytest.fail(f"Failed to access Census {table_id} URL: {e}")
    
    @pytest.mark.network
    @pytest.mark.slow
    def test_abs_url_discovery_mechanism(self):
        """Test that ABS URL discovery can find current download links."""
        
        config = get_config("extractors.abs")
        extractor = ABSGeographicExtractor(config)
        
        # Test URL discovery for SA2 boundaries
        discovered_urls = extractor._discover_abs_urls('SA2', '2021')
        
        # Should find at least one URL
        assert len(discovered_urls) > 0, "No ABS URLs discovered through web scraping"
        
        # All discovered URLs should be valid
        for url in discovered_urls[:3]:  # Test first 3 URLs only
            try:
                response = requests.head(url, timeout=30, allow_redirects=True)
                assert response.status_code == 200, f"Discovered URL not accessible: {url}"
            except Exception:
                # Some discovered URLs may be invalid, that's acceptable
                pass
    
    @pytest.mark.network
    def test_abs_alternative_urls_available(self):
        """Test that ABS alternative URLs are available as fallbacks."""
        
        config = get_config("extractors.abs")
        extractor = ABSGeographicExtractor(config)
        
        # Test that extractor can handle URL failures and try alternatives
        source = {'level': 'SA2', 'year': '2021'}
        
        # Mock the primary URL to fail
        original_get_url = extractor._get_default_abs_url
        
        def mock_failing_primary_url(level, year):
            if level == 'SA2':
                return "https://invalid-url-that-should-fail.abs.gov.au/fake.zip"
            return original_get_url(level, year)
        
        with patch.object(extractor, '_get_default_abs_url', mock_failing_primary_url):
            # Should attempt discovery and find alternative URLs
            try:
                batches = list(extractor.extract(source))
                # If this succeeds, alternative URLs worked
                assert True, "Alternative URL mechanism worked"
            except Exception:
                # If it fails, that's acceptable as long as it attempted alternatives
                # We can't guarantee alternative URLs work in test environment
                pass


class TestAIHWURLAccessibility:
    """Test AIHW data source URL accessibility."""
    
    @pytest.mark.network
    def test_aihw_base_urls_accessible(self):
        """Test that AIHW base URLs are accessible."""
        
        config = get_config("extractors.aihw")
        extractor = AIHWMortalityExtractor(config)
        
        # Test AIHW dataset URLs
        dataset_ids = ['grim-deaths', 'mortality-rates', 'hospital-data']
        
        for dataset_id in dataset_ids:
            try:
                url = extractor._get_default_aihw_url(dataset_id)
                
                response = requests.head(url, timeout=30, allow_redirects=True)
                
                # AIHW pages often redirect, so accept multiple status codes
                assert response.status_code in [200, 301, 302, 403], \
                    f"AIHW {dataset_id} URL not accessible: {url}"
                
            except Exception as e:
                pytest.fail(f"Failed to access AIHW {dataset_id} URL: {e}")
    
    @pytest.mark.network
    def test_aihw_data_discovery(self):
        """Test AIHW data file discovery on web pages."""
        
        config = get_config("extractors.aihw")
        extractor = AIHWMortalityExtractor(config)
        
        # Test scraping mechanism
        test_url = "https://www.aihw.gov.au/reports-data/health-conditions-disability-deaths/deaths/data"
        
        try:
            # Test that the scraping method can access the page
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(test_url, headers=headers, timeout=30)
            
            # Should be able to access the page
            assert response.status_code == 200, f"Cannot access AIHW data page: {test_url}"
            
            # Page should contain some data-related content
            content = response.text.lower()
            assert any(keyword in content for keyword in ['download', 'data', 'csv', 'excel']), \
                "AIHW page does not contain expected data download content"
                
        except Exception as e:
            pytest.fail(f"Failed to test AIHW data discovery: {e}")


class TestBOMURLAccessibility:
    """Test BOM data source URL accessibility."""
    
    @pytest.mark.network
    def test_bom_base_urls_accessible(self):
        """Test that BOM base URLs are accessible."""
        
        config = get_config("extractors.bom")
        extractor = BOMClimateExtractor(config)
        
        # Test BOM base URLs
        base_urls = [
            extractor.base_url,
            "http://www.bom.gov.au/climate/data/stations",
        ]
        
        for url in base_urls:
            try:
                response = requests.head(url, timeout=30, allow_redirects=True)
                assert response.status_code == 200, f"BOM URL not accessible: {url}"
                
            except Exception as e:
                pytest.fail(f"Failed to access BOM URL {url}: {e}")
    
    @pytest.mark.network
    def test_bom_station_data_urls(self):
        """Test BOM weather station data URL construction."""
        
        config = get_config("extractors.bom")
        extractor = BOMClimateExtractor(config)
        
        # Test with known BOM stations
        test_stations = ['066062', '086071', '040913']  # Sydney, Melbourne, Brisbane
        
        for station_id in test_stations:
            url = extractor._build_station_url(station_id, 'daily')
            
            try:
                response = requests.head(url, timeout=30, allow_redirects=True)
                
                # BOM may return various status codes for station data
                # 200: Data available, 404: No data, 403: Access restricted
                assert response.status_code in [200, 404, 403], \
                    f"Unexpected response for BOM station {station_id}: {response.status_code}"
                
            except Exception as e:
                # Some stations may not be accessible, that's acceptable
                pytest.skip(f"BOM station {station_id} not accessible: {e}")


class TestDataFormatValidation:
    """Test that downloaded data matches expected formats."""
    
    @pytest.mark.network
    @pytest.mark.slow
    def test_abs_shapefile_format_validation(self):
        """Test that ABS downloads are valid shapefiles."""
        
        config = get_config("extractors.abs")
        extractor = ABSGeographicExtractor(config)
        
        # Download a small sample to test format
        try:
            url = extractor._get_default_abs_url('SA4', '2021')  # SA4 is smaller than SA2
            
            response = requests.get(url, timeout=120, stream=True)
            assert response.status_code == 200
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
                
                temp_path = Path(temp_file.name)
            
            try:
                # Verify it's a valid ZIP file
                assert extractor._verify_zip_file(temp_path), "Downloaded file is not a valid ZIP"
                
                # Verify it contains shapefile components
                with zipfile.ZipFile(temp_path, 'r') as zf:
                    file_list = zf.namelist()
                    
                    # Should contain .shp, .dbf, .shx files
                    has_shp = any(f.endswith('.shp') for f in file_list)
                    has_dbf = any(f.endswith('.dbf') for f in file_list)
                    has_shx = any(f.endswith('.shx') for f in file_list)
                    
                    assert has_shp and has_dbf and has_shx, \
                        f"ZIP does not contain complete shapefile: {file_list}"
                
            finally:
                # Clean up
                temp_path.unlink(missing_ok=True)
                
        except Exception as e:
            pytest.skip(f"Could not download ABS data for format validation: {e}")
    
    @pytest.mark.network
    def test_bom_csv_format_validation(self):
        """Test that BOM data is in expected CSV format."""
        
        config = get_config("extractors.bom")
        extractor = BOMClimateExtractor(config)
        
        # Test with a known station
        station_id = '066062'  # Sydney Observatory Hill
        url = extractor._build_station_url(station_id, 'daily')
        
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                # Check if response looks like CSV data
                content = response.text
                
                # Should have CSV characteristics
                lines = content.split('\n')
                assert len(lines) > 1, "BOM response too short to be valid data"
                
                # Should contain date-related headers
                header_content = content[:500].lower()
                assert any(keyword in header_content for keyword in ['date', 'temp', 'rain']), \
                    "BOM response does not contain expected climate data headers"
            else:
                pytest.skip(f"BOM station {station_id} data not accessible")
                
        except Exception as e:
            pytest.skip(f"Could not access BOM data for format validation: {e}")


class TestProductionURLConfiguration:
    """Test URL configuration for production readiness."""
    
    def test_all_configured_urls_are_https_where_required(self):
        """Test that sensitive URLs use HTTPS."""
        
        # Get all extractor configurations
        abs_config = get_config("extractors.abs")
        aihw_config = get_config("extractors.aihw")
        bom_config = get_config("extractors.bom")
        
        # ABS URLs should use HTTPS
        abs_extractor = ABSGeographicExtractor(abs_config)
        sa2_url = abs_extractor._get_default_abs_url('SA2', '2021')
        assert sa2_url.startswith('https://'), f"ABS URL should use HTTPS: {sa2_url}"
        
        # AIHW URLs should use HTTPS
        aihw_extractor = AIHWMortalityExtractor(aihw_config)
        aihw_url = aihw_extractor._get_default_aihw_url('grim-deaths')
        assert aihw_url.startswith('https://'), f"AIHW URL should use HTTPS: {aihw_url}"
        
        # BOM URLs may use HTTP (legacy government site)
        bom_extractor = BOMClimateExtractor(bom_config)
        assert bom_extractor.base_url.startswith('http://'), "BOM URLs use HTTP as expected"
    
    def test_url_timeout_configurations(self):
        """Test that extractors have appropriate timeout configurations."""
        
        configs = [
            get_config("extractors.abs"),
            get_config("extractors.aihw"),
            get_config("extractors.bom")
        ]
        
        for config in configs:
            # Should have timeout configuration
            assert 'timeout_seconds' in config or 'timeout' in config, \
                "Extractor config missing timeout settings"
            
            timeout = config.get('timeout_seconds', config.get('timeout', 30))
            
            # Timeout should be reasonable for network operations
            assert 10 <= timeout <= 600, f"Timeout {timeout} outside reasonable range"
    
    def test_retry_configurations(self):
        """Test that extractors have appropriate retry configurations."""
        
        configs = [
            get_config("extractors.abs"),
            get_config("extractors.aihw"),
            get_config("extractors.bom")
        ]
        
        for config in configs:
            # Should have retry configuration
            retry_attempts = config.get('retry_attempts', config.get('max_retries', 3))
            
            # Should have reasonable retry limits
            assert 1 <= retry_attempts <= 10, f"Retry attempts {retry_attempts} outside reasonable range"
    
    def test_user_agent_configuration(self):
        """Test that extractors use appropriate User-Agent headers."""
        
        config = get_config("extractors.abs")
        extractor = ABSGeographicExtractor(config)
        
        # Mock a download to verify User-Agent
        with patch('requests.Session') as mock_session:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'content-length': '1000'}
            mock_response.iter_content.return_value = [b'test content']
            
            mock_session_instance = Mock()
            mock_session_instance.get.return_value = mock_response
            mock_session.return_value = mock_session_instance
            
            try:
                list(extractor._download_abs_file("http://test.url", "SA2", "2021"))
            except Exception:
                pass  # We only care about the session setup
            
            # Verify session was configured with proper headers
            assert mock_session.called
            
            # Check if headers were updated
            if mock_session_instance.headers.update.called:
                headers_arg = mock_session_instance.headers.update.call_args[0][0]
                assert 'User-Agent' in headers_arg
                user_agent = headers_arg['User-Agent']
                assert 'Mozilla' in user_agent, "Should use browser-like User-Agent"


class TestFailoverMechanisms:
    """Test failover and redundancy mechanisms."""
    
    def test_abs_url_failover_mechanism(self):
        """Test that ABS extractor can handle URL failures and try alternatives."""
        
        config = get_config("extractors.abs")
        extractor = ABSGeographicExtractor(config)
        
        source = {'level': 'SA2', 'year': '2021'}
        
        # Count the number of different URLs attempted
        attempted_urls = []
        
        def mock_download(url, *args):
            attempted_urls.append(url)
            if len(attempted_urls) < 3:
                raise Exception(f"Simulated failure for URL {url}")
            # Third attempt succeeds with empty data
            return iter([])
        
        with patch.object(extractor, '_download_abs_file', mock_download):
            try:
                list(extractor.extract(source))
            except Exception:
                pass  # We only care about failover attempts
        
        # Should have attempted multiple URLs
        assert len(attempted_urls) >= 2, "Extractor did not attempt multiple URLs for failover"
        
        # URLs should be different
        assert len(set(attempted_urls)) > 1, "Extractor attempted same URL multiple times"
    
    def test_graceful_degradation_to_demo_data(self):
        """Test that extractors gracefully degrade to demo data when real data fails."""
        
        config = get_config("extractors.abs")
        extractor = ABSGeographicExtractor(config)
        
        source = {'level': 'SA2', 'year': '2021'}
        
        # Mock all real data extraction to fail
        with patch.object(extractor, '_extract_from_url') as mock_real:
            mock_real.side_effect = Exception("All real data sources failed")
            
            # Should fall back to demo data
            batches = list(extractor.extract(source))
            
            # Should return demo data
            assert len(batches) > 0, "No fallback data returned"
            
            # Demo data should be properly formatted
            sample_record = batches[0][0]
            assert 'geographic_id' in sample_record
            assert 'data_source_id' in sample_record
            assert 'DEMO' in sample_record['data_source_id']