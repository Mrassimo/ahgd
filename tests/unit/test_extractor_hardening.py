"""
Test suite for extractor hardening against mock data fallbacks.

This module tests that extractors raise explicit DataExtractionError exceptions
when real data sources fail, instead of silently falling back to demo/mock data.
This is critical for production data integrity.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from pathlib import Path

from src.extractors.abs_extractor import (
    ABSGeographicExtractor,
    ABSCensusExtractor,
    ABSSEIFAExtractor,
    ABSPostcodeExtractor,
)
from src.extractors.aihw_extractor import (
    AIHWMortalityExtractor,
    AIHWHospitalisationExtractor,
    AIHWHealthIndicatorExtractor,
    AIHWMedicareExtractor,
)
from src.extractors.bom_extractor import (
    BOMClimateExtractor,
    BOMWeatherStationExtractor,
    BOMEnvironmentalExtractor,
)
from src.utils.interfaces import DataExtractionError


class TestABSExtractorHardening:
    """Test ABS extractors for proper error handling instead of demo fallbacks."""
    
    def test_abs_geographic_extractor_raises_on_404(self):
        """Test that ABSGeographicExtractor raises DataExtractionError on 404."""
        config = {'batch_size': 100}
        extractor = ABSGeographicExtractor(config)
        
        # Mock requests.get to return 404
        with patch('requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
            mock_get.return_value = mock_response
            
            source = {'path': 'https://abs.gov.au/fake-data.zip', 'level': 'SA2'}
            
            with pytest.raises(DataExtractionError) as exc_info:
                list(extractor.extract(source))
            
            assert "ABS data extraction failed" in str(exc_info.value)
    
    def test_abs_geographic_extractor_raises_on_500(self):
        """Test that ABSGeographicExtractor raises DataExtractionError on 500."""
        config = {'batch_size': 100}
        extractor = ABSGeographicExtractor(config)
        
        # Mock requests.get to return 500
        with patch('requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = requests.HTTPError("500 Internal Server Error")
            mock_get.return_value = mock_response
            
            source = {'path': 'https://abs.gov.au/fake-data.zip', 'level': 'SA2'}
            
            with pytest.raises(DataExtractionError) as exc_info:
                list(extractor.extract(source))
            
            assert "ABS data extraction failed" in str(exc_info.value)
    
    def test_abs_geographic_extractor_raises_on_timeout(self):
        """Test that ABSGeographicExtractor raises DataExtractionError on timeout."""
        config = {'batch_size': 100}
        extractor = ABSGeographicExtractor(config)
        
        # Mock requests.get to timeout
        with patch('requests.Session.get') as mock_get:
            mock_get.side_effect = requests.Timeout("Request timed out")
            
            source = {'path': 'https://abs.gov.au/fake-data.zip', 'level': 'SA2'}
            
            with pytest.raises(DataExtractionError) as exc_info:
                list(extractor.extract(source))
            
            assert "ABS data extraction failed" in str(exc_info.value)
    
    def test_abs_census_extractor_raises_on_404(self):
        """Test that ABSCensusExtractor raises DataExtractionError on 404."""
        config = {'batch_size': 100}
        extractor = ABSCensusExtractor(config)
        
        # Mock requests.get to return 404
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
            mock_get.return_value = mock_response
            
            source = {'url': 'https://abs.gov.au/fake-census.zip', 'table_id': 'G01'}
            
            with pytest.raises(DataExtractionError) as exc_info:
                list(extractor.extract(source))
            
            assert "ABS Census data extraction failed" in str(exc_info.value)
    
    def test_abs_seifa_extractor_raises_on_no_real_data(self):
        """Test that ABSSEIFAExtractor raises DataExtractionError when no real data source provided."""
        config = {'batch_size': 100}
        extractor = ABSSEIFAExtractor(config)
        
        # Current implementation directly calls demo data - should raise error instead
        source = {'url': 'https://abs.gov.au/fake-seifa.zip'}
        
        with pytest.raises(DataExtractionError) as exc_info:
            list(extractor.extract(source))
        
        assert "real data extraction not implemented" in str(exc_info.value).lower()
    
    def test_abs_postcode_extractor_raises_on_no_real_data(self):
        """Test that ABSPostcodeExtractor raises DataExtractionError when no real data source provided."""
        config = {'batch_size': 100}
        extractor = ABSPostcodeExtractor(config)
        
        # Current implementation directly calls demo data - should raise error instead
        source = {'url': 'https://abs.gov.au/fake-postcode.zip'}
        
        with pytest.raises(DataExtractionError) as exc_info:
            list(extractor.extract(source))
        
        assert "real data extraction not implemented" in str(exc_info.value).lower()


class TestAIHWExtractorHardening:
    """Test AIHW extractors for proper error handling instead of demo fallbacks."""
    
    def test_aihw_mortality_extractor_raises_on_404(self):
        """Test that AIHWMortalityExtractor raises DataExtractionError on 404."""
        config = {'batch_size': 100}
        extractor = AIHWMortalityExtractor(config)
        
        # Mock requests.get to return 404
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
            mock_get.return_value = mock_response
            
            source = {'url': 'https://aihw.gov.au/fake-mortality.csv', 'dataset_id': 'grim-deaths'}
            
            with pytest.raises(DataExtractionError) as exc_info:
                list(extractor.extract(source))
            
            assert "AIHW mortality data extraction failed" in str(exc_info.value)
    
    def test_aihw_mortality_extractor_raises_on_timeout(self):
        """Test that AIHWMortalityExtractor raises DataExtractionError on timeout."""
        config = {'batch_size': 100}
        extractor = AIHWMortalityExtractor(config)
        
        # Mock requests.get to timeout
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.Timeout("Request timed out")
            
            source = {'url': 'https://aihw.gov.au/fake-mortality.csv', 'dataset_id': 'grim-deaths'}
            
            with pytest.raises(DataExtractionError) as exc_info:
                list(extractor.extract(source))
            
            assert "AIHW mortality data extraction failed" in str(exc_info.value)
    
    def test_aihw_hospitalisation_extractor_raises_on_no_real_data(self):
        """Test that AIHWHospitalisationExtractor raises DataExtractionError when no real data available."""
        config = {'batch_size': 100}
        extractor = AIHWHospitalisationExtractor(config)
        
        # Current implementation directly calls demo data - should raise error instead
        source = {'url': 'https://aihw.gov.au/fake-hospital.csv'}
        
        with pytest.raises(DataExtractionError) as exc_info:
            list(extractor.extract(source))
        
        assert "real data extraction not implemented" in str(exc_info.value).lower()
    
    def test_aihw_health_indicator_extractor_raises_on_no_real_data(self):
        """Test that AIHWHealthIndicatorExtractor raises DataExtractionError when no real data available."""
        config = {'batch_size': 100}
        extractor = AIHWHealthIndicatorExtractor(config)
        
        # Current implementation directly calls demo data - should raise error instead
        source = {'url': 'https://aihw.gov.au/fake-indicators.csv'}
        
        with pytest.raises(DataExtractionError) as exc_info:
            list(extractor.extract(source))
        
        assert "real data extraction not implemented" in str(exc_info.value).lower()
    
    def test_aihw_medicare_extractor_raises_on_no_real_data(self):
        """Test that AIHWMedicareExtractor raises DataExtractionError when no real data available."""
        config = {'batch_size': 100}
        extractor = AIHWMedicareExtractor(config)
        
        # Current implementation directly calls demo data - should raise error instead
        source = {'url': 'https://aihw.gov.au/fake-medicare.csv'}
        
        with pytest.raises(DataExtractionError) as exc_info:
            list(extractor.extract(source))
        
        assert "real data extraction not implemented" in str(exc_info.value).lower()


class TestBOMExtractorHardening:
    """Test BOM extractors for proper error handling instead of demo fallbacks."""
    
    def test_bom_climate_extractor_raises_on_404(self):
        """Test that BOMClimateExtractor raises DataExtractionError on 404."""
        config = {'batch_size': 100}
        extractor = BOMClimateExtractor(config)
        
        # Mock requests.get to return 404
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
            mock_get.return_value = mock_response
            
            source = {'station_ids': ['066062'], 'start_date': '2023-01-01'}
            
            with pytest.raises(DataExtractionError) as exc_info:
                list(extractor.extract(source))
            
            # The extractor should try real data and fail with error, not fallback to demo
            assert "BOM climate data extraction failed" in str(exc_info.value)
    
    def test_bom_climate_extractor_raises_on_no_stations(self):
        """Test that BOMClimateExtractor raises DataExtractionError when no stations provided and fallback disabled."""
        config = {'batch_size': 100}
        extractor = BOMClimateExtractor(config)
        
        # Current implementation falls back to demo when no station_ids - should raise error instead
        source = {}  # No station_ids
        
        with pytest.raises(DataExtractionError) as exc_info:
            list(extractor.extract(source))
        
        assert "no station data available" in str(exc_info.value).lower() or \
               "real data extraction not implemented" in str(exc_info.value).lower()
    
    def test_bom_weather_station_extractor_raises_on_no_real_data(self):
        """Test that BOMWeatherStationExtractor raises DataExtractionError when no real data available."""
        config = {'batch_size': 100}
        extractor = BOMWeatherStationExtractor(config)
        
        # Current implementation directly calls demo data - should raise error instead
        source = {'url': 'https://bom.gov.au/fake-stations.csv'}
        
        with pytest.raises(DataExtractionError) as exc_info:
            list(extractor.extract(source))
        
        assert "real data extraction not implemented" in str(exc_info.value).lower()
    
    def test_bom_environmental_extractor_raises_on_no_real_data(self):
        """Test that BOMEnvironmentalExtractor raises DataExtractionError when no real data available."""
        config = {'batch_size': 100}
        extractor = BOMEnvironmentalExtractor(config)
        
        # Current implementation directly calls demo data - should raise error instead
        source = {'url': 'https://bom.gov.au/fake-environmental.csv'}
        
        with pytest.raises(DataExtractionError) as exc_info:
            list(extractor.extract(source))
        
        assert "real data extraction not implemented" in str(exc_info.value).lower()


class TestExtractorHardeningIntegration:
    """Integration tests to verify all extractors consistently handle failures."""
    
    def test_all_extractors_reject_invalid_sources(self):
        """Test that all extractors raise DataExtractionError for clearly invalid sources."""
        config = {'batch_size': 100}
        
        # Test all major extractor classes
        extractors = [
            ABSGeographicExtractor(config),
            ABSCensusExtractor(config),
            AIHWMortalityExtractor(config),
            BOMClimateExtractor(config),
        ]
        
        # Invalid source that should cause failures
        invalid_source = {'path': 'https://invalid-domain-that-does-not-exist.com/data.zip'}
        
        for extractor in extractors:
            with pytest.raises(DataExtractionError) as exc_info:
                list(extractor.extract(invalid_source))
            
            # Verify the error message contains useful context
            error_msg = str(exc_info.value).lower()
            assert any(word in error_msg for word in ['failed', 'error', 'not found', 'timeout'])
    
    def test_extractors_provide_context_in_errors(self):
        """Test that extractors provide meaningful context in DataExtractionError messages."""
        config = {'batch_size': 100}
        extractor = ABSGeographicExtractor(config)
        
        # Mock a network failure
        with patch('requests.Session.get') as mock_get:
            mock_get.side_effect = requests.ConnectionError("Network unreachable")
            
            source = {'path': 'https://abs.gov.au/data.zip', 'level': 'SA2'}
            
            with pytest.raises(DataExtractionError) as exc_info:
                list(extractor.extract(source))
            
            error_msg = str(exc_info.value)
            # Error should include source context
            assert "ABS" in error_msg
            assert any(word in error_msg.lower() for word in ['network', 'connection', 'download', 'failed'])


class TestExtractorRealDataPaths:
    """Test that real data paths still work when sources are available."""
    
    @patch('requests.Session.get')
    def test_abs_geographic_extractor_succeeds_with_valid_response(self, mock_get):
        """Test that ABSGeographicExtractor succeeds when real data is available."""
        config = {'batch_size': 100}
        extractor = ABSGeographicExtractor(config)
        
        # Mock successful response with valid ZIP content
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'application/zip', 'content-length': '1000000'}
        mock_response.iter_content = Mock(return_value=[b'fake zip content' for _ in range(10)])
        mock_get.return_value = mock_response
        
        # Mock the ZIP file verification to succeed
        with patch.object(extractor, '_verify_zip_file', return_value=True), \
             patch.object(extractor, '_extract_from_file') as mock_extract:
            
            mock_extract.return_value = iter([])  # Empty successful extraction
            
            source = {'path': 'https://abs.gov.au/real-data.zip', 'level': 'SA2'}
            
            # Should not raise an exception
            try:
                list(extractor.extract(source))
            except DataExtractionError:
                pytest.fail("Extractor should succeed with valid response, not raise DataExtractionError")
    
    @patch('requests.get')
    def test_aihw_mortality_extractor_succeeds_with_valid_csv(self, mock_get):
        """Test that AIHWMortalityExtractor succeeds when real CSV data is available."""
        config = {'batch_size': 100}
        extractor = AIHWMortalityExtractor(config)
        
        # Mock successful CSV response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "SA2_CODE,CAUSE,DEATHS\n101021001,Heart Disease,5"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        source = {'url': 'https://aihw.gov.au/real-data.csv', 'dataset_id': 'grim-deaths'}
        
        # Should not raise an exception
        try:
            result = list(extractor.extract(source))
            # Should return some data
            assert len(result) >= 0  # Empty result is acceptable for valid processing
        except DataExtractionError:
            pytest.fail("Extractor should succeed with valid CSV response, not raise DataExtractionError")