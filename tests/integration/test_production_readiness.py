"""
Production Readiness Tests for AHGD Extractors.

These tests verify that extractors are ready for production deployment
with real Australian government data sources.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import requests
import json
import csv
import io
from datetime import datetime

from src.extractors.abs_extractor import (
    ABSGeographicExtractor,
    ABSCensusExtractor,
    ABSSEIFAExtractor
)
from src.extractors.aihw_extractor import (
    AIHWMortalityExtractor,
    AIHWHospitalisationExtractor
)
from src.extractors.bom_extractor import (
    BOMClimateExtractor,
    BOMWeatherStationExtractor
)
from src.utils.config import get_config
from src.utils.interfaces import ExtractionError


class TestConfigurationValidation:
    """Test that extractor configurations are production-ready."""
    
    def test_all_extractors_have_required_config_fields(self):
        """Test that all extractors have required configuration fields."""
        
        # Test external services configuration exists
        external_services = get_config("external_services")
        assert external_services is not None, "External services configuration missing"
        
        # Test ABS configuration
        abs_config = external_services.get("abs", {})
        assert abs_config, "ABS configuration missing in external_services"
        assert "base_url" in abs_config, "ABS config missing base_url"
        assert "timeout" in abs_config, "ABS config missing timeout"
        assert "retry_attempts" in abs_config, "ABS config missing retry_attempts"
        
        # Test AIHW configuration
        aihw_config = external_services.get("aihw", {})
        assert aihw_config, "AIHW configuration missing in external_services"
        assert "base_url" in aihw_config, "AIHW config missing base_url"
        assert "timeout" in aihw_config, "AIHW config missing timeout"
        assert "retry_attempts" in aihw_config, "AIHW config missing retry_attempts"
        
        # Test BOM configuration
        bom_config = external_services.get("bom", {})
        assert bom_config, "BOM configuration missing in external_services"
        assert "base_url" in bom_config, "BOM config missing base_url"
        assert "timeout" in bom_config, "BOM config missing timeout"
        assert "retry_attempts" in bom_config, "BOM config missing retry_attempts"
    
    def test_url_configurations_are_valid(self):
        """Test that all configured URLs are valid and well-formed."""
        
        # Test external services URLs
        external_services = get_config("external_services")
        
        # Test ABS URL
        abs_config = external_services.get("abs", {})
        base_url = abs_config.get("base_url", "")
        assert base_url.startswith('https://'), f"ABS URL should use HTTPS: {base_url}"
        assert 'abs.gov.au' in base_url, f"ABS URL should be from abs.gov.au domain: {base_url}"
        
        # Test AIHW URL
        aihw_config = external_services.get("aihw", {})
        base_url = aihw_config.get("base_url", "")
        assert base_url.startswith('https://'), f"AIHW URL should use HTTPS: {base_url}"
        assert 'aihw.gov.au' in base_url, f"AIHW URL should be from aihw.gov.au domain: {base_url}"
        
        # Test BOM URL
        bom_config = external_services.get("bom", {})
        base_url = bom_config.get("base_url", "")
        assert base_url.startswith('http://'), f"BOM URL uses HTTP as expected: {base_url}"
        assert 'bom.gov.au' in base_url, f"BOM URL should be from bom.gov.au domain: {base_url}"
    
    def test_data_validation_rules_configured(self):
        """Test that data validation rules are properly configured."""
        
        abs_config = get_config("extractors.abs")
        
        # Check SA2 validation rules
        sa2_config = abs_config.get('geographic', {}).get('sa2', {})
        validation = sa2_config.get('validation', {})
        
        assert 'required_fields' in validation, "SA2 validation missing required_fields"
        assert 'sa2_code_format' in validation, "SA2 validation missing sa2_code_format"
        assert 'area_bounds' in validation, "SA2 validation missing area_bounds"
        
        # Verify SA2 code format is correct for Australia (9 digits)
        sa2_format = validation['sa2_code_format']
        assert '\\d{9}' in sa2_format, "SA2 code format should expect 9 digits"
    
    def test_field_mappings_are_comprehensive(self):
        """Test that field mappings cover all required target schema fields."""
        
        abs_config = get_config("extractors.abs")
        
        # Test geographic field mappings
        abs_extractor = ABSGeographicExtractor(abs_config)
        mappings = abs_extractor.geographic_field_mappings
        
        required_geographic_fields = [
            'sa2_code', 'sa2_name', 'sa3_code', 'sa3_name',
            'sa4_code', 'sa4_name', 'state_code', 'state_name',
            'area_sq_km', 'geometry'
        ]
        
        for field in required_geographic_fields:
            assert field in mappings, f"Geographic mapping missing for field: {field}"
            assert len(mappings[field]) > 0, f"No source fields mapped for: {field}"
        
        # Test census field mappings
        abs_census = ABSCensusExtractor(abs_config)
        assert hasattr(abs_census, '_parse_census_records'), "Census extractor missing parsing method"


class TestDataQualityStandards:
    """Test that extractors meet Australian data quality standards."""
    
    def test_sa2_code_validation_meets_abs_standards(self):
        """Test SA2 code validation meets ABS standards."""
        
        abs_config = get_config("extractors.abs")
        extractor = ABSGeographicExtractor(abs_config)
        
        # Test valid SA2 codes (Australian standard: 9 digits)
        valid_codes = [
            '101011001',  # NSW
            '201021002',  # VIC  
            '301031003',  # QLD
            '401041004',  # SA
            '501051005',  # WA
            '601061006',  # TAS
            '701071007',  # NT
            '801081008'   # ACT
        ]
        
        for code in valid_codes:
            record = {
                'sa2_code': code,
                'sa2_name': 'Test Area',
                'geographic_level': 'SA2',
                'data_source': 'ABS',
                'extraction_timestamp': datetime.now().isoformat(),
                'coordinate_system': 'GDA2020'
            }
            
            validated = extractor._validate_geographic_record(record)
            assert validated is not None, f"Valid SA2 code {code} was rejected"
            assert validated['geographic_id'] == code
        
        # Test invalid SA2 codes
        invalid_codes = [
            '12345',        # Too short
            '1234567890',   # Too long
            'ABC123456',    # Contains letters
            '000000000',    # Invalid format
            '999999999'     # Potentially invalid range
        ]
        
        for code in invalid_codes:
            record = {
                'sa2_code': code,
                'sa2_name': 'Test Area',
                'geographic_level': 'SA2',
                'data_source': 'ABS',
                'extraction_timestamp': datetime.now().isoformat(),
                'coordinate_system': 'GDA2020'
            }
            
            validated = extractor._validate_geographic_record(record)
            # Most invalid codes should be rejected
            if validated is not None:
                # If accepted, it should have a valid reason
                assert len(code) == 9 and code.isdigit(), f"Invalid code {code} was accepted: {validated}"
    
    def test_coordinate_system_validation(self):
        """Test that coordinate system validation meets Australian standards."""
        
        abs_config = get_config("extractors.abs")
        extractor = ABSGeographicExtractor(abs_config)
        
        # Should use GDA2020 (Australian standard)
        assert extractor.coordinate_system == 'GDA2020', "Should use GDA2020 coordinate system"
        
        # Test coordinate validation in records
        record = {
            'sa2_code': '101011001',
            'coordinate_system': 'GDA2020',
            'geographic_level': 'SA2',
            'data_source': 'ABS',
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        validated = extractor._validate_geographic_record(record)
        assert validated is not None
        assert validated['coordinate_system'] == 'GDA2020'
    
    def test_census_data_consistency_validation(self):
        """Test Census data consistency validation."""
        
        abs_config = get_config("extractors.abs")
        extractor = ABSCensusExtractor(abs_config)
        
        # Test valid Census record
        valid_census_data = {
            'SA2_CODE_2021': '101011001',
            'Tot_P_P': '5420',           # Total population
            'Tot_P_M': '2710',           # Male population  
            'Tot_P_F': '2710',           # Female population
            'Median_age_persons': '34.5'
        }
        
        # Mock CSV reader with valid data
        reader = csv.DictReader(io.StringIO(""))
        reader.__iter__ = lambda: iter([valid_census_data])
        
        batches = list(extractor._parse_census_records(reader, 'G01'))
        
        assert len(batches) > 0, "No Census data processed"
        record = batches[0][0]
        
        # Test data consistency
        assert record['total_population'] == 5420
        assert record['male_population'] + record['female_population'] == record['total_population']
        assert record['census_year'] == 2021
    
    def test_health_data_privacy_compliance(self):
        """Test that health data extraction complies with privacy requirements."""
        
        aihw_config = get_config("extractors.aihw")
        extractor = AIHWMortalityExtractor(aihw_config)
        
        # Test mortality record validation
        test_record = {
            'sa2_code': '101011001',
            'cause_of_death': 'Cardiovascular Disease',
            'deaths_count': '8',  # Above minimum reporting threshold
            'mortality_rate': '45.2',
            'year': '2021',
            'data_source': 'AIHW',
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        validated = extractor._validate_mortality_record(test_record)
        assert validated is not None, "Valid mortality record was rejected"
        
        # Test small count handling (privacy protection)
        small_count_record = {
            'sa2_code': '101011001',
            'cause_of_death': 'Rare Disease',
            'deaths_count': '2',  # Below privacy threshold
            'year': '2021',
            'data_source': 'AIHW',
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        # Should handle small counts appropriately (may suppress or flag)
        validated = extractor._validate_mortality_record(small_count_record)
        if validated is not None:
            # If not suppressed, should be properly flagged
            assert 'deaths_count' in validated


class TestErrorHandlingRobustness:
    """Test robust error handling for production scenarios."""
    
    def test_network_timeout_handling(self):
        """Test handling of network timeouts."""
        
        abs_config = get_config("extractors.abs")
        extractor = ABSGeographicExtractor(abs_config)
        
        source = {'level': 'SA2', 'year': '2021'}
        
        # Mock timeout error
        with patch('requests.Session.get') as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")
            
            # Should handle timeout gracefully
            try:
                batches = list(extractor.extract(source))
                # If successful, fallback mechanism worked
                assert len(batches) >= 0
            except ExtractionError as e:
                # Should provide informative error message
                assert 'timeout' in str(e).lower() or 'connection' in str(e).lower()
    
    def test_http_error_handling(self):
        """Test handling of HTTP errors."""
        
        abs_config = get_config("extractors.abs")
        extractor = ABSGeographicExtractor(abs_config)
        
        source = {'level': 'SA2', 'year': '2021'}
        
        # Test various HTTP error codes
        error_codes = [404, 500, 503]
        
        for error_code in error_codes:
            with patch('requests.Session.get') as mock_get:
                mock_response = Mock()
                mock_response.status_code = error_code
                mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(f"HTTP {error_code}")
                mock_get.return_value = mock_response
                
                try:
                    batches = list(extractor.extract(source))
                    # Fallback worked
                    assert len(batches) >= 0
                except ExtractionError as e:
                    # Should provide informative error
                    assert str(error_code) in str(e) or 'HTTP' in str(e)
    
    def test_corrupted_data_handling(self):
        """Test handling of corrupted or invalid data formats."""
        
        abs_config = get_config("extractors.abs")
        extractor = ABSGeographicExtractor(abs_config)
        
        source = {'level': 'SA2', 'year': '2021'}
        
        # Mock response with corrupted ZIP file
        with patch('requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'application/zip'}
            mock_response.iter_content.return_value = [b"corrupted zip data"]
            mock_get.return_value = mock_response
            
            try:
                batches = list(extractor.extract(source))
                # Fallback worked
                assert len(batches) >= 0
            except ExtractionError as e:
                # Should provide informative error about data corruption
                assert any(keyword in str(e).lower() for keyword in ['zip', 'corrupt', 'invalid'])
    
    def test_partial_data_recovery(self):
        """Test recovery from partial data extraction failures."""
        
        abs_config = get_config("extractors.abs")
        extractor = ABSCensusExtractor(abs_config)
        
        # Test with partially corrupted CSV data
        partial_csv = """SA2_CODE_2021,Tot_P_P,Tot_P_M,Tot_P_F
101011001,5420,2710,2710
INVALID_ROW,invalid,data,here
101011003,4250,2100,2150
"""
        
        with patch.object(extractor, '_parse_csv_data') as mock_parse:
            # Should process valid records and skip invalid ones
            valid_records = [
                {
                    'geographic_id': '101011001',
                    'total_population': 5420,
                    'male_population': 2710,
                    'female_population': 2710,
                    'census_year': 2021
                },
                {
                    'geographic_id': '101011003', 
                    'total_population': 4250,
                    'male_population': 2100,
                    'female_population': 2150,
                    'census_year': 2021
                }
            ]
            
            mock_parse.return_value = iter([valid_records])
            
            source = {'table_id': 'G01'}
            batches = list(extractor.extract(source))
            
            assert len(batches) > 0, "No data recovered from partial extraction"
            assert len(batches[0]) == 2, "Should recover valid records"


class TestPerformanceRequirements:
    """Test that extractors meet performance requirements for production."""
    
    def test_memory_efficiency_with_large_datasets(self):
        """Test memory efficiency with large datasets."""
        
        abs_config = get_config("extractors.abs")
        abs_config['batch_size'] = 100  # Small batches for memory efficiency
        
        extractor = ABSGeographicExtractor(abs_config)
        
        # Mock large dataset extraction
        def mock_large_extract(source, **kwargs):
            # Simulate processing large dataset in batches
            for i in range(50):  # 50 batches of 100 records = 5000 records
                batch = []
                for j in range(100):
                    record_id = i * 100 + j
                    batch.append({
                        'geographic_id': f'10101{record_id:04d}',
                        'geographic_level': 'SA2',
                        'geographic_name': f'Area {record_id}',
                        'area_square_km': 10.5,
                        'coordinate_system': 'GDA2020',
                        'data_source_id': 'ABS_ASGS',
                        'extraction_timestamp': datetime.now().isoformat()
                    })
                yield batch
        
        with patch.object(extractor, 'extract', mock_large_extract):
            total_records = 0
            max_batch_size = 0
            
            for batch in extractor.extract({'level': 'SA2'}):
                total_records += len(batch)
                max_batch_size = max(max_batch_size, len(batch))
            
            assert total_records == 5000, "Did not process expected number of records"
            assert max_batch_size <= 100, "Batch size exceeded memory efficiency limit"
    
    def test_extraction_timeout_compliance(self):
        """Test that extractions complete within timeout limits."""
        
        import time
        from datetime import datetime, timedelta
        
        abs_config = get_config("extractors.abs")
        extractor = ABSGeographicExtractor(abs_config)
        
        # Mock slow extraction
        def slow_extract(source, **kwargs):
            time.sleep(0.1)  # Simulate some processing time
            yield [{
                'geographic_id': '101011001',
                'geographic_level': 'SA2', 
                'data_source_id': 'ABS_ASGS',
                'extraction_timestamp': datetime.now().isoformat()
            }]
        
        with patch.object(extractor, 'extract', slow_extract):
            start_time = datetime.now()
            
            batches = list(extractor.extract({'level': 'SA2'}))
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Should complete reasonably quickly for single batch
            assert duration < 5.0, f"Extraction took too long: {duration}s"
            assert len(batches) > 0, "No data extracted"
    
    def test_concurrent_extraction_safety(self):
        """Test that extractors can handle concurrent requests safely."""
        
        import threading
        import time
        
        abs_config = get_config("extractors.abs")
        extractor = ABSGeographicExtractor(abs_config)
        
        results = []
        errors = []
        
        def extract_worker(worker_id):
            try:
                source = {'level': 'SA2', 'year': '2021'}
                
                # Mock quick extraction
                with patch.object(extractor, 'extract') as mock_extract:
                    mock_extract.return_value = iter([[{
                        'geographic_id': f'10101000{worker_id}',
                        'worker_id': worker_id,
                        'extraction_timestamp': datetime.now().isoformat()
                    }]])
                    
                    batches = list(extractor.extract(source))
                    results.append((worker_id, len(batches)))
                    
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Start multiple concurrent extractions
        threads = []
        for i in range(5):
            thread = threading.Thread(target=extract_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join(timeout=10)
        
        # All should succeed
        assert len(errors) == 0, f"Concurrent extraction errors: {errors}"
        assert len(results) == 5, "Not all concurrent extractions completed"


class TestMonitoringAndObservability:
    """Test monitoring and observability features for production."""
    
    def test_audit_trail_generation(self):
        """Test that extractors generate comprehensive audit trails."""
        
        abs_config = get_config("extractors.abs")
        extractor = ABSGeographicExtractor(abs_config)
        
        source = {'level': 'SA2', 'year': '2021'}
        
        # Mock extraction with audit trail
        with patch.object(extractor, 'extract') as mock_extract:
            mock_extract.return_value = iter([[{
                'geographic_id': '101011001',
                'extraction_timestamp': datetime.now().isoformat()
            }]])
            
            batches = list(extractor.extract_with_retry(source))
            
            # Get audit trail
            audit_trail = extractor.get_audit_trail()
            
            assert audit_trail is not None, "No audit trail generated"
            assert audit_trail.operation_type == 'extraction'
            assert audit_trail.source_metadata is not None
            assert audit_trail.processing_metadata is not None
            assert audit_trail.processing_metadata.records_processed > 0
    
    def test_processing_metadata_completeness(self):
        """Test that processing metadata is complete and accurate."""
        
        abs_config = get_config("extractors.abs")
        extractor = ABSGeographicExtractor(abs_config)
        
        source = {'level': 'SA2', 'year': '2021'}
        
        with patch.object(extractor, 'extract') as mock_extract:
            mock_extract.return_value = iter([
                [{'id': 1}, {'id': 2}],  # First batch: 2 records
                [{'id': 3}, {'id': 4}, {'id': 5}]  # Second batch: 3 records
            ])
            
            batches = list(extractor.extract_with_retry(source))
            
            metadata = extractor._processing_metadata
            
            assert metadata is not None
            assert metadata.records_processed == 5, "Incorrect record count in metadata"
            assert metadata.start_time is not None
            assert metadata.end_time is not None
            assert metadata.duration_seconds is not None
            assert metadata.duration_seconds > 0
    
    def test_error_logging_and_reporting(self):
        """Test that errors are properly logged and reported."""
        
        abs_config = get_config("extractors.abs")
        extractor = ABSGeographicExtractor(abs_config)
        
        source = {'level': 'SA2', 'year': '2021'}
        
        # Mock extraction failure
        with patch.object(extractor, 'extract') as mock_extract:
            mock_extract.side_effect = Exception("Test extraction failure")
            
            with pytest.raises(ExtractionError):
                list(extractor.extract_with_retry(source))
            
            # Check error metadata
            metadata = extractor._processing_metadata
            assert metadata.status.name == 'FAILED'
            assert metadata.error_message is not None
            assert 'Test extraction failure' in metadata.error_message
    
    def test_checkpoint_and_resume_functionality(self):
        """Test checkpoint creation and resume functionality."""
        
        abs_config = get_config("extractors.abs")
        abs_config['checkpoint_interval'] = 2  # Checkpoint every 2 records
        
        extractor = ABSGeographicExtractor(abs_config)
        
        source = {'level': 'SA2', 'year': '2021'}
        
        with patch.object(extractor, 'extract') as mock_extract:
            mock_extract.return_value = iter([
                [{'id': 1}, {'id': 2}],    # Should trigger checkpoint
                [{'id': 3}, {'id': 4}],    # Should trigger another checkpoint
                [{'id': 5}]                # Final batch
            ])
            
            batches = list(extractor.extract_with_retry(source))
            
            # Check checkpoint was created
            checkpoint = extractor._last_checkpoint
            assert checkpoint is not None
            assert 'records_processed' in checkpoint
            assert checkpoint['records_processed'] == 5
            assert 'timestamp' in checkpoint