"""
Unit tests for Enhanced Geographic Validator.

This module provides comprehensive unit tests for the EnhancedGeographicValidator
class, testing all five validation types: SA2 coverage, boundary topology,
coordinate reference system, spatial hierarchy, and geographic consistency.
"""

import pytest
import logging
from unittest.mock import Mock, patch, mock_open
from typing import Dict, List, Any

from src.validators.enhanced_geographic import (
    EnhancedGeographicValidator,
    SA2CoverageResult,
    BoundaryTopologyResult,
    CRSValidationResult,
    SpatialHierarchyResult,
    GeographicConsistencyResult
)
from src.utils.interfaces import (
    DataBatch,
    DataRecord,
    ValidationResult,
    ValidationSeverity,
    GeographicValidationError
)


@pytest.fixture
def enhanced_validator():
    """Create an enhanced geographic validator for testing."""
    config = {
        'enhanced_geographic': {
            'sa2_coverage': {
                'official_sa2_codes_file': '/test/sa2_codes.txt'
            },
            'boundary_topology': {
                'tolerance_metres': 100,
                'minimum_area_threshold': 0.0001
            },
            'crs_validation': {
                'target_crs': 7855,  # GDA2020 MGA Zone 55
                'coordinate_precision': 6
            },
            'spatial_hierarchy': {},
            'geographic_consistency': {}
        }
    }
    return EnhancedGeographicValidator(config=config)


@pytest.fixture
def sample_sa2_data() -> DataBatch:
    """Create sample SA2 data for testing."""
    return [
        {
            'sa2_code': '10102100001',
            'sa3_code': '10102',
            'sa4_code': '101',
            'state_code': '1',
            'latitude': -33.8688,
            'longitude': 151.2093,
            'total_population': 5000,
            'geographic_area_sqkm': 2.5,
            'is_coastal': True
        },
        {
            'sa2_code': '20202200002',
            'sa3_code': '20202',
            'sa4_code': '202',
            'state_code': '2',
            'latitude': -37.8136,
            'longitude': 144.9631,
            'total_population': 8000,
            'geographic_area_sqkm': 1.8,
            'is_coastal': False
        },
        {
            'sa2_code': '30303300003',
            'sa3_code': '30303',
            'sa4_code': '303',
            'state_code': '3',
            'latitude': -27.4698,
            'longitude': 153.0251,
            'total_population': 12000,
            'geographic_area_sqkm': 3.2,
            'is_coastal': True
        }
    ]


@pytest.fixture
def invalid_sa2_data() -> DataBatch:
    """Create invalid SA2 data for testing."""
    return [
        {
            'sa2_code': '123',  # Too short
            'sa3_code': '12345',
            'sa4_code': '123',
            'state_code': '1',
            'latitude': 100,  # Invalid latitude
            'longitude': 200,  # Invalid longitude
            'total_population': -100,  # Negative population
            'geographic_area_sqkm': 0,  # Zero area
            'is_coastal': True
        },
        {
            'sa2_code': '90909900009',  # Invalid state prefix (9)
            'sa3_code': '90909',
            'sa4_code': '909',
            'state_code': '9',  # Invalid state code
            'latitude': -33.8688,
            'longitude': 151.2093,
            'total_population': 5000,
            'geographic_area_sqkm': 2.5,
            'is_coastal': True
        }
    ]


class TestEnhancedGeographicValidator:
    """Test cases for EnhancedGeographicValidator."""
    
    def test_validator_initialisation(self, enhanced_validator):
        """Test validator initialisation."""
        assert enhanced_validator.validator_id == "enhanced_geographic_validator"
        assert enhanced_validator.OFFICIAL_SA2_COUNT == 2473
        assert enhanced_validator.target_crs == 7855
        assert enhanced_validator.coordinate_precision == 6
        
    def test_get_validation_rules(self, enhanced_validator):
        """Test validation rules retrieval."""
        rules = enhanced_validator.get_validation_rules()
        
        # Check that enhanced rules are included
        assert "enhanced_sa2_coverage_validation" in rules
        assert "boundary_topology_validation" in rules
        assert "coordinate_reference_system_validation" in rules
        assert "spatial_hierarchy_validation" in rules
        assert "geographic_consistency_validation" in rules
        
        # Check that base rules are still included
        assert "coordinate_validation" in rules
        assert "sa2_boundary_validation" in rules
    
    def test_validate_empty_data(self, enhanced_validator):
        """Test validation with empty data."""
        results = enhanced_validator.validate([])
        
        assert len(results) == 1
        assert not results[0].is_valid
        assert results[0].severity == ValidationSeverity.ERROR
        assert results[0].rule_id == "enhanced_geographic_empty_data"
    
    def test_validate_sa2_coverage_valid_codes(self, enhanced_validator, sample_sa2_data):
        """Test SA2 coverage validation with valid codes."""
        # Mock official SA2 codes
        enhanced_validator.official_sa2_codes = {
            '10102100001', '20202200002', '30303300003'
        }
        
        results = enhanced_validator.validate_sa2_coverage(sample_sa2_data)
        
        # Should find one INFO result for complete coverage
        info_results = [r for r in results if r.severity == ValidationSeverity.INFO]
        assert len(info_results) == 1
        assert info_results[0].rule_id == "sa2_coverage_complete"
    
    def test_validate_sa2_coverage_missing_codes(self, enhanced_validator, sample_sa2_data):
        """Test SA2 coverage validation with missing codes."""
        # Mock official SA2 codes with additional codes not in data
        enhanced_validator.official_sa2_codes = {
            '10102100001', '20202200002', '30303300003',
            '40404400004', '50505500005'  # Missing from data
        }
        
        results = enhanced_validator.validate_sa2_coverage(sample_sa2_data)
        
        # Should find WARNING result for incomplete coverage
        warning_results = [r for r in results if r.severity == ValidationSeverity.WARNING]
        assert len(warning_results) == 1
        assert warning_results[0].rule_id == "sa2_coverage_incomplete"
        assert "Missing 2 SA2 codes" in warning_results[0].message
    
    def test_validate_sa2_coverage_invalid_format(self, enhanced_validator):
        """Test SA2 coverage validation with invalid format codes."""
        invalid_data = [
            {'sa2_code': '123'},  # Too short
            {'sa2_code': '9999999999'},  # Invalid state prefix
            {'sa2_code': 'ABCDEFGHIJK'}  # Non-numeric
        ]
        
        results = enhanced_validator.validate_sa2_coverage(invalid_data)
        
        # Should find ERROR results for invalid formats
        error_results = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(error_results) == 3
        
        for result in error_results:
            assert result.rule_id == "sa2_invalid_format_enhanced"
    
    def test_validate_sa2_coverage_duplicate_codes(self, enhanced_validator):
        """Test SA2 coverage validation with duplicate codes."""
        duplicate_data = [
            {'sa2_code': '10102100001'},
            {'sa2_code': '10102100001'},  # Duplicate
            {'sa2_code': '20202200002'}
        ]
        
        results = enhanced_validator.validate_sa2_coverage(duplicate_data)
        
        # Should find ERROR result for duplicate codes
        error_results = [r for r in results if r.severity == ValidationSeverity.ERROR]
        duplicate_errors = [r for r in error_results if r.rule_id == "sa2_duplicate_codes"]
        
        assert len(duplicate_errors) == 1
        assert "appears 2 times" in duplicate_errors[0].message
    
    @patch('src.validators.enhanced_geographic.SPATIAL_LIBRARIES_AVAILABLE', False)
    def test_boundary_topology_libraries_unavailable(self, enhanced_validator, sample_sa2_data):
        """Test boundary topology validation when spatial libraries unavailable."""
        results = enhanced_validator.validate_boundary_topology(sample_sa2_data)
        
        assert len(results) == 1
        assert results[0].severity == ValidationSeverity.WARNING
        assert results[0].rule_id == "boundary_topology_libraries_unavailable"
    
    @patch('src.validators.enhanced_geographic.SPATIAL_LIBRARIES_AVAILABLE', False)
    def test_crs_validation_libraries_unavailable(self, enhanced_validator, sample_sa2_data):
        """Test CRS validation when spatial libraries unavailable."""
        results = enhanced_validator.validate_coordinate_reference_system(sample_sa2_data)
        
        assert len(results) == 1
        assert results[0].severity == ValidationSeverity.WARNING
        assert results[0].rule_id == "crs_validation_libraries_unavailable"
    
    def test_validate_spatial_hierarchy_valid(self, enhanced_validator, sample_sa2_data):
        """Test spatial hierarchy validation with valid hierarchy."""
        results = enhanced_validator.validate_spatial_hierarchy(sample_sa2_data)
        
        # No hierarchy errors should be found for valid data
        hierarchy_errors = [r for r in results if "hierarchy" in r.rule_id and not r.is_valid]
        assert len(hierarchy_errors) == 0
    
    def test_validate_spatial_hierarchy_invalid(self, enhanced_validator):
        """Test spatial hierarchy validation with invalid hierarchy."""
        invalid_hierarchy_data = [
            {
                'sa2_code': '10102100001',
                'sa3_code': '20202',  # Wrong SA3 for SA2
                'sa4_code': '101',
                'state_code': '1'
            },
            {
                'sa2_code': '20202200002',
                'sa3_code': '20202',
                'sa4_code': '303',  # Wrong SA4 for SA3
                'state_code': '2'
            }
        ]
        
        results = enhanced_validator.validate_spatial_hierarchy(invalid_hierarchy_data)
        
        # Should find hierarchy errors
        hierarchy_errors = [r for r in results if "hierarchy" in r.rule_id and not r.is_valid]
        assert len(hierarchy_errors) >= 1
    
    def test_validate_geographic_consistency_valid(self, enhanced_validator, sample_sa2_data):
        """Test geographic consistency validation with valid data."""
        results = enhanced_validator.validate_geographic_consistency(sample_sa2_data)
        
        # Should not find major consistency errors with valid data
        error_results = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(error_results) == 0
    
    def test_validate_geographic_consistency_invalid_area(self, enhanced_validator):
        """Test geographic consistency validation with invalid area."""
        invalid_area_data = [
            {
                'sa2_code': '10102100001',
                'geographic_area_sqkm': -5.0,  # Negative area
                'total_population': 5000,
                'latitude': -33.8688,
                'longitude': 151.2093
            },
            {
                'sa2_code': '20202200002', 
                'geographic_area_sqkm': 200000,  # Unreasonably large area
                'total_population': 1000,
                'latitude': -37.8136,
                'longitude': 144.9631
            }
        ]
        
        results = enhanced_validator.validate_geographic_consistency(invalid_area_data)
        
        # Should find area validation warnings/errors
        area_issues = [r for r in results if "area" in r.rule_id.lower()]
        assert len(area_issues) >= 1
    
    def test_validate_geographic_consistency_invalid_density(self, enhanced_validator):
        """Test geographic consistency validation with invalid population density."""
        invalid_density_data = [
            {
                'sa2_code': '10102100001',
                'geographic_area_sqkm': 0.001,  # Very small area
                'total_population': 100000,  # Very high population = extreme density
                'latitude': -33.8688,
                'longitude': 151.2093
            }
        ]
        
        results = enhanced_validator.validate_geographic_consistency(invalid_density_data)
        
        # Should find density validation warnings
        density_issues = [r for r in results if "density" in r.rule_id.lower()]
        assert len(density_issues) >= 1
    
    def test_validate_coordinates_outside_australia(self, enhanced_validator):
        """Test coordinate validation with coordinates outside Australia."""
        overseas_data = [
            {
                'sa2_code': '10102100001',
                'latitude': 51.5074,   # London
                'longitude': -0.1278,
                'total_population': 5000,
                'geographic_area_sqkm': 2.5
            }
        ]
        
        results = enhanced_validator.validate_coordinate_reference_system(overseas_data)
        
        # Should find coordinates outside Australian territory
        territory_errors = [r for r in results if "australian_territory" in r.rule_id]
        assert len(territory_errors) == 1
        assert not territory_errors[0].is_valid
    
    def test_validate_coordinate_precision(self, enhanced_validator):
        """Test coordinate precision validation."""
        low_precision_data = [
            {
                'sa2_code': '10102100001',
                'latitude': -33.87,  # Only 2 decimal places
                'longitude': 151.21,  # Only 2 decimal places
                'total_population': 5000,
                'geographic_area_sqkm': 2.5
            }
        ]
        
        results = enhanced_validator.validate_coordinate_reference_system(low_precision_data)
        
        # Should find precision warnings
        precision_warnings = [r for r in results if "precision" in r.rule_id]
        assert len(precision_warnings) >= 1
    
    def test_sa2_code_format_validation(self, enhanced_validator):
        """Test enhanced SA2 code format validation."""
        # Test valid format
        assert enhanced_validator._validate_sa2_code_format_enhanced('10102100001') == True
        assert enhanced_validator._validate_sa2_code_format_enhanced('87654AB1234') == True
        
        # Test invalid formats
        assert enhanced_validator._validate_sa2_code_format_enhanced('123') == False  # Too short
        assert enhanced_validator._validate_sa2_code_format_enhanced('90102100001') == False  # Invalid state prefix
        assert enhanced_validator._validate_sa2_code_format_enhanced('ABCDEFGHIJK') == False  # All letters
    
    def test_load_official_sa2_codes_from_file(self, enhanced_validator):
        """Test loading official SA2 codes from file."""
        sample_codes = "10102100001\n20202200002\n30303300003\n"
        
        with patch('builtins.open', mock_open(read_data=sample_codes)):
            with patch('pathlib.Path.exists', return_value=True):
                codes = enhanced_validator._load_official_sa2_codes()
                
                assert len(codes) == 3
                assert '10102100001' in codes
                assert '20202200002' in codes
                assert '30303300003' in codes
    
    def test_load_official_sa2_codes_file_missing(self, enhanced_validator):
        """Test loading official SA2 codes when file is missing."""
        with patch('pathlib.Path.exists', return_value=False):
            codes = enhanced_validator._load_official_sa2_codes()
            
            # Should generate expected codes
            assert len(codes) > 0
            assert len(codes) <= enhanced_validator.OFFICIAL_SA2_COUNT
    
    def test_enhanced_statistics(self, enhanced_validator, sample_sa2_data):
        """Test enhanced statistics collection."""
        # Run some validations
        enhanced_validator.validate_sa2_coverage(sample_sa2_data)
        enhanced_validator.validate_spatial_hierarchy(sample_sa2_data)
        enhanced_validator.validate_geographic_consistency(sample_sa2_data)
        
        stats = enhanced_validator.get_enhanced_statistics()
        
        # Check that enhanced statistics are tracked
        assert 'sa2_coverage_checks' in stats
        assert 'spatial_hierarchy_checks' in stats
        assert 'geographic_consistency_checks' in stats
        assert stats['sa2_coverage_checks'] >= 1
        assert stats['spatial_hierarchy_checks'] >= 1
        assert stats['geographic_consistency_checks'] >= 1
    
    def test_full_validation_integration(self, enhanced_validator, sample_sa2_data):
        """Test full validation integration."""
        # Mock official SA2 codes
        enhanced_validator.official_sa2_codes = {
            '10102100001', '20202200002', '30303300003'
        }
        
        results = enhanced_validator.validate(sample_sa2_data)
        
        # Should return results from all validation types
        assert len(results) > 0
        
        # Check that different validation types are represented
        rule_ids = {result.rule_id for result in results}
        
        # Should have some base validation rules
        assert any('coordinate' in rule_id for rule_id in rule_ids)
        
        # Should have enhanced validation info
        assert any('sa2_coverage' in rule_id for rule_id in rule_ids)
    
    def test_error_handling_in_validation(self, enhanced_validator):
        """Test error handling during validation."""
        # Create data that will cause validation errors
        problematic_data = [
            {
                'sa2_code': None,  # None value
                'latitude': 'invalid',  # Invalid coordinate
                'longitude': 'invalid',
                'total_population': 'not_a_number',
                'geographic_area_sqkm': 'not_a_number'
            }
        ]
        
        # Should not raise exceptions, but return validation results
        results = enhanced_validator.validate(problematic_data)
        
        # Should contain error results but not crash
        assert len(results) > 0
        error_results = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(error_results) > 0


class TestSA2FormatValidation:
    """Test cases specifically for SA2 format validation."""
    
    @pytest.mark.parametrize("sa2_code,expected", [
        ('10102100001', True),   # Valid NSW SA2
        ('20202200002', True),   # Valid VIC SA2
        ('87654AB1234', True),   # Valid with alpha characters
        ('123', False),          # Too short
        ('90102100001', False),  # Invalid state prefix (9)
        ('ABCDEFGHIJK', False),  # All letters
        ('1010210000A', False),  # Letter in wrong position
        ('', False),             # Empty string
        ('12345678901', True),   # Numeric 11-digit
    ])
    def test_sa2_code_format_validation_parametrised(self, enhanced_validator, sa2_code, expected):
        """Test SA2 code format validation with various inputs."""
        result = enhanced_validator._validate_sa2_code_format_enhanced(sa2_code)
        assert result == expected


class TestAustralianBounds:
    """Test cases for Australian territorial bounds validation."""
    
    @pytest.mark.parametrize("lat,lon,should_be_valid", [
        (-33.8688, 151.2093, True),   # Sydney - valid
        (-37.8136, 144.9631, True),   # Melbourne - valid
        (-27.4698, 153.0251, True),   # Brisbane - valid
        (51.5074, -0.1278, False),    # London - invalid
        (40.7128, -74.0060, False),   # New York - invalid
        (-90, 0, False),              # South Pole - invalid
        (0, 0, False),                # Null Island - invalid
        (-54.0, 158.0, True),         # Near Macquarie Island - valid
        (-9.5, 143.0, True),          # Near Cape York - valid
    ])
    def test_australian_territorial_bounds(self, enhanced_validator, lat, lon, should_be_valid):
        """Test Australian territorial bounds validation."""
        result = enhanced_validator._validate_australian_territorial_bounds(lat, lon, 0)
        
        if should_be_valid:
            assert result is None  # No validation error
        else:
            assert result is not None  # Validation error found
            assert not result.is_valid


if __name__ == '__main__':
    pytest.main([__file__])