"""
Unit tests for geographic standardisation components.

Tests the SA2 mapping accuracy, coordinate transformations, and boundary processing
following Test-Driven Development (TDD) approach.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, List, Any
import pytest

# Import modules under test
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.transformers.geographic_standardiser import (
    GeographicStandardiser,
    SA2MappingEngine,
    PostcodeToSA2Mapper,
    LGAToSA2Mapper,
    PHNToSA2Mapper,
    GeographicMapping,
    GeographicValidationResult
)

from src.transformers.coordinate_transformer import (
    CoordinateSystemTransformer,
    CoordinatePoint,
    GDA2020Transformer,
    MGAZoneTransformer,
    GeographicValidator
)

from src.utils.geographic_utils import (
    AustralianGeographicConstants,
    SA2HierarchyValidator,
    PostcodeValidator,
    DistanceCalculator,
    validate_australian_coordinates,
    determine_state_from_coordinates
)

from src.utils.interfaces import (
    GeographicValidationError,
    ValidationResult,
    ValidationSeverity
)


class TestSA2MappingEngine(unittest.TestCase):
    """Test SA2 mapping engine functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'required_accuracy_metres': 1.0,
            'enable_datum_transformation': True
        }
        self.engine = SA2MappingEngine(self.config)
    
    def test_map_postcode_to_sa2_single_mapping(self):
        """Test mapping a postcode that maps to a single SA2."""
        # Test data - Sydney CBD postcode
        postcode = "2000"
        
        # Expected to map to a single SA2 with high confidence
        mappings = self.engine.map_to_sa2(postcode, "postcode")
        
        self.assertIsInstance(mappings, list)
        self.assertGreater(len(mappings), 0)
        
        # Check first mapping
        mapping = mappings[0]
        self.assertIsInstance(mapping, GeographicMapping)
        self.assertEqual(mapping.source_code, postcode)
        self.assertEqual(mapping.source_type, "postcode")
        self.assertRegex(mapping.target_sa2_code, r'^\d{9}$')  # SA2 format
        self.assertGreater(mapping.confidence, 0.8)  # High confidence
    
    def test_map_postcode_to_sa2_multiple_mappings(self):
        """Test mapping a postcode that spans multiple SA2s."""
        # Large postcode that typically spans multiple SA2s
        postcode = "3000"  # Melbourne CBD area
        
        mappings = self.engine.map_to_sa2(postcode, "postcode")
        
        # Should have at least one mapping
        self.assertGreater(len(mappings), 0)
        
        # Check allocation factors sum to approximately 1.0
        total_allocation = sum(m.allocation_factor for m in mappings)
        self.assertAlmostEqual(total_allocation, 1.0, places=2)
        
        # All mappings should be for the same postcode
        for mapping in mappings:
            self.assertEqual(mapping.source_code, postcode)
            self.assertEqual(mapping.source_type, "postcode")
    
    def test_map_invalid_postcode(self):
        """Test mapping an invalid postcode raises appropriate error."""
        invalid_postcode = "0000"  # Invalid Australian postcode
        
        with self.assertRaises(GeographicValidationError):
            self.engine.map_to_sa2(invalid_postcode, "postcode")
    
    def test_map_sa1_to_sa2_direct_relationship(self):
        """Test SA1 to SA2 mapping (1:1 relationship)."""
        # Valid SA1 code
        sa1_code = "10102100101"
        
        mappings = self.engine.map_to_sa2(sa1_code, "sa1")
        
        # Should have exactly one mapping for SA1->SA2
        self.assertEqual(len(mappings), 1)
        
        mapping = mappings[0]
        self.assertEqual(mapping.source_code, sa1_code)
        self.assertEqual(mapping.source_type, "sa1")
        self.assertEqual(mapping.allocation_factor, 1.0)  # Complete allocation
        self.assertEqual(mapping.mapping_method, "direct")
        self.assertEqual(mapping.confidence, 1.0)  # Perfect confidence
        
        # SA2 code should be prefix of SA1 code
        self.assertTrue(sa1_code.startswith(mapping.target_sa2_code))
    
    def test_map_sa3_to_sa2_hierarchical_decomposition(self):
        """Test SA3 to SA2 mapping (1:many relationship)."""
        sa3_code = "10102"
        
        mappings = self.engine.map_to_sa2(sa3_code, "sa3")
        
        # Should have multiple mappings for SA3->SA2
        self.assertGreater(len(mappings), 1)
        
        # Check all mappings
        for mapping in mappings:
            self.assertEqual(mapping.source_code, sa3_code)
            self.assertEqual(mapping.source_type, "sa3")
            self.assertGreater(mapping.allocation_factor, 0)
            self.assertTrue(mapping.target_sa2_code.startswith(sa3_code))
        
        # Allocation factors should sum to 1.0
        total_allocation = sum(m.allocation_factor for m in mappings)
        self.assertAlmostEqual(total_allocation, 1.0, places=2)
    
    def test_validate_sa2_code_valid(self):
        """Test validation of a valid SA2 code."""
        valid_sa2 = "101021007"
        
        result = self.engine.validate_geographic_hierarchy(valid_sa2)
        
        self.assertIsInstance(result, GeographicValidationResult)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.code, valid_sa2)
        self.assertEqual(result.code_type, "sa2")
        self.assertEqual(result.sa2_code, valid_sa2)
        self.assertEqual(result.confidence, 1.0)
    
    def test_validate_sa2_code_invalid_format(self):
        """Test validation of SA2 code with invalid format."""
        invalid_sa2 = "12345"  # Too short
        
        result = self.engine.validate_geographic_hierarchy(invalid_sa2)
        
        self.assertFalse(result.is_valid)
        self.assertIn("9 digits", result.error_message)
    
    def test_validate_sa2_code_non_numeric(self):
        """Test validation of SA2 code with non-numeric characters."""
        invalid_sa2 = "ABC123DEF"
        
        result = self.engine.validate_geographic_hierarchy(invalid_sa2)
        
        self.assertFalse(result.is_valid)
        self.assertIn("numeric", result.error_message)
    
    def test_cache_performance(self):
        """Test that caching improves performance for repeated lookups."""
        postcode = "2000"
        
        # First lookup - cache miss
        mappings1 = self.engine.map_to_sa2(postcode, "postcode")
        
        # Second lookup - cache hit
        mappings2 = self.engine.map_to_sa2(postcode, "postcode")
        
        # Results should be identical
        self.assertEqual(len(mappings1), len(mappings2))
        
        # Check cache statistics
        stats = self.engine.get_cache_statistics()
        self.assertGreater(stats['cache_hits'], 0)
        self.assertGreater(stats['hit_rate'], 0)


class TestPostcodeToSA2Mapper(unittest.TestCase):
    """Test postcode to SA2 mapping with population weighting."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {}
        self.mapper = PostcodeToSA2Mapper(self.config)
    
    def test_map_postcode_population_weighted(self):
        """Test postcode mapping with population weighting."""
        postcode = "2000"
        
        mappings = self.mapper.map_postcode(postcode, "population")
        
        self.assertIsInstance(mappings, list)
        self.assertGreater(len(mappings), 0)
        
        for mapping in mappings:
            self.assertEqual(mapping.source_code, postcode)
            self.assertEqual(mapping.source_type, "postcode")
            self.assertIn("population_weighted", mapping.mapping_method)
            self.assertGreaterEqual(mapping.confidence, 0.9)  # High confidence for population weighting
    
    def test_validate_postcode_valid_format(self):
        """Test validation of valid postcode format."""
        valid_postcodes = ["2000", "3000", "4000", "5000"]
        
        for postcode in valid_postcodes:
            with self.subTest(postcode=postcode):
                self.assertTrue(self.mapper.validate_postcode(postcode))
    
    def test_validate_postcode_invalid_format(self):
        """Test validation of invalid postcode formats."""
        invalid_postcodes = ["200", "20000", "ABCD", "12.34"]
        
        for postcode in invalid_postcodes:
            with self.subTest(postcode=postcode):
                self.assertFalse(self.mapper.validate_postcode(postcode))
    
    def test_validate_postcode_out_of_range(self):
        """Test validation of postcodes outside Australian ranges."""
        invalid_postcodes = ["0123", "1234", "9876"]
        
        for postcode in invalid_postcodes:
            with self.subTest(postcode=postcode):
                self.assertFalse(self.mapper.validate_postcode(postcode))


class TestCoordinateSystemTransformer(unittest.TestCase):
    """Test coordinate system transformations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'target_system': 'GDA2020',
            'longitude_column': 'longitude',
            'latitude_column': 'latitude',
            'coordinate_system_column': 'coordinate_system'
        }
        self.transformer = CoordinateSystemTransformer(config=self.config)
    
    def test_transform_wgs84_to_gda2020(self):
        """Test transformation from WGS84 to GDA2020."""
        # Sydney Opera House coordinates
        test_data = [{
            'longitude': 151.2153,
            'latitude': -33.8568,
            'coordinate_system': 'WGS84'
        }]
        
        transformed = self.transformer.transform(test_data)
        
        self.assertEqual(len(transformed), 1)
        
        result = transformed[0]
        self.assertIn('longitude_gda2020', result)
        self.assertIn('latitude_gda2020', result)
        self.assertEqual(result['coordinate_system_gda2020'], 'GDA2020')
        
        # WGS84 and GDA2020 should be very close (within mm)
        self.assertAlmostEqual(result['longitude_gda2020'], 151.2153, places=6)
        self.assertAlmostEqual(result['latitude_gda2020'], -33.8568, places=6)
    
    def test_transform_gda94_to_gda2020(self):
        """Test transformation from GDA94 to GDA2020."""
        test_data = [{
            'longitude': 151.2153,
            'latitude': -33.8568,
            'coordinate_system': 'GDA94'
        }]
        
        transformed = self.transformer.transform(test_data)
        
        result = transformed[0]
        # GDA94 to GDA2020 should have small but measurable difference
        lon_diff = abs(result['longitude_gda2020'] - 151.2153)
        lat_diff = abs(result['latitude_gda2020'] - (-33.8568))
        
        # Differences should be small but detectable (centimetre level)
        self.assertLess(lon_diff, 0.001)  # Less than 0.001 degrees
        self.assertLess(lat_diff, 0.001)
    
    def test_transform_invalid_coordinates(self):
        """Test handling of invalid coordinates."""
        test_data = [{
            'longitude': 200.0,  # Invalid longitude
            'latitude': -33.8568,
            'coordinate_system': 'WGS84'
        }]
        
        transformed = self.transformer.transform(test_data)
        
        # Should still return a record but with error information
        self.assertEqual(len(transformed), 1)
        self.assertIn('transformation_error', transformed[0])
    
    def test_determine_mga_zone(self):
        """Test MGA zone determination."""
        # Test coordinates in different MGA zones
        test_cases = [
            (115.0, -32.0, 49),  # Perth - Zone 49
            (138.0, -35.0, 53),  # Adelaide - Zone 53  
            (145.0, -37.0, 54),  # Melbourne - Zone 54
            (153.0, -27.0, 55),  # Brisbane - Zone 55
        ]
        
        for lon, lat, expected_zone in test_cases:
            with self.subTest(lon=lon, lat=lat):
                test_data = [{
                    'longitude': lon,
                    'latitude': lat,
                    'coordinate_system': 'GDA2020'
                }]
                
                transformed = self.transformer.transform(test_data)
                result = transformed[0]
                
                self.assertIn('mga_zone', result)
                if result['mga_zone']:
                    zone_num = int(result['mga_zone'].split('_')[-1])
                    self.assertEqual(zone_num, expected_zone)


class TestGeographicStandardiser(unittest.TestCase):
    """Test the main geographic standardiser."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'geographic_column': 'location_code',
            'geographic_type_column': 'location_type',
            'output_sa2_column': 'sa2_code',
            'batch_size': 10
        }
        self.standardiser = GeographicStandardiser(config=self.config)
    
    def test_transform_postcode_data(self):
        """Test transformation of postcode data to SA2."""
        test_data = [
            {'location_code': '2000', 'location_type': 'postcode', 'value': 100},
            {'location_code': '3000', 'location_type': 'postcode', 'value': 200},
        ]
        
        transformed = self.standardiser.transform(test_data)
        
        # Should have at least as many records as input (possibly more due to 1:many mappings)
        self.assertGreaterEqual(len(transformed), len(test_data))
        
        # All output records should have SA2 codes
        for record in transformed:
            self.assertIn('sa2_code', record)
            self.assertRegex(record['sa2_code'], r'^\d{9}$')
            self.assertIn('allocation_factor', record)
            self.assertGreater(record['allocation_factor'], 0)
    
    def test_transform_mixed_geographic_types(self):
        """Test transformation of mixed geographic data types."""
        test_data = [
            {'location_code': '2000', 'location_type': 'postcode', 'value': 100},
            {'location_code': '101021007', 'location_type': 'sa2', 'value': 200},
            {'location_code': '10102', 'location_type': 'sa3', 'value': 300},
        ]
        
        transformed = self.standardiser.transform(test_data)
        
        # Should handle all types successfully
        self.assertGreaterEqual(len(transformed), len(test_data))
        
        # Check each record has required SA2 standardisation
        for record in transformed:
            self.assertIn('sa2_code', record)
            self.assertIn('allocation_factor', record)
            self.assertIn('mapping_method', record)
    
    def test_validate_geographic_data(self):
        """Test validation of geographic data before transformation."""
        test_data = [
            {'location_code': '2000', 'location_type': 'postcode'},  # Valid
            {'location_type': 'postcode'},  # Missing location_code
            {'location_code': 'INVALID', 'location_type': 'postcode'},  # Invalid code
        ]
        
        validation_results = self.standardiser.validate_geographic_data(test_data)
        
        # Should have validation results
        self.assertGreater(len(validation_results), 0)
        
        # Should detect missing location_code
        missing_code_errors = [r for r in validation_results if 'missing' in r.message.lower()]
        self.assertGreater(len(missing_code_errors), 0)
    
    def test_get_transformation_statistics(self):
        """Test retrieval of transformation statistics."""
        test_data = [
            {'location_code': '2000', 'location_type': 'postcode', 'value': 100}
        ]
        
        self.standardiser.transform(test_data)
        stats = self.standardiser.get_transformation_statistics()
        
        self.assertIn('total_records', stats)
        self.assertIn('successful_mappings', stats)
        self.assertIn('failed_mappings', stats)
        self.assertEqual(stats['total_records'], 1)


class TestAustralianGeographicConstants(unittest.TestCase):
    """Test Australian geographic constants and utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.constants = AustralianGeographicConstants()
    
    def test_states_definition(self):
        """Test state definitions are complete and correct."""
        expected_states = ['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT', 'OT']
        
        actual_states = [info['abbreviation'] for info in self.constants.STATES.values()]
        
        for state in expected_states:
            self.assertIn(state, actual_states)
    
    def test_postcode_ranges_coverage(self):
        """Test postcode ranges cover all states."""
        for state in ['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT']:
            self.assertIn(state, self.constants.POSTCODE_RANGES)
            ranges = self.constants.POSTCODE_RANGES[state]
            self.assertGreater(len(ranges), 0)
    
    def test_mga_zones_definition(self):
        """Test MGA zones are properly defined."""
        expected_zones = [49, 50, 51, 52, 53, 54, 55, 56]
        
        for zone in expected_zones:
            self.assertIn(zone, self.constants.MGA_ZONES)
            zone_info = self.constants.MGA_ZONES[zone]
            self.assertIn('central_meridian', zone_info)
            self.assertIn('bounds', zone_info)
            self.assertIn('states', zone_info)
    
    def test_geographic_bounds_sanity(self):
        """Test geographic bounds are sensible."""
        mainland = self.constants.GEOGRAPHIC_BOUNDS['mainland']
        
        # Check longitude range
        self.assertLess(mainland['min_longitude'], mainland['max_longitude'])
        self.assertGreater(mainland['min_longitude'], 100)  # Reasonable minimum
        self.assertLess(mainland['max_longitude'], 160)     # Reasonable maximum
        
        # Check latitude range (southern hemisphere)
        self.assertLess(mainland['min_latitude'], mainland['max_latitude'])
        self.assertLess(mainland['min_latitude'], 0)        # Southern hemisphere
        self.assertLess(mainland['max_latitude'], 0)        # Southern hemisphere


class TestSA2HierarchyValidator(unittest.TestCase):
    """Test SA2 hierarchy validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = SA2HierarchyValidator()
    
    def test_validate_sa2_code_format(self):
        """Test SA2 code format validation."""
        valid_sa2 = "101021007"
        is_valid, errors = self.validator.validate_hierarchy_code(valid_sa2, "SA2")
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_sa2_code_invalid_format(self):
        """Test SA2 code with invalid format."""
        invalid_sa2 = "12345"  # Too short
        is_valid, errors = self.validator.validate_hierarchy_code(invalid_sa2, "SA2")
        
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_extract_parent_codes_sa1(self):
        """Test extraction of parent codes from SA1."""
        sa1_code = "10102100101"
        parent_codes = self.validator.extract_parent_codes(sa1_code, "SA1")
        
        expected_parents = {
            'SA2': '101021001',
            'SA3': '10102',
            'SA4': '101',
            'STATE': '1'
        }
        
        self.assertEqual(parent_codes, expected_parents)
    
    def test_extract_parent_codes_sa2(self):
        """Test extraction of parent codes from SA2."""
        sa2_code = "101021007"
        parent_codes = self.validator.extract_parent_codes(sa2_code, "SA2")
        
        expected_parents = {
            'SA3': '10102',
            'SA4': '101',
            'STATE': '1'
        }
        
        self.assertEqual(parent_codes, expected_parents)
    
    def test_validate_containment_valid(self):
        """Test valid containment relationship."""
        sa1_code = "10102100101"
        sa2_code = "101021001"
        
        is_valid, errors = self.validator.validate_containment(sa1_code, "SA1", sa2_code, "SA2")
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_containment_invalid(self):
        """Test invalid containment relationship."""
        sa1_code = "10102100101"
        wrong_sa2_code = "201021001"  # Different state
        
        is_valid, errors = self.validator.validate_containment(sa1_code, "SA1", wrong_sa2_code, "SA2")
        
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)


class TestPostcodeValidator(unittest.TestCase):
    """Test postcode validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = PostcodeValidator()
    
    def test_validate_valid_postcodes(self):
        """Test validation of valid postcodes."""
        valid_postcodes = [
            ("2000", "NSW"),  # Sydney
            ("3000", "VIC"),  # Melbourne
            ("4000", "QLD"),  # Brisbane
            ("5000", "SA"),   # Adelaide
            ("6000", "WA"),   # Perth
            ("7000", "TAS"),  # Hobart
            ("0800", "NT"),   # Darwin
            ("2600", "ACT"),  # Canberra
        ]
        
        for postcode, state in valid_postcodes:
            with self.subTest(postcode=postcode, state=state):
                is_valid, errors = self.validator.validate_postcode(postcode, state)
                self.assertTrue(is_valid, f"Postcode {postcode} should be valid for {state}")
                self.assertEqual(len(errors), 0)
    
    def test_validate_invalid_postcodes(self):
        """Test validation of invalid postcodes."""
        invalid_postcodes = [
            "0123",   # Invalid range
            "1234",   # Invalid range
            "ABCD",   # Non-numeric
            "12345",  # Too long
            "123",    # Too short
        ]
        
        for postcode in invalid_postcodes:
            with self.subTest(postcode=postcode):
                is_valid, errors = self.validator.validate_postcode(postcode)
                self.assertFalse(is_valid)
                self.assertGreater(len(errors), 0)
    
    def test_get_postcode_state(self):
        """Test retrieval of state for postcode."""
        test_cases = [
            ("2000", "NSW"),
            ("3000", "VIC"),
            ("4000", "QLD"),
            ("5000", "SA"),
        ]
        
        for postcode, expected_state in test_cases:
            with self.subTest(postcode=postcode):
                actual_state = self.validator.get_postcode_state(postcode)
                self.assertEqual(actual_state, expected_state)
    
    def test_validate_postcode_for_state_mismatch(self):
        """Test validation when postcode doesn't match expected state."""
        # NSW postcode with VIC state
        is_valid = self.validator.validate_postcode_for_state("2000", "VIC")
        self.assertFalse(is_valid)


class TestDistanceCalculator(unittest.TestCase):
    """Test distance calculation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = DistanceCalculator()
    
    def test_calculate_great_circle_distance(self):
        """Test great circle distance calculation."""
        # Distance between Sydney and Melbourne (approximately 713 km)
        sydney_lon, sydney_lat = 151.2093, -33.8688
        melbourne_lon, melbourne_lat = 144.9631, -37.8136
        
        distance = self.calculator.calculate_great_circle_distance(
            sydney_lon, sydney_lat, melbourne_lon, melbourne_lat
        )
        
        # Should be approximately 713 km
        self.assertAlmostEqual(distance, 713.0, delta=20.0)  # ±20km tolerance
    
    def test_calculate_haversine_distance(self):
        """Test haversine distance calculation."""
        # Distance between Perth and Adelaide (approximately 2130 km)
        perth_lon, perth_lat = 115.8605, -31.9505
        adelaide_lon, adelaide_lat = 138.6007, -34.9285
        
        distance = self.calculator.calculate_haversine_distance(
            perth_lon, perth_lat, adelaide_lon, adelaide_lat
        )
        
        # Should be approximately 2130 km
        self.assertAlmostEqual(distance, 2130.0, delta=50.0)  # ±50km tolerance
    
    def test_calculate_bearing(self):
        """Test bearing calculation."""
        # Bearing from Sydney to Melbourne (approximately southwest, ~225°)
        sydney_lon, sydney_lat = 151.2093, -33.8688
        melbourne_lon, melbourne_lat = 144.9631, -37.8136
        
        bearing = self.calculator.calculate_bearing(
            sydney_lon, sydney_lat, melbourne_lon, melbourne_lat
        )
        
        # Should be approximately southwest (225°)
        self.assertAlmostEqual(bearing, 225.0, delta=30.0)  # ±30° tolerance
    
    def test_is_within_distance(self):
        """Test distance threshold checking."""
        # Two points in Sydney CBD (should be within 5km)
        point1_lon, point1_lat = 151.2093, -33.8688  # Sydney Opera House
        point2_lon, point2_lat = 151.2073, -33.8569  # Circular Quay
        
        is_within = self.calculator.is_within_distance(
            point1_lon, point1_lat, point2_lon, point2_lat, 5.0
        )
        
        self.assertTrue(is_within)
    
    def test_find_mga_zone_for_coordinates(self):
        """Test MGA zone determination for coordinates."""
        test_cases = [
            (115.8605, -31.9505, 49),  # Perth - Zone 49
            (138.6007, -34.9285, 53),  # Adelaide - Zone 53
            (144.9631, -37.8136, 54),  # Melbourne - Zone 54
            (151.2093, -33.8688, 55),  # Sydney - Zone 55
            (153.0260, -27.4698, 55),  # Brisbane - Zone 55
        ]
        
        for lon, lat, expected_zone in test_cases:
            with self.subTest(lon=lon, lat=lat):
                zone = self.calculator.find_mga_zone_for_coordinates(lon, lat)
                self.assertEqual(zone, expected_zone)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_validate_australian_coordinates_valid(self):
        """Test validation of valid Australian coordinates."""
        # Sydney coordinates
        is_valid, errors = validate_australian_coordinates(151.2093, -33.8688, True)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_australian_coordinates_invalid(self):
        """Test validation of coordinates outside Australia."""
        # London coordinates
        is_valid, errors = validate_australian_coordinates(-0.1278, 51.5074, True)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_determine_state_from_coordinates(self):
        """Test state determination from coordinates."""
        test_cases = [
            (151.2093, -33.8688, "NSW"),  # Sydney
            (144.9631, -37.8136, "VIC"),  # Melbourne
            (153.0260, -27.4698, "QLD"),  # Brisbane
            (138.6007, -34.9285, "SA"),   # Adelaide
            (115.8605, -31.9505, "WA"),   # Perth
            (147.3257, -42.8821, "TAS"),  # Hobart
            (130.8456, -12.4634, "NT"),   # Darwin
            (149.1300, -35.2809, "ACT"),  # Canberra
        ]
        
        for lon, lat, expected_state in test_cases:
            with self.subTest(lon=lon, lat=lat):
                state = determine_state_from_coordinates(lon, lat)
                self.assertEqual(state, expected_state)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)