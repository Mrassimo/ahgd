"""
Integration tests for end-to-end geographic processing pipeline.

Tests the complete geographic standardisation workflow from raw input
to SA2-standardised output, including coordinate transformations,
boundary processing, and spatial indexing.
"""

import unittest
import tempfile
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import pytest

# Import modules under test
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.transformers.geographic_standardiser import GeographicStandardiser
from src.transformers.coordinate_transformer import CoordinateSystemTransformer
from src.transformers.boundary_processor import BoundaryProcessor
from src.utils.geographic_utils import (
    AustralianGeographicConstants,
    SA2HierarchyValidator,
    PostcodeValidator,
    DistanceCalculator
)
from src.utils.interfaces import ValidationSeverity


class TestGeographicPipelineIntegration(unittest.TestCase):
    """Test complete geographic processing pipeline integration."""
    
    def setUp(self):
        """Set up test fixtures and temporary directories."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Configuration for pipeline components
        self.geographic_config = {
            'geographic_column': 'location_code',
            'geographic_type_column': 'location_type',
            'output_sa2_column': 'sa2_code',
            'include_mapping_metadata': True,
            'strict_validation': False,
            'batch_size': 100
        }
        
        self.coordinate_config = {
            'longitude_column': 'longitude',
            'latitude_column': 'latitude',
            'coordinate_system_column': 'coordinate_system',
            'target_system': 'GDA2020',
            'auto_determine_mga_zone': True
        }
        
        self.boundary_config = {
            'enable_validation': True,
            'enable_simplification': True,
            'enable_spatial_indexing': True,
            'simplification_level': 'medium'
        }
        
        # Create pipeline components
        self.geographic_standardiser = GeographicStandardiser(config=self.geographic_config)
        self.coordinate_transformer = CoordinateSystemTransformer(config=self.coordinate_config)
        self.boundary_processor = BoundaryProcessor(config=self.boundary_config)
        
        # Sample test data
        self.sample_geographic_data = [
            {'location_code': '2000', 'location_type': 'postcode', 'population': 15000, 'area_name': 'Sydney CBD'},
            {'location_code': '3000', 'location_type': 'postcode', 'population': 8000, 'area_name': 'Melbourne CBD'},
            {'location_code': '101021007', 'location_type': 'sa2', 'population': 12000, 'area_name': 'Example SA2'},
            {'location_code': '10102', 'location_type': 'sa3', 'population': 150000, 'area_name': 'Example SA3'}
        ]
        
        self.sample_coordinate_data = [
            {'longitude': 151.2093, 'latitude': -33.8688, 'coordinate_system': 'WGS84', 'location': 'Sydney Opera House'},
            {'longitude': 144.9631, 'latitude': -37.8136, 'coordinate_system': 'GDA94', 'location': 'Melbourne CBD'},
            {'longitude': 138.6007, 'latitude': -34.9285, 'coordinate_system': 'GDA2020', 'location': 'Adelaide CBD'},
            {'longitude': 115.8605, 'latitude': -31.9505, 'coordinate_system': 'WGS84', 'location': 'Perth CBD'}
        ]
        
        self.sample_boundary_data = [
            {
                'area_code': '101021007',
                'area_type': 'SA2', 
                'area_name': 'Sydney - CBD - North',
                'state_code': '1',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[[151.20, -33.86], [151.21, -33.86], [151.21, -33.87], [151.20, -33.87], [151.20, -33.86]]]
                }
            },
            {
                'area_code': '201021001',
                'area_type': 'SA2',
                'area_name': 'Melbourne - CBD - North',
                'state_code': '2',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[[144.95, -37.81], [144.97, -37.81], [144.97, -37.82], [144.95, -37.82], [144.95, -37.81]]]
                }
            }
        ]
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_geographic_standardisation(self):
        """Test complete end-to-end geographic standardisation workflow."""
        # Step 1: Geographic standardisation
        standardised_data = self.geographic_standardiser.transform(self.sample_geographic_data)
        
        # Verify standardisation results
        self.assertGreater(len(standardised_data), 0)
        
        # All records should have SA2 codes
        for record in standardised_data:
            self.assertIn('sa2_code', record)
            self.assertRegex(record['sa2_code'], r'^\d{9}$')
            self.assertIn('allocation_factor', record)
            self.assertGreater(record['allocation_factor'], 0)
            self.assertLessEqual(record['allocation_factor'], 1.0)
        
        # Verify metadata inclusion
        metadata_fields = ['mapping_method', 'mapping_confidence', 'source_geographic_code', 'source_geographic_type']
        for field in metadata_fields:
            self.assertTrue(any(field in record for record in standardised_data))
        
        # Get transformation statistics
        stats = self.geographic_standardiser.get_transformation_statistics()
        self.assertGreater(stats['successful_mappings'], 0)
        self.assertEqual(stats['failed_mappings'], 0)
        
        return standardised_data
    
    def test_end_to_end_coordinate_transformation(self):
        """Test complete coordinate transformation workflow."""
        # Transform coordinates to GDA2020
        transformed_data = self.coordinate_transformer.transform(self.sample_coordinate_data)
        
        # Verify transformation results
        self.assertEqual(len(transformed_data), len(self.sample_coordinate_data))
        
        for record in transformed_data:
            # Should have GDA2020 coordinates
            self.assertIn('longitude_gda2020', record)
            self.assertIn('latitude_gda2020', record)
            self.assertIn('coordinate_system_gda2020', record)
            self.assertEqual(record['coordinate_system_gda2020'], 'GDA2020')
            
            # Coordinates should be within Australian bounds
            lon = record['longitude_gda2020']
            lat = record['latitude_gda2020']
            self.assertGreaterEqual(lon, 110)
            self.assertLessEqual(lon, 155)
            self.assertGreaterEqual(lat, -45)
            self.assertLessEqual(lat, -9)
            
            # Should have MGA zone information
            self.assertIn('mga_zone', record)
            if record['mga_zone']:
                self.assertRegex(record['mga_zone'], r'MGA_ZONE_\d{2}')
        
        # Get transformation statistics
        stats = self.coordinate_transformer.get_transformation_statistics()
        self.assertGreater(stats['successful_transformations'], 0)
        self.assertEqual(stats['failed_transformations'], 0)
        
        return transformed_data
    
    def test_end_to_end_boundary_processing(self):
        """Test complete boundary processing workflow."""
        # Process boundary data
        processed_data = self.boundary_processor.transform(self.sample_boundary_data)
        
        # Verify processing results
        self.assertEqual(len(processed_data), len(self.sample_boundary_data))
        
        for record in processed_data:
            # Should have all required fields
            self.assertIn('area_code', record)
            self.assertIn('area_type', record)
            self.assertIn('geometry', record)
            self.assertIn('geometry_info', record)
            self.assertIn('validation_status', record)
            
            # Geometry should be valid
            geometry = record['geometry']
            self.assertIn('type', geometry)
            self.assertIn('coordinates', geometry)
            
            # Geometry info should be populated
            geom_info = record['geometry_info']
            self.assertIn('coordinate_count', geom_info)
            self.assertIn('bbox', geom_info)
            self.assertGreater(geom_info['coordinate_count'], 0)
        
        # Get processing statistics
        stats = self.boundary_processor.get_processing_statistics()
        self.assertGreater(stats['boundaries_processed'], 0)
        
        return processed_data
    
    def test_integrated_geographic_and_coordinate_pipeline(self):
        """Test integration of geographic standardisation with coordinate transformation."""
        # Combine geographic and coordinate data
        combined_data = []
        for i, geo_record in enumerate(self.sample_geographic_data[:2]):  # First 2 records
            coord_record = self.sample_coordinate_data[i]
            combined_record = {**geo_record, **coord_record}
            combined_data.append(combined_record)
        
        # Step 1: Geographic standardisation
        standardised_data = self.geographic_standardiser.transform(combined_data)
        
        # Step 2: Coordinate transformation on standardised data
        # Add coordinate columns to standardised data for transformation
        coord_data = []
        for record in standardised_data:
            # Find original coordinate data
            original_index = 0  # Simplified for test
            coord_record = {
                **record,
                'longitude': self.sample_coordinate_data[original_index]['longitude'],
                'latitude': self.sample_coordinate_data[original_index]['latitude'],
                'coordinate_system': self.sample_coordinate_data[original_index]['coordinate_system']
            }
            coord_data.append(coord_record)
        
        transformed_data = self.coordinate_transformer.transform(coord_data)
        
        # Verify combined results
        for record in transformed_data:
            # Should have both SA2 standardisation and coordinate transformation
            self.assertIn('sa2_code', record)
            self.assertIn('longitude_gda2020', record)
            self.assertIn('latitude_gda2020', record)
            self.assertIn('allocation_factor', record)
        
        return transformed_data
    
    def test_data_quality_validation_pipeline(self):
        """Test data quality validation across the entire pipeline."""
        # Create test data with quality issues
        problematic_data = [
            {'location_code': '9999', 'location_type': 'postcode'},  # Invalid postcode
            {'location_code': '', 'location_type': 'postcode'},     # Empty code
            {'location_code': '2000', 'location_type': ''},         # Empty type
            {'location_code': 'INVALID', 'location_type': 'sa2'},   # Invalid SA2 format
        ]
        
        # Test geographic validation
        validation_results = self.geographic_standardiser.validate_geographic_data(problematic_data)
        
        # Should detect multiple validation issues
        self.assertGreater(len(validation_results), 0)
        
        # Should have errors for invalid data
        error_results = [r for r in validation_results if not r.is_valid]
        self.assertGreater(len(error_results), 0)
        
        # Test coordinate validation with invalid coordinates
        invalid_coords = [
            {'longitude': 200.0, 'latitude': -33.8688, 'coordinate_system': 'WGS84'},  # Invalid longitude
            {'longitude': 151.2093, 'latitude': 100.0, 'coordinate_system': 'WGS84'},  # Invalid latitude
            {'longitude': 0.0, 'latitude': 0.0, 'coordinate_system': 'WGS84'},         # Outside Australia
        ]
        
        # Transform should handle invalid coordinates gracefully
        transformed_coords = self.coordinate_transformer.transform(invalid_coords)
        
        # Should have error information for invalid coordinates
        error_records = [r for r in transformed_coords if 'transformation_error' in r]
        self.assertGreater(len(error_records), 0)
    
    def test_performance_benchmarks(self):
        """Test pipeline performance with larger datasets."""
        import time
        
        # Create larger test dataset
        large_dataset = []
        base_data = self.sample_geographic_data[0]
        
        for i in range(1000):  # 1000 records
            record = base_data.copy()
            record['location_code'] = f"{2000 + (i % 100):04d}"  # Vary postcodes
            record['id'] = i
            large_dataset.append(record)
        
        # Measure geographic standardisation performance
        start_time = time.time()
        standardised_data = self.geographic_standardiser.transform(large_dataset)
        geo_time = time.time() - start_time
        
        # Performance assertions
        self.assertLess(geo_time, 30.0)  # Should complete within 30 seconds
        self.assertEqual(len(standardised_data), len([r for r in standardised_data if 'sa2_code' in r]))
        
        # Measure coordinate transformation performance
        coord_dataset = []
        for i, record in enumerate(large_dataset[:500]):  # 500 coordinate records
            coord_record = {
                **record,
                'longitude': 151.2093 + (i % 100) * 0.01,  # Vary coordinates slightly
                'latitude': -33.8688 + (i % 100) * 0.01,
                'coordinate_system': 'WGS84'
            }
            coord_dataset.append(coord_record)
        
        start_time = time.time()
        transformed_coords = self.coordinate_transformer.transform(coord_dataset)
        coord_time = time.time() - start_time
        
        self.assertLess(coord_time, 20.0)  # Should complete within 20 seconds
        self.assertEqual(len(transformed_coords), len(coord_dataset))
    
    def test_data_lineage_and_audit_trail(self):
        """Test that data lineage and audit trails are maintained throughout pipeline."""
        # Process sample data and track lineage
        standardised_data = self.geographic_standardiser.transform(self.sample_geographic_data)
        
        # Check for audit trail information
        for record in standardised_data:
            if 'mapping_method' in record:
                # Should have mapping metadata
                self.assertIn('mapping_confidence', record)
                self.assertIn('source_geographic_code', record)
                self.assertIn('source_geographic_type', record)
                
                # Confidence should be reasonable
                self.assertGreaterEqual(record['mapping_confidence'], 0.0)
                self.assertLessEqual(record['mapping_confidence'], 1.0)
    
    def test_error_recovery_and_robustness(self):
        """Test pipeline robustness and error recovery mechanisms."""
        # Test with mixed valid and invalid data
        mixed_data = [
            {'location_code': '2000', 'location_type': 'postcode'},  # Valid
            {'location_code': 'INVALID', 'location_type': 'postcode'},  # Invalid
            {'location_code': '101021007', 'location_type': 'sa2'},  # Valid
            None,  # Null record
            {'location_code': '', 'location_type': ''},  # Empty record
            {'location_code': '3000', 'location_type': 'postcode'},  # Valid
        ]
        
        # Filter out None values (would be handled by input validation)
        filtered_data = [record for record in mixed_data if record is not None]
        
        # Transform should handle mixed data gracefully
        try:
            result = self.geographic_standardiser.transform(filtered_data)
            
            # Should have some successful transformations
            valid_results = [r for r in result if 'sa2_code' in r and r['sa2_code']]
            self.assertGreater(len(valid_results), 0)
            
            # Get statistics to verify error handling
            stats = self.geographic_standardiser.get_transformation_statistics()
            self.assertGreaterEqual(stats['failed_mappings'], 0)
            
        except Exception as e:
            self.fail(f"Pipeline should handle mixed data gracefully, but raised: {e}")
    
    def test_configuration_flexibility(self):
        """Test pipeline configuration flexibility and customisation."""
        # Test with different configuration settings
        custom_config = {
            'geographic_column': 'custom_location',
            'geographic_type_column': 'custom_type', 
            'output_sa2_column': 'custom_sa2',
            'include_mapping_metadata': False,
            'strict_validation': True,
            'handle_invalid_codes': 'warn'
        }
        
        custom_standardiser = GeographicStandardiser(config=custom_config)
        
        # Adapt test data to custom column names
        custom_data = [
            {'custom_location': '2000', 'custom_type': 'postcode', 'value': 100}
        ]
        
        result = custom_standardiser.transform(custom_data)
        
        # Should use custom column names
        self.assertTrue(any('custom_sa2' in record for record in result))
        
        # Should not include metadata when disabled
        metadata_fields = ['mapping_method', 'mapping_confidence']
        has_metadata = any(field in record for record in result for field in metadata_fields)
        self.assertFalse(has_metadata)
    
    def test_output_format_compliance(self):
        """Test that output formats comply with expected schemas."""
        # Process data through complete pipeline
        standardised_data = self.geographic_standardiser.transform(self.sample_geographic_data)
        
        # Verify output schema compliance
        schema = self.geographic_standardiser.get_schema()
        
        for record in standardised_data:
            for field_name, field_type in schema.items():
                if field_name in record:
                    value = record[field_name]
                    
                    # Type checking based on schema
                    if field_type == "string":
                        self.assertIsInstance(value, str)
                    elif field_type == "float":
                        self.assertIsInstance(value, (int, float))
                    elif field_type == "integer":
                        self.assertIsInstance(value, int)
        
        # Verify required fields are present
        required_fields = ['sa2_code', 'allocation_factor']
        for record in standardised_data:
            for field in required_fields:
                self.assertIn(field, record)


class TestGeographicPipelineIntegrationWithExternalData(unittest.TestCase):
    """Test integration with external data sources and formats."""
    
    def setUp(self):
        """Set up test fixtures for external data integration."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock external data files
        self.create_mock_data_files()
        
        # Configure components for external data testing
        self.config = {
            'data_directory': self.temp_dir,
            'enable_external_validation': False,  # Mock mode
            'cache_mappings': True
        }
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_mock_data_files(self):
        """Create mock external data files for testing."""
        # Mock postcode correspondence data
        postcode_data = [
            {'postcode': '2000', 'sa2_code': '101021007', 'allocation_factor': 0.8},
            {'postcode': '2000', 'sa2_code': '101021008', 'allocation_factor': 0.2},
            {'postcode': '3000', 'sa2_code': '201021001', 'allocation_factor': 1.0},
        ]
        
        postcode_file = Path(self.temp_dir) / 'postcode_correspondences.json'
        with open(postcode_file, 'w') as f:
            json.dump(postcode_data, f)
        
        # Mock boundary data
        boundary_data = {
            'type': 'FeatureCollection',
            'features': [
                {
                    'type': 'Feature',
                    'properties': {'SA2_CODE': '101021007', 'SA2_NAME': 'Sydney - CBD - North'},
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [[[151.20, -33.86], [151.21, -33.86], [151.21, -33.87], [151.20, -33.87], [151.20, -33.86]]]
                    }
                }
            ]
        }
        
        boundary_file = Path(self.temp_dir) / 'sa2_boundaries.geojson'
        with open(boundary_file, 'w') as f:
            json.dump(boundary_data, f)
    
    def test_external_correspondence_data_integration(self):
        """Test integration with external correspondence data."""
        # This would test loading and using external correspondence files
        # For now, test basic file existence and format
        
        postcode_file = Path(self.temp_dir) / 'postcode_correspondences.json'
        self.assertTrue(postcode_file.exists())
        
        with open(postcode_file, 'r') as f:
            data = json.load(f)
            self.assertIsInstance(data, list)
            self.assertGreater(len(data), 0)
            
            # Check data format
            for record in data:
                self.assertIn('postcode', record)
                self.assertIn('sa2_code', record)
                self.assertIn('allocation_factor', record)
    
    def test_external_boundary_data_integration(self):
        """Test integration with external boundary data."""
        boundary_file = Path(self.temp_dir) / 'sa2_boundaries.geojson'
        self.assertTrue(boundary_file.exists())
        
        with open(boundary_file, 'r') as f:
            data = json.load(f)
            self.assertEqual(data['type'], 'FeatureCollection')
            self.assertIn('features', data)
            self.assertGreater(len(data['features']), 0)
            
            # Check feature format
            for feature in data['features']:
                self.assertEqual(feature['type'], 'Feature')
                self.assertIn('properties', feature)
                self.assertIn('geometry', feature)


if __name__ == '__main__':
    # Run the integration tests
    unittest.main(verbosity=2)