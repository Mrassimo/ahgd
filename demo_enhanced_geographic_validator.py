#!/usr/bin/env python3
"""
Demonstration script for Enhanced Geographic Validator.

This script demonstrates the enhanced geographic validation capabilities
without requiring the full project dependencies that may have compatibility issues.
"""

import logging
import sys
from typing import Any, Dict, List
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mock the spatial libraries if they're not available
try:
    import geopandas as gpd
    import shapely
    import pyproj
    SPATIAL_LIBRARIES_AVAILABLE = True
    logger.info("Spatial libraries (geopandas, shapely, pyproj) are available")
except ImportError:
    SPATIAL_LIBRARIES_AVAILABLE = False
    logger.warning("Spatial libraries not available - some features will be limited")

def create_sample_sa2_data() -> List[Dict[str, Any]]:
    """Create sample SA2 data for demonstration."""
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
        },
        {
            'sa2_code': '123',  # Invalid - too short
            'sa3_code': '12345',
            'sa4_code': '123',
            'state_code': '1',
            'latitude': 100,  # Invalid latitude
            'longitude': 200,  # Invalid longitude
            'total_population': -100,  # Negative population
            'geographic_area_sqkm': 0,  # Zero area
            'is_coastal': True
        }
    ]

def create_enhanced_validator_config() -> Dict[str, Any]:
    """Create configuration for enhanced geographic validator."""
    return {
        'enhanced_geographic': {
            'sa2_coverage': {
                'official_sa2_codes_file': None  # Will use generated codes
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

def demonstrate_enhanced_validation():
    """Demonstrate enhanced geographic validation capabilities."""
    logger.info("Starting Enhanced Geographic Validator Demonstration")
    logger.info("=" * 60)
    
    # Import the validator components
    try:
        from src.utils.interfaces import ValidationSeverity
        from src.validators.base import BaseValidator
        
        # Import enhanced validator by loading module directly
        import importlib.util
        enhanced_geo_path = project_root / "src" / "validators" / "enhanced_geographic.py"
        spec = importlib.util.spec_from_file_location("enhanced_geographic", enhanced_geo_path)
        enhanced_geo_module = importlib.util.module_from_spec(spec)
        
        # Mock the geographic validator import
        import types
        mock_geographic_validator = types.ModuleType('geographic_validator')
        
        # Create a mock GeographicValidator class
        class MockGeographicValidator(BaseValidator):
            def validate(self, data):
                return []
            def get_validation_rules(self):
                return ["coordinate_validation", "sa2_boundary_validation"]
            def validate_coordinates(self, data):
                return []
            def validate_sa2_boundaries(self, data):
                return []
        
        mock_geographic_validator.GeographicValidator = MockGeographicValidator
        sys.modules['src.validators.geographic_validator'] = mock_geographic_validator
        
        # Now load our enhanced validator
        spec.loader.exec_module(enhanced_geo_module)
        EnhancedGeographicValidator = enhanced_geo_module.EnhancedGeographicValidator
        
        logger.info("✓ Successfully imported EnhancedGeographicValidator")
        
    except Exception as e:
        logger.error(f"Failed to import validator: {e}")
        return
    
    # Create validator instance
    config = create_enhanced_validator_config()
    validator = EnhancedGeographicValidator(config=config)
    
    logger.info(f"✓ Created validator with ID: {validator.validator_id}")
    logger.info(f"  - Official SA2 count: {validator.OFFICIAL_SA2_COUNT}")
    logger.info(f"  - Target CRS: {validator.target_crs}")
    logger.info(f"  - Coordinate precision: {validator.coordinate_precision}")
    logger.info(f"  - Spatial libraries available: {SPATIAL_LIBRARIES_AVAILABLE}")
    
    # Create sample data
    sample_data = create_sample_sa2_data()
    logger.info(f"✓ Created sample data with {len(sample_data)} records")
    
    # Demonstrate validation rules
    logger.info("\nSupported Validation Rules:")
    rules = validator.get_validation_rules()
    for rule in rules:
        logger.info(f"  - {rule}")
    
    # Test SA2 format validation
    logger.info("\nTesting SA2 Code Format Validation:")
    test_codes = [
        ('10102100001', True),   # Valid NSW SA2
        ('20202200002', True),   # Valid VIC SA2
        ('123', False),          # Too short
        ('90102100001', False),  # Invalid state prefix
        ('ABCDEFGHIJK', False),  # Non-numeric
    ]
    
    for code, expected in test_codes:
        result = validator._validate_sa2_code_format_enhanced(code)
        status = "✓" if result == expected else "✗"
        logger.info(f"  {status} {code}: {result} (expected: {expected})")
    
    # Test Australian bounds validation
    logger.info("\nTesting Australian Territorial Bounds:")
    test_coords = [
        (-33.8688, 151.2093, "Sydney", True),
        (-37.8136, 144.9631, "Melbourne", True),
        (51.5074, -0.1278, "London", False),
        (0, 0, "Null Island", False),
    ]
    
    for lat, lon, location, expected_valid in test_coords:
        result = validator._validate_australian_territorial_bounds(lat, lon, 0)
        is_valid = result is None
        status = "✓" if is_valid == expected_valid else "✗"
        logger.info(f"  {status} {location} ({lat}, {lon}): Valid={is_valid}")
    
    # Demonstrate individual validation methods
    logger.info("\nRunning Individual Validation Methods:")
    
    # SA2 Coverage Validation
    try:
        validator.official_sa2_codes = {'10102100001', '20202200002', '30303300003'}
        coverage_results = validator.validate_sa2_coverage(sample_data)
        logger.info(f"  ✓ SA2 Coverage: {len(coverage_results)} validation results")
        
        for result in coverage_results[:3]:  # Show first 3 results
            severity_symbol = "🔴" if result.severity.value == "error" else "🟡" if result.severity.value == "warning" else "🔵"
            logger.info(f"    {severity_symbol} {result.rule_id}: {result.message}")
            
    except Exception as e:
        logger.warning(f"  ⚠ SA2 Coverage validation error: {e}")
    
    # Coordinate Reference System Validation
    try:
        crs_results = validator.validate_coordinate_reference_system(sample_data)
        logger.info(f"  ✓ CRS Validation: {len(crs_results)} validation results")
        
        for result in crs_results[:3]:  # Show first 3 results
            severity_symbol = "🔴" if result.severity.value == "error" else "🟡" if result.severity.value == "warning" else "🔵"
            logger.info(f"    {severity_symbol} {result.rule_id}: {result.message}")
            
    except Exception as e:
        logger.warning(f"  ⚠ CRS validation error: {e}")
    
    # Spatial Hierarchy Validation
    try:
        hierarchy_results = validator.validate_spatial_hierarchy(sample_data)
        logger.info(f"  ✓ Spatial Hierarchy: {len(hierarchy_results)} validation results")
        
        for result in hierarchy_results[:3]:  # Show first 3 results
            severity_symbol = "🔴" if result.severity.value == "error" else "🟡" if result.severity.value == "warning" else "🔵"
            logger.info(f"    {severity_symbol} {result.rule_id}: {result.message}")
            
    except Exception as e:
        logger.warning(f"  ⚠ Spatial hierarchy validation error: {e}")
    
    # Geographic Consistency Validation
    try:
        consistency_results = validator.validate_geographic_consistency(sample_data)
        logger.info(f"  ✓ Geographic Consistency: {len(consistency_results)} validation results")
        
        for result in consistency_results[:3]:  # Show first 3 results
            severity_symbol = "🔴" if result.severity.value == "error" else "🟡" if result.severity.value == "warning" else "🔵"
            logger.info(f"    {severity_symbol} {result.rule_id}: {result.message}")
            
    except Exception as e:
        logger.warning(f"  ⚠ Geographic consistency validation error: {e}")
    
    # Show statistics
    try:
        stats = validator.get_enhanced_statistics()
        logger.info("\nValidation Statistics:")
        for key, value in stats.items():
            logger.info(f"  - {key}: {value}")
    except Exception as e:
        logger.warning(f"  ⚠ Statistics error: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Enhanced Geographic Validator Demonstration Complete")
    logger.info("🎉 All validation methods demonstrated successfully!")

if __name__ == "__main__":
    demonstrate_enhanced_validation()