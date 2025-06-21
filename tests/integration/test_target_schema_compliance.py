"""Test suite for target schema compliance validation.

This module implements Test-Driven Development for the integrated data schema,
validating complete record structures and Australian health data standards.
"""

import pytest
import json
from pathlib import Path
from typing import Dict, Any, List
from decimal import Decimal
from datetime import datetime, date
from pydantic import ValidationError

from src.utils.logging import get_logger
from schemas.integrated_schema import MasterHealthRecord, SA2HealthProfile, HealthIndicatorSummary, DataIntegrationLevel
from schemas.base_schema import VersionedSchema, GeographicBoundary, TemporalData
from schemas.sa2_schema import SA2Coordinates
from schemas.health_schema import HealthIndicator
from schemas.seifa_schema import SEIFAScore, SEIFAIndexType

logger = get_logger(__name__)


# NOTE: Using actual Pydantic schema from schemas.integrated_schema
# This ensures TDD tests validate against real production schema structure


class TestMasterHealthRecordCreation:
    """Test complete integrated record structure creation."""
    
    def test_master_health_record_creation(self):
        """Test that MasterHealthRecord can be created and validated with Pydantic schema.
        
        This test validates that the actual Pydantic MasterHealthRecord schema 
        works correctly and ensures TDD alignment with production schema.
        """
        
        # Test with realistic SA2 data that conforms to actual schema
        test_sa2_code = "101011007"  # Sydney - Haymarket - The Rocks
        
        # Create test data that matches actual Pydantic schema structure
        sample_boundary = GeographicBoundary(
            boundary_id=test_sa2_code,
            boundary_type="SA2",
            name="Sydney - Haymarket - The Rocks",
            state="NSW",
            geometry={"type": "Polygon", "coordinates": [[[151.2073, -33.8688], [151.2093, -33.8688], [151.2093, -33.8668], [151.2073, -33.8668], [151.2073, -33.8688]]]},
            centroid_lat=-33.8688,
            centroid_lon=151.2093,
            area_sq_km=2.54
        )
        
        sample_record_data = {
            # Required fields matching actual schema
            'sa2_code': test_sa2_code,
            'sa2_name': 'Sydney - Haymarket - The Rocks',
            'geographic_hierarchy': {
                'sa3_code': '10101',
                'sa4_code': '101', 
                'state_code': 'NSW'
            },
            'boundary_data': sample_boundary,
            'urbanisation': 'major_urban',
            'remoteness_category': 'Major Cities',
            'demographic_profile': {
                'age_groups': {'0-14': 523, '15-64': 4356, '65+': 553},
                'sex_distribution': {'male': 2687, 'female': 2745}
            },
            'total_population': 5432,
            'population_density_per_sq_km': 2138.6,
            'seifa_scores': {
                SEIFAIndexType.IRSD: 1156.0,
                SEIFAIndexType.IRSAD: 1098.0,
                SEIFAIndexType.IER: 1023.0,
                SEIFAIndexType.IEO: 1134.0
            },
            'seifa_deciles': {
                SEIFAIndexType.IRSD: 8,
                SEIFAIndexType.IRSAD: 7,
                SEIFAIndexType.IER: 6,
                SEIFAIndexType.IEO: 8
            },
            'disadvantage_category': 'Advantaged',
            'health_outcomes_summary': {'life_expectancy': 84.2, 'self_assessed_health_good': 78.5},
            'integration_level': DataIntegrationLevel.STANDARD,
            'data_completeness_score': 89.5,
            'source_datasets': ['ABS_Census_2021', 'SEIFA_2021', 'AIHW_Health_Data'],
            'schema_version': '2.0.0'
        }
        
        # Test Pydantic model creation and validation
        try:
            record = MasterHealthRecord.model_validate(sample_record_data)
            assert isinstance(record, MasterHealthRecord), f"Expected MasterHealthRecord, got {type(record)}"
        except ValidationError as e:
            pytest.fail(f"Failed to create MasterHealthRecord with sample data: {e}")
        
        # Test Pydantic validation by accessing required fields
        assert record.sa2_code == test_sa2_code
        assert len(record.sa2_name) > 0
        
        # Validate geographic hierarchy structure using Pydantic model
        assert 'sa3_code' in record.geographic_hierarchy
        assert 'sa4_code' in record.geographic_hierarchy  
        assert 'state_code' in record.geographic_hierarchy
        
        # Validate geometry through boundary_data
        assert record.boundary_data is not None
        assert record.boundary_data.geometry is not None
        
        # Validate SEIFA indices using proper Pydantic fields
        for seifa_type in SEIFAIndexType:
            assert seifa_type in record.seifa_scores
            assert seifa_type in record.seifa_deciles
            assert 1 <= record.seifa_deciles[seifa_type] <= 10
        
        # Validate health metrics - these are Optional fields in the schema
        if record.life_expectancy:
            assert all(le > 0 for le in record.life_expectancy.values())
        
        if record.gp_services_per_1000:
            assert record.gp_services_per_1000 >= 0
        
        # Validate data integration metadata
        assert record.integration_level == DataIntegrationLevel.STANDARD
        assert 0 <= record.data_completeness_score <= 100
        assert len(record.source_datasets) > 0
        
        # Test Pydantic validation works
        try:
            record.model_validate(record.model_dump())
        except ValidationError as e:
            pytest.fail(f"Record failed Pydantic validation: {e}")
    
    def test_record_schema_validation(self):
        """Test that Pydantic schema validation catches invalid data."""
        # Test invalid SA2 code (wrong length)
        invalid_data = {
            'sa2_code': '12345',  # Too short
            'sa2_name': 'Test SA2',
            'geographic_hierarchy': {'sa3_code': '10101', 'sa4_code': '101', 'state_code': 'NSW'},
            'boundary_data': GeographicBoundary(
                boundary_id='12345',
                boundary_type='SA2',
                name='Test SA2',
                state='NSW',
                geometry={"type": "Point", "coordinates": [151.2093, -33.8688]},
                centroid_lat=-33.8688,
                centroid_lon=151.2093,
                area_sq_km=2.54
            ),
            'urbanisation': 'major_urban',
            'remoteness_category': 'Major Cities',
            'demographic_profile': {'age_groups': {}, 'sex_distribution': {}},
            'total_population': 5432,
            'population_density_per_sq_km': 2138.6,
            'seifa_scores': {SEIFAIndexType.IRSD: 1156.0},
            'seifa_deciles': {SEIFAIndexType.IRSD: 8},
            'disadvantage_category': 'Advantaged',
            'health_outcomes_summary': {},
            'integration_level': DataIntegrationLevel.MINIMAL,
            'data_completeness_score': 50.0,
            'source_datasets': ['test'],
            'schema_version': '2.0.0'
        }
        
        # Test that Pydantic validation catches the invalid SA2 code
        with pytest.raises(ValidationError, match="pattern"):
            MasterHealthRecord.model_validate(invalid_data)
        
        # Test valid data passes validation
        valid_data = invalid_data.copy()
        valid_data['sa2_code'] = '101011007'  # Valid 9-digit code
        
        try:
            record = MasterHealthRecord.model_validate(valid_data)
            
            # Test schema-specific validation methods exist
            integrity_errors = record.validate_data_integrity()
            assert isinstance(integrity_errors, list), "validate_data_integrity should return a list"
            
            # Test required completeness thresholds can be checked
            assert isinstance(record.data_completeness_score, float)
            assert 0 <= record.data_completeness_score <= 100
            
        except ValidationError as e:
            pytest.fail(f"Valid data failed Pydantic schema validation: {e}")
    
    def test_missing_data_handling(self):
        """Test that Pydantic schema handles optional fields correctly."""
        # Test minimal record with only required fields
        minimal_data = {
            'sa2_code': '999999999',
            'sa2_name': 'Minimal Test SA2',
            'geographic_hierarchy': {'sa3_code': '99999', 'sa4_code': '999', 'state_code': 'NSW'},
            'boundary_data': GeographicBoundary(
                boundary_id='999999999',
                boundary_type='SA2',
                name='Minimal Test SA2',
                state='NSW',
                geometry={"type": "Point", "coordinates": [151.0, -33.0]},
                centroid_lat=-33.0,
                centroid_lon=151.0,
                area_sq_km=1.0
            ),
            'urbanisation': 'major_urban',
            'remoteness_category': 'Major Cities',
            'demographic_profile': {'age_groups': {}, 'sex_distribution': {}},
            'total_population': 1000,
            'population_density_per_sq_km': 1000.0,
            'seifa_scores': {SEIFAIndexType.IRSD: 1000.0},
            'seifa_deciles': {SEIFAIndexType.IRSD: 5},
            'disadvantage_category': 'Average',
            'health_outcomes_summary': {},
            'integration_level': DataIntegrationLevel.MINIMAL,
            'data_completeness_score': 30.0,
            'source_datasets': ['minimal'],
            'schema_version': '2.0.0'
        }
        
        # Test that minimal data with optional fields missing still validates
        try:
            record = MasterHealthRecord.model_validate(minimal_data)
            
            # Test that optional fields are None when not provided
            assert record.median_age is None
            assert record.life_expectancy is None
            assert record.gp_services_per_1000 is None
            
            # Test that missing_indicators tracks what's missing
            assert isinstance(record.missing_indicators, list)
            
        except ValidationError as e:
            pytest.fail(f"Minimal valid data failed validation: {e}")
    
    def test_data_type_enforcement(self):
        """Test that Pydantic schema enforces correct data types."""
        # Test that Pydantic rejects completely wrong types
        invalid_types_data = {
            'sa2_code': 101011007,  # Should be string, not int
            'sa2_name': 123,  # Should be string, not int
            'geographic_hierarchy': {'sa3_code': '10101', 'sa4_code': '101', 'state_code': 'NSW'},
            'boundary_data': GeographicBoundary(
                boundary_id='101011007',
                boundary_type='SA2',
                name='Type Test SA2',
                state='NSW',
                geometry={"type": "Point", "coordinates": [151.0, -33.0]},
                centroid_lat=-33.0,
                centroid_lon=151.0,
                area_sq_km=1.0
            ),
            'urbanisation': 'major_urban',
            'remoteness_category': 'Major Cities',
            'demographic_profile': {'age_groups': {}, 'sex_distribution': {}},
            'total_population': "5432",  # Should be int, providing string
            'population_density_per_sq_km': "5432.0",  # Should be float, consistent with population/area
            'seifa_scores': {SEIFAIndexType.IRSD: "1156"},  # Should be float, providing string
            'seifa_deciles': {SEIFAIndexType.IRSD: "8"},  # Should be int, providing string
            'disadvantage_category': 'Advantaged',
            'health_outcomes_summary': {},
            'integration_level': DataIntegrationLevel.STANDARD,
            'data_completeness_score': "89.5",  # Should be float, providing string
            'source_datasets': ['test'],
            'schema_version': '2.0.0'
        }
        
        # Test that Pydantic properly rejects invalid types
        with pytest.raises(ValidationError):
            MasterHealthRecord.model_validate(invalid_types_data)
        
        # Test with correct types that should work
        correct_types_data = invalid_types_data.copy()
        correct_types_data['sa2_code'] = '101011007'  # Fix to string
        correct_types_data['sa2_name'] = 'Type Test SA2'  # Fix to string
        
        try:
            record = MasterHealthRecord.model_validate(correct_types_data)
            
            # Verify types are correct
            assert isinstance(record.sa2_code, str)
            assert isinstance(record.sa2_name, str)
            assert isinstance(record.total_population, int)  # String was coerced to int
            assert isinstance(record.population_density_per_sq_km, float)  # String was coerced to float
            
            # Test SEIFA data structure and types
            assert isinstance(record.seifa_scores, dict)
            assert isinstance(record.seifa_deciles, dict)
            for seifa_type in record.seifa_scores:
                assert isinstance(record.seifa_scores[seifa_type], float)
                assert isinstance(record.seifa_deciles[seifa_type], int)
            
            # Test metadata types
            assert record.integration_level == DataIntegrationLevel.STANDARD
            assert isinstance(record.data_completeness_score, float)
            assert isinstance(record.source_datasets, list)
            assert isinstance(record.missing_indicators, list)
            
            # Test serialization/deserialization round-trip preserves types
            serialized = record.model_dump()
            deserialized = MasterHealthRecord.model_validate(serialized)
            assert type(deserialized.sa2_code) == type(record.sa2_code)
            assert type(deserialized.total_population) == type(record.total_population)
            
        except ValidationError as e:
            pytest.fail(f"Valid data with correct types failed validation: {e}")


class TestSA2HealthProfileValidation:
    """Test SA2-level health profile completeness."""
    
    def test_sa2_health_profile_validation(self):
        """Test complete SA2 health profile validation using actual Pydantic schema.
        
        This validates that each SA2 has a complete health profile conforming
        to the SA2HealthProfile Pydantic schema.
        """
        from src.etl.profile_generator import SA2HealthProfileGenerator
        
        generator = SA2HealthProfileGenerator()
        
        # Generate profile for test SA2
        test_sa2_code = "101011007"
        profile = generator.generate_health_profile(test_sa2_code)
        
        # Validate profile is instance of actual Pydantic schema
        assert isinstance(profile, SA2HealthProfile), f"Expected SA2HealthProfile, got {type(profile)}"
        
        # Test Pydantic validation
        try:
            validated_profile = SA2HealthProfile.model_validate(profile.model_dump())
            assert isinstance(validated_profile, SA2HealthProfile)
        except ValidationError as e:
            pytest.fail(f"SA2HealthProfile failed Pydantic validation: {e}")
        
        # Validate required fields according to actual schema
        assert profile.sa2_code == test_sa2_code
        assert len(profile.sa2_name) > 0
        assert profile.total_population > 0
        assert 1 <= profile.seifa_disadvantage_decile <= 10
        
        # Validate health indicators using Optional field pattern from schema
        if profile.life_expectancy_male is not None:
            assert profile.life_expectancy_male > 0
        if profile.life_expectancy_female is not None:
            assert profile.life_expectancy_female > 0
        
        if profile.gp_visits_per_capita is not None:
            assert profile.gp_visits_per_capita >= 0
        
        if profile.specialist_visits_per_capita is not None:
            assert profile.specialist_visits_per_capita >= 0
        
        # Validate percentage fields are within bounds
        percentage_fields = [
            'excellent_very_good_health_percentage',
            'current_smoking_percentage', 
            'obesity_percentage',
            'bulk_billing_percentage'
        ]
        
        for field_name in percentage_fields:
            value = getattr(profile, field_name, None)
            if value is not None:
                assert 0 <= value <= 100, f"{field_name} {value} outside valid percentage range"
        
        # Validate completeness score
        assert 0 <= profile.profile_completeness_score <= 100
        
        # Test data integrity validation
        integrity_errors = profile.validate_data_integrity()
        assert len(integrity_errors) == 0, f"Data integrity errors: {integrity_errors}"
    
    def test_profile_completeness_scoring(self):
        """Test that profile completeness is accurately calculated using Pydantic schema."""
        from src.etl.profile_generator import SA2HealthProfileGenerator
        
        generator = SA2HealthProfileGenerator()
        
        test_sa2_code = "101011007"
        profile = generator.generate_health_profile(test_sa2_code)
        
        # Use Pydantic schema's completeness field
        completeness = profile.profile_completeness_score
        
        # Require minimum 90% completeness for production
        assert completeness >= 90.0, f"Completeness score {completeness} below required 90%"
        
        # Validate completeness calculation logic according to schema
        assert isinstance(completeness, float)
        assert 0 <= completeness <= 100
        
        # Test that completeness correlates with available data
        non_null_fields = 0
        total_optional_fields = 0
        
        # Count optional health indicator fields that have values
        optional_health_fields = [
            'life_expectancy_male', 'life_expectancy_female',
            'all_cause_mortality_rate', 'diabetes_prevalence',
            'current_smoking_percentage', 'obesity_percentage',
            'gp_visits_per_capita', 'bulk_billing_percentage'
        ]
        
        for field_name in optional_health_fields:
            total_optional_fields += 1
            if getattr(profile, field_name, None) is not None:
                non_null_fields += 1
        
        # Completeness should roughly correlate with available fields
        expected_completeness_rough = (non_null_fields / total_optional_fields) * 100
        # Allow some tolerance for different calculation methods
        assert abs(completeness - expected_completeness_rough) <= 30, \
            f"Completeness {completeness} significantly different from field availability {expected_completeness_rough}"
    
    def test_profile_australian_standards_compliance(self):
        """Test compliance with Australian health data standards."""
        from src.etl.profile_generator import SA2HealthProfileGenerator
        from src.testing.target_validation import ComplianceReporter
        
        generator = SA2HealthProfileGenerator()
        compliance_reporter = ComplianceReporter()
        
        test_sa2_code = "101011007"
        profile = generator.generate_health_profile(test_sa2_code)
        
        # Check AIHW compliance
        aihw_compliance = compliance_reporter.check_aihw_compliance(profile)
        assert aihw_compliance.is_compliant
        assert len(aihw_compliance.violations) == 0
        
        # Check ABS compliance
        abs_compliance = compliance_reporter.check_abs_compliance(profile)
        assert abs_compliance.is_compliant
        assert len(abs_compliance.violations) == 0


class TestGeographicHierarchyConsistency:
    """Test geographic relationships and hierarchy validation."""
    
    def test_geographic_hierarchy_consistency(self):
        """Test that geographic hierarchy relationships are consistent.
        
        Validates SA1→SA2→SA3→SA4→State relationships are maintained
        throughout the integrated dataset.
        """
        from src.etl.geographic_validator import GeographicHierarchyValidator
        
        validator = GeographicHierarchyValidator()
        
        # Test all SA2s have valid hierarchy
        hierarchy_validation = validator.validate_all_hierarchies()
        
        assert hierarchy_validation.is_valid
        assert len(hierarchy_validation.orphaned_sa2s) == 0
        assert len(hierarchy_validation.invalid_codes) == 0
        
        # Test specific hierarchy chain
        test_sa2_code = "101011007"
        chain_validation = validator.validate_hierarchy_chain(test_sa2_code)
        
        assert chain_validation.is_valid
        assert chain_validation.sa2_code == test_sa2_code
        assert len(chain_validation.sa3_code) == 5
        assert len(chain_validation.sa4_code) == 3
        assert len(chain_validation.state_code) == 1
        
        # Validate code relationships
        assert test_sa2_code.startswith(chain_validation.sa3_code[:3])
        assert chain_validation.sa3_code.startswith(chain_validation.sa4_code)
    
    def test_geographic_boundaries_integrity(self):
        """Test that geographic boundaries are spatially consistent."""
        from src.etl.geographic_validator import GeographicHierarchyValidator
        
        validator = GeographicHierarchyValidator()
        
        # Test boundary containment
        test_sa2_code = "101011007"
        boundary_validation = validator.validate_boundary_containment(test_sa2_code)
        
        assert boundary_validation.is_contained_in_sa3
        assert boundary_validation.is_contained_in_sa4
        assert boundary_validation.is_contained_in_state
        
        # Test no boundary overlaps
        overlap_validation = validator.check_boundary_overlaps()
        assert len(overlap_validation.overlapping_pairs) == 0
    
    def test_centroid_calculation_accuracy(self):
        """Test that calculated centroids fall within SA2 boundaries."""
        from src.etl.geographic_validator import GeographicHierarchyValidator
        
        validator = GeographicHierarchyValidator()
        
        test_sa2_code = "101011007"
        centroid_validation = validator.validate_centroid_accuracy(test_sa2_code)
        
        assert centroid_validation.centroid_within_boundary
        assert centroid_validation.calculation_method == "geometric"
        assert -90 <= centroid_validation.latitude <= 90
        assert -180 <= centroid_validation.longitude <= 180


class TestAustralianHealthStandardsCompliance:
    """Test compliance with AIHW and ABS standards."""
    
    def test_australian_health_standards_compliance(self):
        """Test adherence to Australian Institute of Health and Welfare standards.
        
        Validates that all health indicators follow AIHW definitions,
        classification standards, and reporting requirements.
        """
        from src.testing.target_validation import ComplianceReporter
        from src.etl.integration_engine import IntegrationEngine
        
        compliance_reporter = ComplianceReporter()
        engine = IntegrationEngine()
        
        # Test sample of SA2s for compliance
        test_sa2_codes = ["101011007", "201011021", "301011001"]
        
        for sa2_code in test_sa2_codes:
            record = engine.create_master_health_record(sa2_code)
            
            # AIHW standards compliance
            aihw_report = compliance_reporter.generate_aihw_compliance_report(record)
            assert aihw_report.overall_compliance_score >= 0.95
            assert len(aihw_report.critical_violations) == 0
            
            # ABS standards compliance  
            abs_report = compliance_reporter.generate_abs_compliance_report(record)
            assert abs_report.overall_compliance_score >= 0.95
            assert len(abs_report.critical_violations) == 0
    
    def test_data_classification_standards(self):
        """Test that data follows Australian data classification standards."""
        from src.testing.target_validation import ComplianceReporter
        
        compliance_reporter = ComplianceReporter()
        
        # Test classification compliance
        classification_report = compliance_reporter.validate_data_classifications()
        
        # All health data should be classified appropriately
        assert classification_report.unclassified_fields == 0
        assert classification_report.privacy_compliant
        assert classification_report.meets_aihw_standards
    
    def test_metadata_standards_compliance(self):
        """Test that metadata follows Australian government standards."""
        from src.testing.target_validation import ComplianceReporter
        
        compliance_reporter = ComplianceReporter()
        
        metadata_report = compliance_reporter.validate_metadata_standards()
        
        # Metadata should include all required elements
        required_metadata = [
            'data_custodian',
            'collection_method', 
            'reference_period',
            'geographic_coverage',
            'data_quality_statement',
            'privacy_classification'
        ]
        
        for element in required_metadata:
            assert element in metadata_report.present_elements
        
        assert metadata_report.dublin_core_compliant
        assert metadata_report.agls_compliant  # Australian Government Locator Service
