"""Test suite for target schema compliance validation.

This module implements Test-Driven Development for the integrated data schema,
validating complete record structures and Australian health data standards.
"""

import pytest
import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime, date

from src.utils.logging import get_logger
from src.schemas.base import BaseSchemaV1
from src.schemas.sa2_geographic import SA2GeographicSchema
from src.schemas.health_indicators import HealthIndicatorsSchema
from src.schemas.seifa import SEIFASchema

logger = get_logger(__name__)


@dataclass
class MasterHealthRecord:
    """Expected structure for complete integrated health record."""
    
    # Geographic identifiers (mandatory)
    sa2_code: str
    sa2_name: str
    sa3_code: str
    sa3_name: str
    sa4_code: str
    sa4_name: str
    state_code: str
    state_name: str
    
    # Geographic geometry
    geometry: Dict[str, Any]  # GeoJSON format
    centroid_lat: Decimal
    centroid_lon: Decimal
    area_sqkm: Decimal
    
    # Population demographics
    total_population: int
    population_density: Decimal
    median_age: Decimal
    indigenous_population_pct: Decimal
    
    # SEIFA indices (all four indices required)
    seifa_irsad_score: int
    seifa_irsad_decile: int
    seifa_ieo_score: int
    seifa_ieo_decile: int
    seifa_ier_score: int
    seifa_ier_decile: int
    seifa_iod_score: int
    seifa_iod_decile: int
    
    # Health service access
    gp_services_per_1000: Decimal
    specialist_services_per_1000: Decimal
    hospital_beds_per_1000: Decimal
    mental_health_services_count: int
    
    # Health outcomes
    life_expectancy: Decimal
    infant_mortality_rate: Decimal
    preventable_hospitalisations_rate: Decimal
    chronic_disease_prevalence_pct: Decimal
    
    # Pharmaceutical access
    pbs_dispensing_rate_per_1000: Decimal
    high_cost_medicine_access_score: Decimal
    
    # Data lineage and quality
    data_version: str
    last_updated: datetime
    completeness_score: Decimal
    quality_flags: List[str]
    source_datasets: List[str]
    
    # Derived metrics
    health_inequality_index: Decimal
    healthcare_access_index: Decimal
    overall_health_score: Decimal


class TestMasterHealthRecordCreation:
    """Test complete integrated record structure creation."""
    
    def test_master_health_record_creation(self):
        """Test that complete integrated health records can be created with all required fields.
        
        This test will initially FAIL as the ETL pipeline is not yet implemented.
        Success indicates the pipeline can create complete integrated records.
        """
        # This will fail initially - driving implementation
        from src.etl.integration_engine import IntegrationEngine
        
        engine = IntegrationEngine()
        
        # Test with a known SA2 code
        test_sa2_code = "101011007"  # Sydney - Haymarket - The Rocks
        
        # This should create a complete integrated record
        record = engine.create_master_health_record(test_sa2_code)
        
        # Validate all mandatory fields are present
        assert hasattr(record, 'sa2_code')
        assert record.sa2_code == test_sa2_code
        assert hasattr(record, 'sa2_name')
        assert len(record.sa2_name) > 0
        
        # Validate geographic hierarchy
        assert hasattr(record, 'sa3_code')
        assert hasattr(record, 'sa4_code')
        assert hasattr(record, 'state_code')
        assert record.sa2_code.startswith(record.sa3_code[:3])
        
        # Validate geometry is present and valid GeoJSON
        assert hasattr(record, 'geometry')
        assert 'type' in record.geometry
        assert 'coordinates' in record.geometry
        
        # Validate SEIFA indices are complete
        assert hasattr(record, 'seifa_irsad_score')
        assert hasattr(record, 'seifa_ieo_score')
        assert hasattr(record, 'seifa_ier_score')
        assert hasattr(record, 'seifa_iod_score')
        
        # Validate all deciles are between 1-10
        assert 1 <= record.seifa_irsad_decile <= 10
        assert 1 <= record.seifa_ieo_decile <= 10
        assert 1 <= record.seifa_ier_decile <= 10
        assert 1 <= record.seifa_iod_decile <= 10
        
        # Validate health metrics are present
        assert hasattr(record, 'life_expectancy')
        assert record.life_expectancy > 0
        assert hasattr(record, 'gp_services_per_1000')
        assert record.gp_services_per_1000 >= 0
        
        # Validate data lineage
        assert hasattr(record, 'data_version')
        assert hasattr(record, 'completeness_score')
        assert 0 <= record.completeness_score <= 1
        assert hasattr(record, 'source_datasets')
        assert len(record.source_datasets) > 0
    
    def test_record_schema_validation(self):
        """Test that created records conform to schema validation."""
        from src.etl.integration_engine import IntegrationEngine
        from src.testing.target_validation import TargetSchemaValidator
        
        engine = IntegrationEngine()
        validator = TargetSchemaValidator()
        
        # Create record for validation
        test_sa2_code = "101011007"
        record = engine.create_master_health_record(test_sa2_code)
        
        # Validate against target schema
        validation_result = validator.validate_master_record(record)
        
        assert validation_result.is_valid
        assert len(validation_result.errors) == 0
        assert validation_result.completeness_score >= 0.95  # 95% completeness required
    
    def test_missing_data_handling(self):
        """Test graceful handling of missing data in integrated records."""
        from src.etl.integration_engine import IntegrationEngine
        
        engine = IntegrationEngine()
        
        # Test with SA2 that might have missing data
        test_sa2_code = "999999999"  # Non-existent SA2
        
        with pytest.raises(ValueError, match="SA2 not found"):
            engine.create_master_health_record(test_sa2_code)
    
    def test_data_type_enforcement(self):
        """Test that all fields have correct data types."""
        from src.etl.integration_engine import IntegrationEngine
        
        engine = IntegrationEngine()
        test_sa2_code = "101011007"
        record = engine.create_master_health_record(test_sa2_code)
        
        # Test numeric types
        assert isinstance(record.total_population, int)
        assert isinstance(record.population_density, Decimal)
        assert isinstance(record.seifa_irsad_score, int)
        assert isinstance(record.life_expectancy, Decimal)
        
        # Test string types
        assert isinstance(record.sa2_code, str)
        assert isinstance(record.sa2_name, str)
        assert isinstance(record.data_version, str)
        
        # Test datetime types
        assert isinstance(record.last_updated, datetime)
        
        # Test list types
        assert isinstance(record.quality_flags, list)
        assert isinstance(record.source_datasets, list)


class TestSA2HealthProfileValidation:
    """Test SA2-level health profile completeness."""
    
    def test_sa2_health_profile_validation(self):
        """Test complete SA2 health profile validation.
        
        This validates that each SA2 has a complete health profile with
        all required indicators and meets Australian health data standards.
        """
        from src.etl.profile_generator import SA2HealthProfileGenerator
        
        generator = SA2HealthProfileGenerator()
        
        # Generate profile for test SA2
        test_sa2_code = "101011007"
        profile = generator.generate_health_profile(test_sa2_code)
        
        # Validate core health indicators are present
        required_indicators = [
            'life_expectancy',
            'infant_mortality_rate',
            'preventable_hospitalisations_rate',
            'chronic_disease_prevalence_pct',
            'gp_services_per_1000',
            'specialist_services_per_1000',
            'mental_health_services_count'
        ]
        
        for indicator in required_indicators:
            assert hasattr(profile, indicator), f"Missing indicator: {indicator}"
            value = getattr(profile, indicator)
            assert value is not None, f"Null value for indicator: {indicator}"
            assert value >= 0, f"Negative value for indicator: {indicator}"
        
        # Validate derived health scores
        assert hasattr(profile, 'health_inequality_index')
        assert 0 <= profile.health_inequality_index <= 1
        
        assert hasattr(profile, 'healthcare_access_index')
        assert 0 <= profile.healthcare_access_index <= 1
        
        assert hasattr(profile, 'overall_health_score')
        assert 0 <= profile.overall_health_score <= 100
    
    def test_profile_completeness_scoring(self):
        """Test that profile completeness is accurately calculated."""
        from src.etl.profile_generator import SA2HealthProfileGenerator
        from src.testing.target_validation import QualityStandardsChecker
        
        generator = SA2HealthProfileGenerator()
        checker = QualityStandardsChecker()
        
        test_sa2_code = "101011007"
        profile = generator.generate_health_profile(test_sa2_code)
        
        completeness = checker.calculate_profile_completeness(profile)
        
        # Require minimum 90% completeness for production
        assert completeness >= 0.90
        
        # Validate completeness calculation logic
        assert isinstance(completeness, Decimal)
        assert 0 <= completeness <= 1
    
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
