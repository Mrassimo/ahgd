"""
Backwards TDD tests for data extractors.

This module contains Test-Driven Development tests that validate extractors
produce data compatible with target schema requirements. Tests work backwards
from target schemas to ensure successful integration.
"""

import pytest
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch

# Import extractors
from src.extractors.aihw_extractor import (
    AIHWMortalityExtractor,
    AIHWHospitalisationExtractor,
    AIHWHealthIndicatorExtractor,
    AIHWMedicareExtractor,
)
from src.extractors.abs_extractor import (
    ABSGeographicExtractor,
    ABSCensusExtractor,
    ABSSEIFAExtractor,
    ABSPostcodeExtractor,
)
from src.extractors.bom_extractor import (
    BOMClimateExtractor,
    BOMWeatherStationExtractor,
    BOMEnvironmentalExtractor,
)
from src.extractors.medicare_pbs_extractor import (
    MedicareUtilisationExtractor,
    PBSPrescriptionExtractor,
    HealthcareServicesExtractor,
)

# Import registry system
from src.extractors.extractor_registry import (
    ExtractorRegistry,
    ExtractorFactory,
    ExtractorValidator,
    ExtractorType,
    DataCategory,
)

# Import target schemas for validation
from src.schemas.health_schema import HealthIndicatorType, AgeGroupType
from src.schemas.seifa_schema import SEIFAIndexType
from src.schemas.integrated_schema import UrbanRuralClassification

# Import interfaces
from src.utils.interfaces import ExtractionError


class TestTargetSchemaCompatibility:
    """
    Test that extractors produce data compatible with target schemas.
    
    These tests work backwards from target schema requirements to ensure
    successful data integration and pipeline operation.
    """
    
    @pytest.fixture
    def sample_configs(self):
        """Sample configurations for extractors."""
        return {
            'aihw_mortality': {
                'aihw_base_url': 'https://demo.aihw.gov.au',
                'api_key': 'test_key',
                'batch_size': 100,
                'max_retries': 2,
            },
            'abs_geographic': {
                'abs_base_url': 'https://demo.abs.gov.au',
                'coordinate_system': 'GDA2020',
                'batch_size': 50,
            },
            'bom_climate': {
                'bom_base_url': 'http://demo.bom.gov.au',
                'batch_size': 200,
            },
            'medicare_utilisation': {
                'medicare_base_url': 'https://demo.data.gov.au',
                'min_cell_size': 5,
                'batch_size': 100,
            },
        }
    
    def test_aihw_mortality_produces_target_mortality_fields(self, sample_configs):
        """Test AIHW mortality extractor produces fields required by MortalityData schema."""
        extractor = AIHWMortalityExtractor(sample_configs['aihw_mortality'])
        
        # Extract demo data
        batches = list(extractor.extract('demo'))
        assert len(batches) > 0, "Should produce at least one batch"
        
        records = batches[0]
        assert len(records) > 0, "Should produce at least one record"
        
        # Validate target schema compatibility
        sample_record = records[0]
        
        # Required fields for MortalityData schema
        required_fields = [
            'geographic_id',
            'geographic_level', 
            'indicator_name',
            'indicator_code',
            'indicator_type',
            'value',
            'unit',
            'cause_of_death',
            'data_source_id',
            'extraction_timestamp',
        ]
        
        for field in required_fields:
            assert field in sample_record, f"Missing required field: {field}"
        
        # Validate field formats and types
        assert isinstance(sample_record['geographic_id'], str), "geographic_id should be string"
        assert len(sample_record['geographic_id']) == 9, "SA2 code should be 9 digits"
        assert sample_record['geographic_id'].isdigit(), "SA2 code should be numeric"
        
        assert sample_record['geographic_level'] == 'SA2', "Should be SA2 level"
        assert sample_record['indicator_type'] == HealthIndicatorType.MORTALITY.value
        
        assert isinstance(sample_record['value'], (int, float)), "value should be numeric"
        assert sample_record['value'] >= 0, "mortality rate should be non-negative"
        
        # Validate data source identification
        assert 'AIHW' in sample_record['data_source_id'], "Should identify AIHW as source"
        
        # Validate timestamp format
        timestamp_str = sample_record['extraction_timestamp']
        datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))  # Should not raise
    
    def test_aihw_mortality_compatible_with_sa2_health_profile(self, sample_configs):
        """Test AIHW mortality data is compatible with SA2HealthProfile schema."""
        extractor = AIHWMortalityExtractor(sample_configs['aihw_mortality'])
        
        # Extract demo data
        batches = list(extractor.extract('demo'))
        records = batches[0]
        
        # Check compatibility with SA2HealthProfile mortality fields
        for record in records:
            # Should be linkable by SA2 code
            assert 'geographic_id' in record
            assert len(record['geographic_id']) == 9
            
            # Should have mortality indicators
            if 'Cardiovascular' in record['indicator_name']:
                # Expect cardiovascular mortality rate
                assert record['unit'] in ['per 100,000', 'count']
                assert isinstance(record['value'], (int, float))
            
            # Should have age and sex standardisation info if available
            age_group = record.get('age_group', AgeGroupType.ALL_AGES.value)
            assert age_group in [e.value for e in AgeGroupType]
    
    def test_abs_geographic_produces_target_boundary_fields(self, sample_configs):
        """Test ABS geographic extractor produces fields required by GeographicBoundary schema."""
        extractor = ABSGeographicExtractor(sample_configs['abs_geographic'])
        
        # Extract demo SA2 data
        batches = list(extractor.extract({'level': 'SA2'}))
        assert len(batches) > 0
        
        records = batches[0]
        assert len(records) > 0
        
        # Validate target schema compatibility
        sample_record = records[0]
        
        # Required fields for GeographicBoundary schema
        required_fields = [
            'geographic_id',
            'geographic_level',
            'geographic_name',
            'area_square_km',
            'coordinate_system',
            'data_source_id',
            'extraction_timestamp',
        ]
        
        for field in required_fields:
            assert field in sample_record, f"Missing required field: {field}"
        
        # Validate SA2-specific fields
        assert len(sample_record['geographic_id']) == 9, "SA2 code should be 9 digits"
        assert sample_record['geographic_level'] == 'SA2'
        assert isinstance(sample_record['area_square_km'], (int, float))
        assert sample_record['area_square_km'] > 0, "Area should be positive"
        
        # Validate geographic hierarchy for SA2
        assert 'geographic_hierarchy' in sample_record
        hierarchy = sample_record['geographic_hierarchy']
        assert 'sa3_code' in hierarchy
        assert 'sa4_code' in hierarchy
        assert 'state_code' in hierarchy
        
        # Validate urbanisation classification
        assert 'urbanisation' in sample_record
        urbanisation = sample_record['urbanisation']
        assert urbanisation in [e.value for e in UrbanRuralClassification]
    
    def test_abs_seifa_produces_target_seifa_fields(self, sample_configs):
        """Test ABS SEIFA extractor produces fields required by SEIFAIndex schema."""
        extractor = ABSSEIFAExtractor(sample_configs['abs_geographic'])
        
        # Extract demo SEIFA data
        batches = list(extractor.extract('demo'))
        assert len(batches) > 0
        
        records = batches[0]
        assert len(records) > 0
        
        # Should have records for each SEIFA index type
        index_types_found = set()
        for record in records:
            assert 'geographic_id' in record
            assert 'index_type' in record
            assert 'score' in record
            assert 'decile' in record
            
            index_types_found.add(record['index_type'])
            
            # Validate SEIFA index type
            assert record['index_type'] in [e.value for e in SEIFAIndexType]
            
            # Validate score and decile ranges
            assert isinstance(record['score'], (int, float))
            assert 400 <= record['score'] <= 1400, "SEIFA score should be in typical range"
            
            assert isinstance(record['decile'], int)
            assert 1 <= record['decile'] <= 10, "Decile should be 1-10"
        
        # Should have all major SEIFA indices
        expected_indices = {e.value for e in SEIFAIndexType}
        assert index_types_found >= expected_indices, "Should have all SEIFA index types"
    
    def test_bom_climate_produces_target_climate_fields(self, sample_configs):
        """Test BOM climate extractor produces fields required by ClimateData schema."""
        extractor = BOMClimateExtractor(sample_configs['bom_climate'])
        
        # Extract demo climate data
        batches = list(extractor.extract('demo'))
        assert len(batches) > 0
        
        records = batches[0]
        assert len(records) > 0
        
        # Validate target schema compatibility
        sample_record = records[0]
        
        # Required fields for ClimateData schema
        required_fields = [
            'station_id',
            'measurement_date',
            'latitude',
            'longitude',
            'data_source_id',
            'extraction_timestamp',
        ]
        
        for field in required_fields:
            assert field in sample_record, f"Missing required field: {field}"
        
        # Validate climate measurement fields
        climate_fields = [
            'temperature_max_celsius',
            'temperature_min_celsius', 
            'rainfall_mm',
            'relative_humidity_9am_percent',
            'relative_humidity_3pm_percent',
        ]
        
        # At least some climate measurements should be present
        present_climate_fields = [f for f in climate_fields if f in sample_record]
        assert len(present_climate_fields) > 0, "Should have at least some climate measurements"
        
        # Validate data types and ranges
        if 'temperature_max_celsius' in sample_record:
            temp = sample_record['temperature_max_celsius']
            if temp is not None:
                assert isinstance(temp, (int, float))
                assert -20 <= temp <= 60, "Temperature should be in reasonable range"
        
        if 'rainfall_mm' in sample_record:
            rainfall = sample_record['rainfall_mm']
            if rainfall is not None:
                assert isinstance(rainfall, (int, float))
                assert rainfall >= 0, "Rainfall should be non-negative"
        
        # Validate coordinate ranges for Australia
        lat = sample_record['latitude']
        lon = sample_record['longitude']
        assert -45 <= lat <= -10, "Latitude should be within Australia"
        assert 110 <= lon <= 155, "Longitude should be within Australia"
        
        # Validate health-relevant derived indicators
        if 'heat_stress_indicator' in sample_record:
            heat_indicator = sample_record['heat_stress_indicator']
            assert heat_indicator in ['low', 'moderate', 'high'], "Heat stress indicator should be categorical"
    
    def test_medicare_utilisation_produces_target_healthcare_fields(self, sample_configs):
        """Test Medicare extractor produces fields required by HealthcareUtilisation schema."""
        extractor = MedicareUtilisationExtractor(sample_configs['medicare_utilisation'])
        
        # Extract demo Medicare data
        batches = list(extractor.extract('demo'))
        assert len(batches) > 0
        
        records = batches[0]
        assert len(records) > 0
        
        # Validate target schema compatibility
        sample_record = records[0]
        
        # Required fields for HealthcareUtilisation schema
        required_fields = [
            'geographic_id',
            'geographic_level',
            'indicator_name',
            'indicator_code',
            'indicator_type',
            'value',
            'unit',
            'service_type',
            'service_category',
            'data_source_id',
            'extraction_timestamp',
        ]
        
        for field in required_fields:
            assert field in sample_record, f"Missing required field: {field}"
        
        # Validate healthcare utilisation specific fields
        assert sample_record['indicator_type'] == HealthIndicatorType.UTILISATION.value
        assert sample_record['service_category'] in ['primary_care', 'specialist', 'hospital']
        
        # Validate utilisation metrics
        if 'visits_count' in sample_record:
            assert isinstance(sample_record['visits_count'], int)
            assert sample_record['visits_count'] >= 0
        
        if 'utilisation_rate' in sample_record:
            assert isinstance(sample_record['utilisation_rate'], (int, float))
            assert sample_record['utilisation_rate'] >= 0
        
        if 'bulk_billed_percentage' in sample_record:
            bulk_billing = sample_record['bulk_billed_percentage']
            assert isinstance(bulk_billing, (int, float))
            assert 0 <= bulk_billing <= 100, "Bulk billing percentage should be 0-100"
        
        # Validate privacy protection indicators
        assert 'data_suppressed' in sample_record
        assert isinstance(sample_record['data_suppressed'], bool)
    
    def test_data_integration_compatibility(self, sample_configs):
        """Test that extractor outputs are compatible for data integration."""
        # Create extractors
        mortality_extractor = AIHWMortalityExtractor(sample_configs['aihw_mortality'])
        geographic_extractor = ABSGeographicExtractor(sample_configs['abs_geographic'])
        seifa_extractor = ABSSEIFAExtractor(sample_configs['abs_geographic'])
        
        # Extract sample data
        mortality_batches = list(mortality_extractor.extract('demo'))
        geographic_batches = list(geographic_extractor.extract({'level': 'SA2'}))
        seifa_batches = list(seifa_extractor.extract('demo'))
        
        mortality_records = mortality_batches[0]
        geographic_records = geographic_batches[0]
        seifa_records = seifa_batches[0]
        
        # Test linkage compatibility
        # All should use same SA2 code format for linking
        mortality_sa2s = {r['geographic_id'] for r in mortality_records}
        geographic_sa2s = {r['geographic_id'] for r in geographic_records}
        seifa_sa2s = {r['geographic_id'] for r in seifa_records}
        
        # Should have overlapping SA2 codes for integration
        common_sa2s = mortality_sa2s & geographic_sa2s & seifa_sa2s
        assert len(common_sa2s) > 0, "Should have common SA2 codes for integration"
        
        # All SA2 codes should be same format
        all_sa2s = mortality_sa2s | geographic_sa2s | seifa_sa2s
        for sa2_code in all_sa2s:
            assert len(sa2_code) == 9, f"SA2 code {sa2_code} should be 9 digits"
            assert sa2_code.isdigit(), f"SA2 code {sa2_code} should be numeric"
        
        # Test temporal compatibility
        # Records should have compatible temporal references
        for record in mortality_records:
            if 'reference_year' in record:
                year = record['reference_year']
                assert isinstance(year, int)
                assert 2015 <= year <= 2025, "Year should be in reasonable range"


class TestExtractorRegistry:
    """Test the extractor registry and factory system."""
    
    def test_registry_contains_all_extractors(self):
        """Test that registry contains all expected extractors."""
        registry = ExtractorRegistry()
        
        # Check all extractor types are registered
        expected_types = list(ExtractorType)
        registered_types = list(registry._extractors.keys())
        
        for extractor_type in expected_types:
            assert extractor_type in registered_types, f"Missing extractor type: {extractor_type.value}"
    
    def test_registry_dependency_order(self):
        """Test that registry provides correct dependency order."""
        registry = ExtractorRegistry()
        
        # Get extraction order
        order = registry.get_extraction_order()
        
        # Geographic should come before dependent extractors
        geographic_idx = order.index(ExtractorType.ABS_GEOGRAPHIC)
        census_idx = order.index(ExtractorType.ABS_CENSUS)
        seifa_idx = order.index(ExtractorType.ABS_SEIFA)
        
        assert geographic_idx < census_idx, "Geographic should come before Census"
        assert geographic_idx < seifa_idx, "Geographic should come before SEIFA"
    
    def test_factory_creates_extractors(self):
        """Test that factory creates extractor instances."""
        registry = ExtractorRegistry()
        factory = ExtractorFactory(registry)
        
        # Test creating AIHW mortality extractor
        extractor = factory.create_extractor(ExtractorType.AIHW_MORTALITY)
        assert isinstance(extractor, AIHWMortalityExtractor)
        
        # Test creating ABS geographic extractor
        extractor = factory.create_extractor(ExtractorType.ABS_GEOGRAPHIC)
        assert isinstance(extractor, ABSGeographicExtractor)
    
    def test_validator_validates_output_compatibility(self):
        """Test that validator checks output compatibility with target schemas."""
        registry = ExtractorRegistry()
        validator = ExtractorValidator(registry)
        
        # Sample valid mortality records
        sample_records = [
            {
                'geographic_id': '101021001',
                'geographic_level': 'SA2',
                'indicator_name': 'Cardiovascular Disease Mortality',
                'indicator_code': 'MORT_CARDIOVASCULAR_DISEASE',
                'indicator_type': 'mortality',
                'value': 45.2,
                'unit': 'per 100,000',
                'cause_of_death': 'Cardiovascular Disease',
                'data_source_id': 'AIHW_GRIM',
                'extraction_timestamp': datetime.now().isoformat(),
            }
        ]
        
        # Validate mortality extractor output
        validation_result = validator.validate_extractor_output(
            ExtractorType.AIHW_MORTALITY,
            sample_records
        )
        
        assert validation_result['overall_valid'], "Sample mortality records should be valid"
        assert 'MortalityData' in validation_result['schema_compatibility']
        assert validation_result['schema_compatibility']['MortalityData']['compatible']


class TestTargetSchemaFieldMappings:
    """Test that extractors correctly map source fields to target schema fields."""
    
    @pytest.fixture
    def field_mapping_configs(self):
        """Configuration emphasizing field mappings."""
        return {
            'max_retries': 1,
            'batch_size': 10,
        }
    
    def test_aihw_mortality_field_mappings(self, field_mapping_configs):
        """Test AIHW mortality extractor maps fields correctly."""
        extractor = AIHWMortalityExtractor(field_mapping_configs)
        
        # Mock source record with various field name formats
        source_record = {
            'SA2_CODE_2021': '101021001',
            'CAUSE': 'Cardiovascular Disease',
            'ICD10': 'I25.9',
            'DEATHS': 15,
            'RATE': 45.2,
            'YEAR': 2021,
            'AGE_GROUP': '65-74',
            'SEX': 'Persons',
        }
        
        # Map fields using extractor's mapping logic
        mapped = extractor._map_mortality_fields(source_record)
        
        # Verify mapping to target schema fields
        assert mapped['sa2_code'] == '101021001'
        assert mapped['cause_of_death'] == 'Cardiovascular Disease'
        assert mapped['icd10_code'] == 'I25.9'
        assert mapped['deaths_count'] == 15
        assert mapped['mortality_rate'] == 45.2
        assert mapped['year'] == 2021
        assert mapped['age_group'] == '65-74'
        assert mapped['sex'] == 'Persons'
    
    def test_abs_geographic_field_mappings(self, field_mapping_configs):
        """Test ABS geographic extractor maps fields correctly."""
        extractor = ABSGeographicExtractor(field_mapping_configs)
        
        # Mock geographic record
        import pandas as pd
        source_record = pd.Series({
            'SA2_CODE21': '101021001',
            'SA2_NAME21': 'Sydney - Haymarket - The Rocks',
            'SA3_CODE21': '10102',
            'SA4_CODE21': '101',
            'STE_CODE21': '1',
            'AREASQKM21': 2.5,
        })
        
        # Map fields
        mapped = extractor._map_geographic_fields(source_record, 'SA2')
        
        # Verify mapping
        assert mapped['sa2_code'] == '101021001'
        assert mapped['sa2_name'] == 'Sydney - Haymarket - The Rocks'
        assert mapped['sa3_code'] == '10102'
        assert mapped['sa4_code'] == '101'
        assert mapped['state_code'] == '1'
        assert mapped['area_sq_km'] == 2.5
    
    def test_medicare_field_mappings_with_privacy_protection(self, field_mapping_configs):
        """Test Medicare extractor maps fields and applies privacy protection."""
        config = field_mapping_configs.copy()
        config['min_cell_size'] = 5
        
        extractor = MedicareUtilisationExtractor(config)
        
        # Mock Medicare record with small count
        source_record = {
            'SA2_CODE': '101021001',
            'ITEM_GROUP': 'GP Attendances',
            'SERVICES': 3,  # Below minimum cell size
            'PATIENTS': 2,
            'BENEFITS': 150.50,
            'BULK_BILLED': 2,
            'YEAR': 2023,
        }
        
        # Map and apply privacy protection
        mapped = extractor._map_medicare_fields(source_record)
        protected = extractor._apply_privacy_protection(mapped)
        
        # Verify privacy protection applied
        assert protected['services_count'] == '<5'  # Should be suppressed
        assert protected['services_count_suppressed'] == True
        assert protected['benefits_paid'] == 'SUPP'  # Complementary suppression


class TestAustralianHealthDataStandards:
    """Test compliance with Australian health data standards."""
    
    def test_aihw_metadata_standards_compliance(self):
        """Test AIHW extractors comply with METEOR standards."""
        config = {'batch_size': 10}
        extractor = AIHWMortalityExtractor(config)
        
        # Extract sample data
        batches = list(extractor.extract('demo'))
        records = batches[0]
        
        for record in records:
            # Should have proper data source identification
            assert 'data_source_id' in record
            assert 'AIHW' in record['data_source_id']
            
            # Should have extraction timestamp for lineage
            assert 'extraction_timestamp' in record
            
            # Geographic identifiers should follow ABS standards
            sa2_code = record['geographic_id']
            assert len(sa2_code) == 9, "Should use ABS SA2 code format"
            
            # Age groups should follow standard classifications
            if 'age_group' in record:
                age_group = record['age_group']
                standard_age_groups = [e.value for e in AgeGroupType]
                # Should be either a standard age group or 'all_ages'
                assert age_group in standard_age_groups or age_group == 'all_ages'
    
    def test_privacy_act_compliance(self):
        """Test Medicare/PBS extractors comply with Privacy Act 1988."""
        config = {'min_cell_size': 5, 'batch_size': 10}
        extractor = MedicareUtilisationExtractor(config)
        
        # Extract sample data
        batches = list(extractor.extract('demo'))
        records = batches[0]
        
        for record in records:
            # Should have privacy protection indicators
            assert 'data_suppressed' in record
            assert isinstance(record['data_suppressed'], bool)
            
            # Should have proper data source authority
            assert 'data_source_id' in record
            assert any(term in record['data_source_id'] for term in ['MEDICARE', 'HEALTH'])
    
    def test_gda2020_coordinate_system_compliance(self):
        """Test geographic extractors use GDA2020 coordinate system."""
        config = {'coordinate_system': 'GDA2020', 'batch_size': 10}
        extractor = ABSGeographicExtractor(config)
        
        # Extract sample data
        batches = list(extractor.extract({'level': 'SA2'}))
        records = batches[0]
        
        for record in records:
            # Should specify coordinate system
            assert 'coordinate_system' in record
            assert record['coordinate_system'] == 'GDA2020'


# Integration test combining multiple extractors
class TestBackwardsIntegrationWorkflow:
    """Test complete backwards workflow from target schemas to extractor outputs."""
    
    def test_master_health_record_data_assembly(self):
        """Test that extractors produce data that can assemble into MasterHealthRecord."""
        configs = {
            'batch_size': 5,
            'coordinate_system': 'GDA2020',
            'min_cell_size': 5,
        }
        
        # Create extractors
        mortality_extractor = AIHWMortalityExtractor(configs)
        geographic_extractor = ABSGeographicExtractor(configs)
        seifa_extractor = ABSSEIFAExtractor(configs)
        medicare_extractor = MedicareUtilisationExtractor(configs)
        
        # Extract data
        mortality_data = list(mortality_extractor.extract('demo'))[0]
        geographic_data = list(geographic_extractor.extract({'level': 'SA2'}))[0]
        seifa_data = list(seifa_extractor.extract('demo'))[0]
        medicare_data = list(medicare_extractor.extract('demo'))[0]
        
        # Get common SA2 for integration test
        sa2_codes = set()
        sa2_codes.update(r['geographic_id'] for r in mortality_data)
        sa2_codes.update(r['geographic_id'] for r in geographic_data)
        sa2_codes.update(r['geographic_id'] for r in seifa_data)
        sa2_codes.update(r['geographic_id'] for r in medicare_data)
        
        common_sa2 = list(sa2_codes)[0]  # Take first common SA2
        
        # Simulate MasterHealthRecord assembly
        master_record_components = {}
        
        # Geographic components
        geo_record = next(r for r in geographic_data if r['geographic_id'] == common_sa2)
        master_record_components.update({
            'sa2_code': geo_record['geographic_id'],
            'sa2_name': geo_record['geographic_name'],
            'boundary_data': {
                'area_sq_km': geo_record['area_square_km'],
                'coordinate_system': geo_record['coordinate_system'],
            },
            'geographic_hierarchy': geo_record['geographic_hierarchy'],
            'urbanisation': geo_record['urbanisation'],
        })
        
        # SEIFA components
        seifa_records = [r for r in seifa_data if r['geographic_id'] == common_sa2]
        seifa_scores = {}
        seifa_deciles = {}
        for seifa_record in seifa_records:
            index_type = seifa_record['index_type']
            seifa_scores[index_type] = seifa_record['score']
            seifa_deciles[index_type] = seifa_record['decile']
        
        master_record_components.update({
            'seifa_scores': seifa_scores,
            'seifa_deciles': seifa_deciles,
        })
        
        # Health indicators from mortality
        mortality_records = [r for r in mortality_data if r['geographic_id'] == common_sa2]
        mortality_indicators = {}
        for mort_record in mortality_records:
            cause = mort_record['cause_of_death']
            mortality_indicators[f"{cause}_mortality_rate"] = mort_record['value']
        
        master_record_components['mortality_indicators'] = mortality_indicators
        
        # Healthcare utilisation
        medicare_records = [r for r in medicare_data if r['geographic_id'] == common_sa2]
        healthcare_access = {}
        for medicare_record in medicare_records:
            service_type = medicare_record['service_type']
            if 'bulk_billed_percentage' in medicare_record:
                healthcare_access[f"{service_type}_bulk_billing_rate"] = medicare_record['bulk_billed_percentage']
        
        master_record_components['healthcare_access'] = healthcare_access
        
        # Validate assembled record has MasterHealthRecord required components
        required_components = [
            'sa2_code',
            'sa2_name', 
            'boundary_data',
            'geographic_hierarchy',
            'seifa_scores',
            'seifa_deciles',
        ]
        
        for component in required_components:
            assert component in master_record_components, f"Missing component: {component}"
        
        # Validate data types and structures
        assert len(master_record_components['sa2_code']) == 9
        assert isinstance(master_record_components['boundary_data'], dict)
        assert isinstance(master_record_components['seifa_scores'], dict)
        assert len(master_record_components['seifa_scores']) > 0
        
        # Should have SEIFA indices
        expected_seifa_types = [e.value for e in SEIFAIndexType]
        found_seifa_types = set(master_record_components['seifa_scores'].keys())
        assert len(found_seifa_types & set(expected_seifa_types)) > 0, "Should have SEIFA data"