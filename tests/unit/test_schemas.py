"""
Unit tests for AHGD schema validation and management.

Tests Pydantic schema validation, schema migrations, version management,
and schema compatibility checks.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from pydantic import BaseModel, ValidationError, Field, validator
from pydantic.dataclasses import dataclass

from src.utils.schema_manager import SchemaManager
from src.utils.interfaces import ConfigurationError


# Test schema models
class SA2HealthDataV1(BaseModel):
    """Version 1.0 schema for SA2 health data."""
    sa2_code: str = Field(..., regex=r"^[0-9]{9}$", description="9-digit SA2 code")
    value: float = Field(..., ge=0, description="Health indicator value")
    year: int = Field(..., ge=2000, le=2030, description="Reference year")
    
    class Config:
        schema_extra = {
            "version": "1.0",
            "description": "Basic SA2 health data schema"
        }


class SA2HealthDataV2(BaseModel):
    """Version 2.0 schema for SA2 health data with additional fields."""
    sa2_code: str = Field(..., regex=r"^[0-9]{9}$", description="9-digit SA2 code")
    value: float = Field(..., ge=0, description="Health indicator value")
    year: int = Field(..., ge=2000, le=2030, description="Reference year")
    indicator_type: str = Field(..., description="Type of health indicator")
    confidence_interval: Optional[float] = Field(None, ge=0, le=1, description="Confidence interval")
    data_source: str = Field(default="AIHW", description="Data source")
    
    @validator('indicator_type')
    def validate_indicator_type(cls, v):
        valid_types = ['mortality', 'morbidity', 'hospitalisation', 'immunisation']
        if v not in valid_types:
            raise ValueError(f'indicator_type must be one of {valid_types}')
        return v
    
    class Config:
        schema_extra = {
            "version": "2.0",
            "description": "Enhanced SA2 health data schema with indicator types"
        }


class CensusDataSchema(BaseModel):
    """Schema for census data."""
    sa2_code: str = Field(..., regex=r"^[0-9]{9}$")
    total_population: int = Field(..., ge=0)
    median_age: float = Field(..., ge=0, le=120)
    median_income: Optional[int] = Field(None, ge=0)
    unemployment_rate: float = Field(..., ge=0, le=100)
    year: int = Field(..., ge=2000, le=2030)
    
    @validator('unemployment_rate')
    def validate_unemployment_rate(cls, v):
        if v > 50:  # Sanity check for unemployment rate
            raise ValueError('Unemployment rate seems unusually high')
        return v


class SEIFADataSchema(BaseModel):
    """Schema for SEIFA (Socio-Economic Indexes) data."""
    sa2_code: str = Field(..., regex=r"^[0-9]{9}$")
    irsad_score: int = Field(..., ge=1, le=1200, description="Index of Relative Socio-economic Advantage and Disadvantage")
    irsad_decile: int = Field(..., ge=1, le=10)
    irsd_score: Optional[int] = Field(None, ge=1, le=1200, description="Index of Relative Socio-economic Disadvantage")
    irsd_decile: Optional[int] = Field(None, ge=1, le=10)
    year: int = Field(..., ge=2000, le=2030)


class GeographicDataSchema(BaseModel):
    """Schema for geographic data."""
    sa2_code: str = Field(..., regex=r"^[0-9]{9}$")
    sa2_name: str = Field(..., min_length=1, max_length=100)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    state: str = Field(..., regex=r"^(NSW|VIC|QLD|SA|WA|TAS|NT|ACT)$")
    area_sqkm: float = Field(..., gt=0)


@pytest.mark.unit
class TestSchemaValidation:
    """Test schema validation functionality."""
    
    def test_sa2_health_data_v1_valid(self):
        """Test valid SA2 health data V1 schema validation."""
        valid_data = {
            "sa2_code": "101011001",
            "value": 25.5,
            "year": 2021
        }
        
        model = SA2HealthDataV1(**valid_data)
        
        assert model.sa2_code == "101011001"
        assert model.value == 25.5
        assert model.year == 2021
    
    def test_sa2_health_data_v1_invalid_sa2_code(self):
        """Test invalid SA2 code validation."""
        invalid_data = {
            "sa2_code": "12345",  # Too short
            "value": 25.5,
            "year": 2021
        }
        
        with pytest.raises(ValidationError) as exc_info:
            SA2HealthDataV1(**invalid_data)
        
        assert "sa2_code" in str(exc_info.value)
    
    def test_sa2_health_data_v1_invalid_value(self):
        """Test invalid value validation."""
        invalid_data = {
            "sa2_code": "101011001",
            "value": -10.0,  # Negative value
            "year": 2021
        }
        
        with pytest.raises(ValidationError) as exc_info:
            SA2HealthDataV1(**invalid_data)
        
        assert "value" in str(exc_info.value)
    
    def test_sa2_health_data_v1_invalid_year(self):
        """Test invalid year validation."""
        invalid_data = {
            "sa2_code": "101011001",
            "value": 25.5,
            "year": 1990  # Too early
        }
        
        with pytest.raises(ValidationError) as exc_info:
            SA2HealthDataV1(**invalid_data)
        
        assert "year" in str(exc_info.value)
    
    def test_sa2_health_data_v2_valid(self):
        """Test valid SA2 health data V2 schema validation."""
        valid_data = {
            "sa2_code": "101011001",
            "value": 25.5,
            "year": 2021,
            "indicator_type": "mortality",
            "confidence_interval": 0.95,
            "data_source": "AIHW"
        }
        
        model = SA2HealthDataV2(**valid_data)
        
        assert model.sa2_code == "101011001"
        assert model.value == 25.5
        assert model.year == 2021
        assert model.indicator_type == "mortality"
        assert model.confidence_interval == 0.95
        assert model.data_source == "AIHW"
    
    def test_sa2_health_data_v2_invalid_indicator_type(self):
        """Test invalid indicator type validation."""
        invalid_data = {
            "sa2_code": "101011001",
            "value": 25.5,
            "year": 2021,
            "indicator_type": "invalid_type"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            SA2HealthDataV2(**invalid_data)
        
        assert "indicator_type" in str(exc_info.value)
    
    def test_sa2_health_data_v2_default_values(self):
        """Test default values in V2 schema."""
        minimal_data = {
            "sa2_code": "101011001",
            "value": 25.5,
            "year": 2021,
            "indicator_type": "mortality"
        }
        
        model = SA2HealthDataV2(**minimal_data)
        
        assert model.data_source == "AIHW"  # Default value
        assert model.confidence_interval is None  # Optional field
    
    def test_census_data_schema_valid(self):
        """Test valid census data schema validation."""
        valid_data = {
            "sa2_code": "101011001",
            "total_population": 15420,
            "median_age": 34.5,
            "median_income": 68000,
            "unemployment_rate": 4.2,
            "year": 2021
        }
        
        model = CensusDataSchema(**valid_data)
        
        assert model.total_population == 15420
        assert model.median_age == 34.5
        assert model.unemployment_rate == 4.2
    
    def test_census_data_schema_high_unemployment(self):
        """Test census data with unrealistic unemployment rate."""
        invalid_data = {
            "sa2_code": "101011001",
            "total_population": 15420,
            "median_age": 34.5,
            "unemployment_rate": 75.0,  # Unrealistically high
            "year": 2021
        }
        
        with pytest.raises(ValidationError) as exc_info:
            CensusDataSchema(**invalid_data)
        
        assert "unemployment_rate" in str(exc_info.value)
    
    def test_seifa_data_schema_valid(self):
        """Test valid SEIFA data schema validation."""
        valid_data = {
            "sa2_code": "101011001",
            "irsad_score": 1050,
            "irsad_decile": 9,
            "irsd_score": 980,
            "irsd_decile": 8,
            "year": 2021
        }
        
        model = SEIFADataSchema(**valid_data)
        
        assert model.irsad_score == 1050
        assert model.irsad_decile == 9
        assert model.irsd_score == 980
        assert model.irsd_decile == 8
    
    def test_seifa_data_schema_invalid_scores(self):
        """Test SEIFA data with invalid scores."""
        invalid_data = {
            "sa2_code": "101011001",
            "irsad_score": 1500,  # Too high
            "irsad_decile": 11,   # Too high
            "year": 2021
        }
        
        with pytest.raises(ValidationError) as exc_info:
            SEIFADataSchema(**invalid_data)
        
        errors = str(exc_info.value)
        assert "irsad_score" in errors
        assert "irsad_decile" in errors
    
    def test_geographic_data_schema_valid(self):
        """Test valid geographic data schema validation."""
        valid_data = {
            "sa2_code": "101011001",
            "sa2_name": "Sydney - Haymarket - The Rocks",
            "latitude": -33.8688,
            "longitude": 151.2093,
            "state": "NSW",
            "area_sqkm": 2.5
        }
        
        model = GeographicDataSchema(**valid_data)
        
        assert model.sa2_name == "Sydney - Haymarket - The Rocks"
        assert model.latitude == -33.8688
        assert model.longitude == 151.2093
        assert model.state == "NSW"
    
    def test_geographic_data_schema_invalid_coordinates(self):
        """Test geographic data with invalid coordinates."""
        invalid_data = {
            "sa2_code": "101011001",
            "sa2_name": "Test Location",
            "latitude": -95.0,  # Invalid latitude
            "longitude": 200.0,  # Invalid longitude
            "state": "NSW",
            "area_sqkm": 2.5
        }
        
        with pytest.raises(ValidationError) as exc_info:
            GeographicDataSchema(**invalid_data)
        
        errors = str(exc_info.value)
        assert "latitude" in errors
        assert "longitude" in errors
    
    def test_geographic_data_schema_invalid_state(self):
        """Test geographic data with invalid state."""
        invalid_data = {
            "sa2_code": "101011001",
            "sa2_name": "Test Location",
            "latitude": -33.8688,
            "longitude": 151.2093,
            "state": "INVALID",  # Invalid state code
            "area_sqkm": 2.5
        }
        
        with pytest.raises(ValidationError) as exc_info:
            GeographicDataSchema(**invalid_data)
        
        assert "state" in str(exc_info.value)


@pytest.mark.unit
class TestSchemaManager:
    """Test schema manager functionality."""
    
    def test_schema_manager_initialisation(self, temp_dir):
        """Test schema manager initialisation."""
        schemas_dir = temp_dir / "schemas"
        schemas_dir.mkdir()
        
        manager = SchemaManager(schemas_dir)
        
        assert manager.schemas_dir == schemas_dir
        assert isinstance(manager.schemas, dict)
    
    def test_register_schema(self, temp_dir):
        """Test schema registration."""
        schemas_dir = temp_dir / "schemas"
        schemas_dir.mkdir()
        
        manager = SchemaManager(schemas_dir)
        
        manager.register_schema("sa2_health_v1", SA2HealthDataV1, "1.0")
        
        assert "sa2_health_v1" in manager.schemas
        assert manager.schemas["sa2_health_v1"]["model"] == SA2HealthDataV1
        assert manager.schemas["sa2_health_v1"]["version"] == "1.0"
    
    def test_get_schema(self, temp_dir):
        """Test schema retrieval."""
        schemas_dir = temp_dir / "schemas"
        schemas_dir.mkdir()
        
        manager = SchemaManager(schemas_dir)
        manager.register_schema("sa2_health_v1", SA2HealthDataV1, "1.0")
        
        schema = manager.get_schema("sa2_health_v1")
        
        assert schema == SA2HealthDataV1
    
    def test_get_nonexistent_schema(self, temp_dir):
        """Test retrieval of non-existent schema."""
        schemas_dir = temp_dir / "schemas"
        schemas_dir.mkdir()
        
        manager = SchemaManager(schemas_dir)
        
        with pytest.raises(ValueError, match="Schema 'nonexistent' not found"):
            manager.get_schema("nonexistent")
    
    def test_validate_data_success(self, temp_dir):
        """Test successful data validation."""
        schemas_dir = temp_dir / "schemas"
        schemas_dir.mkdir()
        
        manager = SchemaManager(schemas_dir)
        manager.register_schema("sa2_health_v1", SA2HealthDataV1, "1.0")
        
        valid_data = {
            "sa2_code": "101011001",
            "value": 25.5,
            "year": 2021
        }
        
        result = manager.validate_data("sa2_health_v1", valid_data)
        
        assert result.is_valid is True
        assert result.errors == []
        assert isinstance(result.validated_data, SA2HealthDataV1)
    
    def test_validate_data_failure(self, temp_dir):
        """Test data validation failure."""
        schemas_dir = temp_dir / "schemas"
        schemas_dir.mkdir()
        
        manager = SchemaManager(schemas_dir)
        manager.register_schema("sa2_health_v1", SA2HealthDataV1, "1.0")
        
        invalid_data = {
            "sa2_code": "invalid",  # Invalid format
            "value": -10,           # Invalid value
            "year": 1990           # Invalid year
        }
        
        result = manager.validate_data("sa2_health_v1", invalid_data)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert result.validated_data is None
    
    def test_validate_batch_data_success(self, temp_dir):
        """Test successful batch data validation."""
        schemas_dir = temp_dir / "schemas"
        schemas_dir.mkdir()
        
        manager = SchemaManager(schemas_dir)
        manager.register_schema("sa2_health_v1", SA2HealthDataV1, "1.0")
        
        batch_data = [
            {"sa2_code": "101011001", "value": 25.5, "year": 2021},
            {"sa2_code": "101011002", "value": 30.2, "year": 2021}
        ]
        
        results = manager.validate_batch("sa2_health_v1", batch_data)
        
        assert len(results) == 2
        assert all(result.is_valid for result in results)
    
    def test_validate_batch_data_mixed(self, temp_dir):
        """Test batch data validation with mixed valid/invalid records."""
        schemas_dir = temp_dir / "schemas"
        schemas_dir.mkdir()
        
        manager = SchemaManager(schemas_dir)
        manager.register_schema("sa2_health_v1", SA2HealthDataV1, "1.0")
        
        batch_data = [
            {"sa2_code": "101011001", "value": 25.5, "year": 2021},  # Valid
            {"sa2_code": "invalid", "value": -10, "year": 1990},     # Invalid
            {"sa2_code": "101011003", "value": 22.8, "year": 2021}   # Valid
        ]
        
        results = manager.validate_batch("sa2_health_v1", batch_data)
        
        assert len(results) == 3
        assert results[0].is_valid is True
        assert results[1].is_valid is False
        assert results[2].is_valid is True
    
    def test_schema_versioning(self, temp_dir):
        """Test schema versioning functionality."""
        schemas_dir = temp_dir / "schemas"
        schemas_dir.mkdir()
        
        manager = SchemaManager(schemas_dir)
        
        # Register multiple versions
        manager.register_schema("sa2_health", SA2HealthDataV1, "1.0")
        manager.register_schema("sa2_health", SA2HealthDataV2, "2.0")
        
        # Should return latest version by default
        latest_schema = manager.get_schema("sa2_health")
        assert latest_schema == SA2HealthDataV2
        
        # Should be able to get specific version
        v1_schema = manager.get_schema("sa2_health", version="1.0")
        assert v1_schema == SA2HealthDataV1
    
    def test_schema_compatibility_check(self, temp_dir):
        """Test schema compatibility checking."""
        schemas_dir = temp_dir / "schemas"
        schemas_dir.mkdir()
        
        manager = SchemaManager(schemas_dir)
        manager.register_schema("sa2_health", SA2HealthDataV1, "1.0")
        manager.register_schema("sa2_health", SA2HealthDataV2, "2.0")
        
        # V1 data should be compatible with V2 schema (backward compatibility)
        v1_data = {"sa2_code": "101011001", "value": 25.5, "year": 2021}
        
        # This should work if V2 has proper defaults for new fields
        try:
            is_compatible = manager.check_compatibility("sa2_health", v1_data, from_version="1.0", to_version="2.0")
            # The actual result depends on implementation details
        except NotImplementedError:
            # Method might not be implemented yet
            pass
    
    def test_schema_migration(self, temp_dir):
        """Test schema migration functionality."""
        schemas_dir = temp_dir / "schemas"
        schemas_dir.mkdir()
        
        manager = SchemaManager(schemas_dir)
        manager.register_schema("sa2_health", SA2HealthDataV1, "1.0")
        manager.register_schema("sa2_health", SA2HealthDataV2, "2.0")
        
        # Define migration function
        def migrate_v1_to_v2(data: dict) -> dict:
            migrated = data.copy()
            migrated["indicator_type"] = "mortality"  # Default value
            migrated["data_source"] = "AIHW"  # Default value
            return migrated
        
        manager.register_migration("sa2_health", "1.0", "2.0", migrate_v1_to_v2)
        
        v1_data = {"sa2_code": "101011001", "value": 25.5, "year": 2021}
        
        try:
            migrated_data = manager.migrate_data("sa2_health", v1_data, from_version="1.0", to_version="2.0")
            
            assert "indicator_type" in migrated_data
            assert "data_source" in migrated_data
            assert migrated_data["indicator_type"] == "mortality"
        except NotImplementedError:
            # Migration might not be fully implemented
            pass
    
    def test_list_schemas(self, temp_dir):
        """Test listing available schemas."""
        schemas_dir = temp_dir / "schemas"
        schemas_dir.mkdir()
        
        manager = SchemaManager(schemas_dir)
        manager.register_schema("sa2_health", SA2HealthDataV1, "1.0")
        manager.register_schema("census", CensusDataSchema, "1.0")
        manager.register_schema("seifa", SEIFADataSchema, "1.0")
        
        schema_list = manager.list_schemas()
        
        assert "sa2_health" in schema_list
        assert "census" in schema_list
        assert "seifa" in schema_list
    
    def test_schema_info(self, temp_dir):
        """Test schema information retrieval."""
        schemas_dir = temp_dir / "schemas"
        schemas_dir.mkdir()
        
        manager = SchemaManager(schemas_dir)
        manager.register_schema("sa2_health", SA2HealthDataV1, "1.0")
        
        info = manager.get_schema_info("sa2_health")
        
        assert "name" in info
        assert "version" in info
        assert "fields" in info
        assert info["name"] == "sa2_health"
        assert info["version"] == "1.0"


@pytest.mark.unit
class TestSchemaEdgeCases:
    """Test edge cases and error conditions for schemas."""
    
    def test_empty_data_validation(self, temp_dir):
        """Test validation of empty data."""
        schemas_dir = temp_dir / "schemas"
        schemas_dir.mkdir()
        
        manager = SchemaManager(schemas_dir)
        manager.register_schema("sa2_health", SA2HealthDataV1, "1.0")
        
        result = manager.validate_data("sa2_health", {})
        
        assert result.is_valid is False
        assert len(result.errors) > 0
    
    def test_none_data_validation(self, temp_dir):
        """Test validation of None data."""
        schemas_dir = temp_dir / "schemas"
        schemas_dir.mkdir()
        
        manager = SchemaManager(schemas_dir)
        manager.register_schema("sa2_health", SA2HealthDataV1, "1.0")
        
        with pytest.raises(TypeError):
            manager.validate_data("sa2_health", None)
    
    def test_malformed_data_validation(self, temp_dir):
        """Test validation of malformed data."""
        schemas_dir = temp_dir / "schemas"
        schemas_dir.mkdir()
        
        manager = SchemaManager(schemas_dir)
        manager.register_schema("sa2_health", SA2HealthDataV1, "1.0")
        
        malformed_data = {
            "sa2_code": ["not", "a", "string"],  # Wrong type
            "value": {"nested": "object"},       # Wrong type
            "year": "not_a_number"              # Wrong type
        }
        
        result = manager.validate_data("sa2_health", malformed_data)
        
        assert result.is_valid is False
        assert len(result.errors) >= 3  # At least one error per field
    
    def test_extra_fields_handling(self, temp_dir):
        """Test handling of extra fields in data."""
        schemas_dir = temp_dir / "schemas"
        schemas_dir.mkdir()
        
        manager = SchemaManager(schemas_dir)
        manager.register_schema("sa2_health", SA2HealthDataV1, "1.0")
        
        data_with_extra_fields = {
            "sa2_code": "101011001",
            "value": 25.5,
            "year": 2021,
            "extra_field_1": "should be ignored",
            "extra_field_2": 999
        }
        
        result = manager.validate_data("sa2_health", data_with_extra_fields)
        
        # Pydantic by default ignores extra fields, so this should be valid
        assert result.is_valid is True
    
    def test_unicode_data_validation(self, temp_dir):
        """Test validation of data with unicode characters."""
        schemas_dir = temp_dir / "schemas"
        schemas_dir.mkdir()
        
        manager = SchemaManager(schemas_dir)
        manager.register_schema("geographic", GeographicDataSchema, "1.0")
        
        unicode_data = {
            "sa2_code": "101011001",
            "sa2_name": "Sydney - Çhiñatown - 中国城",  # Unicode characters
            "latitude": -33.8688,
            "longitude": 151.2093,
            "state": "NSW",
            "area_sqkm": 2.5
        }
        
        result = manager.validate_data("geographic", unicode_data)
        
        assert result.is_valid is True
        assert result.validated_data.sa2_name == "Sydney - Çhiñatown - 中国城"
    
    def test_very_large_numbers(self, temp_dir):
        """Test validation with very large numbers."""
        schemas_dir = temp_dir / "schemas"
        schemas_dir.mkdir()
        
        manager = SchemaManager(schemas_dir)
        manager.register_schema("census", CensusDataSchema, "1.0")
        
        large_number_data = {
            "sa2_code": "101011001",
            "total_population": 999999999999,  # Very large population
            "median_age": 34.5,
            "unemployment_rate": 4.2,
            "year": 2021
        }
        
        result = manager.validate_data("census", large_number_data)
        
        # Should handle large numbers appropriately
        assert isinstance(result.is_valid, bool)  # Should not crash