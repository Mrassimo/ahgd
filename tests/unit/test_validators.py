"""
Unit tests for AHGD data validators.

Tests validation framework, quality checks, business rules,
geographic validation, and statistical validation methods.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import List

from src.validators.base import BaseValidator
from src.utils.interfaces import (
    DataBatch,
    DataRecord,
    ValidationError,
    ValidationResult,
    ValidationSeverity,
    GeographicValidationError,
)


class ConcreteValidator(BaseValidator):
    """Concrete validator implementation for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._validate_calls = 0
        self._should_fail = False
    
    def validate(self, data: DataBatch) -> List[ValidationResult]:
        """Mock validate implementation."""
        self._validate_calls += 1
        results = []
        
        if self._should_fail:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="custom_validation_failure",
                message="Custom validation failed for testing",
                affected_records=list(range(len(data)))
            ))
        
        # Custom validation: check for test-specific rules
        for record_idx, record in enumerate(data):
            # Check for test marker field
            if record.get('_test_invalid'):
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="test_marker_validation",
                    message="Record marked as invalid for testing",
                    affected_records=[record_idx]
                ))
        
        return results
    
    def get_validation_rules(self) -> List[str]:
        """Mock validation rules implementation."""
        return [
            "schema_validation",
            "business_rules",
            "statistical_validation",
            "geographic_validation",
            "custom_validation"
        ]
    
    def set_failure_mode(self, should_fail: bool):
        """Set the validator to fail for testing."""
        self._should_fail = should_fail


@pytest.mark.unit
class TestBaseValidator:
    """Test cases for BaseValidator."""
    
    def test_validator_initialisation(self, sample_config, mock_logger):
        """Test validator initialisation with configuration."""
        validator_id = "test_validator"
        config = sample_config["validators"]["schema_validator"]
        
        validator = ConcreteValidator(validator_id, config, mock_logger)
        
        assert validator.validator_id == validator_id
        assert validator.config == config
        assert validator.logger == mock_logger
        assert validator.required_columns == {"sa2_code", "value", "year"}
        assert validator.column_types == {"sa2_code": "string", "value": "float", "year": "integer"}
    
    def test_validator_default_configuration(self):
        """Test validator initialisation with default configuration."""
        validator = ConcreteValidator("test", {})
        
        assert validator.required_columns == set()
        assert validator.column_types == {}
        assert validator.business_rules == []
        assert validator.statistical_rules == {}
    
    def test_comprehensive_validation_success(self, sample_config, mock_logger, sample_data_batch):
        """Test comprehensive validation with valid data."""
        validator = ConcreteValidator("test", sample_config["validators"]["schema_validator"], mock_logger)
        
        results = validator.validate_comprehensive(sample_data_batch)
        
        # Should have validation results from all validation types
        assert isinstance(results, list)
        
        # No errors should be found in valid data
        error_results = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(error_results) == 0
    
    def test_comprehensive_validation_with_errors(self, sample_config, mock_logger, invalid_data_batch):
        """Test comprehensive validation with invalid data."""
        validator = ConcreteValidator("test", sample_config["validators"]["schema_validator"], mock_logger)
        
        results = validator.validate_comprehensive(invalid_data_batch)
        
        # Should have error results
        error_results = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(error_results) > 0
        
        # Check for specific validation failures
        rule_ids = [r.rule_id for r in results]
        assert any("schema" in rule_id for rule_id in rule_ids)
    
    def test_schema_validation_success(self, sample_config, mock_logger, sample_data_batch):
        """Test successful schema validation."""
        validator = ConcreteValidator("test", sample_config["validators"]["schema_validator"], mock_logger)
        
        results = validator.validate_schema(sample_data_batch)
        
        # Valid data should pass schema validation
        error_results = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(error_results) == 0
    
    def test_schema_validation_missing_columns(self, sample_config, mock_logger):
        """Test schema validation with missing required columns."""
        validator = ConcreteValidator("test", sample_config["validators"]["schema_validator"], mock_logger)
        
        # Data missing required columns
        incomplete_data = [
            {"sa2_code": "101011001", "value": 25.5},  # Missing year
            {"value": 30.2, "year": 2021}              # Missing sa2_code
        ]
        
        results = validator.validate_schema(incomplete_data)
        
        # Should detect missing columns
        missing_column_errors = [
            r for r in results 
            if r.rule_id == "schema_missing_columns" and r.severity == ValidationSeverity.ERROR
        ]
        assert len(missing_column_errors) > 0
    
    def test_schema_validation_wrong_types(self, sample_config, mock_logger):
        """Test schema validation with incorrect data types."""
        validator = ConcreteValidator("test", sample_config["validators"]["schema_validator"], mock_logger)
        
        # Data with wrong types
        wrong_type_data = [
            {"sa2_code": "101011001", "value": "not_a_number", "year": 2021},
            {"sa2_code": "101011002", "value": 25.5, "year": "not_a_year"}
        ]
        
        results = validator.validate_schema(wrong_type_data)
        
        # Should detect type errors
        type_errors = [
            r for r in results 
            if r.rule_id == "schema_invalid_type" and r.severity == ValidationSeverity.ERROR
        ]
        assert len(type_errors) > 0
    
    def test_schema_validation_empty_data(self, sample_config, mock_logger):
        """Test schema validation with empty data."""
        validator = ConcreteValidator("test", sample_config["validators"]["schema_validator"], mock_logger)
        
        empty_data = []
        results = validator.validate_schema(empty_data)
        
        # Should handle empty data gracefully
        empty_data_errors = [
            r for r in results 
            if r.rule_id == "schema_empty_data" and r.severity == ValidationSeverity.ERROR
        ]
        assert len(empty_data_errors) == 1
    
    def test_business_rules_validation(self, sample_config, mock_logger):
        """Test business rules validation."""
        validator = ConcreteValidator("test", sample_config["validators"]["schema_validator"], mock_logger)
        
        # Data that should pass business rules
        valid_data = [
            {"sa2_code": "101011001", "value": 25.5, "year": 2021}
        ]
        
        results = validator.validate_business_rules(valid_data)
        
        # Should process business rules without errors
        assert isinstance(results, list)
    
    def test_business_rules_range_check(self, sample_config, mock_logger):
        """Test business rules range check validation."""
        config = sample_config["validators"]["schema_validator"].copy()
        config["business_rules"] = [
            {
                "id": "value_range_check",
                "type": "range_check",
                "column": "value",
                "min": 0,
                "max": 100
            }
        ]
        
        validator = ConcreteValidator("test", config, mock_logger)
        
        # Data with values outside range
        out_of_range_data = [
            {"sa2_code": "101011001", "value": -10, "year": 2021},    # Below minimum
            {"sa2_code": "101011002", "value": 150, "year": 2021}     # Above maximum
        ]
        
        results = validator.validate_business_rules(out_of_range_data)
        
        # Should detect range violations
        range_errors = [r for r in results if "range_check" in r.rule_id]
        assert len(range_errors) == 2
    
    def test_business_rules_pattern_match(self, sample_config, mock_logger):
        """Test business rules pattern matching validation."""
        config = sample_config["validators"]["schema_validator"].copy()
        config["business_rules"] = [
            {
                "id": "sa2_pattern_check",
                "type": "pattern_match",
                "column": "sa2_code",
                "pattern": r"^[0-9]{9}$"
            }
        ]
        
        validator = ConcreteValidator("test", config, mock_logger)
        
        # Data with invalid patterns
        invalid_pattern_data = [
            {"sa2_code": "10101", "value": 25.5, "year": 2021},        # Too short
            {"sa2_code": "ABC123456", "value": 30.2, "year": 2021}     # Contains letters
        ]
        
        results = validator.validate_business_rules(invalid_pattern_data)
        
        # Should detect pattern violations
        pattern_errors = [r for r in results if "pattern_check" in r.rule_id]
        assert len(pattern_errors) == 2
    
    def test_business_rules_uniqueness_check(self, sample_config, mock_logger):
        """Test business rules uniqueness validation."""
        config = sample_config["validators"]["schema_validator"].copy()
        config["business_rules"] = [
            {
                "id": "sa2_uniqueness_check",
                "type": "uniqueness_check",
                "column": "sa2_code"
            }
        ]
        
        validator = ConcreteValidator("test", config, mock_logger)
        
        # Data with duplicate SA2 codes
        duplicate_data = [
            {"sa2_code": "101011001", "value": 25.5, "year": 2021},
            {"sa2_code": "101011001", "value": 30.2, "year": 2021},  # Duplicate
            {"sa2_code": "101011002", "value": 22.8, "year": 2021}
        ]
        
        results = validator.validate_business_rules(duplicate_data)
        
        # Should detect duplicates
        uniqueness_errors = [r for r in results if "uniqueness_check" in r.rule_id]
        assert len(uniqueness_errors) == 1
    
    def test_business_rules_completeness_check(self, sample_config, mock_logger):
        """Test business rules completeness validation."""
        config = sample_config["validators"]["schema_validator"].copy()
        config["business_rules"] = [
            {
                "id": "value_completeness_check",
                "type": "completeness_check",
                "column": "value",
                "min_completeness": 0.8  # 80% completeness required
            }
        ]
        
        validator = ConcreteValidator("test", config, mock_logger)
        
        # Data with low completeness (50% null values)
        incomplete_data = [
            {"sa2_code": "101011001", "value": 25.5, "year": 2021},
            {"sa2_code": "101011002", "value": None, "year": 2021},   # Null value
            {"sa2_code": "101011003", "value": 22.8, "year": 2021},
            {"sa2_code": "101011004", "value": None, "year": 2021}    # Null value
        ]
        
        results = validator.validate_business_rules(incomplete_data)
        
        # Should detect low completeness
        completeness_warnings = [r for r in results if "completeness_check" in r.rule_id]
        assert len(completeness_warnings) == 1
        assert completeness_warnings[0].severity == ValidationSeverity.WARNING
    
    def test_statistical_validation_outlier_detection(self, sample_config, mock_logger):
        """Test statistical outlier detection."""
        config = sample_config["validators"]["schema_validator"].copy()
        config["statistical_rules"] = {
            "outlier_detection": [
                {
                    "column": "value",
                    "method": "iqr",
                    "threshold": 1.5
                }
            ]
        }
        
        validator = ConcreteValidator("test", config, mock_logger)
        
        # Data with outliers
        data_with_outliers = [
            {"sa2_code": "101011001", "value": 25.0, "year": 2021},
            {"sa2_code": "101011002", "value": 26.0, "year": 2021},
            {"sa2_code": "101011003", "value": 27.0, "year": 2021},
            {"sa2_code": "101011004", "value": 28.0, "year": 2021},
            {"sa2_code": "101011005", "value": 100.0, "year": 2021}  # Clear outlier
        ]
        
        results = validator.validate_statistics(data_with_outliers)
        
        # Should detect outliers
        outlier_results = [r for r in results if r.rule_id == "statistical_outlier"]
        assert len(outlier_results) > 0
        assert outlier_results[0].severity == ValidationSeverity.INFO  # Outliers are info, not errors
    
    def test_geographic_validation_sa2_codes(self, sample_config, mock_logger, sample_sa2_codes):
        """Test geographic validation of SA2 codes."""
        config = sample_config["validators"]["schema_validator"].copy()
        config["geographic"] = {
            "sa2_column": "sa2_code"
        }
        
        validator = ConcreteValidator("test", config, mock_logger)
        
        # Mock valid SA2 codes
        validator.valid_sa2_codes = set(sample_sa2_codes)
        
        # Data with invalid SA2 codes
        geographic_data = [
            {"sa2_code": "101011001", "value": 25.5, "year": 2021},  # Valid
            {"sa2_code": "999999999", "value": 30.2, "year": 2021},  # Invalid
            {"sa2_code": "101011002", "value": 22.8, "year": 2021}   # Valid
        ]
        
        results = validator.validate_geography(geographic_data)
        
        # Should detect invalid SA2 codes
        geographic_errors = [r for r in results if r.rule_id == "geographic_invalid_sa2"]
        assert len(geographic_errors) == 1
    
    def test_geographic_validation_coordinates(self, sample_config, mock_logger):
        """Test geographic validation of coordinates."""
        config = sample_config["validators"]["schema_validator"].copy()
        config["geographic"] = {
            "latitude_column": "latitude",
            "longitude_column": "longitude"
        }
        
        validator = ConcreteValidator("test", config, mock_logger)
        
        # Data with invalid coordinates
        coordinate_data = [
            {"latitude": -33.8688, "longitude": 151.2093},  # Valid (Sydney)
            {"latitude": -95.0, "longitude": 151.2093},     # Invalid latitude
            {"latitude": -33.8688, "longitude": 200.0},     # Invalid longitude
            {"latitude": 45.0, "longitude": -75.0}          # Valid (Ottawa)
        ]
        
        results = validator.validate_geography(coordinate_data)
        
        # Should detect invalid coordinates
        coordinate_errors = [
            r for r in results 
            if r.rule_id in ["geographic_invalid_latitude", "geographic_invalid_longitude"]
        ]
        assert len(coordinate_errors) == 2
    
    def test_sa2_code_standardisation(self, sample_config, mock_logger):
        """Test SA2 code standardisation."""
        validator = ConcreteValidator("test", sample_config["validators"]["schema_validator"], mock_logger)
        
        # Test various SA2 code formats
        test_codes = [
            ("123456789", "123456789"),  # Already 9 digits
            ("12345678", "012345678"),   # 8 digits - should be padded
            ("1234567", "001234567"),    # 7 digits - should be padded
            (123456789, "123456789"),    # Integer input
            (12345678, "012345678")      # Integer input - should be padded
        ]
        
        for input_code, expected_output in test_codes:
            standardised = validator._standardise_sa2_code(input_code)
            assert standardised == expected_output
    
    def test_data_type_checking(self, sample_config, mock_logger):
        """Test data type checking functionality."""
        validator = ConcreteValidator("test", sample_config["validators"]["schema_validator"], mock_logger)
        
        # Test various data type checks
        type_tests = [
            ("hello", "string", True),
            (123, "string", False),
            (123, "integer", True),
            (123.45, "integer", False),
            (123.45, "float", True),
            ("123.45", "float", False),
            (True, "boolean", True),
            (1, "boolean", False),
            (datetime.now(), "datetime", True),
            ("2021-01-01", "datetime", False)
        ]
        
        for value, expected_type, should_match in type_tests:
            result = validator._check_data_type(value, expected_type)
            assert result == should_match
    
    def test_validation_summary_generation(self, sample_config, mock_logger):
        """Test validation summary generation."""
        validator = ConcreteValidator("test", sample_config["validators"]["schema_validator"], mock_logger)
        
        # Create mock validation results
        validator._validation_results = [
            ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="test_error",
                message="Test error",
                affected_records=[0, 1]
            ),
            ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                rule_id="test_warning",
                message="Test warning",
                affected_records=[2]
            ),
            ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                rule_id="test_info",
                message="Test info",
                affected_records=[]
            )
        ]
        
        summary = validator.get_validation_summary()
        
        assert summary["total_validations"] == 3
        assert summary["errors"] == 1
        assert summary["warnings"] == 1
        assert summary["info"] == 1
        assert summary["affected_records"] == 3  # Records 0, 1, 2
        assert "test_error" in summary["rules_triggered"]
        assert "test_warning" in summary["rules_triggered"]
        assert "test_info" in summary["rules_triggered"]
    
    def test_custom_validation_implementation(self, sample_config, mock_logger):
        """Test custom validation implementation."""
        validator = ConcreteValidator("test", sample_config["validators"]["schema_validator"], mock_logger)
        
        # Data with test markers
        test_data = [
            {"sa2_code": "101011001", "value": 25.5, "year": 2021},
            {"sa2_code": "101011002", "value": 30.2, "year": 2021, "_test_invalid": True}
        ]
        
        results = validator.validate(test_data)
        
        # Should detect test marker
        custom_errors = [r for r in results if r.rule_id == "test_marker_validation"]
        assert len(custom_errors) == 1
        assert custom_errors[0].affected_records == [1]
    
    def test_validation_rules_retrieval(self, sample_config, mock_logger):
        """Test validation rules retrieval."""
        validator = ConcreteValidator("test", sample_config["validators"]["schema_validator"], mock_logger)
        
        rules = validator.get_validation_rules()
        
        assert isinstance(rules, list)
        assert len(rules) == 5
        assert "schema_validation" in rules
        assert "business_rules" in rules
        assert "statistical_validation" in rules
        assert "geographic_validation" in rules
        assert "custom_validation" in rules


@pytest.mark.unit
@pytest.mark.parametrize("validation_severity", [
    ValidationSeverity.ERROR,
    ValidationSeverity.WARNING,
    ValidationSeverity.INFO
])
class TestValidationResultSeverity:
    """Test validation result severity handling."""
    
    def test_validation_severity_filtering(self, sample_config, mock_logger, validation_severity):
        """Test filtering validation results by severity."""
        validator = ConcreteValidator("test", sample_config["validators"]["schema_validator"], mock_logger)
        
        # Create mock results with different severities
        validator._validation_results = [
            ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="error_rule",
                message="Error message"
            ),
            ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                rule_id="warning_rule",
                message="Warning message"
            ),
            ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                rule_id="info_rule",
                message="Info message"
            )
        ]
        
        # Filter by severity
        filtered_results = [
            r for r in validator._validation_results 
            if r.severity == validation_severity
        ]
        
        assert len(filtered_results) == 1
        assert filtered_results[0].severity == validation_severity


@pytest.mark.unit
@pytest.mark.slow
class TestValidatorPerformance:
    """Performance-related tests for validators."""
    
    def test_large_dataset_validation(self, sample_config, mock_logger, performance_data_large):
        """Test validation performance with large dataset."""
        validator = ConcreteValidator("test", sample_config["validators"]["schema_validator"], mock_logger)
        
        start_time = datetime.now()
        
        results = validator.validate_comprehensive(performance_data_large)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        assert isinstance(results, list)
        assert duration < 30.0  # Should complete within 30 seconds
    
    def test_validation_memory_efficiency(self, sample_config, mock_logger, memory_intensive_data):
        """Test memory efficiency with large records."""
        validator = ConcreteValidator("test", sample_config["validators"]["schema_validator"], mock_logger)
        
        # Should handle large records without memory issues
        results = validator.validate_comprehensive(memory_intensive_data)
        
        assert isinstance(results, list)


@pytest.mark.unit
class TestValidatorEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_none_data_handling(self, sample_config, mock_logger):
        """Test handling of None data."""
        validator = ConcreteValidator("test", sample_config["validators"]["schema_validator"], mock_logger)
        
        with pytest.raises(TypeError):
            validator.validate_comprehensive(None)
    
    def test_empty_configuration_handling(self, mock_logger):
        """Test handling of empty configuration."""
        validator = ConcreteValidator("test", {}, mock_logger)
        
        # Should use default values
        assert validator.required_columns == set()
        assert validator.column_types == {}
        assert validator.business_rules == []
    
    def test_malformed_business_rules(self, sample_config, mock_logger):
        """Test handling of malformed business rules."""
        config = sample_config["validators"]["schema_validator"].copy()
        config["business_rules"] = [
            {"id": "malformed_rule"},  # Missing required fields
            {"type": "unknown_type", "column": "test"},  # Unknown rule type
            None  # Invalid rule
        ]
        
        validator = ConcreteValidator("test", config, mock_logger)
        
        # Should handle malformed rules gracefully
        results = validator.validate_business_rules([{"test": "data"}])
        assert isinstance(results, list)
    
    def test_validator_with_no_logger(self, sample_config):
        """Test validator initialisation without logger."""
        validator = ConcreteValidator("test", sample_config["validators"]["schema_validator"])
        
        # Should create default logger
        assert validator.logger is not None
        assert validator.logger.name == "ConcreteValidator"
    
    def test_circular_validation_dependencies(self, sample_config, mock_logger):
        """Test handling of potential circular validation dependencies."""
        class CircularValidator(ConcreteValidator):
            def validate(self, data: DataBatch) -> List[ValidationResult]:
                # Simulate circular dependency by calling comprehensive validation
                try:
                    return super().validate(data)
                except RecursionError:
                    return [ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        rule_id="circular_dependency",
                        message="Circular validation dependency detected"
                    )]
        
        validator = CircularValidator("test", sample_config["validators"]["schema_validator"], mock_logger)
        
        # Should handle gracefully
        results = validator.validate([{"test": "data"}])
        assert isinstance(results, list)
    
    def test_validation_with_missing_sa2_codes_file(self, sample_config, mock_logger):
        """Test handling when SA2 codes file is missing."""
        config = sample_config["validators"]["schema_validator"].copy()
        config["geographic"] = {
            "sa2_codes_file": "/nonexistent/file.csv"
        }
        
        validator = ConcreteValidator("test", config, mock_logger)
        
        # Should handle missing file gracefully
        assert validator.valid_sa2_codes == set()
    
    def test_statistical_validation_insufficient_data(self, sample_config, mock_logger):
        """Test statistical validation with insufficient data."""
        config = sample_config["validators"]["schema_validator"].copy()
        config["statistical_rules"] = {
            "outlier_detection": [
                {
                    "column": "value",
                    "method": "iqr",
                    "threshold": 1.5
                }
            ]
        }
        
        validator = ConcreteValidator("test", config, mock_logger)
        
        # Data with too few points for statistical analysis
        insufficient_data = [
            {"value": 25.0},
            {"value": 26.0}  # Only 2 data points
        ]
        
        results = validator.validate_statistics(insufficient_data)
        
        # Should handle gracefully (no outliers detected due to insufficient data)
        outlier_results = [r for r in results if r.rule_id == "statistical_outlier"]
        assert len(outlier_results) == 0