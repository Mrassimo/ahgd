"""
Abstract base class for data validators in the AHGD ETL pipeline.

This module provides the BaseValidator class which defines the standard interface
for all data validation components.
"""

import re
import statistics
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union
import logging

from ..utils.interfaces import (
    DataBatch,
    DataRecord,
    GeographicValidationError,
    ValidationError,
    ValidationResult,
    ValidationSeverity,
)


class BaseValidator(ABC):
    """
    Abstract base class for data validators.
    
    This class provides the standard interface and common functionality
    for all data validation components in the AHGD ETL pipeline.
    """
    
    def __init__(
        self,
        validator_id: str,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialise the validator.
        
        Args:
            validator_id: Unique identifier for this validator
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.validator_id = validator_id
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Schema validation configuration
        self.schema_config = config.get('schema', {})
        self.required_columns = set(config.get('required_columns', []))
        self.column_types = config.get('column_types', {})
        
        # Business rule configuration
        self.business_rules = config.get('business_rules', [])
        
        # Statistical validation configuration
        self.statistical_rules = config.get('statistical_rules', {})
        
        # Geographic validation configuration
        self.geographic_config = config.get('geographic', {})
        self.valid_sa2_codes = self._load_valid_sa2_codes()
        
        # Validation results
        self._validation_results: List[ValidationResult] = []
    
    @abstractmethod
    def validate(self, data: DataBatch) -> List[ValidationResult]:
        """
        Validate a batch of data records.
        
        Args:
            data: Batch of data records to validate
            
        Returns:
            List[ValidationResult]: Validation results
        """
        pass
    
    @abstractmethod
    def get_validation_rules(self) -> List[str]:
        """
        Get the list of validation rules supported by this validator.
        
        Returns:
            List[str]: List of validation rule identifiers
        """
        pass
    
    def validate_comprehensive(self, data: DataBatch) -> List[ValidationResult]:
        """
        Perform comprehensive validation including all validation types.
        
        Args:
            data: Batch of data records to validate
            
        Returns:
            List[ValidationResult]: Comprehensive validation results
        """
        self._validation_results = []
        
        # Schema validation
        schema_results = self.validate_schema(data)
        self._validation_results.extend(schema_results)
        
        # Business rule validation
        business_results = self.validate_business_rules(data)
        self._validation_results.extend(business_results)
        
        # Statistical validation
        statistical_results = self.validate_statistics(data)
        self._validation_results.extend(statistical_results)
        
        # Geographic validation
        geographic_results = self.validate_geography(data)
        self._validation_results.extend(geographic_results)
        
        # Custom validation (implemented by subclasses)
        custom_results = self.validate(data)
        self._validation_results.extend(custom_results)
        
        # Log validation summary
        error_count = sum(1 for r in self._validation_results if r.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for r in self._validation_results if r.severity == ValidationSeverity.WARNING)
        
        self.logger.info(
            f"Validation completed: {len(data)} records, "
            f"{error_count} errors, {warning_count} warnings"
        )
        
        return self._validation_results
    
    def validate_schema(self, data: DataBatch) -> List[ValidationResult]:
        """
        Validate data against schema requirements.
        
        Args:
            data: Batch of data records to validate
            
        Returns:
            List[ValidationResult]: Schema validation results
        """
        results = []
        
        if not data:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="schema_empty_data",
                message="Data batch is empty"
            ))
            return results
        
        # Get all columns from the first record
        sample_record = data[0]
        actual_columns = set(sample_record.keys())
        
        # Check for missing required columns
        missing_columns = self.required_columns - actual_columns
        if missing_columns:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="schema_missing_columns",
                message=f"Missing required columns: {missing_columns}",
                details={'missing_columns': list(missing_columns)}
            ))
        
        # Check column data types
        for record_idx, record in enumerate(data):
            for column, expected_type in self.column_types.items():
                if column in record and record[column] is not None:
                    if not self._check_data_type(record[column], expected_type):
                        results.append(ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.ERROR,
                            rule_id="schema_invalid_type",
                            message=f"Invalid data type for column '{column}' in record {record_idx}",
                            details={
                                'column': column,
                                'expected_type': expected_type,
                                'actual_value': record[column],
                                'actual_type': type(record[column]).__name__
                            },
                            affected_records=[record_idx]
                        ))
        
        return results
    
    def validate_business_rules(self, data: DataBatch) -> List[ValidationResult]:
        """
        Validate data against business rules.
        
        Args:
            data: Batch of data records to validate
            
        Returns:
            List[ValidationResult]: Business rule validation results
        """
        results = []
        
        for rule in self.business_rules:
            rule_id = rule.get('id', 'unknown_rule')
            rule_type = rule.get('type')
            
            if rule_type == 'range_check':
                results.extend(self._validate_range_check(data, rule))
            elif rule_type == 'pattern_match':
                results.extend(self._validate_pattern_match(data, rule))
            elif rule_type == 'reference_check':
                results.extend(self._validate_reference_check(data, rule))
            elif rule_type == 'uniqueness_check':
                results.extend(self._validate_uniqueness_check(data, rule))
            elif rule_type == 'completeness_check':
                results.extend(self._validate_completeness_check(data, rule))
            else:
                self.logger.warning(f"Unknown business rule type: {rule_type}")
        
        return results
    
    def validate_statistics(self, data: DataBatch) -> List[ValidationResult]:
        """
        Validate data using statistical methods.
        
        Args:
            data: Batch of data records to validate
            
        Returns:
            List[ValidationResult]: Statistical validation results
        """
        results = []
        
        # Outlier detection
        outlier_columns = self.statistical_rules.get('outlier_detection', [])
        for column_config in outlier_columns:
            column = column_config['column']
            method = column_config.get('method', 'iqr')
            threshold = column_config.get('threshold', 1.5)
            
            results.extend(self._detect_outliers(data, column, method, threshold))
        
        # Distribution checks
        distribution_checks = self.statistical_rules.get('distribution_checks', [])
        for check_config in distribution_checks:
            column = check_config['column']
            expected_distribution = check_config['distribution']
            
            results.extend(self._validate_distribution(data, column, expected_distribution))
        
        # Correlation checks
        correlation_checks = self.statistical_rules.get('correlation_checks', [])
        for check_config in correlation_checks:
            column1 = check_config['column1']
            column2 = check_config['column2']
            expected_correlation = check_config['expected_correlation']
            tolerance = check_config.get('tolerance', 0.1)
            
            results.extend(self._validate_correlation(data, column1, column2, expected_correlation, tolerance))
        
        return results
    
    def validate_geography(self, data: DataBatch) -> List[ValidationResult]:
        """
        Validate geographic data, particularly SA2 codes.
        
        Args:
            data: Batch of data records to validate
            
        Returns:
            List[ValidationResult]: Geographic validation results
        """
        results = []
        
        sa2_column = self.geographic_config.get('sa2_column', 'sa2_code')
        
        if sa2_column and self.valid_sa2_codes:
            invalid_records = []
            
            for record_idx, record in enumerate(data):
                sa2_code = record.get(sa2_column)
                
                if sa2_code is not None:
                    # Standardise SA2 code format
                    standardised_code = self._standardise_sa2_code(sa2_code)
                    
                    if standardised_code not in self.valid_sa2_codes:
                        invalid_records.append(record_idx)
                        results.append(ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.ERROR,
                            rule_id="geographic_invalid_sa2",
                            message=f"Invalid SA2 code: {sa2_code}",
                            details={
                                'column': sa2_column,
                                'original_code': sa2_code,
                                'standardised_code': standardised_code
                            },
                            affected_records=[record_idx]
                        ))
        
        # Coordinate validation
        lat_column = self.geographic_config.get('latitude_column')
        lon_column = self.geographic_config.get('longitude_column')
        
        if lat_column and lon_column:
            results.extend(self._validate_coordinates(data, lat_column, lon_column))
        
        return results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the validation results.
        
        Returns:
            Dict[str, Any]: Validation summary
        """
        if not self._validation_results:
            return {'total_validations': 0}
        
        summary = {
            'total_validations': len(self._validation_results),
            'errors': sum(1 for r in self._validation_results if r.severity == ValidationSeverity.ERROR),
            'warnings': sum(1 for r in self._validation_results if r.severity == ValidationSeverity.WARNING),
            'info': sum(1 for r in self._validation_results if r.severity == ValidationSeverity.INFO),
            'rules_triggered': list(set(r.rule_id for r in self._validation_results)),
            'affected_records': len(set(
                record_id 
                for result in self._validation_results 
                for record_id in result.affected_records
            ))
        }
        
        return summary
    
    def _load_valid_sa2_codes(self) -> Set[str]:
        """
        Load valid SA2 codes from configuration or external source.
        
        Returns:
            Set[str]: Set of valid SA2 codes
        """
        # This would typically load from a file or database
        # For now, return an empty set as a placeholder
        sa2_codes_file = self.geographic_config.get('sa2_codes_file')
        if sa2_codes_file:
            try:
                # Implementation would load codes from file
                pass
            except Exception as e:
                self.logger.warning(f"Could not load SA2 codes: {e}")
        
        return set()
    
    def _check_data_type(self, value: Any, expected_type: str) -> bool:
        """
        Check if a value matches the expected data type.
        
        Args:
            value: Value to check
            expected_type: Expected data type
            
        Returns:
            bool: True if type matches
        """
        if expected_type == 'string':
            return isinstance(value, str)
        elif expected_type == 'integer':
            return isinstance(value, int)
        elif expected_type == 'float':
            return isinstance(value, (int, float))
        elif expected_type == 'boolean':
            return isinstance(value, bool)
        elif expected_type == 'datetime':
            return isinstance(value, datetime)
        else:
            return True  # Unknown type, assume valid
    
    def _validate_range_check(self, data: DataBatch, rule: Dict[str, Any]) -> List[ValidationResult]:
        """Validate range constraints."""
        results = []
        column = rule['column']
        min_value = rule.get('min')
        max_value = rule.get('max')
        
        for record_idx, record in enumerate(data):
            value = record.get(column)
            
            if value is not None:
                if min_value is not None and value < min_value:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        rule_id=rule['id'],
                        message=f"Value {value} is below minimum {min_value} for column {column}",
                        affected_records=[record_idx]
                    ))
                
                if max_value is not None and value > max_value:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        rule_id=rule['id'],
                        message=f"Value {value} is above maximum {max_value} for column {column}",
                        affected_records=[record_idx]
                    ))
        
        return results
    
    def _validate_pattern_match(self, data: DataBatch, rule: Dict[str, Any]) -> List[ValidationResult]:
        """Validate pattern matching constraints."""
        results = []
        column = rule['column']
        pattern = rule['pattern']
        
        compiled_pattern = re.compile(pattern)
        
        for record_idx, record in enumerate(data):
            value = record.get(column)
            
            if value is not None and isinstance(value, str):
                if not compiled_pattern.match(value):
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        rule_id=rule['id'],
                        message=f"Value '{value}' does not match pattern '{pattern}' for column {column}",
                        affected_records=[record_idx]
                    ))
        
        return results
    
    def _validate_reference_check(self, data: DataBatch, rule: Dict[str, Any]) -> List[ValidationResult]:
        """Validate reference integrity constraints."""
        # Implementation would check against reference data
        return []
    
    def _validate_uniqueness_check(self, data: DataBatch, rule: Dict[str, Any]) -> List[ValidationResult]:
        """Validate uniqueness constraints."""
        results = []
        column = rule['column']
        
        seen_values = {}
        for record_idx, record in enumerate(data):
            value = record.get(column)
            
            if value is not None:
                if value in seen_values:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        rule_id=rule['id'],
                        message=f"Duplicate value '{value}' found in column {column}",
                        affected_records=[seen_values[value], record_idx]
                    ))
                else:
                    seen_values[value] = record_idx
        
        return results
    
    def _validate_completeness_check(self, data: DataBatch, rule: Dict[str, Any]) -> List[ValidationResult]:
        """Validate completeness constraints."""
        results = []
        column = rule['column']
        min_completeness = rule.get('min_completeness', 1.0)
        
        non_null_count = sum(1 for record in data if record.get(column) is not None)
        completeness = non_null_count / len(data) if data else 0
        
        if completeness < min_completeness:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                rule_id=rule['id'],
                message=f"Column {column} completeness {completeness:.2%} is below threshold {min_completeness:.2%}",
                details={'completeness': completeness, 'threshold': min_completeness}
            ))
        
        return results
    
    def _detect_outliers(self, data: DataBatch, column: str, method: str, threshold: float) -> List[ValidationResult]:
        """Detect statistical outliers."""
        results = []
        values = [record.get(column) for record in data if record.get(column) is not None]
        
        if not values or len(values) < 4:
            return results
        
        if method == 'iqr':
            q1 = statistics.quantiles(values, n=4)[0]
            q3 = statistics.quantiles(values, n=4)[2]
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            for record_idx, record in enumerate(data):
                value = record.get(column)
                if value is not None and (value < lower_bound or value > upper_bound):
                    results.append(ValidationResult(
                        is_valid=True,  # Outliers are info, not errors
                        severity=ValidationSeverity.INFO,
                        rule_id="statistical_outlier",
                        message=f"Potential outlier detected in column {column}: {value}",
                        details={'method': method, 'threshold': threshold},
                        affected_records=[record_idx]
                    ))
        
        return results
    
    def _validate_distribution(self, data: DataBatch, column: str, expected_distribution: str) -> List[ValidationResult]:
        """Validate data distribution."""
        # Implementation would perform distribution tests
        return []
    
    def _validate_correlation(self, data: DataBatch, column1: str, column2: str, expected_correlation: float, tolerance: float) -> List[ValidationResult]:
        """Validate correlation between columns."""
        # Implementation would calculate correlation and compare
        return []
    
    def _validate_coordinates(self, data: DataBatch, lat_column: str, lon_column: str) -> List[ValidationResult]:
        """Validate geographic coordinates."""
        results = []
        
        for record_idx, record in enumerate(data):
            lat = record.get(lat_column)
            lon = record.get(lon_column)
            
            if lat is not None and lon is not None:
                # Validate latitude range
                if not (-90 <= lat <= 90):
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        rule_id="geographic_invalid_latitude",
                        message=f"Invalid latitude: {lat} (must be between -90 and 90)",
                        affected_records=[record_idx]
                    ))
                
                # Validate longitude range
                if not (-180 <= lon <= 180):
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        rule_id="geographic_invalid_longitude",
                        message=f"Invalid longitude: {lon} (must be between -180 and 180)",
                        affected_records=[record_idx]
                    ))
        
        return results
    
    def _standardise_sa2_code(self, sa2_code: Union[str, int]) -> str:
        """
        Standardise SA2 code format.
        
        Args:
            sa2_code: Raw SA2 code
            
        Returns:
            str: Standardised SA2 code
        """
        # Convert to string and pad with zeros if necessary
        code_str = str(sa2_code).strip()
        
        # SA2 codes are typically 9 digits
        if code_str.isdigit() and len(code_str) < 9:
            code_str = code_str.zfill(9)
        
        return code_str