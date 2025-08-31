"""
Core validator for SA1-focused AHGD data validation.

This module provides a consolidated, streamlined validation framework
that focuses on SA1 geographic validation and essential data quality checks.
It replaces the complex multi-validator orchestration with a simple,
efficient approach.
"""

import logging
import re
import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import polars as pl

from schemas.sa1_schema import SA1Coordinates, validate_sa1_hierarchy

from ..utils.interfaces import (
    DataBatch,
    ValidationError,
    ValidationResult,
    ValidationSeverity,
)
from ..utils.logging import get_logger
from .base import BaseValidator


class CoreValidator(BaseValidator):
    """
    Core data validator with SA1-focused validation.

    Consolidates essential validation functionality including:
    - SA1 code validation (11-digit format)
    - Geographic hierarchy validation (SA1->SA2->SA3->SA4)
    - Basic data quality checks
    - Statistical outlier detection
    - British English error messages
    """

    def __init__(
        self, config: Dict[str, Any] = None, logger: Optional[logging.Logger] = None
    ):
        """
        Initialise core validator.

        Args:
            config: Validation configuration
            logger: Optional logger instance
        """
        config = config or {}
        super().__init__(
            validator_id="core_sa1_validator",
            config=config,
            logger=logger or get_logger(__name__),
        )

        # Validation thresholds
        self.quality_threshold = config.get("quality_threshold", 85.0)
        self.error_threshold = config.get("error_threshold", 5.0)  # % of records
        self.outlier_threshold = config.get("outlier_threshold", 3.0)  # std deviations

        # SA1 validation patterns
        self.sa1_code_pattern = re.compile(r"^\d{11}$")
        self.sa2_code_pattern = re.compile(r"^\d{9}$")
        self.sa3_code_pattern = re.compile(r"^\d{5}$")
        self.sa4_code_pattern = re.compile(r"^\d{3}$")

        # Australian state codes mapping
        self.state_mapping = {
            "1": "NSW",
            "2": "VIC",
            "3": "QLD",
            "4": "SA",
            "5": "WA",
            "6": "TAS",
            "7": "NT",
            "8": "ACT",
        }

        self.logger.info(
            "Core SA1 validator initialised", quality_threshold=self.quality_threshold
        )

    def validate_sa1_data(self, data: pl.DataFrame) -> Dict[str, Any]:
        """
        Validate SA1-focused dataset comprehensively.

        Args:
            data: Polars DataFrame with SA1 data

        Returns:
            Dict containing validation results and quality metrics
        """
        self.logger.info(f"Validating SA1 data with {len(data)} records")

        validation_start = datetime.now()
        results = {
            "validation_timestamp": validation_start.isoformat(),
            "total_records": len(data),
            "overall_valid": True,
            "quality_score": 0.0,
            "error_count": 0,
            "warning_count": 0,
            "validation_details": {},
            "errors": [],
            "warnings": [],
            "recommendations": [],
        }

        try:
            # 1. SA1 Code Validation
            sa1_results = self._validate_sa1_codes(data)
            results["validation_details"]["sa1_codes"] = sa1_results
            results["error_count"] += sa1_results.get("error_count", 0)
            results["warnings"].extend(sa1_results.get("warnings", []))

            # 2. Geographic Hierarchy Validation
            hierarchy_results = self._validate_geographic_hierarchy(data)
            results["validation_details"]["hierarchy"] = hierarchy_results
            results["error_count"] += hierarchy_results.get("error_count", 0)
            results["warnings"].extend(hierarchy_results.get("warnings", []))

            # 3. Data Quality Validation
            quality_results = self._validate_data_quality(data)
            results["validation_details"]["data_quality"] = quality_results
            results["error_count"] += quality_results.get("error_count", 0)
            results["warnings"].extend(quality_results.get("warnings", []))

            # 4. Statistical Validation
            statistical_results = self._validate_statistical_consistency(data)
            results["validation_details"]["statistics"] = statistical_results
            results["warning_count"] += statistical_results.get("warning_count", 0)
            results["warnings"].extend(statistical_results.get("warnings", []))

            # Calculate overall quality score
            results["quality_score"] = self._calculate_overall_quality_score(results)

            # Determine if validation passed
            error_rate = (results["error_count"] / results["total_records"]) * 100
            results["overall_valid"] = (
                results["quality_score"] >= self.quality_threshold
                and error_rate <= self.error_threshold
            )

            # Generate recommendations
            results["recommendations"] = self._generate_recommendations(results)

            validation_duration = (datetime.now() - validation_start).total_seconds()
            results["validation_duration_seconds"] = validation_duration

            self.logger.info(
                f"SA1 validation completed",
                quality_score=results["quality_score"],
                error_count=results["error_count"],
                duration=validation_duration,
            )

            return results

        except Exception as e:
            self.logger.error(f"SA1 validation failed: {str(e)}")
            results.update(
                {
                    "overall_valid": False,
                    "validation_error": str(e),
                    "error_count": results[
                        "total_records"
                    ],  # All records considered invalid
                }
            )
            return results

    def _validate_sa1_codes(self, data: pl.DataFrame) -> Dict[str, Any]:
        """Validate SA1 code format and structure."""
        results = {
            "valid_codes": 0,
            "invalid_codes": 0,
            "error_count": 0,
            "warnings": [],
            "invalid_records": [],
        }

        if "sa1_code" not in data.columns:
            results["error_count"] = len(data)
            results["warnings"].append("No SA1 code column found in data")
            return results

        sa1_codes = data.get_column("sa1_code").to_list()

        for i, code in enumerate(sa1_codes):
            if not code or not isinstance(code, str):
                results["invalid_codes"] += 1
                results["invalid_records"].append(
                    f"Record {i}: Empty or invalid SA1 code"
                )
                continue

            # Validate format (11 digits)
            if not self.sa1_code_pattern.match(code):
                results["invalid_codes"] += 1
                results["invalid_records"].append(
                    f"Record {i}: Invalid SA1 code format '{code}' (must be 11 digits)"
                )
                continue

            # Validate state code (first digit)
            state_digit = code[0]
            if state_digit not in self.state_mapping:
                results["invalid_codes"] += 1
                results["invalid_records"].append(
                    f"Record {i}: Invalid state code '{state_digit}' in SA1 '{code}'"
                )
                continue

            results["valid_codes"] += 1

        results["error_count"] = results["invalid_codes"]

        # Generate warnings for high error rates
        if results["invalid_codes"] > 0:
            error_rate = (results["invalid_codes"] / len(data)) * 100
            if error_rate > 10:
                results["warnings"].append(
                    f"High SA1 code error rate: {error_rate:.1f}%"
                )

        return results

    def _validate_geographic_hierarchy(self, data: pl.DataFrame) -> Dict[str, Any]:
        """Validate SA1->SA2->SA3->SA4 geographic hierarchy consistency."""
        results = {
            "consistent_hierarchies": 0,
            "inconsistent_hierarchies": 0,
            "error_count": 0,
            "warnings": [],
            "hierarchy_errors": [],
        }

        required_columns = ["sa1_code", "sa2_code", "sa3_code", "sa4_code"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            results["error_count"] = len(data)
            results["warnings"].append(f"Missing hierarchy columns: {missing_columns}")
            return results

        for i, row in enumerate(data.iter_rows(named=True)):
            sa1_code = row.get("sa1_code", "")
            sa2_code = row.get("sa2_code", "")
            sa3_code = row.get("sa3_code", "")
            sa4_code = row.get("sa4_code", "")

            hierarchy_errors = []

            # Validate SA1 contains SA2 (first 9 digits)
            if sa1_code and sa2_code:
                if not sa1_code.startswith(sa2_code):
                    hierarchy_errors.append(
                        f"SA1 '{sa1_code}' not contained in SA2 '{sa2_code}'"
                    )

            # Validate SA2 contains SA3 (first 5 digits)
            if sa2_code and sa3_code:
                if not sa2_code.startswith(sa3_code):
                    hierarchy_errors.append(
                        f"SA2 '{sa2_code}' not contained in SA3 '{sa3_code}'"
                    )

            # Validate SA3 contains SA4 (first 3 digits)
            if sa3_code and sa4_code:
                if not sa3_code.startswith(sa4_code):
                    hierarchy_errors.append(
                        f"SA3 '{sa3_code}' not contained in SA4 '{sa4_code}'"
                    )

            if hierarchy_errors:
                results["inconsistent_hierarchies"] += 1
                results["hierarchy_errors"].append(
                    f"Record {i}: {'; '.join(hierarchy_errors)}"
                )
            else:
                results["consistent_hierarchies"] += 1

        results["error_count"] = results["inconsistent_hierarchies"]

        return results

    def _validate_data_quality(self, data: pl.DataFrame) -> Dict[str, Any]:
        """Validate general data quality metrics."""
        results = {
            "completeness_score": 0.0,
            "uniqueness_score": 0.0,
            "validity_score": 0.0,
            "error_count": 0,
            "warnings": [],
        }

        total_cells = len(data) * len(data.columns)
        null_cells = 0

        # Check for null values
        for column in data.columns:
            null_count = data.get_column(column).null_count()
            null_cells += null_count

            if null_count > 0:
                null_rate = (null_count / len(data)) * 100
                if null_rate > 20:  # More than 20% null
                    results["warnings"].append(
                        f"High null rate in '{column}': {null_rate:.1f}%"
                    )

        # Completeness score
        results["completeness_score"] = ((total_cells - null_cells) / total_cells) * 100

        # Check SA1 uniqueness
        if "sa1_code" in data.columns:
            unique_sa1s = data.get_column("sa1_code").n_unique()
            total_sa1s = len(data)
            results["uniqueness_score"] = (unique_sa1s / total_sa1s) * 100

            if results["uniqueness_score"] < 95:
                results["warnings"].append(
                    f"Duplicate SA1 codes detected: {100 - results['uniqueness_score']:.1f}% duplicates"
                )

        # Validity checks for numeric columns
        numeric_columns = []
        for column in data.columns:
            if data.get_column(column).dtype in [
                pl.Int64,
                pl.Int32,
                pl.Float64,
                pl.Float32,
            ]:
                numeric_columns.append(column)

                # Check for negative values in population/dwelling counts
                if "population" in column.lower() or "dwelling" in column.lower():
                    negative_count = (data.get_column(column) < 0).sum()
                    if negative_count > 0:
                        results["warnings"].append(
                            f"Negative values in '{column}': {negative_count} records"
                        )

        # Overall validity score (inverse of warning rate)
        warning_rate = len(results["warnings"]) / max(len(data.columns), 1)
        results["validity_score"] = max(0, 100 - (warning_rate * 20))

        return results

    def _validate_statistical_consistency(self, data: pl.DataFrame) -> Dict[str, Any]:
        """Validate statistical consistency and detect outliers."""
        results = {
            "outliers_detected": 0,
            "statistical_warnings": [],
            "warning_count": 0,
            "warnings": [],
        }

        # Check population and dwelling statistics if available
        if "population" in data.columns:
            pop_stats = self._detect_outliers(
                data.get_column("population").to_list(), "population"
            )
            results["outliers_detected"] += pop_stats["outlier_count"]
            results["statistical_warnings"].extend(pop_stats["warnings"])

        if "dwellings" in data.columns:
            dwelling_stats = self._detect_outliers(
                data.get_column("dwellings").to_list(), "dwellings"
            )
            results["outliers_detected"] += dwelling_stats["outlier_count"]
            results["statistical_warnings"].extend(dwelling_stats["warnings"])

        # Check population density if area is available
        if "population" in data.columns and "area_sq_km" in data.columns:
            # Calculate density and check for outliers
            densities = []
            for row in data.iter_rows(named=True):
                pop = row.get("population", 0)
                area = row.get("area_sq_km", 0)
                if area > 0:
                    density = pop / area
                    densities.append(density)

            if densities:
                density_stats = self._detect_outliers(densities, "population_density")
                results["outliers_detected"] += density_stats["outlier_count"]
                results["statistical_warnings"].extend(density_stats["warnings"])

        results["warnings"] = results["statistical_warnings"]
        results["warning_count"] = len(results["warnings"])

        return results

    def _detect_outliers(self, values: List[float], field_name: str) -> Dict[str, Any]:
        """Detect statistical outliers using z-score method."""
        results = {"outlier_count": 0, "warnings": []}

        if not values or len(values) < 3:
            return results

        # Remove None/null values
        clean_values = [v for v in values if v is not None and not np.isnan(v)]

        if len(clean_values) < 3:
            return results

        try:
            mean_val = statistics.mean(clean_values)
            std_val = statistics.stdev(clean_values)

            if std_val == 0:
                return results

            outliers = []
            for i, value in enumerate(clean_values):
                z_score = abs((value - mean_val) / std_val)
                if z_score > self.outlier_threshold:
                    outliers.append((i, value, z_score))

            results["outlier_count"] = len(outliers)

            if outliers:
                outlier_rate = (len(outliers) / len(clean_values)) * 100
                results["warnings"].append(
                    f"Outliers detected in {field_name}: {len(outliers)} values ({outlier_rate:.1f}%)"
                )

                # Log extreme outliers
                extreme_outliers = [o for o in outliers if o[2] > 5.0]  # z-score > 5
                if extreme_outliers:
                    results["warnings"].append(
                        f"Extreme outliers in {field_name}: {len(extreme_outliers)} values (z-score > 5)"
                    )

        except Exception as e:
            results["warnings"].append(
                f"Statistical analysis failed for {field_name}: {str(e)}"
            )

        return results

    def _calculate_overall_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        scores = []

        # SA1 code validity score
        sa1_details = results["validation_details"].get("sa1_codes", {})
        total_sa1 = sa1_details.get("valid_codes", 0) + sa1_details.get(
            "invalid_codes", 0
        )
        if total_sa1 > 0:
            sa1_score = (sa1_details.get("valid_codes", 0) / total_sa1) * 100
            scores.append(sa1_score * 0.3)  # 30% weight

        # Hierarchy consistency score
        hierarchy_details = results["validation_details"].get("hierarchy", {})
        total_hierarchy = hierarchy_details.get(
            "consistent_hierarchies", 0
        ) + hierarchy_details.get("inconsistent_hierarchies", 0)
        if total_hierarchy > 0:
            hierarchy_score = (
                hierarchy_details.get("consistent_hierarchies", 0) / total_hierarchy
            ) * 100
            scores.append(hierarchy_score * 0.3)  # 30% weight

        # Data quality scores
        quality_details = results["validation_details"].get("data_quality", {})
        completeness_score = quality_details.get("completeness_score", 0)
        uniqueness_score = quality_details.get("uniqueness_score", 0)
        validity_score = quality_details.get("validity_score", 0)

        scores.extend(
            [
                completeness_score * 0.2,  # 20% weight
                uniqueness_score * 0.1,  # 10% weight
                validity_score * 0.1,  # 10% weight
            ]
        )

        return sum(scores) if scores else 0.0

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []

        # SA1 code recommendations
        sa1_details = results["validation_details"].get("sa1_codes", {})
        if sa1_details.get("invalid_codes", 0) > 0:
            recommendations.append(
                "Review and correct invalid SA1 codes using ABS correspondence files"
            )

        # Hierarchy recommendations
        hierarchy_details = results["validation_details"].get("hierarchy", {})
        if hierarchy_details.get("inconsistent_hierarchies", 0) > 0:
            recommendations.append(
                "Validate geographic hierarchy using ABS geographic correspondences"
            )

        # Data quality recommendations
        quality_details = results["validation_details"].get("data_quality", {})
        if quality_details.get("completeness_score", 100) < 90:
            recommendations.append(
                "Improve data completeness by addressing missing values"
            )

        if quality_details.get("uniqueness_score", 100) < 95:
            recommendations.append(
                "Remove duplicate SA1 records or investigate data source issues"
            )

        # Statistical recommendations
        statistical_details = results["validation_details"].get("statistics", {})
        if statistical_details.get("outliers_detected", 0) > 0:
            recommendations.append("Investigate statistical outliers for data accuracy")

        # Overall quality recommendations
        if results["quality_score"] < 85:
            recommendations.append(
                "Overall data quality requires improvement before processing"
            )

        return recommendations

    def validate_single_sa1(self, sa1_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single SA1 record.

        Args:
            sa1_data: Dictionary containing SA1 data

        Returns:
            Dict containing validation results
        """
        try:
            # Try to create SA1Coordinates object for validation
            sa1_obj = SA1Coordinates(**sa1_data)
            integrity_errors = sa1_obj.validate_data_integrity()

            return {
                "valid": len(integrity_errors) == 0,
                "errors": integrity_errors,
                "sa1_code": sa1_obj.sa1_code,
                "hierarchy_valid": True,  # If object creation succeeded, hierarchy is valid
            }

        except Exception as e:
            return {
                "valid": False,
                "errors": [f"SA1 validation failed: {str(e)}"],
                "sa1_code": sa1_data.get("sa1_code", "UNKNOWN"),
                "hierarchy_valid": False,
            }

    # Implement abstract methods from BaseValidator

    def validate(self, data: DataBatch) -> List[ValidationResult]:
        """
        Validate a batch of data records (required by BaseValidator).

        Args:
            data: DataBatch containing records to validate

        Returns:
            List[ValidationResult]: Validation results for each record
        """
        results = []

        for i, record in enumerate(data.records):
            try:
                # Convert record to SA1 format if possible
                sa1_record = record.data if hasattr(record, "data") else record
                validation = self.validate_single_sa1(sa1_record)

                result = ValidationResult(
                    record_id=(
                        record.record_id if hasattr(record, "record_id") else str(i)
                    ),
                    is_valid=validation["valid"],
                    errors=[
                        ValidationError(
                            field_name="sa1_validation",
                            error_message=error,
                            severity=ValidationSeverity.ERROR,
                        )
                        for error in validation["errors"]
                    ],
                    warnings=[],
                    metadata={"sa1_code": validation["sa1_code"]},
                )
                results.append(result)

            except Exception as e:
                # Create error result for failed validation
                error_result = ValidationResult(
                    record_id=str(i),
                    is_valid=False,
                    errors=[
                        ValidationError(
                            field_name="validation_exception",
                            error_message=f"Validation failed: {str(e)}",
                            severity=ValidationSeverity.ERROR,
                        )
                    ],
                    warnings=[],
                    metadata={},
                )
                results.append(error_result)

        return results

    def get_validation_rules(self) -> List[str]:
        """
        Get the list of validation rules supported by this validator.

        Returns:
            List[str]: List of validation rule names
        """
        return [
            "sa1_code_format",  # 11-digit SA1 code format validation
            "sa1_state_code",  # Valid Australian state code in SA1
            "geographic_hierarchy",  # SA1->SA2->SA3->SA4 hierarchy consistency
            "data_completeness",  # Missing value detection
            "data_uniqueness",  # Duplicate SA1 code detection
            "population_range",  # Population within expected SA1 range (200-800)
            "dwelling_consistency",  # Dwelling counts consistency with population
            "coordinate_bounds",  # Australian coordinate bounds validation
            "statistical_outliers",  # Statistical outlier detection
            "british_english_validation",  # British English field names and messages
        ]
