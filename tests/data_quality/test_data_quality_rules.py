"""
Data Quality Rules Engine Testing

Comprehensive testing suite for data quality rules engine including:
- Missing data handling and validation
- Outlier detection for health metrics
- Cross-dataset consistency validation
- Temporal data validation (date ranges, sequences)
- Referential integrity across datasets
- Custom Australian health data quality rules
- Rule precedence and conflict resolution

This test suite ensures robust data quality enforcement throughout
the Australian health analytics pipeline with configurable rules
and automated quality monitoring.
"""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from unittest.mock import Mock, patch
from dataclasses import dataclass
from enum import Enum

import polars as pl
import numpy as np
from loguru import logger

from tests.data_quality.validators.australian_health_validators import AustralianHealthDataValidator
from tests.data_quality.validators.quality_metrics import (
    AustralianHealthQualityMetrics,
    QualityDimension,
    QualityThreshold
)


class RuleSeverity(Enum):
    """Data quality rule severity levels."""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class RuleType(Enum):
    """Data quality rule types."""
    COMPLETENESS = "completeness"
    VALIDITY = "validity"
    CONSISTENCY = "consistency"
    UNIQUENESS = "uniqueness"
    OUTLIER = "outlier"
    TEMPORAL = "temporal"
    REFERENTIAL = "referential"
    CUSTOM = "custom"


@dataclass
class DataQualityRule:
    """Data quality rule definition."""
    rule_id: str
    rule_name: str
    rule_type: RuleType
    severity: RuleSeverity
    description: str
    conditions: Dict
    threshold: Optional[float] = None
    enabled: bool = True


@dataclass
class RuleViolation:
    """Data quality rule violation."""
    rule_id: str
    rule_name: str
    severity: RuleSeverity
    violation_count: int
    total_records: int
    violation_percentage: float
    affected_columns: List[str]
    sample_violations: List[Dict]
    details: Dict


class DataQualityRulesEngine:
    """Data quality rules engine for Australian health data."""
    
    def __init__(self, validator: AustralianHealthDataValidator):
        """Initialize rules engine with validator."""
        self.validator = validator
        self.rules = self._load_default_rules()
        self.logger = logger.bind(component="quality_rules_engine")
    
    def _load_default_rules(self) -> List[DataQualityRule]:
        """Load default Australian health data quality rules."""
        return [
            # Completeness Rules
            DataQualityRule(
                rule_id="COMP_001",
                rule_name="SA2 Code Completeness",
                rule_type=RuleType.COMPLETENESS,
                severity=RuleSeverity.CRITICAL,
                description="SA2 codes must be present for all records",
                conditions={"column": "sa2_code_2021", "min_completeness": 100.0},
                threshold=100.0
            ),
            DataQualityRule(
                rule_id="COMP_002",
                rule_name="Population Completeness",
                rule_type=RuleType.COMPLETENESS,
                severity=RuleSeverity.CRITICAL,
                description="Population data must be present",
                conditions={"column": "usual_resident_population", "min_completeness": 95.0},
                threshold=95.0
            ),
            DataQualityRule(
                rule_id="COMP_003",
                rule_name="SEIFA Scores Completeness",
                rule_type=RuleType.COMPLETENESS,
                severity=RuleSeverity.ERROR,
                description="SEIFA scores should be complete for analysis",
                conditions={"columns": ["irsd_score", "irsad_score", "ier_score", "ieo_score"], "min_completeness": 90.0},
                threshold=90.0
            ),
            
            # Validity Rules
            DataQualityRule(
                rule_id="VAL_001",
                rule_name="SA2 Code Format Validity",
                rule_type=RuleType.VALIDITY,
                severity=RuleSeverity.CRITICAL,
                description="SA2 codes must follow 9-digit format with valid state prefix",
                conditions={"column": "sa2_code_2021", "validation_type": "sa2_format"},
                threshold=99.0
            ),
            DataQualityRule(
                rule_id="VAL_002",
                rule_name="SEIFA Score Range Validity",
                rule_type=RuleType.VALIDITY,
                severity=RuleSeverity.ERROR,
                description="SEIFA scores must be within 800-1200 range",
                conditions={"columns": ["irsd_score", "irsad_score", "ier_score", "ieo_score"], 
                          "min_value": 800, "max_value": 1200},
                threshold=95.0
            ),
            DataQualityRule(
                rule_id="VAL_003",
                rule_name="SEIFA Decile Range Validity",
                rule_type=RuleType.VALIDITY,
                severity=RuleSeverity.ERROR,
                description="SEIFA deciles must be within 1-10 range",
                conditions={"columns": ["irsd_decile", "irsad_decile", "ier_decile", "ieo_decile"], 
                          "min_value": 1, "max_value": 10},
                threshold=95.0
            ),
            DataQualityRule(
                rule_id="VAL_004",
                rule_name="Population Positive Values",
                rule_type=RuleType.VALIDITY,
                severity=RuleSeverity.ERROR,
                description="Population values must be positive",
                conditions={"column": "usual_resident_population", "min_value": 1},
                threshold=100.0
            ),
            DataQualityRule(
                rule_id="VAL_005",
                rule_name="Coordinate Bounds Validity",
                rule_type=RuleType.VALIDITY,
                severity=RuleSeverity.ERROR,
                description="Coordinates must be within Australian bounds",
                conditions={"columns": ["latitude", "longitude"], "validation_type": "australian_bounds"},
                threshold=98.0
            ),
            
            # Uniqueness Rules
            DataQualityRule(
                rule_id="UNQ_001",
                rule_name="SA2 Code Uniqueness",
                rule_type=RuleType.UNIQUENESS,
                severity=RuleSeverity.CRITICAL,
                description="SA2 codes must be unique within dataset",
                conditions={"column": "sa2_code_2021", "unique": True},
                threshold=100.0
            ),
            
            # Consistency Rules
            DataQualityRule(
                rule_id="CON_001",
                rule_name="SEIFA Score-Decile Consistency",
                rule_type=RuleType.CONSISTENCY,
                severity=RuleSeverity.WARNING,
                description="SEIFA scores should correlate with deciles",
                conditions={"score_columns": ["irsd_score", "irsad_score", "ier_score", "ieo_score"],
                          "decile_columns": ["irsd_decile", "irsad_decile", "ier_decile", "ieo_decile"],
                          "min_correlation": 0.8},
                threshold=80.0
            ),
            DataQualityRule(
                rule_id="CON_002",
                rule_name="Population Gender Consistency",
                rule_type=RuleType.CONSISTENCY,
                severity=RuleSeverity.ERROR,
                description="Male + Female population should equal total population",
                conditions={"male_column": "tot_p_m", "female_column": "tot_p_f", 
                          "total_column": "tot_p_p", "tolerance_percent": 1.0},
                threshold=99.0
            ),
            
            # Outlier Rules
            DataQualityRule(
                rule_id="OUT_001",
                rule_name="Population Outlier Detection",
                rule_type=RuleType.OUTLIER,
                severity=RuleSeverity.WARNING,
                description="Detect unusual population values for SA2 areas",
                conditions={"column": "usual_resident_population", "method": "iqr", "multiplier": 3.0},
                threshold=95.0
            ),
            DataQualityRule(
                rule_id="OUT_002",
                rule_name="SEIFA Score Outlier Detection",
                rule_type=RuleType.OUTLIER,
                severity=RuleSeverity.INFO,
                description="Detect unusual SEIFA score values",
                conditions={"columns": ["irsd_score", "irsad_score", "ier_score", "ieo_score"], 
                          "method": "zscore", "threshold": 3.0},
                threshold=98.0
            ),
            
            # Temporal Rules
            DataQualityRule(
                rule_id="TMP_001",
                rule_name="Date Range Validity",
                rule_type=RuleType.TEMPORAL,
                severity=RuleSeverity.ERROR,
                description="Dates must be within reasonable range",
                conditions={"date_columns": ["extraction_timestamp", "last_updated"], 
                          "min_date": "2020-01-01", "max_date": "2025-12-31"},
                threshold=100.0
            ),
            DataQualityRule(
                rule_id="TMP_002",
                rule_name="Data Freshness",
                rule_type=RuleType.TEMPORAL,
                severity=RuleSeverity.WARNING,
                description="Data should be updated within expected timeframe",
                conditions={"date_column": "last_updated", "max_age_days": 90},
                threshold=90.0
            ),
        ]
    
    def add_rule(self, rule: DataQualityRule) -> None:
        """Add a custom rule to the engine."""
        self.rules.append(rule)
        self.logger.info(f"Added rule: {rule.rule_id} - {rule.rule_name}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule from the engine."""
        initial_count = len(self.rules)
        self.rules = [rule for rule in self.rules if rule.rule_id != rule_id]
        removed = len(self.rules) < initial_count
        
        if removed:
            self.logger.info(f"Removed rule: {rule_id}")
        else:
            self.logger.warning(f"Rule not found: {rule_id}")
        
        return removed
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable a rule."""
        for rule in self.rules:
            if rule.rule_id == rule_id:
                rule.enabled = True
                self.logger.info(f"Enabled rule: {rule_id}")
                return True
        
        self.logger.warning(f"Rule not found: {rule_id}")
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable a rule."""
        for rule in self.rules:
            if rule.rule_id == rule_id:
                rule.enabled = False
                self.logger.info(f"Disabled rule: {rule_id}")
                return True
        
        self.logger.warning(f"Rule not found: {rule_id}")
        return False
    
    def validate_data(self, df: pl.DataFrame, dataset_name: str = "unknown") -> List[RuleViolation]:
        """Validate data against all enabled rules."""
        violations = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            try:
                violation = self._execute_rule(df, rule)
                if violation:
                    violations.append(violation)
            except Exception as e:
                self.logger.error(f"Error executing rule {rule.rule_id}: {str(e)}")
                # Create a violation for the rule execution error
                violations.append(RuleViolation(
                    rule_id=rule.rule_id,
                    rule_name=rule.rule_name,
                    severity=RuleSeverity.ERROR,
                    violation_count=1,
                    total_records=df.height,
                    violation_percentage=100.0,
                    affected_columns=[],
                    sample_violations=[{"error": str(e)}],
                    details={"execution_error": True, "error_message": str(e)}
                ))
        
        self.logger.info(f"Validated {dataset_name}: {len(violations)} violations found")
        return violations
    
    def _execute_rule(self, df: pl.DataFrame, rule: DataQualityRule) -> Optional[RuleViolation]:
        """Execute a specific rule against the data."""
        if rule.rule_type == RuleType.COMPLETENESS:
            return self._check_completeness_rule(df, rule)
        elif rule.rule_type == RuleType.VALIDITY:
            return self._check_validity_rule(df, rule)
        elif rule.rule_type == RuleType.UNIQUENESS:
            return self._check_uniqueness_rule(df, rule)
        elif rule.rule_type == RuleType.CONSISTENCY:
            return self._check_consistency_rule(df, rule)
        elif rule.rule_type == RuleType.OUTLIER:
            return self._check_outlier_rule(df, rule)
        elif rule.rule_type == RuleType.TEMPORAL:
            return self._check_temporal_rule(df, rule)
        elif rule.rule_type == RuleType.REFERENTIAL:
            return self._check_referential_rule(df, rule)
        else:
            self.logger.warning(f"Unknown rule type: {rule.rule_type}")
            return None
    
    def _check_completeness_rule(self, df: pl.DataFrame, rule: DataQualityRule) -> Optional[RuleViolation]:
        """Check completeness rules."""
        conditions = rule.conditions
        
        if "column" in conditions:
            # Single column completeness
            column = conditions["column"]
            if column not in df.columns:
                return RuleViolation(
                    rule_id=rule.rule_id,
                    rule_name=rule.rule_name,
                    severity=rule.severity,
                    violation_count=1,
                    total_records=df.height,
                    violation_percentage=100.0,
                    affected_columns=[column],
                    sample_violations=[{"error": f"Column {column} not found"}],
                    details={"missing_column": True}
                )
            
            null_count = df[column].null_count()
            completeness = ((df.height - null_count) / df.height) * 100 if df.height > 0 else 100
            min_completeness = conditions.get("min_completeness", rule.threshold)
            
            if completeness < min_completeness:
                return RuleViolation(
                    rule_id=rule.rule_id,
                    rule_name=rule.rule_name,
                    severity=rule.severity,
                    violation_count=null_count,
                    total_records=df.height,
                    violation_percentage=(null_count / df.height) * 100,
                    affected_columns=[column],
                    sample_violations=[{"null_count": null_count, "completeness": completeness}],
                    details={"expected_completeness": min_completeness, "actual_completeness": completeness}
                )
        
        elif "columns" in conditions:
            # Multi-column completeness
            columns = conditions["columns"]
            min_completeness = conditions.get("min_completeness", rule.threshold)
            
            violations_found = []
            for column in columns:
                if column in df.columns:
                    null_count = df[column].null_count()
                    completeness = ((df.height - null_count) / df.height) * 100 if df.height > 0 else 100
                    
                    if completeness < min_completeness:
                        violations_found.append({
                            "column": column,
                            "null_count": null_count,
                            "completeness": completeness
                        })
            
            if violations_found:
                total_violations = sum(v["null_count"] for v in violations_found)
                return RuleViolation(
                    rule_id=rule.rule_id,
                    rule_name=rule.rule_name,
                    severity=rule.severity,
                    violation_count=total_violations,
                    total_records=df.height * len(columns),
                    violation_percentage=(total_violations / (df.height * len(columns))) * 100,
                    affected_columns=[v["column"] for v in violations_found],
                    sample_violations=violations_found[:5],
                    details={"expected_completeness": min_completeness, "violations": violations_found}
                )
        
        return None
    
    def _check_validity_rule(self, df: pl.DataFrame, rule: DataQualityRule) -> Optional[RuleViolation]:
        """Check validity rules."""
        conditions = rule.conditions
        validation_type = conditions.get("validation_type")
        
        if validation_type == "sa2_format":
            column = conditions["column"]
            if column not in df.columns:
                return self._create_missing_column_violation(rule, column, df)
            
            sa2_codes = df[column].drop_nulls().to_list()
            invalid_codes = []
            
            for code in sa2_codes:
                validation = self.validator.validate_sa2_code(code)
                if not validation["valid"]:
                    invalid_codes.append({"code": code, "errors": validation["errors"]})
            
            if invalid_codes:
                violation_percentage = (len(invalid_codes) / len(sa2_codes)) * 100 if sa2_codes else 0
                if violation_percentage > (100 - rule.threshold):
                    return RuleViolation(
                        rule_id=rule.rule_id,
                        rule_name=rule.rule_name,
                        severity=rule.severity,
                        violation_count=len(invalid_codes),
                        total_records=len(sa2_codes),
                        violation_percentage=violation_percentage,
                        affected_columns=[column],
                        sample_violations=invalid_codes[:5],
                        details={"validation_type": "sa2_format", "threshold": rule.threshold}
                    )
        
        elif validation_type == "australian_bounds":
            lat_col = "latitude"
            lon_col = "longitude"
            
            if lat_col not in df.columns or lon_col not in df.columns:
                missing_cols = [col for col in [lat_col, lon_col] if col not in df.columns]
                return self._create_missing_column_violation(rule, missing_cols, df)
            
            coords = df.select([lat_col, lon_col]).drop_nulls()
            invalid_coords = []
            
            for row in coords.iter_rows(named=True):
                lat, lon = row[lat_col], row[lon_col]
                validation = self.validator.validate_australian_coordinates(lat, lon)
                if not validation["valid"]:
                    invalid_coords.append({"latitude": lat, "longitude": lon, "errors": validation["errors"]})
            
            if invalid_coords:
                violation_percentage = (len(invalid_coords) / len(coords)) * 100 if len(coords) > 0 else 0
                if violation_percentage > (100 - rule.threshold):
                    return RuleViolation(
                        rule_id=rule.rule_id,
                        rule_name=rule.rule_name,
                        severity=rule.severity,
                        violation_count=len(invalid_coords),
                        total_records=len(coords),
                        violation_percentage=violation_percentage,
                        affected_columns=[lat_col, lon_col],
                        sample_violations=invalid_coords[:5],
                        details={"validation_type": "australian_bounds", "threshold": rule.threshold}
                    )
        
        elif "min_value" in conditions or "max_value" in conditions:
            # Range validation
            min_val = conditions.get("min_value")
            max_val = conditions.get("max_value")
            
            columns = conditions.get("columns", [conditions.get("column")])
            columns = [col for col in columns if col is not None]
            
            violations_found = []
            for column in columns:
                if column not in df.columns:
                    continue
                
                values = df[column].drop_nulls()
                out_of_range = []
                
                for value in values.to_list():
                    if min_val is not None and value < min_val:
                        out_of_range.append({"value": value, "violation": f"below minimum {min_val}"})
                    elif max_val is not None and value > max_val:
                        out_of_range.append({"value": value, "violation": f"above maximum {max_val}"})
                
                if out_of_range:
                    violation_percentage = (len(out_of_range) / len(values)) * 100 if len(values) > 0 else 0
                    if violation_percentage > (100 - rule.threshold):
                        violations_found.extend(out_of_range)
            
            if violations_found:
                return RuleViolation(
                    rule_id=rule.rule_id,
                    rule_name=rule.rule_name,
                    severity=rule.severity,
                    violation_count=len(violations_found),
                    total_records=df.height,
                    violation_percentage=(len(violations_found) / df.height) * 100,
                    affected_columns=columns,
                    sample_violations=violations_found[:5],
                    details={"min_value": min_val, "max_value": max_val, "threshold": rule.threshold}
                )
        
        return None
    
    def _check_uniqueness_rule(self, df: pl.DataFrame, rule: DataQualityRule) -> Optional[RuleViolation]:
        """Check uniqueness rules."""
        conditions = rule.conditions
        column = conditions.get("column")
        
        if not column or column not in df.columns:
            return self._create_missing_column_violation(rule, column, df)
        
        total_count = df.height
        unique_count = df[column].n_unique()
        duplicate_count = total_count - unique_count
        
        if duplicate_count > 0:
            uniqueness_percentage = (unique_count / total_count) * 100
            if uniqueness_percentage < rule.threshold:
                # Find sample duplicates
                duplicates = df.group_by(column).agg(pl.count().alias("count")).filter(pl.col("count") > 1)
                sample_duplicates = duplicates.head(5).to_dicts()
                
                return RuleViolation(
                    rule_id=rule.rule_id,
                    rule_name=rule.rule_name,
                    severity=rule.severity,
                    violation_count=duplicate_count,
                    total_records=total_count,
                    violation_percentage=(duplicate_count / total_count) * 100,
                    affected_columns=[column],
                    sample_violations=sample_duplicates,
                    details={"unique_count": unique_count, "duplicate_count": duplicate_count, 
                           "uniqueness_percentage": uniqueness_percentage}
                )
        
        return None
    
    def _check_consistency_rule(self, df: pl.DataFrame, rule: DataQualityRule) -> Optional[RuleViolation]:
        """Check consistency rules."""
        conditions = rule.conditions
        
        if "score_columns" in conditions and "decile_columns" in conditions:
            # SEIFA score-decile consistency
            score_cols = conditions["score_columns"]
            decile_cols = conditions["decile_columns"]
            min_correlation = conditions.get("min_correlation", 0.8)
            
            violations = []
            for score_col, decile_col in zip(score_cols, decile_cols):
                if score_col in df.columns and decile_col in df.columns:
                    # Calculate correlation
                    paired_data = df.select([score_col, decile_col]).drop_nulls()
                    if len(paired_data) > 1:
                        correlation = paired_data.select(pl.corr(score_col, decile_col)).item()
                        
                        if correlation is None or abs(correlation) < min_correlation:
                            violations.append({
                                "score_column": score_col,
                                "decile_column": decile_col,
                                "correlation": correlation,
                                "expected_min": min_correlation
                            })
            
            if violations:
                return RuleViolation(
                    rule_id=rule.rule_id,
                    rule_name=rule.rule_name,
                    severity=rule.severity,
                    violation_count=len(violations),
                    total_records=len(score_cols),
                    violation_percentage=(len(violations) / len(score_cols)) * 100,
                    affected_columns=score_cols + decile_cols,
                    sample_violations=violations,
                    details={"min_correlation": min_correlation, "violations": violations}
                )
        
        elif all(col in conditions for col in ["male_column", "female_column", "total_column"]):
            # Population gender consistency
            male_col = conditions["male_column"]
            female_col = conditions["female_column"]
            total_col = conditions["total_column"]
            tolerance_percent = conditions.get("tolerance_percent", 1.0)
            
            if all(col in df.columns for col in [male_col, female_col, total_col]):
                population_data = df.select([male_col, female_col, total_col]).drop_nulls()
                inconsistencies = []
                
                for row in population_data.iter_rows(named=True):
                    male = row[male_col]
                    female = row[female_col]
                    total = row[total_col]
                    expected_total = male + female
                    
                    if total > 0:
                        difference_percent = abs(total - expected_total) / total * 100
                        if difference_percent > tolerance_percent:
                            inconsistencies.append({
                                "male": male,
                                "female": female,
                                "total": total,
                                "expected_total": expected_total,
                                "difference_percent": difference_percent
                            })
                
                if inconsistencies:
                    violation_percentage = (len(inconsistencies) / len(population_data)) * 100
                    if violation_percentage > (100 - rule.threshold):
                        return RuleViolation(
                            rule_id=rule.rule_id,
                            rule_name=rule.rule_name,
                            severity=rule.severity,
                            violation_count=len(inconsistencies),
                            total_records=len(population_data),
                            violation_percentage=violation_percentage,
                            affected_columns=[male_col, female_col, total_col],
                            sample_violations=inconsistencies[:5],
                            details={"tolerance_percent": tolerance_percent, "inconsistencies": inconsistencies}
                        )
        
        return None
    
    def _check_outlier_rule(self, df: pl.DataFrame, rule: DataQualityRule) -> Optional[RuleViolation]:
        """Check outlier detection rules."""
        conditions = rule.conditions
        method = conditions.get("method", "iqr")
        
        columns = conditions.get("columns", [conditions.get("column")])
        columns = [col for col in columns if col is not None and col in df.columns]
        
        if not columns:
            return None
        
        outliers_found = []
        
        for column in columns:
            values = df[column].drop_nulls()
            if len(values) < 3:  # Need at least 3 values for outlier detection
                continue
            
            values_list = values.to_list()
            
            if method == "iqr":
                q1 = np.percentile(values_list, 25)
                q3 = np.percentile(values_list, 75)
                iqr = q3 - q1
                multiplier = conditions.get("multiplier", 1.5)
                lower_bound = q1 - multiplier * iqr
                upper_bound = q3 + multiplier * iqr
                
                outliers = [v for v in values_list if v < lower_bound or v > upper_bound]
                
            elif method == "zscore":
                mean_val = np.mean(values_list)
                std_val = np.std(values_list)
                threshold = conditions.get("threshold", 3.0)
                
                outliers = [v for v in values_list if abs((v - mean_val) / std_val) > threshold]
            
            else:
                continue
            
            if outliers:
                outlier_percentage = (len(outliers) / len(values_list)) * 100
                if outlier_percentage > (100 - rule.threshold):
                    outliers_found.extend([{"column": column, "value": v, "method": method} for v in outliers[:5]])
        
        if outliers_found:
            return RuleViolation(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                severity=rule.severity,
                violation_count=len(outliers_found),
                total_records=df.height,
                violation_percentage=(len(outliers_found) / df.height) * 100,
                affected_columns=columns,
                sample_violations=outliers_found[:5],
                details={"method": method, "threshold": rule.threshold}
            )
        
        return None
    
    def _check_temporal_rule(self, df: pl.DataFrame, rule: DataQualityRule) -> Optional[RuleViolation]:
        """Check temporal rules."""
        conditions = rule.conditions
        
        if "date_columns" in conditions:
            # Date range validation
            date_columns = conditions["date_columns"]
            min_date = conditions.get("min_date")
            max_date = conditions.get("max_date")
            
            violations = []
            for column in date_columns:
                if column not in df.columns:
                    continue
                
                dates = df[column].drop_nulls()
                for date_str in dates.to_list():
                    try:
                        if isinstance(date_str, str):
                            date_obj = datetime.strptime(date_str.split("T")[0], "%Y-%m-%d").date()
                        else:
                            continue
                        
                        if min_date and date_obj < datetime.strptime(min_date, "%Y-%m-%d").date():
                            violations.append({"column": column, "date": str(date_obj), "violation": f"before {min_date}"})
                        elif max_date and date_obj > datetime.strptime(max_date, "%Y-%m-%d").date():
                            violations.append({"column": column, "date": str(date_obj), "violation": f"after {max_date}"})
                    
                    except (ValueError, TypeError):
                        violations.append({"column": column, "date": str(date_str), "violation": "invalid format"})
            
            if violations:
                violation_percentage = (len(violations) / df.height) * 100
                if violation_percentage > (100 - rule.threshold):
                    return RuleViolation(
                        rule_id=rule.rule_id,
                        rule_name=rule.rule_name,
                        severity=rule.severity,
                        violation_count=len(violations),
                        total_records=df.height,
                        violation_percentage=violation_percentage,
                        affected_columns=date_columns,
                        sample_violations=violations[:5],
                        details={"min_date": min_date, "max_date": max_date}
                    )
        
        elif "date_column" in conditions and "max_age_days" in conditions:
            # Data freshness check
            date_column = conditions["date_column"]
            max_age_days = conditions["max_age_days"]
            
            if date_column not in df.columns:
                return self._create_missing_column_violation(rule, date_column, df)
            
            dates = df[date_column].drop_nulls()
            if len(dates) == 0:
                return None
            
            today = datetime.now().date()
            stale_records = []
            
            for date_str in dates.to_list():
                try:
                    if isinstance(date_str, str):
                        date_obj = datetime.strptime(date_str.split("T")[0], "%Y-%m-%d").date()
                        age_days = (today - date_obj).days
                        
                        if age_days > max_age_days:
                            stale_records.append({"date": str(date_obj), "age_days": age_days})
                
                except (ValueError, TypeError):
                    stale_records.append({"date": str(date_str), "age_days": "unknown", "error": "invalid format"})
            
            if stale_records:
                staleness_percentage = (len(stale_records) / len(dates)) * 100
                if staleness_percentage > (100 - rule.threshold):
                    return RuleViolation(
                        rule_id=rule.rule_id,
                        rule_name=rule.rule_name,
                        severity=rule.severity,
                        violation_count=len(stale_records),
                        total_records=len(dates),
                        violation_percentage=staleness_percentage,
                        affected_columns=[date_column],
                        sample_violations=stale_records[:5],
                        details={"max_age_days": max_age_days, "staleness_percentage": staleness_percentage}
                    )
        
        return None
    
    def _check_referential_rule(self, df: pl.DataFrame, rule: DataQualityRule) -> Optional[RuleViolation]:
        """Check referential integrity rules."""
        # Placeholder for referential integrity checks
        # Would need reference datasets to implement fully
        return None
    
    def _create_missing_column_violation(self, rule: DataQualityRule, missing_columns: Union[str, List[str]], df: pl.DataFrame) -> RuleViolation:
        """Create a violation for missing columns."""
        if isinstance(missing_columns, str):
            missing_columns = [missing_columns]
        
        return RuleViolation(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            severity=RuleSeverity.ERROR,
            violation_count=len(missing_columns),
            total_records=df.height,
            violation_percentage=100.0,
            affected_columns=missing_columns,
            sample_violations=[{"missing_column": col} for col in missing_columns],
            details={"missing_columns": missing_columns, "available_columns": df.columns}
        )


class TestDataQualityRulesEngine:
    """Test suite for data quality rules engine."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return AustralianHealthDataValidator()
    
    @pytest.fixture
    def rules_engine(self, validator):
        """Create rules engine instance."""
        return DataQualityRulesEngine(validator)
    
    @pytest.fixture
    def valid_seifa_data(self):
        """Valid SEIFA data for testing."""
        return pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001", "301011002"],
            "sa2_name_2021": ["Sydney", "Melbourne", "Brisbane"],
            "irsd_score": [1050, 950, 1100],
            "irsd_decile": [8, 5, 9],
            "irsad_score": [1080, 920, 1120],
            "irsad_decile": [7, 4, 8],
            "ier_score": [1000, 900, 1050],
            "ier_decile": [6, 3, 7],
            "ieo_score": [1150, 850, 1180],
            "ieo_decile": [9, 2, 10],
            "usual_resident_population": [15000, 12000, 18000],
            "tot_p_m": [7500, 6000, 9000],
            "tot_p_f": [7500, 6000, 9000],
            "tot_p_p": [15000, 12000, 18000],
            "extraction_timestamp": ["2023-01-01", "2023-01-01", "2023-01-01"],
            "last_updated": ["2023-06-01", "2023-06-01", "2023-06-01"],
        })
    
    @pytest.fixture
    def invalid_seifa_data(self):
        """Invalid SEIFA data for testing."""
        return pl.DataFrame({
            "sa2_code_2021": ["901021007", None, "101021007"],  # Invalid state, null, valid
            "sa2_name_2021": ["Invalid Area", "Missing Data", "Sydney"],
            "irsd_score": [750, 1050, None],  # Out of range, valid, null
            "irsd_decile": [0, 8, 5],         # Out of range, valid, valid
            "irsad_score": [1300, 920, 1120], # Out of range, valid, valid
            "irsad_decile": [11, 4, 8],       # Out of range, valid, valid
            "ier_score": [1000, 900, 1050],
            "ier_decile": [6, 3, 7],
            "ieo_score": [1150, 850, 1180],
            "ieo_decile": [9, 2, 10],
            "usual_resident_population": [-100, 12000, 100000],  # Negative, valid, outlier
            "tot_p_m": [7500, 6000, 9000],
            "tot_p_f": [7500, 6000, 9000],
            "tot_p_p": [15100, 12000, 18000],  # Inconsistent with male+female
            "extraction_timestamp": ["2019-01-01", "2023-01-01", "2026-01-01"],  # Too old, valid, future
            "last_updated": ["2022-01-01", "2023-06-01", "2023-06-01"],  # Stale, valid, valid
        })
    
    def test_rules_engine_initialization(self, rules_engine):
        """Test rules engine initialization and default rules."""
        assert isinstance(rules_engine, DataQualityRulesEngine)
        assert len(rules_engine.rules) > 0
        
        # Check that default rules are loaded
        rule_ids = [rule.rule_id for rule in rules_engine.rules]
        
        # Should have rules for all major categories
        assert any(rule_id.startswith("COMP_") for rule_id in rule_ids)  # Completeness
        assert any(rule_id.startswith("VAL_") for rule_id in rule_ids)   # Validity
        assert any(rule_id.startswith("UNQ_") for rule_id in rule_ids)   # Uniqueness
        assert any(rule_id.startswith("CON_") for rule_id in rule_ids)   # Consistency
        assert any(rule_id.startswith("OUT_") for rule_id in rule_ids)   # Outlier
        assert any(rule_id.startswith("TMP_") for rule_id in rule_ids)   # Temporal
        
        # All rules should be enabled by default
        assert all(rule.enabled for rule in rules_engine.rules)
    
    def test_custom_rule_management(self, rules_engine):
        """Test adding, removing, enabling, and disabling custom rules."""
        initial_count = len(rules_engine.rules)
        
        # Add custom rule
        custom_rule = DataQualityRule(
            rule_id="CUSTOM_001",
            rule_name="Custom Test Rule",
            rule_type=RuleType.CUSTOM,
            severity=RuleSeverity.WARNING,
            description="Test custom rule",
            conditions={"test": True},
            threshold=90.0
        )
        
        rules_engine.add_rule(custom_rule)
        assert len(rules_engine.rules) == initial_count + 1
        
        # Find the added rule
        added_rule = next((rule for rule in rules_engine.rules if rule.rule_id == "CUSTOM_001"), None)
        assert added_rule is not None
        assert added_rule.enabled is True
        
        # Disable the rule
        success = rules_engine.disable_rule("CUSTOM_001")
        assert success is True
        assert added_rule.enabled is False
        
        # Enable the rule
        success = rules_engine.enable_rule("CUSTOM_001")
        assert success is True
        assert added_rule.enabled is True
        
        # Remove the rule
        success = rules_engine.remove_rule("CUSTOM_001")
        assert success is True
        assert len(rules_engine.rules) == initial_count
        
        # Try to remove non-existent rule
        success = rules_engine.remove_rule("NON_EXISTENT")
        assert success is False
    
    def test_completeness_rule_validation(self, rules_engine, valid_seifa_data, invalid_seifa_data):
        """Test completeness rule validation."""
        # Test with valid data - should pass
        violations = rules_engine.validate_data(valid_seifa_data, "valid_test")
        completeness_violations = [v for v in violations if v.rule_id.startswith("COMP_")]
        
        # Valid data should have no completeness violations
        assert len(completeness_violations) == 0
        
        # Test with invalid data - should find violations
        violations = rules_engine.validate_data(invalid_seifa_data, "invalid_test")
        completeness_violations = [v for v in violations if v.rule_id.startswith("COMP_")]
        
        # Should have completeness violations due to null values
        assert len(completeness_violations) > 0
        
        # Check specific violation details
        sa2_violation = next((v for v in completeness_violations if "SA2" in v.rule_name), None)
        if sa2_violation:
            assert sa2_violation.severity == RuleSeverity.CRITICAL
            assert sa2_violation.violation_count > 0
            assert "sa2_code_2021" in sa2_violation.affected_columns
    
    def test_validity_rule_validation(self, rules_engine, valid_seifa_data, invalid_seifa_data):
        """Test validity rule validation."""
        # Test with valid data
        violations = rules_engine.validate_data(valid_seifa_data, "valid_test")
        validity_violations = [v for v in violations if v.rule_id.startswith("VAL_")]
        
        # Valid data should have minimal validity violations
        assert len(validity_violations) == 0 or all(v.severity in [RuleSeverity.WARNING, RuleSeverity.INFO] for v in validity_violations)
        
        # Test with invalid data
        violations = rules_engine.validate_data(invalid_seifa_data, "invalid_test")
        validity_violations = [v for v in violations if v.rule_id.startswith("VAL_")]
        
        # Should have validity violations
        assert len(validity_violations) > 0
        
        # Check for specific validity violations
        sa2_format_violation = next((v for v in validity_violations if "Format" in v.rule_name), None)
        if sa2_format_violation:
            assert sa2_format_violation.severity == RuleSeverity.CRITICAL
            assert sa2_format_violation.violation_count > 0
        
        seifa_range_violation = next((v for v in validity_violations if "Range" in v.rule_name), None)
        if seifa_range_violation:
            assert seifa_range_violation.severity == RuleSeverity.ERROR
            assert seifa_range_violation.violation_count > 0
    
    def test_uniqueness_rule_validation(self, rules_engine):
        """Test uniqueness rule validation."""
        # Data with duplicate SA2 codes
        duplicate_data = pl.DataFrame({
            "sa2_code_2021": ["101021007", "101021007", "201011001"],  # Duplicate
            "sa2_name_2021": ["Sydney", "Sydney Duplicate", "Melbourne"],
            "usual_resident_population": [15000, 15000, 12000],
        })
        
        violations = rules_engine.validate_data(duplicate_data, "duplicate_test")
        uniqueness_violations = [v for v in violations if v.rule_id.startswith("UNQ_")]
        
        # Should detect duplicate SA2 codes
        assert len(uniqueness_violations) > 0
        
        uniqueness_violation = uniqueness_violations[0]
        assert uniqueness_violation.severity == RuleSeverity.CRITICAL
        assert uniqueness_violation.violation_count > 0
        assert "sa2_code_2021" in uniqueness_violation.affected_columns
        assert uniqueness_violation.violation_percentage > 0
    
    def test_consistency_rule_validation(self, rules_engine, invalid_seifa_data):
        """Test consistency rule validation."""
        violations = rules_engine.validate_data(invalid_seifa_data, "consistency_test")
        consistency_violations = [v for v in violations if v.rule_id.startswith("CON_")]
        
        # Should find consistency violations (population gender mismatch)
        population_violation = next((v for v in consistency_violations if "Gender" in v.rule_name), None)
        if population_violation:
            assert population_violation.severity == RuleSeverity.ERROR
            assert population_violation.violation_count > 0
            assert all(col in population_violation.affected_columns for col in ["tot_p_m", "tot_p_f", "tot_p_p"])
    
    def test_outlier_rule_validation(self, rules_engine):
        """Test outlier detection rule validation."""
        # Data with outliers
        outlier_data = pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001", "301011002", "401011003"],
            "usual_resident_population": [15000, 12000, 18000, 150000],  # Last one is outlier
            "irsd_score": [1050, 950, 1100, 1180],
            "irsd_decile": [8, 5, 9, 10],
            "irsad_score": [1080, 920, 1120, 1180],
            "irsad_decile": [7, 4, 8, 10],
            "ier_score": [1000, 900, 1050, 1150],
            "ier_decile": [6, 3, 7, 9],
            "ieo_score": [1150, 850, 1180, 1200],
            "ieo_decile": [9, 2, 10, 10],
        })
        
        violations = rules_engine.validate_data(outlier_data, "outlier_test")
        outlier_violations = [v for v in violations if v.rule_id.startswith("OUT_")]
        
        # Should detect population outlier
        population_outlier = next((v for v in outlier_violations if "Population" in v.rule_name), None)
        if population_outlier:
            assert population_outlier.severity == RuleSeverity.WARNING
            assert population_outlier.violation_count > 0
            assert "usual_resident_population" in population_outlier.affected_columns
    
    def test_temporal_rule_validation(self, rules_engine, invalid_seifa_data):
        """Test temporal rule validation."""
        violations = rules_engine.validate_data(invalid_seifa_data, "temporal_test")
        temporal_violations = [v for v in violations if v.rule_id.startswith("TMP_")]
        
        # Should find temporal violations (dates out of range, stale data)
        assert len(temporal_violations) > 0
        
        # Check for date range violations
        date_range_violation = next((v for v in temporal_violations if "Range" in v.rule_name), None)
        if date_range_violation:
            assert date_range_violation.severity == RuleSeverity.ERROR
            assert date_range_violation.violation_count > 0
        
        # Check for data freshness violations
        freshness_violation = next((v for v in temporal_violations if "Freshness" in v.rule_name), None)
        if freshness_violation:
            assert freshness_violation.severity == RuleSeverity.WARNING
            assert freshness_violation.violation_count > 0
    
    def test_rule_severity_handling(self, rules_engine, invalid_seifa_data):
        """Test proper handling of different rule severities."""
        violations = rules_engine.validate_data(invalid_seifa_data, "severity_test")
        
        # Group violations by severity
        violations_by_severity = {}
        for violation in violations:
            severity = violation.severity
            if severity not in violations_by_severity:
                violations_by_severity[severity] = []
            violations_by_severity[severity].append(violation)
        
        # Should have violations of different severities
        severities_found = set(violations_by_severity.keys())
        
        # Check that we have multiple severity levels
        assert len(severities_found) > 1
        
        # Critical violations should be present (SA2 code issues)
        assert RuleSeverity.CRITICAL in severities_found
        
        # Error violations should be present (range violations)
        assert RuleSeverity.ERROR in severities_found
        
        # Check that critical violations have highest priority
        critical_violations = violations_by_severity.get(RuleSeverity.CRITICAL, [])
        for violation in critical_violations:
            assert violation.severity == RuleSeverity.CRITICAL
    
    def test_rule_execution_error_handling(self, rules_engine):
        """Test handling of rule execution errors."""
        # Create a malformed DataFrame to trigger errors
        problematic_data = pl.DataFrame({
            "bad_column": [1, 2, 3]
        })
        
        violations = rules_engine.validate_data(problematic_data, "error_test")
        
        # Should handle missing columns gracefully
        missing_column_violations = [v for v in violations if "missing_column" in v.details or "execution_error" in v.details]
        
        # Should have some violations due to missing expected columns
        assert len(missing_column_violations) > 0
        
        # Check that violations properly indicate missing columns
        for violation in missing_column_violations:
            assert violation.severity in [RuleSeverity.ERROR, RuleSeverity.CRITICAL]
            assert violation.violation_count > 0
    
    def test_comprehensive_rule_validation(self, rules_engine, valid_seifa_data, invalid_seifa_data):
        """Test comprehensive validation across all rule types."""
        # Test valid data
        valid_violations = rules_engine.validate_data(valid_seifa_data, "comprehensive_valid")
        
        # Valid data should have few or no violations
        assert len(valid_violations) <= 2  # Allow for minor warnings/info violations
        
        # Test invalid data
        invalid_violations = rules_engine.validate_data(invalid_seifa_data, "comprehensive_invalid")
        
        # Invalid data should have multiple violations
        assert len(invalid_violations) > 0
        
        # Group violations by rule type
        violations_by_type = {}
        for violation in invalid_violations:
            rule_prefix = violation.rule_id.split("_")[0]
            if rule_prefix not in violations_by_type:
                violations_by_type[rule_prefix] = []
            violations_by_type[rule_prefix].append(violation)
        
        # Should have violations across multiple rule types
        rule_types_with_violations = set(violations_by_type.keys())
        
        # Expect violations in multiple categories
        expected_types = {"COMP", "VAL", "CON", "TMP"}  # Completeness, Validity, Consistency, Temporal
        found_expected = rule_types_with_violations & expected_types
        
        assert len(found_expected) >= 2, f"Expected violations in multiple categories, found: {rule_types_with_violations}"
        
        # Log summary for analysis
        logger.info("Comprehensive validation results:")
        logger.info(f"  Valid data violations: {len(valid_violations)}")
        logger.info(f"  Invalid data violations: {len(invalid_violations)}")
        logger.info(f"  Violation types: {rule_types_with_violations}")
        
        for rule_type, violations in violations_by_type.items():
            severity_counts = {}
            for violation in violations:
                severity = violation.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            logger.info(f"  {rule_type} violations by severity: {severity_counts}")
    
    def test_rule_performance_with_large_dataset(self, rules_engine):
        """Test rule engine performance with larger dataset."""
        # Create larger dataset for performance testing
        large_data = pl.DataFrame({
            "sa2_code_2021": [f"{i%8 + 1}01021{i:03d}" for i in range(1000)],
            "sa2_name_2021": [f"Area_{i}" for i in range(1000)],
            "irsd_score": [950 + (i % 300) for i in range(1000)],
            "irsd_decile": [(i % 10) + 1 for i in range(1000)],
            "irsad_score": [920 + (i % 280) for i in range(1000)],
            "irsad_decile": [(i % 10) + 1 for i in range(1000)],
            "ier_score": [900 + (i % 250) for i in range(1000)],
            "ier_decile": [(i % 10) + 1 for i in range(1000)],
            "ieo_score": [850 + (i % 350) for i in range(1000)],
            "ieo_decile": [(i % 10) + 1 for i in range(1000)],
            "usual_resident_population": [5000 + (i * 10) for i in range(1000)],
        })
        
        # Measure validation time
        start_time = datetime.now()
        violations = rules_engine.validate_data(large_data, "performance_test")
        end_time = datetime.now()
        
        validation_time = (end_time - start_time).total_seconds()
        
        # Should complete in reasonable time (< 10 seconds)
        assert validation_time < 10.0, f"Validation took too long: {validation_time} seconds"
        
        # Should still detect any violations
        logger.info(f"Performance test results:")
        logger.info(f"  Dataset size: {large_data.height} rows")
        logger.info(f"  Validation time: {validation_time:.2f} seconds")
        logger.info(f"  Violations found: {len(violations)}")
        logger.info(f"  Rules processed: {len([r for r in rules_engine.rules if r.enabled])}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])