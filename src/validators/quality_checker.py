"""
Data Quality Assessment Framework

This module provides a comprehensive data quality assessment framework with configurable 
quality rules engine, quality score calculation, anomaly detection algorithms, and 
statistical validation methods for Australian Health Geography Datasets.
"""

import logging
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import hashlib
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from ..utils.interfaces import (
    DataBatch, 
    DataRecord, 
    ValidationResult, 
    ValidationSeverity,
    DataQualityError
)
from .base import BaseValidator


@dataclass
class QualityScore:
    """Data quality score breakdown."""
    overall_score: float
    completeness_score: float
    validity_score: float
    consistency_score: float
    accuracy_score: float
    timeliness_score: float
    uniqueness_score: float
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    quality_grade: str = "UNKNOWN"
    total_records: int = 0
    issues_found: int = 0


@dataclass
class QualityRule:
    """Individual quality rule definition."""
    rule_id: str
    rule_type: str
    description: str
    severity: ValidationSeverity
    weight: float = 1.0
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection analysis."""
    anomalies_detected: int
    anomaly_records: List[int]
    method: str
    threshold: float
    details: Dict[str, Any] = field(default_factory=dict)


class QualityChecker(BaseValidator):
    """
    Comprehensive data quality assessment framework.
    
    This class provides a configurable quality rules engine for assessing
    data quality across multiple dimensions including completeness, validity,
    consistency, accuracy, timeliness, and uniqueness.
    """
    
    def __init__(
        self,
        validator_id: str = "quality_checker",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialise the quality checker.
        
        Args:
            validator_id: Unique identifier for this validator
            config: Configuration dictionary containing quality rules
            logger: Optional logger instance
        """
        super().__init__(validator_id, config or {}, logger)
        
        # Quality assessment configuration
        self.quality_config = self.config.get('quality_rules', {})
        self.quality_weights = self.quality_config.get('quality_weights', {
            'completeness': 0.25,
            'validity': 0.25,
            'consistency': 0.20,
            'accuracy': 0.15,
            'timeliness': 0.10,
            'uniqueness': 0.05
        })
        
        # Quality thresholds
        self.quality_thresholds = self.quality_config.get('quality_thresholds', {
            'excellent': 0.95,
            'good': 0.85,
            'acceptable': 0.70,
            'poor': 0.50
        })
        
        # Performance settings
        self.performance_config = self.quality_config.get('performance', {})
        self.enable_caching = self.performance_config.get('enable_caching', True)
        self.parallel_validation = self.performance_config.get('parallel_validation', True)
        self.max_workers = self.performance_config.get('max_worker_threads', 4)
        self.batch_size = self.performance_config.get('batch_size', 1000)
        
        # Cache for expensive calculations
        self._quality_cache: Dict[str, Any] = {}
        self._anomaly_cache: Dict[str, AnomalyDetectionResult] = {}
        
        # Load quality rules
        self._load_quality_rules()
        
        # Statistics for reporting
        self._quality_stats = defaultdict(int)
        
    def validate(self, data: DataBatch) -> List[ValidationResult]:
        """
        Perform comprehensive quality validation.
        
        Args:
            data: Batch of data records to validate
            
        Returns:
            List[ValidationResult]: Quality validation results
        """
        if not data:
            return [ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="quality_empty_data",
                message="Cannot assess quality of empty dataset"
            )]
        
        results = []
        start_time = datetime.now()
        
        try:
            # Calculate overall quality score
            quality_score = self.calculate_quality_score(data)
            results.extend(self._generate_quality_score_results(quality_score))
            
            # Perform anomaly detection
            anomaly_results = self.detect_anomalies(data)
            results.extend(self._generate_anomaly_results(anomaly_results))
            
            # Validate against configurable rules
            rule_results = self._validate_quality_rules(data)
            results.extend(rule_results)
            
            # Statistical validation
            statistical_results = self._perform_statistical_validation(data)
            results.extend(statistical_results)
            
            # Update statistics
            self._update_quality_statistics(results, len(data))
            
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"Quality assessment completed in {duration:.2f}s: "
                f"{len(results)} issues found across {len(data)} records"
            )
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="quality_assessment_error",
                message=f"Quality assessment failed: {str(e)}"
            ))
        
        return results
    
    def calculate_quality_score(self, data: DataBatch) -> QualityScore:
        """
        Calculate comprehensive quality score.
        
        Args:
            data: Batch of data records
            
        Returns:
            QualityScore: Detailed quality score breakdown
        """
        if not data:
            return QualityScore(
                overall_score=0.0,
                completeness_score=0.0,
                validity_score=0.0,
                consistency_score=0.0,
                accuracy_score=0.0,
                timeliness_score=0.0,
                uniqueness_score=0.0,
                quality_grade="CRITICAL"
            )
        
        # Generate cache key
        cache_key = self._generate_cache_key(data, "quality_score")
        if self.enable_caching and cache_key in self._quality_cache:
            return self._quality_cache[cache_key]
        
        # Calculate individual dimension scores
        completeness_score = self._calculate_completeness_score(data)
        validity_score = self._calculate_validity_score(data)
        consistency_score = self._calculate_consistency_score(data)
        accuracy_score = self._calculate_accuracy_score(data)
        timeliness_score = self._calculate_timeliness_score(data)
        uniqueness_score = self._calculate_uniqueness_score(data)
        
        # Calculate weighted overall score
        overall_score = (
            completeness_score * self.quality_weights.get('completeness', 0.25) +
            validity_score * self.quality_weights.get('validity', 0.25) +
            consistency_score * self.quality_weights.get('consistency', 0.20) +
            accuracy_score * self.quality_weights.get('accuracy', 0.15) +
            timeliness_score * self.quality_weights.get('timeliness', 0.10) +
            uniqueness_score * self.quality_weights.get('uniqueness', 0.05)
        )
        
        # Determine quality grade
        quality_grade = self._determine_quality_grade(overall_score)
        
        quality_score = QualityScore(
            overall_score=overall_score,
            completeness_score=completeness_score,
            validity_score=validity_score,
            consistency_score=consistency_score,
            accuracy_score=accuracy_score,
            timeliness_score=timeliness_score,
            uniqueness_score=uniqueness_score,
            quality_grade=quality_grade,
            total_records=len(data)
        )
        
        # Cache the result
        if self.enable_caching:
            self._quality_cache[cache_key] = quality_score
        
        return quality_score
    
    def detect_anomalies(self, data: DataBatch) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies in the dataset using multiple methods.
        
        Args:
            data: Batch of data records
            
        Returns:
            List[AnomalyDetectionResult]: Anomaly detection results
        """
        results = []
        
        # Get anomaly detection configuration
        anomaly_config = self.quality_config.get('anomaly_detection', {})
        
        # IQR-based outlier detection
        if 'iqr_outliers' in anomaly_config.get('statistical_methods', []):
            iqr_config = next(
                (method for method in anomaly_config['statistical_methods'] 
                 if 'iqr_outliers' in method), {}
            ).get('iqr_outliers', {})
            
            iqr_results = self._detect_iqr_outliers(data, iqr_config)
            results.extend(iqr_results)
        
        # Z-score based outlier detection
        if 'z_score_outliers' in anomaly_config.get('statistical_methods', []):
            zscore_config = next(
                (method for method in anomaly_config['statistical_methods'] 
                 if 'z_score_outliers' in method), {}
            ).get('z_score_outliers', {})
            
            zscore_results = self._detect_zscore_outliers(data, zscore_config)
            results.extend(zscore_results)
        
        # Isolation Forest (if we have scikit-learn available)
        if 'isolation_forest' in anomaly_config.get('statistical_methods', []):
            isolation_config = next(
                (method for method in anomaly_config['statistical_methods'] 
                 if 'isolation_forest' in method), {}
            ).get('isolation_forest', {})
            
            isolation_results = self._detect_isolation_forest_outliers(data, isolation_config)
            results.extend(isolation_results)
        
        return results
    
    def get_validation_rules(self) -> List[str]:
        """
        Get the list of validation rules supported by this validator.
        
        Returns:
            List[str]: List of validation rule identifiers
        """
        return [
            "quality_score_calculation",
            "completeness_assessment",
            "validity_assessment", 
            "consistency_assessment",
            "accuracy_assessment",
            "timeliness_assessment",
            "uniqueness_assessment",
            "anomaly_detection",
            "statistical_validation"
        ]
    
    def get_quality_statistics(self) -> Dict[str, Any]:
        """
        Get quality assessment statistics.
        
        Returns:
            Dict[str, Any]: Quality statistics
        """
        return dict(self._quality_stats)
    
    def _load_quality_rules(self) -> None:
        """Load quality rules from configuration."""
        self.quality_rules = []
        
        # Load completeness rules
        completeness_rules = self.quality_config.get('completeness_rules', {})
        for priority, columns in completeness_rules.items():
            if isinstance(columns, list):
                for column in columns:
                    self.quality_rules.append(QualityRule(
                        rule_id=f"completeness_{priority}_{column}",
                        rule_type="completeness",
                        description=f"Completeness check for {priority} column {column}",
                        severity=ValidationSeverity.WARNING if priority != "critical_columns" else ValidationSeverity.ERROR,
                        parameters={"column": column, "priority": priority}
                    ))
        
        # Load validity rules
        validity_rules = self.quality_config.get('validity_rules', {})
        for rule_name, rule_config in validity_rules.items():
            self.quality_rules.append(QualityRule(
                rule_id=f"validity_{rule_name}",
                rule_type="validity",
                description=rule_config.get("description", f"Validity check for {rule_name}"),
                severity=ValidationSeverity(rule_config.get("severity", "warning")),
                parameters=rule_config
            ))
    
    def _calculate_completeness_score(self, data: DataBatch) -> float:
        """Calculate completeness score."""
        if not data:
            return 0.0
        
        completeness_rules = self.quality_config.get('completeness_rules', {})
        total_weighted_score = 0.0
        total_weight = 0.0
        
        # Weight mapping for different priority columns
        priority_weights = {
            'critical_columns': 1.0,
            'high_priority_columns': 0.8,
            'medium_priority_columns': 0.6,
            'low_priority_columns': 0.4
        }
        
        for priority, columns in completeness_rules.items():
            if isinstance(columns, list):
                weight = priority_weights.get(priority, 0.5)
                
                for column in columns:
                    non_null_count = sum(1 for record in data if record.get(column) is not None and record.get(column) != "")
                    completeness = non_null_count / len(data)
                    
                    total_weighted_score += completeness * weight
                    total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_validity_score(self, data: DataBatch) -> float:
        """Calculate validity score."""
        if not data:
            return 0.0
        
        validity_rules = self.quality_config.get('validity_rules', {})
        total_score = 0.0
        rule_count = 0
        
        for rule_name, rule_config in validity_rules.items():
            valid_count = 0
            total_count = 0
            
            # Get the column to validate (may be implicit from rule name)
            column = rule_config.get('column', rule_name.replace('_', ' '))
            
            for record in data:
                value = record.get(column)
                if value is not None:
                    total_count += 1
                    
                    # Check validity based on rule type
                    if self._is_value_valid(value, rule_config):
                        valid_count += 1
            
            if total_count > 0:
                rule_score = valid_count / total_count
                total_score += rule_score
                rule_count += 1
        
        return total_score / rule_count if rule_count > 0 else 1.0
    
    def _calculate_consistency_score(self, data: DataBatch) -> float:
        """Calculate consistency score."""
        if not data:
            return 0.0
        
        consistency_rules = self.quality_config.get('consistency_rules', {})
        total_score = 0.0
        rule_count = 0
        
        # Population relationships check
        pop_rules = consistency_rules.get('population_relationships', [])
        for rule in pop_rules:
            if rule.get('rule_id') == 'pop_age_consistency':
                score = self._check_population_age_consistency(data, rule)
                total_score += score
                rule_count += 1
        
        # Geographic relationships check
        geo_rules = consistency_rules.get('geographic_relationships', [])
        for rule in geo_rules:
            if rule.get('rule_id') == 'sa2_state_consistency':
                score = self._check_sa2_state_consistency(data, rule)
                total_score += score
                rule_count += 1
        
        # Temporal consistency check
        temporal_rules = consistency_rules.get('temporal_consistency', [])
        for rule in temporal_rules:
            if rule.get('rule_id') == 'year_progression':
                score = self._check_year_progression_consistency(data, rule)
                total_score += score
                rule_count += 1
        
        return total_score / rule_count if rule_count > 0 else 1.0
    
    def _calculate_accuracy_score(self, data: DataBatch) -> float:
        """Calculate accuracy score (placeholder - would require reference data)."""
        # This would typically compare against authoritative reference datasets
        # For now, return a high score as a placeholder
        return 0.9
    
    def _calculate_timeliness_score(self, data: DataBatch) -> float:
        """Calculate timeliness score."""
        if not data:
            return 0.0
        
        timeliness_rules = self.quality_config.get('timeliness_rules', {})
        max_age_months = timeliness_rules.get('maximum_age_months', 24)
        
        current_date = datetime.now()
        fresh_records = 0
        
        for record in data:
            # Try to extract date from various possible columns
            record_date = None
            for date_column in ['data_year', 'year', 'date_updated', 'collection_date']:
                if date_column in record and record[date_column]:
                    try:
                        if isinstance(record[date_column], str):
                            # Try to parse year
                            year = int(record[date_column])
                            record_date = datetime(year, 12, 31)  # Assume end of year
                        elif isinstance(record[date_column], datetime):
                            record_date = record[date_column]
                        break
                    except (ValueError, TypeError):
                        continue
            
            if record_date:
                months_old = (current_date - record_date).days / 30.44  # Average days per month
                if months_old <= max_age_months:
                    fresh_records += 1
        
        return fresh_records / len(data) if data else 0.0
    
    def _calculate_uniqueness_score(self, data: DataBatch) -> float:
        """Calculate uniqueness score."""
        if not data:
            return 0.0
        
        uniqueness_rules = self.quality_config.get('uniqueness_rules', {})
        primary_keys = uniqueness_rules.get('primary_keys', [])
        
        if not primary_keys:
            return 1.0  # No uniqueness constraints defined
        
        total_score = 0.0
        
        for key_config in primary_keys:
            columns = key_config.get('columns', [])
            if not columns:
                continue
            
            # Create composite keys
            keys_seen = set()
            duplicate_count = 0
            
            for record in data:
                key_values = []
                for column in columns:
                    key_values.append(str(record.get(column, '')))
                
                composite_key = '|'.join(key_values)
                
                if composite_key in keys_seen:
                    duplicate_count += 1
                else:
                    keys_seen.add(composite_key)
            
            # Calculate uniqueness score for this key
            uniqueness = 1.0 - (duplicate_count / len(data))
            total_score += uniqueness
        
        return total_score / len(primary_keys)
    
    def _determine_quality_grade(self, score: float) -> str:
        """Determine quality grade based on score."""
        if score >= self.quality_thresholds.get('excellent', 0.95):
            return "EXCELLENT"
        elif score >= self.quality_thresholds.get('good', 0.85):
            return "GOOD"
        elif score >= self.quality_thresholds.get('acceptable', 0.70):
            return "ACCEPTABLE"
        elif score >= self.quality_thresholds.get('poor', 0.50):
            return "POOR"
        else:
            return "CRITICAL"
    
    def _detect_iqr_outliers(
        self, 
        data: DataBatch, 
        config: Dict[str, Any]
    ) -> List[AnomalyDetectionResult]:
        """Detect outliers using IQR method."""
        results = []
        
        multiplier = config.get('multiplier', 1.5)
        columns = config.get('columns', [])
        
        for column in columns:
            values = []
            indices = []
            
            for idx, record in enumerate(data):
                value = record.get(column)
                if value is not None and isinstance(value, (int, float)):
                    values.append(value)
                    indices.append(idx)
            
            if len(values) < 4:  # Need at least 4 values for quartiles
                continue
            
            # Calculate IQR
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr
            
            # Find outliers
            outlier_indices = []
            for i, value in enumerate(values):
                if value < lower_bound or value > upper_bound:
                    outlier_indices.append(indices[i])
            
            if outlier_indices:
                results.append(AnomalyDetectionResult(
                    anomalies_detected=len(outlier_indices),
                    anomaly_records=outlier_indices,
                    method="iqr",
                    threshold=multiplier,
                    details={
                        'column': column,
                        'q1': q1,
                        'q3': q3,
                        'iqr': iqr,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
                ))
        
        return results
    
    def _detect_zscore_outliers(
        self, 
        data: DataBatch, 
        config: Dict[str, Any]
    ) -> List[AnomalyDetectionResult]:
        """Detect outliers using Z-score method."""
        results = []
        
        threshold = config.get('threshold', 3.0)
        columns = config.get('columns', [])
        
        for column in columns:
            values = []
            indices = []
            
            for idx, record in enumerate(data):
                value = record.get(column)
                if value is not None and isinstance(value, (int, float)):
                    values.append(value)
                    indices.append(idx)
            
            if len(values) < 3:  # Need at least 3 values
                continue
            
            # Calculate mean and standard deviation
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val == 0:  # All values are the same
                continue
            
            # Find outliers
            outlier_indices = []
            for i, value in enumerate(values):
                z_score = abs((value - mean_val) / std_val)
                if z_score > threshold:
                    outlier_indices.append(indices[i])
            
            if outlier_indices:
                results.append(AnomalyDetectionResult(
                    anomalies_detected=len(outlier_indices),
                    anomaly_records=outlier_indices,
                    method="z_score",
                    threshold=threshold,
                    details={
                        'column': column,
                        'mean': mean_val,
                        'std': std_val
                    }
                ))
        
        return results
    
    def _detect_isolation_forest_outliers(
        self, 
        data: DataBatch, 
        config: Dict[str, Any]
    ) -> List[AnomalyDetectionResult]:
        """Detect outliers using Isolation Forest (requires scikit-learn)."""
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            self.logger.warning("scikit-learn not available for Isolation Forest")
            return []
        
        results = []
        contamination = config.get('contamination', 0.05)
        columns = config.get('columns', [])
        
        if not columns:
            return results
        
        # Prepare data matrix
        data_matrix = []
        valid_indices = []
        
        for idx, record in enumerate(data):
            row = []
            valid_row = True
            
            for column in columns:
                value = record.get(column)
                if value is not None and isinstance(value, (int, float)):
                    row.append(value)
                else:
                    valid_row = False
                    break
            
            if valid_row:
                data_matrix.append(row)
                valid_indices.append(idx)
        
        if len(data_matrix) < 10:  # Need sufficient data points
            return results
        
        # Apply Isolation Forest
        isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        
        outlier_labels = isolation_forest.fit_predict(data_matrix)
        
        # Find outlier indices
        outlier_indices = [
            valid_indices[i] for i, label in enumerate(outlier_labels) 
            if label == -1
        ]
        
        if outlier_indices:
            results.append(AnomalyDetectionResult(
                anomalies_detected=len(outlier_indices),
                anomaly_records=outlier_indices,
                method="isolation_forest",
                threshold=contamination,
                details={
                    'columns': columns,
                    'contamination': contamination,
                    'total_samples': len(data_matrix)
                }
            ))
        
        return results
    
    def _validate_quality_rules(self, data: DataBatch) -> List[ValidationResult]:
        """Validate data against configurable quality rules."""
        results = []
        
        if self.parallel_validation and len(data) > self.batch_size:
            # Process in parallel for large datasets
            results = self._validate_quality_rules_parallel(data)
        else:
            # Sequential processing for smaller datasets
            for rule in self.quality_rules:
                if rule.enabled:
                    rule_results = self._apply_quality_rule(data, rule)
                    results.extend(rule_results)
        
        return results
    
    def _validate_quality_rules_parallel(self, data: DataBatch) -> List[ValidationResult]:
        """Validate quality rules using parallel processing."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks for each enabled rule
            future_to_rule = {
                executor.submit(self._apply_quality_rule, data, rule): rule
                for rule in self.quality_rules if rule.enabled
            }
            
            # Collect results
            for future in as_completed(future_to_rule):
                try:
                    rule_results = future.result()
                    results.extend(rule_results)
                except Exception as e:
                    rule = future_to_rule[future]
                    self.logger.error(f"Error applying rule {rule.rule_id}: {e}")
        
        return results
    
    def _apply_quality_rule(self, data: DataBatch, rule: QualityRule) -> List[ValidationResult]:
        """Apply a single quality rule to the data."""
        if rule.rule_type == "completeness":
            return self._apply_completeness_rule(data, rule)
        elif rule.rule_type == "validity":
            return self._apply_validity_rule(data, rule)
        else:
            return []
    
    def _apply_completeness_rule(self, data: DataBatch, rule: QualityRule) -> List[ValidationResult]:
        """Apply completeness rule."""
        results = []
        column = rule.parameters.get('column')
        priority = rule.parameters.get('priority', 'medium')
        
        if not column:
            return results
        
        # Define minimum completeness thresholds by priority
        min_completeness_thresholds = {
            'critical_columns': 1.0,
            'high_priority_columns': 0.95,
            'medium_priority_columns': 0.85,
            'low_priority_columns': 0.70
        }
        
        min_threshold = min_completeness_thresholds.get(priority, 0.80)
        
        # Calculate completeness
        non_null_count = sum(
            1 for record in data 
            if record.get(column) is not None and record.get(column) != ""
        )
        completeness = non_null_count / len(data) if data else 0
        
        if completeness < min_threshold:
            results.append(ValidationResult(
                is_valid=False,
                severity=rule.severity,
                rule_id=rule.rule_id,
                message=f"Column '{column}' completeness {completeness:.1%} below threshold {min_threshold:.1%}",
                details={
                    'column': column,
                    'completeness': completeness,
                    'threshold': min_threshold,
                    'priority': priority,
                    'non_null_count': non_null_count,
                    'total_count': len(data)
                }
            ))
        
        return results
    
    def _apply_validity_rule(self, data: DataBatch, rule: QualityRule) -> List[ValidationResult]:
        """Apply validity rule."""
        results = []
        invalid_records = []
        
        for record_idx, record in enumerate(data):
            # Find the column to validate
            column = rule.parameters.get('column')
            if not column:
                # Try to infer column from rule name
                column = rule.rule_id.replace('validity_', '').replace('_', ' ')
            
            value = record.get(column)
            if value is not None and not self._is_value_valid(value, rule.parameters):
                invalid_records.append(record_idx)
        
        if invalid_records:
            results.append(ValidationResult(
                is_valid=False,
                severity=rule.severity,
                rule_id=rule.rule_id,
                message=f"Validity rule '{rule.rule_id}' failed for {len(invalid_records)} records",
                details={
                    'rule_description': rule.description,
                    'invalid_count': len(invalid_records),
                    'total_count': len(data)
                },
                affected_records=invalid_records[:10]  # Limit to first 10 for performance
            ))
        
        return results
    
    def _is_value_valid(self, value: Any, rule_config: Dict[str, Any]) -> bool:
        """Check if a value is valid according to rule configuration."""
        # Pattern matching
        if 'pattern' in rule_config:
            import re
            pattern = rule_config['pattern']
            return bool(re.match(pattern, str(value)))
        
        # Allowed values
        if 'allowed_values' in rule_config:
            return str(value) in rule_config['allowed_values']
        
        # Range checking
        if 'min_value' in rule_config or 'max_value' in rule_config:
            try:
                num_value = float(value)
                if 'min_value' in rule_config and num_value < rule_config['min_value']:
                    return False
                if 'max_value' in rule_config and num_value > rule_config['max_value']:
                    return False
                return True
            except (ValueError, TypeError):
                return False
        
        return True
    
    def _perform_statistical_validation(self, data: DataBatch) -> List[ValidationResult]:
        """Perform statistical validation checks."""
        results = []
        
        # Check for statistical patterns that might indicate data quality issues
        numeric_columns = self._identify_numeric_columns(data)
        
        for column in numeric_columns:
            values = [
                record.get(column) for record in data 
                if record.get(column) is not None and isinstance(record.get(column), (int, float))
            ]
            
            if len(values) < 10:  # Need sufficient data
                continue
            
            # Check for suspicious patterns
            # 1. Too many identical values
            value_counts = {}
            for value in values:
                value_counts[value] = value_counts.get(value, 0) + 1
            
            most_common_count = max(value_counts.values())
            if most_common_count > len(values) * 0.8:  # More than 80% identical
                results.append(ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    rule_id="statistical_suspicious_uniformity",
                    message=f"Column '{column}' has {most_common_count} identical values ({most_common_count/len(values):.1%})",
                    details={'column': column, 'identical_ratio': most_common_count/len(values)}
                ))
            
            # 2. Extreme variance (coefficient of variation)
            if len(values) > 1:
                mean_val = np.mean(values)
                std_val = np.std(values)
                if mean_val != 0:
                    cv = std_val / abs(mean_val)
                    if cv > 2.0:  # High coefficient of variation
                        results.append(ValidationResult(
                            is_valid=True,
                            severity=ValidationSeverity.INFO,
                            rule_id="statistical_high_variance",
                            message=f"Column '{column}' has high variance (CV={cv:.2f})",
                            details={'column': column, 'coefficient_of_variation': cv}
                        ))
        
        return results
    
    def _identify_numeric_columns(self, data: DataBatch) -> List[str]:
        """Identify numeric columns in the dataset."""
        if not data:
            return []
        
        numeric_columns = []
        sample_record = data[0]
        
        for column, value in sample_record.items():
            if isinstance(value, (int, float)):
                numeric_columns.append(column)
        
        return numeric_columns
    
    def _generate_quality_score_results(self, quality_score: QualityScore) -> List[ValidationResult]:
        """Generate validation results from quality score."""
        results = []
        
        # Overall quality assessment
        if quality_score.overall_score < self.quality_thresholds.get('acceptable', 0.70):
            severity = ValidationSeverity.ERROR if quality_score.overall_score < 0.50 else ValidationSeverity.WARNING
            
            results.append(ValidationResult(
                is_valid=quality_score.overall_score >= self.quality_thresholds.get('acceptable', 0.70),
                severity=severity,
                rule_id="overall_quality_score",
                message=f"Overall data quality score: {quality_score.overall_score:.1%} ({quality_score.quality_grade})",
                details={
                    'overall_score': quality_score.overall_score,
                    'completeness_score': quality_score.completeness_score,
                    'validity_score': quality_score.validity_score,
                    'consistency_score': quality_score.consistency_score,
                    'accuracy_score': quality_score.accuracy_score,
                    'timeliness_score': quality_score.timeliness_score,
                    'uniqueness_score': quality_score.uniqueness_score,
                    'quality_grade': quality_score.quality_grade,
                    'total_records': quality_score.total_records
                }
            ))
        
        return results
    
    def _generate_anomaly_results(self, anomaly_results: List[AnomalyDetectionResult]) -> List[ValidationResult]:
        """Generate validation results from anomaly detection."""
        results = []
        
        for anomaly_result in anomaly_results:
            if anomaly_result.anomalies_detected > 0:
                results.append(ValidationResult(
                    is_valid=True,  # Anomalies are informational, not errors
                    severity=ValidationSeverity.INFO,
                    rule_id=f"anomaly_detection_{anomaly_result.method}",
                    message=f"Detected {anomaly_result.anomalies_detected} anomalies using {anomaly_result.method}",
                    details=anomaly_result.details,
                    affected_records=anomaly_result.anomaly_records[:10]  # Limit for performance
                ))
        
        return results
    
    def _check_population_age_consistency(self, data: DataBatch, rule: Dict[str, Any]) -> float:
        """Check consistency between age group populations and total population."""
        consistent_records = 0
        tolerance = rule.get('tolerance', 0.05)
        
        age_columns = [
            'age_0_4_years', 'age_5_14_years', 'age_15_24_years', 'age_25_34_years',
            'age_35_44_years', 'age_45_54_years', 'age_55_64_years', 'age_65_74_years',
            'age_75_84_years', 'age_85_years_over'
        ]
        
        for record in data:
            total_pop = record.get('total_population')
            if total_pop and isinstance(total_pop, (int, float)):
                age_sum = 0
                valid_age_data = True
                
                for age_col in age_columns:
                    age_value = record.get(age_col)
                    if age_value and isinstance(age_value, (int, float)):
                        age_sum += age_value
                    else:
                        valid_age_data = False
                        break
                
                if valid_age_data and total_pop > 0:
                    difference = abs(age_sum - total_pop) / total_pop
                    if difference <= tolerance:
                        consistent_records += 1
        
        return consistent_records / len(data) if data else 0.0
    
    def _check_sa2_state_consistency(self, data: DataBatch, rule: Dict[str, Any]) -> float:
        """Check consistency between SA2 codes and state codes."""
        consistent_records = 0
        
        state_mapping = {
            "1": "NSW", "2": "VIC", "3": "QLD", "4": "SA",
            "5": "WA", "6": "TAS", "7": "NT", "8": "ACT"
        }
        
        for record in data:
            sa2_code = record.get('sa2_code')
            state_code = record.get('state_code')
            
            if sa2_code and state_code:
                sa2_str = str(sa2_code)
                state_str = str(state_code)
                
                if len(sa2_str) >= 1 and sa2_str[0] == state_str:
                    consistent_records += 1
        
        return consistent_records / len(data) if data else 0.0
    
    def _check_year_progression_consistency(self, data: DataBatch, rule: Dict[str, Any]) -> float:
        """Check for reasonable year-over-year population changes."""
        # This would require time series data - placeholder implementation
        return 1.0
    
    def _update_quality_statistics(self, results: List[ValidationResult], record_count: int) -> None:
        """Update quality assessment statistics."""
        self._quality_stats['total_assessments'] += 1
        self._quality_stats['total_records_assessed'] += record_count
        
        for result in results:
            if result.severity == ValidationSeverity.ERROR:
                self._quality_stats['total_errors'] += 1
            elif result.severity == ValidationSeverity.WARNING:
                self._quality_stats['total_warnings'] += 1
            else:
                self._quality_stats['total_info'] += 1
    
    def _generate_cache_key(self, data: DataBatch, operation: str) -> str:
        """Generate cache key for data and operation."""
        # Create hash of data sample for cache key
        data_sample = json.dumps(data[:min(10, len(data))], sort_keys=True, default=str)
        data_hash = hashlib.md5(data_sample.encode()).hexdigest()
        return f"{operation}_{len(data)}_{data_hash}"