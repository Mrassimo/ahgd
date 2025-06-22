"""
Advanced Statistical Validation for Australian Health Data

This module provides comprehensive statistical validation specifically designed
for Australian health and geographic datasets, including range validation,
outlier detection, correlation analysis, and distribution validation with
domain-specific knowledge.
"""

import logging
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..utils.interfaces import (
    DataBatch,
    DataRecord,
    ValidationResult,
    ValidationSeverity,
    DataQualityError
)
from .base import BaseValidator


@dataclass
class RangeValidationConfig:
    """Configuration for range validation rules."""
    column: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    typical_min: Optional[float] = None
    typical_max: Optional[float] = None
    description: str = ""
    severity_if_outside_typical: ValidationSeverity = ValidationSeverity.WARNING
    severity_if_outside_absolute: ValidationSeverity = ValidationSeverity.ERROR


@dataclass
class StatisticalReport:
    """Comprehensive statistical validation report."""
    summary_statistics: Dict[str, Dict[str, float]]
    range_violations: List[Dict[str, Any]]
    outlier_summary: Dict[str, List[Dict[str, Any]]]
    correlation_findings: List[Dict[str, Any]]
    distribution_findings: List[Dict[str, Any]]
    severity_counts: Dict[ValidationSeverity, int]
    visualisation_suggestions: List[Dict[str, Any]]
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.now)


class AdvancedStatisticalValidator(BaseValidator):
    """
    Advanced statistical validation for Australian health data.
    
    This validator implements comprehensive statistical validation with
    domain-specific knowledge of Australian health indicators, SEIFA scores,
    demographic distributions, and geographic patterns.
    """
    
    # Domain-specific range configurations
    HEALTH_INDICATOR_RANGES = {
        # Life expectancy
        'life_expectancy_at_birth': RangeValidationConfig(
            column='life_expectancy_at_birth',
            min_value=60.0,
            max_value=95.0,
            typical_min=70.0,
            typical_max=90.0,
            description="Life expectancy at birth in years"
        ),
        'life_expectancy_male': RangeValidationConfig(
            column='life_expectancy_male',
            min_value=60.0,
            max_value=93.0,
            typical_min=68.0,
            typical_max=88.0,
            description="Male life expectancy in years"
        ),
        'life_expectancy_female': RangeValidationConfig(
            column='life_expectancy_female',
            min_value=62.0,
            max_value=97.0,
            typical_min=72.0,
            typical_max=92.0,
            description="Female life expectancy in years"
        ),
        
        # Mortality rates (per 100,000 population)
        'mortality_rate_all_causes': RangeValidationConfig(
            column='mortality_rate_all_causes',
            min_value=100.0,
            max_value=2000.0,
            typical_min=300.0,
            typical_max=1200.0,
            description="All-cause mortality rate per 100,000"
        ),
        'infant_mortality_rate': RangeValidationConfig(
            column='infant_mortality_rate',
            min_value=0.0,
            max_value=50.0,
            typical_min=2.0,
            typical_max=10.0,
            description="Infant mortality rate per 1,000 live births"
        ),
        
        # Health risk factors (percentages)
        'smoking_rate': RangeValidationConfig(
            column='smoking_rate',
            min_value=0.0,
            max_value=100.0,
            typical_min=5.0,
            typical_max=30.0,
            description="Percentage of population who smoke"
        ),
        'obesity_rate': RangeValidationConfig(
            column='obesity_rate',
            min_value=0.0,
            max_value=100.0,
            typical_min=15.0,
            typical_max=40.0,
            description="Percentage of population classified as obese"
        ),
        'diabetes_prevalence': RangeValidationConfig(
            column='diabetes_prevalence',
            min_value=0.0,
            max_value=100.0,
            typical_min=2.0,
            typical_max=15.0,
            description="Percentage of population with diabetes"
        ),
        
        # SEIFA scores
        'seifa_index_disadvantage': RangeValidationConfig(
            column='seifa_index_disadvantage',
            min_value=400.0,
            max_value=1400.0,
            typical_min=600.0,
            typical_max=1200.0,
            description="SEIFA Index of Relative Socio-economic Disadvantage"
        ),
        'seifa_index_advantage': RangeValidationConfig(
            column='seifa_index_advantage',
            min_value=400.0,
            max_value=1400.0,
            typical_min=600.0,
            typical_max=1200.0,
            description="SEIFA Index of Relative Socio-economic Advantage and Disadvantage"
        ),
        
        # Population and demographics
        'population_total': RangeValidationConfig(
            column='population_total',
            min_value=0,
            max_value=100000,
            typical_min=100,
            typical_max=50000,
            description="Total population count"
        ),
        'median_age': RangeValidationConfig(
            column='median_age',
            min_value=15.0,
            max_value=80.0,
            typical_min=25.0,
            typical_max=55.0,
            description="Median age of population"
        ),
        'population_density': RangeValidationConfig(
            column='population_density',
            min_value=0.0,
            max_value=50000.0,
            typical_min=0.1,
            typical_max=5000.0,
            description="Population per square kilometre"
        ),
        
        # Healthcare access
        'gp_per_1000_population': RangeValidationConfig(
            column='gp_per_1000_population',
            min_value=0.0,
            max_value=10.0,
            typical_min=0.5,
            typical_max=3.0,
            description="General practitioners per 1,000 population"
        ),
        'hospital_beds_per_1000': RangeValidationConfig(
            column='hospital_beds_per_1000',
            min_value=0.0,
            max_value=20.0,
            typical_min=1.0,
            typical_max=8.0,
            description="Hospital beds per 1,000 population"
        )
    }
    
    # Expected correlations in health data
    EXPECTED_CORRELATIONS = [
        {
            'variables': ['seifa_index_disadvantage', 'life_expectancy_at_birth'],
            'expected_correlation': 0.4,
            'tolerance': 0.2,
            'description': 'Higher SEIFA scores typically correlate with higher life expectancy'
        },
        {
            'variables': ['smoking_rate', 'mortality_rate_all_causes'],
            'expected_correlation': 0.3,
            'tolerance': 0.2,
            'description': 'Smoking rates positively correlate with mortality'
        },
        {
            'variables': ['obesity_rate', 'diabetes_prevalence'],
            'expected_correlation': 0.5,
            'tolerance': 0.2,
            'description': 'Obesity and diabetes show strong positive correlation'
        },
        {
            'variables': ['population_density', 'gp_per_1000_population'],
            'expected_correlation': -0.2,
            'tolerance': 0.3,
            'description': 'Urban areas may have lower GP ratios due to efficiency'
        },
        {
            'variables': ['median_age', 'hospital_beds_per_1000'],
            'expected_correlation': 0.3,
            'tolerance': 0.2,
            'description': 'Older populations require more hospital resources'
        }
    ]
    
    def __init__(
        self,
        validator_id: str = "advanced_statistical_validator",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialise the advanced statistical validator.
        
        Args:
            validator_id: Unique identifier for this validator
            config: Configuration dictionary
            logger: Optional logger instance
        """
        super().__init__(validator_id, config or {}, logger)
        
        # Initialise range configurations
        self.range_configs = self._initialise_range_configs()
        
        # Statistical thresholds
        self.outlier_config = self.config.get('outlier_detection', {
            'iqr_multiplier': 1.5,
            'z_score_threshold': 3.0,
            'modified_z_threshold': 3.5,
            'isolation_contamination': 0.05
        })
        
        # Distribution validation config
        self.distribution_config = self.config.get('distribution_validation', {
            'sex_ratio_tolerance': 0.05,  # 45-55% range
            'age_distribution_bins': [0, 15, 25, 45, 65, 85, 120],
            'normality_alpha': 0.05
        })
        
        # Reporting configuration
        self.reporting_config = self.config.get('reporting', {
            'include_visualisations': True,
            'include_recommendations': True,
            'summary_statistics': True
        })
        
        # Cache for statistical calculations
        self._stats_cache: Dict[str, Any] = {}
        
    def validate(self, data: DataBatch) -> List[ValidationResult]:
        """
        Perform comprehensive statistical validation.
        
        Args:
            data: Batch of data records to validate
            
        Returns:
            List[ValidationResult]: Validation results
        """
        if not data:
            return [ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="advanced_statistical_empty_data",
                message="Cannot perform statistical validation on empty dataset"
            )]
        
        results = []
        start_time = datetime.now()
        
        try:
            # 1. Range validation
            self.logger.info("Performing range validation...")
            range_results = self._validate_ranges(data)
            results.extend(range_results)
            
            # 2. Outlier detection
            self.logger.info("Performing outlier detection...")
            outlier_results = self._detect_outliers_comprehensive(data)
            results.extend(outlier_results)
            
            # 3. Correlation validation
            self.logger.info("Performing correlation validation...")
            correlation_results = self._validate_correlations(data)
            results.extend(correlation_results)
            
            # 4. Distribution validation
            self.logger.info("Performing distribution validation...")
            distribution_results = self._validate_distributions(data)
            results.extend(distribution_results)
            
            # 5. Generate comprehensive report if configured
            if self.reporting_config.get('include_report', True):
                report = self._generate_comprehensive_report(data, results)
                self._log_report_summary(report)
            
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"Advanced statistical validation completed in {duration:.2f}s: "
                f"{len(results)} issues found across {len(data)} records"
            )
            
        except Exception as e:
            self.logger.error(f"Advanced statistical validation failed: {e}")
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="advanced_statistical_error",
                message=f"Statistical validation failed: {str(e)}"
            ))
        
        return results
    
    def _initialise_range_configs(self) -> Dict[str, RangeValidationConfig]:
        """Initialise range configurations from defaults and custom config."""
        configs = dict(self.HEALTH_INDICATOR_RANGES)
        
        # Add custom range configurations from config
        custom_ranges = self.config.get('custom_ranges', {})
        for column, range_config in custom_ranges.items():
            configs[column] = RangeValidationConfig(
                column=column,
                **range_config
            )
        
        return configs
    
    def _validate_ranges(self, data: DataBatch) -> List[ValidationResult]:
        """Validate numeric fields against expected ranges."""
        results = []
        
        for column, config in self.range_configs.items():
            violations = []
            typical_violations = []
            
            for idx, record in enumerate(data):
                value = record.get(column)
                
                if value is None or not isinstance(value, (int, float)):
                    continue
                
                # Check absolute bounds
                if config.min_value is not None and value < config.min_value:
                    violations.append({
                        'index': idx,
                        'value': value,
                        'violation': 'below_minimum',
                        'threshold': config.min_value
                    })
                elif config.max_value is not None and value > config.max_value:
                    violations.append({
                        'index': idx,
                        'value': value,
                        'violation': 'above_maximum',
                        'threshold': config.max_value
                    })
                # Check typical bounds
                elif config.typical_min is not None and value < config.typical_min:
                    typical_violations.append({
                        'index': idx,
                        'value': value,
                        'violation': 'below_typical',
                        'threshold': config.typical_min
                    })
                elif config.typical_max is not None and value > config.typical_max:
                    typical_violations.append({
                        'index': idx,
                        'value': value,
                        'violation': 'above_typical',
                        'threshold': config.typical_max
                    })
            
            # Create validation results for absolute violations
            if violations:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=config.severity_if_outside_absolute,
                    rule_id=f"range_validation_{column}",
                    message=f"{len(violations)} values outside absolute range for {column}: {config.description}",
                    details={
                        'column': column,
                        'violations': violations[:10],  # Limit for performance
                        'total_violations': len(violations),
                        'min_allowed': config.min_value,
                        'max_allowed': config.max_value
                    },
                    affected_records=[v['index'] for v in violations[:10]]
                ))
            
            # Create validation results for typical range violations
            if typical_violations:
                results.append(ValidationResult(
                    is_valid=True,
                    severity=config.severity_if_outside_typical,
                    rule_id=f"typical_range_{column}",
                    message=f"{len(typical_violations)} values outside typical range for {column}",
                    details={
                        'column': column,
                        'violations': typical_violations[:10],
                        'total_violations': len(typical_violations),
                        'typical_min': config.typical_min,
                        'typical_max': config.typical_max
                    },
                    affected_records=[v['index'] for v in typical_violations[:10]]
                ))
        
        return results
    
    def _detect_outliers_comprehensive(self, data: DataBatch) -> List[ValidationResult]:
        """Detect outliers using multiple methods."""
        results = []
        numeric_columns = self._identify_numeric_columns(data)
        
        for column in numeric_columns:
            values = self._extract_numeric_values(data, column)
            
            if len(values) < 10:  # Need sufficient data for outlier detection
                continue
            
            outliers_by_method = {}
            
            # 1. IQR method
            iqr_outliers = self._detect_iqr_outliers(values)
            if iqr_outliers:
                outliers_by_method['IQR'] = iqr_outliers
            
            # 2. Z-score method (for larger samples)
            if len(values) >= 30:
                zscore_outliers = self._detect_zscore_outliers(values)
                if zscore_outliers:
                    outliers_by_method['Z-score'] = zscore_outliers
            
            # 3. Modified Z-score (for smaller samples)
            if len(values) < 30:
                modified_zscore_outliers = self._detect_modified_zscore_outliers(values)
                if modified_zscore_outliers:
                    outliers_by_method['Modified Z-score'] = modified_zscore_outliers
            
            # Consensus outliers (detected by multiple methods)
            if len(outliers_by_method) >= 2:
                consensus_outliers = self._find_consensus_outliers(outliers_by_method)
                
                if consensus_outliers:
                    # Find actual indices in original data
                    outlier_indices = []
                    outlier_values = []
                    
                    for idx, record in enumerate(data):
                        value = record.get(column)
                        if value is not None and any(
                            abs(value - values[outlier_idx]) < 1e-10
                            for outlier_idx in consensus_outliers
                        ):
                            outlier_indices.append(idx)
                            outlier_values.append(value)
                    
                    if outlier_indices:
                        # Determine severity based on context
                        severity = self._determine_outlier_severity(column, outlier_values, values)
                        
                        results.append(ValidationResult(
                            is_valid=severity != ValidationSeverity.ERROR,
                            severity=severity,
                            rule_id=f"outlier_detection_{column}",
                            message=f"Detected {len(outlier_indices)} outliers in {column}",
                            details={
                                'column': column,
                                'outlier_count': len(outlier_indices),
                                'methods_agreed': list(outliers_by_method.keys()),
                                'sample_outliers': outlier_values[:5],
                                'statistics': {
                                    'mean': np.mean(values),
                                    'median': np.median(values),
                                    'std': np.std(values),
                                    'min': np.min(values),
                                    'max': np.max(values)
                                }
                            },
                            affected_records=outlier_indices[:10]
                        ))
        
        # Multivariate outlier detection for related columns
        multivariate_results = self._detect_multivariate_outliers(data)
        results.extend(multivariate_results)
        
        return results
    
    def _detect_iqr_outliers(self, values: List[float]) -> List[int]:
        """Detect outliers using IQR method."""
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        multiplier = self.outlier_config['iqr_multiplier']
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        outliers = []
        for i, value in enumerate(values):
            if value < lower_bound or value > upper_bound:
                outliers.append(i)
        
        return outliers
    
    def _detect_zscore_outliers(self, values: List[float]) -> List[int]:
        """Detect outliers using Z-score method."""
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return []
        
        threshold = self.outlier_config['z_score_threshold']
        outliers = []
        
        for i, value in enumerate(values):
            z_score = abs((value - mean_val) / std_val)
            if z_score > threshold:
                outliers.append(i)
        
        return outliers
    
    def _detect_modified_zscore_outliers(self, values: List[float]) -> List[int]:
        """Detect outliers using modified Z-score (MAD method)."""
        median_val = np.median(values)
        mad = np.median(np.abs(values - median_val))
        
        if mad == 0:
            return []
        
        threshold = self.outlier_config['modified_z_threshold']
        outliers = []
        
        for i, value in enumerate(values):
            modified_z = 0.6745 * (value - median_val) / mad
            if abs(modified_z) > threshold:
                outliers.append(i)
        
        return outliers
    
    def _find_consensus_outliers(self, outliers_by_method: Dict[str, List[int]]) -> List[int]:
        """Find outliers detected by multiple methods."""
        all_outliers = []
        for outliers in outliers_by_method.values():
            all_outliers.extend(outliers)
        
        # Count occurrences
        outlier_counts = defaultdict(int)
        for outlier in all_outliers:
            outlier_counts[outlier] += 1
        
        # Return outliers detected by at least 2 methods
        consensus = [
            outlier for outlier, count in outlier_counts.items()
            if count >= 2
        ]
        
        return consensus
    
    def _determine_outlier_severity(
        self,
        column: str,
        outlier_values: List[float],
        all_values: List[float]
    ) -> ValidationSeverity:
        """Determine severity of outliers based on context."""
        # Check if outliers are impossible values
        if column in self.range_configs:
            config = self.range_configs[column]
            
            # Any value outside absolute bounds is an error
            for value in outlier_values:
                if config.min_value is not None and value < config.min_value:
                    return ValidationSeverity.ERROR
                if config.max_value is not None and value > config.max_value:
                    return ValidationSeverity.ERROR
        
        # Check extremity of outliers
        mean_val = np.mean(all_values)
        std_val = np.std(all_values)
        
        for value in outlier_values:
            z_score = abs((value - mean_val) / std_val) if std_val > 0 else 0
            if z_score > 5:  # Extremely unlikely values
                return ValidationSeverity.WARNING
        
        return ValidationSeverity.INFO
    
    def _detect_multivariate_outliers(self, data: DataBatch) -> List[ValidationResult]:
        """Detect multivariate outliers using Isolation Forest."""
        results = []
        
        # Define groups of related variables for multivariate analysis
        variable_groups = [
            {
                'name': 'health_indicators',
                'columns': ['life_expectancy_at_birth', 'mortality_rate_all_causes', 'infant_mortality_rate'],
                'contamination': 0.05
            },
            {
                'name': 'socioeconomic',
                'columns': ['seifa_index_disadvantage', 'median_age', 'population_density'],
                'contamination': 0.05
            },
            {
                'name': 'risk_factors',
                'columns': ['smoking_rate', 'obesity_rate', 'diabetes_prevalence'],
                'contamination': 0.05
            }
        ]
        
        for group in variable_groups:
            # Check if all columns exist
            columns = group['columns']
            if not all(any(col in record for record in data[:10]) for col in columns):
                continue
            
            # Prepare feature matrix
            feature_matrix = []
            valid_indices = []
            
            for idx, record in enumerate(data):
                row = []
                valid_row = True
                
                for col in columns:
                    value = record.get(col)
                    if value is not None and isinstance(value, (int, float)):
                        row.append(float(value))
                    else:
                        valid_row = False
                        break
                
                if valid_row:
                    feature_matrix.append(row)
                    valid_indices.append(idx)
            
            if len(feature_matrix) < 20:  # Need sufficient data
                continue
            
            # Standardise features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_matrix)
            
            # Apply Isolation Forest
            iso_forest = IsolationForest(
                contamination=group['contamination'],
                random_state=42
            )
            outlier_labels = iso_forest.fit_predict(scaled_features)
            
            # Find outliers
            outlier_indices = [
                valid_indices[i] for i, label in enumerate(outlier_labels)
                if label == -1
            ]
            
            if outlier_indices:
                results.append(ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.INFO,
                    rule_id=f"multivariate_outliers_{group['name']}",
                    message=f"Detected {len(outlier_indices)} multivariate outliers in {group['name']} indicators",
                    details={
                        'group': group['name'],
                        'columns': columns,
                        'outlier_count': len(outlier_indices),
                        'total_records': len(feature_matrix),
                        'contamination': group['contamination']
                    },
                    affected_records=outlier_indices[:10]
                ))
        
        return results
    
    def _validate_correlations(self, data: DataBatch) -> List[ValidationResult]:
        """Validate expected correlations in health data."""
        results = []
        
        for corr_config in self.EXPECTED_CORRELATIONS:
            var1, var2 = corr_config['variables']
            
            # Extract paired values
            paired_values = []
            for record in data:
                val1 = record.get(var1)
                val2 = record.get(var2)
                
                if (val1 is not None and isinstance(val1, (int, float)) and
                    val2 is not None and isinstance(val2, (int, float))):
                    paired_values.append((float(val1), float(val2)))
            
            if len(paired_values) < 30:  # Need sufficient data for correlation
                continue
            
            # Calculate correlation
            values1, values2 = zip(*paired_values)
            correlation, p_value = stats.pearsonr(values1, values2)
            
            # Check against expected correlation
            expected = corr_config['expected_correlation']
            tolerance = corr_config['tolerance']
            deviation = abs(correlation - expected)
            
            if deviation > tolerance:
                # Determine if correlation is in wrong direction
                wrong_direction = (expected > 0 and correlation < 0) or (expected < 0 and correlation > 0)
                
                severity = ValidationSeverity.ERROR if wrong_direction else ValidationSeverity.WARNING
                
                results.append(ValidationResult(
                    is_valid=not wrong_direction,
                    severity=severity,
                    rule_id=f"correlation_{var1}_{var2}",
                    message=f"Unexpected correlation between {var1} and {var2}: {correlation:.3f} (expected {expected:.3f} ± {tolerance:.3f})",
                    details={
                        'variable1': var1,
                        'variable2': var2,
                        'observed_correlation': correlation,
                        'expected_correlation': expected,
                        'deviation': deviation,
                        'p_value': p_value,
                        'sample_size': len(paired_values),
                        'description': corr_config['description'],
                        'wrong_direction': wrong_direction
                    }
                ))
        
        # Check for impossible correlations
        impossible_results = self._check_impossible_correlations(data)
        results.extend(impossible_results)
        
        return results
    
    def _check_impossible_correlations(self, data: DataBatch) -> List[ValidationResult]:
        """Check for correlations that should not exist."""
        results = []
        
        # Define pairs that should have near-zero or specific correlations
        impossible_pairs = [
            {
                'variables': ['population_total', 'median_age'],
                'max_correlation': 0.3,
                'description': 'Population size should not strongly correlate with median age'
            },
            {
                'variables': ['life_expectancy_male', 'life_expectancy_female'],
                'min_correlation': 0.7,
                'description': 'Male and female life expectancy should be highly correlated'
            }
        ]
        
        for pair_config in impossible_pairs:
            var1, var2 = pair_config['variables']
            
            # Check if variables exist
            if not all(any(var in record for record in data[:10]) for var in [var1, var2]):
                continue
            
            # Extract paired values
            paired_values = []
            for record in data:
                val1 = record.get(var1)
                val2 = record.get(var2)
                
                if (val1 is not None and isinstance(val1, (int, float)) and
                    val2 is not None and isinstance(val2, (int, float))):
                    paired_values.append((float(val1), float(val2)))
            
            if len(paired_values) < 30:
                continue
            
            values1, values2 = zip(*paired_values)
            correlation, p_value = stats.pearsonr(values1, values2)
            
            # Check constraints
            if 'max_correlation' in pair_config and abs(correlation) > pair_config['max_correlation']:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    rule_id=f"impossible_correlation_{var1}_{var2}",
                    message=f"Unexpected strong correlation between {var1} and {var2}: {correlation:.3f}",
                    details={
                        'variable1': var1,
                        'variable2': var2,
                        'observed_correlation': correlation,
                        'max_expected': pair_config['max_correlation'],
                        'description': pair_config['description']
                    }
                ))
            
            if 'min_correlation' in pair_config and abs(correlation) < pair_config['min_correlation']:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    rule_id=f"weak_expected_correlation_{var1}_{var2}",
                    message=f"Unexpectedly weak correlation between {var1} and {var2}: {correlation:.3f}",
                    details={
                        'variable1': var1,
                        'variable2': var2,
                        'observed_correlation': correlation,
                        'min_expected': pair_config['min_correlation'],
                        'description': pair_config['description']
                    }
                ))
        
        return results
    
    def _validate_distributions(self, data: DataBatch) -> List[ValidationResult]:
        """Validate demographic and health indicator distributions."""
        results = []
        
        # 1. Validate sex distribution
        sex_results = self._validate_sex_distribution(data)
        results.extend(sex_results)
        
        # 2. Validate age distribution
        age_results = self._validate_age_distribution(data)
        results.extend(age_results)
        
        # 3. Validate percentage fields sum to 100
        percentage_results = self._validate_percentage_distributions(data)
        results.extend(percentage_results)
        
        # 4. Validate expected distributions for health indicators
        health_distribution_results = self._validate_health_distributions(data)
        results.extend(health_distribution_results)
        
        return results
    
    def _validate_sex_distribution(self, data: DataBatch) -> List[ValidationResult]:
        """Validate male/female distribution."""
        results = []
        
        # Look for sex-related columns
        male_columns = ['population_male', 'male_population', 'males']
        female_columns = ['population_female', 'female_population', 'females']
        
        male_col = None
        female_col = None
        
        # Find which columns exist
        for col in male_columns:
            if any(col in record for record in data[:10]):
                male_col = col
                break
        
        for col in female_columns:
            if any(col in record for record in data[:10]):
                female_col = col
                break
        
        if male_col and female_col:
            tolerance = self.distribution_config['sex_ratio_tolerance']
            
            for idx, record in enumerate(data):
                male_pop = record.get(male_col)
                female_pop = record.get(female_col)
                
                if (male_pop is not None and isinstance(male_pop, (int, float)) and
                    female_pop is not None and isinstance(female_pop, (int, float))):
                    
                    total_pop = male_pop + female_pop
                    if total_pop > 0:
                        male_ratio = male_pop / total_pop
                        female_ratio = female_pop / total_pop
                        
                        # Check if ratios are within expected bounds (45-55%)
                        if male_ratio < (0.5 - tolerance) or male_ratio > (0.5 + tolerance):
                            results.append(ValidationResult(
                                is_valid=True,
                                severity=ValidationSeverity.WARNING,
                                rule_id="sex_distribution_imbalance",
                                message=f"Unusual sex distribution in record {idx}: {male_ratio:.1%} male",
                                details={
                                    'male_population': male_pop,
                                    'female_population': female_pop,
                                    'male_ratio': male_ratio,
                                    'female_ratio': female_ratio,
                                    'expected_range': f"{50-tolerance*100:.0f}%-{50+tolerance*100:.0f}%"
                                },
                                affected_records=[idx]
                            ))
        
        return results
    
    def _validate_age_distribution(self, data: DataBatch) -> List[ValidationResult]:
        """Validate age distribution patterns."""
        results = []
        
        # Look for age-related columns
        age_columns = [
            'age_0_14', 'age_15_24', 'age_25_44', 'age_45_64', 'age_65_plus',
            'age_0_14_percentage', 'age_15_24_percentage', 'age_25_44_percentage',
            'age_45_64_percentage', 'age_65_plus_percentage'
        ]
        
        # Find which age columns exist
        existing_age_cols = []
        for col in age_columns:
            if any(col in record for record in data[:10]):
                existing_age_cols.append(col)
        
        if len(existing_age_cols) >= 3:  # Need at least 3 age groups
            # Check if percentages sum to 100
            percentage_cols = [col for col in existing_age_cols if 'percentage' in col]
            
            if len(percentage_cols) >= 3:
                for idx, record in enumerate(data):
                    percentages = []
                    for col in percentage_cols:
                        val = record.get(col)
                        if val is not None and isinstance(val, (int, float)):
                            percentages.append(float(val))
                    
                    if len(percentages) == len(percentage_cols):
                        total_percentage = sum(percentages)
                        
                        if abs(total_percentage - 100.0) > 1.0:  # Allow 1% tolerance
                            results.append(ValidationResult(
                                is_valid=False,
                                severity=ValidationSeverity.ERROR,
                                rule_id="age_distribution_sum",
                                message=f"Age percentages do not sum to 100% in record {idx}: {total_percentage:.1f}%",
                                details={
                                    'columns': percentage_cols,
                                    'values': percentages,
                                    'sum': total_percentage
                                },
                                affected_records=[idx]
                            ))
        
        return results
    
    def _validate_percentage_distributions(self, data: DataBatch) -> List[ValidationResult]:
        """Validate that percentage fields are within valid ranges."""
        results = []
        
        # Find all percentage columns
        percentage_columns = []
        if data:
            sample_record = data[0]
            for col in sample_record.keys():
                if any(term in col.lower() for term in ['percentage', 'percent', 'rate', 'proportion']):
                    percentage_columns.append(col)
        
        for col in percentage_columns:
            invalid_percentages = []
            
            for idx, record in enumerate(data):
                value = record.get(col)
                
                if value is not None and isinstance(value, (int, float)):
                    if value < 0 or value > 100:
                        invalid_percentages.append({
                            'index': idx,
                            'value': value
                        })
            
            if invalid_percentages:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id=f"invalid_percentage_{col}",
                    message=f"{len(invalid_percentages)} invalid percentage values in {col}",
                    details={
                        'column': col,
                        'invalid_count': len(invalid_percentages),
                        'sample_violations': invalid_percentages[:5]
                    },
                    affected_records=[v['index'] for v in invalid_percentages[:10]]
                ))
        
        return results
    
    def _validate_health_distributions(self, data: DataBatch) -> List[ValidationResult]:
        """Validate distributions of health indicators."""
        results = []
        
        # Define expected distribution patterns for health indicators
        distribution_checks = [
            {
                'column': 'life_expectancy_at_birth',
                'expected_mean_range': (78, 85),
                'expected_std_range': (1, 5),
                'description': 'Life expectancy should be normally distributed around 80-83 years'
            },
            {
                'column': 'smoking_rate',
                'expected_mean_range': (10, 20),
                'expected_std_range': (2, 8),
                'description': 'Smoking rates typically range from 10-20% with moderate variation'
            }
        ]
        
        for check in distribution_checks:
            column = check['column']
            values = self._extract_numeric_values(data, column)
            
            if len(values) < 30:  # Need sufficient data
                continue
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # Check if distribution parameters are within expected ranges
            mean_min, mean_max = check['expected_mean_range']
            std_min, std_max = check['expected_std_range']
            
            issues = []
            
            if mean_val < mean_min or mean_val > mean_max:
                issues.append(f"mean {mean_val:.2f} outside expected range [{mean_min}, {mean_max}]")
            
            if std_val < std_min or std_val > std_max:
                issues.append(f"standard deviation {std_val:.2f} outside expected range [{std_min}, {std_max}]")
            
            if issues:
                results.append(ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    rule_id=f"distribution_anomaly_{column}",
                    message=f"Unusual distribution for {column}: {', '.join(issues)}",
                    details={
                        'column': column,
                        'observed_mean': mean_val,
                        'observed_std': std_val,
                        'expected_mean_range': check['expected_mean_range'],
                        'expected_std_range': check['expected_std_range'],
                        'description': check['description'],
                        'sample_size': len(values)
                    }
                ))
        
        return results
    
    def _generate_comprehensive_report(
        self,
        data: DataBatch,
        validation_results: List[ValidationResult]
    ) -> StatisticalReport:
        """Generate comprehensive statistical validation report."""
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(data)
        
        # Organise validation results by type
        range_violations = []
        outlier_summary = defaultdict(list)
        correlation_findings = []
        distribution_findings = []
        
        for result in validation_results:
            if 'range_validation' in result.rule_id or 'typical_range' in result.rule_id:
                range_violations.append(result.details)
            elif 'outlier' in result.rule_id:
                method = result.details.get('method', 'unknown')
                outlier_summary[method].append(result.details)
            elif 'correlation' in result.rule_id:
                correlation_findings.append(result.details)
            elif 'distribution' in result.rule_id:
                distribution_findings.append(result.details)
        
        # Count severities
        severity_counts = defaultdict(int)
        for result in validation_results:
            severity_counts[result.severity] += 1
        
        # Generate visualisation suggestions
        vis_suggestions = self._generate_visualisation_suggestions(
            validation_results,
            summary_stats
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(validation_results)
        
        return StatisticalReport(
            summary_statistics=summary_stats,
            range_violations=range_violations,
            outlier_summary=dict(outlier_summary),
            correlation_findings=correlation_findings,
            distribution_findings=distribution_findings,
            severity_counts=dict(severity_counts),
            visualisation_suggestions=vis_suggestions,
            recommendations=recommendations
        )
    
    def _calculate_summary_statistics(self, data: DataBatch) -> Dict[str, Dict[str, float]]:
        """Calculate summary statistics for all numeric columns."""
        summary_stats = {}
        numeric_columns = self._identify_numeric_columns(data)
        
        for column in numeric_columns:
            values = self._extract_numeric_values(data, column)
            
            if values:
                summary_stats[column] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'q1': np.percentile(values, 25),
                    'q3': np.percentile(values, 75),
                    'missing': len(data) - len(values),
                    'missing_percentage': (len(data) - len(values)) / len(data) * 100
                }
        
        return summary_stats
    
    def _generate_visualisation_suggestions(
        self,
        validation_results: List[ValidationResult],
        summary_stats: Dict[str, Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """Generate suggestions for data visualisations."""
        suggestions = []
        
        # Suggest box plots for columns with outliers
        outlier_columns = set()
        for result in validation_results:
            if 'outlier' in result.rule_id:
                column = result.details.get('column')
                if column:
                    outlier_columns.add(column)
        
        if outlier_columns:
            suggestions.append({
                'type': 'box_plot',
                'columns': list(outlier_columns),
                'purpose': 'Visualise outliers and distribution',
                'description': 'Box plots will show median, quartiles, and outliers'
            })
        
        # Suggest scatter plots for correlation issues
        correlation_pairs = []
        for result in validation_results:
            if 'correlation' in result.rule_id:
                var1 = result.details.get('variable1')
                var2 = result.details.get('variable2')
                if var1 and var2:
                    correlation_pairs.append((var1, var2))
        
        if correlation_pairs:
            suggestions.append({
                'type': 'scatter_plot',
                'variable_pairs': correlation_pairs,
                'purpose': 'Examine relationships between variables',
                'description': 'Scatter plots will reveal correlation patterns'
            })
        
        # Suggest histograms for distribution anomalies
        distribution_columns = set()
        for result in validation_results:
            if 'distribution' in result.rule_id:
                column = result.details.get('column')
                if column:
                    distribution_columns.add(column)
        
        if distribution_columns:
            suggestions.append({
                'type': 'histogram',
                'columns': list(distribution_columns),
                'purpose': 'Analyse distribution shapes',
                'description': 'Histograms will show frequency distributions'
            })
        
        # Suggest heatmap for multivariate outliers
        if any('multivariate' in result.rule_id for result in validation_results):
            suggestions.append({
                'type': 'correlation_heatmap',
                'purpose': 'Visualise relationships between multiple variables',
                'description': 'Heatmap will show correlation matrix'
            })
        
        return suggestions
    
    def _generate_recommendations(self, validation_results: List[ValidationResult]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Count issue types
        range_errors = sum(1 for r in validation_results if 'range_validation' in r.rule_id and r.severity == ValidationSeverity.ERROR)
        outlier_warnings = sum(1 for r in validation_results if 'outlier' in r.rule_id)
        correlation_issues = sum(1 for r in validation_results if 'correlation' in r.rule_id)
        distribution_issues = sum(1 for r in validation_results if 'distribution' in r.rule_id)
        
        if range_errors > 0:
            recommendations.append(
                f"Critical: {range_errors} records contain values outside acceptable ranges. "
                "These records should be reviewed for data entry errors or excluded from analysis."
            )
        
        if outlier_warnings > 10:
            recommendations.append(
                "High number of outliers detected. Consider:\n"
                "1. Investigating data collection methods in affected regions\n"
                "2. Checking for data entry or processing errors\n"
                "3. Using robust statistical methods for analysis"
            )
        
        if correlation_issues > 0:
            recommendations.append(
                "Unexpected correlations detected between variables. This may indicate:\n"
                "1. Data quality issues affecting multiple variables\n"
                "2. Unusual patterns requiring further investigation\n"
                "3. Need for domain expert review"
            )
        
        if distribution_issues > 0:
            recommendations.append(
                "Distribution anomalies detected. Recommended actions:\n"
                "1. Verify demographic data against census information\n"
                "2. Check for systematic biases in data collection\n"
                "3. Consider stratified analysis approaches"
            )
        
        # General recommendations
        if len(validation_results) > 50:
            recommendations.append(
                "Large number of validation issues detected. Consider:\n"
                "1. Reviewing data collection and processing pipeline\n"
                "2. Implementing additional data quality checks at source\n"
                "3. Creating a data quality improvement plan"
            )
        
        if not recommendations:
            recommendations.append(
                "Data quality is generally good. Continue monitoring for:\n"
                "1. Temporal changes in data patterns\n"
                "2. New outliers as more data is collected\n"
                "3. Maintaining correlation patterns"
            )
        
        return recommendations
    
    def _log_report_summary(self, report: StatisticalReport) -> None:
        """Log a summary of the statistical report."""
        self.logger.info("Statistical Validation Report Summary:")
        self.logger.info(f"  Total columns analysed: {len(report.summary_statistics)}")
        
        for severity, count in report.severity_counts.items():
            self.logger.info(f"  {severity.value.upper()}: {count} issues")
        
        self.logger.info(f"  Range violations: {len(report.range_violations)}")
        self.logger.info(f"  Outlier detection methods used: {len(report.outlier_summary)}")
        self.logger.info(f"  Correlation findings: {len(report.correlation_findings)}")
        self.logger.info(f"  Distribution findings: {len(report.distribution_findings)}")
        self.logger.info(f"  Recommendations: {len(report.recommendations)}")
    
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
    
    def _extract_numeric_values(self, data: DataBatch, column: str) -> List[float]:
        """Extract numeric values for a specific column."""
        values = []
        
        for record in data:
            value = record.get(column)
            if value is not None and isinstance(value, (int, float)) and not math.isnan(float(value)):
                values.append(float(value))
        
        return values
    
    def get_validation_rules(self) -> List[str]:
        """Get the list of validation rules supported by this validator."""
        rules = []
        
        # Range validation rules
        for column in self.range_configs:
            rules.append(f"range_validation_{column}")
            rules.append(f"typical_range_{column}")
        
        # Outlier detection rules
        rules.extend([
            "outlier_detection_iqr",
            "outlier_detection_zscore",
            "outlier_detection_modified_zscore",
            "outlier_detection_multivariate"
        ])
        
        # Correlation rules
        rules.extend([
            f"correlation_{var1}_{var2}"
            for config in self.EXPECTED_CORRELATIONS
            for var1, var2 in [config['variables']]
        ])
        
        # Distribution rules
        rules.extend([
            "sex_distribution_imbalance",
            "age_distribution_sum",
            "invalid_percentage",
            "distribution_anomaly"
        ])
        
        return rules