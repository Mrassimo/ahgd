"""
Statistical Validation Framework

This module provides comprehensive statistical validation methods including
outlier detection (IQR, Z-score, isolation forest), distribution analysis,
correlation validation, trend analysis, and statistical significance testing
for Australian Health Geography Datasets.
"""

import logging
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
class OutlierDetectionResult:
    """Result of outlier detection analysis."""
    method: str
    column: str
    outliers_detected: int
    outlier_indices: List[int]
    threshold: float
    statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributionTestResult:
    """Result of distribution testing."""
    column: str
    test_name: str
    statistic: float
    p_value: float
    is_normal: bool
    confidence_level: float = 0.05


@dataclass
class CorrelationAnalysisResult:
    """Result of correlation analysis."""
    variable1: str
    variable2: str
    correlation_coefficient: float
    p_value: float
    correlation_type: str  # 'pearson', 'spearman', 'kendall'
    is_significant: bool
    expected_correlation: Optional[float] = None
    deviation: Optional[float] = None


@dataclass
class TrendAnalysisResult:
    """Result of trend analysis."""
    column: str
    trend_direction: str  # 'increasing', 'decreasing', 'stable', 'irregular'
    slope: float
    r_squared: float
    p_value: float
    change_points: List[int] = field(default_factory=list)


class StatisticalValidator(BaseValidator):
    """
    Comprehensive statistical validation framework.
    
    This validator provides advanced statistical methods for data quality
    assessment including outlier detection, distribution analysis, correlation
    validation, trend analysis, and statistical significance testing.
    """
    
    def __init__(
        self,
        validator_id: str = "statistical_validator",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialise the statistical validator.
        
        Args:
            validator_id: Unique identifier for this validator
            config: Configuration dictionary containing statistical rules
            logger: Optional logger instance
        """
        super().__init__(validator_id, config or {}, logger)
        
        # Load statistical validation configuration
        self.statistical_config = self.config.get('statistical_rules', {})
        
        # Outlier detection configuration
        self.outlier_config = self.statistical_config.get('outlier_detection', {})
        
        # Distribution analysis configuration
        self.distribution_config = self.statistical_config.get('distribution_analysis', {})
        
        # Correlation analysis configuration
        self.correlation_config = self.statistical_config.get('correlation_analysis', {})
        
        # Time series analysis configuration
        self.timeseries_config = self.statistical_config.get('time_series_analysis', {})
        
        # Significance testing configuration
        self.significance_config = self.statistical_config.get('significance_testing', {})
        
        # Performance configuration
        self.performance_config = self.statistical_config.get('performance', {})
        self.enable_parallel = self.performance_config.get('enable_parallel', True)
        self.max_workers = self.performance_config.get('max_workers', 4)
        self.enable_sampling = self.performance_config.get('sampling', {}).get('enable_sampling_for_large_datasets', True)
        self.sample_threshold = self.performance_config.get('sampling', {}).get('sample_size_threshold', 10000)
        
        # Caching configuration
        self.enable_caching = self.performance_config.get('caching', {}).get('cache_statistical_calculations', True)
        self._statistical_cache: Dict[str, Any] = {}
        
        # Statistics tracking
        self._validation_statistics = defaultdict(int)
        
    def validate(self, data: DataBatch) -> List[ValidationResult]:
        """
        Perform comprehensive statistical validation.
        
        Args:
            data: Batch of data records to validate
            
        Returns:
            List[ValidationResult]: Statistical validation results
        """
        if not data:
            return [ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="statistical_empty_data",
                message="Cannot perform statistical validation on empty dataset"
            )]
        
        results = []
        start_time = datetime.now()
        
        try:
            # Sample data if it's too large
            working_data = self._prepare_data_for_analysis(data)
            
            # Outlier detection
            outlier_results = self.detect_outliers(working_data)
            results.extend(self._convert_outlier_results_to_validation(outlier_results))
            
            # Distribution analysis
            distribution_results = self.analyze_distributions(working_data)
            results.extend(self._convert_distribution_results_to_validation(distribution_results))
            
            # Correlation analysis
            correlation_results = self.analyze_correlations(working_data)
            results.extend(self._convert_correlation_results_to_validation(correlation_results))
            
            # Trend analysis (if temporal data available)
            trend_results = self.analyze_trends(working_data)
            results.extend(self._convert_trend_results_to_validation(trend_results))
            
            # Statistical significance testing
            significance_results = self.perform_significance_tests(working_data)
            results.extend(significance_results)
            
            # Advanced statistical methods
            if self.statistical_config.get('advanced_methods', {}).get('cluster_analysis', {}).get('enabled', False):
                cluster_results = self.perform_cluster_analysis(working_data)
                results.extend(cluster_results)
            
            # Update statistics
            self._update_validation_statistics(results, len(data))
            
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"Statistical validation completed in {duration:.2f}s: "
                f"{len(results)} statistical issues found across {len(data)} records"
            )
            
        except Exception as e:
            self.logger.error(f"Statistical validation failed: {e}")
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="statistical_validation_error",
                message=f"Statistical validation failed: {str(e)}"
            ))
        
        return results
    
    def detect_outliers(self, data: DataBatch) -> List[OutlierDetectionResult]:
        """
        Detect outliers using multiple statistical methods.
        
        Args:
            data: Batch of data records
            
        Returns:
            List[OutlierDetectionResult]: Outlier detection results
        """
        results = []
        
        # IQR-based outlier detection
        if self.outlier_config.get('iqr_method', {}).get('enabled', True):
            iqr_results = self._detect_iqr_outliers(data)
            results.extend(iqr_results)
        
        # Z-score based outlier detection
        if self.outlier_config.get('z_score_method', {}).get('enabled', True):
            zscore_results = self._detect_zscore_outliers(data)
            results.extend(zscore_results)
        
        # Modified Z-score using median absolute deviation
        if self.outlier_config.get('modified_z_score', {}).get('enabled', True):
            modified_zscore_results = self._detect_modified_zscore_outliers(data)
            results.extend(modified_zscore_results)
        
        # Isolation Forest (multivariate outlier detection)
        if self.outlier_config.get('isolation_forest', {}).get('enabled', True):
            isolation_results = self._detect_isolation_forest_outliers(data)
            results.extend(isolation_results)
        
        return results
    
    def analyze_distributions(self, data: DataBatch) -> List[DistributionTestResult]:
        """
        Analyze data distributions and test for normality.
        
        Args:
            data: Batch of data records
            
        Returns:
            List[DistributionTestResult]: Distribution analysis results
        """
        results = []
        
        # Get numeric columns
        numeric_columns = self._identify_numeric_columns(data)
        
        # Normality tests
        normality_tests = self.distribution_config.get('normality_tests', {})
        
        for column in numeric_columns:
            values = self._extract_numeric_values(data, column)
            if len(values) < 8:  # Need minimum samples for statistical tests
                continue
            
            # Shapiro-Wilk test
            if normality_tests.get('shapiro_wilk', {}).get('enabled', True):
                sw_result = self._perform_shapiro_wilk_test(values, column)
                if sw_result:
                    results.append(sw_result)
            
            # Kolmogorov-Smirnov test
            if normality_tests.get('kolmogorov_smirnov', {}).get('enabled', True):
                ks_result = self._perform_kolmogorov_smirnov_test(values, column)
                if ks_result:
                    results.append(ks_result)
            
            # Anderson-Darling test
            if normality_tests.get('anderson_darling', {}).get('enabled', True):
                ad_result = self._perform_anderson_darling_test(values, column)
                if ad_result:
                    results.append(ad_result)
        
        return results
    
    def analyze_correlations(self, data: DataBatch) -> List[CorrelationAnalysisResult]:
        """
        Analyze correlations between variables.
        
        Args:
            data: Batch of data records
            
        Returns:
            List[CorrelationAnalysisResult]: Correlation analysis results
        """
        results = []
        
        # Get expected correlations from configuration
        expected_correlations = self.correlation_config.get('pearson_correlation', {}).get('expected_correlations', [])
        
        for corr_config in expected_correlations:
            variables = corr_config.get('variables', [])
            if len(variables) != 2:
                continue
            
            var1, var2 = variables
            values1 = self._extract_numeric_values(data, var1)
            values2 = self._extract_numeric_values(data, var2)
            
            # Ensure we have matching pairs
            min_length = min(len(values1), len(values2))
            if min_length < 10:  # Need minimum samples for correlation
                continue
            
            values1 = values1[:min_length]
            values2 = values2[:min_length]
            
            # Calculate Pearson correlation
            correlation, p_value = stats.pearsonr(values1, values2)
            
            # Check against expected correlation
            expected_corr = corr_config.get('expected_correlation', 0)
            tolerance = corr_config.get('tolerance', 0.2)
            deviation = abs(correlation - expected_corr)
            
            results.append(CorrelationAnalysisResult(
                variable1=var1,
                variable2=var2,
                correlation_coefficient=correlation,
                p_value=p_value,
                correlation_type='pearson',
                is_significant=p_value < 0.05,
                expected_correlation=expected_corr,
                deviation=deviation
            ))
        
        return results
    
    def analyze_trends(self, data: DataBatch) -> List[TrendAnalysisResult]:
        """
        Analyze temporal trends in the data.
        
        Args:
            data: Batch of data records
            
        Returns:
            List[TrendAnalysisResult]: Trend analysis results
        """
        results = []
        
        # Check if we have temporal data
        if not self._has_temporal_data(data):
            return results
        
        trend_config = self.timeseries_config.get('trend_analysis', {})
        if not trend_config.get('enabled', True):
            return results
        
        # Group data by entity (e.g., SA2) for time series analysis
        entity_column = 'sa2_code'  # Assume SA2 as the entity
        grouped_data = self._group_data_by_entity(data, entity_column)
        
        # Analyze trends for each entity
        trend_columns = trend_config.get('population_trends', {}).get('columns', [])
        
        for entity_id, entity_records in grouped_data.items():
            if len(entity_records) < 3:  # Need minimum points for trend analysis
                continue
            
            for column in trend_columns:
                trend_result = self._analyze_entity_trend(entity_records, column)
                if trend_result:
                    results.append(trend_result)
        
        return results
    
    def perform_significance_tests(self, data: DataBatch) -> List[ValidationResult]:
        """
        Perform statistical significance tests.
        
        Args:
            data: Batch of data records
            
        Returns:
            List[ValidationResult]: Significance test results
        """
        results = []
        
        significance_config = self.significance_config
        
        # T-tests
        t_tests = significance_config.get('t_tests', {})
        if t_tests.get('enabled', True):
            t_test_results = self._perform_t_tests(data, t_tests.get('comparisons', []))
            results.extend(t_test_results)
        
        # Chi-square tests
        chi_square_tests = significance_config.get('chi_square_tests', {})
        if chi_square_tests.get('enabled', True):
            chi_square_results = self._perform_chi_square_tests(data, chi_square_tests.get('contingency_tables', []))
            results.extend(chi_square_results)
        
        return results
    
    def perform_cluster_analysis(self, data: DataBatch) -> List[ValidationResult]:
        """
        Perform cluster analysis for anomaly detection.
        
        Args:
            data: Batch of data records
            
        Returns:
            List[ValidationResult]: Cluster analysis results
        """
        results = []
        
        cluster_config = self.statistical_config.get('advanced_methods', {}).get('cluster_analysis', {})
        if not cluster_config.get('enabled', True):
            return results
        
        # Prepare feature matrix
        features = cluster_config.get('features', [])
        feature_matrix = self._prepare_feature_matrix(data, features)
        
        if feature_matrix.shape[0] < 10 or feature_matrix.shape[1] < 2:
            return results
        
        # Standardise features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix)
        
        # Perform K-means clustering
        k_range = cluster_config.get('k_range', [3, 8])
        best_k = self._find_optimal_clusters(scaled_features, k_range)
        
        # Apply clustering
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # Identify potential anomalies (small clusters)
        cluster_sizes = np.bincount(cluster_labels)
        min_cluster_size = len(data) * 0.05  # Clusters with < 5% of data are potentially anomalous
        
        anomalous_clusters = np.where(cluster_sizes < min_cluster_size)[0]
        
        if len(anomalous_clusters) > 0:
            anomalous_indices = np.where(np.isin(cluster_labels, anomalous_clusters))[0]
            
            results.append(ValidationResult(
                is_valid=True,  # Anomalies are informational
                severity=ValidationSeverity.INFO,
                rule_id="cluster_analysis_anomalies",
                message=f"Cluster analysis detected {len(anomalous_indices)} potential anomalies",
                details={
                    'method': 'kmeans',
                    'n_clusters': best_k,
                    'anomalous_clusters': anomalous_clusters.tolist(),
                    'features_used': features
                },
                affected_records=anomalous_indices.tolist()[:10]  # Limit for performance
            ))
        
        return results
    
    def get_validation_rules(self) -> List[str]:
        """
        Get the list of validation rules supported by this validator.
        
        Returns:
            List[str]: List of validation rule identifiers
        """
        return [
            "outlier_detection_iqr",
            "outlier_detection_zscore",
            "outlier_detection_modified_zscore",
            "outlier_detection_isolation_forest",
            "normality_test_shapiro_wilk",
            "normality_test_kolmogorov_smirnov",
            "normality_test_anderson_darling",
            "correlation_analysis",
            "trend_analysis",
            "significance_testing",
            "cluster_analysis"
        ]
    
    # Private methods for implementation
    
    def _prepare_data_for_analysis(self, data: DataBatch) -> DataBatch:
        """Prepare data for statistical analysis (sampling if needed)."""
        if not self.enable_sampling or len(data) <= self.sample_threshold:
            return data
        
        # Stratified sampling by state if possible
        state_column = 'state_code'
        if any(state_column in record for record in data):
            return self._stratified_sample(data, state_column)
        else:
            # Simple random sampling
            sample_size = min(self.sample_threshold, len(data))
            return np.random.choice(data, size=sample_size, replace=False).tolist()
    
    def _stratified_sample(self, data: DataBatch, stratify_column: str) -> DataBatch:
        """Perform stratified sampling."""
        # Group by stratification column
        groups = defaultdict(list)
        for record in data:
            key = record.get(stratify_column, 'unknown')
            groups[key].append(record)
        
        # Sample from each group proportionally
        sampled_data = []
        total_sample_size = min(self.sample_threshold, len(data))
        
        for group_key, group_records in groups.items():
            group_proportion = len(group_records) / len(data)
            group_sample_size = max(1, int(total_sample_size * group_proportion))
            
            if len(group_records) <= group_sample_size:
                sampled_data.extend(group_records)
            else:
                sampled_records = np.random.choice(
                    group_records, 
                    size=group_sample_size, 
                    replace=False
                ).tolist()
                sampled_data.extend(sampled_records)
        
        return sampled_data
    
    def _detect_iqr_outliers(self, data: DataBatch) -> List[OutlierDetectionResult]:
        """Detect outliers using IQR method."""
        results = []
        iqr_config = self.outlier_config.get('iqr_method', {})
        applicable_columns = iqr_config.get('applicable_columns', [])
        
        for column_config in applicable_columns:
            column = column_config['column']
            multiplier = column_config.get('multiplier', 1.5)
            
            values = self._extract_numeric_values(data, column)
            if len(values) < 4:
                continue
            
            # Calculate IQR
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr
            
            # Find outliers
            outlier_indices = []
            for idx, record in enumerate(data):
                value = record.get(column)
                if value is not None and isinstance(value, (int, float)):
                    if value < lower_bound or value > upper_bound:
                        outlier_indices.append(idx)
            
            if outlier_indices:
                results.append(OutlierDetectionResult(
                    method='iqr',
                    column=column,
                    outliers_detected=len(outlier_indices),
                    outlier_indices=outlier_indices,
                    threshold=multiplier,
                    statistics={
                        'q1': q1,
                        'q3': q3,
                        'iqr': iqr,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
                ))
        
        return results
    
    def _detect_zscore_outliers(self, data: DataBatch) -> List[OutlierDetectionResult]:
        """Detect outliers using Z-score method."""
        results = []
        zscore_config = self.outlier_config.get('z_score_method', {})
        applicable_columns = zscore_config.get('applicable_columns', [])
        
        for column_config in applicable_columns:
            column = column_config['column']
            threshold = column_config.get('threshold', 3.0)
            
            values = self._extract_numeric_values(data, column)
            if len(values) < 3:
                continue
            
            # Calculate Z-scores
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val == 0:  # All values are identical
                continue
            
            # Find outliers
            outlier_indices = []
            for idx, record in enumerate(data):
                value = record.get(column)
                if value is not None and isinstance(value, (int, float)):
                    z_score = abs((value - mean_val) / std_val)
                    if z_score > threshold:
                        outlier_indices.append(idx)
            
            if outlier_indices:
                results.append(OutlierDetectionResult(
                    method='z_score',
                    column=column,
                    outliers_detected=len(outlier_indices),
                    outlier_indices=outlier_indices,
                    threshold=threshold,
                    statistics={
                        'mean': mean_val,
                        'std': std_val
                    }
                ))
        
        return results
    
    def _detect_modified_zscore_outliers(self, data: DataBatch) -> List[OutlierDetectionResult]:
        """Detect outliers using modified Z-score (using median absolute deviation)."""
        results = []
        modified_zscore_config = self.outlier_config.get('modified_z_score', {})
        applicable_columns = modified_zscore_config.get('applicable_columns', [])
        
        for column_config in applicable_columns:
            column = column_config['column']
            threshold = column_config.get('threshold', 3.5)
            
            values = self._extract_numeric_values(data, column)
            if len(values) < 3:
                continue
            
            # Calculate modified Z-scores using median absolute deviation
            median_val = np.median(values)
            mad = np.median(np.abs(values - median_val))
            
            if mad == 0:  # All values are identical to median
                continue
            
            # Modified Z-score formula: 0.6745 * (x - median) / MAD
            outlier_indices = []
            for idx, record in enumerate(data):
                value = record.get(column)
                if value is not None and isinstance(value, (int, float)):
                    modified_z = 0.6745 * (value - median_val) / mad
                    if abs(modified_z) > threshold:
                        outlier_indices.append(idx)
            
            if outlier_indices:
                results.append(OutlierDetectionResult(
                    method='modified_z_score',
                    column=column,
                    outliers_detected=len(outlier_indices),
                    outlier_indices=outlier_indices,
                    threshold=threshold,
                    statistics={
                        'median': median_val,
                        'mad': mad
                    }
                ))
        
        return results
    
    def _detect_isolation_forest_outliers(self, data: DataBatch) -> List[OutlierDetectionResult]:
        """Detect outliers using Isolation Forest."""
        results = []
        isolation_config = self.outlier_config.get('isolation_forest', {})
        feature_groups = isolation_config.get('feature_groups', {})
        
        for group_name, group_config in feature_groups.items():
            columns = group_config.get('columns', [])
            contamination = group_config.get('contamination', 0.05)
            
            if not columns:
                continue
            
            # Prepare feature matrix
            feature_matrix = self._prepare_feature_matrix(data, columns)
            
            if feature_matrix.shape[0] < 10:  # Need sufficient data points
                continue
            
            # Apply Isolation Forest
            isolation_forest = IsolationForest(
                contamination=contamination,
                random_state=42
            )
            
            outlier_labels = isolation_forest.fit_predict(feature_matrix)
            
            # Find outlier indices
            outlier_indices = [i for i, label in enumerate(outlier_labels) if label == -1]
            
            if outlier_indices:
                results.append(OutlierDetectionResult(
                    method='isolation_forest',
                    column=f"multivariate_{group_name}",
                    outliers_detected=len(outlier_indices),
                    outlier_indices=outlier_indices,
                    threshold=contamination,
                    statistics={
                        'features': columns,
                        'contamination': contamination,
                        'total_samples': feature_matrix.shape[0]
                    }
                ))
        
        return results
    
    def _perform_shapiro_wilk_test(self, values: List[float], column: str) -> Optional[DistributionTestResult]:
        """Perform Shapiro-Wilk normality test."""
        shapiro_config = self.distribution_config.get('normality_tests', {}).get('shapiro_wilk', {})
        
        if not shapiro_config.get('enabled', True):
            return None
        
        if len(values) > 5000:  # Shapiro-Wilk limited to 5000 samples
            return None
        
        try:
            statistic, p_value = stats.shapiro(values)
            alpha = shapiro_config.get('alpha', 0.05)
            
            return DistributionTestResult(
                column=column,
                test_name='shapiro_wilk',
                statistic=statistic,
                p_value=p_value,
                is_normal=p_value > alpha,
                confidence_level=alpha
            )
        except Exception as e:
            self.logger.warning(f"Shapiro-Wilk test failed for column {column}: {e}")
            return None
    
    def _perform_kolmogorov_smirnov_test(self, values: List[float], column: str) -> Optional[DistributionTestResult]:
        """Perform Kolmogorov-Smirnov normality test."""
        ks_config = self.distribution_config.get('normality_tests', {}).get('kolmogorov_smirnov', {})
        
        if not ks_config.get('enabled', True):
            return None
        
        try:
            # Test against standard normal distribution
            standardised_values = stats.zscore(values)
            statistic, p_value = stats.kstest(standardised_values, 'norm')
            alpha = ks_config.get('alpha', 0.05)
            
            return DistributionTestResult(
                column=column,
                test_name='kolmogorov_smirnov',
                statistic=statistic,
                p_value=p_value,
                is_normal=p_value > alpha,
                confidence_level=alpha
            )
        except Exception as e:
            self.logger.warning(f"Kolmogorov-Smirnov test failed for column {column}: {e}")
            return None
    
    def _perform_anderson_darling_test(self, values: List[float], column: str) -> Optional[DistributionTestResult]:
        """Perform Anderson-Darling normality test."""
        ad_config = self.distribution_config.get('normality_tests', {}).get('anderson_darling', {})
        
        if not ad_config.get('enabled', True):
            return None
        
        try:
            result = stats.anderson(values, dist='norm')
            statistic = result.statistic
            
            # Use 5% significance level (index 2 in critical values)
            critical_value = result.critical_values[2]
            alpha = ad_config.get('alpha', 0.05)
            
            return DistributionTestResult(
                column=column,
                test_name='anderson_darling',
                statistic=statistic,
                p_value=0.05 if statistic > critical_value else 0.1,  # Approximation
                is_normal=statistic <= critical_value,
                confidence_level=alpha
            )
        except Exception as e:
            self.logger.warning(f"Anderson-Darling test failed for column {column}: {e}")
            return None
    
    def _perform_t_tests(self, data: DataBatch, comparisons: List[Dict[str, Any]]) -> List[ValidationResult]:
        """Perform t-tests for group comparisons."""
        results = []
        
        for comparison in comparisons:
            name = comparison.get('name', 'unnamed_comparison')
            grouping_column = comparison.get('grouping_column')
            test_column = comparison.get('test_column')
            groups = comparison.get('groups', [])
            
            if not all([grouping_column, test_column]) or len(groups) != 2:
                continue
            
            # Extract data for each group
            group1_values = []
            group2_values = []
            
            for record in data:
                group_value = record.get(grouping_column)
                test_value = record.get(test_column)
                
                if test_value is not None and isinstance(test_value, (int, float)):
                    if str(group_value) == str(groups[0]):
                        group1_values.append(test_value)
                    elif str(group_value) == str(groups[1]):
                        group2_values.append(test_value)
            
            if len(group1_values) < 3 or len(group2_values) < 3:
                continue
            
            try:
                statistic, p_value = stats.ttest_ind(group1_values, group2_values)
                alpha = 0.05
                
                is_significant = p_value < alpha
                
                if is_significant:
                    results.append(ValidationResult(
                        is_valid=True,  # Significant differences are informational
                        severity=ValidationSeverity.INFO,
                        rule_id=f"t_test_{name}",
                        message=f"Significant difference found between {groups[0]} and {groups[1]} for {test_column}",
                        details={
                            'test_type': 't_test',
                            'groups': groups,
                            'test_column': test_column,
                            'statistic': statistic,
                            'p_value': p_value,
                            'group1_mean': np.mean(group1_values),
                            'group2_mean': np.mean(group2_values),
                            'group1_size': len(group1_values),
                            'group2_size': len(group2_values)
                        }
                    ))
            except Exception as e:
                self.logger.warning(f"T-test failed for comparison {name}: {e}")
        
        return results
    
    def _perform_chi_square_tests(self, data: DataBatch, contingency_tables: List[Dict[str, Any]]) -> List[ValidationResult]:
        """Perform chi-square tests for categorical associations."""
        results = []
        
        for table_config in contingency_tables:
            name = table_config.get('name', 'unnamed_chi_square')
            row_variable = table_config.get('row_variable')
            column_variable = table_config.get('column_variable')
            
            if not all([row_variable, column_variable]):
                continue
            
            # Build contingency table
            contingency_data = defaultdict(lambda: defaultdict(int))
            
            for record in data:
                row_value = record.get(row_variable)
                col_value = record.get(column_variable)
                
                if row_value is not None and col_value is not None:
                    contingency_data[str(row_value)][str(col_value)] += 1
            
            if len(contingency_data) < 2:
                continue
            
            # Convert to matrix for chi-square test
            row_labels = list(contingency_data.keys())
            col_labels = list(set(col for row_data in contingency_data.values() for col in row_data.keys()))
            
            matrix = []
            for row_label in row_labels:
                row = [contingency_data[row_label].get(col_label, 0) for col_label in col_labels]
                matrix.append(row)
            
            try:
                chi2, p_value, dof, expected = stats.chi2_contingency(matrix)
                alpha = 0.05
                
                is_significant = p_value < alpha
                
                if is_significant:
                    results.append(ValidationResult(
                        is_valid=True,  # Associations are informational
                        severity=ValidationSeverity.INFO,
                        rule_id=f"chi_square_{name}",
                        message=f"Significant association found between {row_variable} and {column_variable}",
                        details={
                            'test_type': 'chi_square',
                            'row_variable': row_variable,
                            'column_variable': column_variable,
                            'chi2_statistic': chi2,
                            'p_value': p_value,
                            'degrees_of_freedom': dof
                        }
                    ))
            except Exception as e:
                self.logger.warning(f"Chi-square test failed for {name}: {e}")
        
        return results
    
    def _has_temporal_data(self, data: DataBatch) -> bool:
        """Check if data contains temporal information."""
        temporal_columns = ['data_year', 'year', 'date', 'time_period']
        
        for record in data[:10]:  # Check first 10 records
            for col in temporal_columns:
                if col in record and record[col] is not None:
                    return True
        
        return False
    
    def _group_data_by_entity(self, data: DataBatch, entity_column: str) -> Dict[str, List[DataRecord]]:
        """Group data by entity for time series analysis."""
        groups = defaultdict(list)
        
        for record in data:
            entity_id = record.get(entity_column)
            if entity_id:
                groups[str(entity_id)].append(record)
        
        return dict(groups)
    
    def _analyze_entity_trend(self, entity_records: List[DataRecord], column: str) -> Optional[TrendAnalysisResult]:
        """Analyze trend for a single entity."""
        # Extract time series data
        time_points = []
        values = []
        
        for record in entity_records:
            time_value = record.get('data_year') or record.get('year')
            column_value = record.get(column)
            
            if time_value is not None and column_value is not None:
                try:
                    time_points.append(float(time_value))
                    values.append(float(column_value))
                except (ValueError, TypeError):
                    continue
        
        if len(time_points) < 3:
            return None
        
        # Sort by time
        sorted_data = sorted(zip(time_points, values))
        time_points, values = zip(*sorted_data)
        
        # Perform linear regression for trend analysis
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_points, values)
            
            # Determine trend direction
            if p_value < 0.05:  # Significant trend
                if slope > 0:
                    trend_direction = 'increasing'
                else:
                    trend_direction = 'decreasing'
            else:
                trend_direction = 'stable'
            
            return TrendAnalysisResult(
                column=column,
                trend_direction=trend_direction,
                slope=slope,
                r_squared=r_value**2,
                p_value=p_value
            )
            
        except Exception as e:
            self.logger.warning(f"Trend analysis failed for column {column}: {e}")
            return None
    
    def _find_optimal_clusters(self, data: np.ndarray, k_range: List[int]) -> int:
        """Find optimal number of clusters using elbow method."""
        inertias = []
        
        for k in range(k_range[0], k_range[1] + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
        
        # Simple elbow detection - find point with maximum decrease
        if len(inertias) < 2:
            return k_range[0]
        
        decreases = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        optimal_k_idx = np.argmax(decreases)
        
        return k_range[0] + optimal_k_idx
    
    def _prepare_feature_matrix(self, data: DataBatch, columns: List[str]) -> np.ndarray:
        """Prepare feature matrix for multivariate analysis."""
        feature_matrix = []
        
        for record in data:
            row = []
            valid_row = True
            
            for column in columns:
                value = record.get(column)
                if value is not None and isinstance(value, (int, float)):
                    row.append(float(value))
                else:
                    valid_row = False
                    break
            
            if valid_row:
                feature_matrix.append(row)
        
        return np.array(feature_matrix)
    
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
            if value is not None and isinstance(value, (int, float)):
                values.append(float(value))
        
        return values
    
    def _convert_outlier_results_to_validation(self, outlier_results: List[OutlierDetectionResult]) -> List[ValidationResult]:
        """Convert outlier detection results to validation results."""
        validation_results = []
        
        for outlier_result in outlier_results:
            if outlier_result.outliers_detected > 0:
                validation_results.append(ValidationResult(
                    is_valid=True,  # Outliers are informational
                    severity=ValidationSeverity.INFO,
                    rule_id=f"outlier_detection_{outlier_result.method}",
                    message=f"Detected {outlier_result.outliers_detected} outliers in {outlier_result.column} using {outlier_result.method}",
                    details={
                        'method': outlier_result.method,
                        'column': outlier_result.column,
                        'threshold': outlier_result.threshold,
                        'statistics': outlier_result.statistics
                    },
                    affected_records=outlier_result.outlier_indices[:10]  # Limit for performance
                ))
        
        return validation_results
    
    def _convert_distribution_results_to_validation(self, distribution_results: List[DistributionTestResult]) -> List[ValidationResult]:
        """Convert distribution test results to validation results."""
        validation_results = []
        
        for dist_result in distribution_results:
            if not dist_result.is_normal and dist_result.p_value < 0.01:  # Strong evidence of non-normality
                validation_results.append(ValidationResult(
                    is_valid=True,  # Non-normality is informational
                    severity=ValidationSeverity.INFO,
                    rule_id=f"distribution_test_{dist_result.test_name}",
                    message=f"Column {dist_result.column} shows significant deviation from normal distribution ({dist_result.test_name})",
                    details={
                        'test_name': dist_result.test_name,
                        'column': dist_result.column,
                        'statistic': dist_result.statistic,
                        'p_value': dist_result.p_value,
                        'is_normal': dist_result.is_normal
                    }
                ))
        
        return validation_results
    
    def _convert_correlation_results_to_validation(self, correlation_results: List[CorrelationAnalysisResult]) -> List[ValidationResult]:
        """Convert correlation analysis results to validation results."""
        validation_results = []
        
        for corr_result in correlation_results:
            if corr_result.expected_correlation is not None and corr_result.deviation is not None:
                if corr_result.deviation > 0.2:  # Significant deviation from expected
                    severity = ValidationSeverity.WARNING if corr_result.deviation > 0.4 else ValidationSeverity.INFO
                    
                    validation_results.append(ValidationResult(
                        is_valid=corr_result.deviation <= 0.4,
                        severity=severity,
                        rule_id="correlation_analysis",
                        message=f"Correlation between {corr_result.variable1} and {corr_result.variable2} deviates from expected",
                        details={
                            'variable1': corr_result.variable1,
                            'variable2': corr_result.variable2,
                            'observed_correlation': corr_result.correlation_coefficient,
                            'expected_correlation': corr_result.expected_correlation,
                            'deviation': corr_result.deviation,
                            'p_value': corr_result.p_value,
                            'is_significant': corr_result.is_significant
                        }
                    ))
        
        return validation_results
    
    def _convert_trend_results_to_validation(self, trend_results: List[TrendAnalysisResult]) -> List[ValidationResult]:
        """Convert trend analysis results to validation results."""
        validation_results = []
        
        for trend_result in trend_results:
            if trend_result.p_value < 0.05 and abs(trend_result.slope) > 1000:  # Significant and large trend
                validation_results.append(ValidationResult(
                    is_valid=True,  # Trends are informational
                    severity=ValidationSeverity.INFO,
                    rule_id="trend_analysis",
                    message=f"Significant trend detected in {trend_result.column}: {trend_result.trend_direction}",
                    details={
                        'column': trend_result.column,
                        'trend_direction': trend_result.trend_direction,
                        'slope': trend_result.slope,
                        'r_squared': trend_result.r_squared,
                        'p_value': trend_result.p_value
                    }
                ))
        
        return validation_results
    
    def _update_validation_statistics(self, results: List[ValidationResult], record_count: int) -> None:
        """Update statistical validation statistics."""
        self._validation_statistics['total_validations'] += 1
        self._validation_statistics['total_records_validated'] += record_count
        
        for result in results:
            if result.severity == ValidationSeverity.ERROR:
                self._validation_statistics['statistical_errors'] += 1
            elif result.severity == ValidationSeverity.WARNING:
                self._validation_statistics['statistical_warnings'] += 1
            else:
                self._validation_statistics['statistical_info'] += 1
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get statistical validation statistics."""
        return dict(self._validation_statistics)