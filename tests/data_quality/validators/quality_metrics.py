"""
Data Quality Metrics

Comprehensive data quality metrics calculation and monitoring for Australian health data.
Implements industry-standard data quality dimensions and Australian-specific validation rules.

Quality Dimensions Covered:
- Completeness: Percentage of non-null values
- Accuracy: Conformance to expected patterns and formats
- Consistency: Internal logical consistency and cross-dataset alignment
- Validity: Adherence to Australian health data standards
- Timeliness: Data freshness and update frequency
- Uniqueness: Absence of duplicate records

Australian Health Data Specific Metrics:
- SA2 code validity and coverage
- SEIFA compliance with 2021 methodology
- PBS prescription data quality
- Geographic coordinate accuracy
- Population data consistency
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import statistics

import polars as pl
import numpy as np
from loguru import logger

from .australian_health_validators import AustralianHealthDataValidator


class QualityDimension(Enum):
    """Data quality dimensions."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    TIMELINESS = "timeliness"
    UNIQUENESS = "uniqueness"


class QualityThreshold(Enum):
    """Quality threshold levels."""
    EXCELLENT = 95.0
    GOOD = 90.0
    ACCEPTABLE = 80.0
    POOR = 70.0
    UNACCEPTABLE = 0.0


@dataclass
class QualityMetric:
    """Data quality metric result."""
    dimension: str
    metric_name: str
    value: float
    threshold: float
    passed: bool
    details: Dict[str, Any]
    timestamp: str


@dataclass
class QualityReport:
    """Comprehensive data quality report."""
    dataset_name: str
    layer: str
    timestamp: str
    overall_score: float
    quality_classification: str
    metrics: List[QualityMetric]
    recommendations: List[str]
    metadata: Dict[str, Any]


class AustralianHealthQualityMetrics:
    """Australian health data quality metrics calculator."""
    
    def __init__(self, validator: Optional[AustralianHealthDataValidator] = None):
        """
        Initialize quality metrics calculator.
        
        Args:
            validator: Australian health data validator instance
        """
        self.validator = validator or AustralianHealthDataValidator()
        self.logger = logger.bind(component="quality_metrics")
        
        # Quality thresholds for different data types
        self.thresholds = {
            "completeness": {
                "critical_fields": 95.0,  # SA2 codes, population
                "standard_fields": 90.0,  # Most SEIFA fields
                "optional_fields": 80.0   # Supplementary data
            },
            "validity": {
                "sa2_codes": 99.0,
                "seifa_scores": 95.0,
                "coordinates": 98.0,
                "atc_codes": 97.0
            },
            "consistency": {
                "cross_dataset_sa2": 95.0,
                "seifa_score_decile": 90.0,
                "population_estimates": 85.0
            },
            "uniqueness": {
                "sa2_codes": 100.0,
                "identifiers": 100.0
            }
        }
    
    def calculate_completeness_metrics(self, df: pl.DataFrame, critical_fields: Optional[List[str]] = None) -> List[QualityMetric]:
        """
        Calculate data completeness metrics.
        
        Args:
            df: DataFrame to analyze
            critical_fields: List of critical field names
            
        Returns:
            List of completeness metrics
        """
        metrics = []
        critical_fields = critical_fields or ["sa2_code_2021", "usual_resident_population"]
        
        # Overall completeness
        total_cells = df.width * df.height
        null_cells = sum(df[col].null_count() for col in df.columns)
        overall_completeness = ((total_cells - null_cells) / total_cells) * 100
        
        metrics.append(QualityMetric(
            dimension=QualityDimension.COMPLETENESS.value,
            metric_name="overall_completeness",
            value=overall_completeness,
            threshold=self.thresholds["completeness"]["standard_fields"],
            passed=overall_completeness >= self.thresholds["completeness"]["standard_fields"],
            details={
                "total_cells": total_cells,
                "null_cells": null_cells,
                "non_null_percentage": overall_completeness
            },
            timestamp=datetime.now().isoformat()
        ))
        
        # Critical fields completeness
        for field in critical_fields:
            if field in df.columns:
                null_count = df[field].null_count()
                completeness = ((df.height - null_count) / df.height) * 100
                
                metrics.append(QualityMetric(
                    dimension=QualityDimension.COMPLETENESS.value,
                    metric_name=f"{field}_completeness",
                    value=completeness,
                    threshold=self.thresholds["completeness"]["critical_fields"],
                    passed=completeness >= self.thresholds["completeness"]["critical_fields"],
                    details={
                        "field": field,
                        "null_count": null_count,
                        "total_records": df.height,
                        "completeness_percentage": completeness
                    },
                    timestamp=datetime.now().isoformat()
                ))
        
        # Per-column completeness analysis
        column_completeness = {}
        for col in df.columns:
            null_count = df[col].null_count()
            completeness = ((df.height - null_count) / df.height) * 100
            column_completeness[col] = {
                "completeness": completeness,
                "null_count": null_count,
                "passes_threshold": completeness >= self.thresholds["completeness"]["standard_fields"]
            }
        
        metrics.append(QualityMetric(
            dimension=QualityDimension.COMPLETENESS.value,
            metric_name="column_completeness_analysis",
            value=sum(info["completeness"] for info in column_completeness.values()) / len(column_completeness),
            threshold=self.thresholds["completeness"]["standard_fields"],
            passed=all(info["passes_threshold"] for info in column_completeness.values()),
            details=column_completeness,
            timestamp=datetime.now().isoformat()
        ))
        
        return metrics
    
    def calculate_validity_metrics(self, df: pl.DataFrame, data_type: str = "seifa") -> List[QualityMetric]:
        """
        Calculate data validity metrics based on Australian standards.
        
        Args:
            df: DataFrame to analyze
            data_type: Type of data (seifa, health, census, geographic)
            
        Returns:
            List of validity metrics
        """
        metrics = []
        
        # SA2 code validity (common across all data types)
        if "sa2_code_2021" in df.columns:
            sa2_codes = df["sa2_code_2021"].drop_nulls().to_list()
            valid_sa2_count = 0
            validation_details = {"valid_codes": [], "invalid_codes": []}
            
            for code in sa2_codes:
                validation = self.validator.validate_sa2_code(code)
                if validation["valid"]:
                    valid_sa2_count += 1
                    validation_details["valid_codes"].append(code)
                else:
                    validation_details["invalid_codes"].append({
                        "code": code,
                        "errors": validation["errors"]
                    })
            
            sa2_validity = (valid_sa2_count / len(sa2_codes)) * 100 if sa2_codes else 100
            
            metrics.append(QualityMetric(
                dimension=QualityDimension.VALIDITY.value,
                metric_name="sa2_code_validity",
                value=sa2_validity,
                threshold=self.thresholds["validity"]["sa2_codes"],
                passed=sa2_validity >= self.thresholds["validity"]["sa2_codes"],
                details={
                    "total_codes": len(sa2_codes),
                    "valid_codes": valid_sa2_count,
                    "invalid_codes": len(sa2_codes) - valid_sa2_count,
                    "validity_percentage": sa2_validity,
                    "sample_invalid": validation_details["invalid_codes"][:5]  # Show first 5 invalid
                },
                timestamp=datetime.now().isoformat()
            ))
        
        # SEIFA-specific validity
        if data_type == "seifa":
            seifa_metrics = self._calculate_seifa_validity_metrics(df)
            metrics.extend(seifa_metrics)
        
        # Geographic data validity
        if "latitude" in df.columns and "longitude" in df.columns:
            coord_metrics = self._calculate_coordinate_validity_metrics(df)
            metrics.extend(coord_metrics)
        
        # Health data validity (ATC codes, etc.)
        if data_type == "health" and "atc_code" in df.columns:
            atc_metrics = self._calculate_atc_validity_metrics(df)
            metrics.extend(atc_metrics)
        
        return metrics
    
    def calculate_consistency_metrics(self, df: pl.DataFrame, reference_data: Optional[Dict[str, pl.DataFrame]] = None) -> List[QualityMetric]:
        """
        Calculate data consistency metrics.
        
        Args:
            df: Primary DataFrame to analyze
            reference_data: Dictionary of reference DataFrames for cross-dataset consistency
            
        Returns:
            List of consistency metrics
        """
        metrics = []
        
        # Internal consistency - SEIFA score-decile relationships
        if all(col in df.columns for col in ["irsd_score", "irsd_decile"]):
            seifa_consistency = self._calculate_seifa_consistency_metrics(df)
            metrics.extend(seifa_consistency)
        
        # Cross-dataset SA2 consistency
        if reference_data and "sa2_code_2021" in df.columns:
            cross_dataset_metrics = self._calculate_cross_dataset_consistency(df, reference_data)
            metrics.extend(cross_dataset_metrics)
        
        # Population data consistency (if multiple population estimates)
        population_cols = [col for col in df.columns if "population" in col.lower()]
        if len(population_cols) > 1:
            pop_consistency = self._calculate_population_consistency_metrics(df, population_cols)
            metrics.extend(pop_consistency)
        
        # Temporal consistency (if time-series data)
        if "year_month" in df.columns or "date" in df.columns:
            temporal_metrics = self._calculate_temporal_consistency_metrics(df)
            metrics.extend(temporal_metrics)
        
        return metrics
    
    def calculate_uniqueness_metrics(self, df: pl.DataFrame) -> List[QualityMetric]:
        """
        Calculate data uniqueness metrics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of uniqueness metrics
        """
        metrics = []
        
        # SA2 code uniqueness
        if "sa2_code_2021" in df.columns:
            sa2_codes = df["sa2_code_2021"].drop_nulls()
            unique_count = sa2_codes.n_unique()
            total_count = len(sa2_codes)
            uniqueness = (unique_count / total_count) * 100 if total_count > 0 else 100
            
            metrics.append(QualityMetric(
                dimension=QualityDimension.UNIQUENESS.value,
                metric_name="sa2_code_uniqueness",
                value=uniqueness,
                threshold=self.thresholds["uniqueness"]["sa2_codes"],
                passed=uniqueness >= self.thresholds["uniqueness"]["sa2_codes"],
                details={
                    "total_records": total_count,
                    "unique_records": unique_count,
                    "duplicate_records": total_count - unique_count,
                    "uniqueness_percentage": uniqueness
                },
                timestamp=datetime.now().isoformat()
            ))
        
        # Overall record uniqueness (all columns)
        total_records = df.height
        unique_records = df.unique().height
        overall_uniqueness = (unique_records / total_records) * 100 if total_records > 0 else 100
        
        metrics.append(QualityMetric(
            dimension=QualityDimension.UNIQUENESS.value,
            metric_name="overall_record_uniqueness",
            value=overall_uniqueness,
            threshold=95.0,  # Most records should be unique
            passed=overall_uniqueness >= 95.0,
            details={
                "total_records": total_records,
                "unique_records": unique_records,
                "duplicate_records": total_records - unique_records,
                "uniqueness_percentage": overall_uniqueness
            },
            timestamp=datetime.now().isoformat()
        ))
        
        return metrics
    
    def calculate_timeliness_metrics(self, df: pl.DataFrame, date_column: str, expected_frequency: str = "monthly") -> List[QualityMetric]:
        """
        Calculate data timeliness metrics.
        
        Args:
            df: DataFrame to analyze
            date_column: Name of the date column
            expected_frequency: Expected update frequency (daily, weekly, monthly, yearly)
            
        Returns:
            List of timeliness metrics
        """
        metrics = []
        
        if date_column not in df.columns:
            return metrics
        
        dates = df[date_column].drop_nulls()
        if len(dates) == 0:
            return metrics
        
        # Data freshness
        latest_date = dates.max()
        today = datetime.now().date()
        
        # Handle different date formats
        if isinstance(latest_date, str):
            try:
                latest_date = datetime.strptime(latest_date, "%Y-%m-%d").date()
            except ValueError:
                try:
                    latest_date = datetime.strptime(latest_date, "%Y-%m-%d %H:%M:%S").date()
                except ValueError:
                    return metrics
        
        days_since_latest = (today - latest_date).days
        
        # Calculate expected update interval
        frequency_days = {
            "daily": 1,
            "weekly": 7,
            "monthly": 30,
            "quarterly": 90,
            "yearly": 365
        }
        
        expected_interval = frequency_days.get(expected_frequency, 30)
        freshness_score = max(0, 100 - (days_since_latest / expected_interval) * 25)
        
        metrics.append(QualityMetric(
            dimension=QualityDimension.TIMELINESS.value,
            metric_name="data_freshness",
            value=freshness_score,
            threshold=80.0,
            passed=freshness_score >= 80.0,
            details={
                "latest_date": str(latest_date),
                "days_since_latest": days_since_latest,
                "expected_frequency": expected_frequency,
                "freshness_score": freshness_score
            },
            timestamp=datetime.now().isoformat()
        ))
        
        # Data coverage continuity
        earliest_date = dates.min()
        if isinstance(earliest_date, str):
            earliest_date = datetime.strptime(earliest_date, "%Y-%m-%d").date()
        
        coverage_days = (latest_date - earliest_date).days
        expected_records = coverage_days // expected_interval
        actual_records = len(dates)
        coverage_score = min(100, (actual_records / expected_records) * 100) if expected_records > 0 else 100
        
        metrics.append(QualityMetric(
            dimension=QualityDimension.TIMELINESS.value,
            metric_name="data_coverage_continuity",
            value=coverage_score,
            threshold=90.0,
            passed=coverage_score >= 90.0,
            details={
                "earliest_date": str(earliest_date),
                "latest_date": str(latest_date),
                "coverage_days": coverage_days,
                "expected_records": expected_records,
                "actual_records": actual_records,
                "coverage_score": coverage_score
            },
            timestamp=datetime.now().isoformat()
        ))
        
        return metrics
    
    def generate_quality_report(self, 
                              df: pl.DataFrame, 
                              dataset_name: str, 
                              layer: str = "bronze",
                              data_type: str = "seifa",
                              reference_data: Optional[Dict[str, pl.DataFrame]] = None) -> QualityReport:
        """
        Generate comprehensive quality report.
        
        Args:
            df: DataFrame to analyze
            dataset_name: Name of the dataset
            layer: Data layer (bronze, silver, gold)
            data_type: Type of data
            reference_data: Reference data for cross-dataset validation
            
        Returns:
            Comprehensive quality report
        """
        # Calculate all metrics
        all_metrics = []
        
        # Completeness metrics
        completeness_metrics = self.calculate_completeness_metrics(df)
        all_metrics.extend(completeness_metrics)
        
        # Validity metrics
        validity_metrics = self.calculate_validity_metrics(df, data_type)
        all_metrics.extend(validity_metrics)
        
        # Consistency metrics
        consistency_metrics = self.calculate_consistency_metrics(df, reference_data)
        all_metrics.extend(consistency_metrics)
        
        # Uniqueness metrics
        uniqueness_metrics = self.calculate_uniqueness_metrics(df)
        all_metrics.extend(uniqueness_metrics)
        
        # Timeliness metrics (if applicable)
        date_columns = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
        if date_columns:
            timeliness_metrics = self.calculate_timeliness_metrics(df, date_columns[0])
            all_metrics.extend(timeliness_metrics)
        
        # Calculate overall score
        if all_metrics:
            overall_score = sum(metric.value for metric in all_metrics) / len(all_metrics)
        else:
            overall_score = 0.0
        
        # Determine quality classification
        quality_classification = self._classify_quality(overall_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_metrics)
        
        # Create metadata
        metadata = {
            "dataset_rows": df.height,
            "dataset_columns": df.width,
            "analysis_timestamp": datetime.now().isoformat(),
            "data_type": data_type,
            "layer": layer,
            "total_metrics": len(all_metrics),
            "passed_metrics": sum(1 for metric in all_metrics if metric.passed),
            "failed_metrics": sum(1 for metric in all_metrics if not metric.passed)
        }
        
        return QualityReport(
            dataset_name=dataset_name,
            layer=layer,
            timestamp=datetime.now().isoformat(),
            overall_score=overall_score,
            quality_classification=quality_classification,
            metrics=all_metrics,
            recommendations=recommendations,
            metadata=metadata
        )
    
    def _calculate_seifa_validity_metrics(self, df: pl.DataFrame) -> List[QualityMetric]:
        """Calculate SEIFA-specific validity metrics."""
        metrics = []
        seifa_indices = ["irsd", "irsad", "ier", "ieo"]
        
        for index in seifa_indices:
            score_col = f"{index}_score"
            decile_col = f"{index}_decile"
            
            # Score validity
            if score_col in df.columns:
                scores = df[score_col].drop_nulls().to_list()
                valid_scores = sum(1 for score in scores 
                                 if self.validator.SEIFA_SCORE_RANGE[0] <= score <= self.validator.SEIFA_SCORE_RANGE[1])
                score_validity = (valid_scores / len(scores)) * 100 if scores else 100
                
                metrics.append(QualityMetric(
                    dimension=QualityDimension.VALIDITY.value,
                    metric_name=f"{score_col}_validity",
                    value=score_validity,
                    threshold=self.thresholds["validity"]["seifa_scores"],
                    passed=score_validity >= self.thresholds["validity"]["seifa_scores"],
                    details={
                        "total_scores": len(scores),
                        "valid_scores": valid_scores,
                        "invalid_scores": len(scores) - valid_scores,
                        "score_range": self.validator.SEIFA_SCORE_RANGE,
                        "validity_percentage": score_validity
                    },
                    timestamp=datetime.now().isoformat()
                ))
            
            # Decile validity
            if decile_col in df.columns:
                deciles = df[decile_col].drop_nulls().to_list()
                valid_deciles = sum(1 for decile in deciles 
                                  if self.validator.SEIFA_DECILE_RANGE[0] <= decile <= self.validator.SEIFA_DECILE_RANGE[1])
                decile_validity = (valid_deciles / len(deciles)) * 100 if deciles else 100
                
                metrics.append(QualityMetric(
                    dimension=QualityDimension.VALIDITY.value,
                    metric_name=f"{decile_col}_validity",
                    value=decile_validity,
                    threshold=self.thresholds["validity"]["seifa_scores"],
                    passed=decile_validity >= self.thresholds["validity"]["seifa_scores"],
                    details={
                        "total_deciles": len(deciles),
                        "valid_deciles": valid_deciles,
                        "invalid_deciles": len(deciles) - valid_deciles,
                        "decile_range": self.validator.SEIFA_DECILE_RANGE,
                        "validity_percentage": decile_validity
                    },
                    timestamp=datetime.now().isoformat()
                ))
        
        return metrics
    
    def _calculate_coordinate_validity_metrics(self, df: pl.DataFrame) -> List[QualityMetric]:
        """Calculate geographic coordinate validity metrics."""
        metrics = []
        
        coords = df.select(["latitude", "longitude"]).drop_nulls()
        valid_coords = 0
        invalid_details = []
        
        for row in coords.iter_rows(named=True):
            lat, lon = row["latitude"], row["longitude"]
            validation = self.validator.validate_australian_coordinates(lat, lon)
            if validation["valid"]:
                valid_coords += 1
            else:
                invalid_details.append({
                    "latitude": lat,
                    "longitude": lon,
                    "errors": validation["errors"]
                })
        
        coord_validity = (valid_coords / len(coords)) * 100 if len(coords) > 0 else 100
        
        metrics.append(QualityMetric(
            dimension=QualityDimension.VALIDITY.value,
            metric_name="coordinate_validity",
            value=coord_validity,
            threshold=self.thresholds["validity"]["coordinates"],
            passed=coord_validity >= self.thresholds["validity"]["coordinates"],
            details={
                "total_coordinates": len(coords),
                "valid_coordinates": valid_coords,
                "invalid_coordinates": len(coords) - valid_coords,
                "australia_bounds": self.validator.AUSTRALIA_BOUNDS,
                "sample_invalid": invalid_details[:5],
                "validity_percentage": coord_validity
            },
            timestamp=datetime.now().isoformat()
        ))
        
        return metrics
    
    def _calculate_atc_validity_metrics(self, df: pl.DataFrame) -> List[QualityMetric]:
        """Calculate ATC code validity metrics."""
        metrics = []
        
        atc_codes = df["atc_code"].drop_nulls().to_list()
        valid_atc_count = 0
        invalid_details = []
        
        for code in atc_codes:
            validation = self.validator.validate_atc_code(code)
            if validation["valid"]:
                valid_atc_count += 1
            else:
                invalid_details.append({
                    "code": code,
                    "errors": validation["errors"]
                })
        
        atc_validity = (valid_atc_count / len(atc_codes)) * 100 if atc_codes else 100
        
        metrics.append(QualityMetric(
            dimension=QualityDimension.VALIDITY.value,
            metric_name="atc_code_validity",
            value=atc_validity,
            threshold=self.thresholds["validity"]["atc_codes"],
            passed=atc_validity >= self.thresholds["validity"]["atc_codes"],
            details={
                "total_codes": len(atc_codes),
                "valid_codes": valid_atc_count,
                "invalid_codes": len(atc_codes) - valid_atc_count,
                "sample_invalid": invalid_details[:5],
                "validity_percentage": atc_validity
            },
            timestamp=datetime.now().isoformat()
        ))
        
        return metrics
    
    def _calculate_seifa_consistency_metrics(self, df: pl.DataFrame) -> List[QualityMetric]:
        """Calculate SEIFA internal consistency metrics."""
        metrics = []
        seifa_indices = ["irsd", "irsad", "ier", "ieo"]
        
        for index in seifa_indices:
            score_col = f"{index}_score"
            decile_col = f"{index}_decile"
            
            if score_col in df.columns and decile_col in df.columns:
                # Calculate correlation between scores and deciles
                scores_deciles = df.select([score_col, decile_col]).drop_nulls()
                if len(scores_deciles) > 1:
                    correlation = scores_deciles.select(
                        pl.corr(score_col, decile_col)
                    ).item()
                    
                    consistency_score = abs(correlation) * 100 if correlation else 0
                    
                    metrics.append(QualityMetric(
                        dimension=QualityDimension.CONSISTENCY.value,
                        metric_name=f"{index}_score_decile_consistency",
                        value=consistency_score,
                        threshold=self.thresholds["consistency"]["seifa_score_decile"],
                        passed=consistency_score >= self.thresholds["consistency"]["seifa_score_decile"],
                        details={
                            "correlation": correlation,
                            "consistency_score": consistency_score,
                            "records_analyzed": len(scores_deciles)
                        },
                        timestamp=datetime.now().isoformat()
                    ))
        
        return metrics
    
    def _calculate_cross_dataset_consistency(self, df: pl.DataFrame, reference_data: Dict[str, pl.DataFrame]) -> List[QualityMetric]:
        """Calculate cross-dataset consistency metrics."""
        metrics = []
        
        if "sa2_code_2021" not in df.columns:
            return metrics
        
        primary_sa2s = set(df["sa2_code_2021"].drop_nulls().to_list())
        
        for ref_name, ref_df in reference_data.items():
            if "sa2_code_2021" in ref_df.columns:
                ref_sa2s = set(ref_df["sa2_code_2021"].drop_nulls().to_list())
                
                # Calculate overlap
                overlap = primary_sa2s & ref_sa2s
                consistency_score = (len(overlap) / len(primary_sa2s)) * 100 if primary_sa2s else 100
                
                metrics.append(QualityMetric(
                    dimension=QualityDimension.CONSISTENCY.value,
                    metric_name=f"sa2_consistency_with_{ref_name}",
                    value=consistency_score,
                    threshold=self.thresholds["consistency"]["cross_dataset_sa2"],
                    passed=consistency_score >= self.thresholds["consistency"]["cross_dataset_sa2"],
                    details={
                        "primary_sa2_count": len(primary_sa2s),
                        "reference_sa2_count": len(ref_sa2s),
                        "overlap_count": len(overlap),
                        "consistency_percentage": consistency_score,
                        "missing_in_reference": list(primary_sa2s - ref_sa2s)[:10]  # Sample
                    },
                    timestamp=datetime.now().isoformat()
                ))
        
        return metrics
    
    def _calculate_population_consistency_metrics(self, df: pl.DataFrame, population_cols: List[str]) -> List[QualityMetric]:
        """Calculate population data consistency metrics."""
        metrics = []
        
        if len(population_cols) < 2:
            return metrics
        
        # Compare population estimates across different sources
        for i in range(len(population_cols)):
            for j in range(i + 1, len(population_cols)):
                col1, col2 = population_cols[i], population_cols[j]
                
                # Get paired data
                paired_data = df.select([col1, col2]).drop_nulls()
                if len(paired_data) == 0:
                    continue
                
                # Calculate percentage differences
                differences = paired_data.with_columns([
                    (pl.col(col1) - pl.col(col2)).abs().alias("abs_diff"),
                    ((pl.col(col1) - pl.col(col2)).abs() / pl.col(col1) * 100).alias("pct_diff")
                ])
                
                avg_pct_diff = differences["pct_diff"].mean()
                consistency_score = max(0, 100 - avg_pct_diff)
                
                metrics.append(QualityMetric(
                    dimension=QualityDimension.CONSISTENCY.value,
                    metric_name=f"population_consistency_{col1}_vs_{col2}",
                    value=consistency_score,
                    threshold=self.thresholds["consistency"]["population_estimates"],
                    passed=consistency_score >= self.thresholds["consistency"]["population_estimates"],
                    details={
                        "column_1": col1,
                        "column_2": col2,
                        "records_compared": len(paired_data),
                        "avg_percentage_difference": avg_pct_diff,
                        "consistency_score": consistency_score
                    },
                    timestamp=datetime.now().isoformat()
                ))
        
        return metrics
    
    def _calculate_temporal_consistency_metrics(self, df: pl.DataFrame) -> List[QualityMetric]:
        """Calculate temporal consistency metrics."""
        metrics = []
        
        # Look for time-based columns
        time_cols = [col for col in df.columns if any(term in col.lower() for term in ["date", "time", "year", "month"])]
        
        if not time_cols:
            return metrics
        
        # Check for temporal ordering
        time_col = time_cols[0]  # Use first time column found
        
        try:
            # Sort by time column and check for anomalies
            sorted_df = df.sort(time_col)
            
            # For now, just check that we can sort - more complex temporal validation could be added
            temporal_score = 100.0  # Default to valid if sorting works
            
            metrics.append(QualityMetric(
                dimension=QualityDimension.CONSISTENCY.value,
                metric_name="temporal_ordering_consistency",
                value=temporal_score,
                threshold=95.0,
                passed=temporal_score >= 95.0,
                details={
                    "time_column": time_col,
                    "records_analyzed": df.height,
                    "temporal_score": temporal_score
                },
                timestamp=datetime.now().isoformat()
            ))
        
        except Exception as e:
            # If sorting fails, temporal consistency is poor
            metrics.append(QualityMetric(
                dimension=QualityDimension.CONSISTENCY.value,
                metric_name="temporal_ordering_consistency",
                value=0.0,
                threshold=95.0,
                passed=False,
                details={
                    "time_column": time_col,
                    "error": str(e),
                    "temporal_score": 0.0
                },
                timestamp=datetime.now().isoformat()
            ))
        
        return metrics
    
    def _classify_quality(self, overall_score: float) -> str:
        """Classify quality based on overall score."""
        if overall_score >= QualityThreshold.EXCELLENT.value:
            return "Excellent"
        elif overall_score >= QualityThreshold.GOOD.value:
            return "Good"
        elif overall_score >= QualityThreshold.ACCEPTABLE.value:
            return "Acceptable"
        elif overall_score >= QualityThreshold.POOR.value:
            return "Poor"
        else:
            return "Unacceptable"
    
    def _generate_recommendations(self, metrics: List[QualityMetric]) -> List[str]:
        """Generate recommendations based on quality metrics."""
        recommendations = []
        failed_metrics = [metric for metric in metrics if not metric.passed]
        
        if not failed_metrics:
            recommendations.append("Data quality is excellent. No immediate action required.")
            return recommendations
        
        # Group failed metrics by dimension
        failed_by_dimension = {}
        for metric in failed_metrics:
            dimension = metric.dimension
            if dimension not in failed_by_dimension:
                failed_by_dimension[dimension] = []
            failed_by_dimension[dimension].append(metric)
        
        # Generate dimension-specific recommendations
        for dimension, failed_metrics_in_dim in failed_by_dimension.items():
            if dimension == QualityDimension.COMPLETENESS.value:
                recommendations.append(
                    f"Address completeness issues: {len(failed_metrics_in_dim)} metrics failed. "
                    "Consider data collection improvements or imputation strategies."
                )
            elif dimension == QualityDimension.VALIDITY.value:
                recommendations.append(
                    f"Fix validity issues: {len(failed_metrics_in_dim)} metrics failed. "
                    "Review data entry processes and validation rules."
                )
            elif dimension == QualityDimension.CONSISTENCY.value:
                recommendations.append(
                    f"Resolve consistency issues: {len(failed_metrics_in_dim)} metrics failed. "
                    "Check data integration and transformation logic."
                )
            elif dimension == QualityDimension.UNIQUENESS.value:
                recommendations.append(
                    f"Handle uniqueness issues: {len(failed_metrics_in_dim)} metrics failed. "
                    "Identify and resolve duplicate records."
                )
            elif dimension == QualityDimension.TIMELINESS.value:
                recommendations.append(
                    f"Improve timeliness: {len(failed_metrics_in_dim)} metrics failed. "
                    "Review data update schedules and processes."
                )
        
        # Add specific recommendations for critical failures
        critical_failures = [metric for metric in failed_metrics if metric.value < 50.0]
        if critical_failures:
            recommendations.append(
                f"CRITICAL: {len(critical_failures)} metrics show severe quality issues. "
                "Immediate investigation and remediation required."
            )
        
        return recommendations
    
    def save_quality_report(self, report: QualityReport, output_path: Path) -> Path:
        """
        Save quality report to file.
        
        Args:
            report: Quality report to save
            output_path: Directory to save report
            
        Returns:
            Path to saved report file
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"quality_report_{report.dataset_name}_{report.layer}_{timestamp}.json"
        
        # Convert to dictionary for JSON serialization
        report_dict = asdict(report)
        
        with open(report_file, "w") as f:
            json.dump(report_dict, f, indent=2)
        
        self.logger.info(f"Quality report saved: {report_file}")
        return report_file


if __name__ == "__main__":
    # Example usage
    validator = AustralianHealthDataValidator()
    quality_calculator = AustralianHealthQualityMetrics(validator)
    
    # Create sample SEIFA data
    sample_df = pl.DataFrame({
        "sa2_code_2021": ["101021007", "201011001", "301011002"],
        "sa2_name_2021": ["Sydney - Harbour", "Melbourne - CBD", "Brisbane - CBD"],
        "irsd_score": [1050, 950, 1100],
        "irsd_decile": [8, 5, 9],
        "irsad_score": [1080, 920, 1120],
        "irsad_decile": [7, 4, 8],
        "usual_resident_population": [15000, 12000, 18000],
    })
    
    # Generate quality report
    quality_report = quality_calculator.generate_quality_report(
        df=sample_df,
        dataset_name="sample_seifa",
        layer="bronze",
        data_type="seifa"
    )
    
    print(f"Overall Quality Score: {quality_report.overall_score:.2f}")
    print(f"Quality Classification: {quality_report.quality_classification}")
    print(f"Number of Metrics: {len(quality_report.metrics)}")
    print(f"Passed Metrics: {quality_report.metadata['passed_metrics']}")
    print(f"Failed Metrics: {quality_report.metadata['failed_metrics']}")
    
    # Show recommendations
    for rec in quality_report.recommendations:
        print(f"Recommendation: {rec}")