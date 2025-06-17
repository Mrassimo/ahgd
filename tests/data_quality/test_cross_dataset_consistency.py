"""
Cross-Dataset Consistency Validation Tests

Comprehensive testing suite for cross-dataset consistency validation including:
- SA2 code alignment across SEIFA, health, and geographic datasets
- Population estimates consistency between Census and SEIFA data
- Geographic boundary alignment validation
- Health service provider coverage consistency
- Temporal consistency across time-series datasets
- Data lineage integrity across Bronze-Silver-Gold layers

This test suite ensures data consistency across all datasets in the
Australian health analytics platform and validates referential integrity.
"""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from unittest.mock import Mock, patch
from dataclasses import dataclass
from enum import Enum

import polars as pl
import numpy as np
from loguru import logger

from tests.data_quality.validators.australian_health_validators import AustralianHealthDataValidator
from tests.data_quality.validators.quality_metrics import AustralianHealthQualityMetrics


class ConsistencyCheckType(Enum):
    """Types of consistency checks."""
    SA2_ALIGNMENT = "sa2_alignment"
    POPULATION_CONSISTENCY = "population_consistency"
    GEOGRAPHIC_ALIGNMENT = "geographic_alignment"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    SCHEMA_CONSISTENCY = "schema_consistency"


@dataclass
class ConsistencyViolation:
    """Cross-dataset consistency violation."""
    check_type: ConsistencyCheckType
    severity: str
    datasets_involved: List[str]
    violation_count: int
    total_records: int
    violation_percentage: float
    description: str
    sample_violations: List[Dict]
    details: Dict


class CrossDatasetConsistencyValidator:
    """Validator for cross-dataset consistency checks."""
    
    def __init__(self, validator: AustralianHealthDataValidator):
        """Initialize with Australian health data validator."""
        self.validator = validator
        self.logger = logger.bind(component="cross_dataset_consistency")
    
    def validate_sa2_alignment(self, datasets: Dict[str, pl.DataFrame]) -> List[ConsistencyViolation]:
        """
        Validate SA2 code alignment across multiple datasets.
        
        Args:
            datasets: Dictionary of dataset_name -> DataFrame
            
        Returns:
            List of consistency violations
        """
        violations = []
        
        # Extract SA2 codes from each dataset
        sa2_sets = {}
        for dataset_name, df in datasets.items():
            if "sa2_code_2021" in df.columns:
                sa2_codes = set(df["sa2_code_2021"].drop_nulls().to_list())
                sa2_sets[dataset_name] = sa2_codes
        
        if len(sa2_sets) < 2:
            return violations
        
        # Find common and unique SA2 codes
        all_sa2_codes = set()
        for sa2_set in sa2_sets.values():
            all_sa2_codes.update(sa2_set)
        
        # Check pairwise consistency
        dataset_names = list(sa2_sets.keys())
        for i in range(len(dataset_names)):
            for j in range(i + 1, len(dataset_names)):
                dataset1 = dataset_names[i]
                dataset2 = dataset_names[j]
                
                sa2_set1 = sa2_sets[dataset1]
                sa2_set2 = sa2_sets[dataset2]
                
                # Calculate overlap
                intersection = sa2_set1 & sa2_set2
                union = sa2_set1 | sa2_set2
                
                overlap_percentage = (len(intersection) / len(union)) * 100 if union else 100
                
                # Find mismatches
                only_in_1 = sa2_set1 - sa2_set2
                only_in_2 = sa2_set2 - sa2_set1
                
                if overlap_percentage < 90.0:  # Threshold for acceptable overlap
                    violations.append(ConsistencyViolation(
                        check_type=ConsistencyCheckType.SA2_ALIGNMENT,
                        severity="error",
                        datasets_involved=[dataset1, dataset2],
                        violation_count=len(only_in_1) + len(only_in_2),
                        total_records=len(union),
                        violation_percentage=100 - overlap_percentage,
                        description=f"SA2 code alignment between {dataset1} and {dataset2} below threshold",
                        sample_violations=[
                            {"only_in_" + dataset1: list(only_in_1)[:5]},
                            {"only_in_" + dataset2: list(only_in_2)[:5]}
                        ],
                        details={
                            "overlap_percentage": overlap_percentage,
                            "intersection_count": len(intersection),
                            "union_count": len(union),
                            "dataset1_count": len(sa2_set1),
                            "dataset2_count": len(sa2_set2)
                        }
                    ))
        
        return violations
    
    def validate_population_consistency(self, 
                                      census_data: pl.DataFrame, 
                                      seifa_data: pl.DataFrame,
                                      tolerance_percent: float = 5.0) -> List[ConsistencyViolation]:
        """
        Validate population consistency between Census and SEIFA datasets.
        
        Args:
            census_data: Census data with population figures
            seifa_data: SEIFA data with population figures
            tolerance_percent: Acceptable percentage difference
            
        Returns:
            List of consistency violations
        """
        violations = []
        
        # Check if both datasets have population data
        census_pop_col = None
        seifa_pop_col = None
        
        # Find population columns
        for col in census_data.columns:
            if "population" in col.lower() or "tot_p_p" in col.lower():
                census_pop_col = col
                break
        
        for col in seifa_data.columns:
            if "population" in col.lower():
                seifa_pop_col = col
                break
        
        if not census_pop_col or not seifa_pop_col:
            return violations
        
        # Join datasets on SA2 code
        if "sa2_code_2021" not in census_data.columns or "sa2_code_2021" not in seifa_data.columns:
            return violations
        
        joined_data = census_data.select(["sa2_code_2021", census_pop_col]).join(
            seifa_data.select(["sa2_code_2021", seifa_pop_col]),
            on="sa2_code_2021",
            how="inner"
        ).drop_nulls()
        
        if len(joined_data) == 0:
            return violations
        
        # Calculate population differences
        inconsistencies = []
        for row in joined_data.iter_rows(named=True):
            sa2_code = row["sa2_code_2021"]
            census_pop = row[census_pop_col]
            seifa_pop = row[seifa_pop_col]
            
            if census_pop > 0:
                difference_percent = abs(census_pop - seifa_pop) / census_pop * 100
                
                if difference_percent > tolerance_percent:
                    inconsistencies.append({
                        "sa2_code": sa2_code,
                        "census_population": census_pop,
                        "seifa_population": seifa_pop,
                        "difference_percent": difference_percent
                    })
        
        if inconsistencies:
            violation_percentage = (len(inconsistencies) / len(joined_data)) * 100
            
            violations.append(ConsistencyViolation(
                check_type=ConsistencyCheckType.POPULATION_CONSISTENCY,
                severity="warning" if violation_percentage < 10 else "error",
                datasets_involved=["census", "seifa"],
                violation_count=len(inconsistencies),
                total_records=len(joined_data),
                violation_percentage=violation_percentage,
                description=f"Population inconsistency between Census and SEIFA data",
                sample_violations=inconsistencies[:5],
                details={
                    "tolerance_percent": tolerance_percent,
                    "census_column": census_pop_col,
                    "seifa_column": seifa_pop_col,
                    "records_compared": len(joined_data)
                }
            ))
        
        return violations
    
    def validate_geographic_alignment(self, 
                                    geographic_data: pl.DataFrame, 
                                    health_data: pl.DataFrame) -> List[ConsistencyViolation]:
        """
        Validate geographic boundary alignment between datasets.
        
        Args:
            geographic_data: Geographic boundaries data
            health_data: Health data with geographic references
            
        Returns:
            List of consistency violations
        """
        violations = []
        
        # Check coordinate consistency if both datasets have coordinates
        if all(col in geographic_data.columns for col in ["latitude", "longitude"]) and \
           all(col in health_data.columns for col in ["latitude", "longitude"]):
            
            # Join on SA2 code if available
            if "sa2_code_2021" in geographic_data.columns and "sa2_code_2021" in health_data.columns:
                joined_coords = geographic_data.select(["sa2_code_2021", "latitude", "longitude"]).join(
                    health_data.select(["sa2_code_2021", "latitude", "longitude"]),
                    on="sa2_code_2021",
                    how="inner",
                    suffix="_health"
                ).drop_nulls()
                
                coordinate_mismatches = []
                for row in joined_coords.iter_rows(named=True):
                    sa2_code = row["sa2_code_2021"]
                    geo_lat, geo_lon = row["latitude"], row["longitude"]
                    health_lat, health_lon = row["latitude_health"], row["longitude_health"]
                    
                    # Calculate distance (simple Euclidean for validation)
                    distance = ((geo_lat - health_lat)**2 + (geo_lon - health_lon)**2)**0.5
                    
                    # Threshold for acceptable coordinate difference (0.01 degrees ≈ 1km)
                    if distance > 0.01:
                        coordinate_mismatches.append({
                            "sa2_code": sa2_code,
                            "geographic_coords": [geo_lat, geo_lon],
                            "health_coords": [health_lat, health_lon],
                            "distance": distance
                        })
                
                if coordinate_mismatches:
                    violation_percentage = (len(coordinate_mismatches) / len(joined_coords)) * 100
                    
                    violations.append(ConsistencyViolation(
                        check_type=ConsistencyCheckType.GEOGRAPHIC_ALIGNMENT,
                        severity="warning" if violation_percentage < 5 else "error",
                        datasets_involved=["geographic", "health"],
                        violation_count=len(coordinate_mismatches),
                        total_records=len(joined_coords),
                        violation_percentage=violation_percentage,
                        description="Geographic coordinate misalignment between datasets",
                        sample_violations=coordinate_mismatches[:5],
                        details={"distance_threshold": 0.01, "records_compared": len(joined_coords)}
                    ))
        
        return violations
    
    def validate_temporal_consistency(self, datasets: Dict[str, pl.DataFrame]) -> List[ConsistencyViolation]:
        """
        Validate temporal consistency across time-series datasets.
        
        Args:
            datasets: Dictionary of dataset_name -> DataFrame
            
        Returns:
            List of consistency violations
        """
        violations = []
        
        # Find datasets with temporal data
        temporal_datasets = {}
        for dataset_name, df in datasets.items():
            date_columns = [col for col in df.columns if any(term in col.lower() for term in ["date", "time", "year", "month"])]
            if date_columns:
                temporal_datasets[dataset_name] = {
                    "dataframe": df,
                    "date_columns": date_columns
                }
        
        if len(temporal_datasets) < 2:
            return violations
        
        # Check temporal coverage consistency
        dataset_names = list(temporal_datasets.keys())
        for i in range(len(dataset_names)):
            for j in range(i + 1, len(dataset_names)):
                dataset1_name = dataset_names[i]
                dataset2_name = dataset_names[j]
                
                dataset1_info = temporal_datasets[dataset1_name]
                dataset2_info = temporal_datasets[dataset2_name]
                
                # Get date ranges for each dataset
                df1 = dataset1_info["dataframe"]
                df2 = dataset2_info["dataframe"]
                
                date_col1 = dataset1_info["date_columns"][0]
                date_col2 = dataset2_info["date_columns"][0]
                
                # Extract date ranges
                dates1 = df1[date_col1].drop_nulls()
                dates2 = df2[date_col2].drop_nulls()
                
                if len(dates1) == 0 or len(dates2) == 0:
                    continue
                
                # Convert to dates for comparison
                try:
                    min_date1 = min(dates1.to_list())
                    max_date1 = max(dates1.to_list())
                    min_date2 = min(dates2.to_list())
                    max_date2 = max(dates2.to_list())
                    
                    # Check for temporal gaps
                    if isinstance(min_date1, str):
                        min_date1 = datetime.strptime(min_date1.split("T")[0], "%Y-%m-%d").date()
                        max_date1 = datetime.strptime(max_date1.split("T")[0], "%Y-%m-%d").date()
                    if isinstance(min_date2, str):
                        min_date2 = datetime.strptime(min_date2.split("T")[0], "%Y-%m-%d").date()
                        max_date2 = datetime.strptime(max_date2.split("T")[0], "%Y-%m-%d").date()
                    
                    # Check overlap
                    overlap_start = max(min_date1, min_date2)
                    overlap_end = min(max_date1, max_date2)
                    
                    if overlap_start > overlap_end:
                        # No temporal overlap
                        violations.append(ConsistencyViolation(
                            check_type=ConsistencyCheckType.TEMPORAL_CONSISTENCY,
                            severity="warning",
                            datasets_involved=[dataset1_name, dataset2_name],
                            violation_count=1,
                            total_records=2,
                            violation_percentage=100.0,
                            description=f"No temporal overlap between {dataset1_name} and {dataset2_name}",
                            sample_violations=[{
                                "dataset1_range": [str(min_date1), str(max_date1)],
                                "dataset2_range": [str(min_date2), str(max_date2)]
                            }],
                            details={
                                "dataset1_date_column": date_col1,
                                "dataset2_date_column": date_col2,
                                "gap_days": (overlap_start - overlap_end).days
                            }
                        ))
                
                except (ValueError, TypeError) as e:
                    # Date parsing error
                    violations.append(ConsistencyViolation(
                        check_type=ConsistencyCheckType.TEMPORAL_CONSISTENCY,
                        severity="error",
                        datasets_involved=[dataset1_name, dataset2_name],
                        violation_count=1,
                        total_records=2,
                        violation_percentage=100.0,
                        description=f"Date format inconsistency between {dataset1_name} and {dataset2_name}",
                        sample_violations=[{"error": str(e)}],
                        details={"parsing_error": True}
                    ))
        
        return violations
    
    def validate_referential_integrity(self, 
                                     parent_dataset: pl.DataFrame, 
                                     child_dataset: pl.DataFrame,
                                     parent_key: str,
                                     child_key: str,
                                     parent_name: str = "parent",
                                     child_name: str = "child") -> List[ConsistencyViolation]:
        """
        Validate referential integrity between parent and child datasets.
        
        Args:
            parent_dataset: Parent dataset with primary keys
            child_dataset: Child dataset with foreign keys
            parent_key: Column name for parent key
            child_key: Column name for child key (foreign key)
            parent_name: Name of parent dataset
            child_name: Name of child dataset
            
        Returns:
            List of consistency violations
        """
        violations = []
        
        if parent_key not in parent_dataset.columns or child_key not in child_dataset.columns:
            violations.append(ConsistencyViolation(
                check_type=ConsistencyCheckType.REFERENTIAL_INTEGRITY,
                severity="error",
                datasets_involved=[parent_name, child_name],
                violation_count=1,
                total_records=1,
                violation_percentage=100.0,
                description=f"Missing key columns for referential integrity check",
                sample_violations=[{
                    "missing_parent_key": parent_key not in parent_dataset.columns,
                    "missing_child_key": child_key not in child_dataset.columns
                }],
                details={"parent_key": parent_key, "child_key": child_key}
            ))
            return violations
        
        # Get unique keys from both datasets
        parent_keys = set(parent_dataset[parent_key].drop_nulls().to_list())
        child_keys = set(child_dataset[child_key].drop_nulls().to_list())
        
        # Find orphaned records (child keys without parent)
        orphaned_keys = child_keys - parent_keys
        
        if orphaned_keys:
            violation_percentage = (len(orphaned_keys) / len(child_keys)) * 100 if child_keys else 0
            
            violations.append(ConsistencyViolation(
                check_type=ConsistencyCheckType.REFERENTIAL_INTEGRITY,
                severity="error" if violation_percentage > 5 else "warning",
                datasets_involved=[parent_name, child_name],
                violation_count=len(orphaned_keys),
                total_records=len(child_keys),
                violation_percentage=violation_percentage,
                description=f"Orphaned records in {child_name} without corresponding {parent_name} records",
                sample_violations=[{"orphaned_keys": list(orphaned_keys)[:10]}],
                details={
                    "parent_key_count": len(parent_keys),
                    "child_key_count": len(child_keys),
                    "orphaned_count": len(orphaned_keys)
                }
            ))
        
        return violations
    
    def validate_schema_consistency(self, datasets: Dict[str, pl.DataFrame]) -> List[ConsistencyViolation]:
        """
        Validate schema consistency across related datasets.
        
        Args:
            datasets: Dictionary of dataset_name -> DataFrame
            
        Returns:
            List of consistency violations
        """
        violations = []
        
        # Check for common columns that should have consistent types
        common_columns = set.intersection(*[set(df.columns) for df in datasets.values()])
        
        for column in common_columns:
            # Get data types for this column across datasets
            column_types = {}
            for dataset_name, df in datasets.items():
                column_types[dataset_name] = str(df[column].dtype)
            
            # Check if all types are the same
            unique_types = set(column_types.values())
            if len(unique_types) > 1:
                violations.append(ConsistencyViolation(
                    check_type=ConsistencyCheckType.SCHEMA_CONSISTENCY,
                    severity="warning",
                    datasets_involved=list(datasets.keys()),
                    violation_count=len(unique_types) - 1,
                    total_records=len(datasets),
                    violation_percentage=((len(unique_types) - 1) / len(datasets)) * 100,
                    description=f"Inconsistent data types for column '{column}' across datasets",
                    sample_violations=[{"column": column, "types_by_dataset": column_types}],
                    details={"column": column, "unique_types": list(unique_types)}
                ))
        
        return violations


class TestCrossDatasetConsistencyValidation:
    """Test suite for cross-dataset consistency validation."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return AustralianHealthDataValidator()
    
    @pytest.fixture
    def consistency_validator(self, validator):
        """Create consistency validator instance."""
        return CrossDatasetConsistencyValidator(validator)
    
    @pytest.fixture
    def seifa_dataset(self):
        """SEIFA dataset for consistency testing."""
        return pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001", "301011002", "401011003"],
            "sa2_name_2021": ["Sydney Harbour", "Melbourne CBD", "Brisbane Inner", "Adelaide North"],
            "irsd_score": [1050, 950, 1100, 980],
            "irsd_decile": [8, 5, 9, 6],
            "usual_resident_population": [15000, 12000, 18000, 8500],
            "latitude": [-33.8688, -37.8136, -27.4698, -34.9285],
            "longitude": [151.2093, 144.9631, 153.0251, 138.6007],
            "last_updated": ["2023-01-01", "2023-01-01", "2023-01-01", "2023-01-01"]
        })
    
    @pytest.fixture
    def census_dataset(self):
        """Census dataset for consistency testing."""
        return pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001", "301011002", "501011004"],  # Different last SA2
            "sa2_name_2021": ["Sydney Harbour", "Melbourne CBD", "Brisbane Inner", "Perth CBD"],
            "tot_p_p": [15100, 11950, 18050, 22000],  # Slightly different populations
            "tot_p_m": [7550, 5975, 9025, 11000],
            "tot_p_f": [7550, 5975, 9025, 11000],
            "census_date": ["2021-08-10", "2021-08-10", "2021-08-10", "2021-08-10"]
        })
    
    @pytest.fixture
    def health_dataset(self):
        """Health dataset for consistency testing."""
        return pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001", "301011002", "401011003", "601011005"],  # Extra SA2
            "provider_count": [25, 18, 22, 12, 8],
            "service_volume": [1250, 890, 1100, 600, 400],
            "latitude": [-33.8690, -37.8140, -27.4700, -34.9290, -42.8821],  # Slightly different coords
            "longitude": [151.2095, 144.9635, 153.0255, 138.6010, 147.3272],
            "data_period": ["2023-Q1", "2023-Q1", "2023-Q1", "2023-Q1", "2023-Q1"]
        })
    
    @pytest.fixture
    def geographic_dataset(self):
        """Geographic boundaries dataset for consistency testing."""
        return pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001", "301011002", "401011003", "501011004"],
            "sa2_name_2021": ["Sydney Harbour", "Melbourne CBD", "Brisbane Inner", "Adelaide North", "Perth CBD"],
            "latitude": [-33.8688, -37.8136, -27.4698, -34.9285, -31.9505],  # Exact coordinates
            "longitude": [151.2093, 144.9631, 153.0251, 138.6007, 115.8605],
            "area_sqkm": [5.2, 3.8, 4.1, 6.5, 7.2],
            "boundary_updated": ["2021-07-01", "2021-07-01", "2021-07-01", "2021-07-01", "2021-07-01"]
        })
    
    def test_sa2_alignment_validation_perfect_match(self, consistency_validator, seifa_dataset, geographic_dataset):
        """Test SA2 alignment validation with perfect match."""
        # Create datasets with identical SA2 codes
        seifa_subset = seifa_dataset.filter(pl.col("sa2_code_2021").is_in(["101021007", "201011001", "301011002"]))
        geographic_subset = geographic_dataset.filter(pl.col("sa2_code_2021").is_in(["101021007", "201011001", "301011002"]))
        
        datasets = {
            "seifa": seifa_subset,
            "geographic": geographic_subset
        }
        
        violations = consistency_validator.validate_sa2_alignment(datasets)
        
        # Perfect match should have no violations
        assert len(violations) == 0
    
    def test_sa2_alignment_validation_with_mismatches(self, consistency_validator, seifa_dataset, census_dataset, health_dataset):
        """Test SA2 alignment validation with mismatches."""
        datasets = {
            "seifa": seifa_dataset,
            "census": census_dataset,
            "health": health_dataset
        }
        
        violations = consistency_validator.validate_sa2_alignment(datasets)
        
        # Should detect mismatches
        assert len(violations) > 0
        
        # Check violation details
        for violation in violations:
            assert violation.check_type == ConsistencyCheckType.SA2_ALIGNMENT
            assert len(violation.datasets_involved) == 2
            assert violation.violation_count > 0
            assert violation.violation_percentage > 0
            assert "overlap_percentage" in violation.details
    
    def test_population_consistency_validation_good_match(self, consistency_validator):
        """Test population consistency validation with good match."""
        # Create datasets with consistent population data
        census_data = pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001", "301011002"],
            "tot_p_p": [15000, 12000, 18000]
        })
        
        seifa_data = pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001", "301011002"],
            "usual_resident_population": [15050, 11980, 18020]  # Within 5% tolerance
        })
        
        violations = consistency_validator.validate_population_consistency(census_data, seifa_data, tolerance_percent=5.0)
        
        # Should have no violations within tolerance
        assert len(violations) == 0
    
    def test_population_consistency_validation_poor_match(self, consistency_validator, census_dataset, seifa_dataset):
        """Test population consistency validation with poor match."""
        violations = consistency_validator.validate_population_consistency(census_dataset, seifa_dataset, tolerance_percent=1.0)
        
        # Should detect inconsistencies with tight tolerance
        assert len(violations) > 0
        
        violation = violations[0]
        assert violation.check_type == ConsistencyCheckType.POPULATION_CONSISTENCY
        assert "census" in violation.datasets_involved
        assert "seifa" in violation.datasets_involved
        assert violation.violation_count > 0
        assert "tolerance_percent" in violation.details
    
    def test_geographic_alignment_validation(self, consistency_validator, geographic_dataset, health_dataset):
        """Test geographic coordinate alignment validation."""
        violations = consistency_validator.validate_geographic_alignment(geographic_dataset, health_dataset)
        
        # Should detect coordinate misalignments
        assert len(violations) > 0
        
        violation = violations[0]
        assert violation.check_type == ConsistencyCheckType.GEOGRAPHIC_ALIGNMENT
        assert "geographic" in violation.datasets_involved
        assert "health" in violation.datasets_involved
        assert violation.violation_count > 0
        assert "distance_threshold" in violation.details
        
        # Check sample violations have coordinate information
        for sample in violation.sample_violations:
            assert "geographic_coords" in sample
            assert "health_coords" in sample
            assert "distance" in sample
    
    def test_temporal_consistency_validation_no_overlap(self, consistency_validator):
        """Test temporal consistency validation with no overlap."""
        # Create datasets with non-overlapping time periods
        dataset1 = pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001"],
            "data_date": ["2020-01-01", "2020-06-01"],
            "metric": [100, 110]
        })
        
        dataset2 = pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001"],
            "data_date": ["2022-01-01", "2022-06-01"],
            "metric": [120, 130]
        })
        
        datasets = {
            "early_data": dataset1,
            "late_data": dataset2
        }
        
        violations = consistency_validator.validate_temporal_consistency(datasets)
        
        # Should detect temporal gap
        assert len(violations) > 0
        
        violation = violations[0]
        assert violation.check_type == ConsistencyCheckType.TEMPORAL_CONSISTENCY
        assert "early_data" in violation.datasets_involved
        assert "late_data" in violation.datasets_involved
        assert "gap_days" in violation.details
    
    def test_temporal_consistency_validation_overlapping(self, consistency_validator):
        """Test temporal consistency validation with overlapping periods."""
        # Create datasets with overlapping time periods
        dataset1 = pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001"],
            "data_date": ["2023-01-01", "2023-06-01"],
            "metric": [100, 110]
        })
        
        dataset2 = pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001"],
            "data_date": ["2023-03-01", "2023-09-01"],
            "metric": [120, 130]
        })
        
        datasets = {
            "dataset1": dataset1,
            "dataset2": dataset2
        }
        
        violations = consistency_validator.validate_temporal_consistency(datasets)
        
        # Should have no violations with overlapping periods
        assert len(violations) == 0
    
    def test_referential_integrity_validation_valid(self, consistency_validator):
        """Test referential integrity validation with valid references."""
        # Parent dataset (SA2 boundaries)
        parent_data = pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001", "301011002", "401011003"],
            "sa2_name_2021": ["Sydney", "Melbourne", "Brisbane", "Adelaide"]
        })
        
        # Child dataset (health services) - all references valid
        child_data = pl.DataFrame({
            "service_id": [1, 2, 3, 4, 5],
            "sa2_code_2021": ["101021007", "201011001", "301011002", "101021007", "201011001"],
            "service_type": ["GP", "Pharmacy", "Hospital", "GP", "Pharmacy"]
        })
        
        violations = consistency_validator.validate_referential_integrity(
            parent_data, child_data, "sa2_code_2021", "sa2_code_2021", "boundaries", "health_services"
        )
        
        # Should have no violations
        assert len(violations) == 0
    
    def test_referential_integrity_validation_orphaned_records(self, consistency_validator):
        """Test referential integrity validation with orphaned records."""
        # Parent dataset
        parent_data = pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001", "301011002"],
            "sa2_name_2021": ["Sydney", "Melbourne", "Brisbane"]
        })
        
        # Child dataset with orphaned records
        child_data = pl.DataFrame({
            "service_id": [1, 2, 3, 4, 5],
            "sa2_code_2021": ["101021007", "201011001", "401011003", "501011004", "601011005"],  # Last 3 are orphaned
            "service_type": ["GP", "Pharmacy", "Hospital", "GP", "Pharmacy"]
        })
        
        violations = consistency_validator.validate_referential_integrity(
            parent_data, child_data, "sa2_code_2021", "sa2_code_2021", "boundaries", "health_services"
        )
        
        # Should detect orphaned records
        assert len(violations) > 0
        
        violation = violations[0]
        assert violation.check_type == ConsistencyCheckType.REFERENTIAL_INTEGRITY
        assert "boundaries" in violation.datasets_involved
        assert "health_services" in violation.datasets_involved
        assert violation.violation_count == 3  # Three orphaned records
        assert "orphaned_keys" in violation.sample_violations[0]
    
    def test_schema_consistency_validation_consistent_types(self, consistency_validator):
        """Test schema consistency validation with consistent types."""
        # Datasets with consistent column types
        dataset1 = pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001"],
            "metric_value": [100, 110],
            "data_date": ["2023-01-01", "2023-01-02"]
        })
        
        dataset2 = pl.DataFrame({
            "sa2_code_2021": ["301011002", "401011003"],
            "metric_value": [120, 130],
            "data_date": ["2023-01-01", "2023-01-02"]
        })
        
        datasets = {
            "dataset1": dataset1,
            "dataset2": dataset2
        }
        
        violations = consistency_validator.validate_schema_consistency(datasets)
        
        # Should have no violations with consistent types
        assert len(violations) == 0
    
    def test_schema_consistency_validation_inconsistent_types(self, consistency_validator):
        """Test schema consistency validation with inconsistent types."""
        # Datasets with inconsistent column types
        dataset1 = pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001"],
            "metric_value": [100, 110],  # Integer
            "data_date": ["2023-01-01", "2023-01-02"]
        })
        
        dataset2 = pl.DataFrame({
            "sa2_code_2021": ["301011002", "401011003"],
            "metric_value": [120.5, 130.7],  # Float
            "data_date": ["2023-01-01", "2023-01-02"]
        })
        
        datasets = {
            "dataset1": dataset1,
            "dataset2": dataset2
        }
        
        violations = consistency_validator.validate_schema_consistency(datasets)
        
        # Should detect type inconsistencies
        assert len(violations) > 0
        
        violation = violations[0]
        assert violation.check_type == ConsistencyCheckType.SCHEMA_CONSISTENCY
        assert "metric_value" in violation.description
        assert "types_by_dataset" in violation.sample_violations[0]
    
    def test_comprehensive_cross_dataset_validation(self, consistency_validator, seifa_dataset, census_dataset, health_dataset, geographic_dataset):
        """Test comprehensive cross-dataset validation across all consistency types."""
        datasets = {
            "seifa": seifa_dataset,
            "census": census_dataset,
            "health": health_dataset,
            "geographic": geographic_dataset
        }
        
        # Run all consistency checks
        all_violations = []
        
        # SA2 alignment
        sa2_violations = consistency_validator.validate_sa2_alignment(datasets)
        all_violations.extend(sa2_violations)
        
        # Population consistency
        pop_violations = consistency_validator.validate_population_consistency(census_dataset, seifa_dataset)
        all_violations.extend(pop_violations)
        
        # Geographic alignment
        geo_violations = consistency_validator.validate_geographic_alignment(geographic_dataset, health_dataset)
        all_violations.extend(geo_violations)
        
        # Temporal consistency
        temporal_violations = consistency_validator.validate_temporal_consistency(datasets)
        all_violations.extend(temporal_violations)
        
        # Referential integrity (SEIFA -> Geographic)
        ref_violations = consistency_validator.validate_referential_integrity(
            geographic_dataset, seifa_dataset, "sa2_code_2021", "sa2_code_2021", "geographic", "seifa"
        )
        all_violations.extend(ref_violations)
        
        # Schema consistency
        schema_violations = consistency_validator.validate_schema_consistency(datasets)
        all_violations.extend(schema_violations)
        
        # Analyze results
        violations_by_type = {}
        for violation in all_violations:
            check_type = violation.check_type
            if check_type not in violations_by_type:
                violations_by_type[check_type] = []
            violations_by_type[check_type].append(violation)
        
        # Log comprehensive results
        logger.info("Comprehensive cross-dataset validation results:")
        logger.info(f"  Total violations: {len(all_violations)}")
        
        for check_type, violations in violations_by_type.items():
            logger.info(f"  {check_type.value}: {len(violations)} violations")
            for violation in violations:
                logger.info(f"    - {violation.description} ({violation.severity})")
        
        # Validate that we tested multiple consistency types
        assert len(violations_by_type) >= 3, f"Expected violations in multiple categories, got: {list(violations_by_type.keys())}"
        
        # Should have some violations due to intentional mismatches in test data
        assert len(all_violations) > 0, "Expected to find some consistency violations in test data"
        
        # Check that all violation objects have required fields
        for violation in all_violations:
            assert hasattr(violation, 'check_type')
            assert hasattr(violation, 'severity')
            assert hasattr(violation, 'datasets_involved')
            assert hasattr(violation, 'violation_count')
            assert hasattr(violation, 'violation_percentage')
            assert hasattr(violation, 'description')
            assert hasattr(violation, 'sample_violations')
            assert hasattr(violation, 'details')
            
            # Validate severity levels
            assert violation.severity in ["info", "warning", "error", "critical"]
            
            # Validate percentages
            assert 0 <= violation.violation_percentage <= 100
    
    def test_missing_columns_handling(self, consistency_validator):
        """Test handling of missing columns in consistency checks."""
        # Dataset without SA2 codes
        dataset_no_sa2 = pl.DataFrame({
            "region_id": ["R001", "R002", "R003"],
            "metric": [100, 110, 120]
        })
        
        # Dataset with SA2 codes
        dataset_with_sa2 = pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001", "301011002"],
            "metric": [100, 110, 120]
        })
        
        datasets = {
            "no_sa2": dataset_no_sa2,
            "with_sa2": dataset_with_sa2
        }
        
        # SA2 alignment should handle missing columns gracefully
        violations = consistency_validator.validate_sa2_alignment(datasets)
        
        # Should not crash, may return empty violations if no common SA2 columns
        assert isinstance(violations, list)
    
    def test_empty_datasets_handling(self, consistency_validator):
        """Test handling of empty datasets."""
        empty_dataset = pl.DataFrame()
        normal_dataset = pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001"],
            "metric": [100, 110]
        })
        
        datasets = {
            "empty": empty_dataset,
            "normal": normal_dataset
        }
        
        # Should handle empty datasets gracefully
        violations = consistency_validator.validate_sa2_alignment(datasets)
        assert isinstance(violations, list)
        
        # Population consistency with empty data
        pop_violations = consistency_validator.validate_population_consistency(empty_dataset, normal_dataset)
        assert isinstance(pop_violations, list)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])