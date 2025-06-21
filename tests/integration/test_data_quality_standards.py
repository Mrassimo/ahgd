"""Test suite for data quality standards validation.

This module implements Test-Driven Development for data quality requirements,
validating completeness, statistical validity, and Australian data standards.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from statistics import mean, median, stdev

from src.utils.logging import get_logger
from src.schemas.base import BaseSchemaV1

logger = get_logger(__name__)


@dataclass
class CompletenessRequirement:
    """Defines completeness requirements for data fields."""
    field_name: str
    minimum_completeness: Decimal  # 0.0 to 1.0
    mandatory: bool
    business_rule: str
    exemption_conditions: List[str]


@dataclass
class StatisticalValidationThreshold:
    """Defines statistical validation thresholds for numeric fields."""
    field_name: str
    min_value: Optional[Decimal]
    max_value: Optional[Decimal]
    expected_mean_range: tuple[Decimal, Decimal]
    outlier_threshold_stddev: Decimal
    distribution_type: str  # normal, log-normal, uniform, etc.
    allow_zero: bool
    allow_negative: bool


@dataclass
class GeographicValidationRequirement:
    """Defines geographic data validation requirements."""
    field_name: str
    coordinate_system: str  # GDA2020, WGS84, etc.
    precision_meters: Decimal
    boundary_validation: bool
    topology_validation: bool
    projection_validation: bool


class TestCompletenessRequirements:
    """Test field-level completeness standards."""
    
    @pytest.fixture
    def completeness_standards(self):
        """Define completeness standards for all data fields."""
        return [
            CompletenessRequirement(
                field_name="sa2_code",
                minimum_completeness=Decimal("1.0"),
                mandatory=True,
                business_rule="All records must have valid SA2 code",
                exemption_conditions=[]
            ),
            CompletenessRequirement(
                field_name="sa2_name",
                minimum_completeness=Decimal("1.0"),
                mandatory=True,
                business_rule="All SA2 codes must have corresponding names",
                exemption_conditions=[]
            ),
            CompletenessRequirement(
                field_name="total_population",
                minimum_completeness=Decimal("0.99"),
                mandatory=True,
                business_rule="Population data required for health calculations",
                exemption_conditions=["Very remote areas with confidentialised data"]
            ),
            CompletenessRequirement(
                field_name="life_expectancy",
                minimum_completeness=Decimal("0.90"),
                mandatory=False,
                business_rule="Life expectancy calculated where sufficient data available",
                exemption_conditions=["Areas with <1000 population", "Confidentialised areas"]
            ),
            CompletenessRequirement(
                field_name="seifa_irsad_score",
                minimum_completeness=Decimal("0.98"),
                mandatory=True,
                business_rule="SEIFA scores required for socioeconomic analysis",
                exemption_conditions=["Non-standard areas", "Offshore territories"]
            ),
            CompletenessRequirement(
                field_name="gp_services_per_1000",
                minimum_completeness=Decimal("0.85"),
                mandatory=False,
                business_rule="GP service data may be unavailable in remote areas",
                exemption_conditions=["Very remote areas", "Areas without GP services"]
            ),
            CompletenessRequirement(
                field_name="geometry",
                minimum_completeness=Decimal("1.0"),
                mandatory=True,
                business_rule="All SA2s must have valid boundary geometry",
                exemption_conditions=[]
            )
        ]
    
    def test_completeness_requirements(self, completeness_standards):
        """Test that all datasets meet field-level completeness requirements.
        
        This test validates that each field meets its minimum completeness
        threshold and identifies data gaps that need addressing.
        """
        from src.testing.target_validation import QualityStandardsChecker
        from src.etl.data_warehouse import DataWarehouse
        
        checker = QualityStandardsChecker()
        warehouse = DataWarehouse()
        
        # Get complete dataset for validation
        complete_dataset = warehouse.get_master_health_dataset()
        
        # Validate each completeness requirement
        for requirement in completeness_standards:
            completeness_result = checker.validate_field_completeness(
                dataset=complete_dataset,
                field_name=requirement.field_name,
                minimum_completeness=requirement.minimum_completeness,
                exemption_conditions=requirement.exemption_conditions
            )
            
            # Assert completeness meets requirements
            assert completeness_result.actual_completeness >= requirement.minimum_completeness, \
                f"Field {requirement.field_name} completeness {completeness_result.actual_completeness} " \
                f"below required {requirement.minimum_completeness}"
            
            # For mandatory fields, no exemptions allowed
            if requirement.mandatory:
                assert completeness_result.actual_completeness >= Decimal("0.99"), \
                    f"Mandatory field {requirement.field_name} has insufficient completeness"
            
            # Validate exemption handling
            if completeness_result.exempted_records > 0:
                assert len(requirement.exemption_conditions) > 0, \
                    f"Field {requirement.field_name} has exempted records but no defined exemption conditions"
    
    def test_missing_data_patterns(self):
        """Test analysis of missing data patterns for systematic issues."""
        from src.testing.target_validation import QualityStandardsChecker
        from src.etl.data_warehouse import DataWarehouse
        
        checker = QualityStandardsChecker()
        warehouse = DataWarehouse()
        
        complete_dataset = warehouse.get_master_health_dataset()
        
        # Analyse missing data patterns
        missing_patterns = checker.analyse_missing_data_patterns(complete_dataset)
        
        # Check for systematic missing data issues
        assert missing_patterns.systematic_missing_by_state < 0.10, \
            "Systematic missing data by state indicates collection issues"
        
        assert missing_patterns.systematic_missing_by_remoteness < 0.15, \
            "Systematic missing data by remoteness area acceptable up to 15%"
        
        # Validate missing data is not correlated with socioeconomic status
        assert abs(missing_patterns.missing_seifa_correlation) < 0.20, \
            "Missing data should not be strongly correlated with socioeconomic status"
    
    def test_data_availability_by_jurisdiction(self):
        """Test data availability across Australian jurisdictions."""
        from src.testing.target_validation import QualityStandardsChecker
        from src.etl.data_warehouse import DataWarehouse
        
        checker = QualityStandardsChecker()
        warehouse = DataWarehouse()
        
        # Test data availability for each state/territory
        jurisdictions = ['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT']
        
        for jurisdiction in jurisdictions:
            jurisdiction_data = warehouse.get_data_by_jurisdiction(jurisdiction)
            availability_report = checker.assess_jurisdiction_data_availability(jurisdiction_data)
            
            # Each jurisdiction should meet minimum data availability
            assert availability_report.overall_availability >= 0.85, \
                f"Jurisdiction {jurisdiction} data availability below 85%"
            
            # Critical fields must be highly available
            critical_fields = ['sa2_code', 'total_population', 'seifa_irsad_score']
            for field in critical_fields:
                field_availability = availability_report.field_availability[field]
                assert field_availability >= 0.95, \
                    f"Critical field {field} availability in {jurisdiction} below 95%"


class TestStatisticalValidationThresholds:
    """Test outlier detection and distribution compliance."""
    
    @pytest.fixture
    def statistical_thresholds(self):
        """Define statistical validation thresholds for numeric fields."""
        return [
            StatisticalValidationThreshold(
                field_name="total_population",
                min_value=Decimal("0"),
                max_value=Decimal("50000"),  # Largest SA2 in Australia
                expected_mean_range=(Decimal("3000"), Decimal("8000")),
                outlier_threshold_stddev=Decimal("3.0"),
                distribution_type="log-normal",
                allow_zero=False,
                allow_negative=False
            ),
            StatisticalValidationThreshold(
                field_name="life_expectancy",
                min_value=Decimal("70.0"),
                max_value=Decimal("90.0"),
                expected_mean_range=(Decimal("80.0"), Decimal("85.0")),
                outlier_threshold_stddev=Decimal("2.5"),
                distribution_type="normal",
                allow_zero=False,
                allow_negative=False
            ),
            StatisticalValidationThreshold(
                field_name="seifa_irsad_score",
                min_value=Decimal("500"),
                max_value=Decimal("1200"),
                expected_mean_range=(Decimal("950"), Decimal("1050")),
                outlier_threshold_stddev=Decimal("2.0"),
                distribution_type="normal",
                allow_zero=False,
                allow_negative=False
            ),
            StatisticalValidationThreshold(
                field_name="gp_services_per_1000",
                min_value=Decimal("0.0"),
                max_value=Decimal("10.0"),
                expected_mean_range=(Decimal("0.5"), Decimal("2.5")),
                outlier_threshold_stddev=Decimal("2.5"),
                distribution_type="log-normal",
                allow_zero=True,
                allow_negative=False
            ),
            StatisticalValidationThreshold(
                field_name="population_density",
                min_value=Decimal("0.001"),
                max_value=Decimal("20000.0"),
                expected_mean_range=(Decimal("10.0"), Decimal("500.0")),
                outlier_threshold_stddev=Decimal("3.0"),
                distribution_type="log-normal",
                allow_zero=False,
                allow_negative=False
            )
        ]
    
    def test_statistical_validation_thresholds(self, statistical_thresholds):
        """Test that numeric fields comply with statistical validation thresholds.
        
        Validates range constraints, distribution properties, and outlier detection
        for all numeric health and demographic indicators.
        """
        from src.testing.target_validation import QualityStandardsChecker
        from src.etl.data_warehouse import DataWarehouse
        
        checker = QualityStandardsChecker()
        warehouse = DataWarehouse()
        
        complete_dataset = warehouse.get_master_health_dataset()
        
        for threshold in statistical_thresholds:
            field_data = complete_dataset[threshold.field_name].dropna()
            
            # Test range constraints
            if threshold.min_value is not None:
                min_violations = (field_data < threshold.min_value).sum()
                assert min_violations == 0, \
                    f"Field {threshold.field_name} has {min_violations} values below minimum {threshold.min_value}"
            
            if threshold.max_value is not None:
                max_violations = (field_data > threshold.max_value).sum()
                assert max_violations == 0, \
                    f"Field {threshold.field_name} has {max_violations} values above maximum {threshold.max_value}"
            
            # Test zero/negative value constraints
            if not threshold.allow_zero:
                zero_violations = (field_data == 0).sum()
                assert zero_violations == 0, \
                    f"Field {threshold.field_name} has {zero_violations} zero values (not allowed)"
            
            if not threshold.allow_negative:
                negative_violations = (field_data < 0).sum()
                assert negative_violations == 0, \
                    f"Field {threshold.field_name} has {negative_violations} negative values (not allowed)"
            
            # Test expected mean range
            actual_mean = Decimal(str(field_data.mean()))
            assert threshold.expected_mean_range[0] <= actual_mean <= threshold.expected_mean_range[1], \
                f"Field {threshold.field_name} mean {actual_mean} outside expected range {threshold.expected_mean_range}"
            
            # Test outlier detection
            outliers = checker.detect_statistical_outliers(
                field_data, threshold.outlier_threshold_stddev
            )
            outlier_percentage = len(outliers) / len(field_data)
            assert outlier_percentage < 0.05, \
                f"Field {threshold.field_name} has {outlier_percentage:.2%} outliers (>5% indicates data quality issues)"
    
    def test_distribution_conformance(self, statistical_thresholds):
        """Test that numeric fields follow expected statistical distributions."""
        from src.testing.target_validation import QualityStandardsChecker
        from src.etl.data_warehouse import DataWarehouse
        from scipy import stats
        
        checker = QualityStandardsChecker()
        warehouse = DataWarehouse()
        
        complete_dataset = warehouse.get_master_health_dataset()
        
        for threshold in statistical_thresholds:
            field_data = complete_dataset[threshold.field_name].dropna()
            
            # Test distribution type
            if threshold.distribution_type == "normal":
                # Shapiro-Wilk test for normality (for samples < 5000)
                if len(field_data) <= 5000:
                    statistic, p_value = stats.shapiro(field_data)
                    # Accept if p > 0.01 (less strict for health data)
                    assert p_value > 0.01, \
                        f"Field {threshold.field_name} does not follow normal distribution (p={p_value:.4f})"
            
            elif threshold.distribution_type == "log-normal":
                # Test log-normal distribution
                log_data = np.log(field_data + 1)  # Add 1 to handle zeros
                if len(log_data) <= 5000:
                    statistic, p_value = stats.shapiro(log_data)
                    assert p_value > 0.01, \
                        f"Field {threshold.field_name} does not follow log-normal distribution (p={p_value:.4f})"
    
    def test_inter_field_correlations(self):
        """Test expected correlations between related fields."""
        from src.testing.target_validation import QualityStandardsChecker
        from src.etl.data_warehouse import DataWarehouse
        
        checker = QualityStandardsChecker()
        warehouse = DataWarehouse()
        
        complete_dataset = warehouse.get_master_health_dataset()
        
        # Expected correlations
        expected_correlations = [
            ('seifa_irsad_decile', 'life_expectancy', 0.3, 0.7),  # Higher SEIFA -> Higher life expectancy
            ('gp_services_per_1000', 'population_density', -0.8, -0.2),  # Urban areas have lower GP ratio
            ('total_population', 'area_sqkm', -0.6, 0.0),  # Larger areas tend to have lower population
        ]
        
        for field1, field2, min_corr, max_corr in expected_correlations:
            correlation = complete_dataset[field1].corr(complete_dataset[field2])
            assert min_corr <= correlation <= max_corr, \
                f"Correlation between {field1} and {field2} is {correlation:.3f}, expected [{min_corr}, {max_corr}]"


class TestGeographicValidationRequirements:
    """Test spatial data quality validation."""
    
    @pytest.fixture
    def geographic_requirements(self):
        """Define geographic data validation requirements."""
        return [
            GeographicValidationRequirement(
                field_name="geometry",
                coordinate_system="GDA2020",
                precision_meters=Decimal("10.0"),
                boundary_validation=True,
                topology_validation=True,
                projection_validation=True
            ),
            GeographicValidationRequirement(
                field_name="centroid_lat",
                coordinate_system="WGS84",
                precision_meters=Decimal("100.0"),
                boundary_validation=True,
                topology_validation=False,
                projection_validation=True
            ),
            GeographicValidationRequirement(
                field_name="centroid_lon",
                coordinate_system="WGS84",
                precision_meters=Decimal("100.0"),
                boundary_validation=True,
                topology_validation=False,
                projection_validation=True
            )
        ]
    
    def test_geographic_validation_requirements(self, geographic_requirements):
        """Test validation of spatial data quality.
        
        Validates coordinate systems, spatial precision, boundary integrity,
        and topological consistency of geographic data.
        """
        from src.testing.target_validation import QualityStandardsChecker
        from src.etl.data_warehouse import DataWarehouse
        
        checker = QualityStandardsChecker()
        warehouse = DataWarehouse()
        
        geographic_dataset = warehouse.get_geographic_dataset()
        
        for requirement in geographic_requirements:
            validation_result = checker.validate_geographic_field(
                dataset=geographic_dataset,
                field_name=requirement.field_name,
                coordinate_system=requirement.coordinate_system,
                precision_meters=requirement.precision_meters
            )
            
            # Test coordinate system compliance
            assert validation_result.coordinate_system_valid, \
                f"Field {requirement.field_name} coordinate system validation failed"
            
            # Test spatial precision
            assert validation_result.precision_compliant, \
                f"Field {requirement.field_name} does not meet precision requirement of {requirement.precision_meters}m"
            
            # Test boundary validation if required
            if requirement.boundary_validation:
                assert validation_result.boundaries_valid, \
                    f"Field {requirement.field_name} boundary validation failed"
            
            # Test topology validation if required
            if requirement.topology_validation:
                assert validation_result.topology_valid, \
                    f"Field {requirement.field_name} topology validation failed"
    
    def test_australian_geographic_extent_validation(self):
        """Test that all coordinates fall within Australian geographic extent."""
        from src.testing.target_validation import QualityStandardsChecker
        from src.etl.data_warehouse import DataWarehouse
        
        checker = QualityStandardsChecker()
        warehouse = DataWarehouse()
        
        geographic_dataset = warehouse.get_geographic_dataset()
        
        # Australian geographic extent (including external territories)
        AUSTRALIA_BOUNDS = {
            'min_lat': Decimal('-55.0'),  # Includes Antarctic territory
            'max_lat': Decimal('-10.0'),
            'min_lon': Decimal('110.0'),
            'max_lon': Decimal('160.0')
        }
        
        # Validate latitude bounds
        lat_out_of_bounds = (
            (geographic_dataset['centroid_lat'] < AUSTRALIA_BOUNDS['min_lat']) |
            (geographic_dataset['centroid_lat'] > AUSTRALIA_BOUNDS['max_lat'])
        ).sum()
        
        assert lat_out_of_bounds == 0, \
            f"{lat_out_of_bounds} records have latitude outside Australian bounds"
        
        # Validate longitude bounds
        lon_out_of_bounds = (
            (geographic_dataset['centroid_lon'] < AUSTRALIA_BOUNDS['min_lon']) |
            (geographic_dataset['centroid_lon'] > AUSTRALIA_BOUNDS['max_lon'])
        ).sum()
        
        assert lon_out_of_bounds == 0, \
            f"{lon_out_of_bounds} records have longitude outside Australian bounds"
    
    def test_sa2_boundary_area_validation(self):
        """Test that SA2 area calculations are reasonable."""
        from src.testing.target_validation import QualityStandardsChecker
        from src.etl.data_warehouse import DataWarehouse
        
        checker = QualityStandardsChecker()
        warehouse = DataWarehouse()
        
        geographic_dataset = warehouse.get_geographic_dataset()
        
        # SA2 area validation (in square kilometres)
        MIN_SA2_AREA = Decimal('0.01')  # 0.01 sq km minimum
        MAX_SA2_AREA = Decimal('50000.0')  # 50,000 sq km maximum (very remote areas)
        
        area_too_small = (geographic_dataset['area_sqkm'] < MIN_SA2_AREA).sum()
        area_too_large = (geographic_dataset['area_sqkm'] > MAX_SA2_AREA).sum()
        
        assert area_too_small == 0, \
            f"{area_too_small} SA2s have unreasonably small areas (<{MIN_SA2_AREA} sq km)"
        
        assert area_too_large == 0, \
            f"{area_too_large} SA2s have unreasonably large areas (>{MAX_SA2_AREA} sq km)"


class TestDataLineageTracking:
    """Test audit trail completeness."""
    
    def test_data_lineage_tracking(self):
        """Test that complete audit trails are maintained for all data transformations.
        
        Validates that every piece of data can be tracked back to its source,
        including transformation steps, data quality checks, and processing timestamps.
        """
        from src.testing.target_validation import QualityStandardsChecker
        from src.etl.data_warehouse import DataWarehouse
        from src.utils.audit import AuditTrailManager
        
        checker = QualityStandardsChecker()
        warehouse = DataWarehouse()
        audit_manager = AuditTrailManager()
        
        # Get sample of records for lineage testing
        sample_records = warehouse.get_sample_records(n=100)
        
        for record in sample_records:
            lineage_result = audit_manager.trace_data_lineage(record.sa2_code)
            
            # Validate complete lineage chain exists
            assert lineage_result.has_complete_lineage, \
                f"Incomplete lineage for SA2 {record.sa2_code}"
            
            # Validate source tracking
            assert len(lineage_result.source_datasets) > 0, \
                f"No source datasets tracked for SA2 {record.sa2_code}"
            
            # Validate transformation tracking
            assert len(lineage_result.transformation_steps) > 0, \
                f"No transformation steps tracked for SA2 {record.sa2_code}"
            
            # Validate timestamps
            assert lineage_result.extraction_timestamp is not None, \
                f"Missing extraction timestamp for SA2 {record.sa2_code}"
            
            assert lineage_result.processing_timestamp is not None, \
                f"Missing processing timestamp for SA2 {record.sa2_code}"
    
    def test_audit_trail_integrity(self):
        """Test integrity of audit trails across the system."""
        from src.utils.audit import AuditTrailManager
        
        audit_manager = AuditTrailManager()
        
        # Validate audit trail completeness
        integrity_report = audit_manager.validate_audit_integrity()
        
        assert integrity_report.all_records_tracked, \
            "Some records missing from audit trail"
        
        assert integrity_report.no_orphaned_entries, \
            "Orphaned entries found in audit trail"
        
        assert integrity_report.chronological_consistency, \
            "Chronological inconsistencies in audit trail"
    
    def test_data_version_tracking(self):
        """Test that data versions are properly tracked and managed."""
        from src.utils.audit import AuditTrailManager
        from src.etl.data_warehouse import DataWarehouse
        
        audit_manager = AuditTrailManager()
        warehouse = DataWarehouse()
        
        # Validate current data version
        current_version = warehouse.get_current_data_version()
        assert current_version is not None, "No current data version found"
        
        # Validate version history
        version_history = audit_manager.get_version_history()
        assert len(version_history) > 0, "No version history available"
        
        # Validate version increments are logical
        for i in range(1, len(version_history)):
            prev_version = version_history[i-1]
            curr_version = version_history[i]
            
            assert curr_version.timestamp > prev_version.timestamp, \
                "Version timestamps not in chronological order"
            
            assert curr_version.version_number > prev_version.version_number, \
                "Version numbers not incrementing properly"
    
    def test_change_impact_analysis(self):
        """Test that changes can be analysed for downstream impact."""
        from src.utils.audit import AuditTrailManager
        
        audit_manager = AuditTrailManager()
        
        # Test impact analysis for hypothetical change
        test_change = {
            'field': 'seifa_irsad_score',
            'sa2_codes': ['101011007', '201011021'],
            'change_type': 'update'
        }
        
        impact_analysis = audit_manager.analyse_change_impact(test_change)
        
        # Validate impact analysis completeness
        assert impact_analysis.affected_records_count >= 0, \
            "Invalid affected records count"
        
        assert len(impact_analysis.downstream_dependencies) >= 0, \
            "Missing downstream dependencies analysis"
        
        assert impact_analysis.quality_impact_score is not None, \
            "Missing quality impact assessment"
