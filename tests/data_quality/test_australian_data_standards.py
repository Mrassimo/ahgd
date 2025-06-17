"""
Australian Health Data Standards Validation Tests

Comprehensive testing suite for Australian health data compliance including:
- SA2 code format validation (9-digit Australian statistical areas)
- SEIFA index compliance (2021 methodology, 1-10 deciles, 800-1200 scores)
- PBS prescription data validation (ATC codes, dispensing patterns)
- ABS Census data format validation (2021 data structures)
- Geographic boundary coordinate validation (Australian bounds)

This test suite ensures all Australian health data meets government standards
and follows established patterns for data quality and integrity.
"""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import Mock, patch

import polars as pl
import numpy as np
from loguru import logger

from tests.data_quality.validators.australian_health_validators import (
    AustralianHealthDataValidator,
    DataQualityMetricsCalculator
)


class TestAustralianDataStandardsValidation:
    """Test suite for Australian health data standards validation."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return AustralianHealthDataValidator()
    
    @pytest.fixture
    def quality_calculator(self, validator):
        """Create quality metrics calculator."""
        return DataQualityMetricsCalculator(validator)
    
    @pytest.fixture
    def valid_sa2_codes(self):
        """Valid SA2 codes representing all Australian states/territories."""
        return [
            "101021007",  # NSW - Sydney
            "102031008",  # NSW - Regional
            "201011001",  # VIC - Melbourne
            "202051009",  # VIC - Regional
            "301011002",  # QLD - Brisbane
            "302081010",  # QLD - Regional
            "401011003",  # SA - Adelaide
            "402041011",  # SA - Regional
            "501011004",  # WA - Perth
            "502071012",  # WA - Regional
            "601011005",  # TAS - Hobart
            "602021013",  # TAS - Regional
            "701011006",  # NT - Darwin
            "702031014",  # NT - Regional
            "801011007",  # ACT - Canberra
            "802021015",  # ACT - Outer
        ]
    
    @pytest.fixture
    def invalid_sa2_codes(self):
        """Invalid SA2 codes for testing validation."""
        return [
            "12345678",   # Too short
            "1234567890", # Too long
            "901021007",  # Invalid state prefix (9)
            "000021007",  # Invalid state prefix (0)
            "A01021007",  # Contains letters
            "10102100A",  # Letters at end
            "",           # Empty string
            None,         # None value
            "1.1021007",  # Contains decimal point
        ]
    
    @pytest.fixture
    def valid_seifa_2021_data(self):
        """Valid SEIFA 2021 data conforming to ABS methodology."""
        return pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001", "301011002", "401011003", "501011004"],
            "sa2_name_2021": [
                "Sydney - Harbour - South",
                "Melbourne (C) - CBD",
                "Brisbane (C) - Inner",
                "Adelaide (C) - North East",
                "Perth (C) - Inner"
            ],
            "state_name_2021": ["New South Wales", "Victoria", "Queensland", "South Australia", "Western Australia"],
            "irsd_score": [1150, 950, 1100, 980, 1080],
            "irsd_decile": [9, 5, 8, 6, 7],
            "irsd_rank": [156, 1823, 445, 1456, 678],
            "irsad_score": [1180, 920, 1120, 950, 1060],
            "irsad_decile": [10, 4, 8, 5, 7],
            "irsad_rank": [89, 1945, 398, 1678, 789],
            "ier_score": [1000, 900, 1050, 920, 980],
            "ier_decile": [6, 3, 7, 4, 5],
            "ier_rank": [567, 2234, 423, 1987, 1234],
            "ieo_score": [1200, 850, 1180, 880, 1000],
            "ieo_decile": [10, 2, 9, 3, 6],
            "ieo_rank": [45, 2456, 123, 2123, 1000],
            "usual_resident_population": [15000, 12000, 18000, 8500, 22000],
            "tot_p_m": [7500, 6000, 9000, 4250, 11000],
            "tot_p_f": [7500, 6000, 9000, 4250, 11000],
            "tot_p_p": [15000, 12000, 18000, 8500, 22000],
        })
    
    @pytest.fixture
    def invalid_seifa_data(self):
        """Invalid SEIFA data for testing validation."""
        return pl.DataFrame({
            "sa2_code_2021": ["901021007", "101021007", "201011001"],  # First is invalid
            "sa2_name_2021": ["Invalid Area", "Sydney", "Melbourne"],
            "irsd_score": [750, 1050, 1300],  # First and last outside range
            "irsd_decile": [0, 8, 11],        # First and last outside range
            "irsad_score": [1080, None, 1120], # Contains null
            "irsad_decile": [7, 5, 8],
            "ier_score": [1000, 950, 1050],
            "ier_decile": [6, 5, 7],
            "ieo_score": [1150, 900, 1180],
            "ieo_decile": [9, 4, 8],
            "usual_resident_population": [-100, 15000, 75000],  # Negative and too large
        })
    
    def test_sa2_code_format_validation_valid_codes(self, validator, valid_sa2_codes):
        """Test SA2 code format validation with valid codes."""
        for sa2_code in valid_sa2_codes:
            result = validator.validate_sa2_code(sa2_code)
            
            assert result["valid"] is True, f"Valid SA2 code {sa2_code} failed validation"
            assert result["code"] == sa2_code
            assert "state" in result
            assert "state_code" in result
            assert 1 <= result["state_code"] <= 8, f"Invalid state code {result['state_code']}"
            assert len(result.get("errors", [])) == 0
    
    def test_sa2_code_format_validation_invalid_codes(self, validator, invalid_sa2_codes):
        """Test SA2 code format validation with invalid codes."""
        for sa2_code in invalid_sa2_codes:
            result = validator.validate_sa2_code(sa2_code)
            
            assert result["valid"] is False, f"Invalid SA2 code {sa2_code} passed validation"
            assert len(result["errors"]) > 0, f"No errors reported for invalid SA2 code {sa2_code}"
    
    def test_sa2_state_prefix_validation(self, validator):
        """Test SA2 state prefix validation for all Australian states."""
        state_mappings = {
            1: "NSW", 2: "VIC", 3: "QLD", 4: "SA",
            5: "WA", 6: "TAS", 7: "NT", 8: "ACT"
        }
        
        for state_code, state_name in state_mappings.items():
            sa2_code = f"{state_code}01021007"
            result = validator.validate_sa2_code(sa2_code)
            
            assert result["valid"] is True, f"Valid state prefix {state_code} failed validation"
            assert result["state"] == state_name, f"Expected {state_name}, got {result.get('state')}"
            assert result["state_code"] == state_code
    
    def test_seifa_2021_data_validation_valid_data(self, validator, valid_seifa_2021_data):
        """Test SEIFA 2021 data validation with valid data."""
        for row in valid_seifa_2021_data.iter_rows(named=True):
            result = validator.validate_seifa_2021_data(row)
            
            assert result["valid"] is True, f"Valid SEIFA record failed validation: {result['errors']}"
            assert len(result["errors"]) == 0, f"Errors found in valid data: {result['errors']}"
    
    def test_seifa_2021_data_validation_invalid_data(self, validator, invalid_seifa_data):
        """Test SEIFA 2021 data validation with invalid data."""
        errors_found = []
        
        for i, row in enumerate(invalid_seifa_data.iter_rows(named=True)):
            result = validator.validate_seifa_2021_data(row)
            
            if not result["valid"]:
                errors_found.append({
                    "row": i,
                    "errors": result["errors"],
                    "warnings": result.get("warnings", [])
                })
        
        assert len(errors_found) > 0, "No validation errors found in intentionally invalid data"
        
        # Check specific expected errors
        all_errors = [error for error_set in errors_found for error in error_set["errors"]]
        error_text = " ".join(all_errors)
        
        assert "outside valid range" in error_text or "Invalid state code" in error_text, \
            "Expected range validation errors not found"
    
    def test_seifa_score_ranges_validation(self, validator):
        """Test SEIFA score ranges (800-1200) validation."""
        test_cases = [
            (800, True),   # Minimum valid
            (1000, True),  # Mid-range valid
            (1200, True),  # Maximum valid
            (799, False),  # Below minimum
            (1201, False), # Above maximum
            (750, False),  # Well below
            (1300, False), # Well above
        ]
        
        for score, should_be_valid in test_cases:
            test_record = {
                "sa2_code_2021": "101021007",
                "irsd_score": score,
                "irsd_decile": 5,
                "irsad_score": 1000,
                "irsad_decile": 5,
                "ier_score": 1000,
                "ier_decile": 5,
                "ieo_score": 1000,
                "ieo_decile": 5,
                "usual_resident_population": 15000
            }
            
            result = validator.validate_seifa_2021_data(test_record)
            
            if should_be_valid:
                assert result["valid"] is True, f"Valid SEIFA score {score} failed validation"
            else:
                assert result["valid"] is False, f"Invalid SEIFA score {score} passed validation"
    
    def test_seifa_decile_ranges_validation(self, validator):
        """Test SEIFA decile ranges (1-10) validation."""
        test_cases = [
            (1, True),   # Minimum valid
            (5, True),   # Mid-range valid
            (10, True),  # Maximum valid
            (0, False),  # Below minimum
            (11, False), # Above maximum
            (-1, False), # Negative
        ]
        
        for decile, should_be_valid in test_cases:
            test_record = {
                "sa2_code_2021": "101021007",
                "irsd_score": 1000,
                "irsd_decile": decile,
                "irsad_score": 1000,
                "irsad_decile": 5,
                "ier_score": 1000,
                "ier_decile": 5,
                "ieo_score": 1000,
                "ieo_decile": 5,
                "usual_resident_population": 15000
            }
            
            result = validator.validate_seifa_2021_data(test_record)
            
            if should_be_valid:
                assert result["valid"] is True, f"Valid SEIFA decile {decile} failed validation"
            else:
                assert result["valid"] is False, f"Invalid SEIFA decile {decile} passed validation"
    
    def test_australian_coordinate_bounds_validation(self, validator):
        """Test Australian geographic coordinate bounds validation."""
        valid_coordinates = [
            (-33.8688, 151.2093),  # Sydney
            (-37.8136, 144.9631),  # Melbourne
            (-27.4698, 153.0251),  # Brisbane
            (-34.9285, 138.6007),  # Adelaide
            (-31.9505, 115.8605),  # Perth
            (-42.8821, 147.3272),  # Hobart
            (-12.4634, 130.8456),  # Darwin
            (-35.2809, 149.1300),  # Canberra
            (-43.9999, 114.0001),  # Edge cases - just within bounds
            (-10.0001, 153.9999),  # Edge cases - just within bounds
        ]
        
        invalid_coordinates = [
            (-50.0000, 151.2093),  # Too far south
            (-5.0000, 151.2093),   # Too far north
            (-33.8688, 100.0000),  # Too far west
            (-33.8688, 160.0000),  # Too far east
            (-44.0001, 145.0000),  # Just outside south bound
            (-9.9999, 145.0000),   # Just outside north bound
            (-35.0000, 112.9999),  # Just outside west bound
            (-35.0000, 154.0001),  # Just outside east bound
        ]
        
        # Test valid coordinates
        for lat, lon in valid_coordinates:
            result = validator.validate_australian_coordinates(lat, lon)
            assert result["valid"] is True, f"Valid coordinates ({lat}, {lon}) failed validation: {result['errors']}"
        
        # Test invalid coordinates
        for lat, lon in invalid_coordinates:
            result = validator.validate_australian_coordinates(lat, lon)
            assert result["valid"] is False, f"Invalid coordinates ({lat}, {lon}) passed validation"
    
    def test_pbs_atc_code_validation(self, validator):
        """Test PBS ATC (Anatomical Therapeutic Chemical) code validation."""
        valid_atc_codes = [
            "A02BC01",  # Omeprazole (Proton pump inhibitor)
            "C09AA02",  # Enalapril (ACE inhibitor)
            "N06AB03",  # Fluoxetine (SSRI)
            "J01CA04",  # Amoxicillin (Antibiotic)
            "M01AE01",  # Ibuprofen (NSAID)
            "B01AC06",  # Aspirin (Antiplatelet)
            "A10BA02",  # Metformin (Diabetes)
            "C07AB07",  # Bisoprolol (Beta blocker)
        ]
        
        invalid_atc_codes = [
            "A02BC",     # Too short
            "A02BC001",  # Too long
            "X02BC01",   # Invalid anatomical group
            "A2BC01",    # Missing digit
            "A02BC1",    # Missing digit
            "a02bc01",   # Lowercase
            "A02BC0A",   # Letter where digit expected
            "A02B01",    # Missing character
            "",          # Empty string
        ]
        
        # Test valid ATC codes
        for atc_code in valid_atc_codes:
            result = validator.validate_atc_code(atc_code)
            assert result["valid"] is True, f"Valid ATC code {atc_code} failed validation: {result['errors']}"
            assert "anatomical_group" in result
            assert "therapeutic_group" in result
            assert "chemical_group" in result
            assert "substance" in result
        
        # Test invalid ATC codes
        for atc_code in invalid_atc_codes:
            result = validator.validate_atc_code(atc_code)
            assert result["valid"] is False, f"Invalid ATC code {atc_code} passed validation"
    
    def test_australian_postcode_validation(self, validator):
        """Test Australian postcode validation with state consistency."""
        valid_postcodes = [
            ("2000", "NSW"),  # Sydney CBD
            ("3000", "VIC"),  # Melbourne CBD
            ("4000", "QLD"),  # Brisbane CBD
            ("5000", "SA"),   # Adelaide CBD
            ("6000", "WA"),   # Perth CBD
            ("7000", "TAS"),  # Hobart CBD
            ("0800", "NT"),   # Darwin
            ("2600", "ACT"),  # Canberra
            ("9999", "QLD"),  # Maximum postcode
            ("0200", "ACT"),  # Minimum postcode
        ]
        
        invalid_postcodes = [
            ("1000", None),   # Invalid range
            ("12345", None),  # Too long
            ("200", None),    # Too short
            ("ABCD", None),   # Not numeric
            ("2000", "VIC"),  # Valid postcode, wrong state
            ("0199", None),   # Below minimum
        ]
        
        # Test valid postcodes
        for postcode, state in valid_postcodes:
            result = validator.validate_australian_postcode(postcode, state)
            assert result["valid"] is True, f"Valid postcode {postcode} for {state} failed validation: {result['errors']}"
            
            if state:
                # Check state consistency
                inferred_state = result.get("inferred_state")
                assert inferred_state == state, f"State mismatch: expected {state}, inferred {inferred_state}"
        
        # Test invalid postcodes
        for postcode, state in invalid_postcodes:
            result = validator.validate_australian_postcode(postcode, state)
            assert result["valid"] is False, f"Invalid postcode {postcode} passed validation"
    
    def test_abs_census_2021_data_validation(self, validator):
        """Test ABS Census 2021 data structure validation."""
        valid_census_data = {
            "sa1_code_2021": "10102100701",  # 11-digit SA1 code
            "sa2_code_2021": "101021007",    # 9-digit SA2 code
            "tot_p_m": 7500,     # Male population
            "tot_p_f": 7500,     # Female population
            "tot_p_p": 15000,    # Total population
            "age_0_4_yr_m": 500,
            "age_0_4_yr_f": 480,
            "age_5_9_yr_m": 520,
            "age_5_9_yr_f": 510,
            "median_age_persons": 35,
            "median_mortgage_repay_monthly": 2400,
            "median_tot_prsnl_inc_weekly": 1200,
        }
        
        invalid_census_data = {
            "sa1_code_2021": "101021007",    # Wrong length (SA2 instead of SA1)
            "sa2_code_2021": "901021007",    # Invalid state prefix
            "tot_p_m": 7500,
            "tot_p_f": 7500,
            "tot_p_p": 14000,    # Doesn't match male + female
            "age_0_4_yr_m": 500,
            "age_0_4_yr_f": 480,
        }
        
        # Test valid census data
        result = validator.validate_census_2021_data(valid_census_data)
        assert result["valid"] is True, f"Valid census data failed validation: {result['errors']}"
        
        # Test invalid census data
        result = validator.validate_census_2021_data(invalid_census_data)
        assert result["valid"] is False, f"Invalid census data passed validation"
        assert len(result["errors"]) > 0
    
    def test_health_service_data_validation(self, validator):
        """Test health service utilisation data validation."""
        valid_health_data = {
            "service_date": "2023-06-15",
            "provider_number": "123456",
            "schedule_fee": 85.50,
            "benefit_paid": 68.40,
            "patient_contribution": 17.10,
            "item_number": "23",
            "patient_id": "P123456789",
        }
        
        invalid_health_data = {
            "service_date": "2025-12-31",  # Future date
            "provider_number": "12345",    # Too short
            "schedule_fee": -85.50,        # Negative amount
            "benefit_paid": 68.40,
            "patient_contribution": 17.10,
        }
        
        # Test valid health data
        result = validator.validate_health_service_data(valid_health_data)
        assert result["valid"] is True, f"Valid health data failed validation: {result['errors']}"
        
        # Test invalid health data
        result = validator.validate_health_service_data(invalid_health_data)
        assert result["valid"] is False, f"Invalid health data passed validation"
        assert len(result["errors"]) > 0
    
    def test_population_data_consistency_validation(self, validator, valid_seifa_2021_data):
        """Test population data consistency and realistic ranges."""
        populations = valid_seifa_2021_data["usual_resident_population"].to_list()
        
        for population in populations:
            # Test population validation within SEIFA record
            test_record = {
                "sa2_code_2021": "101021007",
                "irsd_score": 1000,
                "irsd_decile": 5,
                "irsad_score": 1000,
                "irsad_decile": 5,
                "ier_score": 1000,
                "ier_decile": 5,
                "ieo_score": 1000,
                "ieo_decile": 5,
                "usual_resident_population": population
            }
            
            result = validator.validate_seifa_2021_data(test_record)
            assert result["valid"] is True, f"Valid population {population} failed validation"
        
        # Test invalid populations
        invalid_populations = [-100, 0, 100000, 500000]
        
        for population in invalid_populations:
            test_record = {
                "sa2_code_2021": "101021007",
                "irsd_score": 1000,
                "irsd_decile": 5,
                "irsad_score": 1000,
                "irsad_decile": 5,
                "ier_score": 1000,
                "ier_decile": 5,
                "ieo_score": 1000,
                "ieo_decile": 5,
                "usual_resident_population": population
            }
            
            result = validator.validate_seifa_2021_data(test_record)
            
            if population <= 0:
                assert result["valid"] is False, f"Invalid population {population} passed validation"
            elif population > 50000:
                # Should generate warnings but might still be valid
                assert len(result.get("warnings", [])) > 0, f"Large population {population} didn't generate warnings"
    
    def test_data_quality_metrics_calculation(self, quality_calculator, valid_seifa_2021_data):
        """Test comprehensive data quality metrics calculation."""
        # Test completeness metrics
        completeness_metrics = quality_calculator.calculate_completeness_metrics(valid_seifa_2021_data)
        
        assert len(completeness_metrics) > 0, "No completeness metrics calculated"
        
        # Find overall completeness metric
        overall_metric = next((m for m in completeness_metrics if m.metric_name == "overall_completeness"), None)
        assert overall_metric is not None, "Overall completeness metric not found"
        assert overall_metric.value == 100.0, f"Expected 100% completeness, got {overall_metric.value}"
        assert overall_metric.passed is True, "Overall completeness should pass"
        
        # Test validity metrics
        validity_metrics = quality_calculator.calculate_validity_metrics(valid_seifa_2021_data, "seifa")
        
        assert len(validity_metrics) > 0, "No validity metrics calculated"
        
        # Find SA2 validity metric
        sa2_metric = next((m for m in validity_metrics if m.metric_name == "sa2_code_validity"), None)
        assert sa2_metric is not None, "SA2 code validity metric not found"
        assert sa2_metric.value == 100.0, f"Expected 100% SA2 validity, got {sa2_metric.value}"
        assert sa2_metric.passed is True, "SA2 validity should pass"
        
        # Test consistency metrics
        consistency_metrics = quality_calculator.calculate_consistency_metrics(valid_seifa_2021_data)
        
        assert len(consistency_metrics) > 0, "No consistency metrics calculated"
        
        # Test uniqueness metrics
        uniqueness_metrics = quality_calculator.calculate_uniqueness_metrics(valid_seifa_2021_data)
        
        assert len(uniqueness_metrics) > 0, "No uniqueness metrics calculated"
        
        # Find SA2 uniqueness metric
        sa2_unique_metric = next((m for m in uniqueness_metrics if m.metric_name == "sa2_code_uniqueness"), None)
        assert sa2_unique_metric is not None, "SA2 uniqueness metric not found"
        assert sa2_unique_metric.value == 100.0, f"Expected 100% SA2 uniqueness, got {sa2_unique_metric.value}"
    
    def test_comprehensive_quality_report_generation(self, quality_calculator, valid_seifa_2021_data):
        """Test comprehensive quality report generation."""
        report = quality_calculator.generate_quality_report(
            df=valid_seifa_2021_data,
            dataset_name="test_seifa_dataset",
            layer="bronze",
            data_type="seifa"
        )
        
        # Validate report structure
        assert report.dataset_name == "test_seifa_dataset"
        assert report.layer == "bronze"
        assert report.overall_score > 0
        assert report.quality_classification in ["Excellent", "Good", "Acceptable", "Poor", "Unacceptable"]
        assert len(report.metrics) > 0
        assert len(report.recommendations) > 0
        assert report.metadata is not None
        
        # Check metadata
        assert report.metadata["dataset_rows"] == valid_seifa_2021_data.height
        assert report.metadata["dataset_columns"] == valid_seifa_2021_data.width
        assert report.metadata["data_type"] == "seifa"
        assert report.metadata["layer"] == "bronze"
        
        # For valid data, should have high quality score
        assert report.overall_score >= 90.0, f"Expected high quality score for valid data, got {report.overall_score}"
        assert report.quality_classification in ["Excellent", "Good"], \
            f"Expected excellent/good quality classification, got {report.quality_classification}"
    
    def test_australian_health_data_patterns_end_to_end(self, validator, quality_calculator):
        """End-to-end test of Australian health data pattern validation."""
        # Create comprehensive test dataset
        test_data = pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001", "301011002", "401011003"],
            "sa2_name_2021": ["Sydney Harbour", "Melbourne CBD", "Brisbane Inner", "Adelaide North"],
            "state_name_2021": ["New South Wales", "Victoria", "Queensland", "South Australia"],
            "irsd_score": [1050, 950, 1100, 980],
            "irsd_decile": [8, 5, 9, 6],
            "irsad_score": [1080, 920, 1120, 950],
            "irsad_decile": [7, 4, 8, 5],
            "ier_score": [1000, 900, 1050, 920],
            "ier_decile": [6, 3, 7, 4],
            "ieo_score": [1150, 850, 1180, 880],
            "ieo_decile": [9, 2, 10, 3],
            "usual_resident_population": [15000, 12000, 18000, 8500],
            "latitude": [-33.8688, -37.8136, -27.4698, -34.9285],
            "longitude": [151.2093, 144.9631, 153.0251, 138.6007],
            "postcode": ["2000", "3000", "4000", "5000"],
        })
        
        # Generate comprehensive quality report
        quality_report = quality_calculator.generate_quality_report(
            df=test_data,
            dataset_name="comprehensive_test",
            layer="bronze",
            data_type="seifa"
        )
        
        # Validate end-to-end results
        assert quality_report.overall_score >= 95.0, \
            f"Expected excellent quality for well-formed test data, got {quality_report.overall_score}"
        
        assert quality_report.quality_classification == "Excellent", \
            f"Expected Excellent classification, got {quality_report.quality_classification}"
        
        # Check that all major validation categories are covered
        metric_names = [metric.metric_name for metric in quality_report.metrics]
        
        assert any("completeness" in name for name in metric_names), "Missing completeness metrics"
        assert any("validity" in name for name in metric_names), "Missing validity metrics"
        assert any("consistency" in name for name in metric_names), "Missing consistency metrics"
        assert any("uniqueness" in name for name in metric_names), "Missing uniqueness metrics"
        
        # Check specific Australian data validations
        assert any("sa2_code" in name for name in metric_names), "Missing SA2 code validation"
        assert any("seifa" in name.lower() for name in metric_names), "Missing SEIFA validation"
        
        # Verify most metrics pass
        passed_metrics = sum(1 for metric in quality_report.metrics if metric.passed)
        total_metrics = len(quality_report.metrics)
        pass_rate = passed_metrics / total_metrics
        
        assert pass_rate >= 0.95, f"Expected >95% pass rate, got {pass_rate:.2%}"
        
        logger.info(f"End-to-end test completed successfully:")
        logger.info(f"  Overall Score: {quality_report.overall_score:.2f}")
        logger.info(f"  Classification: {quality_report.quality_classification}")
        logger.info(f"  Total Metrics: {total_metrics}")
        logger.info(f"  Passed Metrics: {passed_metrics}")
        logger.info(f"  Pass Rate: {pass_rate:.2%}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])