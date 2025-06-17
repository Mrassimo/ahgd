"""
Australian Health Data Pattern Validation Tests

Comprehensive validation of Australian health data patterns including:
- SA2 geographic codes (9-digit pattern validation)
- SEIFA socio-economic indices (1-10 deciles, 800-1200 scores)
- PBS pharmaceutical data (ATC codes, prescription patterns)
- ABS Census 2021 demographic constraints
- Geographic boundary coordinate validation

This module ensures all health data adheres to Australian standards and patterns.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from unittest.mock import Mock, patch

import polars as pl
import pytest
from loguru import logger

from src.data_processing.core import AustralianHealthData


class TestAustralianHealthPatterns:
    """Test suite for Australian health data pattern validation."""
    
    @pytest.fixture
    def health_processor(self, tmp_path):
        """Create a health data processor with temporary directory."""
        return AustralianHealthData(data_dir=tmp_path / "test_data")
    
    @pytest.fixture
    def sample_sa2_codes(self):
        """Sample valid SA2 codes for testing."""
        return [
            "101021007",  # NSW - Sydney
            "201011001",  # VIC - Melbourne
            "301011002",  # QLD - Brisbane
            "401011003",  # SA - Adelaide
            "501011004",  # WA - Perth
            "601011005",  # TAS - Hobart
            "701011006",  # NT - Darwin
            "801011007",  # ACT - Canberra
        ]
    
    @pytest.fixture
    def invalid_sa2_codes(self):
        """Sample invalid SA2 codes for testing."""
        return [
            "12345678",   # Too short
            "1234567890", # Too long
            "901021007",  # Invalid state prefix (9)
            "000021007",  # Invalid state prefix (0)
            "A01021007",  # Contains letters
            "10102100A",  # Letters at end
        ]
    
    @pytest.fixture
    def sample_seifa_data(self):
        """Sample SEIFA data with valid patterns."""
        return pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001", "301011002"],
            "sa2_name_2021": ["Sydney - Harbour", "Melbourne - CBD", "Brisbane - CBD"],
            "irsd_score": [1050, 950, 1100],
            "irsd_decile": [8, 5, 9],
            "irsad_score": [1080, 920, 1120],
            "irsad_decile": [7, 4, 8],
            "ier_score": [1000, 900, 1050],
            "ier_decile": [6, 3, 7],
            "ieo_score": [1150, 850, 1200],
            "ieo_decile": [9, 2, 10],
            "usual_resident_population": [15000, 12000, 18000],
        })
    
    def test_sa2_code_pattern_validation(self, sample_sa2_codes, invalid_sa2_codes):
        """Test SA2 code format validation."""
        # Valid SA2 codes should pass validation
        for sa2_code in sample_sa2_codes:
            assert self._validate_sa2_code(sa2_code), f"Valid SA2 code {sa2_code} failed validation"
        
        # Invalid SA2 codes should fail validation
        for sa2_code in invalid_sa2_codes:
            assert not self._validate_sa2_code(sa2_code), f"Invalid SA2 code {sa2_code} passed validation"
    
    def test_sa2_state_prefix_validation(self):
        """Test SA2 state prefix validation."""
        valid_state_prefixes = {1, 2, 3, 4, 5, 6, 7, 8}  # NSW, VIC, QLD, SA, WA, TAS, NT, ACT
        
        for state_code in valid_state_prefixes:
            sa2_code = f"{state_code}01021007"
            assert self._validate_sa2_code(sa2_code), f"SA2 code with valid state prefix {state_code} failed"
        
        # Test invalid state prefixes
        invalid_prefixes = {0, 9}
        for state_code in invalid_prefixes:
            sa2_code = f"{state_code}01021007"
            assert not self._validate_sa2_code(sa2_code), f"SA2 code with invalid state prefix {state_code} passed"
    
    def test_seifa_score_ranges(self, sample_seifa_data):
        """Test SEIFA score ranges (800-1200)."""
        seifa_columns = ["irsd_score", "irsad_score", "ier_score", "ieo_score"]
        
        for column in seifa_columns:
            scores = sample_seifa_data[column].to_list()
            for score in scores:
                assert 800 <= score <= 1200, f"SEIFA {column} score {score} outside valid range (800-1200)"
    
    def test_seifa_decile_ranges(self, sample_seifa_data):
        """Test SEIFA decile ranges (1-10)."""
        decile_columns = ["irsd_decile", "irsad_decile", "ier_decile", "ieo_decile"]
        
        for column in decile_columns:
            deciles = sample_seifa_data[column].to_list()
            for decile in deciles:
                assert 1 <= decile <= 10, f"SEIFA {column} decile {decile} outside valid range (1-10)"
    
    def test_seifa_score_decile_consistency(self, sample_seifa_data):
        """Test consistency between SEIFA scores and deciles."""
        # For each SEIFA index, higher scores should generally correspond to higher deciles
        # This is a simplified test - in reality, deciles are calculated across all SA2s
        
        indices = ["irsd", "irsad", "ier", "ieo"]
        
        for index in indices:
            score_col = f"{index}_score"
            decile_col = f"{index}_decile"
            
            # Check that scores and deciles are positively correlated
            scores = sample_seifa_data[score_col].to_list()
            deciles = sample_seifa_data[decile_col].to_list()
            
            # Simple correlation check - higher scores should tend to have higher deciles
            high_score_indices = [i for i, score in enumerate(scores) if score > 1000]
            if high_score_indices:
                high_score_deciles = [deciles[i] for i in high_score_indices]
                avg_high_decile = sum(high_score_deciles) / len(high_score_deciles)
                assert avg_high_decile >= 5, f"High {index} scores should correspond to higher deciles"
    
    def test_geographic_coordinate_bounds(self):
        """Test Australian geographic coordinate bounds validation."""
        # Australian mainland bounds: lat -44° to -10°, lon 113° to 154°
        valid_coordinates = [
            (-33.8688, 151.2093),  # Sydney
            (-37.8136, 144.9631),  # Melbourne
            (-27.4698, 153.0251),  # Brisbane
            (-34.9285, 138.6007),  # Adelaide
            (-31.9505, 115.8605),  # Perth
            (-42.8821, 147.3272),  # Hobart
            (-12.4634, 130.8456),  # Darwin
            (-35.2809, 149.1300),  # Canberra
        ]
        
        invalid_coordinates = [
            (-50.0000, 151.2093),  # Too far south
            (-5.0000, 151.2093),   # Too far north
            (-33.8688, 100.0000),  # Too far west
            (-33.8688, 160.0000),  # Too far east
        ]
        
        for lat, lon in valid_coordinates:
            assert self._validate_australian_coordinates(lat, lon), f"Valid coordinates ({lat}, {lon}) failed validation"
        
        for lat, lon in invalid_coordinates:
            assert not self._validate_australian_coordinates(lat, lon), f"Invalid coordinates ({lat}, {lon}) passed validation"
    
    def test_pbs_atc_code_validation(self):
        """Test PBS ATC (Anatomical Therapeutic Chemical) code validation."""
        valid_atc_codes = [
            "A02BC01",  # Omeprazole
            "C09AA02",  # Enalapril
            "N06AB03",  # Fluoxetine
            "J01CA04",  # Amoxicillin
            "M01AE01",  # Ibuprofen
        ]
        
        invalid_atc_codes = [
            "A02BC",     # Too short
            "A02BC001",  # Too long
            "X02BC01",   # Invalid first letter
            "A2BC01",    # Missing digit
            "A02BC1",    # Missing second digit
        ]
        
        for atc_code in valid_atc_codes:
            assert self._validate_atc_code(atc_code), f"Valid ATC code {atc_code} failed validation"
        
        for atc_code in invalid_atc_codes:
            assert not self._validate_atc_code(atc_code), f"Invalid ATC code {atc_code} passed validation"
    
    def test_population_data_consistency(self, sample_seifa_data):
        """Test population data consistency and realistic ranges."""
        populations = sample_seifa_data["usual_resident_population"].to_list()
        
        for population in populations:
            # Population should be positive
            assert population > 0, f"Population {population} should be positive"
            
            # Population should be within realistic SA2 ranges (typically 3,000-25,000)
            assert 100 <= population <= 50000, f"Population {population} outside typical SA2 range"
    
    def test_data_completeness_validation(self, sample_seifa_data):
        """Test data completeness - no null values in critical columns."""
        critical_columns = [
            "sa2_code_2021",
            "sa2_name_2021", 
            "irsd_score",
            "irsd_decile",
            "usual_resident_population"
        ]
        
        for column in critical_columns:
            null_count = sample_seifa_data[column].null_count()
            assert null_count == 0, f"Critical column {column} has {null_count} null values"
    
    def test_data_uniqueness_validation(self, sample_seifa_data):
        """Test data uniqueness - SA2 codes should be unique."""
        sa2_codes = sample_seifa_data["sa2_code_2021"].to_list()
        unique_codes = set(sa2_codes)
        
        assert len(sa2_codes) == len(unique_codes), "Duplicate SA2 codes found in dataset"
    
    def test_temporal_data_consistency(self):
        """Test temporal data consistency for time-series health data."""
        # Mock time-series health data
        sample_time_series = pl.DataFrame({
            "sa2_code_2021": ["101021007"] * 12,
            "year_month": [f"2023-{i:02d}" for i in range(1, 13)],
            "health_metric": [100, 105, 110, 108, 115, 120, 118, 125, 130, 128, 135, 140],
            "population_estimate": [15000] * 12,
        })
        
        # Check temporal ordering
        months = sample_time_series["year_month"].to_list()
        assert months == sorted(months), "Time-series data should be chronologically ordered"
        
        # Check population consistency (should not vary dramatically month-to-month)
        populations = sample_time_series["population_estimate"].to_list()
        pop_variance = max(populations) - min(populations)
        assert pop_variance <= max(populations) * 0.1, "Population estimates vary too much within year"
    
    def test_cross_dataset_sa2_consistency(self):
        """Test SA2 code consistency across multiple datasets."""
        # Mock datasets that should have consistent SA2 codes
        seifa_sa2s = {"101021007", "201011001", "301011002"}
        health_sa2s = {"101021007", "201011001", "301011002", "401011003"}
        boundary_sa2s = {"101021007", "201011001", "301011002", "401011003", "501011004"}
        
        # SEIFA codes should be subset of health codes
        assert seifa_sa2s.issubset(health_sa2s), "SEIFA SA2 codes should be subset of health SA2 codes"
        
        # Health codes should be subset of boundary codes
        assert health_sa2s.issubset(boundary_sa2s), "Health SA2 codes should be subset of boundary SA2 codes"
    
    def test_data_quality_metrics_calculation(self, sample_seifa_data):
        """Test calculation of data quality metrics."""
        metrics = self._calculate_data_quality_metrics(sample_seifa_data)
        
        # Completeness should be 100% for sample data
        assert metrics["completeness"]["overall"] == 100.0, "Sample data should have 100% completeness"
        
        # Validity should be 100% for sample data
        assert metrics["validity"]["sa2_codes"] == 100.0, "Sample SA2 codes should have 100% validity"
        
        # Uniqueness should be 100% for SA2 codes
        assert metrics["uniqueness"]["sa2_codes"] == 100.0, "Sample SA2 codes should be 100% unique"
    
    def test_australian_postcode_validation(self):
        """Test Australian postcode validation patterns."""
        valid_postcodes = [
            "2000",  # NSW - Sydney CBD
            "3000",  # VIC - Melbourne CBD
            "4000",  # QLD - Brisbane CBD
            "5000",  # SA - Adelaide CBD
            "6000",  # WA - Perth CBD
            "7000",  # TAS - Hobart CBD
            "0800",  # NT - Darwin
            "2600",  # ACT - Canberra
        ]
        
        invalid_postcodes = [
            "1000",   # Invalid range
            "12345",  # Too long
            "200",    # Too short
            "ABCD",   # Not numeric
        ]
        
        for postcode in valid_postcodes:
            assert self._validate_australian_postcode(postcode), f"Valid postcode {postcode} failed validation"
        
        for postcode in invalid_postcodes:
            assert not self._validate_australian_postcode(postcode), f"Invalid postcode {postcode} passed validation"
    
    # Helper methods for validation
    
    def _validate_sa2_code(self, sa2_code: str) -> bool:
        """Validate SA2 code format and state prefix."""
        if not isinstance(sa2_code, str) or len(sa2_code) != 9:
            return False
        
        if not sa2_code.isdigit():
            return False
        
        # Check valid state prefix (1-8)
        state_prefix = int(sa2_code[0])
        return 1 <= state_prefix <= 8
    
    def _validate_australian_coordinates(self, latitude: float, longitude: float) -> bool:
        """Validate coordinates are within Australian bounds."""
        # Australian mainland bounds
        return (-44.0 <= latitude <= -10.0) and (113.0 <= longitude <= 154.0)
    
    def _validate_atc_code(self, atc_code: str) -> bool:
        """Validate ATC (Anatomical Therapeutic Chemical) code format."""
        # ATC codes: A02BC01 format (Letter-Digit-Digit-Letter-Letter-Digit-Digit)
        pattern = r'^[A-N][0-9][0-9][A-Z][A-Z][0-9][0-9]$'
        return bool(re.match(pattern, atc_code))
    
    def _validate_australian_postcode(self, postcode: str) -> bool:
        """Validate Australian postcode format."""
        if not isinstance(postcode, str) or len(postcode) != 4:
            return False
        
        if not postcode.isdigit():
            return False
        
        # Australian postcodes range from 0800 to 9999
        postcode_int = int(postcode)
        return 800 <= postcode_int <= 9999
    
    def _calculate_data_quality_metrics(self, df: pl.DataFrame) -> Dict:
        """Calculate comprehensive data quality metrics."""
        total_rows = df.height
        total_cells = df.width * df.height
        
        # Completeness metrics
        null_cells = sum(df[col].null_count() for col in df.columns)
        completeness = ((total_cells - null_cells) / total_cells) * 100
        
        # Validity metrics for SA2 codes
        sa2_codes = df["sa2_code_2021"].to_list()
        valid_sa2_count = sum(1 for code in sa2_codes if self._validate_sa2_code(code))
        sa2_validity = (valid_sa2_count / len(sa2_codes)) * 100
        
        # Uniqueness metrics for SA2 codes
        unique_sa2_count = df["sa2_code_2021"].n_unique()
        sa2_uniqueness = (unique_sa2_count / len(sa2_codes)) * 100
        
        return {
            "completeness": {
                "overall": completeness,
                "null_cells": null_cells,
                "total_cells": total_cells,
            },
            "validity": {
                "sa2_codes": sa2_validity,
                "valid_count": valid_sa2_count,
                "total_count": len(sa2_codes),
            },
            "uniqueness": {
                "sa2_codes": sa2_uniqueness,
                "unique_count": unique_sa2_count,
                "total_count": len(sa2_codes),
            }
        }


class TestHealthDataIntegrity:
    """Advanced health data integrity tests."""
    
    def test_demographic_data_consistency(self):
        """Test demographic data consistency across different sources."""
        # Mock census and health data
        census_data = pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001"],
            "total_population_census": [15234, 12876],
            "median_age": [35, 42],
            "median_income": [75000, 68000],
        })
        
        health_data = pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001"],
            "estimated_population": [15200, 12850],
            "health_service_visits": [2500, 2100],
        })
        
        # Join datasets
        combined = census_data.join(health_data, on="sa2_code_2021")
        
        # Check population consistency (should be within 5%)
        for row in combined.iter_rows(named=True):
            census_pop = row["total_population_census"]
            health_pop = row["estimated_population"]
            difference_pct = abs(census_pop - health_pop) / census_pop * 100
            
            assert difference_pct <= 5.0, f"Population difference {difference_pct:.2f}% exceeds 5% threshold"
    
    def test_longitudinal_data_trends(self):
        """Test longitudinal health data for realistic trends."""
        # Mock longitudinal health data
        longitudinal_data = pl.DataFrame({
            "sa2_code_2021": ["101021007"] * 24,
            "year_month": [f"2022-{i:02d}" if i <= 12 else f"2023-{i-12:02d}" for i in range(1, 25)],
            "health_metric": [100 + i + (i % 3) for i in range(24)],  # Gradual increase with variation
            "population": [15000 + (i * 10) for i in range(24)],  # Gradual population growth
        })
        
        # Check for unrealistic jumps in health metrics
        metrics = longitudinal_data["health_metric"].to_list()
        for i in range(1, len(metrics)):
            change_pct = abs(metrics[i] - metrics[i-1]) / metrics[i-1] * 100
            assert change_pct <= 20.0, f"Health metric change {change_pct:.2f}% exceeds 20% threshold"
    
    def test_spatial_data_consistency(self):
        """Test spatial data consistency and geographic relationships."""
        # Mock spatial health data
        spatial_data = pl.DataFrame({
            "sa2_code_2021": ["101021007", "101021008", "201011001"],
            "latitude": [-33.8688, -33.8700, -37.8136],
            "longitude": [151.2093, 151.2100, 144.9631],
            "health_metric": [105, 108, 95],
            "distance_to_hospital_km": [2.5, 2.8, 1.2],
        })
        
        # Check spatial autocorrelation - nearby areas should have similar metrics
        for i in range(len(spatial_data)):
            for j in range(i + 1, len(spatial_data)):
                row_i = spatial_data.row(i, named=True)
                row_j = spatial_data.row(j, named=True)
                
                # Calculate distance
                lat_diff = row_i["latitude"] - row_j["latitude"]
                lon_diff = row_i["longitude"] - row_j["longitude"]
                distance = (lat_diff**2 + lon_diff**2)**0.5
                
                # If areas are close (same first 6 digits of SA2), metrics should be similar
                if row_i["sa2_code_2021"][:6] == row_j["sa2_code_2021"][:6]:
                    metric_diff = abs(row_i["health_metric"] - row_j["health_metric"])
                    assert metric_diff <= 20, f"Nearby areas have very different health metrics: {metric_diff}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])