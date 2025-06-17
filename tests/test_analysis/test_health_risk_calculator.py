"""
Comprehensive unit tests for Health Risk Calculator.

Tests multi-factor health risk assessment combining:
- SEIFA socio-economic indices (4 indices with weighted contributions)
- Health utilisation patterns (prescription rates, chronic medications)
- Geographic accessibility factors
- Population demographics and risk stratification
- Composite risk score calculations and validation

Validates risk assessment algorithms with known input/output pairs.
"""

import pytest
import polars as pl
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import time
from datetime import datetime, date, timedelta

from src.analysis.risk.health_risk_calculator import HealthRiskCalculator


class TestHealthRiskCalculator:
    """Comprehensive test suite for health risk calculator."""
    
    def test_risk_calculator_initialization(self, mock_data_paths):
        """Test risk calculator initializes with proper configuration."""
        calculator = HealthRiskCalculator(data_dir=mock_data_paths["processed_dir"])
        
        assert calculator.data_dir == mock_data_paths["processed_dir"]
        assert calculator.seifa_data is None
        assert calculator.health_data is None
        assert calculator.boundary_data is None
        assert calculator.integrated_data is None
        
        # Verify SEIFA weights sum appropriately
        total_weight = sum(calculator.SEIFA_WEIGHTS.values())
        assert abs(total_weight - 1.0) < 0.01, "SEIFA weights should sum to approximately 1.0"
        
        # Verify health risk factors sum appropriately
        total_health_weight = sum(calculator.HEALTH_RISK_FACTORS.values())
        assert abs(total_health_weight - 1.0) < 0.01, "Health risk factors should sum to approximately 1.0"
    
    def test_risk_calculator_default_directory(self):
        """Test risk calculator with default data directory."""
        calculator = HealthRiskCalculator()
        assert calculator.data_dir.name == "processed"
    
    def test_load_processed_data_success(self, mock_seifa_data, mock_health_data, mock_boundary_data, mock_data_paths):
        """Test successful loading of all processed data files."""
        calculator = HealthRiskCalculator(data_dir=mock_data_paths["processed_dir"])
        
        # Create mock data files
        seifa_df = mock_seifa_data(num_areas=100)
        health_df = mock_health_data(num_records=500, num_sa2_areas=100)
        boundary_df = mock_boundary_data(num_areas=100)
        
        # Save mock files
        seifa_path = mock_data_paths["processed_dir"] / "seifa_2021_sa2.csv"
        health_path = mock_data_paths["processed_dir"] / "health_data_processed.csv"
        boundary_path = mock_data_paths["processed_dir"] / "sa2_boundaries_processed.csv"
        
        seifa_df.write_csv(seifa_path)
        health_df.write_csv(health_path)
        boundary_df.write_csv(boundary_path)
        
        # Test loading
        result = calculator.load_processed_data()
        
        assert result is True
        assert calculator.seifa_data is not None
        assert calculator.health_data is not None
        assert calculator.boundary_data is not None
        
        assert len(calculator.seifa_data) == 100
        assert len(calculator.health_data) == 500
        assert len(calculator.boundary_data) == 100
    
    def test_load_processed_data_missing_files(self, mock_data_paths):
        """Test loading behavior when data files are missing."""
        calculator = HealthRiskCalculator(data_dir=mock_data_paths["processed_dir"])
        
        # No files exist
        result = calculator.load_processed_data()
        
        # Should handle missing files gracefully
        assert result is False
    
    def test_load_processed_data_with_mock_generation(self, mock_data_paths):
        """Test loading with automatic mock data generation when files missing."""
        calculator = HealthRiskCalculator(data_dir=mock_data_paths["processed_dir"])
        
        # Create only SEIFA and boundary data, health will be mocked
        seifa_df = pl.DataFrame({
            "sa2_code_2021": ["123456789", "234567890"],
            "irsd_decile": [3, 7],
            "irsad_decile": [4, 6]
        })
        boundary_df = pl.DataFrame({
            "sa2_code_2021": ["123456789", "234567890"],
            "sa2_name_2021": ["Area 1", "Area 2"],
            "population_2021": [1000, 1500]
        })
        
        seifa_path = mock_data_paths["processed_dir"] / "seifa_2021_sa2.csv"
        boundary_path = mock_data_paths["processed_dir"] / "sa2_boundaries_processed.csv"
        
        seifa_df.write_csv(seifa_path)
        boundary_df.write_csv(boundary_path)
        
        result = calculator.load_processed_data()
        
        assert result is True
        assert calculator.health_data is not None  # Should have generated mock data
        assert len(calculator.health_data) > 0
    
    def test_calculate_seifa_risk_score(self, mock_seifa_data):
        """Test SEIFA-based risk score calculation with known inputs."""
        calculator = HealthRiskCalculator()
        
        # Create test data with known SEIFA values
        test_df = pl.DataFrame({
            "sa2_code_2021": ["123456789", "234567890", "345678901"],
            "irsd_decile": [1, 5, 10],    # Low, medium, high disadvantage
            "irsad_decile": [2, 6, 9],    # Low, medium, high advantage/disadvantage
            "ier_decile": [1, 4, 8],      # Low, medium, high economic resources
            "ieo_decile": [3, 7, 10]      # Low, medium, high education/occupation
        })
        
        risk_df = calculator._calculate_seifa_risk_score(test_df)
        
        assert "seifa_risk_score" in risk_df.columns
        
        # Verify risk scores are calculated
        risk_scores = risk_df["seifa_risk_score"].to_list()
        
        # First area (all low deciles) should have higher risk
        # Last area (all high deciles) should have lower risk
        assert risk_scores[0] > risk_scores[2], "Lower SEIFA deciles should indicate higher risk"
        
        # Risk scores should be in reasonable range (0-100)
        for score in risk_scores:
            if score is not None:
                assert 0 <= score <= 100
    
    def test_calculate_seifa_risk_score_with_nulls(self):
        """Test SEIFA risk calculation handles missing values appropriately."""
        calculator = HealthRiskCalculator()
        
        test_df = pl.DataFrame({
            "sa2_code_2021": ["123456789", "234567890", "345678901"],
            "irsd_decile": [1, None, 10],     # Missing value
            "irsad_decile": [2, 6, None],     # Missing value
            "ier_decile": [1, 4, 8],
            "ieo_decile": [3, 7, 10]
        })
        
        risk_df = calculator._calculate_seifa_risk_score(test_df)
        
        # Should handle nulls gracefully
        assert len(risk_df) == 3
        
        risk_scores = risk_df["seifa_risk_score"].to_list()
        
        # Areas with partial data should still get risk scores
        # (using available indices with adjusted weights)
        non_null_scores = [score for score in risk_scores if score is not None]
        assert len(non_null_scores) >= 1
    
    def test_calculate_health_utilisation_risk(self, mock_health_data):
        """Test health utilisation risk calculation."""
        calculator = HealthRiskCalculator()
        
        # Create test health data with known patterns
        test_df = pl.DataFrame({
            "sa2_code": ["123456789", "234567890", "345678901"],
            "prescription_count": [5, 25, 50],      # Low, medium, high utilisation
            "chronic_medication": [0, 1, 1],        # No chronic, has chronic, has chronic
            "total_cost": [50.0, 500.0, 1000.0],   # Low, medium, high cost
            "population_2021": [1000, 1000, 1000]  # Same population for comparison
        })
        
        risk_df = calculator._calculate_health_utilisation_risk(test_df)
        
        assert "health_utilisation_risk" in risk_df.columns
        
        risk_scores = risk_df["health_utilisation_risk"].to_list()
        
        # Higher utilisation should indicate higher risk
        assert risk_scores[0] < risk_scores[2], "Higher health utilisation should indicate higher risk"
        
        # Risk scores should be in reasonable range
        for score in risk_scores:
            if score is not None:
                assert 0 <= score <= 100
    
    def test_calculate_geographic_accessibility_risk(self, mock_boundary_data):
        """Test geographic accessibility risk calculation."""
        calculator = HealthRiskCalculator()
        
        # Create test boundary data with accessibility indicators
        test_df = pl.DataFrame({
            "sa2_code_2021": ["123456789", "234567890", "345678901"],
            "remoteness_category": ["Major Cities", "Regional", "Very Remote"],
            "population_density": [5000.0, 100.0, 1.0],  # Urban to very remote
            "area_sqkm": [10.0, 100.0, 1000.0]           # Small to very large areas
        })
        
        risk_df = calculator._calculate_geographic_accessibility_risk(test_df)
        
        assert "geographic_risk" in risk_df.columns
        
        risk_scores = risk_df["geographic_risk"].to_list()
        
        # Very remote areas should have higher geographic risk
        major_cities_risk = risk_scores[0]
        very_remote_risk = risk_scores[2]
        
        assert very_remote_risk > major_cities_risk, "Remote areas should have higher geographic risk"
        
        # Risk scores should be in reasonable range
        for score in risk_scores:
            if score is not None:
                assert 0 <= score <= 100
    
    def test_calculate_composite_risk_score(self, integration_test_data):
        """Test composite risk score calculation combining all factors."""
        calculator = HealthRiskCalculator()
        
        # Create integrated test dataset
        integrated_data = integration_test_data(num_sa2_areas=50, num_health_records=200)
        
        # Merge all datasets for comprehensive testing
        seifa_df = integrated_data["seifa"]
        health_df = integrated_data["health"]
        boundary_df = integrated_data["boundaries"]
        
        # Calculate individual risk components
        seifa_risk = calculator._calculate_seifa_risk_score(seifa_df)
        
        # Aggregate health data by SA2
        health_agg = health_df.group_by("sa2_code").agg([
            pl.col("prescription_count").sum().alias("total_prescriptions"),
            pl.col("chronic_medication").mean().alias("chronic_rate"),
            pl.col("cost_government").sum().alias("total_cost")
        ])
        
        health_risk = calculator._calculate_health_utilisation_risk(health_agg)
        geographic_risk = calculator._calculate_geographic_accessibility_risk(boundary_df)
        
        # Combine all risk factors
        comprehensive_df = seifa_risk.join(
            health_risk, 
            left_on="sa2_code_2021", 
            right_on="sa2_code", 
            how="inner"
        ).join(
            geographic_risk, 
            on="sa2_code_2021", 
            how="inner"
        )
        
        # Calculate composite risk
        composite_df = calculator._calculate_composite_risk_score(comprehensive_df)
        
        assert "composite_risk_score" in composite_df.columns
        
        # Verify composite scores
        composite_scores = composite_df["composite_risk_score"].to_list()
        
        # All scores should be valid
        for score in composite_scores:
            if score is not None:
                assert 0 <= score <= 100
        
        # Verify score distribution
        valid_scores = [s for s in composite_scores if s is not None]
        if len(valid_scores) > 10:
            score_range = max(valid_scores) - min(valid_scores)
            assert score_range > 10, "Composite scores should show meaningful variation"
    
    def test_classify_risk_categories(self):
        """Test risk classification into categorical levels."""
        calculator = HealthRiskCalculator()
        
        # Create test data with known risk scores
        test_df = pl.DataFrame({
            "sa2_code_2021": ["123456789", "234567890", "345678901", "456789012", "567890123"],
            "composite_risk_score": [10.0, 30.0, 50.0, 70.0, 90.0]  # Very Low to Very High
        })
        
        classified_df = calculator._classify_risk_categories(test_df)
        
        assert "risk_category" in classified_df.columns
        
        categories = classified_df["risk_category"].to_list()
        
        # Verify classification logic
        assert categories[0] in ["Very Low", "Low"]      # Score 10
        assert categories[2] in ["Medium", "Moderate"]   # Score 50
        assert categories[4] in ["Very High", "High"]    # Score 90
        
        # All categories should be valid
        valid_categories = ["Very Low", "Low", "Medium", "Moderate", "High", "Very High"]
        for category in categories:
            if category is not None:
                assert category in valid_categories
    
    def test_calculate_population_adjusted_risk(self, mock_boundary_data):
        """Test population-adjusted risk calculations."""
        calculator = HealthRiskCalculator()
        
        # Create test data with different population sizes
        test_df = pl.DataFrame({
            "sa2_code_2021": ["123456789", "234567890", "345678901"],
            "composite_risk_score": [50.0, 50.0, 50.0],    # Same base risk
            "population_2021": [100, 1000, 10000],         # Different populations
            "area_sqkm": [10.0, 10.0, 10.0]               # Same area
        })
        
        adjusted_df = calculator._calculate_population_adjusted_risk(test_df)
        
        assert "population_adjusted_risk" in adjusted_df.columns
        
        # Large populations with same risk score might have different adjusted risk
        # due to scale effects
        adjusted_risks = adjusted_df["population_adjusted_risk"].to_list()
        
        for risk in adjusted_risks:
            if risk is not None:
                assert 0 <= risk <= 150  # Allow for population adjustment factors
    
    def test_generate_risk_summary_statistics(self, integration_test_data):
        """Test generation of comprehensive risk summary statistics."""
        calculator = HealthRiskCalculator()
        
        # Create comprehensive test dataset
        integrated_data = integration_test_data(num_sa2_areas=100, num_health_records=500)
        
        # Simulate complete risk assessment
        test_df = pl.DataFrame({
            "sa2_code_2021": integrated_data["sa2_codes"],
            "composite_risk_score": np.random.uniform(0, 100, 100),
            "risk_category": np.random.choice(["Very Low", "Low", "Medium", "High", "Very High"], 100),
            "seifa_risk_score": np.random.uniform(0, 100, 100),
            "health_utilisation_risk": np.random.uniform(0, 100, 100),
            "geographic_risk": np.random.uniform(0, 100, 100),
            "population_2021": np.random.randint(100, 10000, 100)
        })
        
        summary = calculator._generate_risk_summary(test_df)
        
        # Verify summary structure
        expected_keys = [
            "total_sa2_areas", "risk_distribution", "average_risk_score",
            "high_risk_areas", "population_at_risk", "risk_components"
        ]
        
        for key in expected_keys:
            assert key in summary
        
        # Verify summary calculations
        assert summary["total_sa2_areas"] == 100
        assert isinstance(summary["risk_distribution"], dict)
        assert 0 <= summary["average_risk_score"] <= 100
        
        # Risk distribution should sum to total areas
        if isinstance(summary["risk_distribution"], dict):
            total_distributed = sum(summary["risk_distribution"].values())
            assert total_distributed == 100
    
    def test_validate_risk_calculation_bounds(self):
        """Test risk calculation boundary conditions and edge cases."""
        calculator = HealthRiskCalculator()
        
        # Test extreme SEIFA values
        extreme_seifa = pl.DataFrame({
            "sa2_code_2021": ["123456789", "234567890"],
            "irsd_decile": [1, 10],    # Extreme disadvantage vs advantage
            "irsad_decile": [1, 10],
            "ier_decile": [1, 10],
            "ieo_decile": [1, 10]
        })
        
        risk_df = calculator._calculate_seifa_risk_score(extreme_seifa)
        risk_scores = risk_df["seifa_risk_score"].to_list()
        
        # Extreme cases should produce valid but different scores
        assert risk_scores[0] != risk_scores[1], "Extreme SEIFA values should produce different risk scores"
        
        for score in risk_scores:
            if score is not None:
                assert 0 <= score <= 100, "Risk scores should be within valid bounds"
    
    def test_risk_calculation_performance(self, integration_test_data):
        """Test risk calculation performance with large datasets."""
        calculator = HealthRiskCalculator()
        
        # Create large test dataset
        large_data = integration_test_data(num_sa2_areas=1000, num_health_records=5000)
        
        start_time = time.time()
        
        # Simulate full risk assessment pipeline
        seifa_df = large_data["seifa"]
        seifa_risk = calculator._calculate_seifa_risk_score(seifa_df)
        
        processing_time = time.time() - start_time
        
        # Performance assertion
        assert processing_time < 10.0, "Risk calculation should complete within 10 seconds for 1000 areas"
        
        # Verify processing integrity
        assert len(seifa_risk) == 1000
        assert "seifa_risk_score" in seifa_risk.columns
    
    def test_risk_calculator_with_missing_data_components(self, mock_seifa_data):
        """Test risk calculator handles missing data components gracefully."""
        calculator = HealthRiskCalculator()
        
        # Test with only SEIFA data (no health or boundary data)
        seifa_only = mock_seifa_data(num_areas=50)
        
        # Should still calculate SEIFA-based risk
        risk_df = calculator._calculate_seifa_risk_score(seifa_only)
        
        assert len(risk_df) > 0
        assert "seifa_risk_score" in risk_df.columns
        
        # Risk scores should be valid despite missing other components
        risk_scores = risk_df["seifa_risk_score"].drop_nulls().to_list()
        for score in risk_scores:
            assert 0 <= score <= 100
    
    def test_risk_assessment_consistency(self, mock_seifa_data):
        """Test consistency of risk assessments across multiple runs."""
        calculator = HealthRiskCalculator()
        
        # Same input data
        test_df = mock_seifa_data(num_areas=20)
        
        # Run risk calculation multiple times
        results = []
        for _ in range(3):
            risk_df = calculator._calculate_seifa_risk_score(test_df)
            results.append(risk_df["seifa_risk_score"].to_list())
        
        # Results should be identical (deterministic calculation)
        for i in range(1, len(results)):
            for j in range(len(results[0])):
                if results[0][j] is not None and results[i][j] is not None:
                    assert abs(results[0][j] - results[i][j]) < 0.001, "Risk calculations should be deterministic"
    
    def test_risk_weighting_sensitivity_analysis(self):
        """Test sensitivity of risk scores to weight changes."""
        calculator = HealthRiskCalculator()
        
        # Test data
        test_df = pl.DataFrame({
            "sa2_code_2021": ["123456789"],
            "irsd_decile": [3],
            "irsad_decile": [7],
            "ier_decile": [5],
            "ieo_decile": [6]
        })
        
        # Calculate with default weights
        default_risk = calculator._calculate_seifa_risk_score(test_df)
        default_score = default_risk["seifa_risk_score"].item()
        
        # Modify weights and recalculate
        original_weights = calculator.SEIFA_WEIGHTS.copy()
        calculator.SEIFA_WEIGHTS["irsd_decile"] = 0.7  # Increase IRSD weight
        calculator.SEIFA_WEIGHTS["irsad_decile"] = 0.1
        calculator.SEIFA_WEIGHTS["ier_decile"] = 0.1
        calculator.SEIFA_WEIGHTS["ieo_decile"] = 0.1
        
        modified_risk = calculator._calculate_seifa_risk_score(test_df)
        modified_score = modified_risk["seifa_risk_score"].item()
        
        # Restore original weights
        calculator.SEIFA_WEIGHTS = original_weights
        
        # Scores should be different due to weight changes
        assert abs(default_score - modified_score) > 1.0, "Risk scores should be sensitive to weight changes"
    
    def test_complete_risk_assessment_pipeline(self, integration_test_data, mock_data_paths):
        """Test complete end-to-end risk assessment pipeline."""
        calculator = HealthRiskCalculator(data_dir=mock_data_paths["processed_dir"])
        
        # Setup complete test environment
        integrated_data = integration_test_data(num_sa2_areas=30, num_health_records=150)
        
        # Save integrated data as would be expected
        seifa_path = mock_data_paths["processed_dir"] / "seifa_2021_sa2.csv"
        boundary_path = mock_data_paths["processed_dir"] / "sa2_boundaries_processed.csv"
        
        integrated_data["seifa"].write_csv(seifa_path)
        integrated_data["boundaries"].write_csv(boundary_path)
        
        # Execute complete pipeline
        load_success = calculator.load_processed_data()
        assert load_success is True
        
        # Run full risk assessment
        risk_results = calculator.calculate_comprehensive_risk_assessment()
        
        # Verify comprehensive results
        assert isinstance(risk_results, dict)
        
        expected_components = ["risk_scores", "summary_statistics", "risk_distribution"]
        for component in expected_components:
            if component in risk_results:
                assert risk_results[component] is not None


class TestHealthRiskCalculatorConfiguration:
    """Test health risk calculator configuration and weights."""
    
    def test_seifa_weights_configuration(self):
        """Test SEIFA weights are properly configured."""
        weights = HealthRiskCalculator.SEIFA_WEIGHTS
        
        # All four SEIFA indices should be present
        required_indices = ["irsd_decile", "irsad_decile", "ier_decile", "ieo_decile"]
        for index in required_indices:
            assert index in weights
            assert 0 < weights[index] <= 1.0
        
        # Weights should sum to 1.0
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01
        
        # IRSD should have highest weight (primary disadvantage measure)
        assert weights["irsd_decile"] >= max(weights["irsad_decile"], weights["ier_decile"], weights["ieo_decile"])
    
    def test_health_risk_factors_configuration(self):
        """Test health risk factors are properly configured."""
        factors = HealthRiskCalculator.HEALTH_RISK_FACTORS
        
        required_factors = ["high_prescription_rate", "chronic_medication_use", "geographic_isolation"]
        for factor in required_factors:
            assert factor in factors
            assert 0 < factors[factor] <= 1.0
        
        # Factors should sum to 1.0
        total_weight = sum(factors.values())
        assert abs(total_weight - 1.0) < 0.01
    
    def test_risk_category_thresholds(self):
        """Test risk category classification thresholds are reasonable."""
        calculator = HealthRiskCalculator()
        
        # Test threshold boundaries
        test_scores = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
        
        test_df = pl.DataFrame({
            "sa2_code_2021": [f"12345678{i}" for i in range(10)],
            "composite_risk_score": test_scores
        })
        
        classified_df = calculator._classify_risk_categories(test_df)
        categories = classified_df["risk_category"].to_list()
        
        # Should have progression from low to high risk
        low_risk_count = sum(1 for cat in categories if cat in ["Very Low", "Low"])
        high_risk_count = sum(1 for cat in categories if cat in ["High", "Very High"])
        
        assert low_risk_count > 0, "Should classify some areas as low risk"
        assert high_risk_count > 0, "Should classify some areas as high risk"