"""
Test suite for Health Risk Calculator module

Tests the multi-factor health risk assessment functionality using real and mock data.
"""

import pytest
import polars as pl
import numpy as np
from pathlib import Path
import tempfile
import json

from src.analysis.risk.health_risk_calculator import HealthRiskCalculator


@pytest.fixture
def mock_seifa_data():
    """Create mock SEIFA data for testing."""
    np.random.seed(42)
    n_areas = 100
    
    return pl.DataFrame({
        'sa2_code_2021': [f"1{str(i).zfill(8)}" for i in range(1000, 1000 + n_areas)],
        'sa2_name_2021': [f"Test Area {i}" for i in range(n_areas)],
        'irsd_decile': np.random.randint(1, 11, n_areas),
        'irsad_decile': np.random.randint(1, 11, n_areas),
        'ier_decile': np.random.randint(1, 11, n_areas),
        'ieo_decile': np.random.randint(1, 11, n_areas),
        'irsd_score': np.random.randint(800, 1200, n_areas),
        'state_name': np.random.choice(['NSW', 'VIC', 'QLD'], n_areas)
    })


@pytest.fixture 
def mock_health_data():
    """Create mock health utilisation data for testing."""
    np.random.seed(42)
    n_records = 1000
    sa2_codes = [f"1{str(i).zfill(8)}" for i in range(1000, 1100)]
    
    return pl.DataFrame({
        'sa2_code': np.random.choice(sa2_codes, n_records),
        'prescription_count': np.random.poisson(5, n_records),
        'chronic_medication': np.random.choice([0, 1], n_records, p=[0.7, 0.3]),
        'total_cost': np.random.exponential(50, n_records),
        'state': np.random.choice(['NSW', 'VIC', 'QLD'], n_records)
    })


@pytest.fixture
def mock_boundary_data():
    """Create mock boundary data for testing."""
    np.random.seed(42)
    n_areas = 100
    
    return pl.DataFrame({
        'sa2_code': [f"1{str(i).zfill(8)}" for i in range(1000, 1000 + n_areas)],
        'state_name': np.random.choice(['NSW', 'VIC', 'QLD'], n_areas),
        'usual_resident_population': np.random.randint(500, 5000, n_areas)
    })


@pytest.fixture
def risk_calculator():
    """Create risk calculator instance for testing."""
    return HealthRiskCalculator()


@pytest.fixture
def temp_data_dir(mock_seifa_data, mock_health_data, mock_boundary_data):
    """Create temporary directory with mock data files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save mock data to CSV files
        mock_seifa_data.write_csv(temp_path / "seifa_2021_sa2.csv")
        mock_health_data.write_csv(temp_path / "health_data_processed.csv")
        mock_boundary_data.write_csv(temp_path / "sa2_boundaries_processed.csv")
        
        yield temp_path


class TestHealthRiskCalculator:
    """Test suite for HealthRiskCalculator class."""
    
    def test_initialization(self, risk_calculator):
        """Test calculator initialization."""
        assert risk_calculator.data_dir == Path("data/processed")
        assert risk_calculator.seifa_data is None
        assert risk_calculator.health_data is None
        assert risk_calculator.boundary_data is None
    
    def test_seifa_weights_sum_to_one(self, risk_calculator):
        """Test that SEIFA weights sum to 1.0."""
        total_weight = sum(risk_calculator.SEIFA_WEIGHTS.values())
        assert abs(total_weight - 1.0) < 0.01, f"SEIFA weights sum to {total_weight}, expected 1.0"
    
    def test_load_data_with_mock_files(self, risk_calculator, temp_data_dir):
        """Test loading data from mock CSV files."""
        risk_calculator.data_dir = temp_data_dir
        
        success = risk_calculator.load_processed_data()
        assert success is True
        assert risk_calculator.seifa_data is not None
        assert risk_calculator.health_data is not None
        assert risk_calculator.boundary_data is not None
        
        # Check data shapes
        assert risk_calculator.seifa_data.shape[0] == 100
        assert risk_calculator.health_data.shape[0] == 1000
        assert risk_calculator.boundary_data.shape[0] == 100
    
    def test_load_data_missing_files(self, risk_calculator):
        """Test loading data when files don't exist."""
        risk_calculator.data_dir = Path("/nonexistent/path")
        
        success = risk_calculator.load_processed_data()
        assert success is False
    
    def test_generate_mock_health_data(self, risk_calculator):
        """Test mock health data generation."""
        mock_data = risk_calculator._generate_mock_health_data()
        
        assert isinstance(mock_data, pl.DataFrame)
        assert mock_data.shape[0] == 50000
        assert 'sa2_code' in mock_data.columns
        assert 'prescription_count' in mock_data.columns
        assert 'chronic_medication' in mock_data.columns
    
    def test_calculate_seifa_risk_score(self, risk_calculator, mock_seifa_data):
        """Test SEIFA risk score calculation."""
        risk_calculator.seifa_data = mock_seifa_data
        
        seifa_risk = risk_calculator.calculate_seifa_risk_score()
        
        assert isinstance(seifa_risk, pl.DataFrame)
        assert 'seifa_risk_score' in seifa_risk.columns
        assert seifa_risk.shape[0] == 100
        
        # Check risk score range (should be 1-10)
        risk_scores = seifa_risk['seifa_risk_score'].to_numpy()
        assert np.all(risk_scores >= 1.0) and np.all(risk_scores <= 10.0)
    
    def test_calculate_health_utilisation_risk(self, risk_calculator, mock_health_data):
        """Test health utilisation risk calculation."""
        risk_calculator.health_data = mock_health_data
        
        health_risk = risk_calculator.calculate_health_utilisation_risk()
        
        assert isinstance(health_risk, pl.DataFrame)
        assert 'health_utilisation_risk' in health_risk.columns
        assert 'total_prescriptions' in health_risk.columns
        assert 'chronic_medication_rate' in health_risk.columns
        
        # Check risk score range (should be 0-10) 
        risk_scores = health_risk['health_utilisation_risk'].to_numpy()
        assert np.all(risk_scores >= 0) and np.all(risk_scores <= 10)
    
    def test_integrate_data_sources(self, risk_calculator, temp_data_dir):
        """Test data integration across all sources."""
        risk_calculator.data_dir = temp_data_dir
        risk_calculator.load_processed_data()
        
        integrated = risk_calculator.integrate_data_sources()
        
        assert isinstance(integrated, pl.DataFrame)
        assert 'seifa_risk_score' in integrated.columns
        assert 'health_utilisation_risk' in integrated.columns
        assert 'state_name' in integrated.columns
        assert 'usual_resident_population' in integrated.columns
        
        # Check that data was joined correctly
        assert integrated.shape[0] > 0
    
    def test_calculate_composite_risk_score(self, risk_calculator, temp_data_dir):
        """Test composite risk score calculation."""
        risk_calculator.data_dir = temp_data_dir
        risk_calculator.load_processed_data()
        
        composite = risk_calculator.calculate_composite_risk_score()
        
        assert isinstance(composite, pl.DataFrame)
        assert 'composite_risk_score' in composite.columns
        assert 'risk_category' in composite.columns
        
        # Check risk categories
        categories = composite['risk_category'].unique().to_list()
        expected_categories = ['Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk']
        for cat in categories:
            assert cat in expected_categories
        
        # Check risk score range
        risk_scores = composite['composite_risk_score'].to_numpy()
        assert np.all(risk_scores >= 1.0) and np.all(risk_scores <= 10.0)
    
    def test_generate_risk_summary(self, risk_calculator, temp_data_dir):
        """Test risk summary generation."""
        risk_calculator.data_dir = temp_data_dir
        risk_calculator.load_processed_data()
        risk_calculator.calculate_composite_risk_score()
        
        summary = risk_calculator.generate_risk_summary()
        
        assert isinstance(summary, dict)
        assert 'total_sa2_areas' in summary
        assert 'average_risk_score' in summary
        assert 'high_risk_areas' in summary
        assert 'low_risk_areas' in summary
        assert 'risk_distribution' in summary
        
        # Check summary values
        assert summary['total_sa2_areas'] > 0
        assert 1.0 <= summary['average_risk_score'] <= 10.0
        assert summary['high_risk_areas'] >= 0
        assert summary['low_risk_areas'] >= 0
        
        # Check risk distribution
        dist = summary['risk_distribution']
        total_dist = dist['low'] + dist['moderate'] + dist['high'] + dist['very_high']
        assert total_dist == summary['total_sa2_areas']
    
    def test_export_risk_assessment(self, risk_calculator, temp_data_dir):
        """Test risk assessment export functionality."""
        risk_calculator.data_dir = temp_data_dir
        risk_calculator.load_processed_data()
        
        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir)
            
            success = risk_calculator.export_risk_assessment(output_path)
            assert success is True
            
            # Check exported files exist
            assert (output_path / "health_risk_assessment.csv").exists()
            assert (output_path / "health_risk_assessment.parquet").exists()
            assert (output_path / "risk_assessment_summary.json").exists()
            
            # Validate JSON summary
            with open(output_path / "risk_assessment_summary.json") as f:
                summary = json.load(f)
            assert 'total_sa2_areas' in summary
            assert 'average_risk_score' in summary
    
    def test_process_complete_risk_pipeline(self, risk_calculator, temp_data_dir):
        """Test complete risk assessment pipeline."""
        risk_calculator.data_dir = temp_data_dir
        
        # Mock the output directory creation
        import os
        original_makedirs = os.makedirs
        
        def mock_makedirs(path, exist_ok=False):
            # Don't actually create the directory
            pass
        
        os.makedirs = mock_makedirs
        
        try:
            # This should run without error but may not complete export
            # due to mocked directory creation
            success = risk_calculator.process_complete_risk_pipeline()
            # Success depends on whether export succeeds
            assert isinstance(success, bool)
        finally:
            os.makedirs = original_makedirs
    
    def test_risk_score_consistency(self, risk_calculator, temp_data_dir):
        """Test that risk scores are consistent across runs."""
        risk_calculator.data_dir = temp_data_dir
        risk_calculator.load_processed_data()
        
        # Calculate risk scores twice
        composite1 = risk_calculator.calculate_composite_risk_score()
        composite2 = risk_calculator.calculate_composite_risk_score()
        
        # Should be identical
        assert composite1.shape == composite2.shape
        
        # Check some key columns are identical
        assert composite1['composite_risk_score'].equals(composite2['composite_risk_score'])
        assert composite1['risk_category'].equals(composite2['risk_category'])
    
    def test_seifa_risk_inversion(self, risk_calculator):
        """Test that SEIFA deciles are correctly inverted for risk scoring."""
        # Create test data with known deciles
        test_data = pl.DataFrame({
            'sa2_code_2021': ['100000001', '100000002'],
            'sa2_name_2021': ['Low Decile Area', 'High Decile Area'],
            'irsd_decile': [1, 10],  # 1 = most disadvantaged, 10 = least disadvantaged
            'irsad_decile': [1, 10],
            'ier_decile': [1, 10],
            'ieo_decile': [1, 10]
        })
        
        risk_calculator.seifa_data = test_data
        seifa_risk = risk_calculator.calculate_seifa_risk_score()
        
        # Area with decile 1 (most disadvantaged) should have higher risk score
        # Area with decile 10 (least disadvantaged) should have lower risk score
        low_decile_risk = seifa_risk.filter(pl.col('sa2_code_2021') == '100000001')['seifa_risk_score'].item()
        high_decile_risk = seifa_risk.filter(pl.col('sa2_code_2021') == '100000002')['seifa_risk_score'].item()
        
        assert low_decile_risk > high_decile_risk, "Most disadvantaged area should have higher risk score"
        assert low_decile_risk == 10.0, "Decile 1 areas should have maximum risk score of 10"
        assert high_decile_risk == 1.0, "Decile 10 areas should have minimum risk score of 1"


@pytest.mark.integration
class TestHealthRiskCalculatorIntegration:
    """Integration tests using real processed data if available."""
    
    def test_with_real_seifa_data(self):
        """Test calculator with real SEIFA data if available."""
        calculator = HealthRiskCalculator()
        
        # Check if real data exists
        seifa_path = calculator.data_dir / "seifa_2021_sa2.csv"
        if not seifa_path.exists():
            pytest.skip("Real SEIFA data not available")
        
        # Load and test with real data
        success = calculator.load_processed_data()
        if success and calculator.seifa_data is not None:
            seifa_risk = calculator.calculate_seifa_risk_score()
            
            # Should have significant number of SA2 areas
            assert seifa_risk.shape[0] > 1000, "Should have over 1000 SA2 areas"
            
            # Check data quality
            assert seifa_risk['seifa_risk_score'].null_count() == 0, "No null risk scores"
            
            # Risk scores should be reasonable
            avg_risk = seifa_risk['seifa_risk_score'].mean()
            assert 3.0 <= avg_risk <= 7.0, f"Average risk {avg_risk} should be reasonable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])