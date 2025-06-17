"""
Integration tests for Phase 3 Analytics Modules

Tests the complete analytics pipeline including health risk assessment,
geographic analysis, and health service utilisation analysis.
"""

import pytest
import polars as pl
import numpy as np
from pathlib import Path
import tempfile
import json

from src.analysis.risk.health_risk_calculator import HealthRiskCalculator
from src.analysis.risk.healthcare_access_scorer import HealthcareAccessScorer
from src.analysis.spatial.sa2_health_mapper import SA2HealthMapper
from src.analysis.health.medicare_utilisation_analyzer import MedicareUtilisationAnalyzer
from src.analysis.health.pharmaceutical_analyzer import PharmaceuticalAnalyzer


@pytest.fixture
def comprehensive_mock_data():
    """Create comprehensive mock data for integration testing."""
    np.random.seed(42)
    n_areas = 200
    
    # Mock SEIFA data
    seifa_data = pl.DataFrame({
        'sa2_code_2021': [f"1{str(i).zfill(8)}" for i in range(1000, 1000 + n_areas)],
        'sa2_name_2021': [f"Test Area {i}" for i in range(n_areas)],
        'irsd_decile': np.random.randint(1, 11, n_areas),
        'irsad_decile': np.random.randint(1, 11, n_areas),
        'ier_decile': np.random.randint(1, 11, n_areas),
        'ieo_decile': np.random.randint(1, 11, n_areas),
        'irsd_score': np.random.randint(800, 1200, n_areas),
        'state_name': np.random.choice(['NSW', 'VIC', 'QLD'], n_areas)
    })
    
    # Mock boundary data
    boundary_data = pl.DataFrame({
        'sa2_code': [f"1{str(i).zfill(8)}" for i in range(1000, 1000 + n_areas)],
        'sa2_name': [f"Test Area {i}" for i in range(n_areas)],
        'state_name': np.random.choice(['NSW', 'VIC', 'QLD'], n_areas),
        'usual_resident_population': np.random.randint(500, 5000, n_areas),
        'area_sqkm': np.random.exponential(50, n_areas) + 1,
        'centroid_lat': np.random.uniform(-37, -33, n_areas),
        'centroid_lon': np.random.uniform(144, 151, n_areas)
    })
    
    # Mock health utilisation data
    n_health_records = 5000
    sa2_codes = [f"1{str(i).zfill(8)}" for i in range(1000, 1000 + n_areas)]
    
    health_data = pl.DataFrame({
        'sa2_code': np.random.choice(sa2_codes, n_health_records),
        'prescription_count': np.random.poisson(3, n_health_records),
        'chronic_medication': np.random.choice([0, 1], n_health_records, p=[0.7, 0.3]),
        'total_cost': np.random.exponential(45, n_health_records),
        'state': np.random.choice(['NSW', 'VIC', 'QLD'], n_health_records)
    })
    
    return {
        'seifa': seifa_data,
        'boundary': boundary_data,
        'health': health_data
    }


@pytest.fixture
def temp_integrated_data_dir(comprehensive_mock_data):
    """Create temporary directory with comprehensive mock data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save all mock data
        comprehensive_mock_data['seifa'].write_csv(temp_path / "seifa_2021_sa2.csv")
        comprehensive_mock_data['boundary'].write_csv(temp_path / "sa2_boundaries_processed.csv")
        comprehensive_mock_data['health'].write_csv(temp_path / "health_data_processed.csv")
        
        yield temp_path


class TestPhase3Integration:
    """Integration tests for complete Phase 3 analytics pipeline."""
    
    def test_health_risk_calculator_integration(self, temp_integrated_data_dir):
        """Test health risk calculator with integrated data."""
        calculator = HealthRiskCalculator(data_dir=temp_integrated_data_dir)
        
        # Load data
        assert calculator.load_processed_data() is True
        assert calculator.seifa_data is not None
        assert calculator.health_data is not None
        assert calculator.boundary_data is not None
        
        # Calculate composite risk scores
        composite_data = calculator.calculate_composite_risk_score()
        assert isinstance(composite_data, pl.DataFrame)
        assert 'composite_risk_score' in composite_data.columns
        assert 'risk_category' in composite_data.columns
        assert composite_data.shape[0] == 200  # Should match number of SA2 areas
        
        # Verify risk score range
        risk_scores = composite_data['composite_risk_score'].to_numpy()
        assert np.all(risk_scores >= 1.0) and np.all(risk_scores <= 10.0)
        
        # Generate summary
        summary = calculator.generate_risk_summary()
        assert summary['total_sa2_areas'] == 200
        assert 1.0 <= summary['average_risk_score'] <= 10.0
    
    def test_healthcare_access_scorer_integration(self, temp_integrated_data_dir):
        """Test healthcare access scorer with integrated data."""
        scorer = HealthcareAccessScorer(data_dir=temp_integrated_data_dir)
        
        # Load data
        assert scorer.load_boundary_data() is True
        assert scorer.boundary_data is not None
        
        # Calculate access scores
        access_data = scorer.calculate_composite_access_score()
        assert isinstance(access_data, pl.DataFrame)
        assert 'composite_access_score' in access_data.columns
        assert 'access_category' in access_data.columns
        assert access_data.shape[0] == 200
        
        # Verify access score range
        access_scores = access_data['composite_access_score'].to_numpy()
        assert np.all(access_scores >= 1.0) and np.all(access_scores <= 10.0)
        
        # Generate summary
        summary = scorer.generate_access_summary(access_data)
        assert summary['total_sa2_areas'] == 200
        assert 1.0 <= summary['average_access_score'] <= 10.0
    
    def test_sa2_health_mapper_integration(self, temp_integrated_data_dir):
        """Test SA2 health mapper with integrated data."""
        mapper = SA2HealthMapper(data_dir=temp_integrated_data_dir)
        
        # Load data
        assert mapper.load_spatial_data() is True
        assert mapper.boundary_data is not None
        
        # Create population density map
        density_map = mapper.create_population_density_map()
        assert isinstance(density_map, pl.DataFrame)
        assert 'population_density_per_sqkm' in density_map.columns
        assert 'density_category' in density_map.columns
        assert density_map.shape[0] == 200
        
        # Create state summary
        state_summary = mapper.create_state_level_summary()
        assert isinstance(state_summary, pl.DataFrame)
        assert 'total_sa2_areas' in state_summary.columns
        assert 'total_population' in state_summary.columns
        
        # Should have 3 states (NSW, VIC, QLD)
        assert state_summary.shape[0] == 3
    
    def test_medicare_utilisation_analyzer_integration(self, temp_integrated_data_dir):
        """Test Medicare utilisation analyzer with integrated data."""
        analyzer = MedicareUtilisationAnalyzer(data_dir=temp_integrated_data_dir)
        
        # Load data
        assert analyzer.load_medicare_data() is True
        assert analyzer.medicare_data is not None
        assert analyzer.population_data is not None
        
        # Calculate per capita utilisation
        per_capita = analyzer.calculate_per_capita_utilisation()
        assert isinstance(per_capita, pl.DataFrame)
        assert 'services_per_capita' in per_capita.columns
        assert 'utilisation_category' in per_capita.columns
        
        # Generate summary
        summary = analyzer.generate_utilisation_summary()
        assert 'total_service_episodes' in summary
        assert 'average_services_per_capita' in summary
        assert summary['total_service_episodes'] > 0
    
    def test_pharmaceutical_analyzer_integration(self, temp_integrated_data_dir):
        """Test pharmaceutical analyzer with integrated data."""
        analyzer = PharmaceuticalAnalyzer(data_dir=temp_integrated_data_dir)
        
        # Load data
        assert analyzer.load_pharmaceutical_data() is True
        assert analyzer.pbs_data is not None
        assert analyzer.population_data is not None
        
        # Calculate prescription rates
        prescription_rates = analyzer.calculate_prescription_rates()
        assert isinstance(prescription_rates, pl.DataFrame)
        assert 'prescriptions_per_capita' in prescription_rates.columns
        assert 'usage_category' in prescription_rates.columns
        
        # Analyze therapeutic categories
        atc_analysis = analyzer.analyze_therapeutic_categories()
        assert isinstance(atc_analysis, pl.DataFrame)
        assert 'atc_category' in atc_analysis.columns
        assert 'category_description' in atc_analysis.columns
        
        # Generate summary
        summary = analyzer.generate_pharmaceutical_summary()
        assert 'total_dispensing_episodes' in summary
        assert 'average_prescriptions_per_capita' in summary
        assert summary['total_dispensing_episodes'] > 0
    
    def test_cross_module_data_consistency(self, temp_integrated_data_dir):
        """Test data consistency across multiple analytics modules."""
        # Initialize all analyzers with same data
        risk_calculator = HealthRiskCalculator(data_dir=temp_integrated_data_dir)
        access_scorer = HealthcareAccessScorer(data_dir=temp_integrated_data_dir)
        health_mapper = SA2HealthMapper(data_dir=temp_integrated_data_dir)
        
        # Load data in all modules
        assert risk_calculator.load_processed_data() is True
        assert access_scorer.load_boundary_data() is True
        assert health_mapper.load_spatial_data() is True
        
        # Check data shape consistency
        seifa_areas = risk_calculator.seifa_data.shape[0]
        boundary_areas = access_scorer.boundary_data.shape[0]
        mapper_areas = health_mapper.boundary_data.shape[0]
        
        assert seifa_areas == boundary_areas == mapper_areas == 200
        
        # Check SA2 code consistency
        seifa_codes = set(risk_calculator.seifa_data['sa2_code_2021'].to_list())
        boundary_codes = set(access_scorer.boundary_data['sa2_code'].to_list())
        mapper_codes = set(health_mapper.boundary_data['sa2_code'].to_list())
        
        # Codes should be identical (accounting for column name differences)
        assert len(seifa_codes.intersection(boundary_codes)) > 190  # Allow for minor differences
        assert len(boundary_codes.intersection(mapper_codes)) == 200  # Should be identical
    
    def test_complete_analytics_pipeline(self, temp_integrated_data_dir):
        """Test complete end-to-end analytics pipeline."""
        # Create output directory
        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir)
            
            # Run risk assessment
            risk_calculator = HealthRiskCalculator(data_dir=temp_integrated_data_dir)
            risk_calculator.load_processed_data()
            risk_output_path = output_path / "risk_assessment"
            risk_output_path.mkdir(exist_ok=True)
            assert risk_calculator.export_risk_assessment(risk_output_path) is True
            
            # Verify risk assessment outputs
            assert (risk_output_path / "health_risk_assessment.csv").exists()
            assert (risk_output_path / "health_risk_assessment.parquet").exists()
            assert (risk_output_path / "risk_assessment_summary.json").exists()
            
            # Run access assessment
            access_scorer = HealthcareAccessScorer(data_dir=temp_integrated_data_dir)
            access_scorer.load_boundary_data()
            access_output_path = output_path / "access_assessment"
            access_output_path.mkdir(exist_ok=True)
            assert access_scorer.export_access_assessment(access_output_path) is True
            
            # Verify access assessment outputs
            assert (access_output_path / "healthcare_access_assessment.csv").exists()
            assert (access_output_path / "healthcare_access_assessment.parquet").exists()
            assert (access_output_path / "access_assessment_summary.json").exists()
            
            # Run health mapping
            health_mapper = SA2HealthMapper(data_dir=temp_integrated_data_dir)
            health_mapper.load_spatial_data()
            mapping_output_path = output_path / "health_mapping"
            mapping_output_path.mkdir(exist_ok=True)
            assert health_mapper.export_mapping_data(mapping_output_path) is True
            
            # Verify mapping outputs
            assert (mapping_output_path / "population_density_map.csv").exists()
            assert (mapping_output_path / "state_health_summary.csv").exists()
            assert (mapping_output_path / "sa2_health_mapping.geojson").exists()
    
    def test_analytics_performance_benchmarks(self, temp_integrated_data_dir):
        """Test performance benchmarks for analytics modules."""
        import time
        
        # Test risk calculation performance
        start_time = time.time()
        risk_calculator = HealthRiskCalculator(data_dir=temp_integrated_data_dir)
        risk_calculator.load_processed_data()
        composite_data = risk_calculator.calculate_composite_risk_score()
        risk_time = time.time() - start_time
        
        # Should complete within reasonable time for 200 areas
        assert risk_time < 5.0, f"Risk calculation took {risk_time:.2f}s, expected <5s"
        assert composite_data.shape[0] == 200
        
        # Test access scoring performance
        start_time = time.time()
        access_scorer = HealthcareAccessScorer(data_dir=temp_integrated_data_dir)
        access_scorer.load_boundary_data()
        access_data = access_scorer.calculate_composite_access_score()
        access_time = time.time() - start_time
        
        assert access_time < 5.0, f"Access scoring took {access_time:.2f}s, expected <5s"
        assert access_data.shape[0] == 200
        
        # Test spatial mapping performance
        start_time = time.time()
        health_mapper = SA2HealthMapper(data_dir=temp_integrated_data_dir)
        health_mapper.load_spatial_data()
        density_map = health_mapper.create_population_density_map()
        mapping_time = time.time() - start_time
        
        assert mapping_time < 3.0, f"Spatial mapping took {mapping_time:.2f}s, expected <3s"
        assert density_map.shape[0] == 200
    
    def test_analytics_data_quality_validation(self, temp_integrated_data_dir):
        """Test data quality validation across analytics modules."""
        # Initialize all modules
        risk_calculator = HealthRiskCalculator(data_dir=temp_integrated_data_dir)
        access_scorer = HealthcareAccessScorer(data_dir=temp_integrated_data_dir)
        medicare_analyzer = MedicareUtilisationAnalyzer(data_dir=temp_integrated_data_dir)
        pharma_analyzer = PharmaceuticalAnalyzer(data_dir=temp_integrated_data_dir)
        
        # Load all data
        assert risk_calculator.load_processed_data() is True
        assert access_scorer.load_boundary_data() is True
        assert medicare_analyzer.load_medicare_data() is True
        assert pharma_analyzer.load_pharmaceutical_data() is True
        
        # Validate risk scores
        composite_data = risk_calculator.calculate_composite_risk_score()
        assert composite_data['composite_risk_score'].null_count() == 0, "No null risk scores"
        assert composite_data['risk_category'].null_count() == 0, "No null risk categories"
        
        # Validate access scores
        access_data = access_scorer.calculate_composite_access_score()
        assert access_data['composite_access_score'].null_count() == 0, "No null access scores"
        assert access_data['access_category'].null_count() == 0, "No null access categories"
        
        # Validate utilisation data
        per_capita = medicare_analyzer.calculate_per_capita_utilisation()
        assert per_capita['services_per_capita'].null_count() == 0, "No null per capita values"
        
        # Validate prescription data
        prescription_rates = pharma_analyzer.calculate_prescription_rates()
        assert prescription_rates['prescriptions_per_capita'].null_count() == 0, "No null prescription rates"


@pytest.mark.performance
class TestPhase3Performance:
    """Performance tests for Phase 3 analytics modules."""
    
    def test_large_dataset_performance(self):
        """Test performance with larger mock datasets."""
        np.random.seed(42)
        n_areas = 2000  # Closer to real Australian SA2 count
        
        # Create large mock datasets
        large_seifa = pl.DataFrame({
            'sa2_code_2021': [f"1{str(i).zfill(8)}" for i in range(1000, 1000 + n_areas)],
            'sa2_name_2021': [f"Large Test Area {i}" for i in range(n_areas)],
            'irsd_decile': np.random.randint(1, 11, n_areas),
            'irsad_decile': np.random.randint(1, 11, n_areas),
            'ier_decile': np.random.randint(1, 11, n_areas),
            'ieo_decile': np.random.randint(1, 11, n_areas),
        })
        
        # Test risk calculation performance on large dataset
        import time
        start_time = time.time()
        
        calculator = HealthRiskCalculator()
        calculator.seifa_data = large_seifa
        calculator.health_data = calculator._generate_mock_health_data()
        calculator.boundary_data = calculator._generate_mock_population_data()
        
        composite_data = calculator.calculate_composite_risk_score()
        calculation_time = time.time() - start_time
        
        # Should handle 2000 areas efficiently
        assert calculation_time < 10.0, f"Large dataset calculation took {calculation_time:.2f}s"
        assert composite_data.shape[0] > 1900  # Allow for some integration loss


if __name__ == "__main__":
    pytest.main([__file__, "-v"])