"""
Comprehensive test suite for CensusIntegrator.

Tests all aspects of census data integration including:
- 4-way joins across all transformers
- Data quality validation
- Temporal alignment
- Missing data handling
- Derived indicator calculation
- Performance optimization
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from src.transformers.census.census_integrator import (
    CensusIntegrator,
    DataQualityMetrics,
    CensusIntegratorConfig
)
from src.transformers.census.census_integrator_helpers import CensusIntegratorHelpers
from src.utils.interfaces import ProcessingStatus, TransformationError


class TestCensusIntegrator:
    """Test suite for CensusIntegrator class."""
    
    @pytest.fixture
    def integrator(self):
        """Create CensusIntegrator instance for testing."""
        return CensusIntegrator()
    
    @pytest.fixture
    def sample_demographics_data(self):
        """Create sample demographics data."""
        return pd.DataFrame({
            'geographic_id': ['101021001', '101021002', '101021003'],
            'geographic_level': ['SA2', 'SA2', 'SA2'],
            'geographic_name': ['Area A', 'Area B', 'Area C'],
            'state_territory': ['NSW', 'NSW', 'NSW'],
            'census_year': [2021, 2021, 2021],
            'total_population': [5000, 8000, 3000],
            'males': [2500, 4000, 1500],
            'females': [2500, 4000, 1500],
            'age_0_4': [300, 500, 200],
            'age_5_9': [350, 600, 250],
            'age_15_19': [400, 700, 300],
            'age_65_69': [250, 400, 150],
            'age_85_plus': [100, 200, 80],
            'indigenous': [100, 150, 80],
            'non_indigenous': [4800, 7700, 2850],
            'indigenous_not_stated': [100, 150, 70],
            'total_private_dwellings': [2000, 3200, 1200],
            'occupied_private_dwellings': [1900, 3000, 1100],
            'total_families': [1400, 2200, 800]
        })
    
    @pytest.fixture
    def sample_education_data(self):
        """Create sample education data."""
        return pd.DataFrame({
            'geographic_id': ['101021001', '101021002', '101021003'],
            'geographic_level': ['SA2', 'SA2', 'SA2'],
            'census_year': [2021, 2021, 2021],
            'education_pop_base': [4000, 6400, 2400],
            'year_12_or_equivalent': [2800, 4500, 1600],
            'year_11_or_equivalent': [400, 640, 300],
            'year_10_or_equivalent': [500, 800, 350],
            'year_8_or_below': [300, 460, 150],
            'bachelor_degree': [1200, 2000, 600],
            'postgraduate_degree': [400, 800, 200],
            'certificate_iii_iv': [800, 1200, 400],
            'no_qualification': [600, 900, 400]
        })
    
    @pytest.fixture
    def sample_employment_data(self):
        """Create sample employment data."""
        return pd.DataFrame({
            'geographic_id': ['101021001', '101021002', '101021003'],
            'geographic_level': ['SA2', 'SA2', 'SA2'],
            'census_year': [2021, 2021, 2021],
            'labour_force_pop': [3500, 5600, 2100],
            'employed_full_time': [2500, 4000, 1400],
            'employed_part_time': [700, 1200, 500],
            'unemployed': [300, 400, 200],
            'not_in_labour_force': [1500, 2400, 900],
            'professionals': [800, 1400, 400],
            'managers': [400, 700, 200],
            'technicians_trades': [600, 1000, 350],
            'clerical_administrative': [500, 800, 300],
            'manufacturing': [300, 600, 200],
            'construction': [400, 700, 250],
            'retail_trade': [500, 900, 300]
        })
    
    @pytest.fixture
    def sample_housing_data(self):
        """Create sample housing data."""
        return pd.DataFrame({
            'geographic_id': ['101021001', '101021002', '101021003'],
            'geographic_level': ['SA2', 'SA2', 'SA2'],
            'census_year': [2021, 2021, 2021],
            'separate_house': [1600, 2400, 900],
            'semi_detached': [200, 400, 150],
            'flat_apartment': [200, 400, 150],
            'owned_outright': [800, 1200, 500],
            'owned_with_mortgage': [900, 1500, 400],
            'rented': [200, 500, 300],
            'one_bedroom': [100, 200, 80],
            'two_bedrooms': [400, 600, 300],
            'three_bedrooms': [800, 1300, 500],
            'four_bedrooms': [600, 1100, 320],
            'median_rent_weekly': [450, 520, 380],
            'median_mortgage_monthly': [2800, 3200, 2400],
            'internet_connection': [1800, 2900, 1000],
            'one_motor_vehicle': [800, 1300, 600],
            'two_motor_vehicles': [900, 1500, 400]
        })
    
    @pytest.fixture
    def sample_seifa_data(self):
        """Create sample SEIFA data."""
        return pd.DataFrame({
            'geographic_id': ['101021001', '101021002', '101021003'],
            'geographic_level': ['SA2', 'SA2', 'SA2'],
            'geographic_name': ['Area A', 'Area B', 'Area C'],
            'state_territory': ['NSW', 'NSW', 'NSW'],
            'census_year': [2021, 2021, 2021],
            'irsad_score': [1050, 1200, 900],
            'irsd_score': [1000, 1150, 850],
            'ier_score': [1100, 1250, 950],
            'ieo_score': [1080, 1180, 920],
            'irsad_rank': [15000, 8000, 25000],
            'irsd_rank': [18000, 10000, 28000],
            'irsad_decile': [6, 8, 4],
            'irsd_decile': [5, 7, 3],
            'ier_decile': [7, 8, 5],
            'ieo_decile': [6, 7, 4],
            'disadvantage_severity': ['moderate', 'low', 'high'],
            'population_base': [5000, 8000, 3000]
        })
    
    def test_census_integrator_initialization(self, integrator):
        """Test proper initialization of CensusIntegrator."""
        assert integrator is not None
        assert isinstance(integrator.config, CensusIntegratorConfig)
        assert isinstance(integrator.get_quality_metrics(), DataQualityMetrics)
        assert integrator.integration_sk_counter == 50000
        assert integrator.processing_metadata is None
        
    def test_successful_integration(
        self, integrator, sample_demographics_data, sample_education_data,
        sample_employment_data, sample_housing_data, sample_seifa_data
    ):
        """Test successful 4-way integration of all census data."""
        result = integrator.integrate(
            demographics_data=sample_demographics_data,
            education_data=sample_education_data,
            employment_data=sample_employment_data,
            housing_data=sample_housing_data,
            seifa_data=sample_seifa_data
        )
        
        # Verify result structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # Should have 3 geographic areas
        assert not result.empty
        
        # Verify core columns are present
        core_columns = ['geographic_id', 'geographic_level', 'census_year']
        for col in core_columns:
            assert col in result.columns
        
        # Verify data from all sources is integrated
        assert 'total_population' in result.columns  # Demographics
        # Note: Integration adds suffixes to avoid column conflicts
        education_cols = [col for col in result.columns if 'education' in col]
        employment_cols = [col for col in result.columns if 'employment' in col]
        housing_cols = [col for col in result.columns if 'housing' in col]
        seifa_cols = [col for col in result.columns if any(seifa in col for seifa in ['irsad', 'irsd', 'ier', 'ieo', 'seifa'])]
        
        # At least one column from each domain should be present if data was provided
        assert len(education_cols) > 0   # Education
        assert len(employment_cols) > 0 or 'employed_full_time' in result.columns # Employment  
        assert len(housing_cols) > 0 or 'owned_outright' in result.columns    # Housing
        assert len(seifa_cols) > 0 or 'irsad_score' in result.columns       # SEIFA
        
        # Verify integration metadata
        assert 'integration_sk' in result.columns
        assert 'integration_timestamp' in result.columns
        assert 'data_completeness_score' in result.columns
        
        # Verify processing metadata
        assert integrator.processing_metadata is not None
        assert integrator.processing_metadata.status == ProcessingStatus.COMPLETED
        assert integrator.processing_metadata.records_processed == 3
    
    def test_empty_dataset_handling(self, integrator):
        """Test handling of empty input datasets."""
        empty_df = pd.DataFrame()
        
        result = integrator.integrate(
            demographics_data=empty_df,
            education_data=empty_df,
            employment_data=empty_df,
            housing_data=empty_df,
            seifa_data=empty_df
        )
        
        # Should return empty result with proper schema
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        
    def test_missing_join_keys(self, integrator, sample_demographics_data):
        """Test handling of datasets with missing join keys."""
        # Remove join keys from education data
        bad_education_data = pd.DataFrame({
            'some_column': [1, 2, 3],
            'education_indicator': [100, 200, 300]
        })
        
        result = integrator.integrate(
            demographics_data=sample_demographics_data,
            education_data=bad_education_data,
            employment_data=pd.DataFrame(),
            housing_data=pd.DataFrame(),
            seifa_data=pd.DataFrame()
        )
        
        # Should handle gracefully and continue
        assert isinstance(result, pd.DataFrame)
        assert len(integrator.error_log) > 0
    
    def test_temporal_alignment(self, integrator):
        """Test temporal alignment across different census years."""
        # Create data with different census years
        demographics_2021 = pd.DataFrame({
            'geographic_id': ['101021001'],
            'geographic_level': ['SA2'],
            'census_year': [2021],
            'total_population': [5000]
        })
        
        seifa_2016 = pd.DataFrame({
            'geographic_id': ['101021001'],
            'geographic_level': ['SA2'],
            'census_year': [2016],  # Different year
            'irsad_score': [1050]
        })
        
        result = integrator.integrate(
            demographics_data=demographics_2021,
            education_data=pd.DataFrame(),
            employment_data=pd.DataFrame(),
            housing_data=pd.DataFrame(),
            seifa_data=seifa_2016
        )
        
        # Should align to primary year (2021)
        assert result['census_year'].iloc[0] == 2021
        # Check for SEIFA data integration (may have suffix)
        seifa_cols = [col for col in result.columns if 'irsad' in col or 'seifa' in col]
        assert len(seifa_cols) > 0, f"No SEIFA columns found in: {list(result.columns)}"
    
    def test_data_quality_validation(
        self, integrator, sample_demographics_data, sample_education_data,
        sample_employment_data, sample_housing_data, sample_seifa_data
    ):
        """Test data quality validation and metrics calculation."""
        result = integrator.integrate(
            demographics_data=sample_demographics_data,
            education_data=sample_education_data,
            employment_data=sample_employment_data,
            housing_data=sample_housing_data,
            seifa_data=sample_seifa_data
        )
        
        # Check quality metrics
        quality_metrics = integrator.get_quality_metrics()
        assert isinstance(quality_metrics, DataQualityMetrics)
        
        # Should have completeness scores for all datasets
        assert len(quality_metrics.completeness_scores) > 0
        
        # Should have join success rates
        assert len(quality_metrics.join_success_rates) > 0
        
        # Overall quality score should be calculated
        overall_score = quality_metrics.calculate_overall_quality_score()
        assert 0 <= overall_score <= 100
    
    def test_derived_indicator_calculation(
        self, integrator, sample_demographics_data, sample_education_data,
        sample_employment_data, sample_housing_data, sample_seifa_data
    ):
        """Test calculation of derived cross-domain indicators."""
        result = integrator.integrate(
            demographics_data=sample_demographics_data,
            education_data=sample_education_data,
            employment_data=sample_employment_data,
            housing_data=sample_housing_data,
            seifa_data=sample_seifa_data
        )
        
        # Check for derived indicators (using actual column names from integration)
        derived_indicators = [
            'livability_index',
            'economic_opportunity_score',
            'social_cohesion_index',
            'housing_market_pressure',
            'socioeconomic_profile'
        ]
        
        for indicator in derived_indicators:
            assert indicator in result.columns
    
    def test_parallel_vs_sequential_joins(
        self, integrator, sample_demographics_data, sample_education_data,
        sample_employment_data, sample_housing_data, sample_seifa_data
    ):
        """Test both parallel and sequential join strategies."""
        # Test parallel joins
        integrator.config.parallel_joins = True
        result_parallel = integrator.integrate(
            demographics_data=sample_demographics_data,
            education_data=sample_education_data,
            employment_data=sample_employment_data,
            housing_data=sample_housing_data,
            seifa_data=sample_seifa_data
        )
        
        # Reset integrator for sequential test
        integrator = CensusIntegrator()
        integrator.config.parallel_joins = False
        result_sequential = integrator.integrate(
            demographics_data=sample_demographics_data,
            education_data=sample_education_data,
            employment_data=sample_employment_data,
            housing_data=sample_housing_data,
            seifa_data=sample_seifa_data
        )
        
        # Results should be equivalent (allowing for minor ordering differences)
        assert len(result_parallel) == len(result_sequential)
        assert set(result_parallel.columns) == set(result_sequential.columns)
    
    def test_memory_optimization(self, integrator):
        """Test memory optimization features."""
        # Create large dataset to trigger optimization
        large_demographics = pd.DataFrame({
            'geographic_id': [f"10102100{i:03d}" for i in range(1000)],
            'geographic_level': ['SA2'] * 1000,
            'census_year': [2021] * 1000,
            'total_population': np.random.randint(1000, 10000, 1000),
            'males': np.random.randint(500, 5000, 1000),
            'females': np.random.randint(500, 5000, 1000)
        })
        
        # This should not raise memory errors
        result = integrator.integrate(
            demographics_data=large_demographics,
            education_data=pd.DataFrame(),
            employment_data=pd.DataFrame(),
            housing_data=pd.DataFrame(),
            seifa_data=pd.DataFrame()
        )
        
        assert len(result) == 1000
    
    def test_error_handling_stop_on_error_false(self, integrator):
        """Test error handling when stop_on_error is False."""
        integrator.stop_on_error = False
        
        # Create invalid data that should cause errors
        invalid_demographics = pd.DataFrame({
            'invalid_column': [1, 2, 3],
            'another_invalid': ['a', 'b', 'c']
        })
        
        # Should not raise exception, but continue with fallback
        result = integrator.integrate(
            demographics_data=invalid_demographics,
            education_data=pd.DataFrame(),
            employment_data=pd.DataFrame(),
            housing_data=pd.DataFrame(),
            seifa_data=pd.DataFrame()
        )
        
        # Should have error logs
        assert len(integrator.error_log) > 0
    
    def test_error_handling_stop_on_error_true(self, integrator):
        """Test error handling when stop_on_error is True."""
        integrator.stop_on_error = True
        
        # Create invalid data that should cause errors
        invalid_demographics = pd.DataFrame({
            'invalid_column': [1, 2, 3]
        })
        
        # Should raise TransformationError
        with pytest.raises(TransformationError):
            integrator.integrate(
                demographics_data=invalid_demographics,
                education_data=pd.DataFrame(),
                employment_data=pd.DataFrame(),
                housing_data=pd.DataFrame(),
                seifa_data=pd.DataFrame()
            )
    
    def test_performance_stats_tracking(
        self, integrator, sample_demographics_data, sample_education_data,
        sample_employment_data, sample_housing_data, sample_seifa_data
    ):
        """Test performance statistics tracking."""
        integrator.integrate(
            demographics_data=sample_demographics_data,
            education_data=sample_education_data,
            employment_data=sample_employment_data,
            housing_data=sample_housing_data,
            seifa_data=sample_seifa_data
        )
        
        performance_stats = integrator.get_performance_stats()
        assert isinstance(performance_stats, dict)
        assert 'join_times' in performance_stats
        assert len(performance_stats['join_times']) > 0
    
    def test_integration_includes_lat_long(
        self, integrator, sample_demographics_data, sample_education_data,
        sample_employment_data, sample_housing_data, sample_seifa_data
    ):
        """Test that integration includes latitude and longitude coordinates."""
        # Add latitude and longitude to the demographics data to simulate 
        # geographic transformer output
        sample_demographics_data_with_coords = sample_demographics_data.copy()
        sample_demographics_data_with_coords['latitude'] = [-33.8688, -33.8650, -33.8700]
        sample_demographics_data_with_coords['longitude'] = [151.2093, 151.2100, 151.2080]
        
        result = integrator.integrate(
            demographics_data=sample_demographics_data_with_coords,
            education_data=sample_education_data,
            employment_data=sample_employment_data,
            housing_data=sample_housing_data,
            seifa_data=sample_seifa_data
        )
        
        # Verify latitude and longitude columns are present
        assert 'latitude' in result.columns, "Latitude column missing from integrated data"
        assert 'longitude' in result.columns, "Longitude column missing from integrated data"
        
        # Verify latitude and longitude values are preserved and valid
        assert result['latitude'].notna().all(), "Latitude values should not be null"
        assert result['longitude'].notna().all(), "Longitude values should not be null"
        
        # Verify coordinates are within plausible range for Australia
        assert result['latitude'].between(-44.0, -10.0).all(), "Latitude values outside Australia range"
        assert result['longitude'].between(113.0, 154.0).all(), "Longitude values outside Australia range"
        
        # Verify the coordinate values match the input data
        expected_latitudes = [-33.8688, -33.8650, -33.8700]
        expected_longitudes = [151.2093, 151.2100, 151.2080]
        
        result_sorted = result.sort_values('geographic_id')
        assert result_sorted['latitude'].tolist() == expected_latitudes, "Latitude values not preserved correctly"
        assert result_sorted['longitude'].tolist() == expected_longitudes, "Longitude values not preserved correctly"


class TestCensusIntegratorHelpers:
    """Test suite for CensusIntegratorHelpers class."""
    
    def test_create_empty_dataset_structure(self):
        """Test creation of empty dataset structures."""
        demographics_empty = CensusIntegratorHelpers.create_empty_dataset_structure('demographics')
        assert isinstance(demographics_empty, pd.DataFrame)
        assert 'geographic_id' in demographics_empty.columns
        assert 'total_population' in demographics_empty.columns
        
        education_empty = CensusIntegratorHelpers.create_empty_dataset_structure('education')
        assert 'education_pop_base' in education_empty.columns
    
    def test_add_missing_join_keys(self):
        """Test addition of missing join keys."""
        data = pd.DataFrame({'some_column': [1, 2, 3]})
        missing_keys = ['geographic_id', 'census_year']
        
        result = CensusIntegratorHelpers.add_missing_join_keys(data, missing_keys)
        
        assert 'geographic_id' in result.columns
        assert 'census_year' in result.columns
        assert len(result) == 3
        assert result['census_year'].iloc[0] == 2021
    
    def test_optimize_dtypes_for_join(self):
        """Test data type optimization for joins."""
        data = pd.DataFrame({
            'geographic_id': ['101021001', '101021002'],
            'small_int': [1, 2],  # Should become uint8
            'large_int': [1000000, 2000000],  # Should stay int64/uint32
            'float_val': [1.5, 2.5],  # Should be optimized
            'category_like': ['A', 'A']  # Should become category
        })
        
        optimized = CensusIntegratorHelpers.optimize_dtypes_for_join(data)
        
        # Check optimizations
        assert optimized['small_int'].dtype == 'uint8'
        assert optimized['category_like'].dtype.name == 'category'
        assert optimized['geographic_id'].dtype == 'object'  # Should remain string
    
    def test_calculate_housing_stress_indicator(self):
        """Test housing stress indicator calculation."""
        data = pd.DataFrame({
            'median_rent_weekly': [400, 600, 300],
            'median_household_income': [60000, 80000, 40000],
            'median_mortgage_monthly': [2000, 3000, 1500],
            'owned_outright': [500, 800, 300],
            'owned_with_mortgage': [800, 1200, 400],
            'total_private_dwellings': [2000, 3000, 1000],
            'total_population': [5000, 8000, 3000]
        })
        
        stress_indicator = CensusIntegratorHelpers.calculate_housing_stress_indicator(data)
        
        assert isinstance(stress_indicator, pd.Series)
        assert len(stress_indicator) == 3
        assert not stress_indicator.isna().all()
    
    def test_calculate_area_development_index(self):
        """Test area development index calculation."""
        data = pd.DataFrame({
            'bachelor_degree': [500, 800, 300],
            'education_pop_base': [2000, 3000, 1200],
            'professionals': [400, 700, 200],
            'labour_force_pop': [1500, 2500, 1000],
            'internet_connection': [900, 1400, 600],
            'total_private_dwellings': [1000, 1500, 700],
            'irsad_score': [1050, 1200, 900]
        })
        
        dev_index = CensusIntegratorHelpers.calculate_area_development_index(data)
        
        assert isinstance(dev_index, pd.Series)
        assert len(dev_index) == 3
        assert not dev_index.isna().all()
    
    def test_validate_cross_domain_consistency(self):
        """Test cross-domain data consistency validation."""
        # Create data with intentional inconsistencies
        data = pd.DataFrame({
            'total_population': [5000, 8000, 3000],
            'males': [2000, 4000, 1500],  # Inconsistent with total
            'females': [2500, 4000, 1500],
            'education_pop_base': [6000, 8000, 3000],  # Should not exceed total
            'labour_force_pop': [4000, 8000, 3000],
            'irsad_score': [1050, 2500, 900]  # One score out of range
        })
        
        validated_data = CensusIntegratorHelpers.validate_cross_domain_consistency(data)
        
        assert '_cross_domain_validation_issues' in validated_data.columns
        assert validated_data['_cross_domain_validation_issues'].sum() > 0
    
    def test_calculate_record_completeness(self):
        """Test record-level completeness calculation."""
        # Complete record
        complete_record = pd.Series({
            'total_population': 5000,
            'geographic_id': '101021001',
            'census_year': 2021,
            'year_12_or_equivalent': 3000,
            'employed_full_time': 2000,
            'owned_outright': 800,
            'irsad_score': 1050
        })
        
        completeness = CensusIntegratorHelpers.calculate_record_completeness(complete_record)
        assert completeness == 100.0
        
        # Incomplete record
        incomplete_record = pd.Series({
            'total_population': 5000,
            'geographic_id': '101021001',
            'census_year': None,
            'year_12_or_equivalent': None,
            'employed_full_time': 2000,
            'owned_outright': None,
            'irsad_score': 1050
        })
        
        completeness = CensusIntegratorHelpers.calculate_record_completeness(incomplete_record)
        assert 0 < completeness < 100.0


class TestDataQualityMetrics:
    """Test suite for DataQualityMetrics class."""
    
    def test_data_quality_metrics_initialization(self):
        """Test proper initialization of DataQualityMetrics."""
        metrics = DataQualityMetrics()
        
        assert isinstance(metrics.completeness_scores, dict)
        assert isinstance(metrics.join_success_rates, dict)
        assert isinstance(metrics.temporal_alignment_scores, dict)
        assert isinstance(metrics.validation_pass_rates, dict)
        assert isinstance(metrics.conflict_resolution_counts, dict)
    
    def test_overall_quality_score_calculation(self):
        """Test overall quality score calculation."""
        metrics = DataQualityMetrics()
        
        # Add sample metrics
        metrics.completeness_scores = {'demographics': 95.0, 'education': 90.0}
        metrics.join_success_rates = {'demo_edu': 98.0, 'demo_housing': 95.0}
        metrics.temporal_alignment_scores = {'demographics': 100.0, 'seifa': 80.0}
        metrics.validation_pass_rates = {'cross_domain': 92.0}
        metrics.conflict_resolution_counts = {'population': 2, 'education': 1}
        
        overall_score = metrics.calculate_overall_quality_score()
        
        assert isinstance(overall_score, float)
        assert 0 <= overall_score <= 100
        assert overall_score > 0  # Should be positive with good metrics


class TestCensusIntegratorConfig:
    """Test suite for CensusIntegratorConfig class."""
    
    @patch('src.transformers.census.census_integrator.get_config')
    def test_config_initialization_with_defaults(self, mock_get_config):
        """Test configuration initialization with default values."""
        # Mock configuration values
        def mock_config(key, default):
            return default
        
        mock_get_config.side_effect = mock_config
        
        config = CensusIntegratorConfig(Mock())
        
        assert config.join_strategy == "left"
        assert config.join_keys == ["geographic_id", "geographic_level", "census_year"]
        assert config.parallel_joins == True
        assert config.temporal_tolerance_years == 1
        assert config.minimum_join_success_rate == 0.8
    
    @patch('src.transformers.census.census_integrator.get_config')
    def test_config_initialization_with_custom_values(self, mock_get_config):
        """Test configuration initialization with custom values."""
        custom_values = {
            "transformers.census.integration.join_strategy": "inner",
            "transformers.census.integration.parallel_joins": False,
            "transformers.census.integration.min_join_rate": 0.9
        }
        
        def mock_config(key, default):
            return custom_values.get(key, default)
        
        mock_get_config.side_effect = mock_config
        
        config = CensusIntegratorConfig(Mock())
        
        assert config.join_strategy == "inner"
        assert config.parallel_joins == False
        assert config.minimum_join_success_rate == 0.9


if __name__ == "__main__":
    pytest.main([__file__])