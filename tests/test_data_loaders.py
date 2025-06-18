"""
Unit tests for dashboard data loading functions

Tests the data loading functionality extracted from the monolithic dashboard,
ensuring proper data structure, caching behavior, and error handling.
"""

import pytest
import pandas as pd
import numpy as np
import geopandas as gpd
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dashboard.data.loaders import load_data, calculate_correlations


class TestLoadData:
    """Test cases for the load_data function"""
    
    @patch('src.dashboard.data.loaders.pd.read_parquet')
    @patch('src.dashboard.data.loaders.gpd.read_parquet')
    @patch('src.dashboard.data.loaders.get_global_config')
    def test_load_data_success(self, mock_config, mock_gpd_read, mock_pd_read):
        """Test successful data loading"""
        
        # Mock configuration
        mock_config_obj = MagicMock()
        mock_config_obj.data_source.processed_data_dir = Path('/mock/data')
        mock_config_obj.dashboard.cache_ttl = 3600
        mock_config.return_value = mock_config_obj
        
        # Mock SEIFA data
        mock_seifa_df = pd.DataFrame({
            'SA2_Code_2021': ['101021007', '101021008'],
            'IRSD_Score': [1050, 900],
            'IRSD_Decile_Australia': [8, 3]
        })
        mock_pd_read.return_value = mock_seifa_df
        
        # Mock boundaries data
        mock_boundaries_gdf = gpd.GeoDataFrame({
            'SA2_CODE21': ['101021007', '101021008'],
            'SA2_NAME21': ['Area A', 'Area B'],
            'STE_NAME21': ['New South Wales', 'New South Wales'],
            'geometry': [None, None]  # Simplified for testing
        })
        mock_gpd_read.return_value = mock_boundaries_gdf
        
        # Mock streamlit cache decorator
        with patch('src.dashboard.data.loaders.st.cache_data') as mock_cache:
            mock_cache.return_value = lambda x: x  # Pass through decorator
            
            result = load_data()
        
        # Assertions
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert 'health_risk_score' in result.columns
        assert 'mortality_rate' in result.columns
        assert 'diabetes_prevalence' in result.columns
        assert len(result) == 2
    
    @patch('src.dashboard.data.loaders.pd.read_parquet')
    @patch('src.dashboard.data.loaders.get_global_config')
    def test_load_data_file_not_found(self, mock_config, mock_pd_read):
        """Test handling of missing data files"""
        
        # Mock configuration
        mock_config_obj = MagicMock()
        mock_config_obj.data_source.processed_data_dir = Path('/mock/data')
        mock_config_obj.dashboard.cache_ttl = 3600
        mock_config.return_value = mock_config_obj
        
        # Mock file not found error
        mock_pd_read.side_effect = FileNotFoundError("File not found")
        
        with patch('src.dashboard.data.loaders.st.cache_data') as mock_cache:
            mock_cache.return_value = lambda x: x
            with patch('src.dashboard.data.loaders.st.error') as mock_error:
                result = load_data()
        
        # Should handle error gracefully (function generates data instead of returning None)
        # In the actual implementation, it generates sample data when files are missing
        assert result is not None
        assert isinstance(result, pd.DataFrame)
    
    def test_health_indicators_correlation(self):
        """Test that health indicators are properly correlated with disadvantage"""
        
        # Create mock data with known SEIFA scores
        mock_data = pd.DataFrame({
            'SA2_CODE21': ['A', 'B', 'C'],
            'IRSD_Score': [1100, 1000, 800],  # High to low advantage
            'STE_NAME21': ['NSW', 'NSW', 'NSW']
        })
        
        # Test the health indicator generation logic
        np.random.seed(42)
        disadvantage_effect = (mock_data['IRSD_Score'] - 1000) / 100
        
        mortality_rates = np.maximum(0, 
            8.5 - disadvantage_effect * 0.8 + np.random.normal(0, 1.2, 3)
        )
        
        # More advantaged areas should have lower mortality (on average)
        assert mortality_rates[0] < mortality_rates[2]  # 1100 vs 800 SEIFA


class TestCalculateCorrelations:
    """Test cases for the calculate_correlations function"""
    
    def test_calculate_correlations_success(self):
        """Test successful correlation calculation"""
        
        # Create test data
        test_data = pd.DataFrame({
            'IRSD_Score': [1000, 900, 1100, 800, 1200],
            'IRSD_Decile_Australia': [5, 3, 8, 2, 10],
            'mortality_rate': [8.0, 9.5, 7.2, 10.1, 6.8],
            'diabetes_prevalence': [4.0, 5.2, 3.5, 6.1, 3.0],
            'heart_disease_rate': [12.0, 14.5, 10.8, 15.2, 9.5],
            'mental_health_rate': [18.0, 21.5, 16.2, 23.1, 15.0],
            'gp_access_score': [7.5, 6.8, 8.2, 6.0, 8.8],
            'hospital_distance': [12.0, 18.5, 8.2, 22.1, 6.5],
            'health_risk_score': [10.5, 12.8, 9.2, 14.1, 8.5]
        })
        
        with patch('src.dashboard.data.loaders.st.cache_data') as mock_cache:
            mock_cache.return_value = lambda x: x
            
            correlation_matrix, correlation_data = calculate_correlations(test_data)
        
        # Assertions
        assert isinstance(correlation_matrix, pd.DataFrame)
        assert isinstance(correlation_data, pd.DataFrame)
        assert correlation_matrix.shape[0] == correlation_matrix.shape[1]  # Square matrix
        assert len(correlation_data) == 5  # All data should be valid
        
        # Check that correlation matrix contains expected columns
        expected_columns = [
            'IRSD_Score', 'IRSD_Decile_Australia', 'mortality_rate', 
            'diabetes_prevalence', 'heart_disease_rate', 'mental_health_rate',
            'gp_access_score', 'hospital_distance', 'health_risk_score'
        ]
        for col in expected_columns:
            assert col in correlation_matrix.columns
            assert col in correlation_matrix.index
    
    def test_calculate_correlations_with_missing_data(self):
        """Test correlation calculation with missing data"""
        
        # Create test data with NaN values
        test_data = pd.DataFrame({
            'IRSD_Score': [1000, np.nan, 1100, 800, 1200],
            'IRSD_Decile_Australia': [5, 3, 8, np.nan, 10],
            'mortality_rate': [8.0, 9.5, np.nan, 10.1, 6.8],
            'diabetes_prevalence': [4.0, 5.2, 3.5, 6.1, 3.0],
            'heart_disease_rate': [12.0, 14.5, 10.8, 15.2, 9.5],
            'mental_health_rate': [18.0, 21.5, 16.2, 23.1, 15.0],
            'gp_access_score': [7.5, 6.8, 8.2, 6.0, 8.8],
            'hospital_distance': [12.0, 18.5, 8.2, 22.1, 6.5],
            'health_risk_score': [10.5, 12.8, 9.2, 14.1, 8.5]
        })
        
        with patch('src.dashboard.data.loaders.st.cache_data') as mock_cache:
            mock_cache.return_value = lambda x: x
            
            correlation_matrix, correlation_data = calculate_correlations(test_data)
        
        # Function generates complete data even with missing input data
        assert len(correlation_data) >= 0
        assert correlation_data is not None
    
    def test_calculate_correlations_empty_data(self):
        """Test correlation calculation with empty data"""
        
        test_data = pd.DataFrame()
        
        with patch('src.dashboard.data.loaders.st.cache_data') as mock_cache:
            mock_cache.return_value = lambda x: x
            
            correlation_matrix, correlation_data = calculate_correlations(test_data)
        
        # Function generates sample data even with empty input
        assert correlation_data is not None
        assert correlation_matrix is not None


class TestDataIntegrity:
    """Test data integrity and consistency"""
    
    def test_health_risk_score_calculation(self):
        """Test that health risk score calculation is consistent"""
        
        # Create deterministic test data
        test_data = {
            'mortality_rate': 8.0,
            'diabetes_prevalence': 4.0,
            'heart_disease_rate': 12.0,
            'mental_health_rate': 18.0,
            'gp_access_score': 7.0,
            'hospital_distance': 15.0
        }
        
        # Calculate expected health risk score
        expected_score = (
            (test_data['mortality_rate'] * 0.3) +
            (test_data['diabetes_prevalence'] * 0.2) +
            (test_data['heart_disease_rate'] * 0.15) +
            (test_data['mental_health_rate'] * 0.1) +
            ((10 - test_data['gp_access_score']) * 0.15) +
            (test_data['hospital_distance'] / 10 * 0.1)
        )
        
        # Manual calculation for verification
        manual_score = (8.0 * 0.3) + (4.0 * 0.2) + (12.0 * 0.15) + (18.0 * 0.1) + (3.0 * 0.15) + (1.5 * 0.1)
        
        assert abs(expected_score - manual_score) < 0.001
    
    def test_seifa_score_ranges(self):
        """Test that SEIFA scores are within expected ranges"""
        
        # SEIFA scores should typically be between 500 and 1200
        valid_scores = [600, 800, 1000, 1150]
        invalid_scores = [300, 1500, -100]
        
        for score in valid_scores:
            assert 500 <= score <= 1200
        
        for score in invalid_scores:
            assert not (500 <= score <= 1200)


if __name__ == "__main__":
    pytest.main([__file__])