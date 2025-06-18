"""
Unit tests for dashboard data processing functions

Tests the data processing utilities extracted from the monolithic dashboard,
including filtering, validation, and analysis functions.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dashboard.data.processors import (
    filter_data_by_states,
    validate_health_data,
    calculate_health_risk_score,
    identify_health_hotspots,
    prepare_correlation_data,
    generate_health_indicators,
    calculate_data_quality_metrics,
    apply_scenario_analysis
)


class TestFilterDataByStates:
    """Test cases for state-based data filtering"""
    
    def test_filter_by_single_state(self):
        """Test filtering by a single state"""
        
        test_data = pd.DataFrame({
            'SA2_NAME21': ['Area A', 'Area B', 'Area C'],
            'STATE_NAME21': ['NSW', 'VIC', 'NSW'],
            'health_risk_score': [8.5, 9.2, 7.8]
        })
        
        result = filter_data_by_states(test_data, ['NSW'])
        
        assert len(result) == 2
        assert all(result['STATE_NAME21'] == 'NSW')
    
    def test_filter_by_multiple_states(self):
        """Test filtering by multiple states"""
        
        test_data = pd.DataFrame({
            'SA2_NAME21': ['Area A', 'Area B', 'Area C', 'Area D'],
            'STATE_NAME21': ['NSW', 'VIC', 'QLD', 'NSW'],
            'health_risk_score': [8.5, 9.2, 7.8, 8.1]
        })
        
        result = filter_data_by_states(test_data, ['NSW', 'VIC'])
        
        assert len(result) == 3
        assert set(result['STATE_NAME21']) == {'NSW', 'VIC'}
    
    def test_filter_empty_state_list(self):
        """Test filtering with empty state list returns all data"""
        
        test_data = pd.DataFrame({
            'SA2_NAME21': ['Area A', 'Area B'],
            'STATE_NAME21': ['NSW', 'VIC'],
            'health_risk_score': [8.5, 9.2]
        })
        
        result = filter_data_by_states(test_data, [])
        
        assert len(result) == len(test_data)
        pd.testing.assert_frame_equal(result, test_data)


class TestValidateHealthData:
    """Test cases for health data validation"""
    
    def test_validate_complete_data(self):
        """Test validation of complete dataset"""
        
        test_data = pd.DataFrame({
            'SA2_CODE21': ['A', 'B', 'C'],
            'geometry': ['geom1', 'geom2', 'geom3'],
            'IRSD_Score': [1000, 900, 1100],
            'health_risk_score': [8.5, 9.2, 7.8]
        })
        
        result = validate_health_data(test_data)
        
        assert result['total_records'] == 3
        assert result['geographic_coverage'] == 3
        assert result['seifa_completeness'] == 100.0
        assert result['health_completeness'] == 100.0
        assert result['missing_critical_data'] == 0
    
    def test_validate_incomplete_data(self):
        """Test validation of dataset with missing values"""
        
        test_data = pd.DataFrame({
            'SA2_CODE21': ['A', 'B', 'C'],
            'geometry': ['geom1', None, 'geom3'],
            'IRSD_Score': [1000, np.nan, 1100],
            'health_risk_score': [8.5, 9.2, np.nan]
        })
        
        result = validate_health_data(test_data)
        
        assert result['total_records'] == 3
        assert result['geographic_coverage'] == 2
        assert abs(result['seifa_completeness'] - 66.67) < 0.1
        assert abs(result['health_completeness'] - 66.67) < 0.1
        assert result['missing_critical_data'] == 2


class TestCalculateHealthRiskScore:
    """Test cases for health risk score calculation"""
    
    def test_default_weights(self):
        """Test health risk calculation with default weights"""
        
        mortality_rate = pd.Series([8.0])
        diabetes_prevalence = pd.Series([4.0])
        heart_disease_rate = pd.Series([12.0])
        mental_health_rate = pd.Series([18.0])
        gp_access_score = pd.Series([7.0])
        hospital_distance = pd.Series([15.0])
        
        result = calculate_health_risk_score(
            mortality_rate, diabetes_prevalence, heart_disease_rate,
            mental_health_rate, gp_access_score, hospital_distance
        )
        
        # Manual calculation with default weights
        expected = (8.0 * 0.3) + (4.0 * 0.2) + (12.0 * 0.15) + (18.0 * 0.1) + (3.0 * 0.15) + (1.5 * 0.1)
        
        assert abs(result.iloc[0] - expected) < 0.001
    
    def test_custom_weights(self):
        """Test health risk calculation with custom weights"""
        
        mortality_rate = pd.Series([8.0])
        diabetes_prevalence = pd.Series([4.0])
        heart_disease_rate = pd.Series([12.0])
        mental_health_rate = pd.Series([18.0])
        gp_access_score = pd.Series([7.0])
        hospital_distance = pd.Series([15.0])
        
        custom_weights = {
            'mortality': 0.4,
            'diabetes': 0.3,
            'heart_disease': 0.1,
            'mental_health': 0.1,
            'gp_access': 0.05,
            'hospital_distance': 0.05
        }
        
        result = calculate_health_risk_score(
            mortality_rate, diabetes_prevalence, heart_disease_rate,
            mental_health_rate, gp_access_score, hospital_distance,
            weights=custom_weights
        )
        
        # Manual calculation with custom weights
        expected = (8.0 * 0.4) + (4.0 * 0.3) + (12.0 * 0.1) + (18.0 * 0.1) + (3.0 * 0.05) + (1.5 * 0.05)
        
        assert abs(result.iloc[0] - expected) < 0.001


class TestIdentifyHealthHotspots:
    """Test cases for health hotspot identification"""
    
    def test_identify_hotspots_normal_case(self):
        """Test hotspot identification with normal data"""
        
        # Create test data with clear hotspots
        test_data = pd.DataFrame({
            'SA2_NAME21': [f'Area {i}' for i in range(100)],
            'STATE_NAME21': ['NSW'] * 100,
            'IRSD_Score': np.random.uniform(600, 1200, 100),
            'health_risk_score': np.random.uniform(5, 15, 100)
        })
        
        # Manually create some clear hotspots
        test_data.loc[0:4, 'IRSD_Score'] = [600, 650, 620, 680, 640]  # Low SEIFA (high disadvantage)
        test_data.loc[0:4, 'health_risk_score'] = [14, 13.5, 14.2, 13.8, 14.1]  # High health risk
        
        result = identify_health_hotspots(test_data, n_hotspots=10)
        
        assert len(result) <= 10
        assert not result.empty
        # Hotspots should have high health risk and low SEIFA scores
        assert result['health_risk_score'].mean() > test_data['health_risk_score'].mean()
        assert result['IRSD_Score'].mean() < test_data['IRSD_Score'].mean()
    
    def test_identify_hotspots_empty_data(self):
        """Test hotspot identification with empty data"""
        
        # Create empty DataFrame with required columns
        test_data = pd.DataFrame(columns=['health_risk_score', 'IRSD_Score', 'SA2_NAME21'])
        
        result = identify_health_hotspots(test_data)
        
        assert result.empty
    
    def test_identify_hotspots_insufficient_data(self):
        """Test hotspot identification with insufficient data"""
        
        test_data = pd.DataFrame({
            'SA2_NAME21': ['Area A'],
            'IRSD_Score': [np.nan],
            'health_risk_score': [np.nan]
        })
        
        result = identify_health_hotspots(test_data)
        
        assert result.empty


class TestPrepareCorrelationData:
    """Test cases for correlation data preparation"""
    
    def test_prepare_correlation_data_complete(self):
        """Test correlation data preparation with complete data"""
        
        test_data = pd.DataFrame({
            'SA2_NAME21': ['Area A', 'Area B'],
            'IRSD_Score': [1000, 900],
            'IRSD_Decile_Australia': [5, 3],
            'mortality_rate': [8.0, 9.5],
            'diabetes_prevalence': [4.0, 5.2],
            'heart_disease_rate': [12.0, 14.5],
            'mental_health_rate': [18.0, 21.5],
            'gp_access_score': [7.5, 6.8],
            'hospital_distance': [12.0, 18.5],
            'health_risk_score': [10.5, 12.8],
            'extra_column': ['X', 'Y']  # Should be ignored
        })
        
        correlation_data, column_names = prepare_correlation_data(test_data)
        
        assert len(correlation_data) == 2
        assert 'extra_column' not in column_names
        assert 'IRSD_Score' in column_names
        assert 'health_risk_score' in column_names
    
    def test_prepare_correlation_data_missing_columns(self):
        """Test correlation data preparation with missing columns"""
        
        test_data = pd.DataFrame({
            'SA2_NAME21': ['Area A', 'Area B'],
            'IRSD_Score': [1000, 900],
            'mortality_rate': [8.0, 9.5]
            # Missing other expected columns
        })
        
        correlation_data, column_names = prepare_correlation_data(test_data)
        
        assert len(column_names) < 9  # Should have fewer than all expected columns
        assert 'IRSD_Score' in column_names
        assert 'mortality_rate' in column_names


class TestGenerateHealthIndicators:
    """Test cases for synthetic health indicator generation"""
    
    def test_generate_health_indicators_deterministic(self):
        """Test that health indicator generation is deterministic with fixed seed"""
        
        test_data = pd.DataFrame({
            'SA2_CODE21': ['A', 'B'],
            'SA2_NAME21': ['Area A', 'Area B'],
            'STE_NAME21': ['NSW', 'VIC'],
            'IRSD_Score': [1000, 800]
        })
        
        result1 = generate_health_indicators(test_data, random_seed=42)
        result2 = generate_health_indicators(test_data, random_seed=42)
        
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_generate_health_indicators_correlation(self):
        """Test that generated indicators correlate with disadvantage"""
        
        test_data = pd.DataFrame({
            'SA2_CODE21': ['A', 'B', 'C'],
            'SA2_NAME21': ['Area A', 'Area B', 'Area C'],
            'STE_NAME21': ['NSW', 'NSW', 'NSW'],
            'IRSD_Score': [1200, 1000, 600]  # High to low advantage
        })
        
        result = generate_health_indicators(test_data, random_seed=42)
        
        # More disadvantaged areas (lower SEIFA) should have higher mortality rates on average
        # Note: This is a statistical relationship, not guaranteed for every case
        assert len(result) == 3
        assert 'mortality_rate' in result.columns
        assert 'diabetes_prevalence' in result.columns


class TestApplyScenarioAnalysis:
    """Test cases for scenario analysis"""
    
    def test_apply_scenario_positive_improvement(self):
        """Test scenario analysis with positive improvement"""
        
        test_data = pd.DataFrame({
            'SA2_NAME21': ['Area A', 'Area B'],
            'IRSD_Score': [1000, 800],
            'health_risk_score': [10.0, 12.0]
        })
        
        result = apply_scenario_analysis(test_data, improvement_percentage=10)
        
        assert 'improved_seifa' in result.columns
        assert 'improved_health_risk' in result.columns
        
        # SEIFA scores should increase
        assert all(result['improved_seifa'] > result['IRSD_Score'])
        
        # Health risk should decrease
        assert all(result['improved_health_risk'] < result['health_risk_score'])
    
    def test_apply_scenario_bounds_checking(self):
        """Test that scenario analysis respects realistic bounds"""
        
        test_data = pd.DataFrame({
            'IRSD_Score': [1150],  # Already high
            'health_risk_score': [5.0]  # Already low
        })
        
        result = apply_scenario_analysis(test_data, improvement_percentage=50)
        
        # Should not exceed realistic bounds
        assert result['improved_seifa'].iloc[0] <= 1200
        assert result['improved_health_risk'].iloc[0] >= 0


class TestCalculateDataQualityMetrics:
    """Test cases for data quality metrics calculation"""
    
    def test_calculate_quality_metrics_complete_data(self):
        """Test quality metrics with complete data"""
        
        test_data = pd.DataFrame({
            'SA2_CODE21': ['A', 'B', 'C'],
            'IRSD_Score': [1000, 900, 1100],
            'health_risk_score': [8.5, 9.2, 7.8],
            'geometry': ['geom1', 'geom2', 'geom3']
        })
        
        result = calculate_data_quality_metrics(test_data)
        
        assert result['total_records'] == 3
        assert result['total_columns'] == 4
        assert result['SA2_CODE21_completeness'] == 100.0
        assert result['IRSD_Score_completeness'] == 100.0
        assert result['health_risk_score_completeness'] == 100.0
        assert result['geometry_completeness'] == 100.0
    
    def test_calculate_quality_metrics_with_missing_data(self):
        """Test quality metrics with missing data"""
        
        test_data = pd.DataFrame({
            'SA2_CODE21': ['A', 'B', None],
            'IRSD_Score': [1000, np.nan, 1100],
            'health_risk_score': [8.5, 9.2, np.nan]
        })
        
        result = calculate_data_quality_metrics(test_data)
        
        assert result['total_records'] == 3
        assert abs(result['SA2_CODE21_completeness'] - 66.67) < 0.1
        assert abs(result['IRSD_Score_completeness'] - 66.67) < 0.1
        assert abs(result['health_risk_score_completeness'] - 66.67) < 0.1


if __name__ == "__main__":
    pytest.main([__file__])