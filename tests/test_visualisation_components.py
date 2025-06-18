"""
Tests for visualization components module

This module tests the reusable UI components and utility functions
to ensure they work correctly with Streamlit and format data properly.

Author: Portfolio Demonstration
Date: June 2025
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call

from src.dashboard.visualisation.components import (
    display_key_metrics,
    create_health_indicator_selector,
    format_health_indicator_name,
    display_correlation_insights,
    display_hotspot_card,
    create_data_quality_metrics,
    create_performance_metrics,
    format_number,
    create_data_filter_sidebar
)


@pytest.fixture
def sample_health_data():
    """Create sample health analytics data for testing"""
    np.random.seed(42)
    
    data = {
        'SA2_CODE21': [f'10{i:03d}' for i in range(50)],
        'SA2_NAME21': [f'Area {i}' for i in range(50)],
        'STATE_NAME21': np.random.choice(['NSW', 'VIC', 'QLD', 'WA'], 50),
        'IRSD_Score': np.random.normal(1000, 100, 50),
        'IRSD_Decile_Australia': np.random.randint(1, 11, 50),
        'health_risk_score': np.random.normal(7.5, 2.0, 50),
        'mortality_rate': np.random.normal(10.0, 3.0, 50),
        'diabetes_prevalence': np.random.normal(6.5, 1.5, 50)
    }
    
    # Ensure positive values
    for col in ['IRSD_Score', 'health_risk_score', 'mortality_rate', 'diabetes_prevalence']:
        data[col] = np.abs(data[col])
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_correlation_matrix():
    """Create sample correlation matrix for testing"""
    variables = ['IRSD_Score', 'health_risk_score', 'mortality_rate', 'diabetes_prevalence']
    
    correlation_data = np.array([
        [1.0, -0.65, -0.58, -0.42],
        [-0.65, 1.0, 0.78, 0.61],
        [-0.58, 0.78, 1.0, 0.45],
        [-0.42, 0.61, 0.45, 1.0]
    ])
    
    return pd.DataFrame(correlation_data, index=variables, columns=variables)


@pytest.fixture
def sample_hotspot_data():
    """Create sample hotspot data for testing"""
    return pd.Series({
        'SA2_NAME21': 'Test Area',
        'STATE_NAME21': 'NSW',
        'health_risk_score': 8.5,
        'IRSD_Score': 850.0,
        'IRSD_Decile_Australia': 2.0,
        'mortality_rate': 15.2,
        'diabetes_prevalence': 9.8
    })


class TestDisplayKeyMetrics:
    """Test suite for display_key_metrics function"""
    
    @patch('streamlit.metric')
    @patch('streamlit.columns')
    @patch('streamlit.subheader')
    def test_displays_metrics_with_title(self, mock_subheader, mock_columns, mock_metric, sample_health_data):
        """Test display of key metrics with custom title"""
        # Mock columns context manager
        mock_col1, mock_col2, mock_col3, mock_col4 = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2, mock_col3, mock_col4]
        
        display_key_metrics(sample_health_data, 'health_risk_score', 'Test Title')
        
        # Verify title was set
        mock_subheader.assert_called_once_with('Test Title')
        
        # Verify columns were created
        mock_columns.assert_called_once_with(4)
        
        # Verify metrics were called
        assert mock_metric.call_count >= 4
    
    @patch('streamlit.metric')
    @patch('streamlit.columns')
    @patch('streamlit.warning')
    def test_handles_empty_data(self, mock_warning, mock_columns, mock_metric):
        """Test handling of empty data"""
        empty_data = pd.DataFrame({'health_risk_score': []})
        
        display_key_metrics(empty_data, 'health_risk_score')
        
        mock_warning.assert_called_once()
        mock_metric.assert_not_called()
    
    @patch('streamlit.metric')
    @patch('streamlit.columns')
    @patch('streamlit.warning')
    def test_handles_missing_column(self, mock_warning, mock_columns, mock_metric, sample_health_data):
        """Test handling of missing indicator column"""
        display_key_metrics(sample_health_data, 'nonexistent_column')
        
        mock_warning.assert_called_once()
        mock_metric.assert_not_called()


class TestCreateHealthIndicatorSelector:
    """Test suite for create_health_indicator_selector function"""
    
    def test_returns_dictionary(self):
        """Test that function returns a dictionary"""
        result = create_health_indicator_selector()
        
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_contains_expected_indicators(self):
        """Test that dictionary contains expected health indicators"""
        result = create_health_indicator_selector()
        
        expected_keys = [
            'health_risk_score',
            'mortality_rate', 
            'diabetes_prevalence',
            'heart_disease_rate',
            'mental_health_rate',
            'gp_access_score',
            'hospital_distance'
        ]
        
        for key in expected_keys:
            assert key in result
    
    def test_values_are_formatted_names(self):
        """Test that values are properly formatted display names"""
        result = create_health_indicator_selector()
        
        # Check some specific mappings
        assert result['health_risk_score'] == 'Composite Health Risk Score'
        assert result['mortality_rate'] == 'Mortality Rate'
        assert result['diabetes_prevalence'] == 'Diabetes Prevalence'


class TestFormatHealthIndicatorName:
    """Test suite for format_health_indicator_name function"""
    
    def test_formats_known_indicators(self):
        """Test formatting of known health indicators"""
        assert format_health_indicator_name('health_risk_score') == 'Composite Health Risk Score'
        assert format_health_indicator_name('mortality_rate') == 'Mortality Rate'
        assert format_health_indicator_name('diabetes_prevalence') == 'Diabetes Prevalence'
    
    def test_formats_unknown_indicators(self):
        """Test formatting of unknown indicators falls back to title case"""
        result = format_health_indicator_name('unknown_indicator')
        assert result == 'Unknown Indicator'
        
        result = format_health_indicator_name('test_variable')
        assert result == 'Test Variable'


class TestDisplayCorrelationInsights:
    """Test suite for display_correlation_insights function"""
    
    @patch('streamlit.columns')
    @patch('streamlit.markdown')
    @patch('streamlit.write')
    @patch('streamlit.error')
    def test_displays_insights_successfully(self, mock_error, mock_write, mock_markdown, 
                                          mock_columns, sample_correlation_matrix):
        """Test successful display of correlation insights"""
        mock_col1, mock_col2 = MagicMock(), MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2]
        
        display_correlation_insights(sample_correlation_matrix)
        
        # Should not show error
        mock_error.assert_not_called()
        
        # Should create columns
        mock_columns.assert_called_once_with(2)
        
        # Should write insights
        assert mock_write.call_count >= 2
    
    @patch('streamlit.error')
    def test_handles_missing_target_variable(self, mock_error, sample_correlation_matrix):
        """Test handling when target variable is missing"""
        display_correlation_insights(sample_correlation_matrix, target_variable='missing_var')
        
        mock_error.assert_called_once()


class TestDisplayHotspotCard:
    """Test suite for display_hotspot_card function"""
    
    @patch('streamlit.expander')
    @patch('streamlit.columns')
    @patch('streamlit.markdown')
    @patch('streamlit.write')
    def test_displays_hotspot_card(self, mock_write, mock_markdown, mock_columns, 
                                  mock_expander, sample_hotspot_data):
        """Test display of hotspot card"""
        # Mock expander context manager
        mock_expander_context = MagicMock()
        mock_expander.return_value.__enter__ = MagicMock(return_value=mock_expander_context)
        mock_expander.return_value.__exit__ = MagicMock(return_value=None)
        
        # Mock columns
        mock_col1, mock_col2, mock_col3 = MagicMock(), MagicMock(), MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2, mock_col3]
        
        display_hotspot_card(sample_hotspot_data)
        
        # Verify expander was created with correct title
        assert mock_expander.call_count == 1
        expander_title = mock_expander.call_args[0][0]
        assert 'Test Area' in expander_title
        assert 'NSW' in expander_title
        
        # Verify columns were created
        mock_columns.assert_called_with(3)


class TestCreateDataQualityMetrics:
    """Test suite for create_data_quality_metrics function"""
    
    @patch('streamlit.metric')
    @patch('streamlit.columns')
    def test_displays_quality_metrics(self, mock_columns, mock_metric, sample_health_data):
        """Test display of data quality metrics"""
        mock_col1, mock_col2, mock_col3 = MagicMock(), MagicMock(), MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2, mock_col3]
        
        create_data_quality_metrics(sample_health_data)
        
        # Verify columns and metrics
        mock_columns.assert_called_once_with(3)
        assert mock_metric.call_count >= 1  # At least geographic coverage


class TestCreatePerformanceMetrics:
    """Test suite for create_performance_metrics function"""
    
    @patch('streamlit.metric')
    @patch('streamlit.columns')
    def test_displays_performance_metrics(self, mock_columns, mock_metric, sample_health_data):
        """Test display of performance metrics"""
        mock_col1, mock_col2 = MagicMock(), MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2]
        
        create_performance_metrics(sample_health_data)
        
        # Should display metrics if required columns exist
        if 'IRSD_Score' in sample_health_data.columns and 'health_risk_score' in sample_health_data.columns:
            mock_columns.assert_called_with(2)
            assert mock_metric.call_count >= 2


class TestFormatNumber:
    """Test suite for format_number function"""
    
    def test_formats_integers(self):
        """Test formatting of integer values"""
        assert format_number(1000) == "1,000"
        assert format_number(5) == "5"
        assert format_number(1234567) == "1,234,567"
    
    def test_formats_floats(self):
        """Test formatting of float values"""
        assert format_number(10.123) == "10.12"
        assert format_number(5.5) == "5.50"
        assert format_number(0.001, decimal_places=3) == "0.001"
    
    def test_handles_nan_values(self):
        """Test handling of NaN values"""
        assert format_number(np.nan) == "N/A"
        assert format_number(pd.NA) == "N/A"
    
    def test_custom_decimal_places(self):
        """Test custom decimal places"""
        assert format_number(10.12345, decimal_places=3) == "10.123"
        assert format_number(10.12345, decimal_places=1) == "10.1"


class TestCreateDataFilterSidebar:
    """Test suite for create_data_filter_sidebar function"""
    
    @patch('streamlit.sidebar.header')
    @patch('streamlit.sidebar.multiselect')
    def test_creates_sidebar_filters(self, mock_multiselect, mock_header, sample_health_data):
        """Test creation of sidebar filters"""
        # Mock state selection
        mock_multiselect.return_value = ['NSW', 'VIC']
        
        result = create_data_filter_sidebar(sample_health_data)
        
        # Verify header was created
        mock_header.assert_called_once()
        
        # Verify multiselect was called for states
        mock_multiselect.assert_called_once()
        
        # Verify filtered data is returned
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(sample_health_data)
    
    @patch('streamlit.sidebar.multiselect')
    def test_filters_data_by_state(self, mock_multiselect, sample_health_data):
        """Test that data is properly filtered by selected states"""
        # Mock selecting only NSW
        mock_multiselect.return_value = ['NSW']
        
        result = create_data_filter_sidebar(sample_health_data)
        
        # All returned rows should be NSW
        if len(result) > 0:
            assert all(result['STATE_NAME21'] == 'NSW')
    
    @patch('streamlit.sidebar.multiselect')
    def test_handles_no_state_column(self, mock_multiselect):
        """Test handling when STATE_NAME21 column is missing"""
        data_no_state = pd.DataFrame({'test_col': [1, 2, 3]})
        
        result = create_data_filter_sidebar(data_no_state)
        
        # Should return original data unchanged
        assert result.equals(data_no_state)
        
        # Should not try to create state filter
        mock_multiselect.assert_not_called()


class TestComponentsIntegration:
    """Integration tests for component functions"""
    
    def test_components_work_together(self, sample_health_data, sample_correlation_matrix):
        """Test that components work together without conflicts"""
        
        # Create health indicator selector
        indicators = create_health_indicator_selector()
        assert isinstance(indicators, dict)
        
        # Format indicator names
        formatted_name = format_health_indicator_name('health_risk_score')
        assert isinstance(formatted_name, str)
        
        # Format numbers
        formatted_num = format_number(sample_health_data['health_risk_score'].iloc[0])
        assert isinstance(formatted_num, str)
    
    @patch('streamlit.metric')
    @patch('streamlit.columns')
    @patch('streamlit.sidebar.multiselect')
    def test_full_dashboard_component_flow(self, mock_multiselect, mock_columns, 
                                         mock_metric, sample_health_data, sample_correlation_matrix):
        """Test full component workflow as used in dashboard"""
        # Mock Streamlit components
        mock_multiselect.return_value = ['NSW', 'VIC']
        mock_columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        
        # Simulate dashboard component usage
        filtered_data = create_data_filter_sidebar(sample_health_data)
        indicators = create_health_indicator_selector()
        selected_indicator = list(indicators.keys())[0]
        
        display_key_metrics(filtered_data, selected_indicator)
        display_correlation_insights(sample_correlation_matrix)
        create_data_quality_metrics(filtered_data)
        create_performance_metrics(filtered_data)
        
        # Verify no exceptions occurred and functions ran
        assert isinstance(filtered_data, pd.DataFrame)
        assert isinstance(indicators, dict)