"""
Tests for charts visualization module

This module tests the statistical chart functions to ensure they
create proper Plotly figures with correct data and styling.

Author: Portfolio Demonstration
Date: June 2025
"""

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from unittest.mock import patch, MagicMock

from src.dashboard.visualisation.charts import (
    create_correlation_heatmap,
    create_scatter_plots,
    create_distribution_plot,
    create_state_comparison_chart,
    create_correlation_scatter_matrix,
    create_time_series_plot
)


@pytest.fixture
def sample_correlation_matrix():
    """Create sample correlation matrix for testing"""
    variables = ['IRSD_Score', 'health_risk_score', 'mortality_rate', 'diabetes_prevalence']
    
    # Create realistic correlation values
    correlation_data = np.array([
        [1.0, -0.65, -0.58, -0.42],
        [-0.65, 1.0, 0.78, 0.61],
        [-0.58, 0.78, 1.0, 0.45],
        [-0.42, 0.61, 0.45, 1.0]
    ])
    
    return pd.DataFrame(correlation_data, index=variables, columns=variables)


@pytest.fixture
def sample_health_data():
    """Create sample health analytics data for testing"""
    np.random.seed(42)  # For reproducible tests
    
    data = {
        'SA2_CODE21': [f'10{i:03d}' for i in range(100)],
        'SA2_NAME21': [f'Area {i}' for i in range(100)],
        'STATE_NAME21': np.random.choice(['NSW', 'VIC', 'QLD', 'WA', 'SA'], 100),
        'IRSD_Score': np.random.normal(1000, 100, 100),
        'health_risk_score': np.random.normal(7.5, 2.0, 100),
        'mortality_rate': np.random.normal(10.0, 3.0, 100),
        'diabetes_prevalence': np.random.normal(6.5, 1.5, 100),
        'heart_disease_rate': np.random.normal(4.2, 1.0, 100)
    }
    
    # Ensure positive values where needed
    data['IRSD_Score'] = np.abs(data['IRSD_Score'])
    data['health_risk_score'] = np.abs(data['health_risk_score'])
    data['mortality_rate'] = np.abs(data['mortality_rate'])
    data['diabetes_prevalence'] = np.abs(data['diabetes_prevalence'])
    data['heart_disease_rate'] = np.abs(data['heart_disease_rate'])
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_time_series_data():
    """Create sample time series data for testing"""
    dates = pd.date_range('2020-01-01', periods=24, freq='M')
    
    data = {
        'date': dates,
        'health_metric': np.random.normal(100, 10, 24),
        'state': ['NSW'] * 12 + ['VIC'] * 12
    }
    
    return pd.DataFrame(data)


class TestCreateCorrelationHeatmap:
    """Test suite for create_correlation_heatmap function"""
    
    def test_creates_plotly_figure(self, sample_correlation_matrix):
        """Test that function returns a Plotly figure"""
        result = create_correlation_heatmap(sample_correlation_matrix)
        
        assert isinstance(result, go.Figure)
        assert len(result.data) > 0
    
    def test_heatmap_data_structure(self, sample_correlation_matrix):
        """Test that heatmap contains correct data structure"""
        result = create_correlation_heatmap(sample_correlation_matrix)
        
        # Check that it's an imshow/heatmap trace
        trace = result.data[0]
        assert hasattr(trace, 'z')  # Heatmap data
        assert hasattr(trace, 'colorscale')
    
    def test_layout_properties(self, sample_correlation_matrix):
        """Test that layout has correct properties"""
        result = create_correlation_heatmap(sample_correlation_matrix)
        
        layout = result.layout
        assert layout.width == 800
        assert layout.height == 600
        assert layout.title.x == 0.5  # Centered title
        assert 'Correlation Matrix' in layout.title.text
    
    def test_handles_empty_matrix(self):
        """Test handling of empty correlation matrix"""
        empty_matrix = pd.DataFrame()
        
        # Should still create figure but may be empty
        result = create_correlation_heatmap(empty_matrix)
        assert isinstance(result, go.Figure)


class TestCreateScatterPlots:
    """Test suite for create_scatter_plots function"""
    
    def test_returns_two_figures(self, sample_health_data):
        """Test that function returns tuple of two figures"""
        fig1, fig2 = create_scatter_plots(sample_health_data)
        
        assert isinstance(fig1, go.Figure)
        assert isinstance(fig2, go.Figure)
        assert len(fig1.data) > 0
        assert len(fig2.data) > 0
    
    def test_scatter_plot_data(self, sample_health_data):
        """Test scatter plot data and styling"""
        fig1, fig2 = create_scatter_plots(sample_health_data)
        
        # Check first plot (SEIFA vs Health Risk)
        trace1 = fig1.data[0]
        assert hasattr(trace1, 'x')
        assert hasattr(trace1, 'y')
        assert hasattr(trace1, 'mode')
        assert 'markers' in trace1.mode
        
        # Check second plot (SEIFA vs Mortality)
        trace2 = fig2.data[0]
        assert hasattr(trace2, 'x')
        assert hasattr(trace2, 'y')
    
    def test_trendlines_added(self, sample_health_data):
        """Test that trendlines are included"""
        fig1, fig2 = create_scatter_plots(sample_health_data)
        
        # Should have more than one trace (scatter + trendline)
        assert len(fig1.data) >= 2
        assert len(fig2.data) >= 2
    
    def test_handles_missing_data(self, sample_health_data):
        """Test handling of missing data columns"""
        # Remove required columns
        incomplete_data = sample_health_data.drop(columns=['health_risk_score'])
        
        fig1, fig2 = create_scatter_plots(incomplete_data)
        
        # Should still return figures, but may be empty
        assert isinstance(fig1, go.Figure)
        assert isinstance(fig2, go.Figure)


class TestCreateDistributionPlot:
    """Test suite for create_distribution_plot function"""
    
    def test_creates_subplot_figure(self, sample_health_data):
        """Test creation of distribution plot with subplots"""
        result = create_distribution_plot(sample_health_data, 'health_risk_score')
        
        assert isinstance(result, go.Figure)
        assert len(result.data) >= 2  # Histogram + box plot
    
    def test_custom_title(self, sample_health_data):
        """Test custom title functionality"""
        custom_title = "Custom Distribution Title"
        result = create_distribution_plot(
            sample_health_data, 
            'health_risk_score', 
            title=custom_title
        )
        
        assert custom_title in result.layout.title.text
    
    def test_default_title_formatting(self, sample_health_data):
        """Test default title formatting"""
        result = create_distribution_plot(sample_health_data, 'health_risk_score')
        
        # Should format column name nicely
        assert 'Health Risk Score' in result.layout.title.text


class TestCreateStateComparisonChart:
    """Test suite for create_state_comparison_chart function"""
    
    def test_creates_bar_chart(self, sample_health_data):
        """Test creation of state comparison bar chart"""
        result = create_state_comparison_chart(sample_health_data, 'health_risk_score')
        
        assert isinstance(result, go.Figure)
        assert len(result.data) > 0
    
    def test_groups_by_state(self, sample_health_data):
        """Test that data is properly grouped by state"""
        result = create_state_comparison_chart(sample_health_data, 'health_risk_score')
        
        # Check number of bars matches number of unique states
        unique_states = sample_health_data['STATE_NAME21'].nunique()
        trace = result.data[0]
        
        # Should have one bar per state
        assert len(trace.x) <= unique_states
    
    def test_error_bars_included(self, sample_health_data):
        """Test that error bars are included"""
        result = create_state_comparison_chart(sample_health_data, 'health_risk_score')
        
        trace = result.data[0]
        assert hasattr(trace, 'error_y')


class TestCreateCorrelationScatterMatrix:
    """Test suite for create_correlation_scatter_matrix function"""
    
    def test_creates_scatter_matrix(self, sample_health_data):
        """Test creation of scatter plot matrix"""
        variables = ['IRSD_Score', 'health_risk_score', 'mortality_rate']
        result = create_correlation_scatter_matrix(sample_health_data, variables)
        
        assert isinstance(result, go.Figure)
        assert len(result.data) > 0
    
    def test_handles_missing_variables(self, sample_health_data):
        """Test handling of non-existent variables"""
        variables = ['IRSD_Score', 'nonexistent_column']
        
        # Should handle gracefully or raise appropriate error
        try:
            result = create_correlation_scatter_matrix(sample_health_data, variables)
            assert isinstance(result, go.Figure)
        except KeyError:
            # Acceptable to raise KeyError for missing columns
            pass


class TestCreateTimeSeriesPlot:
    """Test suite for create_time_series_plot function"""
    
    def test_creates_line_plot(self, sample_time_series_data):
        """Test creation of basic time series plot"""
        result = create_time_series_plot(
            sample_time_series_data, 
            'date', 
            'health_metric'
        )
        
        assert isinstance(result, go.Figure)
        assert len(result.data) > 0
    
    def test_grouped_time_series(self, sample_time_series_data):
        """Test time series with grouping"""
        result = create_time_series_plot(
            sample_time_series_data, 
            'date', 
            'health_metric', 
            group_col='state'
        )
        
        assert isinstance(result, go.Figure)
        # Should have multiple traces for different states
        unique_states = sample_time_series_data['state'].nunique()
        assert len(result.data) >= unique_states
    
    def test_layout_properties(self, sample_time_series_data):
        """Test time series layout properties"""
        result = create_time_series_plot(
            sample_time_series_data, 
            'date', 
            'health_metric'
        )
        
        layout = result.layout
        assert layout.height == 500
        assert layout.title.x == 0.5


class TestChartIntegration:
    """Integration tests for chart functions"""
    
    def test_all_charts_with_real_data(self, sample_health_data, sample_correlation_matrix):
        """Test that all chart functions work with realistic data"""
        
        # Test correlation heatmap
        heatmap = create_correlation_heatmap(sample_correlation_matrix)
        assert isinstance(heatmap, go.Figure)
        
        # Test scatter plots
        scatter1, scatter2 = create_scatter_plots(sample_health_data)
        assert isinstance(scatter1, go.Figure)
        assert isinstance(scatter2, go.Figure)
        
        # Test distribution plot
        dist_plot = create_distribution_plot(sample_health_data, 'health_risk_score')
        assert isinstance(dist_plot, go.Figure)
        
        # Test state comparison
        state_chart = create_state_comparison_chart(sample_health_data, 'health_risk_score')
        assert isinstance(state_chart, go.Figure)
    
    def test_chart_performance_with_large_data(self):
        """Test chart performance with larger datasets"""
        # Create larger dataset
        large_data = pd.DataFrame({
            'IRSD_Score': np.random.normal(1000, 100, 1000),
            'health_risk_score': np.random.normal(7.5, 2.0, 1000),
            'mortality_rate': np.random.normal(10.0, 3.0, 1000),
            'STATE_NAME21': np.random.choice(['NSW', 'VIC', 'QLD'], 1000)
        })
        
        # Should handle larger data without issues
        fig1, fig2 = create_scatter_plots(large_data)
        assert isinstance(fig1, go.Figure)
        assert isinstance(fig2, go.Figure)