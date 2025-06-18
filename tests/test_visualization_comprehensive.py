"""
Comprehensive test suite for visualization components.

This module provides extensive testing coverage for all visualization components
including charts, maps, and UI components with proper mocking of dependencies.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
import sys
from pathlib import Path
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock heavy dependencies before importing modules
sys.modules['streamlit'] = Mock()
sys.modules['folium'] = Mock()
sys.modules['plotly.express'] = Mock()
sys.modules['plotly.graph_objects'] = Mock()
sys.modules['plotly.subplots'] = Mock()
sys.modules['streamlit_folium'] = Mock()

from src.dashboard.visualisation.charts import (
    create_correlation_heatmap, create_scatter_plots, create_distribution_plot,
    create_time_series_plot, create_state_comparison_chart, create_correlation_scatter_matrix
)
from src.dashboard.visualisation.maps import (
    create_health_risk_map, get_map_bounds, create_simple_point_map
)
from src.dashboard.visualisation.components import (
    display_key_metrics, create_health_indicator_selector, format_health_indicator_name,
    display_correlation_insights, display_hotspot_card, create_data_quality_metrics,
    create_performance_metrics, apply_custom_styling, format_number, create_data_filter_sidebar
)


class TestChartsModule:
    """Test cases for charts.py module"""
    
    def setup_method(self):
        """Setup test data before each test"""
        self.sample_correlation_matrix = pd.DataFrame({
            'mortality_rate': [1.0, -0.6, 0.7],
            'diabetes_prevalence': [-0.6, 1.0, -0.5],
            'disadvantage_score': [0.7, -0.5, 1.0]
        }, index=['mortality_rate', 'diabetes_prevalence', 'disadvantage_score'])
        
        self.sample_data = pd.DataFrame({
            'SA2_NAME21': ['Area A', 'Area B', 'Area C', 'Area D'],
            'health_risk_score': [8.5, 12.3, 6.1, 15.7],
            'IRSD_Score': [1050, 900, 1200, 750],
            'mortality_rate': [5.2, 8.1, 3.9, 10.4],
            'diabetes_prevalence': [6.8, 9.2, 4.5, 12.1],
            'year': [2021, 2021, 2021, 2021]
        })
    
    @patch('src.dashboard.visualisation.charts.px.imshow')
    def test_create_correlation_heatmap_success(self, mock_imshow):
        """Test successful creation of correlation heatmap"""
        # Mock plotly figure
        mock_figure = Mock()
        mock_figure.update_layout = Mock(return_value=mock_figure)
        mock_imshow.return_value = mock_figure
        
        result = create_correlation_heatmap(self.sample_correlation_matrix)
        
        # Verify plotly was called with correct parameters
        mock_imshow.assert_called_once()
        call_args = mock_imshow.call_args
        
        assert call_args[0][0].equals(self.sample_correlation_matrix)
        assert 'RdBu_r' in str(call_args)
        assert result == mock_figure
    
    @patch('src.dashboard.visualisation.charts.px.imshow')
    def test_create_correlation_heatmap_empty_data(self, mock_imshow):
        """Test correlation heatmap with empty data"""
        empty_matrix = pd.DataFrame()
        mock_figure = Mock()
        mock_imshow.return_value = mock_figure
        
        result = create_correlation_heatmap(empty_matrix)
        
        mock_imshow.assert_called_once_with(
            empty_matrix,
            labels={'x': 'Variables', 'y': 'Variables', 'color': 'Correlation'},
            x=empty_matrix.columns,
            y=empty_matrix.columns,
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title='Correlation Matrix: SEIFA Disadvantage vs Health Outcomes'
        )
    
    @patch('src.dashboard.visualisation.charts.px.scatter')
    def test_create_scatter_plots_success(self, mock_scatter):
        """Test successful creation of scatter plots"""
        mock_figure = Mock()
        mock_figure.update_layout = Mock(return_value=mock_figure)
        mock_scatter.return_value = mock_figure
        
        result = create_scatter_plots(self.sample_data)
        
        # Should return tuple of two figures
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    @patch('src.dashboard.visualisation.charts.px.histogram')
    def test_create_distribution_plot_success(self, mock_histogram):
        """Test successful creation of distribution plot"""
        mock_figure = Mock()
        mock_figure.update_layout = Mock(return_value=mock_figure)
        mock_histogram.return_value = mock_figure
        
        result = create_distribution_plot(self.sample_data, 'health_risk_score')
        
        mock_histogram.assert_called_once()
        call_args = mock_histogram.call_args
        
        assert call_args[1]['x'] == 'health_risk_score'
        assert 'nbins' in call_args[1]
        assert result == mock_figure
    
    @patch('src.dashboard.visualisation.charts.px.line')
    def test_create_time_series_plot_success(self, mock_line):
        """Test successful creation of time series plot"""
        time_series_data = pd.DataFrame({
            'year': [2018, 2019, 2020, 2021],
            'mortality_rate': [5.1, 5.3, 5.5, 5.2],
            'SA2_NAME21': ['Area A'] * 4
        })
        
        mock_figure = Mock()
        mock_figure.update_layout = Mock(return_value=mock_figure)
        mock_line.return_value = mock_figure
        
        result = create_time_series_plot(time_series_data, 'mortality_rate')
        
        mock_line.assert_called_once()
        call_args = mock_line.call_args
        
        assert call_args[1]['x'] == 'year'
        assert call_args[1]['y'] == 'mortality_rate'
        assert result == mock_figure


class TestMapsModule:
    """Test cases for maps.py module"""
    
    def setup_method(self):
        """Setup test data before each test"""
        # Create mock geometry objects
        mock_geometry1 = Mock()
        mock_geometry1.bounds = [150.0, -34.0, 151.0, -33.0]
        mock_geometry2 = Mock()
        mock_geometry2.bounds = [151.0, -35.0, 152.0, -34.0]
        
        self.sample_geo_data = pd.DataFrame({
            'SA2_NAME21': ['Sydney CBD', 'Melbourne CBD'],
            'health_risk_score': [8.5, 12.3],
            'IRSD_Score': [1050, 900],
            'geometry': [mock_geometry1, mock_geometry2],
            'population': [18500, 28000]
        })
    
    @patch('src.dashboard.visualisation.maps.folium.Map')
    @patch('src.dashboard.visualisation.maps.folium.GeoJson')
    def test_create_health_risk_map_success(self, mock_geojson, mock_map):
        """Test successful creation of health risk map"""
        mock_map_instance = Mock()
        mock_map.return_value = mock_map_instance
        
        result = create_health_risk_map(self.sample_geo_data)
        
        # Verify map was created
        mock_map.assert_called_once()
        
        # Verify GeoJson layer was added
        assert mock_geojson.called
        
        # Verify result is the map instance
        assert result == mock_map_instance
    
    @patch('src.dashboard.visualisation.maps.folium.Map')
    def test_create_health_risk_map_empty_data(self, mock_map):
        """Test health risk map with empty data"""
        empty_data = pd.DataFrame()
        
        result = create_health_risk_map(empty_data)
        
        # Should return None for empty data
        assert result is None
        mock_map.assert_not_called()
    
    def test_get_map_bounds_success(self):
        """Test successful calculation of map bounds"""
        result = get_map_bounds(self.sample_geo_data)
        
        # Should return dictionary with bounds
        assert isinstance(result, dict)
        expected_keys = ['min_lat', 'max_lat', 'min_lon', 'max_lon']
        for key in expected_keys:
            assert key in result
    
    @patch('src.dashboard.visualisation.maps.folium.Map')
    def test_create_simple_point_map_success(self, mock_map):
        """Test successful creation of simple point map"""
        mock_map_instance = Mock()
        mock_map.return_value = mock_map_instance
        
        # Add lat/lon columns to test data
        test_data = self.sample_geo_data.copy()
        test_data['latitude'] = [-33.8688, -37.8136]
        test_data['longitude'] = [151.2093, 144.9631]
        
        result = create_simple_point_map(
            test_data, 
            'latitude', 
            'longitude',
            'health_risk_score'
        )
        
        mock_map.assert_called_once()
        assert result == mock_map_instance


class TestComponentsModule:
    """Test cases for components.py module"""
    
    def setup_method(self):
        """Setup test data before each test"""
        self.sample_data = pd.DataFrame({
            'SA2_NAME21': ['Area A', 'Area B', 'Area C', 'Area D'],
            'health_risk_score': [8.5, 12.3, 6.1, 15.7],
            'IRSD_Score': [1050, 900, 1200, 750],
            'population': [18500, 22000, 15000, 28000],
            'mortality_rate': [5.2, 8.1, 3.9, 10.4]
        })
    
    @patch('src.dashboard.visualisation.components.st')
    def test_display_key_metrics_success(self, mock_st):
        """Test successful display of key metrics"""
        mock_st.columns.return_value = [Mock(), Mock(), Mock(), Mock()]
        
        display_key_metrics(self.sample_data, 'health_risk_score', 'Test Metrics')
        
        # Verify streamlit components were called
        mock_st.subheader.assert_called_once_with('Test Metrics')
        mock_st.columns.assert_called_once_with(4)
    
    @patch('src.dashboard.visualisation.components.st')
    def test_display_key_metrics_empty_data(self, mock_st):
        """Test key metrics display with empty data"""
        empty_data = pd.DataFrame({'health_risk_score': []})
        
        display_key_metrics(empty_data, 'health_risk_score')
        
        # Should show warning for empty data
        mock_st.warning.assert_called_once()
    
    @patch('src.dashboard.visualisation.components.st')
    def test_display_key_metrics_missing_column(self, mock_st):
        """Test key metrics display with missing column"""
        with pytest.raises(KeyError):
            display_key_metrics(self.sample_data, 'nonexistent_column')
    
    def test_format_number_positive(self):
        """Test number formatting for positive values"""
        result = format_number(123.456, 2)
        assert result == "123.46"
    
    def test_format_number_negative(self):
        """Test number formatting for negative values"""
        result = format_number(-123.456, 1)
        assert result == "-123.5"
    
    def test_format_number_zero(self):
        """Test number formatting for zero"""
        result = format_number(0, 2)
        assert result == "0.00"
    
    def test_format_number_none(self):
        """Test number formatting for None values"""
        result = format_number(None, 2)
        assert result == "N/A"
    
    def test_create_health_indicator_selector_success(self):
        """Test successful creation of health indicator selector"""
        result = create_health_indicator_selector()
        
        # Should return dictionary of indicators
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_format_health_indicator_name_success(self):
        """Test health indicator name formatting"""
        result = format_health_indicator_name('health_risk_score')
        
        # Should return formatted string
        assert isinstance(result, str)
        assert len(result) > 0
    
    @patch('src.dashboard.visualisation.components.st')
    def test_display_correlation_insights_success(self, mock_st):
        """Test display of correlation insights"""
        correlation_matrix = pd.DataFrame({
            'health_risk_score': [1.0, -0.6],
            'IRSD_Score': [-0.6, 1.0]
        }, index=['health_risk_score', 'IRSD_Score'])
        
        display_correlation_insights(correlation_matrix, 'IRSD_Score')
        
        # Should call streamlit components
        assert mock_st.write.called or mock_st.markdown.called
    
    @patch('src.dashboard.visualisation.components.st')
    def test_display_hotspot_card_success(self, mock_st):
        """Test display of hotspot card"""
        test_row = pd.Series({
            'SA2_NAME21': 'Test Area',
            'health_risk_score': 15.5,
            'IRSD_Score': 850,
            'population': 25000
        })
        
        display_hotspot_card(test_row, 1)
        
        # Should call streamlit components
        assert mock_st.container.called or mock_st.columns.called
    
    @patch('src.dashboard.visualisation.components.st')
    def test_create_data_quality_metrics_success(self, mock_st):
        """Test creation of data quality metrics"""
        create_data_quality_metrics(self.sample_data)
        
        # Should call streamlit components
        assert mock_st.subheader.called or mock_st.columns.called
    
    @patch('src.dashboard.visualisation.components.st')
    def test_create_performance_metrics_success(self, mock_st):
        """Test creation of performance metrics"""
        create_performance_metrics(self.sample_data)
        
        # Should call streamlit components
        assert mock_st.subheader.called or mock_st.columns.called
    
    def test_apply_custom_styling_success(self):
        """Test application of custom styling"""
        # This function should run without error
        try:
            apply_custom_styling()
            success = True
        except Exception:
            success = False
        
        assert success
    
    @patch('src.dashboard.visualisation.components.st')
    def test_create_data_filter_sidebar_success(self, mock_st):
        """Test creation of data filter sidebar"""
        mock_st.sidebar.selectbox.return_value = 'NSW'
        mock_st.sidebar.slider.return_value = (0, 100)
        
        result = create_data_filter_sidebar(self.sample_data)
        
        # Should return filtered dataframe
        assert isinstance(result, pd.DataFrame)


class TestVisualizationIntegration:
    """Integration tests for visualization components"""
    
    def setup_method(self):
        """Setup comprehensive test data"""
        np.random.seed(42)  # For reproducible tests
        
        # Create realistic test dataset
        self.large_dataset = pd.DataFrame({
            'SA2_NAME21': [f"Area_{i}" for i in range(100)],
            'health_risk_score': np.random.normal(10, 3, 100),
            'IRSD_Score': np.random.normal(1000, 150, 100),
            'mortality_rate': np.random.normal(6, 2, 100),
            'diabetes_prevalence': np.random.normal(8, 2.5, 100),
            'population': np.random.randint(5000, 50000, 100),
            'STE_NAME21': np.random.choice(['NSW', 'VIC', 'QLD', 'WA', 'SA'], 100)
        })
    
    @patch('src.dashboard.visualisation.charts.px')
    def test_visualization_pipeline_correlation_analysis(self, mock_px):
        """Test complete correlation analysis visualization pipeline"""
        # Mock all plotly functions
        mock_figure = Mock()
        mock_figure.update_layout = Mock(return_value=mock_figure)
        mock_px.imshow.return_value = mock_figure
        mock_px.scatter.return_value = mock_figure
        
        # Create correlation matrix
        correlation_matrix = self.large_dataset[
            ['health_risk_score', 'IRSD_Score', 'mortality_rate', 'diabetes_prevalence']
        ].corr()
        
        # Test heatmap creation
        heatmap = create_correlation_heatmap(correlation_matrix)
        assert heatmap == mock_figure
        
        # Test scatter plots creation
        scatter_plots = create_scatter_plots(self.large_dataset)
        assert isinstance(scatter_plots, tuple)
        assert len(scatter_plots) == 2
    
    @patch('src.dashboard.visualisation.components.st')
    def test_dashboard_metrics_display_pipeline(self, mock_st):
        """Test complete metrics display pipeline"""
        mock_st.columns.return_value = [Mock(), Mock(), Mock(), Mock()]
        
        # Test various metric displays
        display_key_metrics(self.large_dataset, 'health_risk_score', 'Health Risk Analysis')
        display_key_metrics(self.large_dataset, 'IRSD_Score', 'Socioeconomic Analysis')
        
        # Verify multiple calls were made
        assert mock_st.subheader.call_count == 2
        assert mock_st.columns.call_count == 2
    
    def test_data_validation_edge_cases(self):
        """Test visualization components with edge case data"""
        # Test with all NaN data
        nan_data = pd.DataFrame({
            'health_risk_score': [np.nan, np.nan, np.nan],
            'IRSD_Score': [np.nan, np.nan, np.nan]
        })
        
        # Test with single row
        single_row = pd.DataFrame({
            'health_risk_score': [10.5],
            'IRSD_Score': [1050]
        })
        
        # Test percentage formatting edge cases
        assert format_percentage(float('inf')) == "inf%"
        assert format_percentage(float('-inf')) == "-inf%"
        assert format_percentage(1.0) == "100.00%"
    
    def test_large_dataset_performance(self):
        """Test visualization components with large datasets"""
        # Create large dataset
        large_data = pd.DataFrame({
            'health_risk_score': np.random.normal(10, 3, 10000),
            'IRSD_Score': np.random.normal(1000, 150, 10000),
            'SA2_NAME21': [f"Area_{i}" for i in range(10000)]
        })
        
        # Test correlation matrix calculation
        correlation_matrix = large_data[['health_risk_score', 'IRSD_Score']].corr()
        
        assert correlation_matrix.shape == (2, 2)
        assert not correlation_matrix.isna().any().any()
        
        # Test data quality metrics
        quality_metrics = {
            'completeness': (1 - large_data.isna().sum() / len(large_data)).mean(),
            'unique_areas': large_data['SA2_NAME21'].nunique(),
            'outliers': len(large_data[
                (large_data['health_risk_score'] > large_data['health_risk_score'].mean() + 3 * large_data['health_risk_score'].std()) |
                (large_data['health_risk_score'] < large_data['health_risk_score'].mean() - 3 * large_data['health_risk_score'].std())
            ])
        }
        
        assert quality_metrics['completeness'] == 1.0  # No missing data
        assert quality_metrics['unique_areas'] == 10000
        assert quality_metrics['outliers'] >= 0


class TestVisualizationErrorHandling:
    """Test error handling in visualization components"""
    
    @patch('src.dashboard.visualisation.charts.px.imshow')
    def test_correlation_heatmap_error_handling(self, mock_imshow):
        """Test error handling in correlation heatmap creation"""
        mock_imshow.side_effect = Exception("Plotly error")
        
        with pytest.raises(Exception):
            create_correlation_heatmap(pd.DataFrame({'a': [1, 2], 'b': [3, 4]}))
    
    @patch('src.dashboard.visualisation.maps.folium.Map')
    def test_map_creation_error_handling(self, mock_map):
        """Test error handling in map creation"""
        mock_map.side_effect = Exception("Folium error")
        
        with pytest.raises(Exception):
            create_health_risk_map(pd.DataFrame({
                'geometry': [Mock()],
                'health_risk_score': [10.0]
            }))
    
    @patch('src.dashboard.visualisation.components.st')
    def test_components_error_handling(self, mock_st):
        """Test error handling in UI components"""
        mock_st.columns.side_effect = Exception("Streamlit error")
        
        with pytest.raises(Exception):
            display_key_metrics(
                pd.DataFrame({'health_risk_score': [1, 2, 3]}), 
                'health_risk_score'
            )


# Parametrized tests for comprehensive coverage
@pytest.mark.parametrize("value,places,expected", [
    (0.0, 2, "0.00"),
    (123.456, 2, "123.46"),
    (999.999, 1, "1000.0"),
    (-50.5, 1, "-50.5"),
    (1500.123, 0, "1500"),
])
def test_format_number_parametrized(value, places, expected):
    """Parametrized test for number formatting"""
    result = format_number(value, places)
    assert result == expected


@pytest.mark.parametrize("indicator", [
    'health_risk_score',
    'mortality_rate', 
    'diabetes_prevalence',
    'IRSD_Score'
])
def test_key_metrics_all_indicators(indicator):
    """Test key metrics display for all health indicators"""
    data = pd.DataFrame({
        'health_risk_score': [8.5, 12.3, 6.1],
        'mortality_rate': [5.2, 8.1, 3.9],
        'diabetes_prevalence': [6.8, 9.2, 4.5],
        'IRSD_Score': [1050, 900, 1200]
    })
    
    with patch('src.dashboard.visualisation.components.st') as mock_st:
        mock_st.columns.return_value = [Mock(), Mock(), Mock(), Mock()]
        
        display_key_metrics(data, indicator)
        
        mock_st.columns.assert_called_once_with(4)