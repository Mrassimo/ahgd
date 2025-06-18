"""
Integration Tests for Dashboard UI Components

Tests the complete dashboard functionality including:
- Page navigation and state management
- Filter interactions and data updates
- UI responsiveness and performance
- Complete dashboard functionality
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dashboard.ui.sidebar import SidebarController
from src.dashboard.ui.pages import (
    render_geographic_health_explorer,
    render_correlation_analysis,
    render_health_hotspot_identification,
    render_predictive_risk_analysis,
    render_data_quality_methodology,
    get_page_renderer,
    render_page
)
from src.dashboard.ui.layout import LayoutManager
from src.dashboard.app import HealthAnalyticsDashboard


class TestSidebarController:
    """Test sidebar controls and state management"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.controller = SidebarController()
        self.test_data = pd.DataFrame({
            'STATE_NAME21': ['NSW', 'VIC', 'QLD', 'SA', 'WA'],
            'SA2_NAME21': ['Area1', 'Area2', 'Area3', 'Area4', 'Area5'],
            'IRSD_Score': [1000, 950, 1100, 900, 1050],
            'health_risk_score': [8.5, 9.2, 7.8, 10.1, 8.0]
        })
    
    def test_sidebar_controller_initialization(self):
        """Test sidebar controller initializes correctly"""
        assert self.controller.analysis_types == [
            "Geographic Health Explorer",
            "Correlation Analysis", 
            "Health Hotspot Identification",
            "Predictive Risk Analysis",
            "Data Quality & Methodology"
        ]
    
    @patch('streamlit.sidebar.selectbox')
    @patch('streamlit.sidebar.multiselect')
    @patch('streamlit.sidebar.header')
    def test_render_sidebar_controls(self, mock_header, mock_multiselect, mock_selectbox):
        """Test sidebar controls render correctly"""
        mock_selectbox.return_value = "Geographic Health Explorer"
        mock_multiselect.return_value = ['NSW', 'VIC']
        
        analysis_type, selected_states = self.controller.render_sidebar_controls(self.test_data)
        
        assert analysis_type == "Geographic Health Explorer"
        assert selected_states == ['NSW', 'VIC']
        mock_header.assert_called_once()
        mock_selectbox.assert_called_once()
        mock_multiselect.assert_called_once()
    
    def test_apply_state_filter(self):
        """Test state filtering works correctly"""
        selected_states = ['NSW', 'VIC']
        filtered_data = self.controller.apply_state_filter(self.test_data, selected_states)
        
        assert len(filtered_data) == 2
        assert set(filtered_data['STATE_NAME21']) == {'NSW', 'VIC'}
    
    def test_apply_state_filter_empty_selection(self):
        """Test empty state selection returns all data"""
        filtered_data = self.controller.apply_state_filter(self.test_data, [])
        
        assert len(filtered_data) == len(self.test_data)
        pd.testing.assert_frame_equal(filtered_data, self.test_data)


class TestPageRendering:
    """Test page rendering functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.test_data = pd.DataFrame({
            'STATE_NAME21': ['NSW', 'VIC', 'QLD', 'SA', 'WA'] * 20,
            'SA2_NAME21': [f'Area{i}' for i in range(100)],
            'IRSD_Score': np.random.normal(1000, 100, 100),
            'IRSD_Decile_Australia': np.random.randint(1, 11, 100),
            'health_risk_score': np.random.normal(8.5, 2.0, 100),
            'mortality_rate': np.random.normal(12.0, 3.0, 100),
            'diabetes_prevalence': np.random.normal(6.5, 1.5, 100),
            'geometry': [None] * 100  # Mock geometry for mapping
        })
    
    def test_get_page_renderer(self):
        """Test page renderer lookup works correctly"""
        renderer = get_page_renderer("Geographic Health Explorer")
        assert renderer == render_geographic_health_explorer
        
        renderer = get_page_renderer("Correlation Analysis")
        assert renderer == render_correlation_analysis
        
        renderer = get_page_renderer("Unknown Page")
        assert renderer == render_geographic_health_explorer  # Should fallback
    
    @patch('streamlit.header')
    @patch('streamlit.selectbox')
    @patch('streamlit.subheader')
    @patch('streamlit.dataframe')
    @patch('src.dashboard.visualisation.create_health_indicator_selector')
    @patch('src.dashboard.visualisation.display_key_metrics')
    @patch('src.dashboard.visualisation.create_health_risk_map')
    @patch('streamlit_folium.st_folium')
    def test_render_geographic_health_explorer(self, mock_folium, mock_map, mock_metrics, 
                                             mock_indicators, mock_dataframe, mock_subheader, 
                                             mock_selectbox, mock_header):
        """Test geographic health explorer page renders"""
        mock_indicators.return_value = {'health_risk_score': 'Health Risk Score'}
        mock_selectbox.return_value = 'health_risk_score'
        mock_map.return_value = Mock()
        
        # Should not raise exception
        render_geographic_health_explorer(self.test_data)
        
        mock_header.assert_called_once()
        mock_indicators.assert_called_once()
        mock_selectbox.assert_called_once()
        mock_metrics.assert_called_once()
        mock_map.assert_called_once()
    
    @patch('streamlit.header')
    @patch('streamlit.subheader')
    @patch('streamlit.plotly_chart')
    @patch('src.dashboard.data.loaders.calculate_correlations')
    @patch('src.dashboard.visualisation.create_correlation_heatmap')
    @patch('src.dashboard.visualisation.display_correlation_insights')
    @patch('src.dashboard.visualisation.create_scatter_plots')
    def test_render_correlation_analysis(self, mock_scatter, mock_insights, mock_heatmap,
                                       mock_correlations, mock_plotly, mock_subheader, mock_header):
        """Test correlation analysis page renders"""
        mock_correlations.return_value = (Mock(), Mock())
        mock_heatmap.return_value = Mock()
        mock_scatter.return_value = (Mock(), Mock())
        
        # Should not raise exception
        render_correlation_analysis(self.test_data)
        
        mock_header.assert_called_once()
        mock_correlations.assert_called_once()
        mock_heatmap.assert_called_once()
        mock_insights.assert_called_once()
        mock_scatter.assert_called_once()


class TestLayoutManager:
    """Test layout management functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.layout_manager = LayoutManager()
    
    def test_layout_manager_initialization(self):
        """Test layout manager initializes correctly"""
        assert self.layout_manager.default_column_gap == "medium"
    
    @patch('streamlit.title')
    @patch('streamlit.markdown')
    def test_create_header_section(self, mock_markdown, mock_title):
        """Test header section creation"""
        self.layout_manager.create_header_section("Test Title", "Test Description")
        
        mock_title.assert_called_once_with("Test Title")
        mock_markdown.assert_called_once_with("Test Description")
    
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    def test_create_metrics_row(self, mock_metric, mock_columns):
        """Test metrics row creation"""
        mock_columns.return_value = [Mock(), Mock(), Mock()]
        
        metrics = [
            {'label': 'Metric 1', 'value': '100', 'delta': '+5'},
            {'label': 'Metric 2', 'value': '200', 'delta': '-3'},
            {'label': 'Metric 3', 'value': '300', 'delta': '+10'}
        ]
        
        self.layout_manager.create_metrics_row(metrics)
        
        mock_columns.assert_called_once_with(3)
        assert mock_metric.call_count == 3


class TestHealthAnalyticsDashboard:
    """Test main dashboard application"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.app = HealthAnalyticsDashboard()
    
    def test_dashboard_initialization(self):
        """Test dashboard initializes correctly"""
        assert self.app.config is not None
        assert self.app.sidebar_controller is not None
        assert self.app.data is None
    
    @patch('src.dashboard.data.loaders.load_data')
    def test_load_application_data_success(self, mock_load_data):
        """Test successful data loading"""
        mock_data = pd.DataFrame({'test': [1, 2, 3]})
        mock_load_data.return_value = mock_data
        
        result = self.app.load_application_data()
        
        assert result is True
        pd.testing.assert_frame_equal(self.app.data, mock_data)
    
    @patch('src.dashboard.data.loaders.load_data')
    @patch('streamlit.error')
    def test_load_application_data_failure(self, mock_error, mock_load_data):
        """Test data loading failure"""
        mock_load_data.return_value = None
        
        result = self.app.load_application_data()
        
        assert result is False
        assert self.app.data is None
        mock_error.assert_called_once()
    
    @patch('streamlit.warning')
    def test_render_main_interface_empty_data(self, mock_warning):
        """Test main interface with empty filtered data"""
        self.app.data = pd.DataFrame({
            'STATE_NAME21': ['NSW', 'VIC'],
            'test_col': [1, 2]
        })
        
        with patch.object(self.app.sidebar_controller, 'render_sidebar_controls') as mock_sidebar:
            mock_sidebar.return_value = ("Geographic Health Explorer", [])
            
            with patch.object(self.app.sidebar_controller, 'apply_state_filter') as mock_filter:
                mock_filter.return_value = pd.DataFrame()  # Empty filtered data
                
                self.app.render_main_interface()
                
                mock_warning.assert_called_once()


class TestEndToEndIntegration:
    """Test complete end-to-end dashboard functionality"""
    
    @patch('streamlit.set_page_config')
    @patch('src.dashboard.data.loaders.load_data')
    @patch('streamlit.sidebar.selectbox')
    @patch('streamlit.sidebar.multiselect')
    @patch('streamlit.sidebar.header')
    def test_complete_dashboard_flow(self, mock_header, mock_multiselect, 
                                   mock_selectbox, mock_load_data, mock_page_config):
        """Test complete dashboard flow from initialization to rendering"""
        # Mock data loading
        test_data = pd.DataFrame({
            'STATE_NAME21': ['NSW', 'VIC', 'QLD'],
            'SA2_NAME21': ['Area1', 'Area2', 'Area3'],
            'IRSD_Score': [1000, 950, 1100],
            'IRSD_Decile_Australia': [5, 3, 7],
            'health_risk_score': [8.5, 9.2, 7.8],
            'mortality_rate': [12.0, 13.5, 11.2],
            'diabetes_prevalence': [6.5, 7.2, 5.8]
        })
        mock_load_data.return_value = test_data
        
        # Mock sidebar controls
        mock_selectbox.return_value = "Geographic Health Explorer"
        mock_multiselect.return_value = ['NSW', 'VIC']
        
        # Create and run dashboard
        app = HealthAnalyticsDashboard()
        
        # Test data loading
        assert app.load_application_data() is True
        assert app.data is not None
        assert len(app.data) == 3
        
        # Test sidebar controller
        analysis_type, selected_states = app.sidebar_controller.render_sidebar_controls(app.data)
        assert analysis_type == "Geographic Health Explorer"
        assert selected_states == ['NSW', 'VIC']
        
        # Test state filtering
        filtered_data = app.sidebar_controller.apply_state_filter(app.data, selected_states)
        assert len(filtered_data) == 2
        
        mock_page_config.assert_called_once()
        mock_load_data.assert_called_once()
    
    @pytest.mark.parametrize("analysis_type", [
        "Geographic Health Explorer",
        "Correlation Analysis",
        "Health Hotspot Identification",
        "Predictive Risk Analysis",
        "Data Quality & Methodology"
    ])
    def test_all_analysis_types_have_renderers(self, analysis_type):
        """Test all analysis types have corresponding renderers"""
        renderer = get_page_renderer(analysis_type)
        assert renderer is not None
        assert callable(renderer)
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms"""
        app = HealthAnalyticsDashboard()
        
        # Test with invalid data
        with patch('src.dashboard.data.loaders.load_data') as mock_load:
            mock_load.side_effect = Exception("Database connection failed")
            
            with patch('streamlit.error') as mock_error:
                result = app.load_application_data()
                assert result is False
                mock_error.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])