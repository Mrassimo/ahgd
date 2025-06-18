"""
Integration Tests for Complete Dashboard Modular Architecture

Tests the full dashboard functionality including:
- Application initialization and configuration
- Data loading and processing
- UI component integration
- Page navigation and rendering
- Error handling and recovery
"""

import pytest
import pandas as pd
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dashboard.app import HealthAnalyticsDashboard, create_dashboard_app
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
from src.dashboard.ui.layout import (
    LayoutManager,
    create_dashboard_header,
    create_loading_spinner,
    apply_container_styling
)


class TestDashboardIntegration:
    """Test complete dashboard integration"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create mock data
        self.mock_data = pd.DataFrame({
            'SA2_NAME21': ['Area A', 'Area B', 'Area C'],
            'STATE_NAME21': ['NSW', 'VIC', 'QLD'],
            'IRSD_Score': [950, 1050, 900],
            'IRSD_Decile_Australia': [3, 7, 2],
            'health_risk_score': [8.5, 6.2, 9.1],
            'mortality_rate': [12.3, 8.7, 13.8],
            'diabetes_prevalence': [7.2, 5.1, 8.9],
            'heart_disease_rate': [15.6, 11.2, 17.3],
            'mental_health_rate': [22.1, 18.4, 25.8],
            'gp_access_score': [6.5, 8.2, 5.1],
            'hospital_distance': [45.2, 12.8, 67.3]
        })
        
        # Mock Streamlit session state
        if 'session_state' not in st.__dict__:
            st.session_state = {}
    
    @patch('streamlit.set_page_config')
    @patch('src.dashboard.app.load_data')
    @patch('src.dashboard.app.get_global_config')
    def test_dashboard_initialization(self, mock_config, mock_load_data, mock_set_page_config):
        """Test dashboard application initialization"""
        # Setup mocks
        mock_config.return_value.dashboard.page_title = "Test Dashboard"
        mock_config.return_value.dashboard.page_icon = "🏥"
        mock_config.return_value.dashboard.layout = "wide"
        mock_config.return_value.app.debug = False
        
        mock_load_data.return_value = self.mock_data
        
        # Create dashboard app
        app = create_dashboard_app()
        
        # Verify initialization
        assert isinstance(app, HealthAnalyticsDashboard)
        assert isinstance(app.sidebar_controller, SidebarController)
        mock_set_page_config.assert_called_once()
    
    @patch('streamlit.set_page_config')
    @patch('src.dashboard.app.load_data')
    @patch('src.dashboard.app.get_global_config')
    def test_data_loading_success(self, mock_config, mock_load_data, mock_set_page_config):
        """Test successful data loading"""
        # Setup mocks
        mock_config.return_value.dashboard.page_title = "Test Dashboard"
        mock_config.return_value.dashboard.page_icon = "🏥"
        mock_config.return_value.dashboard.layout = "wide"
        
        mock_load_data.return_value = self.mock_data
        
        # Create and test dashboard
        app = create_dashboard_app()
        success = app.load_application_data()
        
        assert success is True
        assert app.data is not None
        pd.testing.assert_frame_equal(app.data, self.mock_data)
    
    @patch('streamlit.set_page_config')
    @patch('streamlit.error')
    @patch('src.dashboard.app.load_data')
    @patch('src.dashboard.app.get_global_config')
    def test_data_loading_failure(self, mock_config, mock_load_data, mock_error, mock_set_page_config):
        """Test data loading failure handling"""
        # Setup mocks
        mock_config.return_value.dashboard.page_title = "Test Dashboard"
        mock_config.return_value.dashboard.page_icon = "🏥"
        mock_config.return_value.dashboard.layout = "wide"
        
        mock_load_data.return_value = None
        
        # Create and test dashboard
        app = create_dashboard_app()
        success = app.load_application_data()
        
        assert success is False
        mock_error.assert_called_once()
    
    def test_sidebar_controller_initialization(self):
        """Test sidebar controller initialization"""
        sidebar = SidebarController()
        
        expected_types = [
            "Geographic Health Explorer",
            "Correlation Analysis", 
            "Health Hotspot Identification",
            "Predictive Risk Analysis",
            "Data Quality & Methodology"
        ]
        
        assert sidebar.analysis_types == expected_types
    
    @patch('streamlit.sidebar.selectbox')
    @patch('streamlit.sidebar.multiselect')
    @patch('streamlit.sidebar.header')
    def test_sidebar_controls_rendering(self, mock_header, mock_multiselect, mock_selectbox):
        """Test sidebar controls rendering"""
        # Setup mocks
        mock_selectbox.return_value = "Geographic Health Explorer"
        mock_multiselect.return_value = ['NSW', 'VIC']
        
        # Test sidebar rendering
        sidebar = SidebarController()
        analysis_type, selected_states = sidebar.render_sidebar_controls(self.mock_data)
        
        assert analysis_type == "Geographic Health Explorer"
        assert selected_states == ['NSW', 'VIC']
        mock_header.assert_called_once()
        mock_selectbox.assert_called_once()
        mock_multiselect.assert_called_once()
    
    def test_state_filter_application(self):
        """Test state filter application"""
        sidebar = SidebarController()
        
        # Test with selected states
        filtered_data = sidebar.apply_state_filter(self.mock_data, ['NSW', 'VIC'])
        expected_data = self.mock_data[self.mock_data['STATE_NAME21'].isin(['NSW', 'VIC'])]
        pd.testing.assert_frame_equal(filtered_data, expected_data)
        
        # Test with no selected states
        empty_data = sidebar.apply_state_filter(self.mock_data, [])
        pd.testing.assert_frame_equal(empty_data, self.mock_data)
    
    def test_page_renderer_mapping(self):
        """Test page renderer function mapping"""
        # Test all analysis types have renderers
        analysis_types = [
            "Geographic Health Explorer",
            "Correlation Analysis", 
            "Health Hotspot Identification",
            "Predictive Risk Analysis",
            "Data Quality & Methodology"
        ]
        
        for analysis_type in analysis_types:
            renderer = get_page_renderer(analysis_type)
            assert callable(renderer)
        
        # Test unknown analysis type defaults to geographic explorer
        unknown_renderer = get_page_renderer("Unknown Type")
        assert unknown_renderer == render_geographic_health_explorer
    
    @patch('streamlit.header')
    @patch('streamlit.selectbox')
    @patch('streamlit.subheader') 
    @patch('streamlit.dataframe')
    @patch('src.dashboard.ui.pages.create_health_indicator_selector')
    @patch('src.dashboard.ui.pages.display_key_metrics')
    @patch('src.dashboard.ui.pages.create_health_risk_map')
    @patch('streamlit_folium.st_folium')
    def test_geographic_health_explorer_rendering(self, mock_st_folium, mock_create_map, 
                                                 mock_display_metrics, mock_indicator_selector,
                                                 mock_dataframe, mock_subheader, mock_selectbox, mock_header):
        """Test Geographic Health Explorer page rendering"""
        # Setup mocks
        mock_indicator_selector.return_value = {'mortality_rate': 'Mortality Rate'}
        mock_selectbox.return_value = 'mortality_rate'
        mock_create_map.return_value = None  # Return None to trigger warning path
        
        # Test page rendering
        render_geographic_health_explorer(self.mock_data)
        
        mock_header.assert_called_once()
        mock_indicator_selector.assert_called_once()
        mock_display_metrics.assert_called_once()
        mock_create_map.assert_called_once()
    
    @patch('streamlit.header')
    @patch('src.dashboard.ui.pages.calculate_correlations')
    @patch('src.dashboard.ui.pages.create_correlation_heatmap')
    @patch('src.dashboard.ui.pages.display_correlation_insights')
    @patch('streamlit.plotly_chart')
    def test_correlation_analysis_rendering(self, mock_plotly_chart, mock_display_insights,
                                          mock_create_heatmap, mock_calculate_correlations,
                                          mock_header):
        """Test Correlation Analysis page rendering"""
        # Setup mocks
        mock_correlation_matrix = pd.DataFrame({'A': [1.0, 0.5], 'B': [0.5, 1.0]})
        mock_calculate_correlations.return_value = (mock_correlation_matrix, self.mock_data)
        mock_create_heatmap.return_value = Mock()
        
        # Test page rendering
        render_correlation_analysis(self.mock_data)
        
        mock_header.assert_called_once()
        mock_calculate_correlations.assert_called_once()
        mock_create_heatmap.assert_called_once()
        mock_display_insights.assert_called_once()
    
    @patch('streamlit.header')
    @patch('src.dashboard.ui.pages.identify_health_hotspots')
    @patch('streamlit.metric')
    def test_health_hotspot_identification_rendering(self, mock_metric, mock_identify_hotspots,
                                                   mock_header):
        """Test Health Hotspot Identification page rendering"""
        # Setup mocks
        hotspot_data = self.mock_data.copy()
        mock_identify_hotspots.return_value = hotspot_data
        
        # Test page rendering
        render_health_hotspot_identification(self.mock_data)
        
        mock_header.assert_called_once()
        mock_identify_hotspots.assert_called_once()
    
    @patch('streamlit.header')
    @patch('streamlit.slider')
    @patch('streamlit.button')
    def test_predictive_risk_analysis_rendering(self, mock_button, mock_slider, mock_header):
        """Test Predictive Risk Analysis page rendering"""
        # Setup mocks
        mock_slider.return_value = 1000
        mock_button.return_value = False
        
        # Test page rendering
        render_predictive_risk_analysis(self.mock_data)
        
        mock_header.assert_called_once()
        assert mock_slider.call_count >= 4  # Multiple sliders
    
    @patch('streamlit.header')
    @patch('src.dashboard.ui.pages.create_data_quality_metrics')
    @patch('src.dashboard.ui.pages.create_performance_metrics')
    def test_data_quality_methodology_rendering(self, mock_performance_metrics,
                                              mock_quality_metrics, mock_header):
        """Test Data Quality & Methodology page rendering"""
        # Test page rendering
        render_data_quality_methodology(self.mock_data)
        
        mock_header.assert_called_once()
        mock_quality_metrics.assert_called_once()
        mock_performance_metrics.assert_called_once()
    
    @patch('streamlit.error')
    def test_page_rendering_error_handling(self, mock_error):
        """Test error handling in page rendering"""
        # Test with invalid data
        invalid_data = pd.DataFrame()
        
        # This should handle the error gracefully
        render_page("Geographic Health Explorer", invalid_data)
        
        # Error should be caught and displayed
        mock_error.assert_called()
    
    def test_layout_manager_initialization(self):
        """Test layout manager initialization"""
        layout = LayoutManager()
        assert layout.default_column_gap == "medium"
    
    @patch('streamlit.title')
    @patch('streamlit.markdown')
    def test_layout_header_creation(self, mock_markdown, mock_title):
        """Test layout header creation"""
        layout = LayoutManager()
        layout.create_header_section("Test Title", "Test Description")
        
        mock_title.assert_called_once_with("Test Title")
        mock_markdown.assert_called_once_with("Test Description")
    
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    def test_layout_metrics_row(self, mock_metric, mock_columns):
        """Test metrics row creation"""
        # Create proper mock column context managers
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()  
        mock_col3 = MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2, mock_col3]
        
        layout = LayoutManager()
        metrics = [
            {'label': 'Metric 1', 'value': '100', 'delta': '5'},
            {'label': 'Metric 2', 'value': '200', 'delta': '10'}
        ]
        
        layout.create_metrics_row(metrics)
        
        mock_columns.assert_called_once_with(3)
        assert mock_metric.call_count == 2
    
    @patch('streamlit.title')
    @patch('streamlit.markdown')
    def test_dashboard_header_creation(self, mock_markdown, mock_title):
        """Test dashboard header creation"""
        create_dashboard_header()
        
        mock_title.assert_called_once()
        mock_markdown.assert_called_once()
    
    @patch('streamlit.spinner')
    def test_loading_spinner_creation(self, mock_spinner):
        """Test loading spinner creation"""
        create_loading_spinner("Test loading message")
        mock_spinner.assert_called_once_with("Test loading message")
    
    @patch('streamlit.markdown')
    def test_container_styling_application(self, mock_markdown):
        """Test container styling application"""
        apply_container_styling()
        mock_markdown.assert_called_once()
        
        # Check that CSS was applied
        call_args = mock_markdown.call_args
        assert 'unsafe_allow_html=True' in str(call_args)
    
    @patch('streamlit.set_page_config')
    @patch('streamlit.warning')
    @patch('streamlit.error')
    @patch('src.dashboard.app.load_data')
    @patch('src.dashboard.app.get_global_config')
    def test_complete_dashboard_workflow(self, mock_config, mock_load_data, 
                                       mock_error, mock_warning, mock_set_page_config):
        """Test complete dashboard workflow from initialization to rendering"""
        # Setup mocks
        mock_config.return_value.dashboard.page_title = "Test Dashboard"
        mock_config.return_value.dashboard.page_icon = "🏥"
        mock_config.return_value.dashboard.layout = "wide"
        mock_config.return_value.app.debug = False
        
        mock_load_data.return_value = self.mock_data
        
        # Create and run dashboard
        app = create_dashboard_app()
        
        # Test data loading
        success = app.load_application_data()
        assert success is True
        
        # Test sidebar controller
        assert isinstance(app.sidebar_controller, SidebarController)
        
        # Test configuration
        mock_set_page_config.assert_called_once()
    
    @patch('streamlit.set_page_config')
    @patch('streamlit.error')
    @patch('src.dashboard.app.load_data')
    @patch('src.dashboard.app.get_global_config')
    def test_error_recovery_workflow(self, mock_config, mock_load_data, 
                                   mock_error, mock_set_page_config):
        """Test error recovery and graceful degradation"""
        # Setup mocks for error scenario
        mock_config.return_value.dashboard.page_title = "Test Dashboard"
        mock_config.return_value.dashboard.page_icon = "🏥"
        mock_config.return_value.dashboard.layout = "wide"
        mock_config.return_value.app.debug = True
        
        mock_load_data.side_effect = Exception("Test error")
        
        # Create dashboard and test error handling
        app = create_dashboard_app()
        success = app.load_application_data()
        
        assert success is False
        mock_error.assert_called()


class TestLegacyCompatibility:
    """Test backward compatibility with legacy dashboard"""
    
    @patch('src.dashboard.app.main')
    def test_legacy_wrapper_delegation(self, mock_main):
        """Test that legacy wrapper properly delegates to new app"""
        from scripts.streamlit_dashboard import legacy_main
        
        legacy_main()
        mock_main.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])