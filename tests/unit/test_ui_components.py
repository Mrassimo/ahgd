"""
Unit Tests for Dashboard UI Components

Tests individual UI components in isolation:
- Sidebar controllers and filters
- Page rendering functions
- Layout management utilities
- Error handling and edge cases
"""

import pytest
import pandas as pd
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dashboard.ui.sidebar import (
    SidebarController,
    render_analysis_selector,
    render_state_filter,
    create_sidebar_header,
    add_sidebar_info
)
from src.dashboard.ui.pages import (
    get_page_renderer,
    render_geographic_health_explorer,
    render_correlation_analysis,
    render_health_hotspot_identification,
    render_predictive_risk_analysis,
    render_data_quality_methodology
)
from src.dashboard.ui.layout import (
    LayoutManager,
    create_dashboard_header,
    create_loading_spinner,
    create_dashboard_footer,
    format_large_number,
    create_responsive_columns,
    apply_container_styling
)


class TestSidebarController:
    """Test sidebar controller functionality"""
    
    def setup_method(self):
        """Setup test data"""
        self.test_data = pd.DataFrame({
            'STATE_NAME21': ['NSW', 'VIC', 'QLD', 'NSW', 'WA'],
            'IRSD_Score': [950, 1050, 900, 980, 1020],
            'health_risk_score': [8.5, 6.2, 9.1, 7.8, 6.5]
        })
    
    def test_sidebar_controller_init(self):
        """Test SidebarController initialization"""
        controller = SidebarController()
        
        expected_types = [
            "Geographic Health Explorer",
            "Correlation Analysis", 
            "Health Hotspot Identification",
            "Predictive Risk Analysis",
            "Data Quality & Methodology"
        ]
        
        assert controller.analysis_types == expected_types
    
    def test_apply_state_filter_with_selection(self):
        """Test state filter with specific selections"""
        controller = SidebarController()
        
        # Filter for NSW and VIC
        filtered_data = controller.apply_state_filter(self.test_data, ['NSW', 'VIC'])
        expected_states = ['NSW', 'VIC']
        
        assert len(filtered_data) == 3  # 2 NSW + 1 VIC
        assert all(state in expected_states for state in filtered_data['STATE_NAME21'])
    
    def test_apply_state_filter_empty_selection(self):
        """Test state filter with empty selection"""
        controller = SidebarController()
        
        # Empty selection should return original data
        filtered_data = controller.apply_state_filter(self.test_data, [])
        pd.testing.assert_frame_equal(filtered_data, self.test_data)
    
    def test_apply_state_filter_invalid_states(self):
        """Test state filter with invalid state names"""
        controller = SidebarController()
        
        # Filter for non-existent states
        filtered_data = controller.apply_state_filter(self.test_data, ['INVALID'])
        assert len(filtered_data) == 0
    
    def test_get_sidebar_state(self):
        """Test sidebar state retrieval"""
        controller = SidebarController()
        
        # Mock session state
        with patch.object(st, 'session_state', {'test_key': 'test_value'}):
            state = controller.get_sidebar_state()
            
            assert 'session_state_keys' in state
            assert 'sidebar_state' in state
            assert isinstance(state['session_state_keys'], list)
    
    @patch('streamlit.session_state', new_callable=dict)
    def test_reset_sidebar_state(self, mock_session_state):
        """Test sidebar state reset"""
        # Add some test keys
        mock_session_state.update({
            'sidebar_test': 'value',
            'state_filter': ['NSW'],
            'other_key': 'keep_this'
        })
        
        controller = SidebarController()
        controller.reset_sidebar_state()
        
        # Sidebar-related keys should be cleared
        assert 'sidebar_test' not in mock_session_state
        assert 'state_filter' not in mock_session_state
        # Other keys should remain
        assert 'other_key' in mock_session_state


class TestPageRenderers:
    """Test page rendering functions"""
    
    def setup_method(self):
        """Setup test data"""
        self.test_data = pd.DataFrame({
            'SA2_NAME21': ['Area A', 'Area B', 'Area C'],
            'STATE_NAME21': ['NSW', 'VIC', 'QLD'],
            'IRSD_Score': [950, 1050, 900],
            'IRSD_Decile_Australia': [3, 7, 2],
            'health_risk_score': [8.5, 6.2, 9.1],
            'mortality_rate': [12.3, 8.7, 13.8],
            'diabetes_prevalence': [7.2, 5.1, 8.9]
        })
    
    def test_get_page_renderer_valid_types(self):
        """Test page renderer retrieval for valid analysis types"""
        valid_types = [
            ("Geographic Health Explorer", render_geographic_health_explorer),
            ("Correlation Analysis", render_correlation_analysis),
            ("Health Hotspot Identification", render_health_hotspot_identification),
            ("Predictive Risk Analysis", render_predictive_risk_analysis),
            ("Data Quality & Methodology", render_data_quality_methodology)
        ]
        
        for analysis_type, expected_renderer in valid_types:
            renderer = get_page_renderer(analysis_type)
            assert renderer == expected_renderer
    
    def test_get_page_renderer_invalid_type(self):
        """Test page renderer retrieval for invalid analysis type"""
        renderer = get_page_renderer("Invalid Type")
        assert renderer == render_geographic_health_explorer  # Default
    
    def test_get_page_renderer_none_type(self):
        """Test page renderer retrieval for None type"""
        renderer = get_page_renderer(None)
        assert renderer == render_geographic_health_explorer  # Default
    
    @patch('streamlit.header')
    @patch('streamlit.selectbox')
    @patch('streamlit.subheader')
    @patch('streamlit.dataframe')
    def test_geographic_health_explorer_structure(self, mock_dataframe, mock_subheader,
                                                 mock_selectbox, mock_header):
        """Test Geographic Health Explorer page structure"""
        with patch('src.dashboard.ui.pages.create_health_indicator_selector') as mock_selector, \
             patch('src.dashboard.ui.pages.display_key_metrics') as mock_metrics, \
             patch('src.dashboard.ui.pages.create_health_risk_map') as mock_map, \
             patch('streamlit_folium.st_folium') as mock_folium:
            
            mock_selector.return_value = {'mortality_rate': 'Mortality Rate'}
            mock_selectbox.return_value = 'mortality_rate'
            mock_map.return_value = Mock()
            
            render_geographic_health_explorer(self.test_data)
            
            mock_header.assert_called_once()
            mock_selector.assert_called_once()
            mock_metrics.assert_called_once()
            mock_map.assert_called_once()
    
    @patch('streamlit.header')
    @patch('streamlit.subheader')
    @patch('streamlit.plotly_chart')
    def test_correlation_analysis_structure(self, mock_plotly_chart, mock_subheader, mock_header):
        """Test Correlation Analysis page structure"""
        with patch('src.dashboard.ui.pages.calculate_correlations') as mock_calc, \
             patch('src.dashboard.ui.pages.create_correlation_heatmap') as mock_heatmap, \
             patch('src.dashboard.ui.pages.display_correlation_insights') as mock_insights, \
             patch('src.dashboard.ui.pages.create_scatter_plots') as mock_scatter:
            
            mock_correlation_matrix = pd.DataFrame({'A': [1.0, 0.5], 'B': [0.5, 1.0]})
            mock_calc.return_value = (mock_correlation_matrix, self.test_data)
            mock_heatmap.return_value = Mock()
            mock_scatter.return_value = (Mock(), Mock())
            
            render_correlation_analysis(self.test_data)
            
            mock_header.assert_called_once()
            mock_calc.assert_called_once()
            mock_heatmap.assert_called_once()
            mock_insights.assert_called_once()
    
    @patch('streamlit.header')
    @patch('streamlit.subheader')
    @patch('streamlit.markdown')
    @patch('streamlit.metric')
    @patch('streamlit.columns')
    def test_health_hotspot_identification_structure(self, mock_columns, mock_metric,
                                                   mock_markdown, mock_subheader, mock_header):
        """Test Health Hotspot Identification page structure"""
        with patch('src.dashboard.ui.pages.identify_health_hotspots') as mock_hotspots:
            mock_columns.return_value = [Mock(), Mock(), Mock()]
            mock_hotspots.return_value = self.test_data
            
            render_health_hotspot_identification(self.test_data)
            
            mock_header.assert_called_once()
            mock_hotspots.assert_called_once()
            assert mock_metric.call_count >= 3  # At least 3 metrics displayed
    
    @patch('streamlit.header')
    @patch('streamlit.subheader')
    @patch('streamlit.slider')
    @patch('streamlit.button')
    @patch('streamlit.columns')
    def test_predictive_risk_analysis_structure(self, mock_columns, mock_button, mock_slider,
                                              mock_subheader, mock_header):
        """Test Predictive Risk Analysis page structure"""
        mock_columns.return_value = [Mock(), Mock()]
        mock_slider.return_value = 1000
        mock_button.return_value = False
        
        render_predictive_risk_analysis(self.test_data)
        
        mock_header.assert_called_once()
        assert mock_slider.call_count >= 4  # Multiple sliders for inputs
    
    @patch('streamlit.header')
    @patch('streamlit.subheader')
    @patch('streamlit.markdown')
    @patch('streamlit.columns')
    def test_data_quality_methodology_structure(self, mock_columns, mock_markdown,
                                              mock_subheader, mock_header):
        """Test Data Quality & Methodology page structure"""
        with patch('src.dashboard.ui.pages.create_data_quality_metrics') as mock_quality, \
             patch('src.dashboard.ui.pages.create_performance_metrics') as mock_performance:
            
            mock_columns.return_value = [Mock(), Mock()]
            
            render_data_quality_methodology(self.test_data)
            
            mock_header.assert_called_once()
            mock_quality.assert_called_once()
            mock_performance.assert_called_once()


class TestLayoutManager:
    """Test layout management utilities"""
    
    def test_layout_manager_init(self):
        """Test LayoutManager initialization"""
        layout = LayoutManager()
        assert layout.default_column_gap == "medium"
    
    @patch('streamlit.title')
    @patch('streamlit.markdown')
    def test_create_header_section(self, mock_markdown, mock_title):
        """Test header section creation"""
        layout = LayoutManager()
        layout.create_header_section("Test Title", "Test Description")
        
        mock_title.assert_called_once_with("Test Title")
        mock_markdown.assert_called_once_with("Test Description")
    
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    def test_create_metrics_row_default_columns(self, mock_metric, mock_columns):
        """Test metrics row creation with default column count"""
        mock_columns.return_value = [Mock(), Mock(), Mock()]
        
        layout = LayoutManager()
        metrics = [
            {'label': 'Test 1', 'value': '100'},
            {'label': 'Test 2', 'value': '200', 'delta': '10'}
        ]
        
        layout.create_metrics_row(metrics)
        
        mock_columns.assert_called_once_with(3)  # Default 3 columns
        assert mock_metric.call_count == 2
    
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    def test_create_metrics_row_custom_columns(self, mock_metric, mock_columns):
        """Test metrics row creation with custom column count"""
        mock_columns.return_value = [Mock(), Mock(), Mock(), Mock()]
        
        layout = LayoutManager()
        metrics = [{'label': 'Test', 'value': '100'}]
        
        layout.create_metrics_row(metrics, columns=4)
        
        mock_columns.assert_called_once_with(4)
    
    @patch('streamlit.columns')
    def test_create_two_column_layout(self, mock_columns):
        """Test two-column layout creation"""
        mock_col1, mock_col2 = Mock(), Mock()
        mock_columns.return_value = [mock_col1, mock_col2]
        
        layout = LayoutManager()
        
        def left_func():
            pass
        
        def right_func():
            pass
        
        col1, col2 = layout.create_two_column_layout(left_func, right_func, (2, 1))
        
        mock_columns.assert_called_once_with((2, 1))
        assert col1 == mock_col1
        assert col2 == mock_col2
    
    @patch('streamlit.tabs')
    def test_create_tabbed_layout(self, mock_tabs):
        """Test tabbed layout creation"""
        mock_tab1, mock_tab2 = Mock(), Mock()
        mock_tabs.return_value = [mock_tab1, mock_tab2]
        
        layout = LayoutManager()
        tabs = ['Tab 1', 'Tab 2']
        tab_contents = [Mock(), Mock()]
        
        layout.create_tabbed_layout(tabs, tab_contents)
        
        mock_tabs.assert_called_once_with(tabs)
    
    def test_create_tabbed_layout_mismatch(self):
        """Test tabbed layout with mismatched tabs and contents"""
        layout = LayoutManager()
        
        with pytest.raises(ValueError, match="Number of tabs and tab contents must match"):
            layout.create_tabbed_layout(['Tab 1'], [Mock(), Mock()])
    
    @patch('streamlit.markdown')
    def test_add_divider_default(self, mock_markdown):
        """Test default divider addition"""
        layout = LayoutManager()
        layout.add_divider()
        
        mock_markdown.assert_called_once_with("---")
    
    @patch('streamlit.markdown')
    def test_add_divider_thick(self, mock_markdown):
        """Test thick divider addition"""
        layout = LayoutManager()
        layout.add_divider("thick")
        
        assert mock_markdown.call_count == 2  # "---" and empty string
    
    @patch('streamlit.markdown')
    def test_add_divider_dotted(self, mock_markdown):
        """Test dotted divider addition"""
        layout = LayoutManager()
        layout.add_divider("dotted")
        
        mock_markdown.assert_called_once()
        call_args = mock_markdown.call_args
        assert 'unsafe_allow_html=True' in str(call_args)
    
    @patch('streamlit.info')
    @patch('streamlit.warning')
    @patch('streamlit.success')
    @patch('streamlit.error')
    def test_create_info_box_types(self, mock_error, mock_success, mock_warning, mock_info):
        """Test different info box types"""
        layout = LayoutManager()
        
        # Test all box types
        layout.create_info_box("Title", "Content", "info")
        mock_info.assert_called_once()
        
        layout.create_info_box("Title", "Content", "warning")
        mock_warning.assert_called_once()
        
        layout.create_info_box("Title", "Content", "success")
        mock_success.assert_called_once()
        
        layout.create_info_box("Title", "Content", "error")
        mock_error.assert_called_once()
    
    @patch('streamlit.expander')
    def test_create_expandable_section(self, mock_expander):
        """Test expandable section creation"""
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock()
        
        layout = LayoutManager()
        content_func = Mock()
        
        layout.create_expandable_section("Test Section", content_func, expanded=True)
        
        mock_expander.assert_called_once_with("Test Section", expanded=True)


class TestLayoutUtilityFunctions:
    """Test standalone layout utility functions"""
    
    @patch('streamlit.title')
    @patch('streamlit.markdown')
    def test_create_dashboard_header(self, mock_markdown, mock_title):
        """Test dashboard header creation"""
        create_dashboard_header()
        
        mock_title.assert_called_once()
        mock_markdown.assert_called_once()
        
        # Check content
        title_call = mock_title.call_args[0][0]
        assert "Australian Health Analytics Dashboard" in title_call
    
    @patch('streamlit.spinner')
    def test_create_loading_spinner_default(self, mock_spinner):
        """Test loading spinner with default message"""
        create_loading_spinner()
        mock_spinner.assert_called_once_with("Loading data...")
    
    @patch('streamlit.spinner')
    def test_create_loading_spinner_custom(self, mock_spinner):
        """Test loading spinner with custom message"""
        create_loading_spinner("Custom loading message")
        mock_spinner.assert_called_once_with("Custom loading message")
    
    @patch('streamlit.markdown')
    def test_create_dashboard_footer(self, mock_markdown):
        """Test dashboard footer creation"""
        create_dashboard_footer()
        
        assert mock_markdown.call_count == 2  # Divider and content
        
        # Check footer content
        calls = mock_markdown.call_args_list
        footer_content = calls[1][0][0]
        assert "Australian Health Analytics Dashboard" in footer_content
        assert "Portfolio Demonstration Project" in footer_content
    
    def test_format_large_number(self):
        """Test large number formatting"""
        assert format_large_number(1000) == "1,000"
        assert format_large_number(1234567) == "1,234,567"
        assert format_large_number(100) == "100"
    
    @patch('streamlit.columns')
    def test_create_responsive_columns_default(self, mock_columns):
        """Test responsive columns creation with defaults"""
        create_responsive_columns(3)
        mock_columns.assert_called_once_with(3, gap="medium")
    
    @patch('streamlit.columns')
    def test_create_responsive_columns_custom_gap(self, mock_columns):
        """Test responsive columns creation with custom gap"""
        create_responsive_columns(4, gap="large")
        mock_columns.assert_called_once_with(4, gap="large")
    
    @patch('streamlit.markdown')
    def test_apply_container_styling(self, mock_markdown):
        """Test container styling application"""
        apply_container_styling()
        
        mock_markdown.assert_called_once()
        call_args = mock_markdown.call_args
        assert 'unsafe_allow_html=True' in str(call_args)
        
        # Check CSS content
        css_content = call_args[0][0]
        assert '.main-container' in css_content
        assert '.metric-container' in css_content
        assert '.info-section' in css_content


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling in UI components"""
    
    def test_sidebar_controller_empty_data(self):
        """Test sidebar controller with empty data"""
        controller = SidebarController()
        empty_data = pd.DataFrame()
        
        # Should handle empty data gracefully
        filtered_data = controller.apply_state_filter(empty_data, ['NSW'])
        assert len(filtered_data) == 0
    
    def test_sidebar_controller_missing_column(self):
        """Test sidebar controller with missing STATE_NAME21 column"""
        controller = SidebarController()
        invalid_data = pd.DataFrame({'other_column': [1, 2, 3]})
        
        with pytest.raises(KeyError):
            controller.apply_state_filter(invalid_data, ['NSW'])
    
    def test_layout_manager_empty_metrics(self):
        """Test layout manager with empty metrics list"""
        layout = LayoutManager()
        
        with patch('streamlit.columns') as mock_columns:
            mock_columns.return_value = [Mock(), Mock(), Mock()]
            layout.create_metrics_row([])  # Empty metrics
            
            mock_columns.assert_called_once()
    
    def test_format_large_number_edge_cases(self):
        """Test number formatting edge cases"""
        assert format_large_number(0) == "0"
        assert format_large_number(-1000) == "-1,000"
        assert format_large_number(999) == "999"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])