"""
Main Application Entry Point for Australian Health Analytics Dashboard

This module coordinates all dashboard components:
- Data loading and processing
- UI rendering and layout management
- Page routing and navigation
- Error handling and logging
- Session state management
- Performance monitoring and optimization
"""

import streamlit as st
import sys
import warnings
from pathlib import Path
from typing import Optional

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import configuration and core modules
from src.config import get_global_config
from src.dashboard.data.loaders import load_data
from src.dashboard.visualisation import apply_custom_styling
from src.dashboard.ui.sidebar import SidebarController
from src.dashboard.ui.pages import render_page
from src.dashboard.ui.layout import (
    create_dashboard_header,
    create_dashboard_footer,
    create_loading_spinner,
    apply_container_styling
)

# Import performance monitoring modules
from src.performance.monitoring import get_performance_monitor, track_performance
from src.performance.cache import get_cache_manager
from src.performance.production import get_production_manager, ProductionConfig
from src.performance.dashboard import render_performance_sidebar


class HealthAnalyticsDashboard:
    """Main dashboard application class with performance monitoring"""
    
    def __init__(self):
        """Initialise dashboard application"""
        self.config = get_global_config()
        self.sidebar_controller = SidebarController()
        self.data = None
        
        # Initialize performance monitoring
        self.performance_monitor = get_performance_monitor()
        self.cache_manager = get_cache_manager()
        
        # Initialize production manager if in production
        if self.config.environment.value == "production":
            self.production_manager = get_production_manager()
        else:
            self.production_manager = None
        
        self.setup_page_config()
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title=self.config.dashboard.page_title,
            page_icon=self.config.dashboard.page_icon,
            layout=self.config.dashboard.layout,
            initial_sidebar_state="expanded"
        )
        
        # Apply custom CSS styling
        apply_custom_styling()
        apply_container_styling()
    
    @track_performance("load_application_data")
    def load_application_data(self) -> bool:
        """
        Load all required data for the dashboard with performance monitoring
        
        Returns:
            True if data loaded successfully, False otherwise
        """
        try:
            with self.performance_monitor.track_page_load("data_loading"):
                with create_loading_spinner("Loading Australian health and geographic data..."):
                    # Try to get from cache first
                    cache_key = "dashboard_main_data"
                    self.data = self.cache_manager.get(cache_key)
                    
                    if self.data is None:
                        # Load fresh data
                        self.data = load_data()
                        
                        # Cache the loaded data
                        if self.data is not None:
                            self.cache_manager.set(cache_key, self.data, ttl=1800)  # 30 minutes
            
            if self.data is None:
                st.error("Failed to load data. Please check data files are available.")
                return False
            
            # Track successful data load
            self.performance_monitor.add_custom_metric("data_load_success", 1, "application")
            return True
            
        except Exception as e:
            # Track data load failure
            self.performance_monitor.add_custom_metric("data_load_failure", 1, "application")
            
            st.error(f"Error loading data: {str(e)}")
            st.markdown("""
            **Troubleshooting:**
            - Ensure data files are available in the data directory
            - Check database connectivity
            - Verify file permissions
            - Try refreshing the page
            """)
            return False
    
    @track_performance("render_main_interface")
    def render_main_interface(self):
        """Render the main dashboard interface with performance monitoring"""
        with self.performance_monitor.track_page_load("main_interface"):
            # Create header
            create_dashboard_header()
            
            # Track session info
            self.performance_monitor.streamlit_collector.track_session_info()
            
            # Render sidebar controls and get selections
            with self.performance_monitor.track_page_load("sidebar_render"):
                analysis_type, selected_states = self.sidebar_controller.render_sidebar_controls(self.data)
                
                # Add performance monitoring sidebar
                render_performance_sidebar()
            
            # Track user interaction
            self.performance_monitor.track_user_interaction("page_change", analysis_type)
            
            # Apply filters to data
            with self.performance_monitor.track_page_load("data_filtering"):
                filtered_data = self.sidebar_controller.apply_state_filter(self.data, selected_states)
            
            # Validate filtered data
            if filtered_data.empty:
                st.warning("No data available for selected filters. Please adjust your selections.")
                return
            
            # Render selected page
            with self.performance_monitor.track_page_load(f"page_{analysis_type}"):
                render_page(analysis_type, filtered_data)
            
            # Create footer
            create_dashboard_footer()
    
    def handle_error_state(self, error: Exception):
        """
        Handle application errors gracefully with error tracking
        
        Args:
            error: Exception that occurred
        """
        # Track error occurrence
        self.performance_monitor.add_custom_metric("application_error", 1, "errors")
        
        st.error("Dashboard Error")
        st.markdown(f"""
        An error occurred while running the dashboard:
        
        **Error:** {str(error)}
        
        **Troubleshooting Steps:**
        1. Refresh the page
        2. Check your internet connection
        3. Verify data files are available
        4. Clear browser cache
        5. Contact support if issues persist
        """)
        
        # Add debug information in development
        if self.config.dashboard.debug:
            st.markdown("**Debug Information:**")
            st.code(str(error))
            
            # Show performance metrics in debug mode
            with st.expander("Performance Metrics"):
                summary = self.performance_monitor.get_performance_summary()
                st.json(summary)
    
    @track_performance("dashboard_run")
    def run(self):
        """Main application run method with performance monitoring"""
        try:
            # Initialize production optimizations if in production
            if self.production_manager:
                with self.production_manager.production_context():
                    return self._run_dashboard()
            else:
                return self._run_dashboard()
            
        except Exception as e:
            self.handle_error_state(e)
    
    def _run_dashboard(self):
        """Internal dashboard run method"""
        with self.performance_monitor.track_page_load("full_dashboard"):
            # Load data
            if not self.load_application_data():
                return
            
            # Render main interface
            self.render_main_interface()
            
            # Track dashboard session
            self.performance_monitor.add_custom_metric("dashboard_session", 1, "usage")


def create_dashboard_app() -> HealthAnalyticsDashboard:
    """
    Factory function to create dashboard application
    
    Returns:
        Configured dashboard application instance
    """
    return HealthAnalyticsDashboard()


def main():
    """Main entry point for the dashboard application"""
    try:
        app = create_dashboard_app()
        app.run()
        
    except Exception as e:
        st.error("Critical Error: Unable to start dashboard")
        st.markdown(f"Error details: {str(e)}")
        
        # Provide basic troubleshooting
        st.markdown("""
        **Critical Error Recovery:**
        1. Ensure all dependencies are installed
        2. Verify project structure is intact
        3. Check Python environment setup
        4. Contact system administrator
        """)


# Session state debugging utilities
def get_session_info() -> dict:
    """Get current session state information for debugging"""
    return {
        'session_state_keys': list(st.session_state.keys()),
        'session_id': getattr(st.session_state, 'session_id', 'unknown'),
        'data_loaded': 'data' in st.session_state,
        'total_keys': len(st.session_state.keys())
    }


def clear_session_state():
    """Clear all session state - useful for debugging"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# Error recovery utilities
def safe_render_with_fallback(render_func, fallback_message: str = "Content unavailable"):
    """
    Safely render content with fallback on error
    
    Args:
        render_func: Function to render content
        fallback_message: Message to display on error
    """
    try:
        render_func()
    except Exception as e:
        st.warning(f"{fallback_message}: {str(e)}")


if __name__ == "__main__":
    main()