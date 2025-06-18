"""
Sidebar Controls and Filters for Australian Health Analytics Dashboard

This module handles all sidebar UI components including:
- Analysis type selection
- State/territory filters
- Session state management
- Filter state persistence
"""

import streamlit as st
from typing import List, Tuple, Any, Dict


class SidebarController:
    """Manages sidebar controls and filter state"""
    
    def __init__(self):
        """Initialise sidebar controller"""
        self.analysis_types = [
            "Geographic Health Explorer",
            "Correlation Analysis", 
            "Health Hotspot Identification",
            "Predictive Risk Analysis",
            "Data Quality & Methodology"
        ]
    
    def render_sidebar_controls(self, data) -> Tuple[str, List[str]]:
        """
        Render all sidebar controls and return selected values
        
        Args:
            data: Main dataset containing health and geographic data
            
        Returns:
            Tuple of (analysis_type, selected_states)
        """
        st.sidebar.header("🔧 Dashboard Controls")
        
        # Analysis type selection
        analysis_type = st.sidebar.selectbox(
            "Select Analysis Type",
            self.analysis_types
        )
        
        # State filter
        available_states = sorted(data['STATE_NAME21'].dropna().unique())
        selected_states = st.sidebar.multiselect(
            "Filter by State/Territory",
            available_states,
            default=available_states
        )
        
        return analysis_type, selected_states
    
    def apply_state_filter(self, data, selected_states: List[str]):
        """
        Apply state filter to dataset
        
        Args:
            data: Original dataset
            selected_states: List of selected state names
            
        Returns:
            Filtered dataset
        """
        if selected_states:
            return data[data['STATE_NAME21'].isin(selected_states)]
        else:
            return data
    
    def get_sidebar_state(self) -> Dict[str, Any]:
        """
        Get current sidebar state for debugging/logging
        
        Returns:
            Dictionary containing current sidebar state
        """
        return {
            'session_state_keys': list(st.session_state.keys()),
            'sidebar_state': {
                key: value for key, value in st.session_state.items() 
                if key.startswith('sidebar_') or key in self.analysis_types
            }
        }
    
    def reset_sidebar_state(self):
        """Reset sidebar state to defaults"""
        # Clear relevant session state keys
        keys_to_clear = [key for key in st.session_state.keys() 
                        if key.startswith('sidebar_') or 'state' in key.lower()]
        
        for key in keys_to_clear:
            del st.session_state[key]


def render_analysis_selector() -> str:
    """
    Simplified analysis type selector for external use
    
    Returns:
        Selected analysis type
    """
    return st.sidebar.selectbox(
        "Select Analysis Type",
        [
            "Geographic Health Explorer",
            "Correlation Analysis", 
            "Health Hotspot Identification",
            "Predictive Risk Analysis",
            "Data Quality & Methodology"
        ]
    )


def render_state_filter(data) -> List[str]:
    """
    Simplified state filter for external use
    
    Args:
        data: Dataset containing state information
        
    Returns:
        List of selected states
    """
    available_states = sorted(data['STATE_NAME21'].dropna().unique())
    return st.sidebar.multiselect(
        "Filter by State/Territory",
        available_states,
        default=available_states
    )


def create_sidebar_header():
    """Create standardised sidebar header"""
    st.sidebar.header("🔧 Dashboard Controls")


def add_sidebar_info():
    """Add information section to sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("**About This Dashboard**")
    st.sidebar.markdown("""
    Interactive analysis of health outcomes and socio-economic disadvantage across 
    Australian Statistical Areas (SA2 level).
    
    Data sources: ABS, AIHW, PHIDU
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("*Portfolio Demonstration Project*")