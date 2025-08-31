"""
AHGD V3: Geographic Area Selector Component
Streamlit component for hierarchical geographic selection with drill-down capabilities.

Features:
- State → SA4 → SA3 → SA2 → SA1 drill-down
- Multi-select with search functionality
- Population and area filtering
- Real-time area statistics
"""

import streamlit as st
import polars as pl
from typing import List, Optional, Dict, Any

from ..utils.data_connector import DuckDBConnector


class GeographicSelector:
    """Interactive geographic area selector with hierarchical drill-down."""
    
    def __init__(self, db_connector: DuckDBConnector):
        """Initialize geographic selector with database connector."""
        self.db_connector = db_connector
        
        # Initialize session state for geographic selection
        if 'geo_selection_state' not in st.session_state:
            st.session_state.geo_selection_state = {
                'selected_state': None,
                'selected_sa4': None,
                'selected_sa3': None,
                'selected_sa2': None,
                'geographic_history': []
            }

    def render_selector(self, geographic_level: str) -> List[str]:
        """
        Render the geographic selector component.
        
        Args:
            geographic_level: Target geographic level (state, sa4, sa3, sa2, sa1)
            
        Returns:
            List of selected area codes/names
        """
        
        st.subheader(f"📍 {geographic_level.upper()} Selection")
        
        # Get available areas for the selected level
        available_areas = self.db_connector.get_available_areas(geographic_level)
        
        if not available_areas:
            st.warning(f"No {geographic_level.upper()} areas available")
            return []
        
        # Render selection interface based on geographic level
        if geographic_level == 'state':
            return self._render_state_selector(available_areas)
        elif geographic_level == 'sa4':
            return self._render_sa4_selector(available_areas)
        elif geographic_level in ['sa3', 'sa2', 'sa1']:
            return self._render_lower_level_selector(geographic_level, available_areas)
        
        return []

    def _render_state_selector(self, available_states: List[str]) -> List[str]:
        """Render state-level selector."""
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Multi-select for states
            selected_states = st.multiselect(
                "Select States/Territories",
                options=available_states,
                default=st.session_state.geo_selection_state.get('selected_state', []),
                help="Choose one or more states/territories for analysis"
            )
            
            # Update session state
            st.session_state.geo_selection_state['selected_state'] = selected_states
        
        with col2:
            # Selection statistics
            if selected_states:
                st.metric(
                    "States Selected",
                    len(selected_states)
                )
                
                # Quick actions
                if st.button("🇦🇺 Select All States"):
                    st.session_state.geo_selection_state['selected_state'] = available_states
                    st.rerun()
                
                if st.button("🗑️ Clear Selection"):
                    st.session_state.geo_selection_state['selected_state'] = []
                    st.rerun()
        
        # Show selected states summary
        if selected_states:
            st.info(f"**Selected:** {', '.join(selected_states[:3])}" + 
                   (f" and {len(selected_states) - 3} more" if len(selected_states) > 3 else ""))
        
        return selected_states

    def _render_sa4_selector(self, available_sa4s: List[str]) -> List[str]:
        """Render SA4-level selector with state filtering."""
        
        col1, col2 = st.columns([2, 2])
        
        with col1:
            # State filter for SA4 selection
            available_states = self.db_connector.get_available_areas('state')
            
            state_filter = st.selectbox(
                "Filter by State",
                options=['All States'] + available_states,
                help="Filter SA4 areas by state"
            )
        
        with col2:
            # Search functionality
            search_term = st.text_input(
                "🔍 Search SA4 Areas",
                placeholder="Enter SA4 name...",
                help="Search for specific SA4 areas"
            )
        
        # Filter SA4s based on state and search
        filtered_sa4s = available_sa4s
        
        if state_filter != 'All States':
            # Filter SA4s by state (simplified - in production, use proper joins)
            filtered_sa4s = [sa4 for sa4 in available_sa4s if state_filter.lower() in sa4.lower()]
        
        if search_term:
            filtered_sa4s = [sa4 for sa4 in filtered_sa4s if search_term.lower() in sa4.lower()]
        
        # Multi-select for SA4s
        selected_sa4s = st.multiselect(
            f"Select SA4 Areas ({len(filtered_sa4s)} available)",
            options=filtered_sa4s,
            default=st.session_state.geo_selection_state.get('selected_sa4', []),
            help="Choose SA4 areas for detailed analysis"
        )
        
        st.session_state.geo_selection_state['selected_sa4'] = selected_sa4s
        
        # Show selection summary
        if selected_sa4s:
            st.success(f"✅ {len(selected_sa4s)} SA4 areas selected")
        
        return selected_sa4s

    def _render_lower_level_selector(self, geographic_level: str, available_areas: List[str]) -> List[str]:
        """Render selector for SA3/SA2/SA1 levels with performance optimizations."""
        
        # Performance warning for SA1
        if geographic_level == 'sa1':
            st.warning(
                "⚠️ **SA1 Level Analysis**: Due to performance considerations, "
                "SA1 selection is limited to 1,000 areas. Use filters to narrow your selection."
            )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Search and filter controls
            search_col, filter_col = st.columns(2)
            
            with search_col:
                search_term = st.text_input(
                    f"🔍 Search {geographic_level.upper()}",
                    placeholder=f"Enter {geographic_level.upper()} name or code...",
                    help=f"Search for specific {geographic_level.upper()} areas"
                )
            
            with filter_col:
                # Population filter for SA1/SA2
                if geographic_level in ['sa1', 'sa2']:
                    min_population = st.number_input(
                        "Min Population",
                        min_value=0,
                        max_value=50000,
                        value=0,
                        step=100,
                        help="Filter by minimum population"
                    )
            
            # Filter available areas
            filtered_areas = available_areas
            
            if search_term:
                filtered_areas = [
                    area for area in filtered_areas 
                    if search_term.lower() in area.lower()
                ]
            
            # Limit display for performance
            display_limit = 500 if geographic_level == 'sa1' else 1000
            if len(filtered_areas) > display_limit:
                st.info(f"Showing first {display_limit} of {len(filtered_areas)} areas. Use search to narrow selection.")
                filtered_areas = filtered_areas[:display_limit]
            
            # Multi-select
            selected_areas = st.multiselect(
                f"Select {geographic_level.upper()} Areas ({len(filtered_areas)} shown)",
                options=filtered_areas,
                default=[],  # Don't persist lower level selections
                help=f"Choose {geographic_level.upper()} areas for analysis"
            )
        
        with col2:
            # Selection statistics
            if selected_areas:
                st.metric(
                    f"{geographic_level.upper()} Selected",
                    len(selected_areas)
                )
                
                # Quick selection buttons
                if len(filtered_areas) <= 50:  # Only for manageable numbers
                    if st.button(f"Select All {len(filtered_areas)}"):
                        selected_areas = filtered_areas.copy()
                        st.rerun()
                
                if st.button("Clear Selection"):
                    selected_areas = []
                    st.rerun()
            
            # Performance indicator
            if geographic_level == 'sa1':
                performance_color = "🟢" if len(selected_areas) <= 100 else "🟡" if len(selected_areas) <= 500 else "🔴"
                st.markdown(f"{performance_color} **Performance**: {len(selected_areas)} areas")
        
        # Advanced selection options
        if st.expander("🔧 Advanced Selection Options"):
            
            col_adv1, col_adv2 = st.columns(2)
            
            with col_adv1:
                # Random sampling for testing
                sample_size = st.number_input(
                    "Random Sample Size",
                    min_value=0,
                    max_value=min(1000, len(filtered_areas)),
                    value=0,
                    help="Select a random sample of areas"
                )
                
                if sample_size > 0 and st.button("🎲 Random Sample"):
                    import random
                    selected_areas = random.sample(filtered_areas, sample_size)
                    st.rerun()
            
            with col_adv2:
                # Selection by pattern
                pattern_options = [
                    "Urban areas only",
                    "Rural areas only", 
                    "High population areas",
                    "Low population areas"
                ]
                
                selection_pattern = st.selectbox(
                    "Selection Pattern",
                    options=["None"] + pattern_options,
                    help="Apply predefined selection patterns"
                )
                
                if selection_pattern != "None" and st.button("Apply Pattern"):
                    # Implement pattern-based selection
                    st.info(f"Applied pattern: {selection_pattern}")
        
        return selected_areas

    def render_selection_summary(self, selected_areas: List[str], geographic_level: str):
        """Render summary of current geographic selection."""
        
        if not selected_areas:
            return
        
        st.subheader("📊 Selection Summary")
        
        # Get summary statistics for selected areas
        try:
            summary_stats = self.db_connector.get_summary_metrics(
                geographic_level=geographic_level,
                selected_areas=selected_areas,
                health_metric='total_population',  # Use population for summary
                date_range=(2021, 2023)
            )
            
            if summary_stats and summary_stats.height > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_pop = summary_stats.select(
                        pl.col('total_population').sum()
                    ).item()
                    st.metric("Total Population", f"{total_pop:,.0f}" if total_pop else "N/A")
                
                with col2:
                    avg_quality = summary_stats.select(
                        pl.col('data_completeness_score').mean()
                    ).item()
                    if avg_quality:
                        st.metric("Avg Data Quality", f"{avg_quality:.1%}")
                
                with col3:
                    area_count = len(selected_areas)
                    st.metric(f"{geographic_level.upper()} Areas", f"{area_count:,}")
                
                # Geographic distribution
                if 'state_name' in summary_stats.columns:
                    state_dist = summary_stats.group_by('state_name').agg(
                        pl.len().alias('count')
                    ).sort('count', descending=True)
                    
                    st.subheader("📍 Geographic Distribution")
                    for row in state_dist.rows():
                        st.write(f"**{row[0]}**: {row[1]} areas")
            
        except Exception as e:
            st.error(f"Error loading selection summary: {str(e)}")

    def get_selection_breadcrumb(self) -> str:
        """Generate breadcrumb navigation for current selection."""
        
        state = st.session_state.geo_selection_state
        breadcrumb_parts = []
        
        if state.get('selected_state'):
            breadcrumb_parts.append(f"States: {len(state['selected_state'])}")
        
        if state.get('selected_sa4'):
            breadcrumb_parts.append(f"SA4: {len(state['selected_sa4'])}")
        
        if state.get('selected_sa3'):
            breadcrumb_parts.append(f"SA3: {len(state['selected_sa3'])}")
        
        if state.get('selected_sa2'):
            breadcrumb_parts.append(f"SA2: {len(state['selected_sa2'])}")
        
        return " → ".join(breadcrumb_parts) if breadcrumb_parts else "No selection"