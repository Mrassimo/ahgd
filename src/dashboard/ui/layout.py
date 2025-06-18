"""
Layout Management and UI Utilities for Australian Health Analytics Dashboard

This module provides:
- Responsive design utilities
- Container and column management
- Consistent styling and spacing functions
- Layout helpers for different screen sizes
"""

import streamlit as st
from typing import List, Tuple, Dict, Any


class LayoutManager:
    """Manages dashboard layout and responsive design"""
    
    def __init__(self):
        """Initialise layout manager"""
        self.default_column_gap = "medium"
    
    def create_header_section(self, title: str, description: str):
        """
        Create standardised header section
        
        Args:
            title: Page title
            description: Page description
        """
        st.title(title)
        st.markdown(description)
    
    def create_metrics_row(self, metrics: List[Dict[str, Any]], columns: int = 3):
        """
        Create a row of metrics with responsive columns
        
        Args:
            metrics: List of metric dictionaries with label, value, delta
            columns: Number of columns to create
        """
        cols = st.columns(columns)
        
        for i, metric in enumerate(metrics):
            with cols[i % columns]:
                st.metric(
                    metric.get('label', ''),
                    metric.get('value', ''),
                    delta=metric.get('delta')
                )
    
    def create_two_column_layout(self, left_content=None, right_content=None, ratio: Tuple[int, int] = (1, 1)):
        """
        Create a two-column layout with specified ratio
        
        Args:
            left_content: Function to render left content
            right_content: Function to render right content  
            ratio: Column width ratio (left, right)
            
        Returns:
            Tuple of column objects
        """
        col1, col2 = st.columns(ratio)
        
        if left_content:
            with col1:
                left_content()
        
        if right_content:
            with col2:
                right_content()
        
        return col1, col2
    
    def create_tabbed_layout(self, tabs: List[str], tab_contents: List[callable]):
        """
        Create tabbed interface
        
        Args:
            tabs: List of tab names
            tab_contents: List of functions to render tab content
        """
        if len(tabs) != len(tab_contents):
            raise ValueError("Number of tabs and tab contents must match")
        
        tab_objects = st.tabs(tabs)
        
        for tab, content_func in zip(tab_objects, tab_contents):
            with tab:
                content_func()
    
    def add_divider(self, style: str = "default"):
        """
        Add styled divider
        
        Args:
            style: Divider style ('default', 'thick', 'dotted')
        """
        if style == "thick":
            st.markdown("---")
            st.markdown("")
        elif style == "dotted":
            st.markdown('<hr style="border: 1px dotted #ddd;">', unsafe_allow_html=True)
        else:
            st.markdown("---")
    
    def create_info_box(self, title: str, content: str, box_type: str = "info"):
        """
        Create styled information box
        
        Args:
            title: Box title
            content: Box content
            box_type: Box type ('info', 'warning', 'success', 'error')
        """
        if box_type == "warning":
            st.warning(f"**{title}**\n\n{content}")
        elif box_type == "success":
            st.success(f"**{title}**\n\n{content}")
        elif box_type == "error":
            st.error(f"**{title}**\n\n{content}")
        else:
            st.info(f"**{title}**\n\n{content}")
    
    def create_expandable_section(self, title: str, content_func: callable, expanded: bool = False):
        """
        Create expandable section
        
        Args:
            title: Section title
            content_func: Function to render content when expanded
            expanded: Whether section starts expanded
        """
        with st.expander(title, expanded=expanded):
            content_func()


def create_dashboard_header():
    """Create main dashboard header with branding"""
    st.title("🏥 Australian Health Analytics Dashboard")
    st.markdown("""
    **Interactive analysis of health outcomes and socio-economic disadvantage across Australian Statistical Areas**
    
    This dashboard demonstrates correlation analysis between SEIFA (Socio-Economic Indexes for Areas) 
    disadvantage indicators and health outcomes, providing insights for health policy and resource allocation.
    """)


def create_loading_spinner(message: str = "Loading data..."):
    """
    Create loading spinner with custom message
    
    Args:
        message: Loading message to display
        
    Returns:
        Streamlit spinner context manager
    """
    return st.spinner(message)


def create_dashboard_footer():
    """Create standardised dashboard footer"""
    st.markdown("---")
    st.markdown("""
    **Australian Health Analytics Dashboard** | *Portfolio Demonstration Project*
    
    Built with Streamlit, demonstrating modern data science capabilities for health policy analysis.
    Data sources: Australian Bureau of Statistics, Australian Institute of Health and Welfare.
    """)


def format_large_number(number: int) -> str:
    """
    Format large numbers with commas
    
    Args:
        number: Number to format
        
    Returns:
        Formatted number string
    """
    return f"{number:,}"


def create_responsive_columns(num_columns: int, gap: str = "medium") -> List:
    """
    Create responsive columns that adapt to screen size
    
    Args:
        num_columns: Number of columns to create
        gap: Gap size between columns
        
    Returns:
        List of column objects
    """
    return st.columns(num_columns, gap=gap)


def apply_container_styling():
    """Apply custom container styling"""
    st.markdown("""
    <style>
    .main-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .info-section {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #0066cc;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


def create_chart_container(chart_object, title: str = None, full_width: bool = True):
    """
    Create standardised chart container
    
    Args:
        chart_object: Plotly/Altair chart object
        title: Optional chart title
        full_width: Whether to use full container width
    """
    if title:
        st.subheader(title)
    
    st.plotly_chart(chart_object, use_container_width=full_width)


def display_data_table(data, title: str = None, sortable: bool = True, 
                      max_height: int = 400):
    """
    Display data table with consistent formatting
    
    Args:
        data: DataFrame to display
        title: Optional table title
        sortable: Whether table should be sortable
        max_height: Maximum table height in pixels
    """
    if title:
        st.subheader(title)
    
    st.dataframe(
        data,
        use_container_width=True,
        height=max_height
    )