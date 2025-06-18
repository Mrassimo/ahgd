"""
UI Components Package for Australian Health Analytics Dashboard

This package provides modular UI components for the dashboard including:
- Sidebar controls and filters
- Page routing and navigation
- Layout management utilities
- Responsive design helpers
"""

from .sidebar import SidebarController, render_analysis_selector, render_state_filter
from .pages import render_page, get_page_renderer
from .layout import (
    LayoutManager,
    create_dashboard_header,
    create_dashboard_footer,
    create_loading_spinner,
    create_responsive_columns,
    apply_container_styling
)

__all__ = [
    'SidebarController',
    'render_analysis_selector', 
    'render_state_filter',
    'render_page',
    'get_page_renderer',
    'LayoutManager',
    'create_dashboard_header',
    'create_dashboard_footer',
    'create_loading_spinner',
    'create_responsive_columns',
    'apply_container_styling'
]