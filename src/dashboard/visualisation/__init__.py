"""
Australian Health Analytics Dashboard - Visualization Module

This module provides organized visualization components for creating interactive
health analytics dashboards with geographic mapping, statistical charts, and
reusable UI components.

Modules:
--------
- maps: Geographic visualizations using Folium
- charts: Statistical charts and plots using Plotly
- components: Reusable UI components and utilities

Author: Portfolio Demonstration
Date: June 2025
"""

from .maps import (
    create_health_risk_map,
    get_map_bounds,
    create_simple_point_map
)

from .charts import (
    create_correlation_heatmap,
    create_scatter_plots,
    create_distribution_plot,
    create_state_comparison_chart,
    create_correlation_scatter_matrix,
    create_time_series_plot
)

from .components import (
    display_key_metrics,
    create_health_indicator_selector,
    format_health_indicator_name,
    display_correlation_insights,
    display_hotspot_card,
    create_data_quality_metrics,
    create_performance_metrics,
    apply_custom_styling,
    format_number,
    create_data_filter_sidebar
)

__all__ = [
    # Maps
    'create_health_risk_map',
    'get_map_bounds',
    'create_simple_point_map',
    
    # Charts
    'create_correlation_heatmap',
    'create_scatter_plots',
    'create_distribution_plot',
    'create_state_comparison_chart',
    'create_correlation_scatter_matrix',
    'create_time_series_plot',
    
    # Components
    'display_key_metrics',
    'create_health_indicator_selector',
    'format_health_indicator_name',
    'display_correlation_insights',
    'display_hotspot_card',
    'create_data_quality_metrics',
    'create_performance_metrics',
    'apply_custom_styling',
    'format_number',
    'create_data_filter_sidebar'
]