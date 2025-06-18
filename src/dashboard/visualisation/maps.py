"""
Geographic visualization components for Australian Health Analytics Dashboard

This module contains functions for creating interactive geographic visualizations,
primarily using Folium for choropleth maps and spatial data representation.

Author: Portfolio Demonstration
Date: June 2025
"""

import pandas as pd
import numpy as np
import folium
import streamlit as st
from typing import Optional, Dict, Any

from src.config import get_global_config

# Get configuration
config = get_global_config()


def create_health_risk_map(data: pd.DataFrame, indicator: str = 'health_risk_score') -> Optional[folium.Map]:
    """
    Create interactive choropleth map of health indicators
    
    Parameters:
    -----------
    data : pd.DataFrame
        Geospatial dataframe containing SA2 data with health indicators
    indicator : str
        Column name of the health indicator to map (default: 'health_risk_score')
    
    Returns:
    --------
    folium.Map or None
        Interactive folium map object, or None if data is insufficient
    
    Features:
    ---------
    - Choropleth visualization with YlOrRd color scheme
    - Interactive tooltips with detailed area information
    - Marker popups for additional context
    - Automatic map centering and zoom
    - Legend with indicator values
    """
    
    # Filter out rows without geographic data
    map_data = data.dropna(subset=['geometry', indicator])
    
    if map_data.empty:
        st.warning("No geographic data available for mapping")
        return None
    
    # Calculate map center from data bounds
    bounds = map_data.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Create base map using configuration
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=config.dashboard.default_map_zoom,
        tiles='OpenStreetMap'
    )
    
    # Add choropleth layer
    folium.Choropleth(
        geo_data=map_data.__geo_interface__,
        data=map_data,
        columns=['SA2_CODE21', indicator],
        key_on='feature.properties.SA2_CODE21',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f'{indicator.replace("_", " ").title()}',
        bins=9
    ).add_to(m)
    
    # Add tooltip with detailed information
    for idx, row in map_data.iterrows():
        if pd.notna(row[indicator]):
            tooltip_text = _create_tooltip_text(row, indicator)
            
            folium.Marker(
                location=[row.geometry.centroid.y, row.geometry.centroid.x],
                popup=folium.Popup(tooltip_text, max_width=300),
                icon=folium.Icon(color='red', icon='info-sign', prefix='glyphicon')
            ).add_to(m)
    
    return m


def _create_tooltip_text(row: pd.Series, indicator: str) -> str:
    """
    Create formatted tooltip text for map markers
    
    Parameters:
    -----------
    row : pd.Series
        Data row containing area information
    indicator : str
        Health indicator being displayed
    
    Returns:
    --------
    str
        HTML formatted tooltip text
    """
    tooltip_text = f"""
    <b>{row['SA2_NAME21']}</b><br>
    State: {row['STATE_NAME21']}<br>
    {indicator.replace('_', ' ').title()}: {row[indicator]:.2f}<br>
    SEIFA Disadvantage Score: {row['IRSD_Score']:.0f}<br>
    SEIFA Decile: {row['IRSD_Decile_Australia']:.0f}
    """
    
    return tooltip_text


def get_map_bounds(data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate geographic bounds from geospatial data
    
    Parameters:
    -----------
    data : pd.DataFrame
        Geospatial dataframe with geometry column
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing bounds coordinates
    """
    if 'geometry' not in data.columns:
        raise ValueError("Data must contain 'geometry' column")
    
    bounds = data.total_bounds
    
    return {
        'min_lon': bounds[0],
        'min_lat': bounds[1], 
        'max_lon': bounds[2],
        'max_lat': bounds[3],
        'center_lat': (bounds[1] + bounds[3]) / 2,
        'center_lon': (bounds[0] + bounds[2]) / 2
    }


def create_simple_point_map(data: pd.DataFrame, lat_col: str, lon_col: str, 
                           popup_col: str = None) -> folium.Map:
    """
    Create simple point map for locations without complex geometry
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with latitude and longitude columns
    lat_col : str
        Column name containing latitude values
    lon_col : str
        Column name containing longitude values
    popup_col : str, optional
        Column to use for popup text
        
    Returns:
    --------
    folium.Map
        Simple point map with markers
    """
    # Calculate center
    center_lat = data[lat_col].mean()
    center_lon = data[lon_col].mean()
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=config.dashboard.default_map_zoom
    )
    
    # Add points
    for idx, row in data.iterrows():
        popup_text = str(row[popup_col]) if popup_col else f"Point {idx}"
        
        folium.Marker(
            location=[row[lat_col], row[lon_col]],
            popup=popup_text
        ).add_to(m)
    
    return m