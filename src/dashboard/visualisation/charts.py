"""
Statistical chart visualizations for Australian Health Analytics Dashboard

This module contains functions for creating interactive statistical charts and plots,
primarily using Plotly for correlation analysis and scatter plots.

Author: Portfolio Demonstration
Date: June 2025
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, Optional, Dict, Any


def create_correlation_heatmap(correlation_matrix: pd.DataFrame) -> go.Figure:
    """
    Create interactive correlation heatmap using Plotly
    
    Parameters:
    -----------
    correlation_matrix : pd.DataFrame
        Square correlation matrix with variable names as indices and columns
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive correlation heatmap figure
        
    Features:
    ---------
    - Red-Blue diverging color scheme (RdBu_r)
    - Interactive hover tooltips showing exact correlation values
    - Centered title and responsive layout
    - Square aspect ratio for symmetric matrix display
    """
    
    fig = px.imshow(
        correlation_matrix,
        labels=dict(x="Variables", y="Variables", color="Correlation"),
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        color_continuous_scale='RdBu_r',
        aspect="auto",
        title="Correlation Matrix: SEIFA Disadvantage vs Health Outcomes"
    )
    
    fig.update_layout(
        width=800,
        height=600,
        title_x=0.5
    )
    
    return fig


def create_scatter_plots(data: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """
    Create scatter plots showing relationships between key variables
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing health and SEIFA data
    
    Returns:
    --------
    Tuple[go.Figure, go.Figure]
        Two plotly figures: (SEIFA vs Health Risk, SEIFA vs Mortality)
        
    Features:
    ---------
    - Color coding by state/territory
    - Ordinary least squares (OLS) trendlines
    - Interactive hover tooltips
    - Descriptive axis labels and titles
    """
    
    # SEIFA vs Health Risk Score
    fig1 = px.scatter(
        data.dropna(subset=['IRSD_Score', 'health_risk_score']),
        x='IRSD_Score',
        y='health_risk_score',
        color='STATE_NAME21',
        title='SEIFA Disadvantage Score vs Health Risk Score',
        labels={
            'IRSD_Score': 'SEIFA Disadvantage Score (Higher = Less Disadvantaged)',
            'health_risk_score': 'Composite Health Risk Score'
        },
        trendline='ols'
    )
    
    # SEIFA vs Mortality Rate
    fig2 = px.scatter(
        data.dropna(subset=['IRSD_Score', 'mortality_rate']),
        x='IRSD_Score',
        y='mortality_rate',
        color='STATE_NAME21',
        title='SEIFA Disadvantage Score vs Mortality Rate',
        labels={
            'IRSD_Score': 'SEIFA Disadvantage Score (Higher = Less Disadvantaged)',
            'mortality_rate': 'Mortality Rate (per 1,000)'
        },
        trendline='ols'
    )
    
    return fig1, fig2


def create_distribution_plot(data: pd.DataFrame, column: str, title: str = None) -> go.Figure:
    """
    Create distribution plot (histogram + box plot) for a variable
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the data
    column : str
        Column name to plot distribution for
    title : str, optional
        Custom title for the plot
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Combined histogram and box plot figure
    """
    
    if title is None:
        title = f"Distribution of {column.replace('_', ' ').title()}"
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Histogram", "Box Plot"),
        vertical_spacing=0.1
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=data[column].dropna(),
            nbinsx=30,
            name="Distribution",
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Box plot
    fig.add_trace(
        go.Box(
            x=data[column].dropna(),
            name="Summary Statistics",
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=title,
        height=500,
        title_x=0.5
    )
    
    return fig


def create_state_comparison_chart(data: pd.DataFrame, indicator: str) -> go.Figure:
    """
    Create bar chart comparing health indicators across states
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with state and health indicator data
    indicator : str
        Health indicator column to compare
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive bar chart comparing states
    """
    
    # Calculate state averages
    state_data = data.groupby('STATE_NAME21')[indicator].agg(['mean', 'std', 'count']).reset_index()
    state_data = state_data.dropna()
    
    fig = px.bar(
        state_data,
        x='STATE_NAME21',
        y='mean',
        error_y='std',
        title=f'{indicator.replace("_", " ").title()} by State/Territory',
        labels={
            'STATE_NAME21': 'State/Territory',
            'mean': f'Average {indicator.replace("_", " ").title()}',
            'std': 'Standard Deviation'
        }
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        title_x=0.5
    )
    
    return fig


def create_correlation_scatter_matrix(data: pd.DataFrame, variables: list) -> go.Figure:
    """
    Create scatter plot matrix for multiple variables
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the variables
    variables : list
        List of column names to include in the matrix
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Scatter plot matrix figure
    """
    
    # Filter data to only include specified variables
    plot_data = data[variables].dropna()
    
    fig = px.scatter_matrix(
        plot_data,
        dimensions=variables,
        title="Variable Relationships Matrix"
    )
    
    fig.update_layout(
        height=800,
        title_x=0.5
    )
    
    return fig


def create_time_series_plot(data: pd.DataFrame, date_col: str, value_col: str, 
                           group_col: str = None) -> go.Figure:
    """
    Create time series plot for temporal data
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with time series data
    date_col : str
        Column containing date/time values
    value_col : str
        Column containing values to plot
    group_col : str, optional
        Column for grouping (e.g., by state)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive time series plot
    """
    
    if group_col:
        fig = px.line(
            data,
            x=date_col,
            y=value_col,
            color=group_col,
            title=f'{value_col.replace("_", " ").title()} Over Time'
        )
    else:
        fig = px.line(
            data,
            x=date_col,
            y=value_col,
            title=f'{value_col.replace("_", " ").title()} Over Time'
        )
    
    fig.update_layout(
        title_x=0.5,
        height=500
    )
    
    return fig