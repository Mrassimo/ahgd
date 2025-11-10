"""
Reusable chart components for AHGD Dashboard.

This module provides standardised chart templates using Plotly
for consistent visualisations across the dashboard.
"""

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from typing import Optional, List, Dict, Any


def create_bar_chart(
    data: pl.DataFrame,
    x: str,
    y: str,
    title: str,
    orientation: str = "v",
    color: Optional[str] = None,
    color_scale: Optional[List[str]] = None,
    labels: Optional[Dict[str, str]] = None,
    height: int = 400,
    **kwargs,
) -> go.Figure:
    """
    Create a standardised bar chart.

    Args:
        data: Polars DataFrame with chart data
        x: Column name for x-axis
        y: Column name for y-axis
        title: Chart title
        orientation: 'v' for vertical or 'h' for horizontal
        color: Column name for color encoding
        color_scale: Color scale to use
        labels: Custom axis labels
        height: Chart height in pixels
        **kwargs: Additional arguments passed to px.bar

    Returns:
        Plotly Figure object
    """
    fig = px.bar(
        data.to_pandas(),
        x=x,
        y=y,
        title=title,
        orientation=orientation,
        color=color,
        color_continuous_scale=color_scale if color_scale else None,
        color_discrete_sequence=["#1f77b4"] if not color else None,
        labels=labels,
        **kwargs,
    )

    fig.update_layout(
        height=height,
        hovermode="closest",
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=12),
        title_font_size=16,
    )

    return fig


def create_line_chart(
    data: pl.DataFrame,
    x: str,
    y: str,
    title: str,
    color: Optional[str] = None,
    line_group: Optional[str] = None,
    markers: bool = True,
    labels: Optional[Dict[str, str]] = None,
    height: int = 400,
    **kwargs,
) -> go.Figure:
    """
    Create a standardised line chart.

    Args:
        data: Polars DataFrame with chart data
        x: Column name for x-axis
        y: Column name for y-axis
        title: Chart title
        color: Column name for color encoding
        line_group: Column name for grouping lines
        markers: Whether to show markers on line
        labels: Custom axis labels
        height: Chart height in pixels
        **kwargs: Additional arguments passed to px.line

    Returns:
        Plotly Figure object
    """
    fig = px.line(
        data.to_pandas(),
        x=x,
        y=y,
        title=title,
        color=color,
        line_group=line_group,
        markers=markers,
        labels=labels,
        **kwargs,
    )

    fig.update_layout(
        height=height,
        hovermode="x unified",
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=12),
        title_font_size=16,
    )

    return fig


def create_scatter_chart(
    data: pl.DataFrame,
    x: str,
    y: str,
    title: str,
    size: Optional[str] = None,
    color: Optional[str] = None,
    color_scale: Optional[List[str]] = None,
    hover_data: Optional[List[str]] = None,
    labels: Optional[Dict[str, str]] = None,
    height: int = 500,
    **kwargs,
) -> go.Figure:
    """
    Create a standardised scatter chart.

    Args:
        data: Polars DataFrame with chart data
        x: Column name for x-axis
        y: Column name for y-axis
        title: Chart title
        size: Column name for marker size
        color: Column name for color encoding
        color_scale: Color scale to use
        hover_data: Additional columns to show on hover
        labels: Custom axis labels
        height: Chart height in pixels
        **kwargs: Additional arguments passed to px.scatter

    Returns:
        Plotly Figure object
    """
    fig = px.scatter(
        data.to_pandas(),
        x=x,
        y=y,
        title=title,
        size=size,
        color=color,
        color_continuous_scale=color_scale,
        hover_data=hover_data,
        labels=labels,
        **kwargs,
    )

    fig.update_layout(
        height=height,
        hovermode="closest",
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=12),
        title_font_size=16,
    )

    return fig


def create_heatmap(
    data: pl.DataFrame,
    x: str,
    y: str,
    z: str,
    title: str,
    color_scale: Optional[List[str]] = None,
    labels: Optional[Dict[str, str]] = None,
    height: int = 500,
    **kwargs,
) -> go.Figure:
    """
    Create a standardised heatmap.

    Args:
        data: Polars DataFrame with chart data
        x: Column name for x-axis
        y: Column name for y-axis
        z: Column name for color values
        title: Chart title
        color_scale: Color scale to use
        labels: Custom axis labels
        height: Chart height in pixels
        **kwargs: Additional arguments passed to px.density_heatmap

    Returns:
        Plotly Figure object
    """
    fig = px.density_heatmap(
        data.to_pandas(),
        x=x,
        y=y,
        z=z,
        title=title,
        color_continuous_scale=color_scale if color_scale else "RdYlGn",
        labels=labels,
        **kwargs,
    )

    fig.update_layout(
        height=height,
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=12),
        title_font_size=16,
    )

    return fig


def create_histogram(
    data: pl.DataFrame,
    x: str,
    title: str,
    nbins: int = 50,
    color: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
    height: int = 400,
    **kwargs,
) -> go.Figure:
    """
    Create a standardised histogram.

    Args:
        data: Polars DataFrame with chart data
        x: Column name for histogram
        title: Chart title
        nbins: Number of bins
        color: Column name for color encoding
        labels: Custom axis labels
        height: Chart height in pixels
        **kwargs: Additional arguments passed to px.histogram

    Returns:
        Plotly Figure object
    """
    fig = px.histogram(
        data.to_pandas(),
        x=x,
        title=title,
        nbins=nbins,
        color=color,
        color_discrete_sequence=["#1f77b4"] if not color else None,
        labels=labels,
        **kwargs,
    )

    fig.update_layout(
        height=height,
        showlegend=False if not color else True,
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=12),
        title_font_size=16,
    )

    return fig


def create_box_plot(
    data: pl.DataFrame,
    y: str,
    title: str,
    x: Optional[str] = None,
    color: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
    height: int = 400,
    **kwargs,
) -> go.Figure:
    """
    Create a standardised box plot.

    Args:
        data: Polars DataFrame with chart data
        y: Column name for y-axis values
        title: Chart title
        x: Optional column name for x-axis (grouping)
        color: Column name for color encoding
        labels: Custom axis labels
        height: Chart height in pixels
        **kwargs: Additional arguments passed to px.box

    Returns:
        Plotly Figure object
    """
    fig = px.box(
        data.to_pandas(),
        y=y,
        x=x,
        title=title,
        color=color,
        color_discrete_sequence=["#1f77b4"] if not color else None,
        labels=labels,
        **kwargs,
    )

    fig.update_layout(
        height=height,
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=12),
        title_font_size=16,
    )

    return fig


def create_pie_chart(
    data: pl.DataFrame,
    values: str,
    names: str,
    title: str,
    color_scale: Optional[List[str]] = None,
    hole: float = 0,
    height: int = 400,
    **kwargs,
) -> go.Figure:
    """
    Create a standardised pie chart.

    Args:
        data: Polars DataFrame with chart data
        values: Column name for slice values
        names: Column name for slice labels
        title: Chart title
        color_scale: Color scale to use
        hole: Size of hole in center (0 for pie, >0 for donut)
        height: Chart height in pixels
        **kwargs: Additional arguments passed to px.pie

    Returns:
        Plotly Figure object
    """
    fig = px.pie(
        data.to_pandas(),
        values=values,
        names=names,
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=hole,
        **kwargs,
    )

    fig.update_layout(
        height=height,
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=12),
        title_font_size=16,
    )

    return fig


def create_correlation_heatmap(
    corr_matrix: pl.DataFrame,
    title: str = "Correlation Matrix",
    height: int = 600,
) -> go.Figure:
    """
    Create a correlation matrix heatmap.

    Args:
        corr_matrix: Polars DataFrame with correlation values
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly Figure object
    """
    # Convert to pandas for plotly
    corr_df = corr_matrix.to_pandas()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.columns,
            colorscale="RdBu_r",
            zmid=0,
            text=corr_df.values,
            texttemplate="%{text:.2f}",
            textfont={"size": 10},
            colorbar=dict(title="Correlation"),
        )
    )

    fig.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=12),
        title_font_size=16,
        xaxis=dict(tickangle=-45),
    )

    return fig


def create_gauge_chart(
    value: float,
    title: str,
    min_val: float = 0,
    max_val: float = 100,
    threshold_good: float = 75,
    threshold_bad: float = 50,
    height: int = 300,
) -> go.Figure:
    """
    Create a gauge chart for single metric display.

    Args:
        value: Current value to display
        title: Chart title
        min_val: Minimum value on gauge
        max_val: Maximum value on gauge
        threshold_good: Threshold for good performance (green)
        threshold_bad: Threshold for poor performance (red)
        height: Chart height in pixels

    Returns:
        Plotly Figure object
    """
    # Determine color based on thresholds
    if value >= threshold_good:
        color = "#4caf50"  # Green
    elif value >= threshold_bad:
        color = "#ff9800"  # Orange
    else:
        color = "#f44336"  # Red

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=value,
            title={"text": title},
            gauge={
                "axis": {"range": [min_val, max_val]},
                "bar": {"color": color},
                "steps": [
                    {"range": [min_val, threshold_bad], "color": "#ffebee"},
                    {"range": [threshold_bad, threshold_good], "color": "#fff3e0"},
                    {"range": [threshold_good, max_val], "color": "#e8f5e9"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": threshold_good,
                },
            },
        )
    )

    fig.update_layout(height=height, template="plotly_white")

    return fig
