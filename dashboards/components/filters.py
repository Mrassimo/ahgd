"""
Reusable filter widgets for AHGD Dashboard.

This module provides standardised filter components for consistent
filtering across the dashboard.
"""

import streamlit as st
from typing import List, Optional, Tuple, Any
from datetime import datetime, date


def create_state_filter(
    states: List[str],
    default: Optional[List[str]] = None,
    key: str = "state_filter",
    help_text: str = "Filter by state codes",
) -> List[str]:
    """
    Create a multi-select state filter.

    Args:
        states: List of available state codes
        default: Default selected states (all if None)
        key: Unique widget key
        help_text: Help text for the filter

    Returns:
        List of selected state codes
    """
    if default is None:
        default = states

    selected = st.multiselect(
        "States",
        options=states,
        default=default,
        key=key,
        help=help_text,
    )

    return selected


def create_remoteness_filter(
    remoteness_categories: List[str],
    default: Optional[List[str]] = None,
    key: str = "remoteness_filter",
    help_text: str = "Filter by remoteness category",
) -> List[str]:
    """
    Create a multi-select remoteness category filter.

    Args:
        remoteness_categories: List of available remoteness categories
        default: Default selected categories (all if None)
        key: Unique widget key
        help_text: Help text for the filter

    Returns:
        List of selected remoteness categories
    """
    if default is None:
        default = remoteness_categories

    selected = st.multiselect(
        "Remoteness",
        options=remoteness_categories,
        default=default,
        key=key,
        help=help_text,
    )

    return selected


def create_date_range_filter(
    min_date: Optional[date] = None,
    max_date: Optional[date] = None,
    default_range: Optional[Tuple[date, date]] = None,
    key: str = "date_range_filter",
    help_text: str = "Filter by date range",
) -> Tuple[date, date]:
    """
    Create a date range filter.

    Args:
        min_date: Minimum selectable date
        max_date: Maximum selectable date
        default_range: Default date range (full range if None)
        key: Unique widget key
        help_text: Help text for the filter

    Returns:
        Tuple of (start_date, end_date)
    """
    if min_date is None:
        min_date = date(2020, 1, 1)

    if max_date is None:
        max_date = date.today()

    if default_range is None:
        default_range = (min_date, max_date)

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=default_range[0],
            min_value=min_date,
            max_value=max_date,
            key=f"{key}_start",
            help=help_text,
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=default_range[1],
            min_value=min_date,
            max_value=max_date,
            key=f"{key}_end",
            help=help_text,
        )

    return start_date, end_date


def create_numeric_range_filter(
    label: str,
    min_value: float,
    max_value: float,
    default_range: Optional[Tuple[float, float]] = None,
    step: float = 1.0,
    key: str = "numeric_range_filter",
    help_text: str = "Filter by numeric range",
) -> Tuple[float, float]:
    """
    Create a numeric range slider filter.

    Args:
        label: Label for the filter
        min_value: Minimum value
        max_value: Maximum value
        default_range: Default range (full range if None)
        step: Step size for slider
        key: Unique widget key
        help_text: Help text for the filter

    Returns:
        Tuple of (min_selected, max_selected)
    """
    if default_range is None:
        default_range = (min_value, max_value)

    selected_range = st.slider(
        label,
        min_value=min_value,
        max_value=max_value,
        value=default_range,
        step=step,
        key=key,
        help=help_text,
    )

    return selected_range


def create_metric_selector(
    metrics: dict,
    default: Optional[str] = None,
    key: str = "metric_selector",
    help_text: str = "Select metric to visualise",
) -> str:
    """
    Create a metric selector dropdown.

    Args:
        metrics: Dictionary of {display_name: column_name}
        default: Default selected metric
        key: Unique widget key
        help_text: Help text for the selector

    Returns:
        Selected metric column name
    """
    if default is None:
        default = list(metrics.keys())[0]

    selected_label = st.selectbox(
        "Select Metric",
        options=list(metrics.keys()),
        index=list(metrics.keys()).index(default) if default in metrics else 0,
        key=key,
        help=help_text,
    )

    return metrics[selected_label]


def create_comparison_selector(
    options: List[str],
    default: Optional[str] = None,
    key: str = "comparison_selector",
    help_text: str = "Select comparison variable",
) -> str:
    """
    Create a comparison variable selector.

    Args:
        options: List of available options
        default: Default selected option
        key: Unique widget key
        help_text: Help text for the selector

    Returns:
        Selected option
    """
    if default is None:
        default = options[0] if options else None

    selected = st.selectbox(
        "Compare By",
        options=options,
        index=options.index(default) if default in options else 0,
        key=key,
        help=help_text,
    )

    return selected


def create_top_n_filter(
    min_n: int = 5,
    max_n: int = 50,
    default_n: int = 10,
    step: int = 5,
    key: str = "top_n_filter",
    help_text: str = "Number of items to display",
) -> int:
    """
    Create a top N selector for limiting results.

    Args:
        min_n: Minimum number
        max_n: Maximum number
        default_n: Default number
        step: Step size
        key: Unique widget key
        help_text: Help text for the selector

    Returns:
        Selected number
    """
    selected_n = st.slider(
        "Show Top N",
        min_value=min_n,
        max_value=max_n,
        value=default_n,
        step=step,
        key=key,
        help=help_text,
    )

    return selected_n


def create_checkbox_filter(
    label: str,
    default: bool = True,
    key: str = "checkbox_filter",
    help_text: str = "",
) -> bool:
    """
    Create a checkbox filter.

    Args:
        label: Label for the checkbox
        default: Default checked state
        key: Unique widget key
        help_text: Help text for the filter

    Returns:
        Boolean checkbox state
    """
    checked = st.checkbox(
        label,
        value=default,
        key=key,
        help=help_text,
    )

    return checked


def create_filter_sidebar(
    states: List[str],
    remoteness_categories: List[str],
    show_date_filter: bool = False,
    show_metric_selector: bool = False,
    metrics: Optional[dict] = None,
) -> dict:
    """
    Create a complete filter sidebar with common filters.

    Args:
        states: List of available states
        remoteness_categories: List of remoteness categories
        show_date_filter: Whether to include date range filter
        show_metric_selector: Whether to include metric selector
        metrics: Dictionary of metrics for selector

    Returns:
        Dictionary with all filter values
    """
    filters = {}

    with st.sidebar:
        st.header("Filters")

        # State filter
        filters["states"] = create_state_filter(
            states=states,
            key="sidebar_state_filter",
        )

        # Remoteness filter
        filters["remoteness"] = create_remoteness_filter(
            remoteness_categories=remoteness_categories,
            key="sidebar_remoteness_filter",
        )

        # Optional date filter
        if show_date_filter:
            st.markdown("---")
            filters["date_range"] = create_date_range_filter(
                key="sidebar_date_filter",
            )

        # Optional metric selector
        if show_metric_selector and metrics:
            st.markdown("---")
            filters["metric"] = create_metric_selector(
                metrics=metrics,
                key="sidebar_metric_selector",
            )

        st.markdown("---")

        # Refresh button
        if st.button("🔄 Refresh Data", key="sidebar_refresh"):
            st.cache_data.clear()
            st.rerun()

    return filters


def create_search_filter(
    label: str = "Search",
    placeholder: str = "Enter search term...",
    key: str = "search_filter",
    help_text: str = "Search and filter results",
) -> str:
    """
    Create a text search filter.

    Args:
        label: Label for the search box
        placeholder: Placeholder text
        key: Unique widget key
        help_text: Help text for the filter

    Returns:
        Search query string
    """
    search_query = st.text_input(
        label,
        value="",
        placeholder=placeholder,
        key=key,
        help=help_text,
    )

    return search_query.strip()


def create_multi_column_filter(
    columns: List[str],
    default: Optional[List[str]] = None,
    key: str = "column_filter",
    help_text: str = "Select columns to display",
) -> List[str]:
    """
    Create a multi-select column filter for data tables.

    Args:
        columns: List of available columns
        default: Default selected columns (all if None)
        key: Unique widget key
        help_text: Help text for the filter

    Returns:
        List of selected columns
    """
    if default is None:
        default = columns

    selected = st.multiselect(
        "Display Columns",
        options=columns,
        default=default,
        key=key,
        help=help_text,
    )

    return selected
