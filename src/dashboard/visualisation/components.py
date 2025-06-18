"""
Reusable UI components for Australian Health Analytics Dashboard

This module contains utility functions for common visualization patterns,
data formatting, and reusable dashboard components.

Author: Portfolio Demonstration
Date: June 2025
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Any, Optional, Union


def display_key_metrics(data: pd.DataFrame, indicator: str, title: str = None) -> None:
    """
    Display key statistical metrics in a 4-column layout
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the indicator data
    indicator : str
        Column name of the indicator to analyze
    title : str, optional
        Custom title for the metrics section
    """
    
    if title:
        st.subheader(title)
    
    indicator_data = data[indicator].dropna()
    
    if indicator_data.empty:
        st.warning(f"No data available for {indicator}")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Mean Value",
            f"{indicator_data.mean():.2f}",
            delta=f"±{indicator_data.std():.2f}"
        )
    
    with col2:
        st.metric(
            "Minimum",
            f"{indicator_data.min():.2f}"
        )
    
    with col3:
        st.metric(
            "Maximum", 
            f"{indicator_data.max():.2f}"
        )
    
    with col4:
        st.metric(
            "SA2 Areas",
            f"{len(indicator_data):,}"
        )


def create_health_indicator_selector() -> Dict[str, str]:
    """
    Create standardized health indicator options for dashboard
    
    Returns:
    --------
    Dict[str, str]
        Dictionary mapping indicator keys to display names
    """
    
    health_indicators = {
        'health_risk_score': 'Composite Health Risk Score',
        'mortality_rate': 'Mortality Rate',
        'diabetes_prevalence': 'Diabetes Prevalence',
        'heart_disease_rate': 'Heart Disease Rate',
        'mental_health_rate': 'Mental Health Issues Rate',
        'gp_access_score': 'GP Access Score',
        'hospital_distance': 'Distance to Hospital (km)'
    }
    
    return health_indicators


def format_health_indicator_name(indicator: str) -> str:
    """
    Format health indicator names for display
    
    Parameters:
    -----------
    indicator : str
        Raw indicator column name
        
    Returns:
    --------
    str
        Formatted display name
    """
    
    indicator_mapping = create_health_indicator_selector()
    
    return indicator_mapping.get(indicator, indicator.replace('_', ' ').title())


def display_correlation_insights(correlation_matrix: pd.DataFrame, target_variable: str = 'IRSD_Score') -> None:
    """
    Display key correlation insights in a structured format
    
    Parameters:
    -----------
    correlation_matrix : pd.DataFrame
        Correlation matrix containing relationships between variables
    target_variable : str
        Primary variable to analyze correlations for (default: 'IRSD_Score')
    """
    
    if target_variable not in correlation_matrix.columns:
        st.error(f"Target variable '{target_variable}' not found in correlation matrix")
        return
    
    # Find strongest correlations with target variable
    target_correlations = correlation_matrix[target_variable].drop(target_variable).abs().sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Strongest Correlations with {format_health_indicator_name(target_variable)}:**")
        for var, corr in target_correlations.head(5).items():
            direction = "Positive" if correlation_matrix[target_variable][var] > 0 else "Negative"
            formatted_var = format_health_indicator_name(var)
            st.write(f"- **{formatted_var}**: {direction} ({corr:.3f})")
    
    with col2:
        st.markdown("**Statistical Significance:**")
        st.write("- Correlations > 0.3: Moderate relationship")
        st.write("- Correlations > 0.5: Strong relationship") 
        st.write("- Correlations > 0.7: Very strong relationship")
        
        significant_correlations = target_correlations[target_correlations > 0.3]
        st.write(f"- **{len(significant_correlations)}** variables show moderate+ correlation")


def display_hotspot_card(row: pd.Series, index: int = 0) -> None:
    """
    Display a health hotspot area in an expandable card format
    
    Parameters:
    -----------
    row : pd.Series
        Data row containing area information
    index : int
        Index number for display ordering
    """
    
    with st.expander(f"🚨 {row['SA2_NAME21']}, {row['STATE_NAME21']}"):
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Health Indicators**")
            st.write(f"Health Risk Score: {row['health_risk_score']:.2f}")
            if 'mortality_rate' in row:
                st.write(f"Mortality Rate: {row['mortality_rate']:.2f}")
            if 'diabetes_prevalence' in row:
                st.write(f"Diabetes Prevalence: {row['diabetes_prevalence']:.2f}%")
        
        with col2:
            st.markdown("**Socio-Economic Status**")
            st.write(f"SEIFA Score: {row['IRSD_Score']:.0f}")
            st.write(f"SEIFA Decile: {row['IRSD_Decile_Australia']:.0f}/10")
            disadvantage_level = "High" if row['IRSD_Decile_Australia'] <= 3 else "Moderate"
            st.write(f"Disadvantage Level: {disadvantage_level}")
        
        with col3:
            st.markdown("**Intervention Priority**")
            priority_score = (row['health_risk_score'] * 0.6) + ((10 - row['IRSD_Decile_Australia']) * 0.4)
            st.write(f"Priority Score: {priority_score:.2f}/10")
            
            if priority_score >= 7:
                st.markdown("🔴 **Immediate Intervention Required**")
            elif priority_score >= 5:
                st.markdown("🟡 **Medium Priority**")
            else:
                st.markdown("🟢 **Lower Priority**")


def create_data_quality_metrics(data: pd.DataFrame) -> None:
    """
    Display data quality assessment metrics
    
    Parameters:
    -----------
    data : pd.DataFrame
        Primary dataset to assess
    """
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Geographic Coverage",
            f"{len(data):,} SA2 Areas",
            help="Complete coverage of Australian Statistical Areas"
        )
    
    with col2:
        if 'IRSD_Score' in data.columns:
            completeness = (data['IRSD_Score'].notna().sum() / len(data)) * 100
            st.metric(
                "SEIFA Data Completeness",
                f"{completeness:.1f}%",
                help="Percentage of SA2 areas with SEIFA data"
            )
    
    with col3:
        if 'health_risk_score' in data.columns:
            health_completeness = (data['health_risk_score'].notna().sum() / len(data)) * 100
            st.metric(
                "Health Data Completeness",
                f"{health_completeness:.1f}%",
                help="Percentage of areas with health indicators"
            )


def create_performance_metrics(data: pd.DataFrame) -> None:
    """
    Display model performance metrics
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset containing SEIFA and health risk data
    """
    
    if 'IRSD_Score' in data.columns and 'health_risk_score' in data.columns:
        valid_data = data.dropna(subset=['IRSD_Score', 'health_risk_score'])
        
        if len(valid_data) > 0:
            correlation = valid_data['IRSD_Score'].corr(valid_data['health_risk_score'])
            r_squared = correlation ** 2
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "SEIFA-Health Correlation",
                    f"{correlation:.3f}",
                    help="Correlation between disadvantage and health risk"
                )
            
            with col2:
                st.metric(
                    "Explained Variance (R²)",
                    f"{r_squared:.3f}",
                    help="Proportion of health risk explained by disadvantage"
                )


def apply_custom_styling() -> None:
    """
    Apply custom CSS styling to the dashboard
    """
    
    st.markdown("""
    <style>
        .main > div {
            padding-top: 2rem;
        }
        .stMetric {
            background-color: #f0f2f6;
            border: 1px solid #e1e5ea;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .health-hotspot {
            background-color: #ffebee;
            border-left: 4px solid #f44336;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .improvement-opportunity {
            background-color: #e8f5e8;
            border-left: 4px solid #4caf50;
            padding: 1rem;
            margin: 0.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)


def format_number(value: Union[int, float], decimal_places: int = 2) -> str:
    """
    Format numbers for consistent display
    
    Parameters:
    -----------
    value : Union[int, float]
        Numeric value to format
    decimal_places : int
        Number of decimal places (default: 2)
        
    Returns:
    --------
    str
        Formatted number string
    """
    
    if pd.isna(value):
        return "N/A"
    
    if isinstance(value, int) or value.is_integer():
        return f"{int(value):,}"
    
    return f"{value:.{decimal_places}f}"


def create_data_filter_sidebar(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create standardized sidebar filters for data
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to create filters for
        
    Returns:
    --------
    pd.DataFrame
        Filtered dataset based on user selections
    """
    
    st.sidebar.header("🔧 Dashboard Controls")
    
    # State filter
    if 'STATE_NAME21' in data.columns:
        available_states = sorted(data['STATE_NAME21'].dropna().unique())
        selected_states = st.sidebar.multiselect(
            "Filter by State/Territory",
            available_states,
            default=available_states
        )
        
        if selected_states:
            data = data[data['STATE_NAME21'].isin(selected_states)]
    
    # Additional filters can be added here
    
    return data