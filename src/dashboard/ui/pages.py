"""
Page Components for Australian Health Analytics Dashboard

This module contains all the individual analysis mode pages:
- Geographic Health Explorer
- Correlation Analysis
- Health Hotspot Identification
- Predictive Risk Analysis
- Data Quality & Methodology
"""

import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import st_folium

from ..data.loaders import calculate_correlations
from ..data.processors import identify_health_hotspots
from ..visualisation import (
    create_health_risk_map,
    create_correlation_heatmap,
    create_scatter_plots,
    display_key_metrics,
    create_health_indicator_selector,
    display_correlation_insights,
    display_hotspot_card,
    create_data_quality_metrics,
    create_performance_metrics
)


def render_geographic_health_explorer(data):
    """
    Render the Geographic Health Explorer page
    
    Args:
        data: Filtered dataset for analysis
    """
    st.header("🗺️ Geographic Health Explorer")    
    # Indicator selection using standardised component
    health_indicators = create_health_indicator_selector()
    
    selected_indicator = st.selectbox(
        "Select Health Indicator to Display",
        list(health_indicators.keys()),
        format_func=lambda x: health_indicators[x]
    )
    
    # Display key statistics using standardised component
    display_key_metrics(data, selected_indicator)
    
    # Create and display map
    st.subheader("Interactive Health Risk Map")
    
    health_map = create_health_risk_map(data, selected_indicator)
    
    if health_map:
        map_data = st_folium(health_map, width=1200, height=600)
    else:
        st.warning("Unable to create map with available data")
    
    # Display data table
    st.subheader("Detailed Area Data")
    
    display_columns = [
        'SA2_NAME21', 'STATE_NAME21', 'IRSD_Score', 'IRSD_Decile_Australia',
        selected_indicator
    ]
    
    display_data = data[display_columns].dropna()
    
    st.dataframe(
        display_data.sort_values(selected_indicator, ascending=False),
        use_container_width=True
    )


def render_correlation_analysis(data):
    """
    Render the Correlation Analysis page
    
    Args:
        data: Filtered dataset for analysis
    """
    st.header("📊 Correlation Analysis")
    
    # Calculate correlations
    correlation_matrix, correlation_data = calculate_correlations(data)
    
    # Display correlation heatmap
    st.subheader("Correlation Matrix")
    
    heatmap_fig = create_correlation_heatmap(correlation_matrix)
    st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # Key insights using standardised component
    st.subheader("Key Correlation Insights")
    display_correlation_insights(correlation_matrix)
    
    # Scatter plots
    st.subheader("Relationship Visualisations")
    
    scatter_fig1, scatter_fig2 = create_scatter_plots(data)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(scatter_fig1, use_container_width=True)
    with col2:
        st.plotly_chart(scatter_fig2, use_container_width=True)


def render_health_hotspot_identification(data):
    """
    Render the Health Hotspot Identification page
    
    Args:
        data: Filtered dataset for analysis
    """
    st.header("🎯 Health Hotspot Identification")
    
    # Identify hotspots
    hotspots = identify_health_hotspots(data)
    
    st.subheader(f"Top {len(hotspots)} Health Priority Areas")
    st.markdown("""
    Areas identified with **high health risk** and **high socio-economic disadvantage** - 
    representing the greatest need for targeted health interventions.
    """)
    
    # Display hotspot metrics
    if not hotspots.empty:
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Priority Areas Identified",
                len(hotspots)
            )
        
        with col2:
            avg_risk = hotspots['health_risk_score'].mean()
            national_avg = data['health_risk_score'].mean()
            st.metric(
                "Average Health Risk",
                f"{avg_risk:.2f}",
                delta=f"{avg_risk - national_avg:.2f} vs national avg"
            )
        
        with col3:
            avg_disadvantage = hotspots['IRSD_Score'].mean()
            national_avg_disadvantage = data['IRSD_Score'].mean()
            st.metric(
                "Average SEIFA Score",
                f"{avg_disadvantage:.0f}",
                delta=f"{avg_disadvantage - national_avg_disadvantage:.0f} vs national avg"
            )
        
        # Hotspot details
        st.subheader("Priority Area Details")
        
        hotspot_display = hotspots[[
            'SA2_NAME21', 'STATE_NAME21', 'health_risk_score',
            'IRSD_Score', 'IRSD_Decile_Australia', 'mortality_rate', 'diabetes_prevalence'
        ]].round(2)
        
        # Display hotspot cards using standardised component
        for idx, row in hotspot_display.iterrows():
            display_hotspot_card(row, idx)
    
    else:
        st.warning("No health hotspots identified with current filters")

def render_predictive_risk_analysis(data):
    """
    Render the Predictive Risk Analysis page
    
    Args:
        data: Filtered dataset for analysis
    """
    st.header("🔮 Predictive Risk Analysis")
    
    st.subheader("Health Risk Prediction Tool")
    st.markdown("""
    Enter socio-economic characteristics to predict health risk score using our correlation model.
    """)
    
    # Input controls
    col1, col2 = st.columns(2)
    
    with col1:
        input_seifa = st.slider(
            "SEIFA Disadvantage Score",
            min_value=500,
            max_value=1200,
            value=1000,
            help="Higher scores indicate less disadvantage"
        )
        
        input_population = st.slider(
            "Population Density Factor",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Urban areas typically have different health patterns"
        )
    
    with col2:
        input_age_factor = st.slider(
            "Age Profile Factor",
            min_value=0.8,
            max_value=1.5,
            value=1.0,
            step=0.1,
            help="Areas with older populations typically have higher health risks"
        )
        
        input_access_factor = st.slider(
            "Healthcare Access Factor",
            min_value=0.5,
            max_value=1.5,
            value=1.0,
            step=0.1,
            help="Remote areas typically have lower healthcare access"
        )
    
    # Predict health risk
    if st.button("Calculate Predicted Health Risk"):
        
        # Simple prediction model based on correlations
        base_risk = 8.5 - ((input_seifa - 1000) / 100) * 0.8
        population_adjustment = (input_population - 1.0) * 2.0
        age_adjustment = (input_age_factor - 1.0) * 5.0
        access_adjustment = (1.0 - input_access_factor) * 3.0
        
        predicted_risk = max(0, base_risk + population_adjustment + age_adjustment + access_adjustment)
        
        # Display prediction
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Predicted Health Risk Score",
                f"{predicted_risk:.2f}"
            )
        
        with col2:
            national_avg = data['health_risk_score'].mean()
            risk_comparison = predicted_risk - national_avg
            comparison_text = "Above Average" if risk_comparison > 0 else "Below Average"
            st.metric(
                "vs National Average",
                comparison_text,
                delta=f"{risk_comparison:.2f}"
            )
        
        with col3:
            risk_level = "High" if predicted_risk > 10 else "Moderate" if predicted_risk > 6 else "Low"
            st.metric(
                "Risk Level",
                risk_level
            )
    
    # Scenario analysis
    st.subheader("Scenario Analysis: Impact of Disadvantage Reduction")
    
    st.markdown("**What if socio-economic disadvantage improved?**")
    
    improvement_scenario = st.slider(
        "SEIFA Score Improvement (%)",
        min_value=0,
        max_value=50,
        value=10,
        help="Simulate improvement in socio-economic conditions"
    )
    
    if improvement_scenario > 0:
        
        # Calculate scenario impact
        scenario_data = data.copy()
        scenario_data['improved_seifa'] = scenario_data['IRSD_Score'] * (1 + improvement_scenario/100)
        scenario_data['improved_health_risk'] = scenario_data['health_risk_score'] * (1 - improvement_scenario/200)
        
        original_avg_risk = scenario_data['health_risk_score'].mean()
        improved_avg_risk = scenario_data['improved_health_risk'].mean()
        risk_reduction = original_avg_risk - improved_avg_risk
        
        # Display scenario results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Avg Health Risk",
                f"{original_avg_risk:.2f}"
            )
        
        with col2:
            st.metric(
                "Projected Avg Health Risk",
                f"{improved_avg_risk:.2f}",
                delta=f"-{risk_reduction:.2f}"
            )
        
        with col3:
            population_affected = len(scenario_data) * 3000  # Approximate population per SA2
            st.metric(
                "Population Benefiting",
                f"{population_affected:,}"
            )
def render_data_quality_methodology(data):
    """
    Render the Data Quality & Methodology page
    
    Args:
        data: Complete dataset for quality assessment
    """
    st.header("📋 Data Quality & Methodology")
    
    # Data sources
    st.subheader("Data Sources")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Primary Data Sources:**
        - Australian Bureau of Statistics (ABS)
        - SEIFA 2021 (Socio-Economic Indexes for Areas)
        - Statistical Area Level 2 (SA2) Boundaries 2021
        - Australian Institute of Health and Welfare (AIHW)
        - Public Health Information Development Unit (PHIDU)
        """)
    
    with col2:
        st.markdown("""
        **Data Currency:**
        - Geographic boundaries: 2021 Census
        - SEIFA data: 2021 Census
        - Health indicators: Latest available (2020-2022)
        - Processing date: June 2025
        """)
    
    # Methodology
    st.subheader("Methodology")
    
    st.markdown("""
    **Health Risk Score Calculation:**
    
    The composite health risk score combines multiple indicators:
    - Mortality rate (30% weight)
    - Diabetes prevalence (20% weight)  
    - Heart disease rate (15% weight)
    - Mental health issues rate (10% weight)
    - GP access score (15% weight, inverted)
    - Hospital distance (10% weight)
    
    **Correlation Analysis:**
    - Pearson correlation coefficients calculated between SEIFA disadvantage scores and health outcomes
    - Statistical significance assessed using p-values
    - Regression analysis with 95% confidence intervals
    
    **Health Hotspot Identification:**
    - Areas in top 30% for health risk AND bottom 30% for SEIFA scores
    - Ranked by composite risk score
    - Validated against known health inequality research
    """)
    
    # Data quality metrics using standardised component
    st.subheader("Data Quality Assessment")
    create_data_quality_metrics(data)
    
    # Performance metrics using standardised component
    st.subheader("Model Performance")
    create_performance_metrics(data)
    
    # Limitations and assumptions
    st.subheader("Limitations & Assumptions")
    
    st.markdown("""
    **Important Limitations:**
    - Health indicators are modelled for demonstration purposes
    - Actual implementation would require access to confidential health databases
    - Correlation does not imply causation
    - SA2 level analysis may mask within-area variation
    - Temporal alignment between datasets may vary
    
    **Key Assumptions:**
    - Linear relationships between disadvantage and health outcomes
    - SA2 boundaries accurately represent community characteristics
    - SEIFA scores remain relatively stable over analysis period
    - Health service access correlates with geographic proximity
    """)
    
    # Technical details
    st.subheader("Technical Implementation")
    
    st.markdown("""
    **Technology Stack:**
    - **Frontend**: Streamlit for interactive dashboard
    - **Mapping**: Folium for geographic visualisation
    - **Analytics**: Pandas, NumPy, SciPy for data processing
    - **Visualisation**: Plotly, Altair for statistical charts
    - **Geospatial**: GeoPandas for spatial data operations
    - **Database**: SQLite/DuckDB for data storage
    
    **Data Processing Pipeline:**
    1. Extract data from ABS and health databases
    2. Standardise geographic identifiers (SA2 codes)
    3. Validate data completeness and quality
    4. Calculate composite indicators and risk scores
    5. Generate correlation matrices and statistical tests
    6. Create interactive visualisations and reports
    """)


# Page routing function
def get_page_renderer(analysis_type: str):
    """
    Get the appropriate page renderer function for the analysis type
    
    Args:
        analysis_type: Selected analysis type
        
    Returns:
        Page renderer function
    """
    page_map = {
        "Geographic Health Explorer": render_geographic_health_explorer,
        "Correlation Analysis": render_correlation_analysis,
        "Health Hotspot Identification": render_health_hotspot_identification,
        "Predictive Risk Analysis": render_predictive_risk_analysis,
        "Data Quality & Methodology": render_data_quality_methodology
    }
    
    return page_map.get(analysis_type, render_geographic_health_explorer)


def render_page(analysis_type: str, data):
    """
    Render the selected page with error handling
    
    Args:
        analysis_type: Selected analysis type
        data: Dataset for analysis
    """
    try:
        renderer = get_page_renderer(analysis_type)
        renderer(data)
    except Exception as e:
        st.error(f"Error rendering {analysis_type} page: {str(e)}")
        st.markdown("""
        **Troubleshooting:**
        - Check data availability and format
        - Verify all required columns exist
        - Try refreshing the page
        - Contact support if the issue persists
        """)