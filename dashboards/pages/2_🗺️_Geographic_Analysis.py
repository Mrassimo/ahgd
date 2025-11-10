"""
Geographic Analysis Dashboard Page

Interactive maps and spatial analysis of health indicators across Australia's
SA2 regions with state comparisons and remoteness analysis.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import numpy as np
from pathlib import Path
import sys
import json
from typing import Optional, Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboards.config import (
    DB_PATH,
    HEALTH_COLOR_SCALE,
    METRICS_CONFIG,
    MAP_CENTER,
    MAP_ZOOM,
    STATE_CODES,
)
from dashboards.utils.database import get_db_connection

st.set_page_config(
    page_title="Geographic Analysis | AHGD Dashboard",
    page_icon="🗺️",
    layout="wide"
)

# Header
st.title("🗺️ Geographic Analysis")
st.markdown("Interactive spatial analysis of health indicators across Australian SA2 regions")

# Get database connection
try:
    db = get_db_connection(str(DB_PATH))
except Exception as e:
    st.error(f"Database connection error: {str(e)}")
    st.info("Please ensure the AHGD pipeline has been executed and the database exists.")
    st.stop()


# Helper functions for geographic queries
@st.cache_data(ttl=300)
def get_geographic_health_data(_db, states: List[str] = None, remoteness: List[str] = None) -> pl.DataFrame:
    """
    Get health data with geographic information.

    Args:
        _db: Database connection
        states: List of state codes to filter
        remoteness: List of remoteness categories to filter

    Returns:
        Polars DataFrame with health and geographic data
    """
    sql = """
        SELECT
            mhr.sa2_code,
            mhr.sa2_name,
            mhr.state_code,
            mhr.remoteness_category,
            mhr.latitude,
            mhr.longitude,
            mhr.total_population,
            mhr.mortality_rate,
            mhr.utilisation_rate,
            mhr.bulk_billed_percentage,
            mhr.median_household_income,
            mhr.unemployment_rate,
            mhr.seifa_irsad_score,
            mhr.seifa_irsd_decile,
            dhi.composite_health_index,
            dhi.health_accessibility_score,
            dhi.socioeconomic_health_score
        FROM master_health_record mhr
        LEFT JOIN derived_health_indicators dhi
            ON mhr.sa2_code = dhi.sa2_code
        WHERE mhr.latitude IS NOT NULL
            AND mhr.longitude IS NOT NULL
    """

    conditions = []
    if states:
        state_list = ",".join([f"'{s}'" for s in states])
        conditions.append(f"mhr.state_code IN ({state_list})")

    if remoteness:
        remoteness_list = ",".join([f"'{r}'" for r in remoteness])
        conditions.append(f"mhr.remoteness_category IN ({remoteness_list})")

    if conditions:
        sql += " AND " + " AND ".join(conditions)

    try:
        return _db.query(sql)
    except Exception as e:
        st.error(f"Error loading geographic data: {str(e)}")
        return pl.DataFrame()


@st.cache_data(ttl=300)
def get_state_aggregates(_db) -> pl.DataFrame:
    """Get aggregated health metrics by state."""
    sql = """
        SELECT
            state_code,
            COUNT(DISTINCT sa2_code) as total_sa2_regions,
            SUM(total_population) as total_population,
            AVG(mortality_rate) as avg_mortality_rate,
            AVG(utilisation_rate) as avg_utilisation_rate,
            AVG(bulk_billed_percentage) as avg_bulk_billing,
            AVG(median_household_income) as avg_income,
            AVG(seifa_irsad_score) as avg_seifa_score
        FROM master_health_record
        WHERE state_code IS NOT NULL
        GROUP BY state_code
        ORDER BY state_code
    """

    try:
        return _db.query(sql)
    except Exception as e:
        st.error(f"Error loading state aggregates: {str(e)}")
        return pl.DataFrame()


@st.cache_data(ttl=300)
def get_remoteness_aggregates(_db) -> pl.DataFrame:
    """Get aggregated health metrics by remoteness category."""
    sql = """
        SELECT
            remoteness_category,
            COUNT(DISTINCT sa2_code) as total_sa2_regions,
            SUM(total_population) as total_population,
            AVG(mortality_rate) as avg_mortality_rate,
            AVG(utilisation_rate) as avg_utilisation_rate,
            AVG(bulk_billed_percentage) as avg_bulk_billing,
            AVG(median_household_income) as avg_income,
            AVG(seifa_irsad_score) as avg_seifa_score
        FROM master_health_record
        WHERE remoteness_category IS NOT NULL
        GROUP BY remoteness_category
        ORDER BY
            CASE remoteness_category
                WHEN 'Major Cities of Australia' THEN 1
                WHEN 'Inner Regional Australia' THEN 2
                WHEN 'Outer Regional Australia' THEN 3
                WHEN 'Remote Australia' THEN 4
                WHEN 'Very Remote Australia' THEN 5
                ELSE 6
            END
    """

    try:
        return _db.query(sql)
    except Exception as e:
        st.error(f"Error loading remoteness aggregates: {str(e)}")
        return pl.DataFrame()


@st.cache_data(ttl=300)
def get_spatial_clusters(_db, metric: str, n_clusters: int = 5) -> pl.DataFrame:
    """
    Identify spatial clusters using quintile-based grouping.

    Args:
        _db: Database connection
        metric: Metric to cluster on
        n_clusters: Number of clusters

    Returns:
        DataFrame with cluster assignments
    """
    sql = f"""
        WITH percentiles AS (
            SELECT
                sa2_code,
                {metric},
                NTILE({n_clusters}) OVER (ORDER BY {metric}) as cluster
            FROM master_health_record
            WHERE {metric} IS NOT NULL
                AND latitude IS NOT NULL
                AND longitude IS NOT NULL
        )
        SELECT
            mhr.sa2_code,
            mhr.sa2_name,
            mhr.state_code,
            mhr.latitude,
            mhr.longitude,
            mhr.{metric},
            p.cluster
        FROM master_health_record mhr
        INNER JOIN percentiles p ON mhr.sa2_code = p.sa2_code
    """

    try:
        return _db.query(sql)
    except Exception as e:
        st.error(f"Error calculating spatial clusters: {str(e)}")
        return pl.DataFrame()


# Sidebar filters
with st.sidebar:
    st.header("🎛️ Filters")

    # Metric selection
    metric_options = {
        "Mortality Rate": "mortality_rate",
        "Healthcare Utilisation": "utilisation_rate",
        "Bulk Billing %": "bulk_billed_percentage",
        "Composite Health Index": "composite_health_index",
        "SEIFA Score": "seifa_irsad_score",
        "Median Income": "median_household_income",
        "Unemployment Rate": "unemployment_rate",
    }

    selected_metric_name = st.selectbox(
        "Select Health Indicator",
        options=list(metric_options.keys()),
        index=0,
        help="Choose the health indicator to visualise on the map",
    )
    selected_metric = metric_options[selected_metric_name]

    st.markdown("---")

    # State filter
    states = db.get_states()
    selected_states = st.multiselect(
        "Filter by State",
        options=states,
        default=states,
        help="Select states to display (leave empty for all)",
    )

    # Remoteness filter
    remoteness_cats = db.get_remoteness_categories()
    selected_remoteness = st.multiselect(
        "Filter by Remoteness",
        options=remoteness_cats,
        default=remoteness_cats,
        help="Filter by remoteness classification",
    )

    st.markdown("---")

    # Map settings
    st.subheader("Map Settings")

    map_style = st.selectbox(
        "Map Style",
        options=["open-street-map", "carto-positron", "carto-darkmatter"],
        index=1,
        help="Select the base map style",
    )

    point_size = st.slider(
        "Point Size",
        min_value=3,
        max_value=20,
        value=8,
        help="Size of points on the map",
    )

    st.markdown("---")

    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Interactive Map",
    "📊 State Comparison",
    "🏞️ Remoteness Analysis",
    "🎯 Spatial Clusters",
    "📍 Region Details"
])

with tab1:
    st.subheader(f"Interactive Map: {selected_metric_name}")

    try:
        # Load geographic data
        geo_data = get_geographic_health_data(
            db,
            states=selected_states if selected_states else None,
            remoteness=selected_remoteness if selected_remoteness else None
        )

        if geo_data.is_empty():
            st.warning("No geographic data available with current filters.")
        else:
            # Convert to pandas for Plotly
            df = geo_data.to_pandas()

            # Create scatter map
            fig = px.scatter_mapbox(
                df,
                lat="latitude",
                lon="longitude",
                color=selected_metric,
                size="total_population",
                hover_name="sa2_name",
                hover_data={
                    "sa2_code": True,
                    "state_code": True,
                    "remoteness_category": True,
                    "total_population": ":,.0f",
                    selected_metric: ":.2f",
                    "latitude": False,
                    "longitude": False,
                },
                color_continuous_scale="RdYlGn_r" if selected_metric in ["mortality_rate", "unemployment_rate"] else "RdYlGn",
                size_max=point_size * 3,
                zoom=MAP_ZOOM,
                center={"lat": MAP_CENTER[0], "lon": MAP_CENTER[1]},
                mapbox_style=map_style,
                title=f"{selected_metric_name} across Australian SA2 Regions",
            )

            fig.update_layout(
                height=700,
                margin={"r": 0, "t": 40, "l": 0, "b": 0},
            )

            st.plotly_chart(fig, use_container_width=True)

            # Summary statistics for visible data
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "SA2 Regions Shown",
                    f"{len(df):,}",
                    help="Number of SA2 regions displayed on map"
                )

            with col2:
                st.metric(
                    f"Avg {selected_metric_name}",
                    f"{df[selected_metric].mean():.2f}",
                    help=f"Average {selected_metric_name} for visible regions"
                )

            with col3:
                st.metric(
                    f"Min {selected_metric_name}",
                    f"{df[selected_metric].min():.2f}",
                    help=f"Minimum {selected_metric_name} value"
                )

            with col4:
                st.metric(
                    f"Max {selected_metric_name}",
                    f"{df[selected_metric].max():.2f}",
                    help=f"Maximum {selected_metric_name} value"
                )

    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        st.info("Please verify that the database contains geographic data (latitude/longitude).")

with tab2:
    st.subheader("State-by-State Comparison")

    try:
        state_data = get_state_aggregates(db)

        if state_data.is_empty():
            st.warning("No state data available.")
        else:
            df = state_data.to_pandas()

            # Add state names
            df['state_name'] = df['state_code'].map(STATE_CODES)

            # Create comparison charts
            col1, col2 = st.columns(2)

            with col1:
                # Mortality rate by state
                fig = px.bar(
                    df,
                    x='state_name',
                    y='avg_mortality_rate',
                    title='Average Mortality Rate by State',
                    color='avg_mortality_rate',
                    color_continuous_scale='Reds',
                    labels={'state_name': 'State', 'avg_mortality_rate': 'Avg Mortality Rate'},
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Utilisation rate by state
                fig = px.bar(
                    df,
                    x='state_name',
                    y='avg_utilisation_rate',
                    title='Average Healthcare Utilisation by State',
                    color='avg_utilisation_rate',
                    color_continuous_scale='Blues',
                    labels={'state_name': 'State', 'avg_utilisation_rate': 'Avg Utilisation Rate (%)'},
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Population and regions comparison
            col1, col2 = st.columns(2)

            with col1:
                fig = px.pie(
                    df,
                    values='total_population',
                    names='state_name',
                    title='Population Distribution by State',
                    hole=0.4,
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.pie(
                    df,
                    values='total_sa2_regions',
                    names='state_name',
                    title='SA2 Regions by State',
                    hole=0.4,
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            # Detailed comparison table
            st.markdown("### Detailed State Comparison")

            # Format the dataframe for display
            display_df = df[['state_name', 'total_sa2_regions', 'total_population',
                           'avg_mortality_rate', 'avg_utilisation_rate',
                           'avg_bulk_billing', 'avg_income', 'avg_seifa_score']].copy()

            display_df.columns = [
                'State', 'SA2 Regions', 'Population',
                'Avg Mortality', 'Avg Utilisation (%)',
                'Avg Bulk Billing (%)', 'Avg Income ($)', 'Avg SEIFA'
            ]

            st.dataframe(
                display_df.style.format({
                    'SA2 Regions': '{:,.0f}',
                    'Population': '{:,.0f}',
                    'Avg Mortality': '{:.2f}',
                    'Avg Utilisation (%)': '{:.1f}',
                    'Avg Bulk Billing (%)': '{:.1f}',
                    'Avg Income ($)': '{:,.0f}',
                    'Avg SEIFA': '{:.0f}',
                }).background_gradient(subset=['Avg Mortality'], cmap='Reds')
                  .background_gradient(subset=['Avg Utilisation (%)'], cmap='Greens'),
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Error loading state comparison: {str(e)}")

with tab3:
    st.subheader("Health Indicators by Remoteness Category")

    try:
        remoteness_data = get_remoteness_aggregates(db)

        if remoteness_data.is_empty():
            st.warning("No remoteness data available.")
        else:
            df = remoteness_data.to_pandas()

            # Trend line chart
            st.markdown("### Health Trends by Remoteness")

            fig = go.Figure()

            # Add mortality rate trend
            fig.add_trace(go.Scatter(
                x=df['remoteness_category'],
                y=df['avg_mortality_rate'],
                name='Mortality Rate',
                mode='lines+markers',
                line=dict(color='red', width=3),
                marker=dict(size=10),
            ))

            # Add utilisation rate (scaled for visibility)
            fig.add_trace(go.Scatter(
                x=df['remoteness_category'],
                y=df['avg_utilisation_rate'] / 10,  # Scale down
                name='Utilisation Rate (÷10)',
                mode='lines+markers',
                line=dict(color='blue', width=3),
                marker=dict(size=10),
                yaxis='y2',
            ))

            fig.update_layout(
                title='Health Indicators Trend by Remoteness',
                xaxis_title='Remoteness Category',
                yaxis_title='Mortality Rate',
                yaxis2=dict(
                    title='Utilisation Rate (÷10)',
                    overlaying='y',
                    side='right'
                ),
                height=500,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Comparison bars
            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    df,
                    x='remoteness_category',
                    y='avg_bulk_billing',
                    title='Average Bulk Billing % by Remoteness',
                    color='avg_bulk_billing',
                    color_continuous_scale='Greens',
                    labels={'remoteness_category': 'Remoteness', 'avg_bulk_billing': 'Bulk Billing (%)'},
                )
                fig.update_layout(height=400, showlegend=False)
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.bar(
                    df,
                    x='remoteness_category',
                    y='avg_income',
                    title='Average Household Income by Remoteness',
                    color='avg_income',
                    color_continuous_scale='Blues',
                    labels={'remoteness_category': 'Remoteness', 'avg_income': 'Income ($)'},
                )
                fig.update_layout(height=400, showlegend=False)
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

            # Population distribution
            st.markdown("### Population Distribution by Remoteness")

            fig = px.bar(
                df,
                x='remoteness_category',
                y='total_population',
                title='Population by Remoteness Category',
                color='total_sa2_regions',
                color_continuous_scale='Viridis',
                labels={
                    'remoteness_category': 'Remoteness Category',
                    'total_population': 'Total Population',
                    'total_sa2_regions': 'SA2 Regions'
                },
            )
            fig.update_layout(height=400)
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

            # Detailed table
            st.markdown("### Detailed Remoteness Comparison")

            display_df = df[['remoteness_category', 'total_sa2_regions', 'total_population',
                           'avg_mortality_rate', 'avg_utilisation_rate',
                           'avg_bulk_billing', 'avg_income', 'avg_seifa_score']].copy()

            display_df.columns = [
                'Remoteness Category', 'SA2 Regions', 'Population',
                'Avg Mortality', 'Avg Utilisation (%)',
                'Avg Bulk Billing (%)', 'Avg Income ($)', 'Avg SEIFA'
            ]

            st.dataframe(
                display_df.style.format({
                    'SA2 Regions': '{:,.0f}',
                    'Population': '{:,.0f}',
                    'Avg Mortality': '{:.2f}',
                    'Avg Utilisation (%)': '{:.1f}',
                    'Avg Bulk Billing (%)': '{:.1f}',
                    'Avg Income ($)': '{:,.0f}',
                    'Avg SEIFA': '{:.0f}',
                }).background_gradient(subset=['Avg Mortality'], cmap='Reds')
                  .background_gradient(subset=['Avg SEIFA'], cmap='Greens'),
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Error loading remoteness analysis: {str(e)}")

with tab4:
    st.subheader("Spatial Clustering Analysis")
    st.markdown("Identify geographic clusters of health indicator values")

    # Cluster settings
    col1, col2 = st.columns([2, 1])

    with col1:
        cluster_metric_name = st.selectbox(
            "Select Metric for Clustering",
            options=list(metric_options.keys()),
            index=0,
            key="cluster_metric",
        )
        cluster_metric = metric_options[cluster_metric_name]

    with col2:
        n_clusters = st.slider(
            "Number of Clusters",
            min_value=3,
            max_value=7,
            value=5,
            help="Number of groups to divide regions into"
        )

    try:
        cluster_data = get_spatial_clusters(db, cluster_metric, n_clusters)

        if cluster_data.is_empty():
            st.warning("No cluster data available.")
        else:
            df = cluster_data.to_pandas()

            # Create cluster map
            fig = px.scatter_mapbox(
                df,
                lat="latitude",
                lon="longitude",
                color="cluster",
                hover_name="sa2_name",
                hover_data={
                    "sa2_code": True,
                    "state_code": True,
                    cluster_metric: ":.2f",
                    "cluster": True,
                    "latitude": False,
                    "longitude": False,
                },
                color_continuous_scale="Viridis",
                zoom=MAP_ZOOM,
                center={"lat": MAP_CENTER[0], "lon": MAP_CENTER[1]},
                mapbox_style=map_style,
                title=f"Spatial Clusters based on {cluster_metric_name} ({n_clusters} clusters)",
            )

            fig.update_layout(
                height=600,
                margin={"r": 0, "t": 40, "l": 0, "b": 0},
            )

            st.plotly_chart(fig, use_container_width=True)

            # Cluster statistics
            st.markdown("### Cluster Statistics")

            cluster_stats = df.groupby('cluster').agg({
                'sa2_code': 'count',
                cluster_metric: ['mean', 'min', 'max', 'std']
            }).reset_index()

            cluster_stats.columns = ['Cluster', 'SA2 Count', 'Mean', 'Min', 'Max', 'Std Dev']

            st.dataframe(
                cluster_stats.style.format({
                    'SA2 Count': '{:.0f}',
                    'Mean': '{:.2f}',
                    'Min': '{:.2f}',
                    'Max': '{:.2f}',
                    'Std Dev': '{:.2f}',
                }).background_gradient(subset=['Mean'], cmap='RdYlGn_r' if cluster_metric in ['mortality_rate', 'unemployment_rate'] else 'RdYlGn'),
                use_container_width=True
            )

            # Cluster distribution
            col1, col2 = st.columns(2)

            with col1:
                fig = px.histogram(
                    df,
                    x='cluster',
                    title='Distribution of Regions by Cluster',
                    color='cluster',
                    color_continuous_scale='Viridis',
                )
                fig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.box(
                    df,
                    x='cluster',
                    y=cluster_metric,
                    title=f'{cluster_metric_name} Distribution by Cluster',
                    color='cluster',
                    color_continuous_scale='Viridis',
                )
                fig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error performing clustering analysis: {str(e)}")

with tab5:
    st.subheader("SA2 Region Details")
    st.markdown("Search and explore detailed information for specific SA2 regions")

    try:
        # Get all SA2 data
        all_data = get_geographic_health_data(db)

        if all_data.is_empty():
            st.warning("No SA2 data available.")
        else:
            df = all_data.to_pandas()

            # Search box
            search_term = st.text_input(
                "Search by SA2 Code or Name",
                placeholder="e.g., 101021007 or Sydney",
                help="Enter SA2 code or region name to search"
            )

            if search_term:
                # Filter data
                mask = (
                    df['sa2_code'].astype(str).str.contains(search_term, case=False, na=False) |
                    df['sa2_name'].astype(str).str.contains(search_term, case=False, na=False)
                )
                filtered_df = df[mask]

                if len(filtered_df) == 0:
                    st.warning(f"No regions found matching '{search_term}'")
                else:
                    st.success(f"Found {len(filtered_df)} matching region(s)")

                    # Display results
                    for idx, row in filtered_df.head(10).iterrows():
                        with st.expander(f"📍 {row['sa2_name']} ({row['sa2_code']})"):
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.markdown("**Geographic Information**")
                                st.write(f"**State:** {STATE_CODES.get(str(row['state_code']), row['state_code'])}")
                                st.write(f"**Remoteness:** {row['remoteness_category']}")
                                st.write(f"**Population:** {row['total_population']:,.0f}")
                                st.write(f"**Location:** {row['latitude']:.4f}, {row['longitude']:.4f}")

                            with col2:
                                st.markdown("**Health Indicators**")
                                st.write(f"**Mortality Rate:** {row['mortality_rate']:.2f}")
                                st.write(f"**Utilisation Rate:** {row['utilisation_rate']:.1f}%")
                                st.write(f"**Bulk Billing:** {row['bulk_billed_percentage']:.1f}%")
                                if not pd.isna(row.get('composite_health_index')):
                                    st.write(f"**Health Index:** {row['composite_health_index']:.1f}")

                            with col3:
                                st.markdown("**Socioeconomic Factors**")
                                st.write(f"**Median Income:** ${row['median_household_income']:,.0f}")
                                st.write(f"**Unemployment:** {row['unemployment_rate']:.1f}%")
                                st.write(f"**SEIFA Score:** {row['seifa_irsad_score']:.0f}")
                                st.write(f"**SEIFA Decile:** {row['seifa_irsd_decile']:.0f}")

                            # Mini map for this region
                            mini_df = pd.DataFrame([row])
                            fig = px.scatter_mapbox(
                                mini_df,
                                lat="latitude",
                                lon="longitude",
                                hover_name="sa2_name",
                                zoom=10,
                                height=300,
                                mapbox_style=map_style,
                            )
                            fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
                            st.plotly_chart(fig, use_container_width=True)

            else:
                st.info("Enter a search term to explore SA2 region details")

                # Show some example regions
                st.markdown("### Example Regions")

                sample_df = df.sample(min(5, len(df)))

                for idx, row in sample_df.iterrows():
                    st.markdown(f"- **{row['sa2_name']}** ({row['sa2_code']}) - {STATE_CODES.get(str(row['state_code']), row['state_code'])}")

    except Exception as e:
        st.error(f"Error loading region details: {str(e)}")

# Footer
st.markdown("---")
st.caption("Data source: AHGD Pipeline | Geographic visualizations powered by Plotly")

# Import pandas for region details tab
import pandas as pd
