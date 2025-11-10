"""
Climate & Environment Dashboard Page

Analysis of climate and environmental factors and their impact on health outcomes,
including temperature, rainfall, air quality, and environmental health indicators.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import polars as pl
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboards.config import DB_PATH, HEALTH_COLOR_SCALE, METRICS_CONFIG
from dashboards.utils.database import get_db_connection

st.set_page_config(
    page_title="Climate & Environment | AHGD Dashboard",
    page_icon="🌡️",
    layout="wide"
)

# Header
st.title("🌡️ Climate & Environment Analysis")
st.markdown("Explore environmental factors and their impact on health outcomes")

# Get database connection
try:
    db = get_db_connection(str(DB_PATH))
except Exception as e:
    st.error(f"Database connection error: {str(e)}")
    st.stop()

# Sidebar filters
with st.sidebar:
    st.header("Filters")

    # State filter
    states = db.get_states()
    selected_states = st.multiselect(
        "States",
        options=states,
        default=states,
        help="Filter by state codes",
    )

    # Remoteness filter
    remoteness_cats = db.get_remoteness_categories()
    selected_remoteness = st.multiselect(
        "Remoteness",
        options=remoteness_cats,
        default=remoteness_cats,
        help="Filter by remoteness category",
    )

    st.markdown("---")

    # Analysis options
    st.subheader("Analysis Options")

    show_trendlines = st.checkbox("Show Trendlines", value=True)
    log_scale = st.checkbox("Use Log Scale (where applicable)", value=False)

    st.markdown("---")

    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

# Get filtered data
try:
    # Build filter query
    filter_conditions = []
    if selected_states:
        states_str = "', '".join(selected_states)
        filter_conditions.append(f"state_code IN ('{states_str}')")
    if selected_remoteness:
        remoteness_str = "', '".join(selected_remoteness)
        filter_conditions.append(f"remoteness_category IN ('{remoteness_str}')")

    where_clause = " AND ".join(filter_conditions) if filter_conditions else "1=1"

    # Get data with environmental indicators
    # Note: Some environmental fields may not exist yet, so we'll handle gracefully
    sql = f"""
        SELECT
            sa2_code,
            sa2_name,
            state_code,
            remoteness_category,
            total_population,
            latitude,
            longitude,
            mortality_rate,
            utilisation_rate,
            seifa_irsd_decile,
            population_density_per_sq_km
        FROM master_health_record
        WHERE {where_clause}
    """
    data = db.query(sql)

    # Check if environmental columns exist and add them if available
    try:
        env_check_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'master_health_record'
                AND column_name IN (
                    'avg_temperature',
                    'avg_rainfall',
                    'avg_max_temperature',
                    'avg_min_temperature',
                    'air_quality_index',
                    'green_space_access',
                    'heat_wave_days',
                    'extreme_heat_days'
                )
        """
        available_cols = db.query(env_check_sql)
        has_env_data = available_cols.height > 0
    except:
        has_env_data = False

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Main content
if not has_env_data:
    # Show warning if environmental data not available
    st.warning("""
    ⚠️ **Environmental data not yet available**

    The climate and environmental fields are not currently populated in the database.
    This page will display simulated data patterns to demonstrate functionality.

    To populate real environmental data:
    1. Run BOM data extractors
    2. Integrate climate data with SA2 regions
    3. Execute the full ETL pipeline
    """)

    # Generate simulated environmental data for demonstration
    # Add simulated temperature based on latitude (southern = cooler)
    data = data.with_columns([
        (25 - (pl.col("latitude") + 25) * 0.5 + pl.lit(np.random.randn(data.height) * 2)).alias("avg_temperature"),
        (500 + (pl.col("latitude") + 25) * 20 + pl.lit(np.random.randn(data.height) * 100)).alias("avg_rainfall"),
        (30 - (pl.col("latitude") + 25) * 0.6 + pl.lit(np.random.randn(data.height) * 3)).alias("avg_max_temperature"),
        (15 - (pl.col("latitude") + 25) * 0.4 + pl.lit(np.random.randn(data.height) * 2)).alias("avg_min_temperature"),
        (50 + pl.lit(np.random.randn(data.height) * 15)).alias("air_quality_index"),
        (60 + pl.lit(np.random.randn(data.height) * 20)).clip(0, 100).alias("green_space_access"),
    ])

tab1, tab2, tab3, tab4 = st.tabs([
    "🌡️ Temperature & Health",
    "🌧️ Rainfall & Climate",
    "🌿 Environmental Quality",
    "🗺️ Geographic Patterns"
])

# Tab 1: Temperature & Health
with tab1:
    st.subheader("Temperature and Health Outcomes")

    if not has_env_data:
        st.info("📊 Displaying simulated data for demonstration purposes")

    # Temperature summary
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_temp = data["avg_temperature"].mean()
        st.metric(
            "Avg Temperature",
            f"{avg_temp:.1f}°C",
            help="Average annual temperature across regions",
        )

    with col2:
        max_temp = data["avg_max_temperature"].mean()
        st.metric(
            "Avg Maximum",
            f"{max_temp:.1f}°C",
            help="Average maximum temperature",
        )

    with col3:
        min_temp = data["avg_min_temperature"].mean()
        st.metric(
            "Avg Minimum",
            f"{min_temp:.1f}°C",
            help="Average minimum temperature",
        )

    with col4:
        temp_range = (data["avg_max_temperature"] - data["avg_min_temperature"]).mean()
        st.metric(
            "Avg Diurnal Range",
            f"{temp_range:.1f}°C",
            help="Average daily temperature range",
        )

    st.markdown("---")

    # Temperature vs Health
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Temperature vs Mortality Rate")

        fig = px.scatter(
            data.to_pandas(),
            x="avg_temperature",
            y="mortality_rate",
            size="total_population",
            color="remoteness_category",
            hover_data=["sa2_name", "state_code"],
            title="Average Temperature vs Mortality Rate",
            labels={
                "avg_temperature": "Average Temperature (°C)",
                "mortality_rate": "Mortality Rate (per 1,000)",
                "remoteness_category": "Remoteness",
            },
            trendline="ols" if show_trendlines else None,
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Temperature vs Healthcare Utilisation")

        fig = px.scatter(
            data.to_pandas(),
            x="avg_temperature",
            y="utilisation_rate",
            size="total_population",
            color="remoteness_category",
            hover_data=["sa2_name", "state_code"],
            title="Average Temperature vs Healthcare Utilisation",
            labels={
                "avg_temperature": "Average Temperature (°C)",
                "utilisation_rate": "Healthcare Utilisation Rate (%)",
                "remoteness_category": "Remoteness",
            },
            trendline="ols" if show_trendlines else None,
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    # Temperature distribution by state
    st.markdown("---")
    st.markdown("### Temperature Distribution by State")

    fig = px.box(
        data.to_pandas(),
        x="state_code",
        y="avg_temperature",
        color="state_code",
        title="Temperature Distribution Across States",
        labels={
            "state_code": "State",
            "avg_temperature": "Average Temperature (°C)",
        },
        points="outliers",
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Temperature extremes
    st.markdown("---")
    st.markdown("### Extreme Temperatures Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Hottest regions
        hottest = data.sort("avg_max_temperature", descending=True).head(10)

        fig = px.bar(
            hottest.to_pandas(),
            x="avg_max_temperature",
            y="sa2_name",
            orientation="h",
            title="Top 10 Hottest Regions (Avg Maximum Temperature)",
            labels={
                "avg_max_temperature": "Average Max Temperature (°C)",
                "sa2_name": "SA2 Region",
            },
            color="avg_max_temperature",
            color_continuous_scale="Reds",
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Coldest regions
        coldest = data.sort("avg_min_temperature").head(10)

        fig = px.bar(
            coldest.to_pandas(),
            x="avg_min_temperature",
            y="sa2_name",
            orientation="h",
            title="Top 10 Coldest Regions (Avg Minimum Temperature)",
            labels={
                "avg_min_temperature": "Average Min Temperature (°C)",
                "sa2_name": "SA2 Region",
            },
            color="avg_min_temperature",
            color_continuous_scale="Blues_r",
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Rainfall & Climate
with tab2:
    st.subheader("Rainfall and Climate Patterns")

    if not has_env_data:
        st.info("📊 Displaying simulated data for demonstration purposes")

    # Rainfall summary
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_rainfall = data["avg_rainfall"].mean()
        st.metric(
            "Avg Annual Rainfall",
            f"{avg_rainfall:.0f}mm",
            help="Average annual rainfall across regions",
        )

    with col2:
        min_rainfall = data["avg_rainfall"].min()
        st.metric(
            "Driest Region",
            f"{min_rainfall:.0f}mm",
            help="Lowest average annual rainfall",
        )

    with col3:
        max_rainfall = data["avg_rainfall"].max()
        st.metric(
            "Wettest Region",
            f"{max_rainfall:.0f}mm",
            help="Highest average annual rainfall",
        )

    with col4:
        # Climate zones based on rainfall
        arid = data.filter(pl.col("avg_rainfall") < 300).height
        st.metric(
            "Arid Regions",
            f"{arid:,}",
            help="Regions with <300mm annual rainfall",
        )

    st.markdown("---")

    # Rainfall vs Health
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Rainfall vs Mortality Rate")

        fig = px.scatter(
            data.to_pandas(),
            x="avg_rainfall",
            y="mortality_rate",
            size="total_population",
            color="state_code",
            hover_data=["sa2_name", "remoteness_category"],
            title="Average Rainfall vs Mortality Rate",
            labels={
                "avg_rainfall": "Average Annual Rainfall (mm)",
                "mortality_rate": "Mortality Rate (per 1,000)",
                "state_code": "State",
            },
            trendline="ols" if show_trendlines else None,
            log_x=log_scale,
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Rainfall Distribution")

        fig = px.histogram(
            data.to_pandas(),
            x="avg_rainfall",
            nbins=50,
            title="Distribution of Average Annual Rainfall",
            labels={"avg_rainfall": "Average Annual Rainfall (mm)", "count": "Number of Regions"},
            color_discrete_sequence=["#3498db"],
        )
        fig.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Climate zones analysis
    st.markdown("---")
    st.markdown("### Climate Zone Classification")

    # Define climate zones based on rainfall and temperature
    data = data.with_columns([
        pl.when(pl.col("avg_rainfall") < 300)
        .then(pl.lit("Arid"))
        .when(pl.col("avg_rainfall") < 600)
        .then(pl.lit("Semi-Arid"))
        .when(pl.col("avg_rainfall") < 1000)
        .then(pl.lit("Temperate"))
        .otherwise(pl.lit("Humid"))
        .alias("climate_zone")
    ])

    col1, col2 = st.columns(2)

    with col1:
        # Climate zone distribution
        zone_counts = (
            data.group_by("climate_zone")
            .agg(pl.count("sa2_code").alias("count"))
            .sort("count", descending=True)
        )

        fig = px.pie(
            zone_counts.to_pandas(),
            values="count",
            names="climate_zone",
            title="Distribution of SA2 Regions by Climate Zone",
            color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Health outcomes by climate zone
        zone_health = (
            data.group_by("climate_zone")
            .agg([
                pl.mean("mortality_rate").alias("avg_mortality"),
                pl.mean("utilisation_rate").alias("avg_utilisation"),
                pl.count("sa2_code").alias("count")
            ])
        )

        fig = px.bar(
            zone_health.to_pandas(),
            x="climate_zone",
            y=["avg_mortality", "avg_utilisation"],
            title="Average Health Metrics by Climate Zone",
            labels={
                "value": "Rate",
                "climate_zone": "Climate Zone",
                "variable": "Metric",
            },
            barmode="group",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Temperature-Rainfall relationship
    st.markdown("---")
    st.markdown("### Temperature-Rainfall Relationship")

    fig = px.scatter(
        data.to_pandas(),
        x="avg_rainfall",
        y="avg_temperature",
        size="total_population",
        color="climate_zone",
        hover_data=["sa2_name", "state_code"],
        title="Rainfall vs Temperature by Climate Zone",
        labels={
            "avg_rainfall": "Average Annual Rainfall (mm)",
            "avg_temperature": "Average Temperature (°C)",
            "climate_zone": "Climate Zone",
        },
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: Environmental Quality
with tab3:
    st.subheader("Environmental Quality and Health")

    if not has_env_data:
        st.info("📊 Displaying simulated data for demonstration purposes")

    # Environmental quality summary
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_aqi = data["air_quality_index"].mean()
        st.metric(
            "Avg Air Quality Index",
            f"{avg_aqi:.1f}",
            help="Average air quality index (lower is better)",
        )

    with col2:
        good_air = data.filter(pl.col("air_quality_index") < 50).height
        st.metric(
            "Good Air Quality",
            f"{good_air:,}",
            help="Regions with good air quality (AQI < 50)",
        )

    with col3:
        avg_green = data["green_space_access"].mean()
        st.metric(
            "Avg Green Space Access",
            f"{avg_green:.1f}%",
            help="Average green space access percentage",
        )

    with col4:
        high_green = data.filter(pl.col("green_space_access") > 70).height
        st.metric(
            "High Green Space",
            f"{high_green:,}",
            help="Regions with >70% green space access",
        )

    st.markdown("---")

    # Air quality analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Air Quality vs Mortality Rate")

        fig = px.scatter(
            data.to_pandas(),
            x="air_quality_index",
            y="mortality_rate",
            size="total_population",
            color="remoteness_category",
            hover_data=["sa2_name", "state_code"],
            title="Air Quality Index vs Mortality Rate",
            labels={
                "air_quality_index": "Air Quality Index",
                "mortality_rate": "Mortality Rate (per 1,000)",
                "remoteness_category": "Remoteness",
            },
            trendline="ols" if show_trendlines else None,
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Air Quality Distribution")

        fig = px.histogram(
            data.to_pandas(),
            x="air_quality_index",
            nbins=40,
            title="Distribution of Air Quality Index",
            labels={"air_quality_index": "Air Quality Index", "count": "Number of Regions"},
            color_discrete_sequence=["#e74c3c"],
        )
        fig.add_vline(
            x=50,
            line_dash="dash",
            line_color="green",
            annotation_text="Good threshold",
        )
        fig.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Green space analysis
    st.markdown("---")
    st.markdown("### Green Space and Health Outcomes")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Green Space Access vs Mortality")

        fig = px.scatter(
            data.to_pandas(),
            x="green_space_access",
            y="mortality_rate",
            size="total_population",
            color="seifa_irsd_decile",
            hover_data=["sa2_name", "state_code"],
            title="Green Space Access vs Mortality Rate",
            labels={
                "green_space_access": "Green Space Access (%)",
                "mortality_rate": "Mortality Rate (per 1,000)",
                "seifa_irsd_decile": "SEIFA Decile",
            },
            trendline="ols" if show_trendlines else None,
            color_continuous_scale="RdYlGn",
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Green Space by Remoteness")

        fig = px.box(
            data.to_pandas(),
            x="remoteness_category",
            y="green_space_access",
            color="remoteness_category",
            title="Green Space Access by Remoteness Category",
            labels={
                "remoteness_category": "Remoteness",
                "green_space_access": "Green Space Access (%)",
            },
        )
        fig.update_layout(height=450, showlegend=False)
        fig.update_xaxis(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # Environmental quality composite
    st.markdown("---")
    st.markdown("### Environmental Quality Score")

    # Calculate composite environmental score
    data = data.with_columns([
        (
            (100 - pl.col("air_quality_index")) * 0.5 +
            pl.col("green_space_access") * 0.5
        ).alias("environmental_score")
    ])

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            data.to_pandas(),
            x="environmental_score",
            nbins=40,
            title="Distribution of Environmental Quality Score",
            labels={
                "environmental_score": "Environmental Score (0-100)",
                "count": "Number of Regions",
            },
            color_discrete_sequence=["#27ae60"],
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            data.to_pandas(),
            x="environmental_score",
            y="mortality_rate",
            size="total_population",
            color="remoteness_category",
            hover_data=["sa2_name", "state_code"],
            title="Environmental Quality vs Mortality Rate",
            labels={
                "environmental_score": "Environmental Quality Score",
                "mortality_rate": "Mortality Rate (per 1,000)",
                "remoteness_category": "Remoteness",
            },
            trendline="ols" if show_trendlines else None,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# Tab 4: Geographic Patterns
with tab4:
    st.subheader("Geographic and Climate Patterns")

    # Climate gradient analysis
    st.markdown("### Climate Gradient: North to South")

    # Group by latitude bands
    data = data.with_columns([
        pl.col("latitude").round(1).alias("lat_band")
    ])

    lat_summary = (
        data.group_by("lat_band")
        .agg([
            pl.mean("avg_temperature").alias("avg_temp"),
            pl.mean("avg_rainfall").alias("avg_rain"),
            pl.mean("mortality_rate").alias("avg_mortality"),
            pl.count("sa2_code").alias("count")
        ])
        .filter(pl.col("count") > 5)  # Only include bands with sufficient data
        .sort("lat_band")
    )

    # Create dual-axis plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=lat_summary["lat_band"].to_list(),
            y=lat_summary["avg_temp"].to_list(),
            name="Temperature",
            line=dict(color="#e74c3c", width=3),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=lat_summary["lat_band"].to_list(),
            y=lat_summary["avg_rain"].to_list(),
            name="Rainfall",
            line=dict(color="#3498db", width=3),
        ),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Latitude (degrees)")
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
    fig.update_yaxes(title_text="Rainfall (mm)", secondary_y=True)
    fig.update_layout(
        title="Climate Gradient: Temperature and Rainfall by Latitude",
        height=450,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Remoteness and environmental factors
    st.markdown("---")
    st.markdown("### Environmental Factors by Remoteness")

    remoteness_env = (
        data.group_by("remoteness_category")
        .agg([
            pl.mean("avg_temperature").alias("Temperature"),
            pl.mean("avg_rainfall").alias("Rainfall"),
            pl.mean("air_quality_index").alias("Air Quality Index"),
            pl.mean("green_space_access").alias("Green Space Access"),
        ])
    )

    # Normalize for radar chart
    fig = go.Figure()

    for i, row in enumerate(remoteness_env.iter_rows(named=True)):
        fig.add_trace(go.Scatterpolar(
            r=[
                row["Temperature"] / 30 * 100,  # Normalize to 0-100
                row["Rainfall"] / 1500 * 100,
                100 - row["Air Quality Index"],  # Invert so higher is better
                row["Green Space Access"],
            ],
            theta=["Temperature", "Rainfall", "Air Quality", "Green Space"],
            fill="toself",
            name=row["remoteness_category"],
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Environmental Profile by Remoteness Category (normalized)",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    # State comparison
    st.markdown("---")
    st.markdown("### State-Level Environmental Comparison")

    state_env = (
        data.group_by("state_code")
        .agg([
            pl.mean("avg_temperature").alias("avg_temp"),
            pl.mean("avg_rainfall").alias("avg_rain"),
            pl.mean("air_quality_index").alias("avg_aqi"),
            pl.mean("green_space_access").alias("avg_green"),
            pl.mean("mortality_rate").alias("avg_mortality"),
            pl.count("sa2_code").alias("region_count"),
        ])
        .sort("state_code")
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            state_env.to_pandas(),
            x="state_code",
            y=["avg_temp", "avg_rain"],
            title="Average Temperature and Rainfall by State",
            labels={"value": "Value", "state_code": "State", "variable": "Metric"},
            barmode="group",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            state_env.to_pandas(),
            x="state_code",
            y=["avg_aqi", "avg_green"],
            title="Air Quality and Green Space by State",
            labels={"value": "Value", "state_code": "State", "variable": "Metric"},
            barmode="group",
            color_discrete_sequence=["#e74c3c", "#27ae60"],
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Combined environmental-health view
    st.markdown("---")
    st.markdown("### Environmental Health Risk Assessment")

    st.markdown("""
    This composite view combines environmental stressors (temperature extremes, air quality)
    with health outcomes to identify regions with elevated environmental health risks.
    """)

    # Calculate environmental risk score
    data = data.with_columns([
        (
            (pl.col("air_quality_index") - 50).clip(0, None) * 0.3 +  # AQI above 50
            (pl.col("avg_max_temperature") - 35).clip(0, None) * 2 +  # Extreme heat
            (100 - pl.col("green_space_access")) * 0.2  # Low green space
        ).alias("env_risk_score")
    ])

    # Top environmental risk regions
    high_risk = data.sort("env_risk_score", descending=True).head(20)

    fig = px.scatter(
        high_risk.to_pandas(),
        x="env_risk_score",
        y="mortality_rate",
        size="total_population",
        color="climate_zone",
        hover_data=["sa2_name", "state_code", "remoteness_category"],
        title="Top 20 Regions: Environmental Risk vs Mortality Rate",
        labels={
            "env_risk_score": "Environmental Risk Score",
            "mortality_rate": "Mortality Rate (per 1,000)",
            "climate_zone": "Climate Zone",
        },
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Display table of high-risk regions
    with st.expander("View High Environmental Risk Regions"):
        display_cols = [
            "sa2_name",
            "state_code",
            "remoteness_category",
            "avg_max_temperature",
            "air_quality_index",
            "green_space_access",
            "env_risk_score",
            "mortality_rate",
        ]
        st.dataframe(
            high_risk.select(display_cols).to_pandas(),
            use_container_width=True,
        )

# Footer
st.markdown("---")
if not has_env_data:
    st.info("""
    💡 **Note:** This dashboard is displaying simulated environmental data for demonstration.
    Real climate data from BOM will be integrated in future pipeline runs.
    """)

st.caption("Data source: AHGD Pipeline | Climate data from BOM | Environmental indicators from multiple sources")
