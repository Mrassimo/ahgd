"""
Overview Dashboard Page

High-level summary of health indicators across Australia with KPIs,
interactive visualizations, and regional comparisons.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboards.config import DB_PATH, HEALTH_COLOR_SCALE, METRICS_CONFIG
from dashboards.utils.database import get_db_connection

st.set_page_config(page_title="Overview | AHGD Dashboard", page_icon="📊", layout="wide")

# Header
st.title("📊 Overview Dashboard")
st.markdown("High-level summary of health indicators across Australia")

# Get database connection
try:
    db = get_db_connection(str(DB_PATH))
except Exception as e:
    st.error(f"Database connection error: {str(e)}")
    st.stop()

# Filters
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

    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

# Main content
tab1, tab2, tab3 = st.tabs(["📈 Key Metrics", "🏆 Top/Bottom Regions", "📊 Distributions"])

with tab1:
    st.subheader("Key Health Indicators")

    # Summary metrics
    try:
        summary = db.get_health_summary()

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Total SA2 Regions",
                f"{summary['total_sa2']:,}",
                help="Number of Statistical Area Level 2 regions",
            )

        with col2:
            mortality = summary["avg_mortality_rate"]
            mortality_config = METRICS_CONFIG["mortality_rate"]
            delta_color = (
                "normal"
                if mortality < mortality_config["good_threshold"]
                else "inverse"
            )
            st.metric(
                mortality_config["label"],
                mortality_config["format"].format(mortality),
                help=mortality_config["description"],
            )

        with col3:
            utilisation = summary["avg_utilisation_rate"]
            util_config = METRICS_CONFIG["utilisation_rate"]
            st.metric(
                util_config["label"],
                util_config["format"].format(utilisation),
                help=util_config["description"],
            )

        with col4:
            st.metric(
                "Total Population",
                f"{summary['total_population']:,.0f}",
                help="Total population across all SA2 regions",
            )

        with col5:
            composite = summary["avg_composite_index"]
            composite_config = METRICS_CONFIG["composite_health_index"]
            st.metric(
                composite_config["label"],
                composite_config["format"].format(composite),
                help=composite_config["description"],
            )

    except Exception as e:
        st.error(f"Error loading summary metrics: {str(e)}")

    st.markdown("---")

    # Health indicator trends
    st.subheader("Health Indicator Distribution")

    try:
        data = db.get_master_health_record()

        # Create histogram for mortality rate
        fig = px.histogram(
            data.to_pandas(),
            x="mortality_rate",
            nbins=50,
            title="Distribution of Mortality Rates",
            labels={"mortality_rate": "Mortality Rate (per 1,000)", "count": "Number of SA2 Regions"},
            color_discrete_sequence=["#1f77b4"],
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading health indicator trends: {str(e)}")

    # Scatter plot: Mortality vs Utilisation
    st.subheader("Mortality Rate vs Healthcare Utilisation")

    try:
        data = db.get_master_health_record(limit=1000)

        fig = px.scatter(
            data.to_pandas(),
            x="utilisation_rate",
            y="mortality_rate",
            size="total_population",
            color="seifa_irsd_decile",
            hover_data=["sa2_code"],
            title="Relationship between Healthcare Utilisation and Mortality",
            labels={
                "utilisation_rate": "Healthcare Utilisation Rate (%)",
                "mortality_rate": "Mortality Rate (per 1,000)",
                "seifa_irsd_decile": "SEIFA Decile",
            },
            color_continuous_scale="RdYlGn",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error creating scatter plot: {str(e)}")

with tab2:
    st.subheader("Top and Bottom Performing Regions")

    col1, col2 = st.columns(2)

    try:
        regions = db.get_top_bottom_regions(metric="composite_health_index", n=10)

        with col1:
            st.markdown("### 🏆 Top 10 Regions")
            top_data = regions["top"].to_pandas()

            fig = px.bar(
                top_data,
                x="composite_health_index",
                y="sa2_code",
                orientation="h",
                title="Highest Composite Health Index",
                color="composite_health_index",
                color_continuous_scale="Greens",
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### ⚠️ Bottom 10 Regions")
            bottom_data = regions["bottom"].to_pandas()

            fig = px.bar(
                bottom_data,
                x="composite_health_index",
                y="sa2_code",
                orientation="h",
                title="Lowest Composite Health Index",
                color="composite_health_index",
                color_continuous_scale="Reds",
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading top/bottom regions: {str(e)}")

    # Remoteness comparison
    st.markdown("---")
    st.subheader("Health Indicators by Remoteness Category")

    try:
        data = db.get_master_health_record()

        # Box plot by remoteness
        fig = go.Figure()

        for remoteness in data["remoteness_category"].unique():
            subset = data.filter(pl.col("remoteness_category") == remoteness)
            fig.add_trace(
                go.Box(
                    y=subset["mortality_rate"].to_list(),
                    name=remoteness,
                    boxmean="sd",
                )
            )

        fig.update_layout(
            title="Mortality Rate Distribution by Remoteness",
            yaxis_title="Mortality Rate (per 1,000)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error creating remoteness comparison: {str(e)}")

with tab3:
    st.subheader("Indicator Distributions")

    # Select metric
    metric_options = {
        "Mortality Rate": "mortality_rate",
        "Utilisation Rate": "utilisation_rate",
        "Bulk Billing %": "bulk_billed_percentage",
        "SEIFA IRSAD Score": "seifa_irsad_score",
        "Unemployment Rate": "unemployment_rate",
        "Median Income": "median_household_income",
    }

    selected_metric = st.selectbox(
        "Select Indicator",
        options=list(metric_options.keys()),
    )

    metric_column = metric_options[selected_metric]

    try:
        data = db.get_master_health_record()

        col1, col2 = st.columns(2)

        with col1:
            # Histogram
            fig = px.histogram(
                data.to_pandas(),
                x=metric_column,
                nbins=50,
                title=f"Distribution of {selected_metric}",
                color_discrete_sequence=["#1f77b4"],
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Box plot
            fig = px.box(
                data.to_pandas(),
                y=metric_column,
                title=f"{selected_metric} - Box Plot",
                color_discrete_sequence=["#1f77b4"],
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        st.markdown("### Summary Statistics")

        stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5)

        with stats_col1:
            st.metric("Mean", f"{data[metric_column].mean():.2f}")

        with stats_col2:
            st.metric("Median", f"{data[metric_column].median():.2f}")

        with stats_col3:
            st.metric("Std Dev", f"{data[metric_column].std():.2f}")

        with stats_col4:
            st.metric("Min", f"{data[metric_column].min():.2f}")

        with stats_col5:
            st.metric("Max", f"{data[metric_column].max():.2f}")

    except Exception as e:
        st.error(f"Error creating distributions: {str(e)}")

# Footer
st.markdown("---")
st.caption("Data source: AHGD Pipeline | Updated: Real-time from DuckDB")
