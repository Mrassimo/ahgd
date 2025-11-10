"""
Health Indicators Deep Dive Page

Comprehensive analysis of health metrics including mortality rates,
Medicare utilisation, correlations, and risk factors.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboards.config import DB_PATH, HEALTH_COLOR_SCALE, METRICS_CONFIG, STATE_CODES
from dashboards.utils.database import get_db_connection

st.set_page_config(
    page_title="Health Indicators | AHGD Dashboard",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .risk-medium {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
    }
    .risk-low {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.title("🏥 Health Indicators Deep Dive")
st.markdown("Comprehensive analysis of mortality, utilisation, and health outcomes")

# Get database connection
try:
    db = get_db_connection(str(DB_PATH))
except Exception as e:
    st.error(f"Database connection error: {str(e)}")
    st.stop()

# Sidebar Filters
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
        "Remoteness Categories",
        options=remoteness_cats,
        default=remoteness_cats,
        help="Filter by remoteness category",
    )

    # Population threshold
    pop_threshold = st.slider(
        "Minimum Population",
        min_value=0,
        max_value=50000,
        value=0,
        step=1000,
        help="Filter regions by minimum population",
    )

    st.markdown("---")

    # Metric selection for analysis
    st.subheader("Analysis Focus")

    health_metric = st.selectbox(
        "Primary Health Metric",
        options=["mortality_rate", "utilisation_rate", "bulk_billed_percentage"],
        format_func=lambda x: {
            "mortality_rate": "Mortality Rate",
            "utilisation_rate": "Utilisation Rate",
            "bulk_billed_percentage": "Bulk Billing %",
        }[x],
    )

    # Statistical threshold for risk identification
    risk_threshold = st.slider(
        "Risk Threshold (Percentile)",
        min_value=75,
        max_value=95,
        value=90,
        step=5,
        help="Define high-risk as top N percentile",
    )

    st.markdown("---")

    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

# Load and filter data
@st.cache_data(ttl=300)
def load_filtered_data(states, remoteness, pop_threshold):
    """Load data with filters applied."""
    data = db.get_master_health_record()

    # Apply filters
    if states:
        data = data.filter(pl.col("state_code").is_in(states))
    if remoteness:
        data = data.filter(pl.col("remoteness_category").is_in(remoteness))
    if pop_threshold > 0:
        data = data.filter(pl.col("total_population") >= pop_threshold)

    return data

try:
    df = load_filtered_data(selected_states, selected_remoteness, pop_threshold)

    if len(df) == 0:
        st.warning("No data available with current filters. Please adjust filter settings.")
        st.stop()

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Mortality Analysis",
    "🏥 Medicare Utilisation",
    "🔗 Correlations",
    "⚠️ Risk Factors",
    "📈 Detailed Statistics"
])

# TAB 1: Mortality Analysis
with tab1:
    st.subheader("Mortality Rate Analysis")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_mortality = df["mortality_rate"].mean()
        st.metric(
            "Average Mortality Rate",
            f"{avg_mortality:.2f}",
            help="Deaths per 1,000 population",
        )

    with col2:
        median_mortality = df["mortality_rate"].median()
        st.metric(
            "Median Mortality Rate",
            f"{median_mortality:.2f}",
            help="Middle value of mortality distribution",
        )

    with col3:
        std_mortality = df["mortality_rate"].std()
        st.metric(
            "Std Deviation",
            f"{std_mortality:.2f}",
            help="Variation in mortality rates",
        )

    with col4:
        mortality_range = df["mortality_rate"].max() - df["mortality_rate"].min()
        st.metric(
            "Range",
            f"{mortality_range:.2f}",
            help="Difference between highest and lowest",
        )

    st.markdown("---")

    # Distribution visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Histogram
        fig = px.histogram(
            df.to_pandas(),
            x="mortality_rate",
            nbins=50,
            title="Mortality Rate Distribution",
            labels={"mortality_rate": "Mortality Rate (per 1,000)", "count": "Number of SA2 Regions"},
            color_discrete_sequence=["#e74c3c"],
            marginal="box",
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Box plot by remoteness
        fig = px.box(
            df.to_pandas(),
            x="remoteness_category",
            y="mortality_rate",
            title="Mortality Rate by Remoteness",
            labels={
                "remoteness_category": "Remoteness Category",
                "mortality_rate": "Mortality Rate (per 1,000)",
            },
            color="remoteness_category",
            color_discrete_sequence=px.colors.sequential.Reds,
        )
        fig.update_layout(height=400, showlegend=False)
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    # State comparison
    st.markdown("### Mortality Breakdown by State")

    # Calculate state statistics
    state_stats = (
        df.group_by("state_code")
        .agg([
            pl.col("mortality_rate").mean().alias("avg_mortality"),
            pl.col("mortality_rate").median().alias("median_mortality"),
            pl.col("mortality_rate").std().alias("std_mortality"),
            pl.count().alias("region_count"),
            pl.col("total_population").sum().alias("total_pop"),
        ])
        .sort("avg_mortality", descending=True)
    )

    # Create bar chart
    fig = px.bar(
        state_stats.to_pandas(),
        x="state_code",
        y="avg_mortality",
        title="Average Mortality Rate by State",
        labels={
            "state_code": "State",
            "avg_mortality": "Average Mortality Rate",
        },
        color="avg_mortality",
        color_continuous_scale="Reds",
        text="avg_mortality",
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Detailed state table
    with st.expander("📋 Detailed State Statistics"):
        display_df = state_stats.to_pandas()
        display_df.columns = [
            "State",
            "Avg Mortality",
            "Median Mortality",
            "Std Dev",
            "Region Count",
            "Total Population",
        ]
        st.dataframe(display_df, use_container_width=True)

    # Mortality vs Population scatter
    st.markdown("### Mortality Rate vs Population Size")

    # Sample data if too large
    plot_df = df.sample(n=min(2000, len(df)), seed=42)

    fig = px.scatter(
        plot_df.to_pandas(),
        x="total_population",
        y="mortality_rate",
        color="remoteness_category",
        size="total_population",
        hover_data=["sa2_code", "state_code"],
        title="Relationship between Population and Mortality",
        labels={
            "total_population": "Total Population",
            "mortality_rate": "Mortality Rate (per 1,000)",
            "remoteness_category": "Remoteness",
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# TAB 2: Medicare Utilisation
with tab2:
    st.subheader("Medicare Utilisation Analysis")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_util = df["utilisation_rate"].mean()
        st.metric(
            "Average Utilisation Rate",
            f"{avg_util:.1f}%",
            help="Percentage of population using Medicare services",
        )

    with col2:
        avg_bulk_bill = df["bulk_billed_percentage"].mean()
        st.metric(
            "Average Bulk Billing",
            f"{avg_bulk_bill:.1f}%",
            help="Percentage of services bulk billed",
        )

    with col3:
        # Calculate access disparity (std dev of utilisation)
        util_disparity = df["utilisation_rate"].std()
        st.metric(
            "Access Disparity",
            f"{util_disparity:.1f}%",
            help="Variation in utilisation rates (lower is better)",
        )

    with col4:
        # Regions with low utilisation (below 70%)
        low_util_count = len(df.filter(pl.col("utilisation_rate") < 70))
        st.metric(
            "Low Utilisation Regions",
            f"{low_util_count}",
            help="Regions with <70% utilisation rate",
        )

    st.markdown("---")

    # Utilisation distribution
    col1, col2 = st.columns(2)

    with col1:
        # Histogram
        fig = px.histogram(
            df.to_pandas(),
            x="utilisation_rate",
            nbins=50,
            title="Healthcare Utilisation Rate Distribution",
            labels={
                "utilisation_rate": "Utilisation Rate (%)",
                "count": "Number of SA2 Regions"
            },
            color_discrete_sequence=["#3498db"],
            marginal="box",
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Bulk billing distribution
        fig = px.histogram(
            df.to_pandas(),
            x="bulk_billed_percentage",
            nbins=50,
            title="Bulk Billing Percentage Distribution",
            labels={
                "bulk_billed_percentage": "Bulk Billing Rate (%)",
                "count": "Number of SA2 Regions"
            },
            color_discrete_sequence=["#2ecc71"],
            marginal="box",
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Utilisation vs Bulk Billing
    st.markdown("### Utilisation Rate vs Bulk Billing Rate")

    plot_df = df.sample(n=min(2000, len(df)), seed=42)

    fig = px.scatter(
        plot_df.to_pandas(),
        x="bulk_billed_percentage",
        y="utilisation_rate",
        color="remoteness_category",
        size="total_population",
        hover_data=["sa2_code", "state_code"],
        title="Healthcare Access: Utilisation vs Bulk Billing",
        labels={
            "bulk_billed_percentage": "Bulk Billing Rate (%)",
            "utilisation_rate": "Utilisation Rate (%)",
            "remoteness_category": "Remoteness",
        },
        color_discrete_sequence=px.colors.qualitative.Pastel,
        trendline="ols",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Service usage by remoteness
    st.markdown("### Medicare Access by Remoteness")

    remoteness_stats = (
        df.group_by("remoteness_category")
        .agg([
            pl.col("utilisation_rate").mean().alias("avg_utilisation"),
            pl.col("bulk_billed_percentage").mean().alias("avg_bulk_billing"),
            pl.count().alias("region_count"),
        ])
        .sort("avg_utilisation", descending=True)
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Utilisation Rate',
        x=remoteness_stats["remoteness_category"].to_list(),
        y=remoteness_stats["avg_utilisation"].to_list(),
        marker_color='#3498db',
    ))

    fig.add_trace(go.Bar(
        name='Bulk Billing Rate',
        x=remoteness_stats["remoteness_category"].to_list(),
        y=remoteness_stats["avg_bulk_billing"].to_list(),
        marker_color='#2ecc71',
    ))

    fig.update_layout(
        title="Average Utilisation and Bulk Billing by Remoteness",
        xaxis_title="Remoteness Category",
        yaxis_title="Percentage (%)",
        barmode='group',
        height=400,
        xaxis_tickangle=45,
    )

    st.plotly_chart(fig, use_container_width=True)

# TAB 3: Correlations
with tab3:
    st.subheader("Health vs Socioeconomic Correlations")

    st.markdown("""
    Explore relationships between health indicators and socioeconomic factors.
    Stronger correlations (closer to -1 or 1) indicate stronger relationships.
    """)

    # Calculate correlation matrix
    correlation_columns = [
        "mortality_rate",
        "utilisation_rate",
        "bulk_billed_percentage",
        "median_household_income",
        "unemployment_rate",
        "seifa_irsad_score",
        "seifa_irsd_decile",
    ]

    # Filter to only include columns that exist
    available_columns = [col for col in correlation_columns if col in df.columns]

    # Calculate correlation matrix
    corr_df = df.select(available_columns).to_pandas()
    correlation_matrix = corr_df.corr()

    # Create heatmap
    fig = px.imshow(
        correlation_matrix,
        title="Correlation Matrix: Health and Socioeconomic Indicators",
        labels=dict(x="Indicator", y="Indicator", color="Correlation"),
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        text_auto=".2f",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Key correlation insights
    st.markdown("### Key Correlation Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Mortality Rate Correlations")

        # Get correlations with mortality
        mortality_corrs = correlation_matrix["mortality_rate"].sort_values(ascending=False)
        mortality_corrs = mortality_corrs[mortality_corrs.index != "mortality_rate"]

        fig = px.bar(
            x=mortality_corrs.values,
            y=mortality_corrs.index,
            orientation='h',
            title="Correlations with Mortality Rate",
            labels={"x": "Correlation Coefficient", "y": "Indicator"},
            color=mortality_corrs.values,
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Utilisation Rate Correlations")

        # Get correlations with utilisation
        util_corrs = correlation_matrix["utilisation_rate"].sort_values(ascending=False)
        util_corrs = util_corrs[util_corrs.index != "utilisation_rate"]

        fig = px.bar(
            x=util_corrs.values,
            y=util_corrs.index,
            orientation='h',
            title="Correlations with Utilisation Rate",
            labels={"x": "Correlation Coefficient", "y": "Indicator"},
            color=util_corrs.values,
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Scatter plots for strongest correlations
    st.markdown("### Detailed Correlation Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Mortality vs Income
        plot_df = df.sample(n=min(2000, len(df)), seed=42)

        fig = px.scatter(
            plot_df.to_pandas(),
            x="median_household_income",
            y="mortality_rate",
            color="seifa_irsd_decile",
            size="total_population",
            hover_data=["sa2_code"],
            title="Mortality Rate vs Median Household Income",
            labels={
                "median_household_income": "Median Household Income ($)",
                "mortality_rate": "Mortality Rate (per 1,000)",
                "seifa_irsd_decile": "SEIFA Decile",
            },
            color_continuous_scale="RdYlGn",
            trendline="ols",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Mortality vs Unemployment
        fig = px.scatter(
            plot_df.to_pandas(),
            x="unemployment_rate",
            y="mortality_rate",
            color="seifa_irsd_decile",
            size="total_population",
            hover_data=["sa2_code"],
            title="Mortality Rate vs Unemployment Rate",
            labels={
                "unemployment_rate": "Unemployment Rate (%)",
                "mortality_rate": "Mortality Rate (per 1,000)",
                "seifa_irsd_decile": "SEIFA Decile",
            },
            color_continuous_scale="RdYlGn",
            trendline="ols",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# TAB 4: Risk Factors
with tab4:
    st.subheader("Risk Factor Identification")

    st.markdown(f"""
    Identifying high-risk regions based on the **{risk_threshold}th percentile** threshold.
    Regions in the top {100 - risk_threshold}% are flagged as high-risk.
    """)

    # Calculate risk thresholds
    mortality_threshold = df["mortality_rate"].quantile(risk_threshold / 100)
    low_util_threshold = df["utilisation_rate"].quantile((100 - risk_threshold) / 100)
    low_bulk_bill_threshold = df["bulk_billed_percentage"].quantile((100 - risk_threshold) / 100)

    # Identify high-risk regions
    high_risk_mortality = df.filter(pl.col("mortality_rate") >= mortality_threshold)
    low_access_regions = df.filter(
        (pl.col("utilisation_rate") <= low_util_threshold) |
        (pl.col("bulk_billed_percentage") <= low_bulk_bill_threshold)
    )

    # Combined risk score
    df_risk = df.with_columns([
        ((pl.col("mortality_rate") - df["mortality_rate"].mean()) / df["mortality_rate"].std()).alias("mortality_zscore"),
        ((df["utilisation_rate"].mean() - pl.col("utilisation_rate")) / df["utilisation_rate"].std()).alias("access_zscore"),
    ])

    df_risk = df_risk.with_columns([
        (pl.col("mortality_zscore") + pl.col("access_zscore")).alias("combined_risk_score")
    ])

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "High Mortality Regions",
            f"{len(high_risk_mortality)}",
            help=f"Regions with mortality ≥ {mortality_threshold:.2f}",
        )

    with col2:
        st.metric(
            "Low Access Regions",
            f"{len(low_access_regions)}",
            help="Regions with low utilisation or bulk billing",
        )

    with col3:
        high_risk_pop = high_risk_mortality["total_population"].sum()
        st.metric(
            "Population at Risk",
            f"{high_risk_pop:,.0f}",
            help="Population in high-mortality regions",
        )

    with col4:
        risk_percentage = (len(high_risk_mortality) / len(df)) * 100
        st.metric(
            "% Regions at Risk",
            f"{risk_percentage:.1f}%",
            help="Percentage of regions with high mortality",
        )

    st.markdown("---")

    # Top risk regions
    st.markdown("### Highest Risk Regions")

    top_risk_regions = (
        df_risk.sort("combined_risk_score", descending=True)
        .head(20)
        .select([
            "sa2_code",
            "state_code",
            "remoteness_category",
            "mortality_rate",
            "utilisation_rate",
            "bulk_billed_percentage",
            "combined_risk_score",
        ])
    )

    # Display as styled dataframe
    st.dataframe(
        top_risk_regions.to_pandas().style.background_gradient(
            subset=["combined_risk_score"],
            cmap="Reds"
        ),
        use_container_width=True,
        height=400,
    )

    # Geographic distribution of risk
    st.markdown("### Risk Distribution by State and Remoteness")

    col1, col2 = st.columns(2)

    with col1:
        # Risk by state
        state_risk = (
            df_risk.group_by("state_code")
            .agg([
                pl.col("combined_risk_score").mean().alias("avg_risk_score"),
                (pl.col("mortality_rate") >= mortality_threshold).sum().alias("high_risk_count"),
                pl.count().alias("total_regions"),
            ])
            .with_columns([
                (pl.col("high_risk_count") / pl.col("total_regions") * 100).alias("risk_percentage")
            ])
            .sort("avg_risk_score", descending=True)
        )

        fig = px.bar(
            state_risk.to_pandas(),
            x="state_code",
            y="risk_percentage",
            title="Percentage of High-Risk Regions by State",
            labels={
                "state_code": "State",
                "risk_percentage": "% High-Risk Regions",
            },
            color="risk_percentage",
            color_continuous_scale="Reds",
            text="risk_percentage",
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk by remoteness
        remoteness_risk = (
            df_risk.group_by("remoteness_category")
            .agg([
                pl.col("combined_risk_score").mean().alias("avg_risk_score"),
                (pl.col("mortality_rate") >= mortality_threshold).sum().alias("high_risk_count"),
                pl.count().alias("total_regions"),
            ])
            .with_columns([
                (pl.col("high_risk_count") / pl.col("total_regions") * 100).alias("risk_percentage")
            ])
            .sort("avg_risk_score", descending=True)
        )

        fig = px.bar(
            remoteness_risk.to_pandas(),
            x="remoteness_category",
            y="avg_risk_score",
            title="Average Risk Score by Remoteness",
            labels={
                "remoteness_category": "Remoteness Category",
                "avg_risk_score": "Average Risk Score",
            },
            color="avg_risk_score",
            color_continuous_scale="Reds",
            text="avg_risk_score",
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    # Risk score distribution
    st.markdown("### Combined Risk Score Distribution")

    fig = px.histogram(
        df_risk.to_pandas(),
        x="combined_risk_score",
        nbins=50,
        title="Distribution of Combined Risk Scores",
        labels={
            "combined_risk_score": "Combined Risk Score (Z-score)",
            "count": "Number of SA2 Regions"
        },
        color_discrete_sequence=["#e74c3c"],
        marginal="box",
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# TAB 5: Detailed Statistics
with tab5:
    st.subheader("Detailed Statistical Analysis")

    # Statistical comparison tool
    st.markdown("### Compare Regions")

    col1, col2 = st.columns(2)

    with col1:
        comparison_metric = st.selectbox(
            "Select Metric for Comparison",
            options=[
                "mortality_rate",
                "utilisation_rate",
                "bulk_billed_percentage",
                "median_household_income",
                "unemployment_rate",
            ],
            format_func=lambda x: {
                "mortality_rate": "Mortality Rate",
                "utilisation_rate": "Utilisation Rate",
                "bulk_billed_percentage": "Bulk Billing %",
                "median_household_income": "Median Income",
                "unemployment_rate": "Unemployment Rate",
            }.get(x, x),
        )

    with col2:
        comparison_group = st.selectbox(
            "Group By",
            options=["state_code", "remoteness_category"],
            format_func=lambda x: {
                "state_code": "State",
                "remoteness_category": "Remoteness",
            }.get(x, x),
        )

    # Calculate comparison statistics
    comparison_stats = (
        df.group_by(comparison_group)
        .agg([
            pl.col(comparison_metric).mean().alias("mean"),
            pl.col(comparison_metric).median().alias("median"),
            pl.col(comparison_metric).std().alias("std_dev"),
            pl.col(comparison_metric).min().alias("min"),
            pl.col(comparison_metric).max().alias("max"),
            pl.col(comparison_metric).quantile(0.25).alias("q1"),
            pl.col(comparison_metric).quantile(0.75).alias("q3"),
            pl.count().alias("count"),
        ])
        .sort("mean", descending=True)
    )

    # Display comparison table
    st.dataframe(
        comparison_stats.to_pandas().style.format({
            "mean": "{:.2f}",
            "median": "{:.2f}",
            "std_dev": "{:.2f}",
            "min": "{:.2f}",
            "max": "{:.2f}",
            "q1": "{:.2f}",
            "q3": "{:.2f}",
        }),
        use_container_width=True,
    )

    # Violin plot for distribution comparison
    st.markdown(f"### {comparison_metric.replace('_', ' ').title()} Distribution Comparison")

    fig = px.violin(
        df.to_pandas(),
        x=comparison_group,
        y=comparison_metric,
        box=True,
        points="outliers",
        title=f"Distribution of {comparison_metric.replace('_', ' ').title()} by {comparison_group.replace('_', ' ').title()}",
        color=comparison_group,
    )
    fig.update_layout(height=500, showlegend=False)
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Detailed data table with filters
    st.markdown("### Detailed Data Explorer")

    st.markdown("""
    Explore and filter the complete dataset. You can sort by clicking column headers.
    """)

    # Select columns to display
    display_columns = st.multiselect(
        "Select Columns to Display",
        options=[
            "sa2_code",
            "state_code",
            "remoteness_category",
            "mortality_rate",
            "utilisation_rate",
            "bulk_billed_percentage",
            "median_household_income",
            "unemployment_rate",
            "seifa_irsad_score",
            "total_population",
        ],
        default=[
            "sa2_code",
            "state_code",
            "remoteness_category",
            "mortality_rate",
            "utilisation_rate",
            "total_population",
        ],
    )

    # Number of rows to display
    num_rows = st.slider(
        "Number of rows to display",
        min_value=10,
        max_value=500,
        value=50,
        step=10,
    )

    # Sort options
    sort_col = st.selectbox(
        "Sort by",
        options=display_columns,
        index=3 if "mortality_rate" in display_columns else 0,
    )

    sort_order = st.radio(
        "Sort order",
        options=["Descending", "Ascending"],
        horizontal=True,
    )

    # Apply sorting and display
    display_df = df.select(display_columns).sort(
        sort_col,
        descending=(sort_order == "Descending")
    ).head(num_rows)

    st.dataframe(display_df.to_pandas(), use_container_width=True, height=400)

    # Export options
    st.markdown("### Export Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Export Filtered Data (CSV)"):
            csv_data = display_df.to_pandas().to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="ahgd_health_indicators.csv",
                mime="text/csv",
            )

    with col2:
        st.info("Export to Excel and Parquet coming soon")

    # Summary statistics for current view
    st.markdown("### Summary Statistics for Current Selection")

    summary_cols = [col for col in display_columns if col in [
        "mortality_rate",
        "utilisation_rate",
        "bulk_billed_percentage",
        "median_household_income",
        "unemployment_rate",
        "total_population",
    ]]

    if summary_cols:
        summary_df = df.select(summary_cols).describe()
        st.dataframe(summary_df.to_pandas(), use_container_width=True)

# Footer
st.markdown("---")
st.caption(f"""
**Data Source:** AHGD Pipeline | **Regions Displayed:** {len(df):,} SA2 areas |
**Last Updated:** Real-time from DuckDB
""")
