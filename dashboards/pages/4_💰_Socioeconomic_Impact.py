"""
Socioeconomic Impact Dashboard Page

Analysis of socioeconomic factors and their correlation with health outcomes,
including SEIFA indices, income, unemployment, education, and demographics.
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

from dashboards.config import DB_PATH, SEIFA_COLOR_SCALE, METRICS_CONFIG
from dashboards.utils.database import get_db_connection

st.set_page_config(
    page_title="Socioeconomic Impact | AHGD Dashboard",
    page_icon="💰",
    layout="wide"
)

# Header
st.title("💰 Socioeconomic Impact Analysis")
st.markdown("Explore the relationship between socioeconomic factors and health outcomes")

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

    # SEIFA decile filter
    seifa_deciles = list(range(1, 11))
    selected_deciles = st.multiselect(
        "SEIFA Deciles",
        options=seifa_deciles,
        default=seifa_deciles,
        help="Filter by SEIFA disadvantage decile (1=most disadvantaged)",
    )

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
    if selected_deciles:
        deciles_str = ", ".join(map(str, selected_deciles))
        filter_conditions.append(f"seifa_irsd_decile IN ({deciles_str})")

    where_clause = " AND ".join(filter_conditions) if filter_conditions else "1=1"

    # Get data with filters
    sql = f"""
        SELECT
            sa2_code,
            sa2_name,
            state_code,
            remoteness_category,
            total_population,
            population_density_per_sq_km,
            median_age,
            seifa_irsad_score,
            seifa_irsd_score,
            seifa_irsad_decile,
            seifa_irsd_decile,
            median_household_income,
            unemployment_rate,
            mortality_rate,
            utilisation_rate,
            bulk_billed_percentage
        FROM master_health_record
        WHERE {where_clause}
            AND seifa_irsad_score IS NOT NULL
            AND seifa_irsd_score IS NOT NULL
    """
    data = db.query(sql)

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Main content
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 SEIFA Analysis",
    "💵 Income & Health",
    "👥 Demographics",
    "🔍 Correlation Analysis"
])

# Tab 1: SEIFA Analysis
with tab1:
    st.subheader("SEIFA Index Analysis")

    # SEIFA summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_irsad = data["seifa_irsad_score"].mean()
        st.metric(
            "Avg IRSAD Score",
            f"{avg_irsad:.0f}",
            help="Index of Relative Socio-economic Advantage and Disadvantage (mean=1000)",
        )

    with col2:
        avg_irsd = data["seifa_irsd_score"].mean()
        st.metric(
            "Avg IRSD Score",
            f"{avg_irsd:.0f}",
            help="Index of Relative Socio-economic Disadvantage (mean=1000)",
        )

    with col3:
        most_disadvantaged = data.filter(pl.col("seifa_irsd_decile") <= 2).height
        st.metric(
            "Most Disadvantaged",
            f"{most_disadvantaged:,}",
            help="SA2 regions in deciles 1-2",
        )

    with col4:
        least_disadvantaged = data.filter(pl.col("seifa_irsd_decile") >= 9).height
        st.metric(
            "Least Disadvantaged",
            f"{least_disadvantaged:,}",
            help="SA2 regions in deciles 9-10",
        )

    st.markdown("---")

    # SEIFA distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### SEIFA Decile Distribution")

        # Count by decile
        decile_counts = (
            data.group_by("seifa_irsd_decile")
            .agg(pl.count("sa2_code").alias("count"))
            .sort("seifa_irsd_decile")
        )

        fig = px.bar(
            decile_counts.to_pandas(),
            x="seifa_irsd_decile",
            y="count",
            title="SA2 Regions by SEIFA IRSD Decile",
            labels={
                "seifa_irsd_decile": "SEIFA Decile (1=Most Disadvantaged)",
                "count": "Number of SA2 Regions",
            },
            color="seifa_irsd_decile",
            color_continuous_scale=SEIFA_COLOR_SCALE,
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### SEIFA Score Distribution")

        fig = px.histogram(
            data.to_pandas(),
            x="seifa_irsad_score",
            nbins=50,
            title="Distribution of SEIFA IRSAD Scores",
            labels={"seifa_irsad_score": "SEIFA IRSAD Score", "count": "Frequency"},
            color_discrete_sequence=["#5ab4ac"],
        )
        fig.add_vline(
            x=1000,
            line_dash="dash",
            line_color="red",
            annotation_text="National Mean (1000)",
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # SEIFA vs Health Outcomes
    st.markdown("### SEIFA vs Health Outcomes")

    col1, col2 = st.columns(2)

    with col1:
        # SEIFA vs Mortality
        fig = px.scatter(
            data.to_pandas(),
            x="seifa_irsd_score",
            y="mortality_rate",
            size="total_population",
            color="seifa_irsd_decile",
            hover_data=["sa2_name", "state_code"],
            title="SEIFA Score vs Mortality Rate",
            labels={
                "seifa_irsd_score": "SEIFA IRSD Score",
                "mortality_rate": "Mortality Rate (per 1,000)",
                "seifa_irsd_decile": "SEIFA Decile",
            },
            color_continuous_scale=SEIFA_COLOR_SCALE,
            trendline="ols",
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # SEIFA vs Healthcare Utilisation
        fig = px.scatter(
            data.to_pandas(),
            x="seifa_irsd_score",
            y="utilisation_rate",
            size="total_population",
            color="seifa_irsd_decile",
            hover_data=["sa2_name", "state_code"],
            title="SEIFA Score vs Healthcare Utilisation",
            labels={
                "seifa_irsd_score": "SEIFA IRSD Score",
                "utilisation_rate": "Healthcare Utilisation Rate (%)",
                "seifa_irsd_decile": "SEIFA Decile",
            },
            color_continuous_scale=SEIFA_COLOR_SCALE,
            trendline="ols",
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Income & Health
with tab2:
    st.subheader("Income and Health Outcomes")

    # Filter data with income
    income_data = data.filter(pl.col("median_household_income").is_not_null())

    # Income summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_income = income_data["median_household_income"].mean()
        st.metric(
            "Avg Median Income",
            f"${avg_income:,.0f}",
            help="Average median weekly household income",
        )

    with col2:
        min_income = income_data["median_household_income"].min()
        st.metric(
            "Lowest Income",
            f"${min_income:,.0f}",
            help="Lowest median household income",
        )

    with col3:
        max_income = income_data["median_household_income"].max()
        st.metric(
            "Highest Income",
            f"${max_income:,.0f}",
            help="Highest median household income",
        )

    with col4:
        # Income quintiles
        income_quintiles = income_data["median_household_income"].qcut(
            5, labels=False, allow_duplicates=True, maintain_order=True
        )
        st.metric(
            "Income Range",
            f"${max_income - min_income:,.0f}",
            help="Difference between highest and lowest",
        )

    st.markdown("---")

    # Income vs Health scatter plots
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Income vs Mortality Rate")

        fig = px.scatter(
            income_data.to_pandas(),
            x="median_household_income",
            y="mortality_rate",
            size="total_population",
            color="seifa_irsd_decile",
            hover_data=["sa2_name", "state_code"],
            title="Median Income vs Mortality Rate",
            labels={
                "median_household_income": "Median Weekly Household Income ($)",
                "mortality_rate": "Mortality Rate (per 1,000)",
                "seifa_irsd_decile": "SEIFA Decile",
            },
            color_continuous_scale=SEIFA_COLOR_SCALE,
            trendline="ols",
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Income vs Bulk Billing Rate")

        bulk_bill_data = income_data.filter(
            pl.col("bulk_billed_percentage").is_not_null()
        )

        fig = px.scatter(
            bulk_bill_data.to_pandas(),
            x="median_household_income",
            y="bulk_billed_percentage",
            size="total_population",
            color="seifa_irsd_decile",
            hover_data=["sa2_name", "state_code"],
            title="Median Income vs Bulk Billing Rate",
            labels={
                "median_household_income": "Median Weekly Household Income ($)",
                "bulk_billed_percentage": "Bulk Billing Rate (%)",
                "seifa_irsd_decile": "SEIFA Decile",
            },
            color_continuous_scale=SEIFA_COLOR_SCALE,
            trendline="ols",
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    # Unemployment analysis
    st.markdown("---")
    st.markdown("### Unemployment Impact on Health")

    unemployment_data = data.filter(pl.col("unemployment_rate").is_not_null())

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            unemployment_data.to_pandas(),
            x="unemployment_rate",
            y="mortality_rate",
            size="total_population",
            color="seifa_irsd_decile",
            hover_data=["sa2_name", "state_code"],
            title="Unemployment Rate vs Mortality Rate",
            labels={
                "unemployment_rate": "Unemployment Rate (%)",
                "mortality_rate": "Mortality Rate (per 1,000)",
                "seifa_irsd_decile": "SEIFA Decile",
            },
            color_continuous_scale=SEIFA_COLOR_SCALE,
            trendline="ols",
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Box plot by unemployment quartiles
        unemp_quartile = unemployment_data["unemployment_rate"].qcut(
            4, labels=["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"], allow_duplicates=True
        )
        plot_data = unemployment_data.to_pandas()
        plot_data["unemployment_quartile"] = unemp_quartile.to_pandas()

        fig = px.box(
            plot_data,
            x="unemployment_quartile",
            y="mortality_rate",
            title="Mortality Rate by Unemployment Quartile",
            labels={
                "unemployment_quartile": "Unemployment Quartile",
                "mortality_rate": "Mortality Rate (per 1,000)",
            },
            color="unemployment_quartile",
        )
        fig.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Demographics
with tab3:
    st.subheader("Population Demographics")

    # Demographics summary
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_pop = data["total_population"].sum()
        st.metric(
            "Total Population",
            f"{total_pop:,.0f}",
            help="Total population in filtered regions",
        )

    with col2:
        avg_age = data["median_age"].mean()
        st.metric(
            "Avg Median Age",
            f"{avg_age:.1f} years",
            help="Average median age across regions",
        )

    with col3:
        avg_density = data["population_density_per_sq_km"].mean()
        st.metric(
            "Avg Population Density",
            f"{avg_density:,.1f}/km²",
            help="Average population density",
        )

    with col4:
        num_regions = data.height
        st.metric(
            "SA2 Regions",
            f"{num_regions:,}",
            help="Number of SA2 regions in filtered data",
        )

    st.markdown("---")

    # Age analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Age Distribution")

        age_data = data.filter(pl.col("median_age").is_not_null())

        fig = px.histogram(
            age_data.to_pandas(),
            x="median_age",
            nbins=30,
            title="Distribution of Median Age Across SA2 Regions",
            labels={"median_age": "Median Age (years)", "count": "Number of Regions"},
            color_discrete_sequence=["#1f77b4"],
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Age vs Health Outcomes")

        fig = px.scatter(
            age_data.to_pandas(),
            x="median_age",
            y="mortality_rate",
            size="total_population",
            color="seifa_irsd_decile",
            hover_data=["sa2_name", "state_code"],
            title="Median Age vs Mortality Rate",
            labels={
                "median_age": "Median Age (years)",
                "mortality_rate": "Mortality Rate (per 1,000)",
                "seifa_irsd_decile": "SEIFA Decile",
            },
            color_continuous_scale=SEIFA_COLOR_SCALE,
            trendline="ols",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Population density analysis
    st.markdown("---")
    st.markdown("### Population Density and Health")

    col1, col2 = st.columns(2)

    with col1:
        # Log scale for density
        density_data = data.filter(pl.col("population_density_per_sq_km") > 0)

        fig = px.scatter(
            density_data.to_pandas(),
            x="population_density_per_sq_km",
            y="mortality_rate",
            size="total_population",
            color="remoteness_category",
            hover_data=["sa2_name", "state_code"],
            title="Population Density vs Mortality Rate",
            labels={
                "population_density_per_sq_km": "Population Density (per km²)",
                "mortality_rate": "Mortality Rate (per 1,000)",
                "remoteness_category": "Remoteness",
            },
            log_x=True,
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            density_data.to_pandas(),
            x="population_density_per_sq_km",
            y="utilisation_rate",
            size="total_population",
            color="remoteness_category",
            hover_data=["sa2_name", "state_code"],
            title="Population Density vs Healthcare Utilisation",
            labels={
                "population_density_per_sq_km": "Population Density (per km²)",
                "utilisation_rate": "Healthcare Utilisation Rate (%)",
                "remoteness_category": "Remoteness",
            },
            log_x=True,
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

# Tab 4: Correlation Analysis
with tab4:
    st.subheader("Correlation Analysis")

    st.markdown("""
    Explore correlations between socioeconomic indicators and health outcomes.
    Positive correlations indicate variables that increase together,
    while negative correlations indicate inverse relationships.
    """)

    # Calculate correlations
    try:
        # Get correlation data
        corr_data = data.filter(
            pl.col("seifa_irsad_score").is_not_null()
            & pl.col("median_household_income").is_not_null()
            & pl.col("unemployment_rate").is_not_null()
            & pl.col("mortality_rate").is_not_null()
            & pl.col("utilisation_rate").is_not_null()
        )

        # Select numeric columns for correlation
        numeric_cols = [
            "seifa_irsad_score",
            "seifa_irsd_score",
            "median_household_income",
            "unemployment_rate",
            "mortality_rate",
            "utilisation_rate",
            "bulk_billed_percentage",
            "median_age",
            "population_density_per_sq_km",
        ]

        # Calculate correlation matrix
        corr_df = corr_data.select(numeric_cols).to_pandas()
        correlation_matrix = corr_df.corr()

        # Create heatmap
        fig = px.imshow(
            correlation_matrix,
            labels=dict(color="Correlation"),
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation Matrix: Socioeconomic Indicators and Health Outcomes",
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Key correlations table
        st.markdown("### Key Correlations with Health Outcomes")

        # Extract correlations with mortality and utilisation
        mortality_corrs = correlation_matrix["mortality_rate"].sort_values(ascending=False)
        utilisation_corrs = correlation_matrix["utilisation_rate"].sort_values(
            ascending=False
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Correlations with Mortality Rate")
            mortality_df = mortality_corrs.to_frame(name="Correlation")
            mortality_df = mortality_df[mortality_df.index != "mortality_rate"]
            st.dataframe(
                mortality_df.style.background_gradient(
                    cmap="RdYlGn_r", vmin=-1, vmax=1
                ),
                use_container_width=True,
            )

        with col2:
            st.markdown("#### Correlations with Healthcare Utilisation")
            utilisation_df = utilisation_corrs.to_frame(name="Correlation")
            utilisation_df = utilisation_df[
                utilisation_df.index != "utilisation_rate"
            ]
            st.dataframe(
                utilisation_df.style.background_gradient(
                    cmap="RdYlGn", vmin=-1, vmax=1
                ),
                use_container_width=True,
            )

        # Scatter matrix for selected variables
        st.markdown("---")
        st.markdown("### Multi-Variable Relationships")

        selected_vars = st.multiselect(
            "Select variables for scatter matrix (max 5)",
            options=[
                "seifa_irsad_score",
                "median_household_income",
                "unemployment_rate",
                "mortality_rate",
                "utilisation_rate",
                "median_age",
            ],
            default=[
                "seifa_irsad_score",
                "median_household_income",
                "mortality_rate",
            ],
            max_selections=5,
        )

        if len(selected_vars) >= 2:
            fig = px.scatter_matrix(
                corr_df[selected_vars],
                dimensions=selected_vars,
                color=corr_df["seifa_irsd_decile"] if "seifa_irsd_decile" in corr_df.columns else None,
                title="Scatter Matrix of Selected Variables",
                height=800,
            )
            fig.update_traces(diagonal_visible=False)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error calculating correlations: {str(e)}")

# Footer
st.markdown("---")
st.caption("Data source: AHGD Pipeline | SEIFA data from ABS | Income and employment from Census")
