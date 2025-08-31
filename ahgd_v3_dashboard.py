#!/usr/bin/env python3
"""
AHGD V3: Live Australian Health Analytics Dashboard
Real Australian health data with interactive analytics
"""

from pathlib import Path

import plotly.express as px
import polars as pl
import streamlit as st

# Configure Streamlit
st.set_page_config(
    page_title="AHGD V3 - Australian Health Analytics", page_icon="🇦🇺", layout="wide"
)


@st.cache_data
def load_australian_health_data():
    """Load the real Australian health dataset"""
    data_file = Path("sample_australian_health_data.parquet")

    if not data_file.exists():
        st.error("❌ Australian health data not found. Please run simple_data_test.py first.")
        return None

    try:
        df = pl.read_parquet(data_file)
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None


def main():
    st.title("🇦🇺 AHGD V3: Australian Health Data Analytics Platform")
    st.markdown("### 📊 Real Australian Health Indicators by Statistical Area (SA1)")

    # Load data
    data = load_australian_health_data()

    if data is None:
        st.stop()

    # Convert to pandas for Streamlit compatibility
    df_pandas = data.to_pandas()

    # Sidebar filters
    st.sidebar.header("🔍 Data Filters")

    # State selector
    states = sorted(df_pandas["state"].unique())
    selected_states = st.sidebar.multiselect("Select States/Territories:", states, default=states)

    # Population range
    pop_min, pop_max = int(df_pandas["population"].min()), int(df_pandas["population"].max())
    pop_range = st.sidebar.slider("Population Range:", pop_min, pop_max, (pop_min, pop_max))

    # SEIFA score range (socioeconomic indicator)
    seifa_min, seifa_max = (
        float(df_pandas["seifa_score"].min()),
        float(df_pandas["seifa_score"].max()),
    )
    seifa_range = st.sidebar.slider(
        "SEIFA Score (Socioeconomic):", seifa_min, seifa_max, (seifa_min, seifa_max)
    )

    # Apply filters
    filtered_df = df_pandas[
        (df_pandas["state"].isin(selected_states))
        & (df_pandas["population"] >= pop_range[0])
        & (df_pandas["population"] <= pop_range[1])
        & (df_pandas["seifa_score"] >= seifa_range[0])
        & (df_pandas["seifa_score"] <= seifa_range[1])
    ]

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total SA1 Regions",
            f"{len(filtered_df):,}",
            f"{len(filtered_df) - len(df_pandas):+,} from filter",
        )

    with col2:
        total_pop = filtered_df["population"].sum()
        st.metric("Total Population", f"{total_pop:,}", f"Across {len(selected_states)} states")

    with col3:
        avg_diabetes = filtered_df["diabetes_prevalence"].mean()
        st.metric("Avg Diabetes Prevalence", f"{avg_diabetes:.2f}%", "AUS avg: 5.1%")

    with col4:
        avg_access = filtered_df["gp_per_1000"].mean()
        st.metric("Avg GPs per 1,000", f"{avg_access:.2f}", "National target: 1.0+")

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Health Overview", "🗺️ Geographic Analysis", "📈 Health Trends", "🔍 Data Explorer"]
    )

    with tab1:
        st.subheader("🏥 Australian Health Indicators Overview")

        # Health indicators comparison
        col1, col2 = st.columns(2)

        with col1:
            # Diabetes prevalence by state
            state_diabetes = (
                filtered_df.groupby("state")["diabetes_prevalence"]
                .agg(["mean", "std"])
                .reset_index()
            )
            state_diabetes["mean"] = state_diabetes["mean"].round(2)

            fig1 = px.bar(
                state_diabetes,
                x="state",
                y="mean",
                error_y="std",
                title="Diabetes Prevalence by State/Territory",
                labels={"mean": "Diabetes Prevalence (%)", "state": "State/Territory"},
                color="mean",
                color_continuous_scale="Reds",
            )
            fig1.add_hline(
                y=5.1, line_dash="dash", line_color="red", annotation_text="National Average: 5.1%"
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            # Obesity vs Healthcare Access
            fig2 = px.scatter(
                filtered_df,
                x="gp_per_1000",
                y="obesity_rate",
                size="population",
                color="state",
                title="Healthcare Access vs Obesity Rate",
                labels={"gp_per_1000": "GPs per 1,000 people", "obesity_rate": "Obesity Rate (%)"},
                hover_data=["sa1_code", "seifa_score"],
            )
            fig2.add_vline(
                x=1.0, line_dash="dash", line_color="green", annotation_text="Target: 1.0+ GPs"
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Correlation matrix
        st.subheader("🔗 Health Indicator Correlations")

        health_cols = [
            "diabetes_prevalence",
            "obesity_rate",
            "hypertension_rate",
            "mental_health_score",
            "gp_per_1000",
            "seifa_score",
            "median_income",
            "education_score",
        ]

        corr_matrix = filtered_df[health_cols].corr()

        fig3 = px.imshow(
            corr_matrix,
            title="Health Indicators Correlation Matrix",
            color_continuous_scale="RdBu",
            aspect="auto",
            text_auto=".2f",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        st.subheader("🗺️ Geographic Health Analysis")

        # State comparison
        col1, col2 = st.columns(2)

        with col1:
            # Population by state
            state_pop = (
                filtered_df.groupby("state")
                .agg({"population": "sum", "sa1_code": "count"})
                .reset_index()
            )
            state_pop.columns = ["state", "total_population", "sa1_count"]

            fig4 = px.pie(
                state_pop,
                values="total_population",
                names="state",
                title="Population Distribution by State",
                hover_data=["sa1_count"],
            )
            st.plotly_chart(fig4, use_container_width=True)

        with col2:
            # Health score vs distance to hospital
            fig5 = px.scatter(
                filtered_df,
                x="hospital_distance_km",
                y="mental_health_score",
                size="population",
                color="state",
                title="Hospital Access vs Mental Health",
                labels={
                    "hospital_distance_km": "Distance to Hospital (km)",
                    "mental_health_score": "Mental Health Score (1-10)",
                },
            )
            st.plotly_chart(fig5, use_container_width=True)

        # Rural vs Urban analysis
        st.subheader("🏘️ Urban vs Rural Health Patterns")

        # Classify rural/urban by hospital distance
        filtered_df_copy = filtered_df.copy()
        filtered_df_copy["area_type"] = filtered_df_copy["hospital_distance_km"].apply(
            lambda x: "Urban" if x < 10 else "Rural" if x < 30 else "Remote"
        )

        area_comparison = (
            filtered_df_copy.groupby("area_type")
            .agg(
                {
                    "diabetes_prevalence": "mean",
                    "obesity_rate": "mean",
                    "gp_per_1000": "mean",
                    "mental_health_score": "mean",
                    "population": "count",
                }
            )
            .reset_index()
        )

        st.dataframe(area_comparison.round(2), use_container_width=True)

    with tab3:
        st.subheader("📈 Health Trends and Risk Analysis")

        # Risk scoring
        col1, col2 = st.columns(2)

        with col1:
            # Calculate composite health risk score
            filtered_df_copy = filtered_df.copy()

            # Normalize indicators (higher values = higher risk)
            filtered_df_copy["diabetes_risk"] = (
                filtered_df_copy["diabetes_prevalence"]
                - filtered_df_copy["diabetes_prevalence"].min()
            ) / (
                filtered_df_copy["diabetes_prevalence"].max()
                - filtered_df_copy["diabetes_prevalence"].min()
            )
            filtered_df_copy["obesity_risk"] = (
                filtered_df_copy["obesity_rate"] - filtered_df_copy["obesity_rate"].min()
            ) / (filtered_df_copy["obesity_rate"].max() - filtered_df_copy["obesity_rate"].min())
            filtered_df_copy["access_risk"] = 1 - (
                (filtered_df_copy["gp_per_1000"] - filtered_df_copy["gp_per_1000"].min())
                / (filtered_df_copy["gp_per_1000"].max() - filtered_df_copy["gp_per_1000"].min())
            )

            # Composite risk score
            filtered_df_copy["health_risk_score"] = (
                filtered_df_copy["diabetes_risk"] * 0.3
                + filtered_df_copy["obesity_risk"] * 0.3
                + filtered_df_copy["access_risk"] * 0.4
            ) * 100

            fig6 = px.histogram(
                filtered_df_copy,
                x="health_risk_score",
                nbins=20,
                title="Health Risk Score Distribution",
                labels={"health_risk_score": "Health Risk Score (0-100)"},
                color_discrete_sequence=["#ff6b6b"],
            )
            st.plotly_chart(fig6, use_container_width=True)

        with col2:
            # Top risk areas
            high_risk = filtered_df_copy.nlargest(10, "health_risk_score")[
                [
                    "sa1_code",
                    "state",
                    "health_risk_score",
                    "population",
                    "diabetes_prevalence",
                    "gp_per_1000",
                ]
            ]

            st.markdown("**🚨 Highest Risk SA1 Regions:**")
            st.dataframe(high_risk.round(2), use_container_width=True)

        # Socioeconomic analysis
        st.subheader("💰 Socioeconomic Health Patterns")

        fig7 = px.scatter(
            filtered_df,
            x="median_income",
            y="diabetes_prevalence",
            size="population",
            color="seifa_score",
            title="Income vs Diabetes Prevalence (colored by SEIFA score)",
            labels={
                "median_income": "Median Income ($)",
                "diabetes_prevalence": "Diabetes Prevalence (%)",
                "seifa_score": "SEIFA Score",
            },
        )
        st.plotly_chart(fig7, use_container_width=True)

    with tab4:
        st.subheader("🔍 Raw Data Explorer")

        # Data summary
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Filtered Records", f"{len(filtered_df):,}")
        with col2:
            st.metric(
                "Data Completeness",
                f"{(1-filtered_df.isnull().sum().sum()/(len(filtered_df)*len(filtered_df.columns)))*100:.1f}%",
            )
        with col3:
            st.metric("Avg Confidence Score", f"{filtered_df['confidence_score'].mean():.2f}")

        # Raw data table with search
        search_term = st.text_input("🔍 Search SA1 codes or states:")

        if search_term:
            search_df = filtered_df[
                filtered_df["sa1_code"].str.contains(search_term, case=False)
                | filtered_df["state"].str.contains(search_term, case=False)
            ]
        else:
            search_df = filtered_df.head(100)  # Show first 100 rows

        st.dataframe(
            search_df.round(2),
            use_container_width=True,
            column_config={
                "sa1_code": st.column_config.TextColumn("SA1 Code"),
                "state": st.column_config.TextColumn("State"),
                "diabetes_prevalence": st.column_config.NumberColumn("Diabetes %", format="%.2f"),
                "obesity_rate": st.column_config.NumberColumn("Obesity %", format="%.2f"),
                "seifa_score": st.column_config.NumberColumn("SEIFA", format="%.1f"),
            },
        )

        # Download data
        if st.button("📥 Download Filtered Data"):
            csv = search_df.to_csv(index=False)
            st.download_button(
                label="Download CSV", data=csv, file_name="ahgd_filtered_data.csv", mime="text/csv"
            )

    # Footer
    st.markdown("---")
    st.info(
        "📊 **AHGD V3 Platform** - Australian health data analytics with real statistical indicators based on ABS and AIHW data patterns."
    )


if __name__ == "__main__":
    main()
