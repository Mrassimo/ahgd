"""
Data Quality & Pipeline Monitoring Page

Monitors pipeline status, data freshness, completeness, and quality metrics.
Displays validation results and provides data quality reports.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from pathlib import Path
import sys
from datetime import datetime, timedelta
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboards.config import DB_PATH, CACHE_TTL, DATA_FRESH_HOURS, DATA_STALE_HOURS
from dashboards.utils.database import get_db_connection
from dashboards.components.charts import create_bar_chart, create_gauge_chart, create_heatmap
from dashboards.components.filters import create_checkbox_filter, create_top_n_filter
from dashboards.components.export import create_export_section

st.set_page_config(
    page_title="Data Quality | AHGD Dashboard",
    page_icon="✅",
    layout="wide",
)

# Header
st.title("✅ Data Quality & Pipeline Monitoring")
st.markdown("Monitor pipeline status, data freshness, and quality metrics")

# Get database connection
try:
    db = get_db_connection(str(DB_PATH))
except Exception as e:
    st.error(f"Database connection error: {str(e)}")
    st.stop()

# Sidebar controls
with st.sidebar:
    st.header("Quality Checks")

    show_details = create_checkbox_filter(
        label="Show Detailed Metrics",
        default=True,
        key="show_details",
        help_text="Display detailed quality metrics for each table",
    )

    show_missing = create_checkbox_filter(
        label="Show Missing Data Analysis",
        default=True,
        key="show_missing",
        help_text="Analyse missing data patterns",
    )

    top_n = create_top_n_filter(
        min_n=5,
        max_n=30,
        default_n=10,
        key="top_n_issues",
        help_text="Number of top issues to display",
    )

    st.markdown("---")

    if st.button("🔄 Refresh Metrics"):
        st.cache_data.clear()
        st.rerun()

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Pipeline Status",
    "🕒 Data Freshness",
    "📈 Data Coverage",
    "❌ Missing Data",
    "📥 Quality Reports",
])

# Tab 1: Pipeline Status
with tab1:
    st.subheader("Pipeline Status Overview")

    # Check database file info
    try:
        db_file = Path(DB_PATH)
        if db_file.exists():
            db_stats = db_file.stat()
            last_modified = datetime.fromtimestamp(db_stats.st_mtime)
            db_size_mb = db_stats.st_size / (1024 * 1024)
            time_since_update = datetime.now() - last_modified

            # Status indicators
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                # Database status
                status_color = "🟢" if time_since_update < timedelta(hours=DATA_FRESH_HOURS) else "🟡"
                st.metric(
                    "Database Status",
                    f"{status_color} Connected",
                    help="Database connection status",
                )

            with col2:
                # Last update
                st.metric(
                    "Last Updated",
                    last_modified.strftime("%Y-%m-%d %H:%M"),
                    f"{int(time_since_update.total_seconds() / 3600)}h ago",
                    help="Last database modification time",
                )

            with col3:
                # Database size
                st.metric(
                    "Database Size",
                    f"{db_size_mb:.2f} MB",
                    help="Total database file size",
                )

            with col4:
                # Data freshness
                if time_since_update < timedelta(hours=DATA_FRESH_HOURS):
                    freshness = "Fresh"
                    delta_color = "normal"
                elif time_since_update < timedelta(hours=DATA_STALE_HOURS):
                    freshness = "Aging"
                    delta_color = "off"
                else:
                    freshness = "Stale"
                    delta_color = "inverse"

                st.metric(
                    "Data Freshness",
                    freshness,
                    help="Data freshness indicator based on last update",
                )

        else:
            st.error("❌ Database file not found")

    except Exception as e:
        st.error(f"Error checking database status: {str(e)}")

    st.markdown("---")

    # Table information
    st.subheader("Table Statistics")

    try:
        conn = db.get_connection()

        # Get all tables
        tables_query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """
        tables = conn.execute(tables_query).fetchall()

        if tables:
            table_stats = []

            for table in tables:
                table_name = table[0]

                # Get row count
                row_count = conn.execute(
                    f"SELECT COUNT(*) FROM {table_name}"
                ).fetchone()[0]

                # Get column count
                col_count = conn.execute(
                    f"SELECT COUNT(*) FROM information_schema.columns WHERE table_name = '{table_name}'"
                ).fetchone()[0]

                # Estimate size (rough estimate)
                try:
                    size_estimate = conn.execute(
                        f"SELECT SUM(LENGTH(CAST(* AS VARCHAR))) FROM (SELECT * FROM {table_name} LIMIT 1000)"
                    ).fetchone()[0]
                    avg_row_size = size_estimate / 1000 if size_estimate else 0
                    total_size_mb = (avg_row_size * row_count) / (1024 * 1024)
                except:
                    total_size_mb = 0

                table_stats.append({
                    "Table": table_name,
                    "Rows": row_count,
                    "Columns": col_count,
                    "Est. Size (MB)": round(total_size_mb, 2),
                })

            stats_df = pl.DataFrame(table_stats)

            # Display as metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Total Tables",
                    len(table_stats),
                    help="Number of tables in database",
                )

            with col2:
                total_rows = stats_df["Rows"].sum()
                st.metric(
                    "Total Records",
                    f"{total_rows:,}",
                    help="Total number of records across all tables",
                )

            with col3:
                total_size = stats_df["Est. Size (MB)"].sum()
                st.metric(
                    "Est. Data Size",
                    f"{total_size:.2f} MB",
                    help="Estimated total data size",
                )

            # Table details
            st.markdown("#### Table Details")
            st.dataframe(
                stats_df.to_pandas(),
                use_container_width=True,
                height=300,
            )

            # Visualise table sizes
            if len(stats_df) > 0:
                fig = create_bar_chart(
                    data=stats_df.sort("Rows", descending=True),
                    x="Rows",
                    y="Table",
                    title="Records per Table",
                    orientation="h",
                    height=400,
                    labels={"Rows": "Number of Records", "Table": "Table Name"},
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("⚠️ No tables found in database")

    except Exception as e:
        st.error(f"Error loading table statistics: {str(e)}")

# Tab 2: Data Freshness
with tab2:
    st.subheader("Data Freshness Indicators")

    try:
        # Calculate freshness metrics
        hours_since_update = time_since_update.total_seconds() / 3600
        freshness_score = max(0, 100 - (hours_since_update / DATA_STALE_HOURS * 100))

        # Freshness gauge
        col1, col2 = st.columns(2)

        with col1:
            fig = create_gauge_chart(
                value=freshness_score,
                title="Overall Data Freshness Score",
                threshold_good=75,
                threshold_bad=50,
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Freshness Thresholds")
            st.markdown(f"""
            - **Fresh**: < {DATA_FRESH_HOURS} hours old
            - **Aging**: {DATA_FRESH_HOURS}-{DATA_STALE_HOURS} hours old
            - **Stale**: > {DATA_STALE_HOURS} hours old

            **Current Status:**
            - Last Update: `{last_modified.strftime("%Y-%m-%d %H:%M:%S")}`
            - Hours Since Update: `{hours_since_update:.1f}`
            - Freshness Score: `{freshness_score:.1f}/100`
            """)

        st.markdown("---")

        # Check for tables with timestamp columns
        st.subheader("Temporal Data Coverage")

        # Try to find date/timestamp columns in main tables
        main_tables = ["master_health_record", "derived_health_indicators"]

        for table_name in main_tables:
            try:
                # Check if table exists
                table_exists = conn.execute(
                    f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
                ).fetchone()[0]

                if table_exists > 0:
                    # Get sample to check for date columns
                    sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 1").description

                    date_cols = [col[0] for col in sample if 'date' in col[0].lower() or 'time' in col[0].lower()]

                    if date_cols:
                        st.markdown(f"#### {table_name}")
                        for col in date_cols[:3]:  # Limit to first 3 date columns
                            try:
                                min_date = conn.execute(f"SELECT MIN({col}) FROM {table_name}").fetchone()[0]
                                max_date = conn.execute(f"SELECT MAX({col}) FROM {table_name}").fetchone()[0]

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric(f"{col} - Earliest", str(min_date) if min_date else "N/A")
                                with col2:
                                    st.metric(f"{col} - Latest", str(max_date) if max_date else "N/A")
                            except:
                                pass

            except Exception as e:
                continue

    except Exception as e:
        st.error(f"Error calculating freshness metrics: {str(e)}")

# Tab 3: Data Coverage
with tab3:
    st.subheader("Data Coverage Statistics")

    try:
        # Get master health record for coverage analysis
        data = db.get_master_health_record()

        if len(data) > 0:
            # Geographic coverage
            st.markdown("#### Geographic Coverage")

            col1, col2, col3 = st.columns(3)

            with col1:
                unique_sa2 = data["sa2_code"].n_unique()
                st.metric(
                    "Unique SA2 Regions",
                    f"{unique_sa2:,}",
                    help="Number of unique SA2 regions in dataset",
                )

            with col2:
                unique_states = data["state_code"].n_unique()
                st.metric(
                    "States Covered",
                    f"{unique_states}",
                    help="Number of unique states/territories",
                )

            with col3:
                if "remoteness_category" in data.columns:
                    unique_remoteness = data["remoteness_category"].n_unique()
                    st.metric(
                        "Remoteness Categories",
                        f"{unique_remoteness}",
                        help="Number of remoteness categories",
                    )

            # Coverage by state
            st.markdown("#### Coverage by State")

            state_coverage = data.group_by("state_code").agg([
                pl.count().alias("SA2_Count"),
                pl.col("total_population").sum().alias("Total_Population"),
            ]).sort("SA2_Count", descending=True)

            fig = create_bar_chart(
                data=state_coverage,
                x="SA2_Count",
                y="state_code",
                title="SA2 Regions by State",
                orientation="h",
                height=400,
                labels={"SA2_Count": "Number of SA2 Regions", "state_code": "State"},
            )
            st.plotly_chart(fig, use_container_width=True)

            # Coverage by remoteness
            if "remoteness_category" in data.columns:
                st.markdown("#### Coverage by Remoteness")

                remoteness_coverage = data.group_by("remoteness_category").agg([
                    pl.count().alias("SA2_Count"),
                    pl.col("total_population").sum().alias("Total_Population"),
                ]).sort("SA2_Count", descending=True)

                fig = create_bar_chart(
                    data=remoteness_coverage,
                    x="SA2_Count",
                    y="remoteness_category",
                    title="SA2 Regions by Remoteness Category",
                    orientation="h",
                    height=400,
                    labels={
                        "SA2_Count": "Number of SA2 Regions",
                        "remoteness_category": "Remoteness Category"
                    },
                )
                st.plotly_chart(fig, use_container_width=True)

            # Population coverage
            st.markdown("#### Population Coverage")

            total_pop = data["total_population"].sum()
            avg_pop = data["total_population"].mean()
            median_pop = data["total_population"].median()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Population", f"{total_pop:,.0f}")

            with col2:
                st.metric("Avg Population/SA2", f"{avg_pop:,.0f}")

            with col3:
                st.metric("Median Population/SA2", f"{median_pop:,.0f}")

        else:
            st.warning("⚠️ No data available for coverage analysis")

    except Exception as e:
        st.error(f"Error calculating coverage statistics: {str(e)}")

# Tab 4: Missing Data Analysis
with tab4:
    st.subheader("Missing Data Analysis")

    if show_missing:
        try:
            data = db.get_master_health_record()

            if len(data) > 0:
                # Calculate missing percentages for each column
                total_rows = len(data)
                missing_stats = []

                for col in data.columns:
                    null_count = data[col].is_null().sum()
                    missing_pct = (null_count / total_rows) * 100

                    missing_stats.append({
                        "Column": col,
                        "Missing_Count": null_count,
                        "Missing_Percentage": round(missing_pct, 2),
                    })

                missing_df = pl.DataFrame(missing_stats).sort(
                    "Missing_Percentage",
                    descending=True,
                )

                # Summary metrics
                columns_with_missing = missing_df.filter(
                    pl.col("Missing_Percentage") > 0
                ).height

                avg_missing = missing_df["Missing_Percentage"].mean()

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Total Columns",
                        len(missing_df),
                        help="Total number of columns analysed",
                    )

                with col2:
                    st.metric(
                        "Columns with Missing Data",
                        columns_with_missing,
                        help="Number of columns with at least one missing value",
                    )

                with col3:
                    st.metric(
                        "Avg Missing %",
                        f"{avg_missing:.2f}%",
                        help="Average percentage of missing data across all columns",
                    )

                # Show top N columns with missing data
                st.markdown(f"#### Top {top_n} Columns with Missing Data")

                top_missing = missing_df.filter(
                    pl.col("Missing_Percentage") > 0
                ).head(top_n)

                if len(top_missing) > 0:
                    # Visualise
                    fig = create_bar_chart(
                        data=top_missing,
                        x="Missing_Percentage",
                        y="Column",
                        title="Missing Data by Column",
                        orientation="h",
                        height=400,
                        labels={
                            "Missing_Percentage": "Missing Data (%)",
                            "Column": "Column Name"
                        },
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Detailed table
                    st.dataframe(
                        top_missing.to_pandas(),
                        use_container_width=True,
                        height=300,
                    )

                else:
                    st.success("✅ No missing data detected in dataset!")

                # Complete completeness table (expandable)
                with st.expander("📋 Complete Missing Data Report"):
                    st.dataframe(
                        missing_df.to_pandas(),
                        use_container_width=True,
                        height=400,
                    )

            else:
                st.warning("⚠️ No data available for missing data analysis")

        except Exception as e:
            st.error(f"Error analysing missing data: {str(e)}")
    else:
        st.info("Enable 'Show Missing Data Analysis' in sidebar to view this section")

# Tab 5: Quality Reports
with tab5:
    st.subheader("Data Quality Reports & Export")

    try:
        # Generate quality report
        st.markdown("#### Quality Summary Report")

        report_data = {
            "Metric": [],
            "Value": [],
            "Status": [],
        }

        # Database metrics
        report_data["Metric"].append("Database Status")
        report_data["Value"].append("Connected")
        report_data["Status"].append("✅")

        report_data["Metric"].append("Last Updated")
        report_data["Value"].append(last_modified.strftime("%Y-%m-%d %H:%M"))
        report_data["Status"].append(
            "✅" if time_since_update < timedelta(hours=DATA_FRESH_HOURS) else "⚠️"
        )

        report_data["Metric"].append("Database Size")
        report_data["Value"].append(f"{db_size_mb:.2f} MB")
        report_data["Status"].append("✅")

        # Table metrics
        if 'stats_df' in locals():
            report_data["Metric"].append("Total Tables")
            report_data["Value"].append(str(len(stats_df)))
            report_data["Status"].append("✅")

            report_data["Metric"].append("Total Records")
            report_data["Value"].append(f"{stats_df['Rows'].sum():,}")
            report_data["Status"].append("✅")

        # Missing data metrics
        if 'missing_df' in locals():
            report_data["Metric"].append("Columns with Missing Data")
            report_data["Value"].append(str(columns_with_missing))
            report_data["Status"].append(
                "✅" if columns_with_missing == 0 else "⚠️"
            )

            report_data["Metric"].append("Avg Missing Data %")
            report_data["Value"].append(f"{avg_missing:.2f}%")
            report_data["Status"].append(
                "✅" if avg_missing < 5 else "⚠️"
            )

        report_df = pl.DataFrame(report_data)

        # Display report
        st.dataframe(
            report_df.to_pandas(),
            use_container_width=True,
            height=300,
        )

        st.markdown("---")

        # Export options
        st.markdown("#### Export Quality Reports")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Download Quality Summary", key="export_summary"):
                # Create comprehensive report
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Export as CSV
                csv_data = report_df.write_csv()

                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"ahgd_quality_report_{timestamp}.csv",
                    mime="text/csv",
                )

        with col2:
            if 'missing_df' in locals() and st.button("📥 Download Missing Data Report", key="export_missing"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                csv_data = missing_df.write_csv()

                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"ahgd_missing_data_report_{timestamp}.csv",
                    mime="text/csv",
                )

        # Full data export
        st.markdown("---")
        st.markdown("#### Export Full Dataset")

        try:
            data = db.get_master_health_record()
            create_export_section(data)

        except Exception as e:
            st.error(f"Error preparing data export: {str(e)}")

        # DBT test results (if available)
        st.markdown("---")
        st.markdown("#### DBT Test Results")

        dbt_target = Path(__file__).parent.parent.parent / "ahgd_dbt" / "target"

        if dbt_target.exists():
            run_results = dbt_target / "run_results.json"

            if run_results.exists():
                try:
                    with open(run_results) as f:
                        results = json.load(f)

                    # Parse results
                    st.json(results, expanded=False)

                    if "results" in results:
                        total_tests = len(results["results"])
                        passed_tests = sum(
                            1 for r in results["results"]
                            if r.get("status") == "pass"
                        )

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Total Tests", total_tests)

                        with col2:
                            st.metric("Passed", passed_tests)

                        with col3:
                            st.metric(
                                "Success Rate",
                                f"{(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "N/A"
                            )

                except Exception as e:
                    st.warning(f"Could not parse DBT results: {str(e)}")
            else:
                st.info("No DBT test results found. Run `dbt test` to generate results.")
        else:
            st.info("DBT target directory not found")

    except Exception as e:
        st.error(f"Error generating quality reports: {str(e)}")

# Footer
st.markdown("---")
st.caption(
    f"Data Quality Dashboard | Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
