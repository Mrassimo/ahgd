"""
AHGD Dashboard - Main Application

Interactive dashboard for Australian Health Geography Data.
Provides visualizations and analysis of health indicators, demographics,
and socioeconomic factors across Australian Statistical Areas.
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboards.config import (
    DASHBOARD_TITLE,
    DASHBOARD_ICON,
    PAGE_LAYOUT,
    PAGES,
    DB_PATH,
)
from dashboards.utils.database import get_db_connection

# Page configuration
st.set_page_config(
    page_title=DASHBOARD_TITLE,
    page_icon=DASHBOARD_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown(f"# {DASHBOARD_ICON} {DASHBOARD_TITLE}")
    st.markdown("---")

    # Database status
    try:
        db = get_db_connection(str(DB_PATH))
        st.success("✅ Database Connected")

        # Show quick stats
        summary = db.get_health_summary()
        st.metric("Total SA2 Regions", f"{summary['total_sa2']:,}")
        st.metric("Total Population", f"{summary['total_population']:,.0f}")

    except FileNotFoundError:
        st.error("❌ Database Not Found")
        st.warning(f"Expected location: {DB_PATH}")
        st.info(
            """
            **To fix this:**
            1. Run the Airflow ETL pipeline
            2. Ensure `ahgd.db` exists in project root
            3. Refresh this page
            """
        )
        st.stop()
    except Exception as e:
        st.error(f"❌ Database Error: {str(e)}")
        st.stop()

    st.markdown("---")

    # Navigation info
    st.markdown("### 📑 Available Pages")
    for page_name, page_info in PAGES.items():
        st.markdown(f"{page_info['icon']} **{page_name}**")
        st.caption(page_info["description"])

    st.markdown("---")

    # About section
    with st.expander("ℹ️ About"):
        st.markdown(
            """
            **AHGD Dashboard v1.0**

            This dashboard visualizes Australian health geography data,
            combining health indicators, demographics, socioeconomic factors,
            and geographic information at the SA2 level.

            **Data Sources:**
            - Australian Bureau of Statistics (ABS)
            - Australian Institute of Health and Welfare (AIHW)
            - Medicare data
            - Bureau of Meteorology (BOM)

            **Last Updated:** Real-time from DuckDB
            """
        )

    # Footer
    st.markdown("---")
    st.caption("Built with Streamlit + DuckDB")

# Main content
st.markdown(
    f'<div class="main-header">{DASHBOARD_ICON} {DASHBOARD_TITLE}</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Interactive Analysis of Australian Health Geography Data</div>',
    unsafe_allow_html=True,
)

# Welcome message
st.markdown(
    """
    <div class="info-box">
    <strong>👋 Welcome to the AHGD Dashboard!</strong><br>
    Use the sidebar to navigate between different analysis pages. Each page provides
    interactive visualizations and insights into Australian health geography data.
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick overview
col1, col2, col3, col4 = st.columns(4)

try:
    summary = db.get_health_summary()

    with col1:
        st.metric(
            "SA2 Regions",
            f"{summary['total_sa2']:,}",
            help="Total number of Statistical Area Level 2 regions",
        )

    with col2:
        st.metric(
            "Avg Mortality Rate",
            f"{summary['avg_mortality_rate']:.2f}",
            help="Average mortality rate per 1,000 population",
        )

    with col3:
        st.metric(
            "Avg Utilisation Rate",
            f"{summary['avg_utilisation_rate']:.1f}%",
            help="Average healthcare service utilisation rate",
        )

    with col4:
        st.metric(
            "Composite Health Index",
            f"{summary['avg_composite_index']:.1f}",
            help="Average composite health score (higher is better)",
        )

except Exception as e:
    st.error(f"Error loading summary data: {str(e)}")

st.markdown("---")

# Feature highlights
st.subheader("🚀 Dashboard Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        **📊 Data Visualization**
        - Interactive choropleth maps
        - Health indicator trends
        - Correlation analysis
        - Geographic clustering
        """
    )

    st.markdown(
        """
        **🔍 Analysis Tools**
        - Filter by state, remoteness
        - Compare regions
        - Export data to CSV/Excel
        - Drill-down capabilities
        """
    )

with col2:
    st.markdown(
        """
        **🗺️ Geographic Analysis**
        - SA2-level health mapping
        - Remoteness categories
        - State comparisons
        - Spatial patterns
        """
    )

    st.markdown(
        """
        **💡 Insights**
        - Socioeconomic correlations
        - Climate impact analysis
        - Risk factor identification
        - Data quality monitoring
        """
    )

st.markdown("---")

# Getting started
st.subheader("🎯 Getting Started")

st.markdown(
    """
    1. **Explore Overview** - Start with the high-level summary
    2. **Geographic Analysis** - Dive into interactive maps
    3. **Health Indicators** - Analyze specific health metrics
    4. **Socioeconomic Impact** - Understand correlations
    5. **Data Quality** - Check pipeline status and data freshness

    Use the **sidebar navigation** to switch between pages.
    """
)

# Data refresh info
st.info(
    """
    **🔄 Data Updates**

    This dashboard connects directly to the DuckDB database populated by the Airflow ETL pipeline.
    Data is cached for 5 minutes for performance. To see the latest data, wait for the cache to expire
    or restart the dashboard.
    """
)

# Recent activity (placeholder for future enhancement)
with st.expander("📈 Recent Activity"):
    st.markdown(
        """
        - ✅ Dashboard initialized
        - ✅ Database connection established
        - ✅ Summary metrics loaded

        *Future: Show ETL pipeline run history and data updates*
        """
    )
