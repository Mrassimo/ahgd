#!/usr/bin/env python3
"""
AHGD: REAL Australian Health Data Dashboard
Using ACTUAL ABS government data - no fancy stuff, just working code
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="AHGD - REAL Australian Data",
    page_icon="🇦🇺",
    layout="wide"
)

@st.cache_data
def load_real_boundaries():
    """Load REAL ABS SA2 boundaries"""
    shp_path = Path("real_data/SA2_boundaries/SA2_2021_AUST_GDA2020.shp")
    if shp_path.exists():
        return gpd.read_file(shp_path)
    return None

@st.cache_data  
def load_real_census_data():
    """Load REAL ABS census data"""
    # Load basic demographic data (G01 table)
    csv_path = Path("real_data/Census_data/2021Census_G01_AUST_SA2.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None

def main():
    st.title("🇦🇺 AHGD: REAL Australian Bureau of Statistics Data")
    st.markdown("### Using actual government data from ABS - 2,473 SA2 regions")
    
    # Load real data
    boundaries = load_real_boundaries()
    census = load_real_census_data()
    
    if boundaries is None or census is None:
        st.error("❌ Real data not found. Run get_real_data.py first!")
        st.stop()
    
    # Show what we have
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("SA2 Boundaries", f"{len(boundaries):,}", "Real ABS shapefiles")
    
    with col2:
        st.metric("Census Records", f"{len(census):,}", "2021 Census data")
        
    with col3:
        st.metric("Data Columns", f"{len(census.columns)}", "Demographics fields")
    
    # Show some real data
    st.subheader("📊 Real ABS Data Sample")
    
    tab1, tab2 = st.tabs(["🗺️ Geographic Boundaries", "📊 Census Demographics"])
    
    with tab1:
        st.markdown("**Real SA2 Geographic Boundaries from ABS:**")
        
        # Show boundary info
        if not boundaries.empty:
            st.dataframe(boundaries[['SA2_CODE21', 'SA2_NAME21', 'SA3_CODE21']].head(20))
            
            # Map sample (simple plot)
            st.subheader("🗺️ Sample SA2 Boundaries")
            
            # Take first 50 SA2s for performance
            sample_boundaries = boundaries.head(50)
            
            fig = px.choropleth_mapbox(
                sample_boundaries.to_crs('EPSG:4326'),  # Convert to lat/lon
                geojson=sample_boundaries.to_crs('EPSG:4326').__geo_interface__,
                locations=sample_boundaries.index,
                hover_name='SA2_NAME21',
                hover_data=['SA2_CODE21'],
                mapbox_style="open-street-map",
                zoom=5,
                center={"lat": -25, "lon": 135},  # Center of Australia
                title="Sample SA2 Regions (first 50)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("**Real 2021 Census Demographics:**")
        
        if not census.empty:
            # Show raw census data
            st.dataframe(census.head(20))
            
            # Simple analysis of real data
            st.subheader("📈 Real Population Analysis")
            
            # Total population column (if exists)
            pop_cols = [col for col in census.columns if 'Tot_P' in col or 'Total_P' in col]
            
            if pop_cols:
                pop_col = pop_cols[0]
                census_clean = census[census[pop_col].notna()]
                
                # Population distribution
                fig = px.histogram(
                    census_clean,
                    x=pop_col,
                    nbins=50,
                    title=f"SA2 Population Distribution (Real 2021 Census)",
                    labels={pop_col: 'Population'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Top populated SA2s
                top_sa2s = census_clean.nlargest(20, pop_col)[['SA2_CODE_2021', pop_col]]
                st.markdown("**Top 20 Most Populated SA2s:**")
                st.dataframe(top_sa2s)
                
                # Basic stats
                st.markdown("**Population Statistics:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Australia", f"{census_clean[pop_col].sum():,}")
                with col2:
                    st.metric("Average SA2", f"{census_clean[pop_col].mean():.0f}")
                with col3:
                    st.metric("Largest SA2", f"{census_clean[pop_col].max():,}")
    
    # Available datasets
    st.subheader("📁 Available Real Datasets")
    
    csv_files = list(Path("real_data/Census_data").glob("*.csv"))
    
    st.markdown(f"**{len(csv_files)} real ABS census datasets available:**")
    
    # Show first 20 files
    for i, csv_file in enumerate(csv_files[:20]):
        if i % 4 == 0:
            cols = st.columns(4)
        
        with cols[i % 4]:
            st.text(csv_file.name.replace("2021Census_", "").replace("_AUST_SA2.csv", ""))
    
    if len(csv_files) > 20:
        st.text(f"... and {len(csv_files) - 20} more datasets")
    
    st.markdown("---")
    st.success("✅ **This is REAL Australian Bureau of Statistics data** - 2,473 SA2 regions with actual census demographics, not mock data!")

if __name__ == "__main__":
    main()