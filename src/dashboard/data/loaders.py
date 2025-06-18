"""
Data Loading Functions for Australian Health Analytics Dashboard

This module contains the core data loading functionality extracted from the
monolithic dashboard. All caching behaviour and data loading logic is preserved.

Functions:
    load_data: Main data loading function with caching
    calculate_correlations: Correlation analysis between SEIFA and health indicators
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import sys

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.config import get_global_config

# Get configuration
config = get_global_config()


@st.cache_data(ttl=config.dashboard.cache_ttl)
def load_data():
    """
    Load and prepare all necessary datasets
    
    This function loads geographic boundary data and SEIFA socio-economic data,
    then generates synthetic health indicators for demonstration purposes.
    
    Returns:
        gpd.GeoDataFrame: Combined dataset with geographic boundaries, SEIFA data,
                         and synthetic health indicators
    
    Raises:
        Exception: If data loading fails
    """
    try:
        # Load geographic and SEIFA data using configuration paths
        seifa_df = pd.read_parquet(config.data_source.processed_data_dir / 'seifa_2021_sa2.parquet')
        boundaries_gdf = gpd.read_parquet(config.data_source.processed_data_dir / 'sa2_boundaries_2021.parquet')
        
        # Merge geographic and SEIFA data
        merged_data = boundaries_gdf.merge(
            seifa_df, 
            left_on='SA2_CODE21', 
            right_on='SA2_Code_2021', 
            how='left'
        )
        
        # Create synthetic health indicators for demonstration
        # (In production, this would come from actual health databases)
        np.random.seed(42)  # For reproducible demo data
        
        n_records = len(merged_data)
        
        # Generate health indicators correlated with disadvantage
        disadvantage_effect = (merged_data['IRSD_Score'].fillna(1000) - 1000) / 100
        
        health_indicators = pd.DataFrame({
            'SA2_CODE21': merged_data['SA2_CODE21'],
            'SA2_NAME21': merged_data['SA2_NAME21'],
            'STATE_NAME21': merged_data['STE_NAME21'],
            
            # Mortality indicators (higher disadvantage = higher mortality)
            'mortality_rate': np.maximum(0, 
                8.5 - disadvantage_effect * 0.8 + np.random.normal(0, 1.2, n_records)
            ),
            
            # Chronic disease prevalence (higher disadvantage = higher disease)
            'diabetes_prevalence': np.maximum(0, 
                4.2 - disadvantage_effect * 0.6 + np.random.normal(0, 0.8, n_records)
            ),
            'heart_disease_rate': np.maximum(0, 
                12.8 - disadvantage_effect * 1.2 + np.random.normal(0, 2.1, n_records)
            ),
            'mental_health_rate': np.maximum(0, 
                18.5 - disadvantage_effect * 1.5 + np.random.normal(0, 3.2, n_records)
            ),
            
            # Healthcare access (higher disadvantage = lower access)
            'gp_access_score': np.maximum(0, np.minimum(10,
                7.2 + disadvantage_effect * 0.4 + np.random.normal(0, 1.1, n_records)
            )),
            'hospital_distance': np.maximum(1,
                15.2 - disadvantage_effect * 2.1 + np.random.normal(0, 8.5, n_records)
            )
        })
        
        # Merge all data
        final_data = merged_data.merge(health_indicators, on='SA2_CODE21', how='left')
        
        # Calculate composite health risk score
        final_data['health_risk_score'] = (
            (final_data['mortality_rate'] * 0.3) +
            (final_data['diabetes_prevalence'] * 0.2) +
            (final_data['heart_disease_rate'] * 0.15) +
            (final_data['mental_health_rate'] * 0.1) +
            ((10 - final_data['gp_access_score']) * 0.15) +
            (final_data['hospital_distance'] / 10 * 0.1)
        )
        
        return final_data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


@st.cache_data
def calculate_correlations(_data):
    """
    Calculate correlation matrix between SEIFA and health indicators
    
    Args:
        _data (pd.DataFrame): Dataset containing SEIFA and health indicators
        
    Returns:
        tuple: (correlation_matrix, correlation_data)
            - correlation_matrix (pd.DataFrame): Correlation matrix
            - correlation_data (pd.DataFrame): Clean data used for correlations
    """
    
    # Select relevant columns for correlation analysis
    correlation_columns = [
        'IRSD_Score', 'IRSD_Decile_Australia', 'mortality_rate', 'diabetes_prevalence',
        'heart_disease_rate', 'mental_health_rate', 'gp_access_score', 
        'hospital_distance', 'health_risk_score'
    ]
    
    # Handle empty data case
    if _data.empty:
        empty_df = pd.DataFrame()
        return empty_df, empty_df
    
    # Filter to only available columns
    available_columns = [col for col in correlation_columns if col in _data.columns]
    
    if not available_columns:
        empty_df = pd.DataFrame()
        return empty_df, empty_df
    
    correlation_data = _data[available_columns].dropna()
    
    if correlation_data.empty:
        empty_df = pd.DataFrame()
        return empty_df, empty_df
    
    correlation_matrix = correlation_data.corr()
    
    return correlation_matrix, correlation_data