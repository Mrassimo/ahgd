"""
Data Processing Utilities for Australian Health Analytics Dashboard

This module contains data transformation, filtering, and processing utilities
extracted from the monolithic dashboard.

Functions:
    filter_data_by_states: Filter data by selected states/territories
    validate_health_data: Validate health data completeness and quality
    calculate_health_risk_score: Calculate composite health risk scores
    identify_health_hotspots: Identify priority areas for health interventions
    prepare_correlation_data: Prepare data for correlation analysis
    generate_health_indicators: Generate synthetic health indicators (demo purposes)
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple


def filter_data_by_states(data: pd.DataFrame, selected_states: List[str]) -> pd.DataFrame:
    """
    Filter dataset by selected states/territories
    
    Args:
        data (pd.DataFrame): Input dataset with STATE_NAME21 column
        selected_states (List[str]): List of state names to include
        
    Returns:
        pd.DataFrame: Filtered dataset
    """
    if not selected_states:
        return data
    
    return data[data['STATE_NAME21'].isin(selected_states)]


def validate_health_data(data: pd.DataFrame) -> dict:
    """
    Validate health data completeness and quality
    
    Args:
        data (pd.DataFrame): Dataset to validate
        
    Returns:
        dict: Validation results with completeness metrics
    """
    validation_results = {
        'total_records': len(data),
        'geographic_coverage': len(data[data['geometry'].notna()]),
        'seifa_completeness': data['IRSD_Score'].notna().sum() / len(data) * 100,
        'health_completeness': data['health_risk_score'].notna().sum() / len(data) * 100,
        'missing_critical_data': len(data[data[['IRSD_Score', 'health_risk_score']].isna().any(axis=1)])
    }
    
    return validation_results


def calculate_health_risk_score(
    mortality_rate: pd.Series,
    diabetes_prevalence: pd.Series,
    heart_disease_rate: pd.Series,  
    mental_health_rate: pd.Series,
    gp_access_score: pd.Series,
    hospital_distance: pd.Series,
    weights: Optional[dict] = None
) -> pd.Series:
    """
    Calculate composite health risk score from individual indicators
    
    Args:
        mortality_rate: Mortality rate per 1,000
        diabetes_prevalence: Diabetes prevalence percentage
        heart_disease_rate: Heart disease rate per 1,000
        mental_health_rate: Mental health issues rate per 1,000
        gp_access_score: GP access score (0-10, higher is better)
        hospital_distance: Distance to nearest hospital (km)
        weights: Optional custom weights dictionary
        
    Returns:
        pd.Series: Composite health risk score
    """
    if weights is None:
        weights = {
            'mortality': 0.3,
            'diabetes': 0.2,
            'heart_disease': 0.15,
            'mental_health': 0.1,
            'gp_access': 0.15,
            'hospital_distance': 0.1
        }
    
    health_risk_score = (
        (mortality_rate * weights['mortality']) +
        (diabetes_prevalence * weights['diabetes']) +
        (heart_disease_rate * weights['heart_disease']) +
        (mental_health_rate * weights['mental_health']) +
        ((10 - gp_access_score) * weights['gp_access']) +
        (hospital_distance / 10 * weights['hospital_distance'])
    )
    
    return health_risk_score


def identify_health_hotspots(data: pd.DataFrame, n_hotspots: int = 20) -> pd.DataFrame:
    """
    Identify areas with poor health outcomes and high disadvantage
    
    Args:
        data (pd.DataFrame): Dataset with health risk and SEIFA data
        n_hotspots (int): Number of hotspots to identify
        
    Returns:
        pd.DataFrame: Top priority areas for health interventions
    """
    # Filter valid data
    valid_data = data.dropna(subset=['health_risk_score', 'IRSD_Score'])
    
    if valid_data.empty:
        return pd.DataFrame()
    
    # Define hotspots as areas with:
    # 1. High health risk (top 30%)
    # 2. High disadvantage (bottom 30% SEIFA scores)
    
    health_risk_threshold = valid_data['health_risk_score'].quantile(0.7)
    disadvantage_threshold = valid_data['IRSD_Score'].quantile(0.3)
    
    hotspots = valid_data[
        (valid_data['health_risk_score'] >= health_risk_threshold) &
        (valid_data['IRSD_Score'] <= disadvantage_threshold)
    ].nlargest(n_hotspots, 'health_risk_score')
    
    return hotspots


def prepare_correlation_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare data for correlation analysis by selecting relevant columns
    
    Args:
        data (pd.DataFrame): Input dataset
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: (prepared_data, column_names)
    """
    # Select relevant columns for correlation analysis
    correlation_columns = [
        'IRSD_Score', 'IRSD_Decile_Australia', 'mortality_rate', 'diabetes_prevalence',
        'heart_disease_rate', 'mental_health_rate', 'gp_access_score', 
        'hospital_distance', 'health_risk_score'
    ]
    
    # Filter to available columns and clean data
    available_columns = [col for col in correlation_columns if col in data.columns]
    correlation_data = data[available_columns].dropna()
    
    return correlation_data, available_columns


def generate_health_indicators(
    merged_data: pd.DataFrame, 
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic health indicators correlated with disadvantage
    
    Note: This is for demonstration purposes. In production, this would
    come from actual health databases.
    
    Args:
        merged_data (pd.DataFrame): Base data with SEIFA scores
        random_seed (int): Random seed for reproducible results
        
    Returns:
        pd.DataFrame: Health indicators dataset
    """
    np.random.seed(random_seed)
    
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
    
    return health_indicators


def calculate_data_quality_metrics(data: pd.DataFrame) -> dict:
    """
    Calculate comprehensive data quality metrics
    
    Args:
        data (pd.DataFrame): Dataset to analyse
        
    Returns:
        dict: Data quality metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['total_records'] = len(data)
    metrics['total_columns'] = len(data.columns)
    
    # Completeness metrics
    key_columns = ['SA2_CODE21', 'IRSD_Score', 'health_risk_score', 'geometry']
    for col in key_columns:
        if col in data.columns:
            completeness = data[col].notna().sum() / len(data) * 100
            metrics[f'{col}_completeness'] = completeness
    
    # Data range validation
    if 'IRSD_Score' in data.columns:
        irsd_data = data['IRSD_Score'].dropna()
        metrics['irsd_range_valid'] = ((irsd_data >= 500) & (irsd_data <= 1200)).sum() / len(irsd_data) * 100
    
    if 'health_risk_score' in data.columns:
        health_data = data['health_risk_score'].dropna()
        metrics['health_risk_range'] = {
            'min': health_data.min(),
            'max': health_data.max(),
            'mean': health_data.mean(),
            'std': health_data.std()
        }
    
    return metrics


def apply_scenario_analysis(
    data: pd.DataFrame,
    improvement_percentage: float
) -> pd.DataFrame:
    """
    Apply scenario analysis for SEIFA improvement impact
    
    Args:
        data (pd.DataFrame): Base dataset
        improvement_percentage (float): Percentage improvement in SEIFA scores
        
    Returns:
        pd.DataFrame: Dataset with scenario projections
    """
    scenario_data = data.copy()
    
    # Improve SEIFA scores
    scenario_data['improved_seifa'] = scenario_data['IRSD_Score'] * (1 + improvement_percentage/100)
    
    # Calculate improved health risk (simplified linear relationship)
    scenario_data['improved_health_risk'] = scenario_data['health_risk_score'] * (1 - improvement_percentage/200)
    
    # Ensure realistic bounds
    scenario_data['improved_seifa'] = np.clip(scenario_data['improved_seifa'], 500, 1200)
    scenario_data['improved_health_risk'] = np.maximum(0, scenario_data['improved_health_risk'])
    
    return scenario_data