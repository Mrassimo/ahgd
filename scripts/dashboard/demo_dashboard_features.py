#!/usr/bin/env python3
"""
Dashboard Feature Demonstration Script

This script demonstrates the core analytical capabilities of the Australian Health Analytics Dashboard
by running key analyses and generating sample outputs that would be shown in the dashboard.

Useful for portfolio presentations and understanding dashboard capabilities without running the full UI.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class HealthAnalyticsDemonstration:
    """Demonstration class for health analytics capabilities"""
    
    def __init__(self):
        """Initialize the demonstration with data loading"""
        self.data = None
        self.correlation_matrix = None
        self.hotspots = None
        
    def load_data(self):
        """Load and prepare demonstration data"""
        print("🔄 Loading Australian health and geographic data...")
        
        try:
            # Load geographic and SEIFA data
            seifa_df = pd.read_parquet('data/processed/seifa_2021_sa2.parquet')
            boundaries_gdf = gpd.read_parquet('data/processed/sa2_boundaries_2021.parquet')
            
            # Merge geographic and SEIFA data
            merged_data = boundaries_gdf.merge(
                seifa_df, 
                left_on='SA2_CODE21', 
                right_on='SA2_Code_2021', 
                how='left'
            )
            
            # Generate synthetic health indicators for demonstration
            np.random.seed(42)  # For reproducible demo
            
            n_records = len(merged_data)
            disadvantage_effect = (merged_data['IRSD_Score'].fillna(1000) - 1000) / 100
            
            # Create health indicators correlated with disadvantage
            health_indicators = pd.DataFrame({
                'SA2_CODE21': merged_data['SA2_CODE21'],
                'SA2_NAME21': merged_data['SA2_NAME21'],
                'STATE_NAME21': merged_data['STE_NAME21'],
                
                'mortality_rate': np.maximum(0, 
                    8.5 - disadvantage_effect * 0.8 + np.random.normal(0, 1.2, n_records)
                ),
                'diabetes_prevalence': np.maximum(0, 
                    4.2 - disadvantage_effect * 0.6 + np.random.normal(0, 0.8, n_records)
                ),
                'heart_disease_rate': np.maximum(0, 
                    12.8 - disadvantage_effect * 1.2 + np.random.normal(0, 2.1, n_records)
                ),
                'mental_health_rate': np.maximum(0, 
                    18.5 - disadvantage_effect * 1.5 + np.random.normal(0, 3.2, n_records)
                ),
                'gp_access_score': np.maximum(0, np.minimum(10,
                    7.2 + disadvantage_effect * 0.4 + np.random.normal(0, 1.1, n_records)
                )),
                'hospital_distance': np.maximum(1,
                    15.2 - disadvantage_effect * 2.1 + np.random.normal(0, 8.5, n_records)
                )
            })
            
            # Merge all data
            self.data = merged_data.merge(health_indicators, on='SA2_CODE21', how='left')
            
            # Calculate composite health risk score
            self.data['health_risk_score'] = (
                (self.data['mortality_rate'] * 0.3) +
                (self.data['diabetes_prevalence'] * 0.2) +
                (self.data['heart_disease_rate'] * 0.15) +
                (self.data['mental_health_rate'] * 0.1) +
                ((10 - self.data['gp_access_score']) * 0.15) +
                (self.data['hospital_distance'] / 10 * 0.1)
            )
            
            print(f"✅ Data loaded successfully: {len(self.data):,} SA2 areas")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def demonstrate_geographic_analysis(self):
        """Demonstrate geographic health analysis capabilities"""
        print("\n🗺️ GEOGRAPHIC HEALTH ANALYSIS DEMONSTRATION")
        print("=" * 55)
        
        # Geographic coverage summary
        print("📊 Geographic Coverage:")
        state_coverage = self.data['STATE_NAME21'].value_counts()
        for state, count in state_coverage.items():
            print(f"   {state}: {count:,} SA2 areas")
        
        print(f"\n📍 Total Coverage: {len(self.data):,} Statistical Areas across Australia")
        
        # Health indicator statistics
        print("\n🏥 Health Indicator Summary:")
        health_cols = ['mortality_rate', 'diabetes_prevalence', 'heart_disease_rate', 
                      'mental_health_rate', 'gp_access_score', 'hospital_distance']
        
        for col in health_cols:
            data_col = self.data[col].dropna()
            print(f"   {col.replace('_', ' ').title()}: "
                  f"Mean={data_col.mean():.2f}, "
                  f"Range={data_col.min():.2f}-{data_col.max():.2f}")
        
        # State-level health risk comparison
        print("\n🎯 Health Risk by State/Territory:")
        state_risk = self.data.groupby('STATE_NAME21')['health_risk_score'].agg(['mean', 'std', 'count']).round(2)
        state_risk = state_risk.sort_values('mean', ascending=False)
        
        for state, row in state_risk.iterrows():
            if pd.notna(row['mean']):
                print(f"   {state}: Risk Score {row['mean']:.2f} (±{row['std']:.2f}) - {row['count']} areas")
    
    def demonstrate_correlation_analysis(self):
        """Demonstrate correlation analysis capabilities"""
        print("\n📊 CORRELATION ANALYSIS DEMONSTRATION")
        print("=" * 45)
        
        # Calculate correlation matrix
        correlation_columns = [
            'IRSD_Score', 'IRSD_Decile_Australia', 'mortality_rate', 'diabetes_prevalence',
            'heart_disease_rate', 'mental_health_rate', 'gp_access_score', 
            'hospital_distance', 'health_risk_score'
        ]
        
        correlation_data = self.data[correlation_columns].dropna()
        self.correlation_matrix = correlation_data.corr()
        
        print(f"📈 Correlation Analysis based on {len(correlation_data):,} complete records")
        
        # Key correlations with SEIFA disadvantage
        seifa_correlations = self.correlation_matrix['IRSD_Score'].drop('IRSD_Score')
        seifa_correlations = seifa_correlations.sort_values(key=abs, ascending=False)
        
        print("\n🔗 Strongest Correlations with SEIFA Disadvantage Score:")
        print("   (Higher SEIFA scores = less disadvantaged)")
        
        for variable, correlation in seifa_correlations.head(6).items():
            direction = "↗️ Positive" if correlation > 0 else "↘️ Negative"
            strength = "Very Strong" if abs(correlation) > 0.7 else "Strong" if abs(correlation) > 0.5 else "Moderate" if abs(correlation) > 0.3 else "Weak"
            print(f"   {variable.replace('_', ' ').title()}: {direction} ({correlation:.3f}) - {strength}")
        
        # Statistical significance
        print(f"\n📊 Correlation Strength Summary:")
        strong_corr = len(seifa_correlations[abs(seifa_correlations) > 0.5])
        moderate_corr = len(seifa_correlations[(abs(seifa_correlations) > 0.3) & (abs(seifa_correlations) <= 0.5)])
        
        print(f"   Strong correlations (>0.5): {strong_corr}")
        print(f"   Moderate correlations (0.3-0.5): {moderate_corr}")
        print(f"   R² for Health Risk Score: {(seifa_correlations['health_risk_score']**2):.3f}")
    
    def demonstrate_hotspot_identification(self):
        """Demonstrate health hotspot identification"""
        print("\n🎯 HEALTH HOTSPOT IDENTIFICATION DEMONSTRATION")
        print("=" * 55)
        
        # Identify hotspots
        valid_data = self.data.dropna(subset=['health_risk_score', 'IRSD_Score'])
        
        health_risk_threshold = valid_data['health_risk_score'].quantile(0.7)
        disadvantage_threshold = valid_data['IRSD_Score'].quantile(0.3)
        
        self.hotspots = valid_data[
            (valid_data['health_risk_score'] >= health_risk_threshold) &
            (valid_data['IRSD_Score'] <= disadvantage_threshold)
        ].nlargest(20, 'health_risk_score')
        
        print(f"🚨 Identified {len(self.hotspots)} Priority Health Hotspots")
        print(f"   Criteria: Health Risk ≥ {health_risk_threshold:.2f} AND SEIFA Score ≤ {disadvantage_threshold:.0f}")
        
        # Hotspot statistics
        print(f"\n📊 Hotspot vs National Comparison:")
        national_health_risk = valid_data['health_risk_score'].mean()
        national_seifa = valid_data['IRSD_Score'].mean()
        
        hotspot_health_risk = self.hotspots['health_risk_score'].mean()
        hotspot_seifa = self.hotspots['IRSD_Score'].mean()
        
        print(f"   Average Health Risk: {hotspot_health_risk:.2f} vs {national_health_risk:.2f} (national)")
        print(f"   Average SEIFA Score: {hotspot_seifa:.0f} vs {national_seifa:.0f} (national)")
        print(f"   Risk Difference: +{(hotspot_health_risk - national_health_risk):.2f} points")
        print(f"   Disadvantage Difference: {(hotspot_seifa - national_seifa):.0f} points")
        
        # Top 10 hotspots
        print(f"\n🏆 Top 10 Priority Areas for Intervention:")
        top_hotspots = self.hotspots.head(10)
        
        # Debug: Check what columns we actually have
        available_name_cols = [col for col in top_hotspots.columns if 'SA2_NAME' in col]
        available_state_cols = [col for col in top_hotspots.columns if 'STATE' in col or 'STE_NAME' in col]
        
        name_col = available_name_cols[0] if available_name_cols else 'SA2_CODE21'
        state_col = available_state_cols[0] if available_state_cols else 'STATE_NAME21'
        
        for idx, (_, row) in enumerate(top_hotspots.iterrows(), 1):
            priority_level = "🔴 Critical" if row['health_risk_score'] > 12 else "🟡 High"
            name = row.get(name_col, row.get('SA2_CODE21', 'Unknown'))
            state = row.get(state_col, 'Unknown')
            print(f"   {idx:2d}. {name}, {state}")
            print(f"       Risk: {row['health_risk_score']:.2f} | SEIFA: {row['IRSD_Score']:.0f} | {priority_level}")
        
        # State distribution of hotspots
        print(f"\n📍 Hotspot Distribution by State:")
        if state_col in self.hotspots.columns:
            hotspot_states = self.hotspots[state_col].value_counts()
            for state, count in hotspot_states.items():
                print(f"   {state}: {count} priority areas")
        else:
            print("   State distribution not available")
    
    def demonstrate_predictive_analysis(self):
        """Demonstrate predictive risk analysis capabilities"""
        print("\n🔮 PREDICTIVE RISK ANALYSIS DEMONSTRATION")
        print("=" * 50)
        
        # Simple prediction model
        valid_data = self.data.dropna(subset=['IRSD_Score', 'health_risk_score'])
        
        # Calculate regression parameters
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(
            valid_data['IRSD_Score'], valid_data['health_risk_score']
        )
        
        print(f"📈 Predictive Model Performance:")
        print(f"   Correlation coefficient (r): {r_value:.3f}")
        print(f"   R-squared (explained variance): {r_value**2:.3f}")
        print(f"   Statistical significance (p-value): {p_value:.2e}")
        print(f"   Standard error: {std_err:.3f}")
        
        # Prediction examples
        print(f"\n🎯 Risk Prediction Examples:")
        
        seifa_examples = [600, 800, 1000, 1200]  # Range from high to low disadvantage
        
        for seifa_score in seifa_examples:
            predicted_risk = slope * seifa_score + intercept
            disadvantage_level = "Very High" if seifa_score < 700 else "High" if seifa_score < 900 else "Moderate" if seifa_score < 1100 else "Low"
            
            print(f"   SEIFA Score {seifa_score}: Predicted Risk = {predicted_risk:.2f} ({disadvantage_level} disadvantage)")
        
        # Scenario analysis
        print(f"\n🌟 Scenario Analysis: Impact of Socio-Economic Improvement")
        
        improvement_scenarios = [5, 10, 20, 30]  # Percentage improvements
        
        for improvement in improvement_scenarios:
            
            # Calculate current and improved scenarios
            current_avg_risk = valid_data['health_risk_score'].mean()
            
            # Simulate improvement (simplified model)
            improved_seifa = valid_data['IRSD_Score'] * (1 + improvement/100)
            improved_risk = slope * improved_seifa + intercept
            improved_avg_risk = improved_risk.mean()
            
            risk_reduction = current_avg_risk - improved_avg_risk
            population_affected = len(valid_data) * 3000  # Approximate population per SA2
            
            print(f"   {improvement}% SEIFA improvement: Risk reduction = {risk_reduction:.2f} points")
            print(f"                                   Population benefiting ≈ {population_affected:,}")
    
    def demonstrate_data_quality(self):
        """Demonstrate data quality assessment capabilities"""
        print("\n📋 DATA QUALITY & METHODOLOGY DEMONSTRATION")
        print("=" * 55)
        
        print(f"📊 Data Coverage Assessment:")
        print(f"   Total SA2 Areas: {len(self.data):,}")
        print(f"   Geographic Coverage: {len(self.data['STATE_NAME21'].unique())} states/territories")
        
        # Data completeness
        key_variables = ['IRSD_Score', 'health_risk_score', 'mortality_rate', 'diabetes_prevalence']
        
        print(f"\n📈 Data Completeness:")
        for var in key_variables:
            completeness = (self.data[var].notna().sum() / len(self.data)) * 100
            print(f"   {var.replace('_', ' ').title()}: {completeness:.1f}%")
        
        # Data quality metrics
        print(f"\n🔍 Data Quality Metrics:")
        
        # Check for outliers
        health_risk_data = self.data['health_risk_score'].dropna()
        q1, q3 = health_risk_data.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = health_risk_data[(health_risk_data < q1 - 1.5*iqr) | (health_risk_data > q3 + 1.5*iqr)]
        
        print(f"   Outliers in Health Risk Score: {len(outliers)} ({len(outliers)/len(health_risk_data)*100:.1f}%)")
        print(f"   Data Range: {health_risk_data.min():.2f} to {health_risk_data.max():.2f}")
        print(f"   Standard Deviation: {health_risk_data.std():.2f}")
        
        # Temporal consistency
        print(f"\n📅 Data Currency:")
        print(f"   Geographic Boundaries: 2021 Australian Census")
        print(f"   SEIFA Data: 2021 Census (Most Recent)")
        print(f"   Health Indicators: Modelled for demonstration")
        print(f"   Processing Date: June 2025")
        
        # Methodology transparency
        print(f"\n🔬 Methodology Summary:")
        print(f"   Health Risk Score Components:")
        print(f"   • Mortality Rate (30% weight)")
        print(f"   • Diabetes Prevalence (20% weight)")
        print(f"   • Heart Disease Rate (15% weight)")
        print(f"   • Mental Health Rate (10% weight)")
        print(f"   • GP Access Score (15% weight)")
        print(f"   • Hospital Distance (10% weight)")
        
        print(f"\n⚠️  Important Limitations:")
        print(f"   • Health indicators modelled for portfolio demonstration")
        print(f"   • Actual deployment requires confidential health databases")
        print(f"   • Correlation does not imply causation")
        print(f"   • SA2-level analysis may mask within-area variation")
    
    def run_full_demonstration(self):
        """Run complete demonstration of all dashboard capabilities"""
        print("🏥 AUSTRALIAN HEALTH ANALYTICS DASHBOARD")
        print("📊 COMPREHENSIVE CAPABILITY DEMONSTRATION")
        print("=" * 60)
        
        # Load data
        if not self.load_data():
            return False
        
        # Run all demonstrations
        self.demonstrate_geographic_analysis()
        self.demonstrate_correlation_analysis()
        self.demonstrate_hotspot_identification()
        self.demonstrate_predictive_analysis()
        self.demonstrate_data_quality()
        
        # Summary
        print("\n🎯 DEMONSTRATION SUMMARY")
        print("=" * 30)
        print("✅ Geographic Health Mapping: Interactive choropleth maps with 2,454 SA2 areas")
        print("✅ Correlation Analysis: Statistical relationships between disadvantage and health")
        print("✅ Hotspot Identification: 20 priority areas for targeted intervention")
        print("✅ Predictive Modelling: Risk assessment and scenario analysis capabilities")
        print("✅ Data Quality Assurance: Comprehensive methodology and validation metrics")
        
        print(f"\n💡 Key Portfolio Insights:")
        print(f"   • Demonstrates modern data science and visualisation capabilities")
        print(f"   • Shows understanding of health policy and geographic analysis")
        print(f"   • Exhibits statistical modelling and predictive analytics skills")
        print(f"   • Highlights data quality management and methodology transparency")
        
        print(f"\n🚀 Dashboard Features Ready for Interactive Exploration:")
        print(f"   Run: python run_dashboard.py")
        print(f"   URL: http://localhost:8501")
        
        return True

def main():
    """Run the demonstration"""
    demo = HealthAnalyticsDemonstration()
    demo.run_full_demonstration()

if __name__ == "__main__":
    main()