"""
Example: Basic analysis of Australian Health and Geographic Data

This example demonstrates how to load and analyse the AHGD dataset
using Python and common data science libraries.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(format_type='parquet'):
    """Load AHGD dataset in specified format."""
    if format_type == 'parquet':
        return pd.read_parquet('ahgd_data.parquet')
    elif format_type == 'csv':
        return pd.read_csv('ahgd_data.csv')
    elif format_type == 'json':
        import json
        with open('ahgd_data.json', 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data['data'])
    else:
        raise ValueError(f"Unsupported format: {format_type}")

def basic_analysis():
    """Perform basic statistical analysis."""
    # Load data
    df = load_dataset('parquet')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Summary statistics
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    print("\nSummary Statistics:")
    print(df[numeric_cols].describe())
    
    # State-level aggregations
    if 'state_name' in df.columns and 'life_expectancy_years' in df.columns:
        state_health = df.groupby('state_name').agg({
            'life_expectancy_years': 'mean',
            'smoking_prevalence_percent': 'mean',
            'obesity_prevalence_percent': 'mean'
        }).round(2)
        
        print("\nHealth Indicators by State:")
        print(state_health)
    
    return df

def create_visualisations(df):
    """Create basic visualisations."""
    plt.style.use('seaborn-v0_8')
    
    # Life expectancy distribution
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    df['life_expectancy_years'].hist(bins=20, alpha=0.7)
    plt.title('Distribution of Life Expectancy')
    plt.xlabel('Years')
    
    # Health indicators correlation
    if all(col in df.columns for col in ['life_expectancy_years', 'smoking_prevalence_percent']):
        plt.subplot(2, 2, 2)
        plt.scatter(df['smoking_prevalence_percent'], df['life_expectancy_years'], alpha=0.6)
        plt.xlabel('Smoking Prevalence (%)')
        plt.ylabel('Life Expectancy (Years)')
        plt.title('Smoking vs Life Expectancy')
    
    plt.tight_layout()
    plt.savefig('ahgd_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Run basic analysis
    data = basic_analysis()
    
    # Create visualisations
    create_visualisations(data)
    
    print("\nAnalysis complete! Check ahgd_analysis.png for visualisations.")
