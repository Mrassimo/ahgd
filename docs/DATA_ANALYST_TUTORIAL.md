# AHGD Data Analyst Tutorial

*Working with Australian Health Data for Analysis*

This tutorial is designed for data analysts who want to leverage AHGD for Australian health data analysis, visualisation, and reporting. You'll learn how to extract meaningful insights from integrated health and geographic datasets.

## Prerequisites

Before starting this tutorial, ensure you have:

- **Completed the [Quick Start Guide](QUICK_START_GUIDE.md)**
- **Python experience** with pandas, numpy, and matplotlib
- **Basic understanding** of Australian geography (SA2, SA3, LGA boundaries)
- **Statistical knowledge** for health data interpretation

## Analysis Environment Setup

### Install Analysis Dependencies

```bash
# Activate your AHGD environment
source venv/bin/activate

# Install analysis-specific packages
pip install -r requirements-analysis.txt

# Or install individually
pip install jupyter geopandas matplotlib seaborn plotly folium statsmodels scikit-learn
```

### Launch Jupyter Environment

```bash
# Start Jupyter with AHGD configuration
export AHGD_CONFIG=configs/analysis.yaml
jupyter lab

# Alternative: Use the provided analysis notebook
jupyter lab examples/analysis_starter.ipynb
```

## Core Health Datasets

### Understanding Available Data Sources

AHGD integrates multiple Australian health data sources:

| Source | Dataset | Description | Geographic Level |
|--------|---------|-------------|------------------|
| **AIHW** | disease_prevalence | Chronic disease rates | SA2, SA3 |
| **AIHW** | health_services | GP visits, hospital admissions | SA2 |
| **AIHW** | mental_health | Mental health indicators | SA3 |
| **ABS** | mortality_stats | Death rates by cause | SA2 |
| **ABS** | disability_stats | Disability prevalence | SA2 |
| **SEIFA** | socioeconomic_index | Advantage/disadvantage indices | SA2 |

### Extract Health Indicator Data

Let's start with a comprehensive health analysis dataset:

```bash
# Extract multiple health datasets
ahgd-extract --source aihw --dataset disease_prevalence --years 2019-2023 --output data_raw/
ahgd-extract --source aihw --dataset health_services --years 2019-2023 --output data_raw/
ahgd-extract --source abs --dataset mortality_stats --years 2019-2023 --output data_raw/
ahgd-extract --source seifa --dataset socioeconomic_index --year 2021 --output data_raw/

# Transform and integrate the datasets
ahgd-transform --input data_raw/ --output data_processed/ --schema integrated_health
```

## Analytical Workflows

### 1. Exploratory Data Analysis

Create a comprehensive analysis notebook:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from src.utils.analysis import HealthDataAnalyser, GeographicAnalyser

# Load integrated health data
health_data = pd.read_csv('data_processed/integrated_health_data.csv')

# Basic dataset overview
print(f"Dataset shape: {health_data.shape}")
print(f"Date range: {health_data['year'].min()} - {health_data['year'].max()}")
print(f"Geographic coverage: {health_data['sa2_code'].nunique()} SA2 areas")

# Health indicator summary
health_indicators = [col for col in health_data.columns if 'rate' in col or 'prevalence' in col]
print(f"Health indicators available: {len(health_indicators)}")

# Data quality assessment
quality_summary = health_data.groupby('state')['data_quality_score'].agg(['mean', 'min', 'max'])
print("\nData quality by state:")
print(quality_summary)
```

### 2. Geographic Health Analysis

Analyse health patterns across Australian geographic areas:

```python
# Load geographic boundaries
aus_boundaries = gpd.read_file('data_processed/spatial/sa2_boundaries.shp')

# Merge health data with geographic boundaries
health_geo = aus_boundaries.merge(health_data, on='sa2_code', how='inner')

# Calculate state-level health indicators
state_health = health_data.groupby('state').agg({
    'diabetes_prevalence_rate': 'mean',
    'heart_disease_rate': 'mean',
    'mental_health_service_rate': 'mean',
    'gp_visit_rate': 'mean',
    'seifa_index': 'mean'
}).round(2)

print("State-level health indicators:")
print(state_health)

# Identify health hotspots (areas with multiple high-risk indicators)
health_risk_score = health_data.assign(
    diabetes_risk=pd.cut(health_data['diabetes_prevalence_rate'], bins=5, labels=range(1,6)),
    heart_risk=pd.cut(health_data['heart_disease_rate'], bins=5, labels=range(1,6)),
    mental_risk=pd.cut(health_data['mental_health_service_rate'], bins=5, labels=range(1,6), include_lowest=True),
    overall_risk=lambda x: (
        x['diabetes_risk'].astype(int) + 
        x['heart_risk'].astype(int) + 
        x['mental_risk'].astype(int)
    )
)

# Top 20 highest risk areas
high_risk_areas = health_risk_score.nlargest(20, 'overall_risk')[
    ['sa2_code', 'sa2_name', 'state', 'overall_risk', 
     'diabetes_prevalence_rate', 'heart_disease_rate', 'seifa_index']
]
print("\nTop 20 highest health risk areas:")
print(high_risk_areas)
```

### 3. Temporal Health Trends

Analyse health trends over time:

```python
# Trend analysis for key health indicators
trend_indicators = ['diabetes_prevalence_rate', 'heart_disease_rate', 'mental_health_service_rate']

# National trends
national_trends = health_data.groupby('year')[trend_indicators].mean()

# Plot national trends
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, indicator in enumerate(trend_indicators):
    axes[i].plot(national_trends.index, national_trends[indicator], marker='o', linewidth=2)
    axes[i].set_title(f'National {indicator.replace("_", " ").title()}')
    axes[i].set_xlabel('Year')
    axes[i].set_ylabel('Rate per 1,000')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/national_health_trends.png', dpi=300, bbox_inches='tight')
plt.show()

# State-by-state comparison
state_comparison = health_data.groupby(['state', 'year'])[trend_indicators].mean().unstack('state')

# Plot state comparisons
for indicator in trend_indicators:
    plt.figure(figsize=(12, 6))
    for state in state_comparison[indicator].columns:
        plt.plot(state_comparison.index, state_comparison[indicator][state], 
                marker='o', label=state, linewidth=2)
    
    plt.title(f'{indicator.replace("_", " ").title()} by State (2019-2023)')
    plt.xlabel('Year')
    plt.ylabel('Rate per 1,000')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'outputs/{indicator}_by_state.png', dpi=300, bbox_inches='tight')
    plt.show()
```

### 4. Socioeconomic Health Analysis

Examine the relationship between socioeconomic factors and health outcomes:

```python
# Correlation analysis between SEIFA index and health indicators
correlation_data = health_data[['seifa_index'] + trend_indicators].corr()

# Visualise correlations
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_data, annot=True, cmap='RdBu_r', center=0, 
            square=True, cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation: Socioeconomic Index vs Health Indicators')
plt.tight_layout()
plt.savefig('outputs/seifa_health_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

# Quintile analysis - divide areas by socioeconomic advantage
health_data['seifa_quintile'] = pd.qcut(health_data['seifa_index'], q=5, 
                                       labels=['Most Disadvantaged', 'Disadvantaged', 
                                              'Average', 'Advantaged', 'Most Advantaged'])

quintile_analysis = health_data.groupby('seifa_quintile')[trend_indicators].agg(['mean', 'std'])

print("Health indicators by socioeconomic quintile:")
print(quintile_analysis)

# Statistical significance testing
from scipy.stats import f_oneway

for indicator in trend_indicators:
    groups = [health_data[health_data['seifa_quintile'] == q][indicator].dropna() 
              for q in health_data['seifa_quintile'].unique()]
    f_stat, p_value = f_oneway(*groups)
    print(f"\n{indicator}: F-statistic = {f_stat:.3f}, p-value = {p_value:.6f}")
    if p_value < 0.001:
        print("*** Highly significant difference between socioeconomic groups")
    elif p_value < 0.01:
        print("** Significant difference between socioeconomic groups")
    elif p_value < 0.05:
        print("* Statistically significant difference between socioeconomic groups")
```

### 5. Interactive Visualisation

Create interactive maps and dashboards:

```python
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Interactive choropleth map
def create_health_map(data, indicator, title):
    """Create interactive choropleth map for health indicator"""
    
    # Merge with geographic data
    map_data = aus_boundaries.merge(data, on='sa2_code', how='inner')
    
    # Create base map centered on Australia
    m = folium.Map(location=[-25.2744, 133.7751], zoom_start=5)
    
    # Add choropleth layer
    folium.Choropleth(
        geo_data=map_data,
        name=title,
        data=map_data,
        columns=['sa2_code', indicator],
        key_on='feature.properties.sa2_code',
        fill_color='RdYlBu_r',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f'{indicator.replace("_", " ").title()} (per 1,000)'
    ).add_to(m)
    
    # Add popup information
    folium.GeoJson(
        map_data,
        popup=folium.GeoJsonPopup(
            fields=['sa2_name', 'state', indicator, 'seifa_index'],
            aliases=['SA2 Name', 'State', indicator.replace('_', ' ').title(), 'SEIFA Index']
        )
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    return m

# Create maps for each health indicator
for indicator in trend_indicators:
    latest_data = health_data[health_data['year'] == health_data['year'].max()]
    health_map = create_health_map(latest_data, indicator, 
                                  f'{indicator.replace("_", " ").title()} (2023)')
    health_map.save(f'outputs/{indicator}_map.html')

# Interactive dashboard with Plotly
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Diabetes Prevalence by State', 'Heart Disease Trends', 
                   'Mental Health Services', 'Socioeconomic vs Health'),
    specs=[[{"type": "bar"}, {"type": "scatter"}],
           [{"type": "box"}, {"type": "scatter"}]]
)

# Bar chart: Diabetes prevalence by state
state_diabetes = health_data[health_data['year'] == 2023].groupby('state')['diabetes_prevalence_rate'].mean()
fig.add_trace(
    go.Bar(x=state_diabetes.index, y=state_diabetes.values, name='Diabetes Rate'),
    row=1, col=1
)

# Line chart: Heart disease trends
for state in health_data['state'].unique():
    state_data = health_data[health_data['state'] == state].groupby('year')['heart_disease_rate'].mean()
    fig.add_trace(
        go.Scatter(x=state_data.index, y=state_data.values, mode='lines+markers', 
                  name=f'{state} Heart Disease'),
        row=1, col=2
    )

# Box plot: Mental health services by quintile
for quintile in health_data['seifa_quintile'].unique():
    quintile_data = health_data[health_data['seifa_quintile'] == quintile]['mental_health_service_rate']
    fig.add_trace(
        go.Box(y=quintile_data, name=quintile),
        row=2, col=1
    )

# Scatter plot: SEIFA vs combined health risk
fig.add_trace(
    go.Scatter(
        x=health_data['seifa_index'], 
        y=health_data['diabetes_prevalence_rate'] + health_data['heart_disease_rate'],
        mode='markers',
        name='Combined Health Risk',
        text=health_data['sa2_name'],
        hovertemplate='<b>%{text}</b><br>SEIFA: %{x}<br>Combined Risk: %{y}<extra></extra>'
    ),
    row=2, col=2
)

fig.update_layout(height=800, showlegend=False, title_text="Australian Health Data Dashboard")
fig.write_html('outputs/health_dashboard.html')
fig.show()
```

## Advanced Analysis Techniques

### 1. Spatial Autocorrelation

Test for geographic clustering of health outcomes:

```python
from pysal.lib import weights
from pysal.explore import esda

# Create spatial weights matrix (this requires geographic data)
# Note: This is a simplified example - actual implementation requires proper spatial data
def spatial_analysis(data, indicator):
    """Perform spatial autocorrelation analysis"""
    
    # Calculate Moran's I (measure of spatial autocorrelation)
    # This is conceptual - requires proper spatial weights
    print(f"Spatial Analysis for {indicator}:")
    print("- Moran's I: Measures spatial clustering")
    print("- LISA: Local indicators of spatial association")
    print("- Hotspot detection: Areas with similar high/low values")
    
    # Placeholder for actual spatial analysis
    # In practice, you would use:
    # w = weights.Queen.from_dataframe(geo_data)
    # moran = esda.Moran(data[indicator], w)
    # print(f"Moran's I: {moran.I:.4f}, p-value: {moran.p_norm:.4f}")

# Run spatial analysis for each indicator
for indicator in trend_indicators:
    spatial_analysis(health_data, indicator)
```

### 2. Time Series Analysis

Forecast health trends:

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

def time_series_analysis(data, indicator, state=None):
    """Perform time series analysis and forecasting"""
    
    if state:
        ts_data = data[data['state'] == state].groupby('year')[indicator].mean()
        title_suffix = f" ({state})"
    else:
        ts_data = data.groupby('year')[indicator].mean()
        title_suffix = " (National)"
    
    # Seasonal decomposition
    if len(ts_data) >= 4:  # Need at least 4 periods
        decomposition = seasonal_decompose(ts_data, model='additive', period=2)
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        decomposition.observed.plot(ax=axes[0], title=f'Original{title_suffix}')
        decomposition.trend.plot(ax=axes[1], title='Trend')
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
        decomposition.resid.plot(ax=axes[3], title='Residuals')
        
        plt.tight_layout()
        plt.savefig(f'outputs/{indicator}_{state or "national"}_decomposition.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    # ARIMA forecasting
    try:
        model = ARIMA(ts_data, order=(1, 1, 1))
        fitted_model = model.fit()
        
        # Forecast next 2 years
        forecast = fitted_model.forecast(steps=2)
        forecast_ci = fitted_model.get_forecast(steps=2).conf_int()
        
        print(f"\n{indicator}{title_suffix} Forecast:")
        print(f"2024: {forecast.iloc[0]:.2f}")
        print(f"2025: {forecast.iloc[1]:.2f}")
        
        # Plot forecast
        plt.figure(figsize=(10, 6))
        plt.plot(ts_data.index, ts_data.values, 'o-', label='Historical')
        plt.plot([2024, 2025], forecast.values, 's-', color='red', label='Forecast')
        plt.fill_between([2024, 2025], 
                        forecast_ci.iloc[:, 0], 
                        forecast_ci.iloc[:, 1], 
                        color='red', alpha=0.2, label='95% Confidence Interval')
        plt.title(f'{indicator.replace("_", " ").title()} Forecast{title_suffix}')
        plt.xlabel('Year')
        plt.ylabel('Rate per 1,000')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'outputs/{indicator}_{state or "national"}_forecast.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Could not fit ARIMA model for {indicator}: {e}")

# Perform time series analysis
for indicator in trend_indicators:
    time_series_analysis(health_data, indicator)
    
    # State-specific analysis for major states
    for state in ['NSW', 'VIC', 'QLD']:
        time_series_analysis(health_data, indicator, state)
```

### 3. Health Equity Analysis

Assess health disparities across different population groups:

```python
def health_equity_analysis(data):
    """Analyse health equity across socioeconomic groups"""
    
    # Calculate health equity metrics
    equity_metrics = {}
    
    for indicator in trend_indicators:
        # Rate ratio (highest vs lowest quintile)
        quintile_means = data.groupby('seifa_quintile')[indicator].mean()
        rate_ratio = quintile_means.max() / quintile_means.min()
        
        # Rate difference (absolute difference)
        rate_difference = quintile_means.max() - quintile_means.min()
        
        # Population attributable fraction
        overall_rate = data[indicator].mean()
        disadvantaged_rate = data[data['seifa_quintile'] == 'Most Disadvantaged'][indicator].mean()
        paf = (disadvantaged_rate - overall_rate) / disadvantaged_rate * 100
        
        equity_metrics[indicator] = {
            'rate_ratio': rate_ratio,
            'rate_difference': rate_difference,
            'paf': paf
        }
    
    # Create equity dashboard
    equity_df = pd.DataFrame(equity_metrics).T
    
    print("Health Equity Metrics:")
    print("Rate Ratio: Higher values indicate greater inequality")
    print("Rate Difference: Absolute difference between groups")
    print("PAF: Population Attributable Fraction (% difference)")
    print("\n", equity_df.round(2))
    
    # Visualise equity metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    equity_df['rate_ratio'].plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Rate Ratio (Inequality Measure)')
    axes[0].set_ylabel('Ratio')
    axes[0].tick_params(axis='x', rotation=45)
    
    equity_df['rate_difference'].plot(kind='bar', ax=axes[1], color='lightcoral')
    axes[1].set_title('Rate Difference (Absolute)')
    axes[1].set_ylabel('Difference per 1,000')
    axes[1].tick_params(axis='x', rotation=45)
    
    equity_df['paf'].plot(kind='bar', ax=axes[2], color='lightgreen')
    axes[2].set_title('Population Attributable Fraction')
    axes[2].set_ylabel('Percentage')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('outputs/health_equity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return equity_df

equity_results = health_equity_analysis(health_data)
```

## Reporting and Communication

### 1. Automated Report Generation

Create standardised health reports:

```python
def generate_health_report(data, output_file='health_report.html'):
    """Generate comprehensive health analysis report"""
    
    from datetime import datetime
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Australian Health Data Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
            .metric {{ background-color: #e9ecef; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .alert {{ background-color: #fff3cd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Australian Health Data Analysis Report</h1>
            <p>Generated: {datetime.now().strftime('%d %B %Y at %H:%M')}</p>
            <p>Data Period: {data['year'].min()} - {data['year'].max()}</p>
            <p>Geographic Coverage: {data['sa2_code'].nunique():,} SA2 Areas</p>
        </div>
        
        <h2>Executive Summary</h2>
        <div class="metric">
            <h3>Key Findings</h3>
            <ul>
                <li>National diabetes prevalence: {data['diabetes_prevalence_rate'].mean():.1f} per 1,000 population</li>
                <li>Heart disease rate: {data['heart_disease_rate'].mean():.1f} per 1,000 population</li>
                <li>Mental health service utilisation: {data['mental_health_service_rate'].mean():.1f} per 1,000 population</li>
                <li>Data quality score: {data['data_quality_score'].mean():.2f} (out of 1.0)</li>
            </ul>
        </div>
        
        <h2>State Comparison</h2>
        {data.groupby('state')[trend_indicators + ['seifa_index']].mean().round(2).to_html()}
        
        <h2>Recommendations</h2>
        <div class="alert">
            <h3>Priority Areas for Intervention</h3>
            <p>Based on the analysis, areas with high disease rates and low socioeconomic indices should be prioritised for health interventions.</p>
        </div>
        
        <h2>Data Quality Notes</h2>
        <p>This analysis is based on official Australian health data sources. All rates are age-standardised where appropriate.</p>
        
    </body>
    </html>
    """
    
    with open(f'outputs/{output_file}', 'w') as f:
        f.write(html_template)
    
    print(f"Report generated: outputs/{output_file}")

# Generate report
generate_health_report(health_data)
```

### 2. Export Analysis Results

Prepare data for external use:

```bash
# Export processed analysis data
ahgd-loader --input data_processed/ --output analysis_exports/ --formats csv,excel,json --include-metadata

# Create analysis summary tables
python -c "
import pandas as pd
health_data = pd.read_csv('data_processed/integrated_health_data.csv')

# State summary
state_summary = health_data.groupby('state').agg({
    'diabetes_prevalence_rate': ['mean', 'std'],
    'heart_disease_rate': ['mean', 'std'],
    'mental_health_service_rate': ['mean', 'std'],
    'seifa_index': 'mean',
    'data_quality_score': 'mean'
}).round(2)

state_summary.to_csv('analysis_exports/state_health_summary.csv')
print('State summary exported to analysis_exports/state_health_summary.csv')
"
```

## Best Practices for Health Data Analysis

### 1. Data Quality Considerations

- **Always check data quality scores** before analysis
- **Validate geographic codes** against official boundaries
- **Handle missing data appropriately** (don't ignore it)
- **Consider temporal consistency** across datasets
- **Document all assumptions** and limitations

### 2. Statistical Considerations

- **Use appropriate denominators** (population, age-adjusted rates)
- **Account for multiple comparisons** when testing hypotheses
- **Consider spatial autocorrelation** in geographic analysis
- **Use confidence intervals** for all estimates
- **Be cautious with small area analysis** (check cell counts)

### 3. Ethical Considerations

- **Protect individual privacy** (no re-identification attempts)
- **Consider health equity implications** in all analyses
- **Avoid stigmatising interpretations** of findings
- **Acknowledge data limitations** clearly
- **Consider vulnerable populations** in recommendations

## Troubleshooting Analysis Issues

### Common Problems and Solutions

**Problem**: Geographic codes don't match between datasets
```python
# Check for mismatched codes
geo_codes = pd.read_csv('data_processed/spatial/sa2_boundaries.csv')['sa2_code']
data_codes = health_data['sa2_code']
missing_codes = set(data_codes) - set(geo_codes)
print(f"Mismatched codes: {len(missing_codes)}")
```

**Problem**: Temporal data inconsistencies
```python
# Check for data gaps
year_coverage = health_data.groupby(['year', 'state']).size().unstack(fill_value=0)
print("Year coverage by state:")
print(year_coverage)
```

**Problem**: Statistical tests failing
```python
# Check data distribution and sample sizes
for indicator in trend_indicators:
    print(f"\n{indicator}:")
    print(f"  Mean: {health_data[indicator].mean():.2f}")
    print(f"  Std: {health_data[indicator].std():.2f}")
    print(f"  Missing: {health_data[indicator].isna().sum()}")
    print(f"  Min sample size: {health_data.groupby('state')[indicator].count().min()}")
```

## Next Steps

You've now mastered Australian health data analysis with AHGD! Consider exploring:

- **[Researcher Guide](RESEARCHER_GUIDE.md)**: Advanced statistical methods and research workflows
- **[Developer Tutorial](DEVELOPER_TUTORIAL.md)**: Extending AHGD with custom analysis functions
- **Machine Learning Applications**: Predictive modeling with health data
- **Real-time Analysis**: Setting up automated monitoring systems

## Resources

- **Australian Bureau of Statistics**: [Health Statistics](https://www.abs.gov.au/statistics/health)
- **AIHW Data Portal**: [Australian Institute of Health and Welfare](https://www.aihw.gov.au/reports-data)
- **SEIFA Documentation**: [Socio-Economic Indexes](https://www.abs.gov.au/ausstats/abs@.nsf/mf/2033.0.55.001)
- **SA2 Geography**: [Australian Statistical Geography Standard](https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026)

Happy analysing! 📊🇦🇺