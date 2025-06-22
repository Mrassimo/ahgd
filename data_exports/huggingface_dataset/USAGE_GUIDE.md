# Australian Health and Geographic Data (AHGD) - Usage Guide

## Quick Start

### Loading the Dataset

#### Using Pandas (CSV)
```python
import pandas as pd

# Load the CSV version
df = pd.read_csv('ahgd_data.csv')
print(f"Dataset shape: {df.shape}")
```

#### Using PyArrow (Parquet)
```python
import pandas as pd

# Load the Parquet version (recommended for large datasets)
df = pd.read_parquet('ahgd_data.parquet')
print(f"Dataset shape: {df.shape}")
```

#### Using GeoPandas (GeoJSON)
```python
import geopandas as gpd

# Load the GeoJSON version for spatial analysis
gdf = gpd.read_file('ahgd_data.geojson')
print(f"Geographic dataset shape: {gdf.shape}")
```

#### Using JSON
```python
import json
import pandas as pd

# Load the JSON version
with open('ahgd_data.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data['data'])
metadata = data['metadata']
```

## Available Formats

| Format | File Size | Recommended For | Description |
|--------|-----------|-----------------|-------------|
| PARQUET | 0.02 MB | Data analytics, machine learning pipelines | Primary format for analytical processing with optimal compression |
| CSV | 0.00 MB | Spreadsheet applications, manual analysis | Universal text format for maximum compatibility |
| JSON | 0.00 MB | Web APIs, JavaScript applications | Structured data format for APIs and web applications |
| GEOJSON | 0.00 MB | GIS applications, spatial analysis | Geographic data format with geometry information for GIS |


## Data Dictionary

| Column Name | Description | Data Type | Example Values |
|-------------|-------------|-----------|----------------|
| geographic_id | SA2 Geographic Identifier | string | "101021001" |
| geographic_name | SA2 Area Name | string | "Sydney - Haymarket - The Rocks" |
| state_name | State/Territory Name | string | "New South Wales" |
| life_expectancy_years | Life Expectancy (Years) | float | 82.5 |
| smoking_prevalence_percent | Smoking Prevalence (%) | float | 14.2 |
| obesity_prevalence_percent | Obesity Prevalence (%) | float | 31.8 |
| avg_temp_max | Average Maximum Temperature (°C) | float | 25.5 |
| total_rainfall | Total Rainfall (mm) | float | 1200.0 |

## Example Analyses

### Basic Statistics
```python
# Get summary statistics
print(df.describe())

# Check data coverage
print(f"States covered: {df['state_name'].unique()}")
print(f"SA2 areas: {df['geographic_id'].nunique()}")
```

### Health Analysis
```python
# Life expectancy by state
life_exp_by_state = df.groupby('state_name')['life_expectancy_years'].mean()
print(life_exp_by_state)

# Correlation between environmental and health factors
corr_matrix = df[['life_expectancy_years', 'avg_temp_max', 'total_rainfall']].corr()
print(corr_matrix)
```

### Spatial Analysis (with GeoPandas)
```python
import matplotlib.pyplot as plt

# Plot life expectancy by geographic area
fig, ax = plt.subplots(figsize=(12, 8))
gdf.plot(column='life_expectancy_years', 
         cmap='viridis', 
         legend=True,
         ax=ax)
ax.set_title('Life Expectancy by SA2 Area')
plt.show()
```

## Data Quality

- **Completeness**: 98.5% complete across all indicators
- **Validation**: All records pass geographic and statistical validation
- **Update Frequency**: Annual updates (reference year 2021)

## Support and Issues

For questions about this dataset:
1. Check the data dictionary and examples above
2. Review the validation reports in the documentation
3. Refer to the original data source documentation

## Attribution Requirements

When using this dataset, please cite:
- The original data sources (AIHW, ABS, BOM)
- This integrated dataset
- Maintain the CC BY 4.0 license terms

## Legal and Ethical Considerations

- Data is aggregated at SA2 level to protect privacy
- No individual-level information is included
- Use should comply with ethical research practices
- Commercial use is permitted under CC BY 4.0
