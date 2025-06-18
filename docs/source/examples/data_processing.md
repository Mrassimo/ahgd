
# Data Processing Examples

## AIHW Data Processing

```python
from src.dashboard.data.processors import HealthDataProcessor
import pandas as pd

# Initialize processor
processor = HealthDataProcessor()

# Load raw AIHW data
raw_data = pd.read_csv("data/raw/health/aihw_mort_table1_2025.csv")

# Clean and standardise data
cleaned_data = processor.clean_mortality_data(raw_data)

# Calculate health indicators
health_indicators = processor.calculate_health_indicators(cleaned_data)

# Aggregate by geographic region
regional_data = processor.aggregate_by_region(health_indicators, "SA2")
```

## Geographic Data Processing

```python
from src.dashboard.data.processors import GeographicProcessor
import geopandas as gpd

# Initialize processor
geo_processor = GeographicProcessor()

# Load boundary data
boundaries = gpd.read_file("data/raw/geographic/SA2_boundaries.shp")

# Simplify geometries for web display
simplified = geo_processor.simplify_geometries(
    boundaries, 
    tolerance=0.001
)

# Reproject to Web Mercator
web_ready = geo_processor.reproject_data(
    simplified, 
    target_crs="EPSG:3857"
)
```
