# AHGD API Quick Start Guide
### Get up and running in 5 minutes

This guide will have you making your first API calls to the Australian Health Geography Data platform in under 5 minutes.

---

## 🚀 Step 1: Get Your API Key

### Sign Up (Free)
1. Visit [dashboard.ahgd.dev](https://dashboard.ahgd.dev/signup)
2. Create your free account
3. Copy your API key from the dashboard

### Free Tier Includes:
- **1,000 requests/hour**
- **All endpoints** (health, geographic, analytics)
- **61,845 SA1 areas** of health data
- **Sub-second responses**

---

## 📋 Step 2: Your First API Call

### Test with curl
```bash
# Test the API (replace with your actual key)
curl -H "X-API-Key: ahgd_v3_your_api_key_here" \
     https://api.ahgd.dev/v1/system/health

# Expected response:
{
  "status": "healthy",
  "version": "3.0.0",
  "uptime_seconds": 2847293
}
```

### Get Health Data for Sydney CBD
```bash
curl -H "X-API-Key: your-key" \
     https://api.ahgd.dev/v1/health/sa1/101011001

# Returns comprehensive health profile including:
# - Diabetes prevalence: 4.2%
# - Life expectancy: 83.2 years  
# - Healthcare utilization rates
# - Risk assessment scores
```

---

## 🛠️ Step 3: Choose Your Development Language

### Python (Most Popular)
```python
# Install the SDK
pip install ahgd-python-sdk

# Your first health query
from ahgd import HealthAPI

client = HealthAPI(api_key="your-key")
profile = client.get_health_profile("101011001")

print(f"Area: {profile.area_name}")
print(f"Diabetes rate: {profile.diabetes_prevalence}%") 
print(f"Life expectancy: {profile.life_expectancy} years")
```

### R (Academic/Research)
```r
# Install the package
devtools::install_github("massimoraso/ahgd-r-sdk")

# Health data analysis
library(ahgd)
client <- ahgd_client("your-api-key")

# Get health data for Sydney areas
sydney_health <- get_health_indicators(
  client,
  areas = c("101011001", "101011002", "101011003"),
  indicators = c("diabetes", "life_expectancy", "mental_health")
)

# Quick analysis
summary(sydney_health)
```

### JavaScript/Node.js (Web Development)
```javascript
// Install the SDK
npm install @ahgd/js-sdk

// Health dashboard data
import { HealthAPI } from '@ahgd/js-sdk';

const client = new HealthAPI('your-api-key');

async function getDashboardData() {
  // Get multiple health profiles
  const areas = ['101011001', '201031245', '301051289'];
  const profiles = await Promise.all(
    areas.map(code => client.getHealthProfile(code))
  );
  
  console.log('Health Data Retrieved:', profiles.length);
  return profiles;
}
```

---

## 📊 Step 4: Explore the Data

### Find Areas with High Diabetes Rates
```python
# Search for areas with diabetes concerns
high_diabetes_areas = client.search_areas({
    "filters": {
        "diabetes_rate": {"min": 8.0},
        "population": {"min": 500}
    },
    "sort_by": "diabetes_rate",
    "limit": 20
})

for area in high_diabetes_areas:
    print(f"{area.area_name}: {area.diabetes_rate}% diabetes")
```

### Compare Health Across Capital Cities
```python
# Health comparison across major cities
capital_areas = {
    "Sydney CBD": "101011001",
    "Melbourne CBD": "201031245", 
    "Brisbane CBD": "301051289",
    "Perth CBD": "501071234"
}

comparison = client.compare_health_metrics({
    "areas": list(capital_areas.values()),
    "indicators": ["diabetes_prevalence", "life_expectancy"]
})

print("Capital City Health Comparison:")
for code, data in comparison.results.items():
    city = [k for k, v in capital_areas.items() if v == code][0]
    print(f"{city}: {data.life_expectancy} years, {data.diabetes_prevalence}% diabetes")
```

### Geographic Analysis
```python
# Find areas near Sydney Opera House
from ahgd import GeoAPI

geo_client = GeoAPI(api_key="your-key")

nearby_areas = geo_client.find_nearby_areas({
    "center": {"latitude": -33.8568, "longitude": 151.2153},
    "radius_km": 2.0,
    "limit": 10
})

print("Areas near Sydney Opera House:")
for area in nearby_areas:
    print(f"{area.area_name}: {area.distance_km:.1f}km away")
```

---

## 📈 Step 5: Advanced Analytics

### Health Risk Assessment
```python
from ahgd import AnalyticsAPI

analytics = AnalyticsAPI(api_key="your-key")

# Assess health risks for multiple areas
risk_assessment = analytics.risk_assessment({
    "areas": ["101011001", "101011002"],
    "risk_factors": [
        "chronic_disease_prevalence", 
        "healthcare_access",
        "socioeconomic_disadvantage"
    ]
})

for area_risk in risk_assessment.results:
    print(f"{area_risk.area_name}:")
    print(f"  Overall risk: {area_risk.overall_risk_score}/10")
    print(f"  Risk level: {area_risk.risk_level}")
```

### Correlation Analysis
```python
# Analyze health correlations
correlations = analytics.correlations({
    "indicators": [
        "diabetes_prevalence",
        "life_expectancy", 
        "seifa_disadvantage_rank"
    ],
    "geographic_scope": {"state": ["NSW", "VIC"]}
})

print("Key Health Correlations:")
matrix = correlations.correlation_matrix
print(f"Diabetes vs Life Expectancy: r = {matrix['diabetes_prevalence']['life_expectancy']:.3f}")
```

---

## 🗺️ Step 6: Create Your First Health Map

### Python with Folium
```python
import folium
from ahgd import HealthAPI, GeoAPI

health_client = HealthAPI(api_key="your-key")
geo_client = GeoAPI(api_key="your-key")

# Get health data for Sydney inner areas
sydney_areas = ["101011001", "101011002", "101011003"]
health_data = {}

for area_code in sydney_areas:
    profile = health_client.get_health_profile(area_code)
    health_data[area_code] = profile

# Get geographic boundaries
boundaries = geo_client.get_boundaries(sydney_areas, format="geojson")

# Create interactive map
m = folium.Map(location=[-33.8568, 151.2153], zoom_start=14)

def get_color(diabetes_rate):
    if diabetes_rate < 4: return 'green'
    elif diabetes_rate < 6: return 'yellow'
    elif diabetes_rate < 8: return 'orange'
    else: return 'red'

# Add health data to map
for feature in boundaries['features']:
    area_code = feature['properties']['sa1_code']
    health_profile = health_data[area_code]
    
    folium.GeoJson(
        feature,
        style_function=lambda x, rate=health_profile.diabetes_prevalence: {
            'fillColor': get_color(rate),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7
        },
        popup=f"""
        <b>{health_profile.area_name}</b><br>
        Diabetes: {health_profile.diabetes_prevalence}%<br>
        Life Expectancy: {health_profile.life_expectancy} years
        """
    ).add_to(m)

m.save('sydney_health_map.html')
print("Health map saved to sydney_health_map.html")
```

### R with ggplot2
```r
library(ahgd)
library(ggplot2)
library(sf)

client <- ahgd_client("your-api-key")
geo_client <- ahgd_geo_client("your-api-key")

# Get health data for Melbourne
melbourne_health <- get_health_indicators(
  client,
  state = "VIC",
  metro_area = "Melbourne",
  indicators = c("diabetes", "life_expectancy")
)

# Get boundaries
melbourne_boundaries <- get_boundaries(
  geo_client,
  sa1_codes = melbourne_health$sa1_code
)

# Create choropleth map
ggplot(melbourne_boundaries) +
  geom_sf(aes(fill = diabetes_rate), color = "white", size = 0.1) +
  scale_fill_viridis_c(name = "Diabetes\nRate (%)") +
  theme_void() +
  labs(title = "Diabetes Prevalence in Melbourne",
       subtitle = "SA1 Areas - 2023 Data")
```

---

## 🎯 Step 7: Common Use Cases

### Public Health Research
```python
# Research workflow: Compare health outcomes by socioeconomic status
research_data = analytics.correlations({
    "indicators": ["diabetes_prevalence", "seifa_irsad", "healthcare_access"],
    "geographic_scope": {"state": "NSW"},
    "method": "pearson"
})

# Statistical significance
for correlation, stats in research_data.significance.items():
    if stats.significant:
        print(f"{correlation}: r={stats.correlation:.3f}, p={stats.p_value:.6f}")
```

### Government Planning
```python
# Infrastructure planning: Find underserved areas
underserved = health_client.search_areas({
    "filters": {
        "healthcare_access_score": {"max": 3.0},
        "population": {"min": 1000},
        "chronic_disease_burden": {"min": 6.0}
    },
    "sort_by": "healthcare_access_score",
    "limit": 50
})

print(f"Found {len(underserved)} underserved areas needing healthcare facilities")
```

### Commercial Health Analytics
```python
# Market analysis: Healthcare service opportunities
market_analysis = analytics.clustering({
    "features": [
        "population_density",
        "healthcare_access_score", 
        "chronic_disease_prevalence"
    ],
    "num_clusters": 5,
    "geographic_scope": {"state": ["NSW", "VIC"]}
})

# Identify market opportunities
for cluster in market_analysis.clusters.values():
    if cluster.characteristics and "low healthcare access" in cluster.characteristics:
        print(f"Market opportunity: {cluster.name} - {cluster.size} areas")
```

---

## 📚 Next Steps

### Explore More Endpoints
- **[Health API](health-api.md)**: Comprehensive health indicators
- **[Geographic API](geographic-api.md)**: Spatial data and boundaries  
- **[Analytics API](analytics-api.md)**: Advanced statistical analysis
- **[System API](system-api.md)**: Monitoring and performance

### Advanced Features
- **Predictive modeling** for health outcomes
- **Time series analysis** of health trends
- **Spatial autocorrelation** analysis
- **Custom report generation**

### Get Help
- **[API Documentation Hub](README.md)**: Complete reference
- **[GitHub Discussions](https://github.com/massimoraso/AHGD/discussions)**: Community support
- **[Email Support](mailto:support@ahgd.dev)**: Direct assistance
- **[Code Examples](https://github.com/massimoraso/AHGD/tree/main/examples)**: Sample projects

---

## 🚨 Common Issues & Solutions

### API Key Issues
```bash
# Error: Invalid API key
# Solution: Check your key format
curl -H "X-API-Key: ahgd_v3_your_actual_key_here" \
     https://api.ahgd.dev/v1/system/health
```

### Rate Limiting
```python
# Error: 429 Too Many Requests
# Solution: Implement retry with backoff
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

retry_strategy = Retry(
    total=3,
    status_forcelist=[429, 500, 502, 503, 504],
    backoff_factor=1
)
```

### Large Data Requests
```python
# For large datasets, use pagination
def get_all_areas_with_high_diabetes():
    all_areas = []
    offset = 0
    limit = 100
    
    while True:
        batch = client.search_areas({
            "filters": {"diabetes_rate": {"min": 8.0}},
            "limit": limit,
            "offset": offset
        })
        
        if not batch.results:
            break
            
        all_areas.extend(batch.results)
        offset += limit
        
    return all_areas
```

---

**🎉 Congratulations! You're now ready to build amazing health analytics applications with the AHGD API.**

**[← Back to API Hub](README.md)** | **[Explore Examples →](../examples/)**

---

*Last updated: August 2024 • Get started in minutes with the world's fastest health geography API*