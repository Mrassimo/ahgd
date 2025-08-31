# Health Data API
### High-Performance Health Analytics

The Health Data API provides lightning-fast access to comprehensive health indicators at SA1 level (61,845 areas) with sub-second response times powered by Polars and DuckDB.

---

## 🏥 Core Endpoints

### Get SA1 Health Profile
Get comprehensive health indicators for a specific SA1 area.

```http
GET /v1/health/sa1/{sa1_code}
```

**Parameters:**
- `sa1_code` (required): 11-digit SA1 area code

**Response Time:** <100ms

**Example:**
```bash
curl -H "X-API-Key: your-key" \
     https://api.ahgd.dev/v1/health/sa1/101011001
```

**Response:**
```json
{
  "sa1_code": "101011001",
  "area_name": "Sydney - Circular Quay",
  "state": "NSW",
  "population": 623,
  "health_indicators": {
    "chronic_disease": {
      "diabetes_prevalence": 4.2,
      "diabetes_rank_national": 2847,
      "cardiovascular_disease_rate": 12.8,
      "cancer_incidence_rate": 89.3,
      "mental_health_conditions": 18.5
    },
    "mortality": {
      "life_expectancy": 83.2,
      "age_standardised_death_rate": 245.7,
      "leading_cause": "cardiovascular_disease",
      "premature_mortality_rate": 12.4
    },
    "healthcare_utilization": {
      "gp_services_per_1000": 342.8,
      "specialist_services_per_1000": 127.4,
      "mental_health_services_per_1000": 45.7,
      "pharmaceutical_costs_avg": 847.20
    }
  },
  "risk_assessment": {
    "overall_health_score": 7.8,
    "risk_level": "LOW",
    "priority_interventions": [
      "mental_health_services",
      "preventive_care"
    ]
  },
  "data_quality": {
    "completeness": 0.94,
    "last_updated": "2024-08-31T10:30:00Z",
    "sources": ["aihw", "phidu", "abs"]
  }
}
```

### Advanced Health Search
Search for areas based on multiple health criteria with high-performance filtering.

```http
POST /v1/health/search
```

**Response Time:** <200ms

**Request Body:**
```json
{
  "filters": {
    "diabetes_rate": {"min": 3.0, "max": 8.0},
    "life_expectancy": {"min": 80.0},
    "state": ["NSW", "VIC", "QLD"],
    "population": {"min": 500},
    "seifa_disadvantage": {"max": 5}
  },
  "sort_by": "diabetes_rate",
  "sort_order": "desc",
  "limit": 100,
  "offset": 0,
  "include_fields": [
    "basic_info",
    "health_indicators",
    "socioeconomic"
  ]
}
```

**Response:**
```json
{
  "total_results": 1247,
  "page_info": {
    "limit": 100,
    "offset": 0,
    "has_more": true
  },
  "results": [
    {
      "sa1_code": "201031245",
      "area_name": "Melbourne - Docklands",
      "state": "VIC",
      "diabetes_rate": 7.8,
      "life_expectancy": 81.4,
      "health_score": 6.2
    }
  ],
  "processing_time_ms": 145,
  "cached": false
}
```

### Compare Health Metrics
Compare health indicators across multiple SA1 areas.

```http
POST /v1/health/compare
```

**Response Time:** <300ms

**Request Body:**
```json
{
  "areas": [
    "101011001",
    "201031245",
    "301051289"
  ],
  "indicators": [
    "diabetes_prevalence",
    "life_expectancy",
    "mental_health_services_rate"
  ],
  "comparison_type": "absolute"
}
```

**Response:**
```json
{
  "comparison_id": "cmp_2024_health_analysis",
  "areas_compared": 3,
  "indicators": ["diabetes_prevalence", "life_expectancy", "mental_health_services_rate"],
  "results": {
    "101011001": {
      "area_name": "Sydney - Circular Quay",
      "diabetes_prevalence": 4.2,
      "life_expectancy": 83.2,
      "mental_health_services_rate": 45.7
    },
    "201031245": {
      "area_name": "Melbourne - Docklands",
      "diabetes_prevalence": 7.8,
      "life_expectancy": 81.4,
      "mental_health_services_rate": 38.2
    },
    "301051289": {
      "area_name": "Brisbane - CBD",
      "diabetes_prevalence": 5.6,
      "life_expectancy": 82.1,
      "mental_health_services_rate": 42.3
    }
  },
  "statistics": {
    "diabetes_prevalence": {
      "min": 4.2,
      "max": 7.8,
      "mean": 5.87,
      "std_dev": 1.82
    }
  },
  "processing_time_ms": 267
}
```

### Health Trends Analysis
Analyze temporal trends in health indicators over time.

```http
GET /v1/health/trends?sa1_code={code}&indicators={list}&years={range}
```

**Parameters:**
- `sa1_code`: Target SA1 area
- `indicators`: Comma-separated list of health indicators
- `years`: Year range (e.g., "2019-2023")

**Response Time:** <500ms

**Example:**
```bash
curl -H "X-API-Key: your-key" \
     "https://api.ahgd.dev/v1/health/trends?sa1_code=101011001&indicators=diabetes_rate,life_expectancy&years=2019-2023"
```

**Response:**
```json
{
  "sa1_code": "101011001",
  "time_period": "2019-2023",
  "trends": {
    "diabetes_rate": {
      "2019": 3.8,
      "2020": 4.0,
      "2021": 4.1,
      "2022": 4.2,
      "2023": 4.3,
      "trend": "increasing",
      "annual_change_rate": 0.125,
      "significance": "p<0.05"
    },
    "life_expectancy": {
      "2019": 82.8,
      "2020": 82.2,
      "2021": 82.9,
      "2022": 83.1,
      "2023": 83.2,
      "trend": "stable",
      "annual_change_rate": 0.1,
      "significance": "ns"
    }
  },
  "forecast": {
    "diabetes_rate": {
      "2024": 4.4,
      "2025": 4.5,
      "confidence_interval": [4.1, 4.9]
    }
  }
}
```

---

## 📊 Health Indicators Reference

### Chronic Disease Indicators
| Indicator | Unit | Range | Source |
|-----------|------|-------|--------|
| `diabetes_prevalence` | % population | 0-20 | AIHW, PHIDU |
| `cardiovascular_disease_rate` | per 1000 | 5-50 | AIHW |
| `cancer_incidence_rate` | per 100,000 | 200-800 | Cancer registries |
| `mental_health_conditions` | % population | 5-35 | PHIDU |
| `chronic_kidney_disease` | % population | 1-15 | AIHW |

### Mortality Indicators
| Indicator | Unit | Range | Source |
|-----------|------|-------|--------|
| `life_expectancy` | years | 75-90 | ABS, AIHW |
| `age_standardised_death_rate` | per 100,000 | 200-800 | ABS |
| `premature_mortality_rate` | per 100,000 | 50-300 | AIHW |
| `infant_mortality_rate` | per 1,000 births | 2-8 | ABS |

### Healthcare Utilization
| Indicator | Unit | Range | Source |
|-----------|------|-------|--------|
| `gp_services_per_1000` | services | 100-800 | MBS |
| `specialist_services_per_1000` | services | 20-300 | MBS |
| `mental_health_services_per_1000` | services | 10-150 | MBS |
| `pharmaceutical_costs_avg` | AUD | 200-2000 | PBS |

---

## 🔍 Filtering & Search Options

### Numeric Filters
```json
{
  "diabetes_rate": {
    "min": 3.0,           // Greater than or equal
    "max": 8.0,           // Less than or equal
    "eq": 5.5,            // Exactly equal
    "ne": 0.0             // Not equal
  }
}
```

### Categorical Filters
```json
{
  "state": ["NSW", "VIC"],              // Any of these values
  "risk_level": "HIGH",                 // Exact match
  "leading_cause": {"ne": "unknown"}    // Not equal to
}
```

### Geographic Filters
```json
{
  "near": {
    "sa1_code": "101011001",
    "radius_km": 5.0
  },
  "bounding_box": {
    "north": -33.8,
    "south": -34.0,
    "east": 151.3,
    "west": 151.1
  }
}
```

### Sort Options
- `diabetes_rate`, `life_expectancy`, `population`
- `health_score`, `seifa_rank`, `area_name`
- Custom composite scoring available

---

## 📈 Performance Optimization

### Response Caching
- **Automatic caching** for frequently requested data
- **Cache TTL**: 1 hour for health indicators, 24 hours for geographic data
- **Cache keys** include all query parameters
- **Cache hit rate**: >85% for common queries

### Lazy Loading
```json
{
  "include_fields": [
    "basic_info",        // Always included
    "health_indicators", // Optional, adds ~50ms
    "socioeconomic",     // Optional, adds ~30ms
    "geographic"         // Optional, adds ~20ms
  ]
}
```

### Pagination
```json
{
  "limit": 100,     // Max 1000 per request
  "offset": 0,      // For pagination
  "cursor": "xyz"   // Alternative cursor-based pagination
}
```

---

## 🚨 Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| `INVALID_SA1_CODE` | SA1 code format invalid | Use 11-digit numeric string |
| `AREA_NOT_FOUND` | SA1 area doesn't exist | Check code with /geo/search |
| `INVALID_INDICATOR` | Health indicator not available | See indicators reference |
| `DATE_RANGE_INVALID` | Invalid year range specified | Use format "2019-2023" |
| `INSUFFICIENT_DATA` | Not enough data for analysis | Try broader criteria |

---

## 💡 Usage Examples

### Python SDK
```python
from ahgd import HealthAPI

client = HealthAPI(api_key="your-key")

# Get single area profile
profile = client.get_health_profile("101011001")
print(f"Diabetes rate: {profile.diabetes_prevalence}%")

# Search high-risk areas
high_risk = client.search_areas({
    "diabetes_rate": {"min": 8.0},
    "life_expectancy": {"max": 78.0},
    "limit": 50
})

for area in high_risk:
    print(f"{area.area_name}: {area.diabetes_rate}%")
```

### R Package
```r
library(ahgd)

client <- ahgd_client("your-api-key")

# Get health data for analysis
health_data <- get_health_indicators(
  client,
  areas = c("101011001", "201031245"),
  indicators = c("diabetes", "life_expectancy", "mental_health")
)

# Statistical analysis
correlation <- cor(health_data$diabetes_rate, health_data$seifa_rank)
```

### JavaScript/Node.js
```javascript
import { HealthAPI } from '@ahgd/js-sdk';

const client = new HealthAPI('your-api-key');

// Async health data fetching
const profile = await client.getHealthProfile('101011001');
console.log(`Health score: ${profile.health_score}/10`);

// Batch processing
const areas = ['101011001', '201031245', '301051289'];
const profiles = await Promise.all(
  areas.map(code => client.getHealthProfile(code))
);
```

---

**[← Back to API Hub](README.md)** | **[Next: Geographic API →](geographic-api.md)**

---

*Last updated: August 2024 • Powered by Polars & DuckDB for maximum performance*
