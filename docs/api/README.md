# AHGD V3 API Documentation Hub
### High-Performance Health Analytics API

Welcome to the comprehensive API documentation for the Australian Health Geography Data (AHGD) V3 platform. Our modern API delivers **10-100x performance improvements** through Polars, DuckDB, and Parquet-first architecture.

---

## 🚀 Quick Start

### Base URL
```
Production: https://api.ahgd.dev/v1
Development: http://localhost:8000/v1  
```

### Authentication
```bash
# Get API key from dashboard
curl -H "X-API-Key: your-api-key" https://api.ahgd.dev/v1/health/status
```

### Example Request
```bash
# Get SA1 health profile (sub-second response)
curl -H "X-API-Key: your-key" \
     https://api.ahgd.dev/v1/health/sa1/101011001
```

---

## 📊 API Endpoints Reference

### 🏥 Health Data API
High-performance health analytics with SA1-level granularity.

| Endpoint | Method | Description | Performance |
|----------|--------|-------------|-------------|
| `/health/sa1/{code}` | GET | Get comprehensive health profile | <100ms |
| `/health/search` | POST | Advanced health indicator search | <200ms |
| `/health/compare` | POST | Compare health metrics across areas | <300ms |
| `/health/trends` | GET | Temporal health trend analysis | <500ms |

**[📖 Health API Documentation →](health-api.md)**

### 🗺️ Geographic API  
Lightning-fast geographic data with 61,845 SA1 areas.

| Endpoint | Method | Description | Performance |
|----------|--------|-------------|-------------|
| `/geo/sa1/{code}` | GET | Get SA1 area details | <50ms |
| `/geo/boundaries` | GET | Get area boundaries (GeoJSON) | <100ms |
| `/geo/nearby` | POST | Find nearby areas by distance | <150ms |
| `/geo/hierarchy` | GET | Get geographic hierarchy | <75ms |

**[📖 Geographic API Documentation →](geographic-api.md)**

### 📈 Analytics API
Advanced analytics powered by DuckDB and Polars.

| Endpoint | Method | Description | Performance |
|----------|--------|-------------|-------------|
| `/analytics/correlations` | POST | Health correlation analysis | <400ms |
| `/analytics/clustering` | POST | Geographic health clustering | <600ms |
| `/analytics/risk-assessment` | POST | Population health risk scoring | <350ms |
| `/analytics/reports` | POST | Generate analytics reports | <1s |

**[📖 Analytics API Documentation →](analytics-api.md)**

### 🔧 System API
System monitoring and performance metrics.

| Endpoint | Method | Description | Response |
|----------|--------|-------------|----------|
| `/system/health` | GET | System health check | Health status |
| `/system/performance` | GET | Performance metrics | Real-time stats |
| `/system/version` | GET | API version info | Version details |

**[📖 System API Documentation →](system-api.md)**

---

## 🎯 Data Models

### SA1 Health Profile
```json
{
  "sa1_code": "101011001",
  "area_name": "Sydney - Circular Quay",
  "state": "NSW",
  "population": 623,
  "health_indicators": {
    "diabetes_prevalence": 4.2,
    "cardiovascular_risk": "LOW", 
    "mental_health_services_rate": 45.7,
    "life_expectancy": 83.2
  },
  "socioeconomic": {
    "seifa_irsad": 1094,
    "seifa_rank": 85,
    "disadvantage_level": "LOW"
  },
  "geographic": {
    "latitude": -33.8568,
    "longitude": 151.2153,
    "area_sqkm": 0.15
  },
  "data_quality": {
    "completeness": 0.94,
    "last_updated": "2024-08-31T10:30:00Z",
    "source_reliability": "HIGH"
  }
}
```

### Search Request
```json
{
  "filters": {
    "diabetes_rate": {"min": 3.0, "max": 8.0},
    "state": ["NSW", "VIC"],
    "population": {"min": 500}
  },
  "sort_by": "diabetes_rate",
  "limit": 100,
  "format": "json"
}
```

### Analytics Report
```json
{
  "report_id": "rpt_health_analysis_2024",
  "areas_analyzed": 1247,
  "correlation_matrix": {
    "diabetes_seifa": -0.73,
    "mental_health_income": -0.61,
    "life_expectancy_education": 0.82
  },
  "risk_areas": [
    {
      "sa1_code": "301011234",
      "risk_score": 8.2,
      "primary_concerns": ["diabetes", "cardiovascular"]
    }
  ],
  "processing_time_ms": 340,
  "cached": true
}
```

---

## ⚡ Performance Features

### Lightning-Fast Responses
- **Sub-second queries** on multi-million record datasets
- **Intelligent caching** with Parquet storage
- **Parallel processing** across multiple cores
- **Lazy evaluation** for memory efficiency

### High Availability
- **99.9% uptime** SLA
- **Auto-scaling** based on demand  
- **Load balancing** across regions
- **Circuit breakers** for fault tolerance

### Rate Limiting
```
Free Tier:     1,000 requests/hour
Professional:  10,000 requests/hour  
Enterprise:    Unlimited
```

---

## 🔐 Authentication & Security

### API Key Authentication
```bash
# Include in header
X-API-Key: ahgd_v3_your_api_key_here

# Or as query parameter
?api_key=ahgd_v3_your_api_key_here
```

### OAuth2 (Enterprise)
```bash
# Get access token
curl -X POST https://api.ahgd.dev/oauth/token \
  -d "grant_type=client_credentials" \
  -d "client_id=your_client_id" \
  -d "client_secret=your_secret"

# Use token
curl -H "Authorization: Bearer your_access_token" \
     https://api.ahgd.dev/v1/health/sa1/101011001
```

### Security Features
- **HTTPS encryption** for all endpoints
- **Input validation** with Pydantic V2
- **Rate limiting** and DDoS protection
- **Audit logging** for all API calls
- **Data privacy** compliance (Australian Privacy Principles)

---

## 📊 Data Sources & Quality

### Government Data Sources
- **ABS Census 2021**: Demographics at SA1 level
- **AIHW Health Data**: Mortality and morbidity statistics  
- **PHIDU Health Atlas**: Population health indicators
- **MBS/PBS Data**: Healthcare utilization (modeled to SA1)

### Data Quality Metrics
| Metric | Score | Notes |
|--------|-------|-------|
| **Completeness** | 94.2% | Average across all datasets |
| **Accuracy** | 98.7% | Validated against source systems |
| **Currency** | 2021-2024 | Most recent available data |
| **Consistency** | 96.1% | Standardized to SA1 framework |

### Update Frequency
- **Health indicators**: Annual updates
- **Census data**: 5-year cycle (next: 2026)
- **Healthcare utilization**: Quarterly updates
- **Geographic boundaries**: As needed (stable)

---

## 🛠️ Developer Tools

### Interactive API Explorer
**[🚀 Try the API Live →](https://api.ahgd.dev/docs)**
- Interactive Swagger/OpenAPI documentation
- Test endpoints with real data
- Code generation for multiple languages
- Authentication testing

### SDKs & Libraries

#### Python SDK
```python
pip install ahgd-python-sdk

from ahgd import HealthAPI

client = HealthAPI(api_key="your-key")
profile = client.get_sa1_health_profile("101011001")
print(f"Diabetes rate: {profile.diabetes_prevalence}%")
```

#### R Package  
```r
# Install from GitHub
devtools::install_github("massimoraso/ahgd-r-sdk")

library(ahgd)
client <- ahgd_client("your-api-key")
profile <- get_sa1_health_profile(client, "101011001")
```

#### JavaScript SDK
```javascript
npm install @ahgd/js-sdk

import { HealthAPI } from '@ahgd/js-sdk';

const client = new HealthAPI('your-api-key');
const profile = await client.getHealthProfile('101011001');
console.log(`Life expectancy: ${profile.life_expectancy}`);
```

### Code Examples
**[📚 Complete Code Examples →](code-examples.md)**
- Python data analysis workflows
- R statistical modeling examples  
- JavaScript dashboard integration
- Jupyter notebook tutorials

---

## 📈 Use Cases & Examples

### 🏥 Public Health Analysis
```python
# Find areas with high diabetes rates and low service access
import requests

search_payload = {
    "filters": {
        "diabetes_rate": {"min": 8.0},
        "mental_health_services_rate": {"max": 20.0}
    },
    "sort_by": "diabetes_rate",
    "limit": 50
}

response = requests.post(
    "https://api.ahgd.dev/v1/health/search",
    json=search_payload,
    headers={"X-API-Key": "your-key"}
)

high_need_areas = response.json()
```

### 🏛️ Government Planning
```python
# Identify underserved areas for new health facilities
correlation_analysis = client.analytics.correlations({
    "indicators": ["healthcare_access", "population_density", "age_65_plus"],
    "geographic_scope": {"state": "NSW"},
    "method": "pearson"
})

# Find optimal locations based on multiple criteria
optimal_locations = client.analytics.facility_planning({
    "service_type": "GP_clinic",
    "population_catchment": 5000,
    "max_travel_time_minutes": 20
})
```

### 🔬 Research Applications
```r
# Health equity analysis
library(ahgd)
library(ggplot2)

# Get health data for specific regions
health_data <- get_health_indicators(
  client,
  areas = get_areas_by_state(client, "VIC"),
  indicators = c("diabetes", "mental_health", "life_expectancy")
)

# Analyze relationship with socioeconomic factors
model <- lm(life_expectancy ~ seifa_irsad + diabetes_rate, data = health_data)
summary(model)
```

---

## 🚨 Error Handling

### HTTP Status Codes
| Code | Status | Description |
|------|--------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid parameters |
| 401 | Unauthorized | Invalid API key |
| 404 | Not Found | Resource not found |
| 429 | Rate Limited | Too many requests |
| 500 | Server Error | Internal error |

### Error Response Format
```json
{
  "error": {
    "code": "INVALID_SA1_CODE",
    "message": "SA1 code '999999999' is not valid",
    "details": {
      "parameter": "sa1_code",
      "expected_format": "11-digit numeric string",
      "suggestion": "Use /geo/search to find valid codes"
    },
    "request_id": "req_12345abcde",
    "timestamp": "2024-08-31T10:30:00Z"
  }
}
```

### Retry Guidelines
```python
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure automatic retries
retry_strategy = Retry(
    total=3,
    status_forcelist=[429, 500, 502, 503, 504],
    backoff_factor=1
)

adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
```

---

## 📚 Additional Resources

### Documentation
- **[🚀 Getting Started Guide](../guides/getting-started.md)**
- **[🎯 SA1 Analysis Tutorial](../guides/sa1-analysis.md)**
- **[🏥 Health Analytics Cookbook](../guides/health-analytics.md)**
- **[📊 Data Dictionary](../data-dictionary/data_dictionary.md)**

### Community & Support
- **[💬 GitHub Discussions](https://github.com/massimoraso/AHGD/discussions)**
- **[🐛 Bug Reports](https://github.com/massimoraso/AHGD/issues)**
- **[📧 Email Support](mailto:support@ahgd.dev)**
- **[📖 Developer Blog](https://blog.ahgd.dev)**

### Changelog & Updates
- **[📝 API Changelog](changelog.md)**
- **[🔔 Breaking Changes](breaking-changes.md)**  
- **[🆕 What's New](whats-new.md)**
- **[🗺️ Roadmap](roadmap.md)**

---

## 🎯 Performance Benchmarks

### Response Times (95th percentile)
```
GET /health/sa1/{code}           <100ms
POST /health/search              <200ms  
GET /geo/boundaries              <150ms
POST /analytics/correlations     <400ms
```

### Throughput
```
Concurrent users:     50+
Requests per second:  1000+
Data processing:      10-100x faster than pandas
Memory efficiency:    75% reduction vs legacy
```

### Availability
```
Uptime SLA:          99.9%
Response success:    99.95%
Error rate:          <0.05%
```

---

**🚀 Ready to build amazing health analytics applications? [Get your API key →](https://dashboard.ahgd.dev/signup)**

---

*Last updated: August 2024 • API Version: 3.0.0 • Built with ❤️ for Australian health research*