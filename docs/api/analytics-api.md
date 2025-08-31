# Analytics API
### Advanced Health Analytics & Machine Learning

The Analytics API provides sophisticated health analytics powered by DuckDB and Polars, delivering complex statistical analysis, machine learning insights, and predictive modeling with sub-second performance.

---

## 📊 Core Analytics Endpoints

### Health Correlation Analysis
Analyze correlations between health indicators and socioeconomic factors.

```http
POST /v1/analytics/correlations
```

**Response Time:** <400ms

**Request Body:**
```json
{
  "indicators": [
    "diabetes_prevalence",
    "life_expectancy",
    "seifa_irsad",
    "mental_health_services_rate"
  ],
  "geographic_scope": {
    "state": ["NSW", "VIC"],
    "population_min": 500
  },
  "method": "pearson",
  "significance_level": 0.05
}
```

**Response:**
```json
{
  "correlation_analysis": {
    "method": "pearson",
    "sample_size": 12847,
    "matrix": {
      "diabetes_prevalence": {
        "life_expectancy": -0.734,
        "seifa_irsad": -0.681,
        "mental_health_services_rate": 0.342
      },
      "life_expectancy": {
        "seifa_irsad": 0.792,
        "mental_health_services_rate": -0.156
      },
      "seifa_irsad": {
        "mental_health_services_rate": -0.289
      }
    },
    "significance": {
      "diabetes_prevalence_life_expectancy": {
        "p_value": 0.000001,
        "significant": true,
        "confidence_interval": [-0.756, -0.712]
      }
    }
  },
  "insights": [
    "Strong negative correlation between diabetes and life expectancy (r=-0.734, p<0.001)",
    "Socioeconomic advantage strongly correlates with better health outcomes"
  ],
  "processing_time_ms": 342
}
```

### Health Risk Clustering
Identify clusters of areas with similar health risk profiles using machine learning.

```http
POST /v1/analytics/clustering
```

**Response Time:** <600ms

**Request Body:**
```json
{
  "features": [
    "diabetes_prevalence",
    "cardiovascular_disease_rate",
    "mental_health_conditions",
    "life_expectancy",
    "seifa_disadvantage_rank"
  ],
  "algorithm": "kmeans",
  "num_clusters": 5,
  "geographic_scope": {
    "state": "NSW"
  },
  "standardize_features": true
}
```

**Response:**
```json
{
  "clustering_analysis": {
    "algorithm": "kmeans",
    "num_clusters": 5,
    "areas_analyzed": 19368,
    "silhouette_score": 0.67,
    "clusters": {
      "cluster_0": {
        "name": "Very High Risk",
        "size": 2847,
        "percentage": 14.7,
        "centroid": {
          "diabetes_prevalence": 12.3,
          "life_expectancy": 76.8,
          "seifa_disadvantage_rank": 8.2
        },
        "characteristics": [
          "Highest diabetes rates",
          "Lowest life expectancy",
          "Most socioeconomically disadvantaged"
        ]
      },
      "cluster_1": {
        "name": "High Risk",
        "size": 3456,
        "percentage": 17.8,
        "centroid": {
          "diabetes_prevalence": 8.7,
          "life_expectancy": 79.2,
          "seifa_disadvantage_rank": 6.4
        }
      }
    },
    "area_assignments": [
      {
        "sa1_code": "101234567",
        "cluster_id": 0,
        "cluster_name": "Very High Risk",
        "distance_to_centroid": 0.23
      }
    ]
  },
  "recommendations": [
    "Target preventive interventions in Cluster 0 areas",
    "Focus on diabetes prevention programs in high-risk clusters"
  ],
  "processing_time_ms": 567
}
```

### Population Health Risk Assessment
Calculate comprehensive health risk scores for areas or populations.

```http
POST /v1/analytics/risk-assessment
```

**Response Time:** <350ms

**Request Body:**
```json
{
  "areas": ["101011001", "201031245", "301051289"],
  "risk_factors": [
    "chronic_disease_prevalence",
    "healthcare_access",
    "socioeconomic_disadvantage",
    "environmental_factors"
  ],
  "weighting": {
    "chronic_disease_prevalence": 0.4,
    "healthcare_access": 0.3,
    "socioeconomic_disadvantage": 0.2,
    "environmental_factors": 0.1
  },
  "benchmark": "national_average"
}
```

**Response:**
```json
{
  "risk_assessment": {
    "benchmark": "national_average",
    "total_population": 2971,
    "results": [
      {
        "sa1_code": "101011001",
        "area_name": "Sydney - Circular Quay",
        "overall_risk_score": 3.2,
        "risk_level": "LOW",
        "risk_factors": {
          "chronic_disease_prevalence": {
            "score": 2.8,
            "percentile": 25,
            "contribution": 1.12
          },
          "healthcare_access": {
            "score": 8.9,
            "percentile": 95,
            "contribution": 2.67
          },
          "socioeconomic_disadvantage": {
            "score": 1.4,
            "percentile": 15,
            "contribution": 0.28
          }
        },
        "priority_interventions": [
          "Maintain excellent healthcare access",
          "Monitor diabetes prevention programs"
        ],
        "peer_comparison": {
          "similar_areas": ["201031245", "501071389"],
          "ranking": "Top 20%"
        }
      }
    ]
  },
  "population_summary": {
    "average_risk_score": 4.7,
    "high_risk_population": 847,
    "priority_areas_count": 1
  }
}
```

### Predictive Health Modeling
Generate health outcome predictions using machine learning models.

```http
POST /v1/analytics/predictions
```

**Response Time:** <800ms

**Request Body:**
```json
{
  "target": "diabetes_prevalence",
  "prediction_horizon": "2025",
  "features": [
    "current_diabetes_rate",
    "aging_population_trend",
    "socioeconomic_factors",
    "healthcare_access_changes"
  ],
  "geographic_scope": {
    "sa1_codes": ["101011001", "101011002"]
  },
  "model_type": "gradient_boosting",
  "confidence_interval": 0.95
}
```

**Response:**
```json
{
  "predictive_model": {
    "target": "diabetes_prevalence",
    "prediction_year": 2025,
    "model_type": "gradient_boosting",
    "model_performance": {
      "r_squared": 0.87,
      "mae": 0.43,
      "rmse": 0.61
    },
    "predictions": [
      {
        "sa1_code": "101011001",
        "current_value": 4.2,
        "predicted_value": 4.8,
        "confidence_interval": [4.1, 5.5],
        "change_percentage": 14.3,
        "trend": "increasing",
        "risk_factors": [
          "aging population",
          "lifestyle factors"
        ]
      }
    ],
    "feature_importance": {
      "current_diabetes_rate": 0.45,
      "aging_population_trend": 0.28,
      "socioeconomic_factors": 0.18,
      "healthcare_access_changes": 0.09
    }
  },
  "recommendations": [
    "Implement early intervention programs in SA1 101011001",
    "Monitor aging population health trends"
  ]
}
```

---

## 📈 Report Generation

### Comprehensive Health Reports
Generate detailed analytics reports for specific areas or regions.

```http
POST /v1/analytics/reports
```

**Response Time:** <1s

**Request Body:**
```json
{
  "report_type": "health_profile_comprehensive",
  "geographic_scope": {
    "sa4_code": "101",
    "include_sa1_details": true
  },
  "sections": [
    "executive_summary",
    "demographic_profile",
    "health_indicators",
    "risk_assessment",
    "comparative_analysis",
    "recommendations"
  ],
  "comparison_benchmarks": [
    "state_average",
    "national_average",
    "peer_areas"
  ],
  "format": "json",
  "include_visualizations": true
}
```

**Response:**
```json
{
  "report": {
    "id": "rpt_health_comprehensive_101_2024",
    "title": "Sydney City and Inner South - Health Profile 2024",
    "generated_at": "2024-08-31T10:30:00Z",
    "geographic_scope": {
      "sa4_code": "101",
      "sa4_name": "Sydney - City and Inner South",
      "total_population": 567890,
      "sa1_areas_included": 1847
    },
    "executive_summary": {
      "overall_health_score": 7.2,
      "key_findings": [
        "Above average life expectancy (82.1 vs 80.9 national)",
        "Below average diabetes rates (5.8% vs 7.2% national)",
        "High healthcare service utilization",
        "Significant health disparities between areas"
      ],
      "priority_recommendations": [
        "Address health inequities in disadvantaged pockets",
        "Strengthen diabetes prevention programs",
        "Maintain excellent healthcare access"
      ]
    },
    "health_indicators": {
      "chronic_disease": {
        "diabetes_prevalence": {
          "value": 5.8,
          "national_percentile": 35,
          "trend": "stable"
        }
      },
      "mortality": {
        "life_expectancy": {
          "value": 82.1,
          "national_percentile": 78,
          "trend": "improving"
        }
      }
    },
    "risk_assessment": {
      "areas_by_risk_level": {
        "very_high": 89,
        "high": 234,
        "moderate": 567,
        "low": 689,
        "very_low": 268
      },
      "population_at_risk": 145670
    }
  },
  "visualizations": [
    {
      "type": "choropleth_map",
      "title": "Health Risk Distribution",
      "url": "https://api.ahgd.dev/v1/reports/visualizations/choropleth_101_risk.png"
    }
  ],
  "processing_time_ms": 890
}
```

---

## 🤖 Machine Learning Models

### Available Models

| Model Type | Use Case | Performance | Training Data |
|------------|----------|-------------|---------------|
| **Gradient Boosting** | Disease prediction | R²=0.87 | 5+ years health data |
| **Random Forest** | Risk classification | AUC=0.92 | Multi-indicator analysis |
| **Neural Network** | Complex patterns | R²=0.84 | Deep feature learning |
| **Linear Regression** | Trend analysis | R²=0.76 | Simple relationships |
| **K-Means** | Area clustering | Silhouette=0.67 | Unsupervised grouping |

### Model Validation
```json
{
  "cross_validation": {
    "folds": 10,
    "stratified": true,
    "metrics": {
      "accuracy": 0.89,
      "precision": 0.87,
      "recall": 0.91,
      "f1_score": 0.89
    }
  },
  "holdout_performance": {
    "test_size": 0.2,
    "r_squared": 0.85,
    "mae": 0.47
  }
}
```

---

## 📊 Statistical Analysis

### Hypothesis Testing
```http
POST /v1/analytics/hypothesis-test
```

**Request Body:**
```json
{
  "null_hypothesis": "No difference in diabetes rates between urban and rural areas",
  "groups": {
    "urban": {
      "filter": {"urban_classification": "Major Urban"}
    },
    "rural": {
      "filter": {"urban_classification": "Rural Balance"}
    }
  },
  "variable": "diabetes_prevalence",
  "test_type": "t_test_independent",
  "alpha": 0.05
}
```

### Regression Analysis
```http
POST /v1/analytics/regression
```

**Request Body:**
```json
{
  "dependent_variable": "life_expectancy",
  "independent_variables": [
    "seifa_irsad",
    "healthcare_access_score",
    "air_quality_index",
    "population_density"
  ],
  "model_type": "multiple_linear",
  "include_diagnostics": true
}
```

---

## 🔍 Advanced Query Operations

### Time Series Analysis
```http
POST /v1/analytics/time-series
```

**Request Body:**
```json
{
  "indicator": "diabetes_prevalence",
  "areas": ["101011001"],
  "time_period": "2015-2023",
  "analysis": {
    "trend": true,
    "seasonality": true,
    "forecast": {
      "periods": 3,
      "method": "arima"
    }
  }
}
```

### Spatial Autocorrelation
```http
POST /v1/analytics/spatial-autocorr
```

**Request Body:**
```json
{
  "variable": "diabetes_prevalence",
  "geographic_scope": {"state": "NSW"},
  "spatial_weights": "queen_contiguity",
  "significance_test": true
}
```

---

## 📈 Performance Optimization

### Query Optimization
- **Lazy evaluation** for complex analytics pipelines
- **Parallel processing** across multiple CPU cores
- **Memory mapping** for large dataset operations
- **Query plan optimization** with DuckDB

### Caching Strategy
```json
{
  "cache_levels": {
    "raw_data": "24 hours",
    "aggregated_results": "6 hours",
    "model_predictions": "2 hours",
    "correlation_matrices": "1 hour"
  },
  "cache_keys": "query_hash + parameters + data_version"
}
```

---

## 💡 Usage Examples

### Python - Health Analytics
```python
from ahgd import AnalyticsAPI
import pandas as pd
import matplotlib.pyplot as plt

client = AnalyticsAPI(api_key="your-key")

# Correlation analysis
correlations = client.correlations({
    "indicators": ["diabetes_rate", "life_expectancy", "seifa_rank"],
    "geographic_scope": {"state": "NSW"}
})

# Risk assessment for multiple areas
risk_scores = client.risk_assessment({
    "areas": ["101011001", "201031245"],
    "risk_factors": ["chronic_disease", "healthcare_access"]
})

# Predictive modeling
predictions = client.predict({
    "target": "diabetes_prevalence",
    "prediction_horizon": "2025",
    "areas": ["101011001"]
})
```

### R - Statistical Analysis
```r
library(ahgd)

client <- ahgd_analytics_client("your-api-key")

# Health clustering analysis
clusters <- health_clustering(
  client,
  features = c("diabetes_rate", "life_expectancy", "seifa_rank"),
  num_clusters = 5,
  state = "VIC"
)

# Regression analysis
regression_results <- health_regression(
  client,
  dependent = "life_expectancy",
  independent = c("seifa_irsad", "diabetes_rate", "air_quality"),
  areas = get_metro_areas(client, "Melbourne")
)

# Generate comprehensive report
report <- generate_health_report(
  client,
  sa4_code = "205",  # Greater Melbourne
  sections = c("demographics", "health_indicators", "risk_assessment")
)
```

---

**[← Back to API Hub](README.md)** | **[Next: System API →](system-api.md)**

---

*Last updated: August 2024 • Powered by DuckDB + Polars for maximum analytical performance*
