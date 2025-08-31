# System API
### Monitoring, Performance & Administration

The System API provides real-time monitoring, performance metrics, and administrative capabilities for the AHGD platform, delivering comprehensive system health and operational insights.

---

## 🔧 Core System Endpoints

### System Health Check
Get overall system health status and availability.

```http
GET /v1/system/health
```

**Response Time:** <50ms

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-08-31T10:30:00Z",
  "version": "3.0.0",
  "uptime_seconds": 2847293,
  "services": {
    "api_server": {
      "status": "healthy",
      "response_time_ms": 12,
      "last_check": "2024-08-31T10:29:55Z"
    },
    "database": {
      "status": "healthy", 
      "connection_pool": {
        "active": 8,
        "idle": 12,
        "max": 50
      },
      "query_performance_ms": 23
    },
    "cache": {
      "status": "healthy",
      "hit_rate": 0.87,
      "memory_usage_mb": 2048,
      "eviction_rate": 0.02
    },
    "parquet_storage": {
      "status": "healthy",
      "disk_usage_gb": 127.4,
      "read_performance_mb_s": 450.2,
      "write_performance_mb_s": 234.7
    }
  },
  "data_freshness": {
    "health_indicators": "2024-08-30T02:00:00Z",
    "geographic_data": "2024-08-01T00:00:00Z",
    "demographics": "2024-07-01T00:00:00Z"
  }
}
```

### Performance Metrics
Get detailed performance metrics and system utilization.

```http
GET /v1/system/performance
```

**Response Time:** <100ms

**Query Parameters:**
- `window` - Time window: `1h`, `24h`, `7d`, `30d`
- `metrics` - Specific metrics: `cpu,memory,disk,network`

**Response:**
```json
{
  "timestamp": "2024-08-31T10:30:00Z",
  "time_window": "1h",
  "performance_metrics": {
    "api_performance": {
      "requests_per_second": 247.3,
      "average_response_time_ms": 145,
      "p95_response_time_ms": 380,
      "p99_response_time_ms": 890,
      "error_rate": 0.003,
      "success_rate": 0.997
    },
    "query_performance": {
      "polars_operations_per_sec": 1834,
      "duckdb_queries_per_sec": 456,
      "cache_hit_rate": 0.87,
      "average_query_time_ms": 67,
      "memory_efficiency": {
        "peak_usage_gb": 3.2,
        "average_usage_gb": 1.8,
        "gc_frequency_per_hour": 12
      }
    },
    "data_processing": {
      "parquet_read_mb_s": 450.2,
      "parquet_write_mb_s": 234.7,
      "compression_ratio": 4.7,
      "concurrent_operations": 8,
      "queue_depth": 2
    },
    "system_resources": {
      "cpu_usage_percent": 34.2,
      "memory_usage_gb": 14.7,
      "memory_total_gb": 32.0,
      "disk_usage_percent": 67.8,
      "network_throughput_mbps": 89.4
    }
  },
  "performance_trends": {
    "response_time_trend": "stable",
    "throughput_trend": "increasing",
    "resource_utilization_trend": "stable"
  }
}
```

### Data Quality Metrics
Monitor data quality, completeness, and accuracy.

```http
GET /v1/system/data-quality
```

**Response Time:** <200ms

**Response:**
```json
{
  "data_quality_summary": {
    "overall_score": 94.7,
    "last_assessment": "2024-08-31T06:00:00Z",
    "assessment_frequency": "daily",
    "trending": "improving"
  },
  "data_sources": {
    "aihw_health_data": {
      "quality_score": 96.2,
      "completeness": 0.94,
      "accuracy": 0.98,
      "consistency": 0.96,
      "currency_days": 2,
      "issues": []
    },
    "abs_census_data": {
      "quality_score": 98.5,
      "completeness": 0.99,
      "accuracy": 0.99,
      "consistency": 0.98,
      "currency_days": 45,
      "issues": [
        {
          "type": "minor",
          "description": "3 SA1 areas with estimated population",
          "impact": "minimal"
        }
      ]
    },
    "phidu_health_atlas": {
      "quality_score": 91.3,
      "completeness": 0.89,
      "accuracy": 0.95,
      "consistency": 0.93,
      "currency_days": 180,
      "issues": [
        {
          "type": "moderate",
          "description": "Remote area data sparsity",
          "impact": "affects 247 SA1 areas"
        }
      ]
    }
  },
  "validation_rules": {
    "total_rules": 234,
    "passing": 228,
    "failing": 6,
    "warnings": 12
  },
  "anomalies": {
    "detected": 3,
    "resolved": 1,
    "pending_review": 2
  }
}
```

### System Configuration
Get current system configuration and feature flags.

```http
GET /v1/system/config
```

**Response:**
```json
{
  "api_version": "3.0.0",
  "build_info": {
    "commit_hash": "a1b2c3d4e5f6",
    "build_date": "2024-08-30T12:00:00Z",
    "environment": "production"
  },
  "feature_flags": {
    "polars_processing": true,
    "parquet_caching": true,
    "ml_predictions": true,
    "real_time_analytics": true,
    "experimental_endpoints": false
  },
  "rate_limits": {
    "default": 1000,
    "premium": 10000,
    "enterprise": -1
  },
  "data_retention": {
    "raw_data_days": 365,
    "processed_data_days": 1095,
    "logs_days": 90,
    "cache_hours": 24
  }
}
```

---

## 📊 Monitoring & Alerting

### System Alerts
Get active system alerts and notifications.

```http
GET /v1/system/alerts?severity={level}&status={status}
```

**Parameters:**
- `severity`: `low`, `medium`, `high`, `critical`
- `status`: `active`, `resolved`, `acknowledged`

**Response:**
```json
{
  "active_alerts": 2,
  "total_alerts_24h": 8,
  "alerts": [
    {
      "id": "alert_disk_usage_high",
      "severity": "medium",
      "status": "active",
      "title": "Disk usage above 75%",
      "description": "Parquet storage disk usage at 78.4%",
      "triggered_at": "2024-08-31T09:15:00Z",
      "affected_services": ["parquet_storage"],
      "recommended_actions": [
        "Archive old data",
        "Increase storage capacity"
      ]
    },
    {
      "id": "alert_cache_hit_rate_low", 
      "severity": "low",
      "status": "active",
      "title": "Cache hit rate below threshold",
      "description": "Cache hit rate at 82% (threshold: 85%)",
      "triggered_at": "2024-08-31T08:30:00Z",
      "affected_services": ["cache"],
      "auto_resolve": true
    }
  ]
}
```

### Performance Benchmarks
Get performance benchmarks and SLA compliance.

```http
GET /v1/system/benchmarks
```

**Response:**
```json
{
  "sla_compliance": {
    "availability": {
      "target": 99.9,
      "actual_30d": 99.97,
      "status": "exceeding"
    },
    "response_time": {
      "target_p95_ms": 500,
      "actual_p95_ms": 380,
      "status": "meeting"
    },
    "error_rate": {
      "target": 0.01,
      "actual": 0.003,
      "status": "exceeding"
    }
  },
  "performance_benchmarks": {
    "health_api_p95_ms": 145,
    "geo_api_p95_ms": 98,
    "analytics_api_p95_ms": 420,
    "data_processing_throughput_mbs": 450.2,
    "concurrent_users_supported": 250,
    "queries_per_second": 1500
  },
  "resource_utilization": {
    "cpu_efficiency": 0.89,
    "memory_efficiency": 0.92,
    "storage_efficiency": 0.87,
    "network_efficiency": 0.94
  }
}
```

---

## 🔐 Administrative Operations

### Data Refresh Status
Monitor data refresh operations and pipeline status.

```http
GET /v1/system/data-refresh
```

**Response:**
```json
{
  "pipeline_status": {
    "health_data_pipeline": {
      "status": "running",
      "started_at": "2024-08-31T06:00:00Z",
      "progress": 0.78,
      "estimated_completion": "2024-08-31T11:30:00Z",
      "records_processed": 12847293,
      "current_stage": "aihw_mortality_processing"
    },
    "geographic_pipeline": {
      "status": "completed",
      "completed_at": "2024-08-31T02:15:00Z",
      "records_processed": 61845,
      "duration_minutes": 12
    }
  },
  "last_refresh": {
    "health_indicators": "2024-08-30T02:00:00Z",
    "geographic_boundaries": "2024-08-01T00:00:00Z", 
    "demographic_data": "2024-07-01T00:00:00Z"
  },
  "next_scheduled": {
    "health_indicators": "2024-09-01T02:00:00Z",
    "quality_checks": "2024-08-31T18:00:00Z"
  }
}
```

### Cache Management
Monitor and manage system caches.

```http
GET /v1/system/cache
POST /v1/system/cache/clear
```

**GET Response:**
```json
{
  "cache_layers": {
    "query_cache": {
      "size_mb": 1024,
      "entries": 15847,
      "hit_rate": 0.87,
      "eviction_policy": "LRU",
      "ttl_hours": 1
    },
    "data_cache": {
      "size_mb": 2048, 
      "entries": 5673,
      "hit_rate": 0.92,
      "eviction_policy": "LFU",
      "ttl_hours": 6
    },
    "parquet_cache": {
      "size_gb": 8.4,
      "files": 234,
      "hit_rate": 0.94,
      "compression_ratio": 4.2,
      "ttl_hours": 24
    }
  },
  "cache_performance": {
    "read_operations_per_sec": 2847,
    "write_operations_per_sec": 456,
    "evictions_per_hour": 23,
    "memory_pressure": "normal"
  }
}
```

---

## 📈 Usage Analytics

### API Usage Statistics
Get detailed API usage and consumption metrics.

```http
GET /v1/system/usage?window={period}&breakdown={dimension}
```

**Parameters:**
- `window`: `1h`, `24h`, `7d`, `30d`
- `breakdown`: `endpoint`, `user`, `plan`, `geography`

**Response:**
```json
{
  "usage_period": "24h",
  "summary": {
    "total_requests": 184567,
    "unique_users": 1247,
    "data_transferred_gb": 23.4,
    "average_requests_per_user": 148
  },
  "endpoint_usage": {
    "/v1/health/sa1": {
      "requests": 67834,
      "percentage": 36.8,
      "avg_response_time_ms": 89,
      "error_rate": 0.002
    },
    "/v1/geo/boundaries": {
      "requests": 34567,
      "percentage": 18.7,
      "avg_response_time_ms": 145,
      "error_rate": 0.001
    },
    "/v1/analytics/correlations": {
      "requests": 8945,
      "percentage": 4.8,
      "avg_response_time_ms": 567,
      "error_rate": 0.008
    }
  },
  "geographic_usage": {
    "NSW": 45.2,
    "VIC": 28.7,
    "QLD": 15.3,
    "WA": 6.8,
    "other": 4.0
  },
  "usage_trends": {
    "requests": "increasing",
    "response_times": "stable", 
    "error_rates": "decreasing"
  }
}
```

### User Analytics
Monitor user behavior and API consumption patterns.

```http
GET /v1/system/users?plan={tier}&active={period}
```

**Response:**
```json
{
  "user_statistics": {
    "total_users": 3247,
    "active_24h": 892,
    "active_7d": 1456,
    "active_30d": 2134,
    "new_users_30d": 234
  },
  "plan_distribution": {
    "free": 2456,
    "professional": 678,
    "enterprise": 113
  },
  "usage_patterns": {
    "peak_hours": [9, 10, 11, 14, 15],
    "geographic_distribution": {
      "australia": 0.78,
      "international": 0.22
    },
    "common_use_cases": [
      "research_analysis",
      "government_planning",
      "commercial_analytics",
      "academic_projects"
    ]
  }
}
```

---

## 🚨 Error Monitoring

### Error Analytics
Monitor and analyze system errors and exceptions.

```http
GET /v1/system/errors?window={period}&severity={level}
```

**Response:**
```json
{
  "error_summary": {
    "total_errors_24h": 45,
    "error_rate": 0.0024,
    "most_common": "RATE_LIMIT_EXCEEDED",
    "trending": "stable"
  },
  "error_breakdown": {
    "RATE_LIMIT_EXCEEDED": {
      "count": 18,
      "percentage": 40.0,
      "avg_per_hour": 0.75,
      "affected_endpoints": ["/v1/health/search"]
    },
    "INVALID_SA1_CODE": {
      "count": 12,
      "percentage": 26.7,
      "common_patterns": ["999999999", "12345"]
    },
    "TIMEOUT": {
      "count": 8,
      "percentage": 17.8,
      "avg_duration_ms": 5000
    }
  },
  "resolution_metrics": {
    "auto_resolved": 38,
    "manual_intervention": 7,
    "average_resolution_time_minutes": 4.2
  }
}
```

---

## ⚡ Performance Tools

### Query Profiler
Profile and optimize slow queries.

```http
POST /v1/system/profile
```

**Request Body:**
```json
{
  "query": {
    "endpoint": "/v1/health/search",
    "parameters": {
      "filters": {"diabetes_rate": {"min": 8.0}},
      "limit": 100
    }
  },
  "profile_level": "detailed"
}
```

**Response:**
```json
{
  "query_profile": {
    "total_time_ms": 234,
    "stages": [
      {
        "stage": "query_parsing",
        "time_ms": 12,
        "percentage": 5.1
      },
      {
        "stage": "data_filtering",
        "time_ms": 145,
        "percentage": 62.0
      },
      {
        "stage": "result_serialization", 
        "time_ms": 77,
        "percentage": 32.9
      }
    ],
    "optimization_suggestions": [
      "Add index on diabetes_rate column",
      "Use columnar filtering for better performance"
    ],
    "memory_usage_mb": 23.4,
    "rows_scanned": 1284567,
    "rows_returned": 100
  }
}
```

---

## 💡 Usage Examples

### Python - System Monitoring
```python
from ahgd import SystemAPI
import time

client = SystemAPI(api_key="your-key")

# Check system health
health = client.get_health()
print(f"System status: {health['status']}")

# Monitor performance
performance = client.get_performance(window="1h")
print(f"API throughput: {performance['requests_per_second']:.1f} req/s")

# Set up monitoring loop
while True:
    metrics = client.get_performance()
    if metrics['error_rate'] > 0.01:
        print("⚠️  High error rate detected!")
    
    if metrics['p95_response_time_ms'] > 1000:
        print("⚠️  High response times detected!")
    
    time.sleep(60)
```

### System Health Dashboard
```javascript
import { SystemAPI } from '@ahgd/js-sdk';

const systemClient = new SystemAPI('your-api-key');

// Real-time system monitoring
async function updateDashboard() {
  const [health, performance, alerts] = await Promise.all([
    systemClient.getHealth(),
    systemClient.getPerformance('1h'),
    systemClient.getAlerts('active')
  ]);
  
  // Update dashboard elements
  document.getElementById('status').textContent = health.status;
  document.getElementById('response-time').textContent = 
    `${performance.average_response_time_ms}ms`;
  document.getElementById('alert-count').textContent = alerts.active_alerts;
}

// Update every 30 seconds
setInterval(updateDashboard, 30000);
```

---

**[← Back to API Hub](README.md)**

---

*Last updated: August 2024 • Real-time monitoring powered by high-performance metrics collection*