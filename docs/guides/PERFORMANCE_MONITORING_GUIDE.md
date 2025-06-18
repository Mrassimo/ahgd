# Performance Monitoring and Optimization Guide

## Overview

The Australian Health Analytics Dashboard includes a comprehensive performance monitoring and optimization system designed for enterprise-grade deployment. This system provides real-time monitoring, intelligent caching, automated alerting, and production-ready optimizations.

## System Architecture

### Core Components

1. **Performance Monitoring** (`src/performance/monitoring.py`)
   - Real-time metrics collection
   - System resource tracking
   - Application performance analytics
   - Database query monitoring

2. **Advanced Caching** (`src/performance/cache.py`)
   - Multi-tier caching (Memory, File, Redis)
   - Intelligent cache invalidation
   - Compression and optimization
   - Streamlit session integration

3. **Query Optimization** (`src/performance/optimization.py`)
   - Database connection pooling
   - Query result caching
   - Lazy loading for large datasets
   - Pagination support

4. **Health Monitoring** (`src/performance/health.py`)
   - System health checks
   - Application readiness monitoring
   - Resource usage validation
   - Production health endpoints

5. **Automated Alerting** (`src/performance/alerts.py`)
   - Performance threshold monitoring
   - Multi-channel notifications
   - Alert aggregation and rate limiting
   - Escalation policies

6. **Production Features** (`src/performance/production.py`)
   - Environment-specific configurations
   - Graceful shutdown handling
   - Resource optimization
   - Load balancer compatibility

## Quick Start

### Basic Setup

```python
from src.performance.monitoring import get_performance_monitor
from src.performance.cache import get_cache_manager

# Initialize monitoring
monitor = get_performance_monitor()
cache = get_cache_manager()

# Start monitoring
monitor.start_monitoring()
```

### Production Setup

```python
from src.performance.production import setup_production_environment

# Setup production environment with all optimizations
production_manager = setup_production_environment()
```

### Environment Variables

```bash
# Production configuration
export AHGD_ENVIRONMENT=production
export AHGD_PERFORMANCE_MONITORING=true
export AHGD_CACHING=true

# Redis configuration (optional)
export AHGD_REDIS_ENABLED=true
export AHGD_REDIS_HOST=localhost
export AHGD_REDIS_PORT=6379

# Alert configuration
export AHGD_ALERT_EMAIL=true
export AHGD_ALERT_EMAIL_RECIPIENTS=admin@example.com,ops@example.com
export AHGD_ALERT_WEBHOOK_URLS=https://webhook.example.com/alerts

# Resource limits
export AHGD_MAX_MEMORY_MB=2048
export AHGD_MAX_CPU_PERCENT=80
```

## Performance Monitoring

### Metrics Collection

The system automatically collects metrics in several categories:

- **System Metrics**: CPU, Memory, Disk usage
- **Application Metrics**: Function execution times, page load times
- **Database Metrics**: Query duration, connection pool status
- **Cache Metrics**: Hit rates, storage usage
- **User Metrics**: Session information, interaction tracking

### Performance Dashboard

Access the performance dashboard at: `http://localhost:8503`

```bash
streamlit run performance_dashboard.py --server.port 8503
```

The dashboard provides:
- Real-time system status
- Performance trends and analytics
- Health check results
- Alert status and history
- Cache performance metrics
- Database query analysis

### Custom Metrics

Add custom metrics to track application-specific performance:

```python
from src.performance.monitoring import get_performance_monitor

monitor = get_performance_monitor()

# Add custom metric
monitor.add_custom_metric("user_login", 1, "authentication")

# Track function performance
@monitor.track_function_performance("data_processing")
def process_data(data):
    # Your processing logic
    return processed_data
```

## Caching System

### Cache Backends

The system supports multiple cache backends with automatic fallback:

1. **Streamlit Session Cache**: Fast in-memory caching for session data
2. **File Cache**: Persistent caching across sessions
3. **Redis Cache**: Distributed caching for production deployment

### Cache Configuration

```python
from src.performance.cache import CacheManager, CacheConfig

config = CacheConfig(
    # Enable Redis for production
    redis_enabled=True,
    redis_host="localhost",
    redis_port=6379,
    
    # File cache settings
    file_cache_enabled=True,
    max_file_cache_size_mb=1024,
    
    # Compression settings
    compression_enabled=True,
    compression_threshold=1024  # Compress objects > 1KB
)

cache_manager = CacheManager(config)
```

### Using the Cache

```python
from src.performance.cache import cached, get_cache_manager

# Decorator for automatic caching
@cached(ttl=3600)  # Cache for 1 hour
def expensive_computation(param1, param2):
    # Expensive operation
    return result

# Manual cache operations
cache = get_cache_manager()
cache.set("key", data, ttl=1800)
result = cache.get("key")
```

## Query Optimization

### Database Optimization

```python
from src.performance.optimization import QueryOptimizer

optimizer = QueryOptimizer("database.db", cache_manager, monitor)

# Execute optimized query with caching
result = optimizer.execute_query(
    "SELECT * FROM health_data WHERE state = ?",
    parameters=["NSW"],
    cache_ttl=3600
)

# Create database indexes for better performance
optimizer.create_index("health_data", ["state", "year"])
```

### Lazy Loading

For large datasets, use lazy loading to improve initial load times:

```python
from src.performance.optimization import PaginatedQuery

paginated = PaginatedQuery(optimizer, "SELECT * FROM large_table", page_size=1000)
lazy_df = paginated.to_lazy_dataframe()

# Access data on-demand
first_page = lazy_df.head(1000)
specific_rows = lazy_df[5000:6000]
```

## Health Monitoring

### Health Checks

The system includes comprehensive health checks:

- **Database Connectivity**: Connection and query performance
- **System Resources**: Memory, CPU, disk usage
- **Data Integrity**: File availability and data quality
- **Application Health**: Session state and component status

### Health Endpoints

For load balancer integration, health endpoints are available:

- `GET /health` - Overall health status
- `GET /health/ready` - Readiness check
- `GET /health/live` - Liveness check
- `GET /metrics` - Prometheus-compatible metrics

```python
from src.performance.health import get_health_checker

health_checker = get_health_checker()

# Get health status
status = health_checker.run_all_checks()
print(f"Overall status: {status.status.value}")

# Get endpoint responses
health_response = health_checker.get_health_endpoint_response()
readiness_response = health_checker.get_readiness_check()
```

## Alerting System

### Alert Rules

Define custom alert rules for monitoring thresholds:

```python
from src.performance.alerts import AlertRule, AlertSeverity, AlertChannel

rule = AlertRule(
    name="high_memory_usage",
    condition="name == 'memory_usage_percent' and value > 85",
    severity=AlertSeverity.HIGH,
    channels=[AlertChannel.EMAIL, AlertChannel.WEBHOOK],
    message_template="High memory usage: {value:.1f}%",
    cooldown_minutes=15
)

alert_manager.add_rule(rule)
```

### Alert Channels

Configure multiple notification channels:

```python
from src.performance.alerts import AlertConfig

config = AlertConfig(
    # Email settings
    smtp_host="smtp.gmail.com",
    smtp_port=587,
    smtp_username="alerts@example.com",
    smtp_password="app_password",
    email_to=["admin@example.com", "ops@example.com"],
    
    # Webhook settings
    webhook_urls=["https://hooks.slack.com/webhook_url"],
    
    # File logging
    alert_log_file=Path("alerts.log")
)
```

### Default Alert Rules

The system includes default rules for common scenarios:

- High CPU usage (>80%)
- Critical memory usage (>90%)
- Slow database queries (>5s)
- Health check failures
- Slow page loads (>10s)

## Production Deployment

### Docker Configuration

```dockerfile
FROM python:3.11-slim

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Environment variables
ENV AHGD_ENVIRONMENT=production
ENV AHGD_REDIS_ENABLED=true
ENV AHGD_REDIS_HOST=redis

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD python -c "from src.performance.health import get_health_checker; \
                 hc = get_health_checker(); \
                 status = hc.get_liveness_check(); \
                 exit(0 if status['alive'] else 1)"

# Run application
CMD ["streamlit", "run", "main.py", "--server.port", "8501"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ahgd-dashboard
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ahgd-dashboard
  template:
    metadata:
      labels:
        app: ahgd-dashboard
    spec:
      containers:
      - name: dashboard
        image: ahgd-dashboard:latest
        ports:
        - containerPort: 8501
        - containerPort: 8502  # Health endpoint
        env:
        - name: AHGD_ENVIRONMENT
          value: "production"
        - name: AHGD_REDIS_HOST
          value: "redis-service"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8502
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8502
          initialDelaySeconds: 30
          periodSeconds: 10
```

### Load Balancer Configuration

```nginx
upstream ahgd_backend {
    server ahgd-1:8501 max_fails=3 fail_timeout=30s;
    server ahgd-2:8501 max_fails=3 fail_timeout=30s;
    server ahgd-3:8501 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name dashboard.example.com;
    
    # Health check endpoint
    location /health {
        proxy_pass http://ahgd_backend:8502;
        proxy_set_header Host $host;
        access_log off;
    }
    
    # Main application
    location / {
        proxy_pass http://ahgd_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # WebSocket support for Streamlit
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Performance Tuning

### Memory Optimization

```python
# Optimize DataFrame memory usage
from src.performance.optimization import optimize_dataframe_memory

df = optimize_dataframe_memory(df)

# Configure cache memory limits
cache_config = CacheConfig(max_memory_mb=512)
```

### Database Optimization

```python
# Create indexes for frequently queried columns
optimizer.create_index("health_metrics", ["sa2_code", "year"])
optimizer.create_index("sa2_boundaries", ["state_code"])

# Optimize table statistics
optimizer.optimize_table("health_metrics")
```

### Session Management

```python
# Configure session cleanup
production_config = ProductionConfig(
    session_cleanup_enabled=True,
    session_max_age_hours=24
)
```

## Monitoring and Observability

### Metrics Export

Export metrics to external monitoring systems:

```python
# Get metrics in Prometheus format
monitor = get_performance_monitor()
metrics = monitor.get_performance_summary()

# Custom metrics export
def export_to_prometheus():
    summary = monitor.get_performance_summary()
    # Format for Prometheus ingestion
    return prometheus_format(summary)
```

### Logging Configuration

```python
import logging

# Production logging configuration
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ahgd_production.log')
    ]
)
```

### Performance Benchmarking

```python
# Benchmark application performance
from src.performance.monitoring import track_performance

@track_performance("benchmark_test")
def benchmark_operation():
    # Operation to benchmark
    pass

# Run multiple iterations and analyze results
for i in range(100):
    benchmark_operation()

# Get performance statistics
stats = monitor.get_performance_summary()
print(f"Average execution time: {stats['function_avg_duration_benchmark_test']:.2f}s")
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Check cache configuration
   - Optimize DataFrame operations
   - Enable garbage collection
   - Review session state size

2. **Slow Query Performance**
   - Add database indexes
   - Enable query caching
   - Optimize query structure
   - Check connection pool settings

3. **Cache Misses**
   - Verify cache backend availability
   - Check TTL settings
   - Review cache key generation
   - Monitor cache memory limits

### Debug Mode

Enable debug mode for detailed diagnostics:

```python
config = ProductionConfig(
    debug_mode=True,
    log_level="DEBUG"
)
```

### Performance Profiling

```python
# Enable detailed performance tracking
@monitor.track_function_performance("detailed_analysis")
def analyze_performance():
    with monitor.track_page_load("specific_operation"):
        # Your operation here
        pass
```

## API Reference

### Core Classes

- `PerformanceMonitor`: Main monitoring coordinator
- `CacheManager`: Multi-tier cache management
- `QueryOptimizer`: Database optimization
- `HealthChecker`: System health monitoring
- `AlertManager`: Alert processing and delivery
- `ProductionManager`: Production deployment management

### Configuration Classes

- `CacheConfig`: Cache system configuration
- `AlertConfig`: Alert system configuration
- `ProductionConfig`: Production deployment configuration

### Utilities

- `@track_performance`: Function performance decorator
- `@cached`: Result caching decorator
- `optimize_dataframe_memory()`: Memory optimization utility
- `compress_data()` / `decompress_data()`: Data compression utilities

## Best Practices

1. **Production Deployment**
   - Use environment variables for configuration
   - Enable Redis for distributed caching
   - Configure appropriate resource limits
   - Set up health check endpoints

2. **Performance Optimization**
   - Cache frequently accessed data
   - Use lazy loading for large datasets
   - Optimize database queries with indexes
   - Monitor memory usage regularly

3. **Monitoring and Alerting**
   - Set appropriate alert thresholds
   - Configure multiple notification channels
   - Monitor both system and application metrics
   - Regular health check validation

4. **Security Considerations**
   - Secure cache backends (Redis authentication)
   - Validate alert webhook endpoints
   - Monitor for unusual resource usage patterns
   - Regular security updates

## Support and Contribution

For issues and feature requests related to the performance monitoring system, please refer to the main project documentation.

The performance monitoring system is designed to be extensible and customizable for specific deployment requirements.