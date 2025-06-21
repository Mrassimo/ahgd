# AHGD Structured Logging Framework

A comprehensive logging framework for the Australian Health Geography Data (AHGD) project, providing structured logging, performance monitoring, health checks, and data lineage tracking.

## Features

- **Structured Logging**: Using loguru and structlog for consistent, machine-readable logs
- **Rotating File Handlers**: Automatic log rotation with compression
- **JSON Formatting**: Machine-readable logs for integration with log aggregation systems
- **Performance Monitoring**: Decorators and context managers for operation timing
- **Data Lineage Tracking**: Track data transformations through ETL pipelines
- **Health Checks**: System and component health monitoring
- **System Resource Monitoring**: CPU, memory, disk, and network monitoring
- **Alerting**: Configurable alerts with multiple notification channels
- **Environment-Specific Configuration**: Different settings for dev, staging, production

## Quick Start

### 1. Basic Setup

```python
from src.utils import setup_development_logging

# Quick setup for development
logger = setup_development_logging()

# Basic logging
logger.log.info("Application started", component="main")
logger.log.error("Something went wrong", error_code="E001")
```

### 2. Environment-Specific Setup

```python
from src.utils import setup_environment_logging, detect_environment

# Auto-detect environment and setup logging
env = detect_environment()  # Returns 'development', 'production', etc.
logger = setup_environment_logging(env)

# Or explicitly specify environment
logger = setup_environment_logging('production')
```

### 3. Operation Context Tracking

```python
from src.utils import get_logger

logger = get_logger()

# Track operations with context
with logger.operation_context("data_processing", dataset="health_data", user_id="12345"):
    logger.log.info("Starting data validation")
    # Your processing code here
    logger.log.info("Validation completed", records_processed=5000)
```

### 4. Performance Monitoring

```python
from src.utils import monitor_performance

@monitor_performance("data_extraction")
def extract_health_data():
    """This function is automatically monitored for performance"""
    # Your extraction code here
    return data

# Get performance metrics
logger = get_logger()
metrics = logger.get_performance_metrics("data_extraction")
```

### 5. Data Lineage Tracking

```python
from src.utils import track_lineage

# Track data transformations
track_lineage(
    source_id="raw_mortality_data",
    target_id="clean_mortality_data", 
    operation="data_cleaning",
    row_count=50000,
    transformations=["null_removal", "validation"],
    validation_status="passed"
)
```

## Configuration

### Environment Detection

The framework automatically detects the environment from:
- `AHGD_ENV` environment variable
- `ENV` environment variable
- Deployment indicators (Docker, Kubernetes)
- Git repository presence

### Configuration File

Configuration is loaded from `configs/logging_config.yaml` with environment-specific sections:

```yaml
development:
  log_level: DEBUG
  log_dir: logs/dev
  console_logs: true
  json_logs: true
  performance_logging: true

production:
  log_level: INFO
  log_dir: logs/prod
  console_logs: false
  json_logs: true
  monitoring:
    enabled: true
    alerts:
      high_cpu:
        threshold: 80.0
        severity: high
```

### Validate Configuration

```python
from src.utils import validate_environment_config, print_config_summary

# Validate configuration
validation = validate_environment_config('production')
if not validation['valid']:
    print("Configuration errors:", validation['errors'])

# Print detailed summary
print_config_summary('production')
```

## Monitoring and Health Checks

### System Monitoring

```python
from src.utils import get_system_monitor

monitor = get_system_monitor()

# Get current system metrics
metrics = monitor.get_current_metrics()
print(f"CPU: {metrics.cpu_percent}%, Memory: {metrics.memory_percent}%")

# Start continuous monitoring
monitor.start_monitoring(interval=60)  # Check every 60 seconds
```

### Health Checks

```python
from src.utils import get_health_checker

health_checker = get_health_checker()

# Register custom health check
def check_database():
    # Your database check logic
    return True

health_checker.register_health_check(
    "database",
    check_database,
    "Database connectivity check"
)

# Run health checks
health_summary = health_checker.get_health_summary()
print(f"Overall status: {health_summary['overall_status']}")
```

### Error Tracking

```python
from src.utils import get_error_tracker

error_tracker = get_error_tracker()

try:
    # Your code that might fail
    risky_operation()
except Exception as e:
    error_tracker.track_error(e, {
        'component': 'data_processor',
        'operation': 'data_validation'
    })

# Get error summary
summary = error_tracker.get_error_summary(hours=24)
print(f"Errors in last 24h: {summary['total_errors']}")
```

## Log Files Structure

The framework creates organised log files by environment and type:

```
logs/
├── dev/
│   ├── etl/
│   │   ├── info.log
│   │   ├── error.log
│   │   └── info.json
│   ├── validation/
│   └── performance/
├── prod/
│   ├── etl/
│   ├── validation/
│   ├── errors/
│   ├── audit/
│   └── lineage/
└── test/
    └── test/
```

## Integration Examples

### ETL Pipeline Integration

```python
from src.utils import get_logger, track_lineage

logger = get_logger()

# Complete ETL pipeline with logging
with logger.operation_context("etl_pipeline", source="aihw_data"):
    
    # Extract
    with logger.operation_context("extract"):
        logger.log.info("Starting extraction")
        data = extract_data()
        track_lineage("aihw_api", "raw_staging", "extraction", row_count=len(data))
    
    # Transform  
    with logger.operation_context("transform"):
        logger.log.info("Starting transformation")
        clean_data = transform_data(data)
        track_lineage("raw_staging", "clean_data", "transformation", 
                     row_count=len(clean_data), transformations=["cleaning", "validation"])
    
    # Load
    with logger.operation_context("load"):
        logger.log.info("Starting load")
        load_data(clean_data)
        track_lineage("clean_data", "health_db.facts", "loading", row_count=len(clean_data))
```

### Web Application Integration

```python
from src.utils import setup_production_logging, get_health_checker
from flask import Flask

app = Flask(__name__)

# Setup logging for web app
logger = setup_production_logging()

# Register health checks
health_checker = get_health_checker()
health_checker.register_health_check("database", check_db_connection)
health_checker.register_health_check("cache", check_cache_connection)

@app.route('/health')
def health_check():
    health_summary = health_checker.get_health_summary()
    return health_summary, 200 if health_summary['overall_status'] == 'healthy' else 503
```

## Production Deployment

### Environment Variables

Set environment variables for production:

```bash
export AHGD_ENV=production
export AHGD_VERSION=1.2.0
export DEPLOYMENT_ID=prod-001
```

### Docker Integration

```dockerfile
# Dockerfile
COPY configs/logging_config.yaml /app/configs/
RUN mkdir -p /app/logs/prod
ENV AHGD_ENV=production
```

### Monitoring Setup

For production monitoring, configure alerts in `logging_config.yaml`:

```yaml
production:
  monitoring:
    enabled: true
    notifications:
      slack:
        enabled: true
        webhook_url: ${SLACK_WEBHOOK_URL}
      email:
        enabled: true
        to:
          - devops@company.com
          - data-team@company.com
```

## Best Practices

1. **Use Operation Context**: Always wrap operations in context managers for better tracking
2. **Monitor Performance**: Use decorators on critical functions
3. **Track Data Lineage**: Essential for data governance and debugging
4. **Configure Alerts**: Set appropriate thresholds for your environment
5. **Regular Health Checks**: Monitor system health proactively
6. **Structured Logging**: Include relevant context in log messages
7. **Environment-Specific Config**: Different settings for dev/staging/prod

## Troubleshooting

### Configuration Issues

```python
from src.utils import validate_environment_config, print_config_summary

# Check configuration
validation = validate_environment_config()
if not validation['valid']:
    print("Errors:", validation['errors'])
    print("Warnings:", validation['warnings'])

# Print detailed summary
print_config_summary()
```

### Log Directory Permissions

```python
from src.utils import create_log_directories

# Create required directories
try:
    create_log_directories('production')
    print("Directories created successfully")
except Exception as e:
    print(f"Failed to create directories: {e}")
```

### Performance Issues

If logging impacts performance:

1. Disable console logging in production
2. Reduce log levels (INFO instead of DEBUG)
3. Enable log sampling for high-volume applications
4. Use async logging (enabled by default)

## Advanced Features

### Log Sampling

For high-volume applications, configure sampling in `logging_config.yaml`:

```yaml
sampling:
  enabled: true
  rules:
    - level: DEBUG
      sample_rate: 0.01  # Sample 1% of DEBUG logs
    - level: INFO  
      sample_rate: 0.1   # Sample 10% of INFO logs
```

### Integration with External Systems

Configure integrations for log forwarding:

```yaml
integrations:
  elasticsearch:
    enabled: true
    hosts: ["localhost:9200"]
    index_pattern: "ahgd-logs-{date}"
  
  prometheus:
    enabled: true
    port: 8000
    path: /metrics
```

### Security Features

- Automatic PII masking
- Sensitive data pattern detection
- Log encryption support
- Audit trail compliance

## API Reference

See `docs/logging_examples.py` for comprehensive examples and `src/utils/` for full API documentation.

## Support

For issues or questions:
1. Check the examples in `docs/logging_examples.py`
2. Validate your configuration with `print_config_summary()`
3. Review log files in the configured log directory
4. Check system health with health checks