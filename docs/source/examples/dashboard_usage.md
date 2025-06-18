
# Dashboard Usage Example

## Basic Dashboard Setup

```python
from src.dashboard.app import create_dashboard
from src.config import Config

# Load configuration
config = Config()

# Create and run dashboard
app = create_dashboard(config)
app.run(
    host=config.dashboard.host,
    port=config.dashboard.port,
    debug=config.dashboard.debug
)
```

## Data Loading Example

```python
from src.dashboard.data.loaders import HealthDataLoader
from src.config import Config

# Initialize data loader
config = Config()
loader = HealthDataLoader(config)

# Load health data
health_data = loader.load_aihw_mortality_data()
demographic_data = loader.load_demographic_data()

# Process and merge data
processed_data = loader.merge_health_demographic_data(
    health_data, 
    demographic_data
)
```

## Performance Monitoring Example

```python
from src.performance.monitoring import PerformanceMonitor
from src.config import Config

# Initialize monitoring
config = Config()
monitor = PerformanceMonitor(config)

# Start monitoring
monitor.start_monitoring()

# Your application code here
# ...

# Get performance metrics
metrics = monitor.get_metrics()
print(f"Memory usage: {metrics['memory_usage_mb']}MB")
print(f"CPU usage: {metrics['cpu_usage_percent']}%")
```
