Performance Package
==================

.. automodule:: src.performance
   :members:
   :undoc-members:
   :show-inheritance:

The performance package provides comprehensive monitoring, caching, optimisation,
and alerting capabilities for the Australian Health Analytics Dashboard.

Monitoring Module
-----------------

.. automodule:: src.performance.monitoring
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
~~~~~~~~~~~

.. autoclass:: src.performance.monitoring.PerformanceMonitor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.performance.monitoring.MetricsCollector
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
~~~~~~~~~~~~~~

.. code-block:: python

   from src.performance.monitoring import PerformanceMonitor
   
   # Create performance monitor
   monitor = PerformanceMonitor()
   
   # Start monitoring
   monitor.start()
   
   # Get current metrics
   metrics = monitor.get_metrics()
   print(f"CPU Usage: {metrics['cpu_percent']}%")
   print(f"Memory Usage: {metrics['memory_percent']}%")
   
   # Stop monitoring
   monitor.stop()

Health Monitoring
-----------------

.. automodule:: src.performance.health
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
~~~~~~~~~~~

.. autoclass:: src.performance.health.HealthChecker
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.performance.health.DatabaseHealthCheck
   :members:
   :undoc-members:
   :show-inheritance:.. autoclass:: src.performance.health.DataQualityCheck
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
~~~~~~~~~~~~~~

.. code-block:: python

   from src.performance.health import HealthChecker
   
   # Create health checker
   health_checker = HealthChecker()
   
   # Check all health metrics
   health_status = health_checker.check_all()
   
   # Check specific components
   db_health = health_checker.check_database_health()
   data_health = health_checker.check_data_quality()
   
   # Print health report
   for check, status in health_status.items():
       print(f"{check}: {'✓' if status['healthy'] else '✗'}")
       if not status['healthy']:
           print(f"  Issue: {status['message']}")

Caching Module
--------------

.. automodule:: src.performance.cache
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
~~~~~~~~~~~

.. autoclass:: src.performance.cache.CacheManager
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.performance.cache.MemoryCache
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.performance.cache.RedisCache
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
~~~~~~~~~~~~~

.. autofunction:: src.performance.cache.cache_data

.. autofunction:: src.performance.cache.invalidate_cache

.. autofunction:: src.performance.cache.get_cache_stats

Usage Examples
~~~~~~~~~~~~~~

.. code-block:: python

   from src.performance.cache import cache_data, CacheManager
   
   # Use decorator for caching
   @cache_data(ttl=3600)  # Cache for 1 hour
   def expensive_computation(data):
       # Simulate expensive operation
       result = data.groupby('region').sum()
       return result
   
   # Use cache manager directly
   cache_manager = CacheManager()
   
   # Store data in cache
   cache_manager.set('user_data', user_data, ttl=1800)
   
   # Retrieve from cache
   cached_data = cache_manager.get('user_data')
   
   # Check cache statistics
   stats = cache_manager.get_stats()
   print(f"Cache hit rate: {stats['hit_rate']:.2%}")

Optimisation Module
-------------------

.. automodule:: src.performance.optimization
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
~~~~~~~~~~~

.. autoclass:: src.performance.optimization.QueryOptimizer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.performance.optimization.DataOptimizer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.performance.optimization.MemoryOptimizer
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
~~~~~~~~~~~~~~

.. code-block:: python

   from src.performance.optimization import (
       QueryOptimizer,
       DataOptimizer,
       MemoryOptimizer
   )
   
   # Optimise database queries
   query_optimizer = QueryOptimizer()
   optimised_query = query_optimizer.optimise_query(sql_query)
   
   # Optimise data processing
   data_optimizer = DataOptimizer()
   optimised_data = data_optimizer.optimise_dataframe(df)
   
   # Optimise memory usage
   memory_optimizer = MemoryOptimizer()
   memory_optimizer.cleanup_unused_data()
   memory_optimizer.optimise_memory_usage()

Alerts Module
-------------

.. automodule:: src.performance.alerts
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
~~~~~~~~~~~

.. autoclass:: src.performance.alerts.AlertManager
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.performance.alerts.PerformanceAlert
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.performance.alerts.HealthAlert
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
~~~~~~~~~~~~~~

.. code-block:: python

   from src.performance.alerts import AlertManager, PerformanceAlert
   
   # Create alert manager
   alert_manager = AlertManager()
   
   # Set up performance alerts
   cpu_alert = PerformanceAlert(
       metric='cpu_percent',
       threshold=80,
       message='High CPU usage detected'
   )
   
   memory_alert = PerformanceAlert(
       metric='memory_percent',
       threshold=85,
       message='High memory usage detected'
   )
   
   # Register alerts
   alert_manager.register_alert(cpu_alert)
   alert_manager.register_alert(memory_alert)
   
   # Check for alerts
   active_alerts = alert_manager.check_alerts()
   for alert in active_alerts:
       print(f"ALERT: {alert.message}")

Dashboard Module
----------------

.. automodule:: src.performance.dashboard
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
~~~~~~~~~~~~~

.. autofunction:: src.performance.dashboard.create_performance_dashboard

.. autofunction:: src.performance.dashboard.create_metrics_display

.. autofunction:: src.performance.dashboard.create_health_display

.. autofunction:: src.performance.dashboard.create_alerts_display

Usage Examples
~~~~~~~~~~~~~~

.. code-block:: python

   from src.performance.dashboard import create_performance_dashboard
   from src.performance.monitoring import PerformanceMonitor
   from src.performance.health import HealthChecker
   
   # Set up monitoring components
   monitor = PerformanceMonitor()
   health_checker = HealthChecker()
   
   # Create performance dashboard
   dashboard = create_performance_dashboard(monitor, health_checker)
   
   # The dashboard will automatically display:
   # - Real-time performance metrics
   # - System health status
   # - Active alerts
   # - Historical trends

Production Module
-----------------

.. automodule:: src.performance.production
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
~~~~~~~~~~~

.. autoclass:: src.performance.production.ProductionMonitor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.performance.production.LoadBalancer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.performance.production.HealthEndpoint
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
~~~~~~~~~~~~~~

.. code-block:: python

   from src.performance.production import (
       ProductionMonitor,
       LoadBalancer,
       HealthEndpoint
   )
   
   # Set up production monitoring
   prod_monitor = ProductionMonitor()
   
   # Start comprehensive monitoring
   prod_monitor.start_monitoring()
   
   # Set up load balancing
   load_balancer = LoadBalancer()
   load_balancer.configure_backends([
       'http://app1:8501',
       'http://app2:8501',
       'http://app3:8501'
   ])
   
   # Create health endpoint for load balancer checks
   health_endpoint = HealthEndpoint()
   health_status = health_endpoint.get_health_status()

Complete Performance Setup
--------------------------

Comprehensive Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.performance.monitoring import PerformanceMonitor
   from src.performance.health import HealthChecker
   from src.performance.alerts import AlertManager
   from src.performance.cache import CacheManager
   from src.performance.optimization import DataOptimizer
   
   # Set up all performance components
   monitor = PerformanceMonitor()
   health_checker = HealthChecker()
   alert_manager = AlertManager()
   cache_manager = CacheManager()
   optimizer = DataOptimizer()
   
   # Start monitoring
   monitor.start()
   
   # Configure alerts
   alert_manager.configure_default_alerts()
   
   # Optimise system
   optimizer.apply_optimizations()
   
   # Regular health checks
   def check_system_health():
       health_status = health_checker.check_all()
       alerts = alert_manager.check_alerts()
       
       if alerts:
           for alert in alerts:
               print(f"ALERT: {alert.message}")
       
       return health_status

Production Deployment
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.performance.production import ProductionMonitor
   from src.config import get_config, Environment
   
   # Get production configuration
   config = get_config(Environment.PRODUCTION)
   
   # Set up production monitoring
   prod_monitor = ProductionMonitor(config)
   
   # Start all production services
   prod_monitor.start_all_services()
   
   # The production monitor will:
   # - Monitor system health
   # - Collect performance metrics
   # - Send alerts when needed
   # - Optimise resource usage
   # - Provide health endpoints