Configuration Reference
======================

This reference provides comprehensive information about configuring the
Australian Health Analytics Dashboard.

Configuration Files
-------------------

The AHGD platform uses several configuration files:

**pyproject.toml**
  Main project configuration including dependencies, build settings, and tool configurations

**.env**
  Environment variables for runtime configuration (create from .env.template)

**src/config.py**
  Application configuration classes and settings

Environment Variables
---------------------

Core Settings
~~~~~~~~~~~~~

.. envvar:: AHGD_ENV

   Deployment environment
   
   :Default: ``development``
   :Options: ``development``, ``staging``, ``production``
   :Example: ``AHGD_ENV=production``

.. envvar:: DATABASE_PATH

   Path to SQLite database file
   
   :Default: ``./health_analytics.db``
   :Example: ``DATABASE_PATH=/opt/ahgd/data/analytics.db``

.. envvar:: DATA_DIR

   Root directory for data files
   
   :Default: ``./data``
   :Example: ``DATA_DIR=/opt/ahgd/data``

Dashboard Settings
~~~~~~~~~~~~~~~~~~

.. envvar:: DASHBOARD_HOST

   Host address for the Streamlit dashboard
   
   :Default: ``localhost``
   :Example: ``DASHBOARD_HOST=0.0.0.0``

.. envvar:: DASHBOARD_PORT

   Port number for the dashboard
   
   :Default: ``8501``
   :Example: ``DASHBOARD_PORT=80``

.. envvar:: DASHBOARD_TITLE

   Custom title for the dashboard
   
   :Default: ``Australian Health Analytics Dashboard``
   :Example: ``DASHBOARD_TITLE=NSW Health Analytics``

Data Processing Settings
~~~~~~~~~~~~~~~~~~~~~~~~

.. envvar:: PROCESSING_BATCH_SIZE

   Batch size for data processing operations
   
   :Default: ``10000``
   :Example: ``PROCESSING_BATCH_SIZE=50000``

.. envvar:: PROCESSING_WORKERS

   Number of worker processes for parallel processing
   
   :Default: ``4``
   :Example: ``PROCESSING_WORKERS=8``

.. envvar:: CACHE_TTL

   Time-to-live for cached data (seconds)
   
   :Default: ``3600``
   :Example: ``CACHE_TTL=7200``

Logging Settings
~~~~~~~~~~~~~~~~

.. envvar:: LOG_LEVEL

   Logging level
   
   :Default: ``INFO``
   :Options: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``
   :Example: ``LOG_LEVEL=DEBUG``

.. envvar:: LOG_FILE

   Path to log file
   
   :Default: ``./logs/ahgd.log``
   :Example: ``LOG_FILE=/var/log/ahgd/application.log``

.. envvar:: LOG_FORMAT

   Log message format
   
   :Default: ``%(asctime)s - %(name)s - %(levelname)s - %(message)s``

Performance Settings
~~~~~~~~~~~~~~~~~~~~

.. envvar:: PERFORMANCE_MONITORING

   Enable performance monitoring
   
   :Default: ``true``
   :Options: ``true``, ``false``
   :Example: ``PERFORMANCE_MONITORING=false``

.. envvar:: MEMORY_LIMIT_MB

   Memory limit for the application (MB)
   
   :Default: ``8192``
   :Example: ``MEMORY_LIMIT_MB=16384``

.. envvar:: CPU_LIMIT_PERCENT

   CPU usage limit (percentage)
   
   :Default: ``80``
   :Example: ``CPU_LIMIT_PERCENT=90``

Security Settings
~~~~~~~~~~~~~~~~~

.. envvar:: SECRET_KEY

   Secret key for session management
   
   :Default: Generated automatically
   :Example: ``SECRET_KEY=your-secret-key-here``

.. envvar:: ALLOWED_HOSTS

   Comma-separated list of allowed host names
   
   :Default: ``localhost,127.0.0.1``
   :Example: ``ALLOWED_HOSTS=localhost,dashboard.example.com``

Configuration Classes
---------------------

Main Configuration
~~~~~~~~~~~~~~~~~~

The main configuration is handled by the ``Config`` class:

.. code-block:: python

   from src.config import get_config, Environment
   
   # Get configuration for current environment
   config = get_config()
   
   # Get configuration for specific environment
   prod_config = get_config(Environment.PRODUCTION)
   
   # Access configuration values
   print(f"Database: {config.database.path}")
   print(f"Environment: {config.environment}")

Database Configuration
~~~~~~~~~~~~~~~~~~~~~~

Database settings are managed by ``DatabaseConfig``:

.. code-block:: python

   from src.config import DatabaseConfig
   
   db_config = DatabaseConfig(
       name="custom_analytics.db",
       connection_timeout=60,
       backup_enabled=True
   )
   
   print(f"Connection string: {db_config.connection_string}")

Data Source Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

Data source paths are configured via ``DataSourceConfig``:

.. code-block:: python

   from src.config import DataSourceConfig
   from pathlib import Path
   
   data_config = DataSourceConfig(
       base_path=Path("/opt/ahgd/data"),
       health_data_subdir="health",
       geographic_data_subdir="geographic"
   )

Dashboard Configuration
~~~~~~~~~~~~~~~~~~~~~~~

Dashboard settings use ``DashboardConfig``:

.. code-block:: python

   from src.config import DashboardConfig
   
   dashboard_config = DashboardConfig(
       host="0.0.0.0",
       port=8501,
       title="Custom Health Dashboard",
       theme="light",
       sidebar_state="expanded"
   )

Processing Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

Data processing settings via ``ProcessingConfig``:

.. code-block:: python

   from src.config import ProcessingConfig
   
   processing_config = ProcessingConfig(
       batch_size=50000,
       max_workers=8,
       chunk_size=10000,
       parallel_enabled=True
   )

Logging Configuration
~~~~~~~~~~~~~~~~~~~~~

Logging is configured with ``LoggingConfig``:

.. code-block:: python

   from src.config import LoggingConfig, setup_logging
   import logging
   
   log_config = LoggingConfig(
       level=logging.INFO,
       format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
       file_path="logs/ahgd.log",
       max_file_size_mb=100,
       backup_count=5
   )
   
   # Set up logging
   logger = setup_logging(log_config)

Configuration Validation
-------------------------

The configuration system includes validation to ensure settings are correct:

.. code-block:: python

   from src.config import get_config, ConfigurationError
   
   try:
       config = get_config()
       print("Configuration is valid")
   except ConfigurationError as e:
       print(f"Configuration error: {e}")
       # Handle configuration error

Custom Configuration
--------------------

Creating Custom Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can create custom configurations for specific environments:

.. code-block:: python

   from src.config import (
       Config, DatabaseConfig, DataSourceConfig,
       DashboardConfig, Environment
   )
   from pathlib import Path
   
   # Create custom database configuration
   custom_db = DatabaseConfig(
       name="test_analytics.db",
       path=Path("./test_data/test_analytics.db"),
       connection_timeout=30
   )
   
   # Create custom data source configuration
   custom_data = DataSourceConfig(
       base_path=Path("./test_data"),
       health_data_subdir="test_health",
       geographic_data_subdir="test_geo"
   )
   
   # Create custom dashboard configuration
   custom_dashboard = DashboardConfig(
       host="localhost",
       port=8502,
       title="Test Dashboard",
       theme="dark"
   )
   
   # Combine into custom configuration
   custom_config = Config(
       environment=Environment.DEVELOPMENT,
       database=custom_db,
       data_sources=custom_data,
       dashboard=custom_dashboard
   )

Environment-Specific Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different environments can have different default settings:

.. code-block:: python

   import os
   from src.config import get_config, Environment
   
   # Set environment
   os.environ['AHGD_ENV'] = 'production'
   
   # Get environment-specific configuration
   config = get_config()
   
   # Production environment will have:
   # - More restrictive security settings
   # - Optimised performance settings
   # - Reduced logging verbosity
   # - Enhanced monitoring

Configuration Files Examples
----------------------------

.env File Example
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Environment
   AHGD_ENV=production
   
   # Database
   DATABASE_PATH=/opt/ahgd/data/analytics.db
   
   # Data directories
   DATA_DIR=/opt/ahgd/data
   PROCESSED_DATA_DIR=/opt/ahgd/data/processed
   RAW_DATA_DIR=/opt/ahgd/data/raw
   
   # Dashboard
   DASHBOARD_HOST=0.0.0.0
   DASHBOARD_PORT=80
   DASHBOARD_TITLE=Australian Health Analytics
   
   # Performance
   PROCESSING_BATCH_SIZE=50000
   PROCESSING_WORKERS=8
   CACHE_TTL=7200
   
   # Logging
   LOG_LEVEL=INFO
   LOG_FILE=/var/log/ahgd/application.log
   
   # Security
   SECRET_KEY=your-production-secret-key
   ALLOWED_HOSTS=localhost,dashboard.health.gov.au

Docker Environment File
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Docker-specific configuration
   AHGD_ENV=production
   DATABASE_PATH=/app/data/analytics.db
   DATA_DIR=/app/data
   DASHBOARD_HOST=0.0.0.0
   DASHBOARD_PORT=8501
   LOG_LEVEL=INFO
   PROCESSING_WORKERS=4
   MEMORY_LIMIT_MB=4096

Kubernetes ConfigMap
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: ahgd-config
   data:
     AHGD_ENV: "production"
     DATABASE_PATH: "/data/analytics.db"
     DATA_DIR: "/data"
     DASHBOARD_HOST: "0.0.0.0"
     DASHBOARD_PORT: "8501"
     LOG_LEVEL: "INFO"
     PROCESSING_WORKERS: "4"
     CACHE_TTL: "3600"

Configuration Best Practices
-----------------------------

Security
~~~~~~~~

1. **Never commit secrets to version control**
   
   * Use .env files for local development
   * Use secure secret management in production
   * Rotate secrets regularly

2. **Use environment-specific settings**
   
   * Different security levels for different environments
   * Appropriate logging levels
   * Suitable performance settings

3. **Validate configuration on startup**
   
   * Check required settings are present
   * Validate file paths and permissions
   * Test database connectivity

Performance
~~~~~~~~~~~

1. **Tune for your environment**
   
   * Adjust worker counts based on CPU cores
   * Set appropriate memory limits
   * Configure caching based on available memory

2. **Monitor configuration impact**
   
   * Track performance metrics
   * Adjust settings based on usage patterns
   * Test configuration changes in staging

3. **Use appropriate batch sizes**
   
   * Larger batches for more memory
   * Smaller batches for limited resources
   * Balance between memory usage and performance

Maintenance
~~~~~~~~~~~

1. **Document configuration changes**
   
   * Keep configuration changelog
   * Document environment-specific requirements
   * Include rollback procedures

2. **Test configuration changes**
   
   * Use staging environment for testing
   * Validate before production deployment
   * Have rollback plan ready

3. **Regular configuration review**
   
   * Review settings periodically
   * Update based on new requirements
   * Remove obsolete settings

Troubleshooting Configuration
-----------------------------

Common Issues
~~~~~~~~~~~~~

**Configuration Not Loading**

Check the configuration loading process:

.. code-block:: python

   import os
   from src.config import get_config
   
   # Check environment variables
   print("Environment variables:")
   for key, value in os.environ.items():
       if key.startswith('AHGD_') or key in ['DATABASE_PATH', 'DATA_DIR']:
           print(f"  {key}={value}")
   
   # Check configuration loading
   try:
       config = get_config()
       print(f"Configuration loaded successfully")
       print(f"Environment: {config.environment}")
       print(f"Database: {config.database.path}")
   except Exception as e:
       print(f"Configuration error: {e}")

**Path Issues**

Verify file and directory paths:

.. code-block:: python

   from src.config import get_config
   import os
   
   config = get_config()
   
   # Check database path
   db_path = config.database.path
   print(f"Database path: {db_path}")
   print(f"Database exists: {db_path.exists()}")
   print(f"Database parent exists: {db_path.parent.exists()}")
   
   # Check data directories
   data_dir = config.data_sources.base_path
   print(f"Data directory: {data_dir}")
   print(f"Data directory exists: {data_dir.exists()}")
   print(f"Data directory writable: {os.access(data_dir, os.W_OK)}")

**Permission Issues**

Check file and directory permissions:

.. code-block:: bash

   # Check database permissions
   ls -la health_analytics.db
   
   # Check data directory permissions
   ls -la data/
   
   # Check log directory permissions
   ls -la logs/

**Environment Variable Issues**

Debug environment variable loading:

.. code-block:: python

   import os
   from dotenv import load_dotenv
   
   # Load .env file explicitly
   load_dotenv()
   
   # Check specific variables
   variables_to_check = [
       'AHGD_ENV', 'DATABASE_PATH', 'DATA_DIR',
       'DASHBOARD_HOST', 'DASHBOARD_PORT', 'LOG_LEVEL'
   ]
   
   for var in variables_to_check:
       value = os.getenv(var)
       print(f"{var}: {value if value else 'NOT SET'}")

Getting Help
~~~~~~~~~~~~

If you continue to have configuration issues:

1. Check the :doc:`troubleshooting` guide
2. Verify your environment matches the requirements
3. Check the logs for specific error messages
4. Consult the :doc:`../api/config` API documentation
5. Contact the development team with specific error details