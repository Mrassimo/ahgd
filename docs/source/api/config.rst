Configuration Module
===================

.. automodule:: src.config
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Classes
---------------------

Main Configuration
~~~~~~~~~~~~~~~~~~

.. autoclass:: src.config.Config
   :members:
   :undoc-members:
   :show-inheritance:

Database Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: src.config.DatabaseConfig
   :members:
   :undoc-members:
   :show-inheritance:

Data Source Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: src.config.DataSourceConfig
   :members:
   :undoc-members:
   :show-inheritance:

Dashboard Configuration
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: src.config.DashboardConfig
   :members:
   :undoc-members:
   :show-inheritance:

Processing Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: src.config.ProcessingConfig
   :members:
   :undoc-members:
   :show-inheritance:

Logging Configuration
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: src.config.LoggingConfig
   :members:
   :undoc-members:
   :show-inheritance:

Environment Enumeration
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: src.config.Environment
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Functions
-----------------------

.. autofunction:: src.config.get_config

.. autofunction:: src.config.get_global_config

.. autofunction:: src.config.get_project_root

.. autofunction:: src.config.setup_logging

.. autofunction:: src.config.reset_global_config

Usage Examples
--------------

Basic Configuration Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.config import get_config, Environment
   
   # Get default configuration
   config = get_config()
   
   # Access database configuration
   print(f"Database path: {config.database.path}")
   print(f"Connection string: {config.database.connection_string}")
   
   # Access data source configuration
   print(f"Data directory: {config.data_sources.base_path}")
   print(f"Health data: {config.data_sources.health_data_path}")

Environment-Specific Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.config import get_config, Environment
   
   # Production configuration
   prod_config = get_config(Environment.PRODUCTION)
   
   # Development configuration
   dev_config = get_config(Environment.DEVELOPMENT)
   
   # Staging configuration
   staging_config = get_config(Environment.STAGING)

Custom Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.config import Config, DatabaseConfig, get_project_root
   
   # Create custom database configuration
   custom_db = DatabaseConfig(
       name="custom_analytics.db",
       path=get_project_root() / "custom_data" / "custom_analytics.db",
       connection_timeout=60,
       backup_enabled=True
   )
   
   # Create custom configuration
   custom_config = Config(database=custom_db)

Logging Setup
~~~~~~~~~~~~~

.. code-block:: python

   from src.config import setup_logging, get_config
   
   # Set up logging with default configuration
   config = get_config()
   logger = setup_logging(config.logging)
   
   # Use the logger
   logger.info("Application started")
   logger.warning("This is a warning")
   logger.error("This is an error")

Configuration Validation
~~~~~~~~~~~~~~~~~~~~~~~~~

The configuration system includes validation to ensure all required settings
are properly configured:

.. code-block:: python

   from src.config import get_config, ConfigurationError
   
   try:
       config = get_config()
       # Configuration is automatically validated
       print("Configuration is valid")
   except ConfigurationError as e:
       print(f"Configuration error: {e}")

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

The configuration system supports environment variables for deployment flexibility:

.. code-block:: bash

   # Set environment variables
   export AHGD_ENV=production
   export DATABASE_PATH=/opt/ahgd/data/analytics.db
   export DATA_DIR=/opt/ahgd/data
   export LOG_LEVEL=INFO

.. code-block:: python

   # These will be automatically picked up by get_config()
   config = get_config()
   print(f"Environment: {config.environment}")
   print(f"Database: {config.database.path}")