API Reference
=============

This section provides detailed documentation for all modules, classes, and functions
in the Australian Health Analytics Dashboard.

The API is organised into several main packages:

.. toctree::
   :maxdepth: 2

   config
   dashboard
   performance

Core Modules
------------

Configuration Module
~~~~~~~~~~~~~~~~~~~~

.. automodule:: src.config
   :members:
   :undoc-members:
   :show-inheritance:

Dashboard Package
~~~~~~~~~~~~~~~~~

The dashboard package contains all components for the web interface:

.. automodule:: src.dashboard
   :members:
   :undoc-members:
   :show-inheritance:

Data Subpackage
~~~~~~~~~~~~~~~

.. automodule:: src.dashboard.data
   :members:
   :undoc-members:
   :show-inheritance:

Data Loaders
^^^^^^^^^^^^

.. automodule:: src.dashboard.data.loaders
   :members:
   :undoc-members:
   :show-inheritance:

Data Processors
^^^^^^^^^^^^^^^

.. automodule:: src.dashboard.data.processors
   :members:
   :undoc-members:
   :show-inheritance:UI Subpackage
~~~~~~~~~~~~~

.. automodule:: src.dashboard.ui
   :members:
   :undoc-members:
   :show-inheritance:

Layout Components
^^^^^^^^^^^^^^^^^

.. automodule:: src.dashboard.ui.layout
   :members:
   :undoc-members:
   :show-inheritance:

Page Components
^^^^^^^^^^^^^^^

.. automodule:: src.dashboard.ui.pages
   :members:
   :undoc-members:
   :show-inheritance:

Sidebar Components
^^^^^^^^^^^^^^^^^^

.. automodule:: src.dashboard.ui.sidebar
   :members:
   :undoc-members:
   :show-inheritance:

Visualisation Subpackage
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: src.dashboard.visualisation
   :members:
   :undoc-members:
   :show-inheritance:

Charts
^^^^^^

.. automodule:: src.dashboard.visualisation.charts
   :members:
   :undoc-members:
   :show-inheritance:

Components
^^^^^^^^^^

.. automodule:: src.dashboard.visualisation.components
   :members:
   :undoc-members:
   :show-inheritance:

Maps
^^^^

.. automodule:: src.dashboard.visualisation.maps
   :members:
   :undoc-members:
   :show-inheritance:

Performance Package
~~~~~~~~~~~~~~~~~~~

.. automodule:: src.performance
   :members:
   :undoc-members:
   :show-inheritance:

Alerts
^^^^^^

.. automodule:: src.performance.alerts
   :members:
   :undoc-members:
   :show-inheritance:

Caching
^^^^^^^

.. automodule:: src.performance.cache
   :members:
   :undoc-members:
   :show-inheritance:

Health Monitoring
^^^^^^^^^^^^^^^^^

.. automodule:: src.performance.health
   :members:
   :undoc-members:
   :show-inheritance:

Monitoring
^^^^^^^^^^

.. automodule:: src.performance.monitoring
   :members:
   :undoc-members:
   :show-inheritance:

Optimisation
^^^^^^^^^^^^

.. automodule:: src.performance.optimization
   :members:
   :undoc-members:
   :show-inheritance:

Scripts and Utilities
---------------------

The project includes various utility scripts for data processing and analysis.
These are documented separately as they are not part of the main API but are
important for understanding the data processing pipeline.

Data Processing Scripts
~~~~~~~~~~~~~~~~~~~~~~~

* :py:mod:`scripts.download_data` - Downloads health data from Australian sources
* :py:mod:`scripts.process_data` - Processes raw data into analysis-ready formats
* :py:mod:`scripts.geographic_mapping` - Handles geographic data processing
* :py:mod:`scripts.health_correlation_analysis` - Performs correlation analysis

Main Application
~~~~~~~~~~~~~~~~

.. automodule:: main
   :members:
   :undoc-members:

Examples
--------

Configuration Usage
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.config import get_config, Environment
   
   # Get default configuration
   config = get_config()
   
   # Get configuration for specific environment
   prod_config = get_config(Environment.PRODUCTION)
   
   # Access configuration values
   db_path = config.database.path
   data_dir = config.data_sources.base_path

Data Loading
~~~~~~~~~~~~

.. code-block:: python

   from src.dashboard.data.loaders import load_health_data, load_geographic_data
   from src.config import get_config
   
   config = get_config()
   
   # Load health data
   health_data = load_health_data(config.data_sources.health_data_path)
   
   # Load geographic boundaries
   boundaries = load_geographic_data(config.data_sources.geographic_data_path)

Dashboard Creation
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.dashboard.app import create_dashboard
   from src.dashboard.data.loaders import load_all_data
   
   # Load all required data
   data = load_all_data()
   
   # Create and configure dashboard
   dashboard = create_dashboard(data)

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.performance.monitoring import PerformanceMonitor
   from src.performance.health import HealthChecker
   
   # Set up monitoring
   monitor = PerformanceMonitor()
   health_checker = HealthChecker()
   
   # Start monitoring
   monitor.start()
   
   # Check system health
   health_status = health_checker.check_all()

Error Handling
~~~~~~~~~~~~~~

The API uses custom exceptions for better error handling:

.. code-block:: python

   from src.config import ConfigurationError
   from src.dashboard.data.exceptions import DataLoadError
   
   try:
       config = get_config()
       data = load_health_data(config.data_sources.health_data_path)
   except ConfigurationError as e:
       print(f"Configuration error: {e}")
   except DataLoadError as e:
       print(f"Data loading error: {e}")