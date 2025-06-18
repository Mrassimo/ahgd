Australian Health Analytics Dashboard Documentation
==================================================

Welcome to the Australian Health Analytics Dashboard (AHGD) documentation.
This comprehensive platform provides health analytics and visualisation capabilities
for Australian health data, including demographic, geographic, and health outcome analysis.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api/index
   guides/index
   tutorials/index
   deployment/index
   reference/index

Quick Start
-----------

The Australian Health Analytics Dashboard is a production-ready analytics platform
that combines health data from multiple Australian sources including:

* Australian Institute of Health and Welfare (AIHW)
* Public Health Information Development Unit (PHIDU)
* Australian Bureau of Statistics (ABS)
* Socio-Economic Indexes for Areas (SEIFA)

Key Features
~~~~~~~~~~~~

* **Interactive Dashboards**: Streamlit-based web interface for data exploration
* **Geographic Visualisation**: Interactive maps showing health data by SA2 regions
* **Performance Monitoring**: Real-time performance metrics and health checks
* **Automated Data Processing**: ETL pipelines for Australian health datasets
* **Comprehensive Testing**: Full test suite with 85%+ code coverage

Architecture Overview
~~~~~~~~~~~~~~~~~~~~

The platform is organised into several key modules:

* **Data Module**: Data loading, processing, and transformation
* **Dashboard Module**: Web interface and user interactions
* **Visualisation Module**: Charts, maps, and interactive components
* **Performance Module**: Monitoring, caching, and optimisation
* **Configuration**: Centralised configuration management

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-org/ahgd.git
   cd ahgd

   # Install with uv (recommended)
   uv pip install -e .

   # Or install with pip
   pip install -e .

Quick Example
~~~~~~~~~~~~~

.. code-block:: python

   from src.config import get_config
   from src.dashboard.data.loaders import load_health_data
   from src.dashboard.app import create_dashboard

   # Load configuration
   config = get_config()

   # Load health data
   data = load_health_data(config.data_sources.health_data_path)

   # Create dashboard
   dashboard = create_dashboard(data)

API Reference
~~~~~~~~~~~~~

The API documentation provides detailed information about all modules, classes,
and functions in the AHGD platform.

.. autosummary::
   :toctree: api/
   :template: module.rst

   src.config
   src.dashboard
   src.performance

Contributing
~~~~~~~~~~~~

We welcome contributions to the Australian Health Analytics Dashboard.
Please see our :doc:`guides/contributing` guide for details on how to contribute.

License
~~~~~~~

This project is licensed under the MIT License. See the LICENSE file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`