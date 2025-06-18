Getting Started
===============

This guide will help you get the Australian Health Analytics Dashboard up and running
on your local machine or in a production environment.

Prerequisites
-------------

System Requirements
~~~~~~~~~~~~~~~~~~~

* Python 3.11 or higher
* 8GB+ RAM (recommended for processing large datasets)
* 5GB+ available disk space
* Internet connection for data downloads

Required Software
~~~~~~~~~~~~~~~~~

* `UV <https://docs.astral.sh/uv/>`_ (recommended) or pip for package management
* Git for version control
* A modern web browser for the dashboard interface

Installation
------------

1. Clone the Repository
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/your-org/ahgd.git
   cd ahgd

2. Set Up Python Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using UV (Recommended):

.. code-block:: bash

   # Install UV if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install the project and dependencies
   uv pip install -e .

   # Install development dependencies (optional)
   uv pip install -e .[dev]

Using pip:

.. code-block:: bash   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install the project
   pip install -e .

   # Install development dependencies (optional)
   pip install -e .[dev]

3. Configure Environment
~~~~~~~~~~~~~~~~~~~~~~~~~

Create a `.env` file in the project root (copy from `.env.template` if available):

.. code-block:: bash

   # Database configuration
   DATABASE_PATH=./health_analytics.db
   
   # Dashboard configuration
   DASHBOARD_HOST=localhost
   DASHBOARD_PORT=8501
   
   # Data paths
   DATA_DIR=./data
   PROCESSED_DATA_DIR=./data/processed
   RAW_DATA_DIR=./data/raw

4. Download and Process Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The platform includes scripts to download and process Australian health data:

.. code-block:: bash

   # Download health data from Australian sources
   python scripts/download_data.py

   # Process the downloaded data
   python scripts/process_data.py

   # Verify data integrity
   python verify_data.py

5. Run the Dashboard
~~~~~~~~~~~~~~~~~~~~

Start the Streamlit dashboard:

.. code-block:: bash

   # Run the main dashboard
   python run_dashboard.py

   # Or run with custom configuration
   streamlit run src/dashboard/app.py --server.port 8501

The dashboard will be available at `http://localhost:8501`.

Configuration
-------------

The platform uses a centralised configuration system located in `src/config.py`.

Key Configuration Options
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.config import get_config

   config = get_config()
   
   # Database settings
   print(config.database.path)
   print(config.database.connection_string)
   
   # Data source paths
   print(config.data_sources.health_data_path)
   print(config.data_sources.geographic_data_path)
   
   # Dashboard settings
   print(config.dashboard.host)
   print(config.dashboard.port)

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

The following environment variables can be used to override default settings:

* `AHGD_ENV`: Environment (development, staging, production)
* `DATABASE_PATH`: Path to SQLite database file
* `DATA_DIR`: Root directory for data files
* `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

Development Setup
-----------------

For development work, install the additional development dependencies:

.. code-block:: bash

   # Install development dependencies
   uv pip install -e .[dev,test]

   # Set up pre-commit hooks
   pre-commit install

   # Run tests
   python -m pytest

   # Run linting
   ruff check src/ tests/
   
   # Run type checking
   mypy src/

Testing
-------

The project includes comprehensive tests located in the `tests/` directory:

.. code-block:: bash

   # Run all tests
   python -m pytest

   # Run with coverage report
   python -m pytest --cov=src --cov-report=html

   # Run specific test categories
   python -m pytest -m "unit"
   python -m pytest -m "integration"

Common Issues
-------------

Data Download Issues
~~~~~~~~~~~~~~~~~~~~

If you encounter issues downloading data:

1. Check your internet connection
2. Verify that the data source URLs are accessible
3. Check the logs in `logs/data_download.log`

Dashboard Not Loading
~~~~~~~~~~~~~~~~~~~~~

If the dashboard doesn't load:

1. Ensure all dependencies are installed
2. Check that the database exists and is accessible
3. Verify that the port (8501) is not in use by another application
4. Check the logs for error messages

Performance Issues
~~~~~~~~~~~~~~~~~~

If the dashboard is slow:

1. Ensure you have sufficient RAM (8GB+ recommended)
2. Check that processed data files exist in `data/processed/`
3. Monitor performance using the built-in performance dashboard

Next Steps
----------

* Explore the :doc:`tutorials/index` for detailed walkthroughs
* Read the :doc:`api/index` for technical details
* Check the :doc:`guides/index` for advanced usage patterns
* See :doc:`deployment/index` for production deployment guidance