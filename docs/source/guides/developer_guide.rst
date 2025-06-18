Developer Guide
===============

This guide provides comprehensive information for developers working on the
Australian Health Analytics Dashboard (AHGD) project.

Architecture Overview
---------------------

System Architecture
~~~~~~~~~~~~~~~~~~~

The AHGD follows a modular, layered architecture designed for scalability,
maintainability, and testability:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────┐
   │                    Presentation Layer                    │
   │  ┌─────────────────┐  ┌─────────────────┐              │
   │  │   Streamlit UI  │  │  Performance    │              │
   │  │   Components    │  │   Dashboard     │              │
   │  └─────────────────┘  └─────────────────┘              │
   └─────────────────────────────────────────────────────────┘
   ┌─────────────────────────────────────────────────────────┐
   │                    Application Layer                     │
   │  ┌─────────────────┐  ┌─────────────────┐              │
   │  │   Dashboard     │  │  Visualisation  │              │
   │  │   Logic         │  │   Components    │              │
   │  └─────────────────┘  └─────────────────┘              │
   └─────────────────────────────────────────────────────────┘
   ┌─────────────────────────────────────────────────────────┐
   │                     Business Layer                      │
   │  ┌─────────────────┐  ┌─────────────────┐              │
   │  │   Data          │  │   Performance   │              │
   │  │   Processing    │  │   Monitoring    │              │
   │  └─────────────────┘  └─────────────────┘              │
   └─────────────────────────────────────────────────────────┘
   ┌─────────────────────────────────────────────────────────┐
   │                      Data Layer                         │
   │  ┌─────────────────┐  ┌─────────────────┐              │
   │  │   Data          │  │   Configuration │              │
   │  │   Loaders       │  │   Management    │              │
   │  └─────────────────┘  └─────────────────┘              │
   └─────────────────────────────────────────────────────────┘

Package Structure
~~~~~~~~~~~~~~~~~

The codebase is organised into logical packages:

.. code-block:: text

   src/
   ├── config.py                 # Configuration management
   ├── dashboard/                # Main dashboard package
   │   ├── app.py               # Main application entry point
   │   ├── data/                # Data handling
   │   │   ├── loaders.py       # Data loading functions
   │   │   └── processors.py    # Data processing functions
   │   ├── ui/                  # User interface components
   │   │   ├── layout.py        # Page layouts
   │   │   ├── pages.py         # Page components
   │   │   └── sidebar.py       # Sidebar components
   │   └── visualisation/       # Visualisation components
   │       ├── charts.py        # Chart creation functions
   │       ├── components.py    # UI components
   │       └── maps.py          # Map visualisations
   └── performance/             # Performance monitoring
       ├── alerts.py            # Alert system
       ├── cache.py             # Caching mechanisms
       ├── health.py            # Health checks
       ├── monitoring.py        # Performance monitoring
       └── optimization.py      # Performance optimisation

Development Environment Setup
-----------------------------

Prerequisites
~~~~~~~~~~~~~

* Python 3.11 or higher
* Git
* UV package manager (recommended) or pip
* IDE with Python support (VS Code, PyCharm, etc.)

Initial Setup
~~~~~~~~~~~~~

1. **Clone Repository**

.. code-block:: bash

   git clone https://github.com/your-org/ahgd.git
   cd ahgd

2. **Set Up Development Environment**

.. code-block:: bash

   # Install UV if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install project with development dependencies
   uv pip install -e .[dev,test]

3. **Configure Pre-commit Hooks**

.. code-block:: bash

   pre-commit install

4. **Set Up Environment Variables**

.. code-block:: bash

   cp .env.template .env
   # Edit .env with your local settings

5. **Run Initial Tests**

.. code-block:: bash

   python -m pytest
   python -m pytest --cov=src --cov-report=html

Development Workflow
-------------------

Branch Strategy
~~~~~~~~~~~~~~~

The project uses a Git Flow branching strategy:

* **main**: Production-ready code
* **develop**: Integration branch for features
* **feature/***: Feature development branches
* **bugfix/***: Bug fix branches
* **release/***: Release preparation branches
* **hotfix/***: Critical production fixes

Feature Development Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Create Feature Branch**

.. code-block:: bash

   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name

2. **Develop Feature**

   * Write code following project standards
   * Add comprehensive tests
   * Update documentation
   * Ensure code passes all quality checks

3. **Test Thoroughly**

.. code-block:: bash

   # Run all tests
   python -m pytest

   # Check code coverage
   python -m pytest --cov=src --cov-report=term-missing

   # Run linting
   ruff check src/ tests/
   black --check src/ tests/

   # Type checking
   mypy src/

4. **Create Pull Request**

   * Push branch to remote
   * Create pull request to develop branch
   * Add descriptive title and description
   * Request review from team members

Code Standards
--------------

Python Style Guide
~~~~~~~~~~~~~~~~~~

The project follows PEP 8 with some modifications:

* **Line Length**: 100 characters (configured in pyproject.toml)
* **Import Organisation**: Use isort with force_single_line = true
* **String Formatting**: Prefer f-strings for Python 3.6+
* **Type Hints**: Required for all public functions and methods

Example of well-formatted code:

.. code-block:: python

   from typing import Dict, List, Optional, Union
   import pandas as pd
   from src.config import get_config

   def process_health_data(
       data: pd.DataFrame,
       region_filter: Optional[List[str]] = None,
       time_period: Optional[str] = None,
   ) -> Dict[str, Union[pd.DataFrame, Dict[str, float]]]:
       """
       Process health data with optional filtering.
       
       Args:
           data: Raw health data DataFrame
           region_filter: List of regions to include
           time_period: Time period filter (e.g., '2020-2023')
           
       Returns:
           Dictionary containing processed data and summary statistics
           
       Raises:
           ValueError: If data is empty or invalid
           ProcessingError: If processing fails
       """
       if data.empty:
           raise ValueError("Input data cannot be empty")
       
       config = get_config()
       processed_data = data.copy()
       
       # Apply filters
       if region_filter:
           processed_data = processed_data[
               processed_data['region'].isin(region_filter)
           ]
       
       if time_period:
           start_date, end_date = parse_time_period(time_period)
           processed_data = processed_data[
               (processed_data['date'] >= start_date) &
               (processed_data['date'] <= end_date)
           ]
       
       # Calculate summary statistics
       summary_stats = {
           'total_records': len(processed_data),
           'unique_regions': processed_data['region'].nunique(),
           'date_range': (
               processed_data['date'].min(),
               processed_data['date'].max()
           )
       }
       
       return {
           'data': processed_data,
           'summary': summary_stats
       }

Documentation Standards
~~~~~~~~~~~~~~~~~~~~~~~

All code must include comprehensive documentation:

**Docstring Format**: Use Google-style docstrings

.. code-block:: python

   def example_function(param1: str, param2: int = 10) -> bool:
       """
       Brief description of the function.
       
       Longer description if needed, explaining the purpose,
       algorithm, or important considerations.
       
       Args:
           param1: Description of parameter 1
           param2: Description of parameter 2 with default value
           
       Returns:
           Description of return value
           
       Raises:
           ValueError: When param1 is empty
           ProcessingError: When processing fails
           
       Example:
           >>> result = example_function("test", 20)
           >>> print(result)
           True
       """

**Module Documentation**: Include module-level docstrings

.. code-block:: python

   """
   Health Data Processing Module
   
   This module provides functions for loading, processing, and analysing
   Australian health data from various sources including AIHW, PHIDU,
   and ABS datasets.
   
   The module handles:
   * Data loading from multiple formats (CSV, Excel, Parquet)
   * Data cleaning and validation
   * Geographic data processing and mapping
   * Statistical analysis and aggregation
   
   Example:
       >>> from src.dashboard.data.processors import process_health_data
       >>> data = load_health_data('path/to/data.csv')
       >>> processed = process_health_data(data)
   """

Testing Standards
-----------------

Test Structure
~~~~~~~~~~~~~~

Tests are organised to mirror the source code structure:

.. code-block:: text

   tests/
   ├── unit/                    # Unit tests
   │   ├── test_config.py
   │   ├── test_data_processing.py
   │   └── test_ui_components.py
   ├── integration/             # Integration tests
   │   ├── test_dashboard_integration.py
   │   └── test_database_operations.py
   ├── fixtures/                # Test fixtures and data
   │   └── sample_data.py
   └── conftest.py             # Pytest configuration

Test Categories
~~~~~~~~~~~~~~~

**Unit Tests**: Test individual functions and methods

.. code-block:: python

   import pytest
   import pandas as pd
   from src.dashboard.data.processors import process_health_data
   
   class TestProcessHealthData:
       """Test cases for process_health_data function."""
       
       def test_process_health_data_basic(self, sample_health_data):
           """Test basic data processing functionality."""
           result = process_health_data(sample_health_data)
           
           assert 'data' in result
           assert 'summary' in result
           assert isinstance(result['data'], pd.DataFrame)
           assert len(result['data']) > 0
       
       def test_process_health_data_with_region_filter(self, sample_health_data):
           """Test data processing with region filtering."""
           regions = ['NSW', 'VIC']
           result = process_health_data(sample_health_data, region_filter=regions)
           
           assert all(
               region in regions 
               for region in result['data']['region'].unique()
           )
       
       def test_process_health_data_empty_input(self):
           """Test handling of empty input data."""
           empty_df = pd.DataFrame()
           
           with pytest.raises(ValueError, match="Input data cannot be empty"):
               process_health_data(empty_df)

**Integration Tests**: Test component interactions

.. code-block:: python

   import pytest
   from src.dashboard.app import create_dashboard
   from src.dashboard.data.loaders import load_all_data
   from src.config import get_config
   
   class TestDashboardIntegration:
       """Integration tests for dashboard components."""
       
       @pytest.fixture
       def dashboard_config(self):
           """Create test configuration."""
           return get_config(environment='test')
       
       def test_dashboard_creation_with_real_data(self, dashboard_config):
           """Test dashboard creation with actual data."""
           data = load_all_data(dashboard_config)
           dashboard = create_dashboard(data, dashboard_config)
           
           assert dashboard is not None
           assert hasattr(dashboard, 'render')
           
       def test_dashboard_performance_monitoring(self, dashboard_config):
           """Test performance monitoring integration."""
           data = load_all_data(dashboard_config)
           dashboard = create_dashboard(data, dashboard_config)
           
           # Test that performance monitoring is active
           assert dashboard.performance_monitor.is_active()
           
           # Test health checks
           health_status = dashboard.health_checker.check_all()
           assert health_status['overall']['healthy']

Test Fixtures
~~~~~~~~~~~~~

Use fixtures for reusable test data:

.. code-block:: python

   # tests/fixtures/sample_data.py
   import pytest
   import pandas as pd
   import numpy as np
   
   @pytest.fixture
   def sample_health_data():
       """Create sample health data for testing."""
       np.random.seed(42)
       
       data = pd.DataFrame({
           'region': np.random.choice(['NSW', 'VIC', 'QLD', 'SA', 'WA'], 1000),
           'date': pd.date_range('2020-01-01', periods=1000, freq='D'),
           'health_metric': np.random.normal(75, 10, 1000),
           'population': np.random.randint(1000, 100000, 1000),
           'age_group': np.random.choice(['0-17', '18-64', '65+'], 1000)
       })
       
       return data
   
   @pytest.fixture
   def sample_geographic_data():
       """Create sample geographic data for testing."""
       return pd.DataFrame({
           'sa2_code': ['101011001', '101011002', '101011003'],
           'sa2_name': ['Region A', 'Region B', 'Region C'],
           'state': ['NSW', 'NSW', 'VIC'],
           'area_sqkm': [10.5, 15.2, 8.7],
           'population': [5000, 7500, 3200]
       })

Performance Testing
~~~~~~~~~~~~~~~~~~~

Include performance tests for critical functions:

.. code-block:: python

   import pytest
   import time
   from src.dashboard.data.loaders import load_large_dataset
   
   class TestPerformance:
       """Performance tests for data processing functions."""
       
       @pytest.mark.slow
       def test_large_dataset_loading_performance(self):
           """Test performance of loading large datasets."""
           start_time = time.time()
           
           # Load large dataset
           data = load_large_dataset(size=100000)
           
           load_time = time.time() - start_time
           
           # Assert reasonable performance
           assert load_time < 10.0  # Should load in under 10 seconds
           assert len(data) == 100000
       
       def test_data_processing_memory_usage(self):
           """Test memory usage during data processing."""
           import psutil
           import os
           
           process = psutil.Process(os.getpid())
           initial_memory = process.memory_info().rss
           
           # Process data
           large_data = create_large_test_dataset(size=50000)
           processed_data = process_health_data(large_data)
           
           final_memory = process.memory_info().rss
           memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
           
           # Assert reasonable memory usage
           assert memory_increase < 500  # Less than 500MB increase

Performance Considerations
--------------------------

Code Optimisation
~~~~~~~~~~~~~~~~~

**Data Processing**

* Use vectorised operations with pandas/polars
* Implement caching for expensive computations
* Use appropriate data types (categories, int32 vs int64)
* Leverage multiprocessing for CPU-bound tasks

.. code-block:: python

   import pandas as pd
   from src.performance.cache import cache_data
   
   @cache_data(ttl=3600)  # Cache for 1 hour
   def expensive_health_analysis(data: pd.DataFrame) -> pd.DataFrame:
       """Perform expensive health data analysis with caching."""
       # Use vectorised operations
       data['standardised_rate'] = (
           data['raw_rate'] * data['standard_population'] / data['actual_population']
       )
       
       # Use categorical data for memory efficiency
       data['region'] = data['region'].astype('category')
       data['age_group'] = data['age_group'].astype('category')
       
       return data

**Memory Management**

.. code-block:: python

   import gc
   import pandas as pd
   
   def process_large_dataset_efficiently(file_path: str) -> pd.DataFrame:
       """Process large datasets with memory management."""
       # Use chunking for large files
       chunk_size = 10000
       processed_chunks = []
       
       for chunk in pd.read_csv(file_path, chunksize=chunk_size):
           # Process chunk
           processed_chunk = process_health_data(chunk)
           processed_chunks.append(processed_chunk)
           
           # Explicit garbage collection
           del chunk
           gc.collect()
       
       # Combine results
       result = pd.concat(processed_chunks, ignore_index=True)
       
       # Clean up
       del processed_chunks
       gc.collect()
       
       return result

Database Optimisation
~~~~~~~~~~~~~~~~~~~~~

**Query Optimisation**

.. code-block:: python

   import duckdb
   from src.config import get_config
   
   def optimised_health_query(
       region_filter: List[str],
       date_range: Tuple[str, str]
   ) -> pd.DataFrame:
       """Execute optimised health data query."""
       config = get_config()
       
       # Use parameterised queries
       query = """
           SELECT 
               region,
               date,
               health_metric,
               population
           FROM health_data 
           WHERE 
               region IN ({region_placeholders})
               AND date BETWEEN ? AND ?
           ORDER BY region, date
       """.format(
           region_placeholders=','.join(['?'] * len(region_filter))
       )
       
       params = region_filter + list(date_range)
       
       with duckdb.connect(str(config.database.path)) as conn:
           result = conn.execute(query, params).fetchdf()
       
       return result

Monitoring and Profiling
~~~~~~~~~~~~~~~~~~~~~~~~

**Performance Monitoring**

.. code-block:: python

   from src.performance.monitoring import performance_timer
   import logging
   
   logger = logging.getLogger(__name__)
   
   @performance_timer
   def monitored_data_processing(data: pd.DataFrame) -> pd.DataFrame:
       """Data processing with performance monitoring."""
       logger.info(f"Processing {len(data)} records")
       
       # Processing logic here
       result = process_health_data(data)
       
       logger.info(f"Processed to {len(result)} records")
       return result

**Memory Profiling**

.. code-block:: python

   from memory_profiler import profile
   
   @profile
   def memory_intensive_function(data: pd.DataFrame) -> pd.DataFrame:
       """Function with memory profiling."""
       # Memory-intensive operations
       large_result = data.groupby(['region', 'age_group']).agg({
           'health_metric': ['mean', 'std', 'count'],
           'population': 'sum'
       })
       
       return large_result

Deployment
----------

Local Development
~~~~~~~~~~~~~~~~~

For local development and testing:

.. code-block:: bash

   # Start development server
   streamlit run src/dashboard/app.py --server.port 8501

   # With debugging
   streamlit run src/dashboard/app.py --server.port 8501 --server.runOnSave true

Production Deployment
~~~~~~~~~~~~~~~~~~~~~

See :doc:`../deployment/index` for detailed production deployment instructions.

**Docker Deployment**

.. code-block:: dockerfile

   FROM python:3.11-slim

   WORKDIR /app

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       build-essential \
       && rm -rf /var/lib/apt/lists/*

   # Install Python dependencies
   COPY pyproject.toml ./
   RUN pip install -e .

   # Copy application code
   COPY src/ ./src/
   COPY data/ ./data/

   # Expose port
   EXPOSE 8501

   # Run application
   CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]

Contributing Guidelines
-----------------------

Code Review Process
~~~~~~~~~~~~~~~~~~~

All code changes must go through review:

1. **Self Review**: Review your own code before submitting
2. **Automated Checks**: Ensure all CI checks pass
3. **Peer Review**: At least one team member must approve
4. **Documentation Review**: Ensure documentation is updated
5. **Testing Review**: Verify adequate test coverage

Review Checklist
~~~~~~~~~~~~~~~~

Reviewers should check:

* **Functionality**: Does the code work as intended?
* **Tests**: Are there adequate tests with good coverage?
* **Documentation**: Is the code well-documented?
* **Performance**: Are there any performance concerns?
* **Security**: Are there any security vulnerabilities?
* **Style**: Does the code follow project standards?
* **Maintainability**: Is the code easy to understand and maintain?

Release Process
~~~~~~~~~~~~~~~

1. **Feature Freeze**: No new features after feature freeze
2. **Release Branch**: Create release branch from develop
3. **Testing**: Comprehensive testing of release candidate
4. **Documentation**: Update version numbers and documentation
5. **Release**: Merge to main and tag release
6. **Deployment**: Deploy to production environment
7. **Post-Release**: Merge main back to develop

Troubleshooting Development Issues
----------------------------------

Common Development Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Import Errors**

.. code-block:: bash

   # Ensure project is installed in development mode
   uv pip install -e .
   
   # Check Python path
   python -c "import sys; print('\n'.join(sys.path))"

**Test Failures**

.. code-block:: bash

   # Run specific test with verbose output
   python -m pytest tests/test_specific.py::test_function -v
   
   # Run with debugging
   python -m pytest tests/test_specific.py::test_function -s --pdb

**Performance Issues**

.. code-block:: bash

   # Profile code execution
   python -m cProfile -o profile_output.prof your_script.py
   
   # Analyse profile
   python -c "import pstats; pstats.Stats('profile_output.prof').sort_stats('cumulative').print_stats(10)"

**Memory Issues**

.. code-block:: bash

   # Monitor memory usage
   python -m memory_profiler your_script.py
   
   # Use memory profiling decorator
   from memory_profiler import profile

Getting Help
~~~~~~~~~~~~

* **Documentation**: Check this developer guide and API reference
* **Code Examples**: Look at existing code for patterns
* **Team Chat**: Use team communication channels
* **Issue Tracker**: Search existing issues or create new ones
* **Code Review**: Ask specific questions during code review