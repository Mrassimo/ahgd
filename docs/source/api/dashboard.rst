Dashboard Package
================

.. automodule:: src.dashboard
   :members:
   :undoc-members:
   :show-inheritance:

Main Dashboard Application
--------------------------

.. automodule:: src.dashboard.app
   :members:
   :undoc-members:
   :show-inheritance:

Data Subpackage
---------------

The data subpackage handles all data loading and processing operations.

.. toctree::
   :maxdepth: 2

Data Loaders
~~~~~~~~~~~~

.. automodule:: src.dashboard.data.loaders
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
^^^^^^^^^^^^^

.. autofunction:: src.dashboard.data.loaders.load_health_data

.. autofunction:: src.dashboard.data.loaders.load_geographic_data

.. autofunction:: src.dashboard.data.loaders.load_demographic_data

.. autofunction:: src.dashboard.data.loaders.load_all_data

Usage Examples
^^^^^^^^^^^^^^

.. code-block:: python

   from src.dashboard.data.loaders import (
       load_health_data,
       load_geographic_data,
       load_all_data
   )
   from src.config import get_config
   
   config = get_config()
   
   # Load specific data types
   health_data = load_health_data(config.data_sources.health_data_path)
   geo_data = load_geographic_data(config.data_sources.geographic_data_path)
   
   # Load all data at once
   all_data = load_all_data(config)

Data Processors
~~~~~~~~~~~~~~~

.. automodule:: src.dashboard.data.processors
   :members:
   :undoc-members:
   :show-inheritance:Key Functions
^^^^^^^^^^^^^

.. autofunction:: src.dashboard.data.processors.process_health_data

.. autofunction:: src.dashboard.data.processors.process_geographic_data

.. autofunction:: src.dashboard.data.processors.merge_health_geographic

.. autofunction:: src.dashboard.data.processors.calculate_health_indicators

Usage Examples
^^^^^^^^^^^^^^

.. code-block:: python

   from src.dashboard.data.processors import (
       process_health_data,
       process_geographic_data,
       merge_health_geographic
   )
   
   # Process raw health data
   processed_health = process_health_data(raw_health_data)
   
   # Process geographic data
   processed_geo = process_geographic_data(raw_geo_data)
   
   # Merge datasets
   merged_data = merge_health_geographic(processed_health, processed_geo)

UI Subpackage
-------------

The UI subpackage contains all user interface components for the dashboard.

Layout Components
~~~~~~~~~~~~~~~~~

.. automodule:: src.dashboard.ui.layout
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
^^^^^^^^^^^^^

.. autofunction:: src.dashboard.ui.layout.create_main_layout

.. autofunction:: src.dashboard.ui.layout.create_header

.. autofunction:: src.dashboard.ui.layout.create_footer

.. autofunction:: src.dashboard.ui.layout.create_navigation

Page Components
~~~~~~~~~~~~~~~

.. automodule:: src.dashboard.ui.pages
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
^^^^^^^^^^^^^

.. autofunction:: src.dashboard.ui.pages.render_overview_page

.. autofunction:: src.dashboard.ui.pages.render_analysis_page

.. autofunction:: src.dashboard.ui.pages.render_geographic_page

.. autofunction:: src.dashboard.ui.pages.render_performance_page

Sidebar Components
~~~~~~~~~~~~~~~~~~

.. automodule:: src.dashboard.ui.sidebar
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
^^^^^^^^^^^^^

.. autofunction:: src.dashboard.ui.sidebar.create_sidebar

.. autofunction:: src.dashboard.ui.sidebar.create_filters

.. autofunction:: src.dashboard.ui.sidebar.create_controls

Visualisation Subpackage
-------------------------

The visualisation subpackage provides all charting and mapping capabilities.

Charts
~~~~~~

.. automodule:: src.dashboard.visualisation.charts
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
^^^^^^^^^^^^^

.. autofunction:: src.dashboard.visualisation.charts.create_bar_chart

.. autofunction:: src.dashboard.visualisation.charts.create_line_chart

.. autofunction:: src.dashboard.visualisation.charts.create_scatter_plot

.. autofunction:: src.dashboard.visualisation.charts.create_histogram

Usage Examples
^^^^^^^^^^^^^^

.. code-block:: python

   from src.dashboard.visualisation.charts import (
       create_bar_chart,
       create_line_chart,
       create_scatter_plot
   )
   import pandas as pd
   
   # Sample data
   data = pd.DataFrame({
       'region': ['NSW', 'VIC', 'QLD', 'SA', 'WA'],
       'population': [8000000, 6500000, 5100000, 1750000, 2650000],
       'health_score': [75, 78, 72, 80, 74]
   })
   
   # Create bar chart
   bar_chart = create_bar_chart(
       data, 
       x='region', 
       y='population',
       title='Population by State'
   )
   
   # Create scatter plot
   scatter_plot = create_scatter_plot(
       data,
       x='population',
       y='health_score',
       title='Health Score vs Population'
   )

Components
~~~~~~~~~~

.. automodule:: src.dashboard.visualisation.components
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
^^^^^^^^^^^^^

.. autofunction:: src.dashboard.visualisation.components.create_metric_card

.. autofunction:: src.dashboard.visualisation.components.create_data_table

.. autofunction:: src.dashboard.visualisation.components.create_filter_panel

.. autofunction:: src.dashboard.visualisation.components.create_export_panel

Maps
~~~~

.. automodule:: src.dashboard.visualisation.maps
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
^^^^^^^^^^^^^

.. autofunction:: src.dashboard.visualisation.maps.create_choropleth_map

.. autofunction:: src.dashboard.visualisation.maps.create_point_map

.. autofunction:: src.dashboard.visualisation.maps.create_heatmap

.. autofunction:: src.dashboard.visualisation.maps.add_map_controls

Usage Examples
^^^^^^^^^^^^^^

.. code-block:: python

   from src.dashboard.visualisation.maps import (
       create_choropleth_map,
       create_point_map
   )
   import geopandas as gpd
   
   # Load geographic data
   boundaries = gpd.read_file('data/boundaries.geojson')
   
   # Create choropleth map
   choropleth = create_choropleth_map(
       boundaries,
       value_column='health_score',
       title='Health Scores by SA2 Region'
   )
   
   # Create point map for specific locations
   point_map = create_point_map(
       locations_df,
       lat_column='latitude',
       lon_column='longitude',
       size_column='population'
   )

Dashboard Integration
---------------------

Complete Dashboard Setup
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.dashboard.app import create_dashboard
   from src.dashboard.data.loaders import load_all_data
   from src.config import get_config
   
   # Load configuration
   config = get_config()
   
   # Load all required data
   data = load_all_data(config)
   
   # Create dashboard with all components
   dashboard = create_dashboard(data, config)

Custom Page Creation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.dashboard.ui.layout import create_main_layout
   from src.dashboard.ui.sidebar import create_sidebar
   from src.dashboard.visualisation.charts import create_bar_chart
   import streamlit as st
   
   def create_custom_page(data):
       # Create layout
       layout = create_main_layout("Custom Analysis")
       
       # Create sidebar with filters
       sidebar = create_sidebar(data)
       
       # Create visualisations
       chart = create_bar_chart(
           data,
           x='region',
           y='metric',
           title='Custom Metric Analysis'
       )
       
       # Display in Streamlit
       st.plotly_chart(chart, use_container_width=True)

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~

The dashboard includes several performance optimisations:

.. code-block:: python

   from src.dashboard.data.loaders import load_all_data
   from src.performance.cache import cache_data
   
   # Cache frequently accessed data
   @cache_data(ttl=3600)  # Cache for 1 hour
   def load_cached_data():
       return load_all_data()
   
   # Use cached data
   data = load_cached_data()

Error Handling
~~~~~~~~~~~~~~

.. code-block:: python

   from src.dashboard.data.loaders import load_health_data
   from src.dashboard.exceptions import DataLoadError
   import streamlit as st
   
   try:
       data = load_health_data(data_path)
   except DataLoadError as e:
       st.error(f"Failed to load health data: {e}")
       st.stop()
   except Exception as e:
       st.error(f"Unexpected error: {e}")
       st.stop()