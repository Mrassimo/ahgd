Basic Usage Tutorial
====================

This tutorial will walk you through the basic features of the Australian Health
Analytics Dashboard, from initial setup to creating your first analysis.

Prerequisites
-------------

Before starting, ensure you have:

* AHGD installed and running
* Sample data downloaded
* Web browser open to the dashboard

Tutorial Steps
--------------

Step 1: Accessing the Dashboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Start the Dashboard**

   If not already running, start the dashboard:

   .. code-block:: bash

      python run_dashboard.py

2. **Open in Browser**

   Navigate to `http://localhost:8501` in your web browser.

3. **Verify Data Loading**

   You should see the main dashboard page with health data loaded.
   If you see error messages about missing data, run:

   .. code-block:: bash

      python scripts/download_data.py
      python scripts/process_data.py

Step 2: Understanding the Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dashboard interface has several key areas:

**Header Section**
  * Application title and logo
  * Navigation menu
  * User controls and settings

**Sidebar (Left Panel)**
  * Data filters and controls
  * Region selection
  * Time period settings
  * Export options

**Main Content Area**
  * Charts and visualisations
  * Data tables
  * Analysis results

**Status Bar (Bottom)**
  * Data freshness indicators
  * Performance metrics
  * Help and documentation links

Step 3: Basic Navigation
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Page Navigation**

   Use the sidebar or header menu to navigate between pages:
   
   * **Overview**: Summary statistics and key indicators
   * **Geographic**: Maps and regional analysis
   * **Health Analytics**: Detailed health outcome analysis
   * **Demographics**: Population and demographic insights

2. **Using Filters**

   In the sidebar, you'll find various filter options:
   
   * **State/Territory**: Select specific regions
   * **Time Period**: Choose date ranges
   * **Age Groups**: Filter by demographic categories
   * **Health Metrics**: Select specific health indicators

Step 4: Your First Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's create a basic analysis of health outcomes by state:

1. **Navigate to Geographic Page**

   Click "Geographic Analysis" in the sidebar.

2. **Select Health Metric**

   In the sidebar filters:
   
   * Set "Health Metric" to "Life Expectancy"
   * Ensure "All States" is selected
   * Set time period to "Latest Available"

3. **View the Map**

   You should see a choropleth map showing life expectancy across Australia.
   The map uses colour coding to show variations between states and territories.

4. **Interpret the Results**

   * Darker colours typically indicate higher values
   * Hover over regions to see exact values
   * Use the legend to understand the colour scale

Step 5: Exploring Health Trends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now let's look at trends over time:

1. **Navigate to Health Analytics**

   Click "Health Analytics" in the sidebar.

2. **Configure Time Series Analysis**

   Set the following filters:
   
   * **Health Metric**: "Mortality Rate"
   * **Region**: "New South Wales"
   * **Time Period**: "2015-2023"
   * **Age Group**: "All Ages"

3. **View Trend Chart**

   You should see a line chart showing mortality trends over time.

4. **Analyse the Trend**

   * Look for patterns (increasing, decreasing, stable)
   * Note any significant changes or anomalies
   * Consider external factors that might explain patterns

Step 6: Comparing Regions
~~~~~~~~~~~~~~~~~~~~~~~~~

Let's compare health outcomes between different states:

1. **Multi-Region Selection**

   In the sidebar:
   
   * Hold Ctrl (or Cmd on Mac) and select multiple states:
     * New South Wales
     * Victoria
     * Queensland

2. **Create Comparison Chart**

   * Set "Chart Type" to "Bar Chart"
   * Set "Health Metric" to "Health Score"
   * Set "Time Period" to "Latest Available"

3. **Interpret Comparison**

   * Compare values across selected regions
   * Look for significant differences
   * Consider factors that might explain variations

Step 7: Working with Demographics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Explore how health outcomes vary by demographic groups:

1. **Navigate to Demographics Page**

   Click "Demographic Analysis" in the sidebar.

2. **Age Group Analysis**

   Configure filters:
   
   * **Region**: "Australia"
   * **Health Metric**: "Health Risk Score"
   * **Breakdown**: "Age Group"

3. **View Results**

   You should see health metrics broken down by age groups.

4. **Identify Patterns**

   * Note which age groups have higher/lower health risks
   * Consider implications for health policy
   * Look for unexpected patterns

Step 8: Exporting Data and Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Learn how to export your analysis:

1. **Export Chart**

   * Click the export button on any chart
   * Choose format (PNG, SVG, PDF)
   * Save to your computer

2. **Export Data**

   * Use the "Export Data" button in the sidebar
   * Choose format (CSV, Excel, JSON)
   * Download the filtered dataset

3. **Create Report**

   * Click "Generate Report" in the sidebar
   * Configure report settings
   * Download PDF report with charts and analysis

Step 9: Understanding Data Quality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Learn to assess data quality:

1. **Check Data Freshness**

   Look at the status bar for data update information:
   
   * Green indicators: Recent, high-quality data
   * Yellow indicators: Older data, use with caution
   * Red indicators: Outdated or low-quality data

2. **Review Data Coverage**

   Check which regions and time periods have complete data:
   
   * Some regions may have limited data
   * Recent time periods may have provisional data
   * Historical data may use different methodologies

3. **Validate Results**

   * Compare results with known benchmarks
   * Check for outliers or unexpected values
   * Cross-reference with official statistics

Practical Exercise
------------------

Complete this exercise to practice what you've learned:

**Exercise: Analyse Health Inequality Across Australia**

1. **Objective**
   
   Investigate how health outcomes vary by socioeconomic status across Australian regions.

2. **Steps**

   a. Navigate to the Geographic Analysis page
   
   b. Set filters:
      * Health Metric: "Health Score"
      * Socioeconomic Filter: "By SEIFA Decile"
      * Time Period: "Latest Available"
   
   c. Create visualisations:
      * Map showing health scores by region
      * Chart comparing health scores across SEIFA deciles
      * Trend analysis showing changes over time
   
   d. Export your results:
      * Save maps and charts as images
      * Export the underlying data
      * Generate a summary report

3. **Analysis Questions**

   Answer these questions based on your analysis:
   
   * Which regions have the highest/lowest health scores?
   * How does socioeconomic status relate to health outcomes?
   * Are there geographic patterns in health inequality?
   * What trends do you observe over time?

4. **Expected Results**

   You should find:
   
   * Clear correlation between socioeconomic status and health
   * Geographic clustering of similar health outcomes
   * Variations between urban and rural areas
   * Gradual improvements in some metrics over time

Tips for Success
----------------

**Navigation Tips**
~~~~~~~~~~~~~~~~~~~

* Use the browser back button to return to previous views
* Bookmark frequently used filter combinations
* Use keyboard shortcuts where available
* Keep multiple browser tabs open for comparison

**Analysis Tips**
~~~~~~~~~~~~~~~~~

* Start with broad overviews before drilling down
* Always check data quality and currency
* Compare multiple metrics for comprehensive analysis
* Look for patterns across different time periods

**Data Interpretation**
~~~~~~~~~~~~~~~~~~~~~~~

* Consider population size when comparing regions
* Account for demographic differences between areas
* Be cautious about causal interpretations
* Validate surprising results with additional analysis

**Performance Tips**
~~~~~~~~~~~~~~~~~~~~

* Limit date ranges for large datasets
* Use sampling options for initial exploration
* Clear browser cache if performance is slow
* Close unused browser tabs

Common Issues and Solutions
---------------------------

**Dashboard Not Loading**

* Check that the server is running
* Verify your browser supports JavaScript
* Try refreshing the page
* Clear browser cache and cookies

**Data Not Displaying**

* Check filter settings - they may be too restrictive
* Verify data is available for selected time period
* Look for error messages in the interface
* Try resetting filters to defaults

**Charts Not Rendering**

* Ensure browser JavaScript is enabled
* Try different chart types
* Reduce data complexity by filtering
* Check browser console for error messages

**Export Not Working**

* Check browser pop-up blocker settings
* Ensure sufficient disk space
* Try different export formats
* Verify file permissions

Next Steps
----------

After completing this tutorial, you're ready to:

* Explore the :doc:`data_analysis` tutorial for advanced techniques
* Learn about :doc:`custom_visualizations` for specialized charts
* Check the :doc:`../guides/user_guide` for comprehensive feature documentation
* Practice with your own research questions and hypotheses

Additional Resources
-------------------

* :doc:`../guides/user_guide` - Comprehensive user documentation
* :doc:`../reference/index` - Configuration and troubleshooting
* :doc:`../api/index` - Technical API reference
* Project repository - For sample data and examples