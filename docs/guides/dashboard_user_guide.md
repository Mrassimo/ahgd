# Australian Health Analytics Dashboard - User Guide

## Overview

The Australian Health Analytics Dashboard is an interactive web application built with Streamlit that demonstrates correlation analysis and risk scoring between socio-economic disadvantage and health outcomes across Australia. This portfolio showcase project highlights modern data science capabilities for health policy analysis.

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Required packages (see requirements.txt)

### Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Data Files Are Available**:
   The dashboard requires processed data files in the `data/processed/` directory:
   - `seifa_2021_sa2.parquet` - SEIFA disadvantage data
   - `sa2_boundaries_2021.parquet` - Geographic boundaries

3. **Launch Dashboard**:
   ```bash
   streamlit run scripts/streamlit_dashboard.py
   ```

4. **Access Dashboard**:
   Open your web browser to `http://localhost:8501`

## Dashboard Features

### 1. Geographic Health Explorer 🗺️

**Purpose**: Interactive mapping of health indicators across Australian Statistical Areas

**Key Features**:
- **Interactive Choropleth Maps**: Colour-coded regions showing health risk levels
- **Multiple Health Indicators**: Switch between different health metrics
- **State/Territory Filtering**: Focus analysis on specific jurisdictions
- **Detailed Tooltips**: Click areas for comprehensive health profiles
- **Statistical Summary**: Key metrics and distribution statistics

**How to Use**:
1. Select your preferred health indicator from the dropdown menu
2. Use the sidebar to filter by specific states/territories
3. Hover over map regions to see basic information
4. Click on regions for detailed health and socio-economic data
5. Review the data table below the map for comprehensive area information

**Available Health Indicators**:
- Composite Health Risk Score
- Mortality Rate (per 1,000 population)
- Diabetes Prevalence (%)
- Heart Disease Rate (per 1,000)
- Mental Health Issues Rate (per 1,000)
- GP Access Score (0-10 scale)
- Distance to Hospital (kilometres)

### 2. Correlation Analysis 📊

**Purpose**: Statistical analysis of relationships between socio-economic disadvantage and health outcomes

**Key Features**:
- **Interactive Correlation Matrix**: Heatmap showing all variable relationships
- **Statistical Significance**: Colour-coded correlation strengths
- **Scatter Plot Analysis**: Visualise specific variable relationships
- **Trend Line Analysis**: Regression lines with confidence intervals
- **Key Insights Summary**: Automated identification of strongest correlations

**How to Use**:
1. Review the correlation heatmap to identify relationship patterns
2. Focus on SEIFA disadvantage correlations (key policy variable)
3. Examine scatter plots for detailed relationship visualisation
4. Note statistical significance levels in the insights panel
5. Use state filtering to analyse regional patterns

**Interpretation Guide**:
- **Strong Positive Correlation (0.5 to 1.0)**: Variables increase together
- **Strong Negative Correlation (-0.5 to -1.0)**: One increases as other decreases
- **Moderate Correlation (0.3 to 0.5)**: Clear but not dominant relationship
- **Weak Correlation (0.0 to 0.3)**: Limited relationship

### 3. Health Hotspot Identification 🎯

**Purpose**: Identify priority areas for health intervention combining poor health outcomes with high socio-economic disadvantage

**Key Features**:
- **Automated Hotspot Detection**: Areas with high health risk AND high disadvantage
- **Priority Ranking**: Sorted by composite risk factors
- **Intervention Recommendations**: Risk-based priority classifications
- **Comparative Analysis**: Hotspot metrics vs national averages
- **Detailed Area Profiles**: Comprehensive statistics for each priority area

**How to Use**:
1. Review the identified priority areas in the summary metrics
2. Examine individual hotspot details by expanding area cards
3. Compare hotspot averages against national benchmarks
4. Use the priority classifications to guide resource allocation
5. Export data for further policy analysis

**Priority Classifications**:
- 🔴 **Immediate Intervention Required** (Score ≥7/10): Critical priority areas
- 🟡 **Medium Priority** (Score 5-7/10): Important but less urgent
- 🟢 **Lower Priority** (Score <5/10): Monitor and maintain current services

### 4. Predictive Risk Analysis 🔮

**Purpose**: Model health risk outcomes based on socio-economic characteristics and explore policy scenarios

**Key Features**:
- **Health Risk Prediction Tool**: Input characteristics to predict health outcomes
- **Scenario Analysis**: Model impact of socio-economic improvements
- **Population Impact Calculations**: Estimate benefiting populations
- **Interactive Parameter Controls**: Adjust multiple risk factors
- **Policy Impact Modelling**: Quantify potential intervention effects

**How to Use**:
1. **Risk Prediction**:
   - Adjust SEIFA disadvantage score slider
   - Modify population density, age profile, and healthcare access factors
   - Click "Calculate Predicted Health Risk" to see results
   - Compare predicted risk against national averages

2. **Scenario Analysis**:
   - Set improvement percentage for socio-economic conditions
   - Review projected health risk reductions
   - Assess population impact estimates
   - Use results for policy case development

**Input Parameters**:
- **SEIFA Disadvantage Score** (500-1200): Higher = less disadvantaged
- **Population Density Factor** (0.5-2.0): Urban vs rural characteristics
- **Age Profile Factor** (0.8-1.5): Population age distribution impact
- **Healthcare Access Factor** (0.5-1.5): Service accessibility measure

### 5. Data Quality & Methodology 📋

**Purpose**: Transparency about data sources, processing methods, and analytical limitations

**Key Features**:
- **Data Source Documentation**: Complete listing of datasets and provenance
- **Methodology Explanation**: Risk score calculations and statistical methods
- **Quality Assessment Metrics**: Data completeness and coverage statistics
- **Performance Indicators**: Model validation and correlation measures
- **Limitations and Assumptions**: Critical analytical caveats
- **Technical Implementation Details**: Technology stack and processing pipeline

**Key Sections**:
- **Data Sources**: ABS, AIHW, PHIDU datasets with update dates
- **Risk Score Methodology**: Weighted component calculation explanation
- **Quality Metrics**: Coverage percentages and completeness measures
- **Model Performance**: Statistical validation measures
- **Technical Details**: Implementation architecture and tools

## Navigation Tips

### Sidebar Controls
- **Analysis Type Selector**: Switch between the five main dashboard sections
- **State/Territory Filter**: Focus analysis on specific jurisdictions
- **Additional Filters**: Context-specific controls appear based on selected analysis

### Performance Optimisation
- **Data Caching**: Dashboard automatically caches processed data for faster performance
- **State Filtering**: Reduce data volume by selecting specific states for analysis
- **Browser Compatibility**: Best performance in Chrome, Firefox, or Safari
- **Mobile Responsive**: Dashboard adapts to tablet and mobile screens

### Export Options
- **Data Tables**: All data tables can be sorted and are ready for copy/paste
- **Visualisations**: Right-click charts to save as images
- **Maps**: Use browser tools to capture screenshots of interactive maps

## Data Understanding

### Geographic Coverage
- **2,454 Statistical Area Level 2 (SA2) regions** across Australia
- Complete coverage of all states and territories
- Population-weighted analysis capability
- Urban and rural area inclusion

### Health Indicators
**Note**: For portfolio demonstration purposes, health indicators are modelled based on established correlations with socio-economic disadvantage. In production deployment, these would be replaced with actual health surveillance data from confidential databases.

### Socio-Economic Data
- **SEIFA 2021**: Official ABS disadvantage indexes
- **Contemporary Data**: Based on 2021 Census
- **Multiple Dimensions**: Income, education, employment, housing
- **National Comparability**: Standardised scores and rankings

## Troubleshooting

### Common Issues

**Dashboard Won't Load**:
- Ensure all required packages are installed: `pip install -r requirements.txt`
- Verify data files exist in `data/processed/` directory
- Check Python version (3.8+ required)

**Map Not Displaying**:
- Verify internet connection (maps require online tile services)
- Check browser JavaScript is enabled
- Try refreshing the page or restarting the dashboard

**Slow Performance**:
- Use state filtering to reduce data volume
- Close other browser tabs or applications
- Ensure sufficient system memory (4GB+ recommended)

**Data Issues**:
- Run `verify_data.py` to check data file integrity
- Re-run data processing pipeline if necessary
- Check file permissions in data directories

### Getting Help

**For Technical Issues**:
- Check error messages displayed in the dashboard
- Review browser console for JavaScript errors
- Verify system requirements and dependencies

**For Data Questions**:
- Consult the "Data Quality & Methodology" section
- Review source documentation in the `docs/` directory
- Check data processing reports for quality metrics

## Portfolio Context

### Demonstration Purpose
This dashboard is designed as a **portfolio showcase project** demonstrating:
- Modern data science and visualisation skills
- Understanding of health policy and public health analytics
- Ability to translate complex analysis into user-friendly interfaces
- Technical proficiency with current data science tools and methods

### Professional Applications
The techniques and approaches demonstrated could be applied to:
- **Health Department Analysis**: Resource allocation and service planning
- **Policy Development**: Evidence-based health inequality interventions
- **Academic Research**: Geographic health outcome studies
- **Consultancy Projects**: Health system performance analysis

### Technical Skills Demonstrated
- **Data Engineering**: ETL pipelines and data quality management
- **Statistical Analysis**: Correlation analysis and predictive modelling
- **Geographic Information Systems**: Spatial data processing and visualisation
- **Web Development**: Interactive dashboard creation and deployment
- **Data Visualisation**: Chart design and user experience optimisation

---

**Last Updated**: June 2025
**Dashboard Version**: 1.0
**Contact**: Portfolio demonstration project