# Australian Health Analytics Dashboard 🏥

**Interactive Streamlit dashboard showcasing correlation analysis and risk scoring between socio-economic disadvantage and health outcomes across Australia**

![Dashboard Preview](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)

## 🎯 Portfolio Showcase Objectives

This dashboard demonstrates advanced data science and visualisation capabilities specifically for **health policy analysis** and **geographic data science**. Perfect for showcasing skills to:

- **Health departments and policy organisations**
- **Public health research institutions** 
- **Healthcare consulting firms**
- **Government analytics roles**
- **Academic research positions**

## ✨ Key Features & Capabilities

### 🗺️ Geographic Health Explorer
- **Interactive choropleth mapping** of 2,454 Australian Statistical Areas
- **Multi-indicator visualisation** (mortality, chronic disease, healthcare access)
- **Dynamic state/territory filtering** for targeted analysis
- **Detailed area profiles** with socio-economic context

### 📊 Statistical Correlation Analysis  
- **Interactive correlation matrices** between SEIFA disadvantage and health outcomes
- **Scatter plot analysis** with regression lines and confidence intervals
- **Statistical significance testing** and effect size interpretation
- **Automated insight generation** highlighting strongest relationships

### 🎯 Health Hotspot Identification
- **Algorithm-driven identification** of priority intervention areas
- **Composite risk scoring** combining health outcomes and disadvantage
- **Comparative analysis** against national and state benchmarks
- **Resource allocation guidance** with priority classifications

### 🔮 Predictive Risk Modelling
- **Interactive prediction tool** for health risk assessment
- **Scenario analysis** modelling socio-economic improvements
- **Population impact calculations** for policy planning
- **Evidence-based intervention recommendations**

### 📋 Methodology Transparency
- **Complete data provenance** documentation
- **Statistical methodology** explanation and validation
- **Quality assessment metrics** and limitations discussion
- **Reproducible analysis pipeline** documentation

## 🛠️ Technical Implementation

### Modern Data Science Stack
```python
Frontend:       Streamlit (interactive web apps)
Mapping:        Folium + Streamlit-Folium (geographic visualisation)
Analytics:      Pandas, NumPy, SciPy (data processing & statistics)  
Visualisation:  Plotly, Altair (interactive charts)
Geospatial:     GeoPandas (spatial data operations)
Database:       DuckDB (embedded analytics)
```

### Key Technical Features
- **Cached data processing** for optimal performance
- **Responsive design** for desktop and mobile
- **Modular architecture** for easy extension
- **Error handling** and graceful degradation
- **Export-ready** visualisations and data

## 🚀 Quick Start

### 1. Installation
```bash
# Clone repository
git clone <repository-url>
cd AHGD

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
```bash
# Ensure processed data files exist
python setup_and_run.py  # If needed for data processing
```

### 3. Launch Dashboard
```bash
# Using the launcher script (recommended)
python run_dashboard.py

# Or directly with Streamlit
streamlit run scripts/streamlit_dashboard.py
```

### 4. Access Dashboard
Open your browser to: **http://localhost:8501**

## 📊 Demo & Portfolio Presentation

### Command-Line Demonstration
```bash
# Run comprehensive feature demonstration
python scripts/demo_dashboard_features.py
```

This generates a detailed analysis report showing:
- Geographic coverage statistics
- Correlation analysis results  
- Health hotspot identification
- Predictive model performance
- Data quality assessment

### Portfolio Screenshots
*[Dashboard screenshots would be included here in a real deployment]*

## 🎓 Skills Demonstrated

### Data Science & Analytics
- **Statistical Analysis**: Correlation analysis, regression modelling, significance testing
- **Predictive Modelling**: Risk assessment algorithms and scenario analysis
- **Data Quality Management**: Validation, completeness assessment, outlier detection

### Geographic Information Systems (GIS)
- **Spatial Data Processing**: GeoPandas operations on Australian Bureau of Statistics boundaries
- **Choropleth Mapping**: Interactive geographic visualisation with Folium
- **Spatial Statistics**: Geographic correlation and hotspot analysis

### Software Engineering
- **Web Application Development**: Production-ready Streamlit dashboard
- **Modular Design**: Clean, maintainable code architecture
- **Error Handling**: Robust exception management and user feedback
- **Performance Optimisation**: Data caching and efficient processing

### Domain Expertise
- **Health Policy Understanding**: SEIFA disadvantage indexes and health outcome relationships
- **Public Health Analytics**: Population health indicators and intervention prioritisation
- **Australian Data Sources**: ABS, AIHW, PHIDU dataset integration

## 📈 Real-World Applications

### Health Department Use Cases
- **Resource Allocation**: Identify areas needing additional health services
- **Policy Development**: Evidence-based targeting of health inequality interventions
- **Performance Monitoring**: Track health outcomes across geographic areas
- **Budget Planning**: Quantify population impact of proposed interventions

### Research Applications
- **Academic Studies**: Geographic health outcome analysis
- **Grant Applications**: Demonstrate analytical capabilities and methodology
- **Policy Evaluation**: Assess effectiveness of existing health programs
- **Collaborative Research**: Provide analytical infrastructure for multi-institution studies

## 🔍 Data Sources & Methodology

### Primary Data Sources
- **Australian Bureau of Statistics (ABS)**
  - SEIFA 2021: Socio-Economic Indexes for Areas
  - SA2 Digital Boundaries 2021
  
- **Australian Institute of Health and Welfare (AIHW)**
  - National mortality and morbidity databases
  
- **Public Health Information Development Unit (PHIDU)**
  - Social health atlas data

### Statistical Methodology
- **Correlation Analysis**: Pearson correlation coefficients with significance testing
- **Composite Scoring**: Weighted health risk indicators
- **Hotspot Detection**: Quantile-based identification algorithms
- **Predictive Modelling**: Linear regression with scenario analysis

### Quality Assurance
- **Data Validation**: Automated completeness and consistency checks
- **Outlier Detection**: Statistical identification of anomalous values
- **Methodology Documentation**: Complete analytical transparency
- **Reproducible Pipeline**: Version-controlled processing scripts

## 📝 Documentation

### User Guides
- **[Dashboard User Guide](dashboard_user_guide.md)**: Comprehensive usage instructions
- **[Technical Documentation](../README.md)**: System architecture and setup
- **[Data Processing Report](DATA_PROCESSING_REPORT.md)**: Pipeline documentation

### Portfolio Materials
- **Feature demonstrations**: Command-line showcase scripts
- **Methodology explanations**: Statistical approach documentation  
- **Use case examples**: Real-world application scenarios
- **Performance metrics**: System capability benchmarks

## 🌟 Portfolio Value Proposition

### For Employers
- **Immediate Value**: Production-ready analytical capabilities
- **Technical Skills**: Modern data science stack proficiency
- **Domain Knowledge**: Understanding of health policy and geographic analysis
- **Delivery Focus**: User-centric dashboard design and functionality

### For Collaborators
- **Analytical Infrastructure**: Ready-to-use health analytics platform
- **Extensible Design**: Easy integration of additional data sources
- **Methodology Transparency**: Reproducible and auditable analysis
- **Knowledge Transfer**: Comprehensive documentation and training materials

## 🔧 Development & Extension

### Adding New Data Sources
The dashboard architecture supports easy integration of additional health indicators:

```python
# Example: Adding new health indicator
def load_additional_health_data():
    # Process new data source
    new_data = process_health_indicator(source_file)
    
    # Integrate with existing pipeline
    merged_data = integrate_with_sa2_boundaries(new_data)
    
    # Update dashboard displays
    add_indicator_to_dashboard(merged_data, 'new_indicator')
```

### Customisation Options
- **Geographic Scope**: Adapt for different countries or regions
- **Health Indicators**: Modify composite scoring weights and components
- **Analysis Methods**: Implement different statistical approaches
- **Visualisation Themes**: Customise colours, layouts, and branding

## 📞 Portfolio Contact

**Demonstration Project**: Australian Health Analytics Dashboard
**Technical Showcase**: Modern data science capabilities for health policy analysis
**Skills Highlighted**: Statistical analysis, geographic data science, web application development

---

*This dashboard represents a portfolio demonstration of advanced data science capabilities applied to Australian health data. In production deployment, synthetic health indicators would be replaced with actual health surveillance data from confidential databases.*

**Last Updated**: June 2025 | **Version**: 1.0 | **Purpose**: Portfolio Showcase