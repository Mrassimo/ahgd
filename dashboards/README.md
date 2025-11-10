# AHGD Dashboard

Interactive dashboard for Australian Health Geography Data, providing visualizations and analysis of health indicators, demographics, and socioeconomic factors across Australian Statistical Areas.

![Dashboard](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=flat-square&logo=streamlit)
![Database](https://img.shields.io/badge/Database-DuckDB-FFF000?style=flat-square&logo=duckdb)
![Python](https://shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- DuckDB database (`ahgd.db`) populated by the AHGD ETL pipeline
- Required Python packages (see requirements.txt)

### Installation

```bash
# Navigate to the dashboards directory
cd dashboards

# Install dependencies
pip install -r requirements.txt
```

### Running the Dashboard

```bash
# From the dashboards directory
streamlit run app.py

# Or from the project root
cd ..
streamlit run dashboards/app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

### Docker Deployment

```bash
# Using Docker Compose (recommended)
cd dashboards
docker-compose up -d

# Or build and run manually
docker build -t ahgd-dashboard -f Dockerfile ..
docker run -d -p 8501:8501 -v /path/to/ahgd.db:/app/ahgd.db:ro ahgd-dashboard
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

---

## 📊 Dashboard Features

### 🏠 Main Dashboard
- Real-time database connection status
- Quick summary KPIs (SA2 regions, population, health metrics)
- Feature highlights and navigation guide
- Custom styling with metric cards
- About section and help text

### Page 1: 📊 Overview Dashboard
- **Key Metrics Tab:**
  - 5 KPI cards (SA2 regions, mortality, utilisation, population, composite index)
  - Health indicator distribution histogram
  - Scatter plot: Mortality vs Utilisation with SEIFA coloring
- **Top/Bottom Regions Tab:**
  - Top 10 regions by composite health index
  - Bottom 10 regions by composite health index
  - Remoteness category box plots for mortality
- **Distributions Tab:**
  - Interactive metric selection (6 health indicators)
  - Histogram and box plot visualizations
  - Summary statistics (mean, median, std dev, min, max)

### Page 2: 🗺️ Geographic Analysis
- **Interactive Map Tab:**
  - Scatter mapbox visualization of Australia
  - Color-coded by selected health indicator
  - Size-scaled by population
  - Multiple map styles (OpenStreetMap, Carto)
  - Hover tooltips with SA2 details
- **State Comparison Tab:**
  - Bar charts comparing health metrics by state
  - Pie charts showing population distribution
  - Detailed comparison table with gradients
- **Remoteness Analysis Tab:**
  - Trend lines across remoteness categories
  - Dual-axis charts for multiple metrics
  - Population and health distribution
- **Spatial Clusters Tab:**
  - Quintile-based clustering algorithm
  - Interactive cluster maps
  - Cluster statistics and distributions
- **Region Details Tab:**
  - Search by SA2 code or name
  - Detailed region cards with mini maps
  - Comprehensive health and demographic info

### Page 3: 🏥 Health Indicators Deep Dive
- **Mortality Rate Analysis:**
  - Distribution histograms and box plots
  - Mortality by remoteness and state
  - Mortality vs population analysis
- **Medicare Utilisation Analysis:**
  - Utilisation and bulk billing metrics
  - Access disparity measurements
  - Low utilisation region identification
- **Correlation Matrix:**
  - Full correlation heatmap
  - Individual correlation analysis
  - Scatter plots with trendlines
- **Risk Factor Identification:**
  - Configurable risk thresholds (75th-95th percentile)
  - Combined risk score using z-scores
  - High-risk region table
  - Risk distribution by state/remoteness
- **Statistical Analysis:**
  - Comparison tool with grouping
  - Violin plots for distributions
  - Interactive data explorer
  - CSV export functionality

### Page 4: 💰 Socioeconomic Impact
- **SEIFA Analysis:**
  - IRSAD/IRSD score distributions
  - SEIFA vs health outcome correlations
  - Disadvantaged region identification
- **Income & Health:**
  - Income vs mortality/bulk billing analysis
  - Unemployment impact visualization
  - Mortality by unemployment quartiles
- **Demographics:**
  - Population and age statistics
  - Age vs mortality correlations
  - Population density vs health outcomes
- **Correlation Analysis:**
  - Full socioeconomic correlation matrix
  - Multi-variable scatter matrices
  - Key correlation tables

### Page 5: 🌡️ Climate & Environment
- **Temperature & Health:**
  - Temperature vs mortality analysis
  - Temperature distribution by state
  - Hottest/coldest region analysis
- **Rainfall & Climate:**
  - Rainfall correlations with health
  - Climate zone classification (Arid, Temperate, Humid)
  - Health metrics by climate zone
- **Environmental Quality:**
  - Air Quality Index (AQI) analysis
  - Green space access metrics
  - Environmental quality by remoteness
- **Geographic Patterns:**
  - Climate gradient visualizations
  - Environmental profile radar charts
  - State-level comparisons
  - Environmental risk assessment

### Page 6: ✅ Data Quality & Pipeline Monitoring
- **Pipeline Status:**
  - Database connection health
  - Last update timestamp
  - Table statistics and record counts
- **Data Freshness:**
  - Freshness score gauge
  - Temporal coverage analysis
  - Threshold indicators (fresh/stale/outdated)
- **Data Coverage:**
  - Geographic coverage (SA2, states, remoteness)
  - Population coverage statistics
- **Missing Data Analysis:**
  - Column-by-column missing data
  - Visualizations and percentages
- **Quality Reports:**
  - Exportable quality reports
  - DBT test results display
  - Full dataset export

---

## 🛠️ Architecture

```
Streamlit Dashboard (Frontend)
       ↓ queries
 DuckDB Database (ahgd.db)
  ├── master_health_record
  └── derived_health_indicators
       ↓ populated by
 Airflow ETL Pipeline
  ├── Extract → Load → dbt Transform
  └── Scheduled daily/weekly updates
```

### Directory Structure

```
dashboards/
├── app.py                          # Main Streamlit application
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── DEPLOYMENT.md                   # Deployment guide
├── Dockerfile                      # Docker image configuration
├── docker-compose.yml              # Docker Compose setup
├── .dockerignore                   # Docker ignore patterns
├── pages/                          # Multi-page dashboard pages
│   ├── 1_📊_Overview.py            # Overview dashboard
│   ├── 2_🗺️_Geographic_Analysis.py # Geographic maps and analysis
│   ├── 3_🏥_Health_Indicators.py   # Health deep dive
│   ├── 4_💰_Socioeconomic_Impact.py # Socioeconomic analysis
│   ├── 5_🌡️_Climate_Environment.py # Climate correlations
│   └── 6_✅_Data_Quality.py        # Pipeline monitoring
├── components/                     # Reusable UI components
│   ├── charts.py                   # Chart templates
│   ├── filters.py                  # Filter widgets
│   └── export.py                   # Data export functionality
└── utils/                          # Utility modules
    └── database.py                 # DuckDB connection management
```

---

## 🎨 Features & Capabilities

### Interactive Visualizations
- **Plotly Charts:** Hover, zoom, pan on all charts
- **Geographic Maps:** Interactive choropleth and scatter maps
- **Correlation Matrices:** Heatmaps with color scales
- **Distribution Plots:** Histograms, box plots, violin plots
- **Trend Analysis:** Line charts with dual axes
- **Comparison Views:** Bar charts, pie charts, radar charts

### Data Analysis Tools
- **Filters:** State, remoteness, metrics, thresholds
- **Search:** Find regions by SA2 code or name
- **Clustering:** Spatial and statistical clustering
- **Risk Scoring:** Composite risk assessment
- **Statistical Tests:** Correlations, aggregations
- **Exports:** CSV, Excel, Parquet formats

### Performance Features
- **Query Caching:** 5-minute TTL for fast responses
- **Lazy Loading:** Data loaded per tab/page
- **Polars DataFrames:** 5-10x faster than Pandas
- **Read-only DB:** Safe concurrent access
- **Sample Limiting:** For large visualizations

### User Experience
- **Responsive Layout:** Works on desktop, tablet, mobile
- **Custom Styling:** Professional UI with metric cards
- **Help Text:** Contextual tooltips and descriptions
- **Error Handling:** Graceful degradation
- **Refresh Button:** Clear cache and reload data

---

## 📈 Data Connection

The dashboard connects to DuckDB database at:
```
../ahgd.db  (relative to dashboards directory)
```

### Required Tables
- `master_health_record` - Main health and demographic data
- `derived_health_indicators` - Calculated health metrics

### Data Schema
See [AHGD Data Dictionary](../docs/data_dictionary/) for complete schema.

Key fields:
- **Geographic:** sa2_code, sa2_name, state_code, remoteness_category, coordinates
- **Health:** mortality_rate, utilisation_rate, bulk_billed_percentage
- **Socioeconomic:** median_household_income, unemployment_rate, seifa_scores
- **Demographics:** total_population, median_age, population_density
- **Environment:** temperature, rainfall, air_quality, green_space

---

## 🔧 Customization

### Configuration

Edit `config.py` to customize:
- **Database Path:** Location of DuckDB file
- **Color Schemes:** Health and SEIFA color scales
- **Map Settings:** Center, zoom, styles
- **Cache Duration:** Query cache TTL (default 5 minutes)
- **Metrics Thresholds:** Good/bad thresholds for indicators
- **Export Limits:** Max rows for exports

### Adding New Pages

1. Create file: `pages/N_icon_PageName.py`
2. Follow existing page structure
3. Import utilities from `utils/` and `components/`
4. Streamlit auto-adds to sidebar navigation

Example:
```python
# pages/7_🔍_Custom_Analysis.py
import streamlit as st
from dashboards.utils.database import get_db_connection
from dashboards.config import DB_PATH

st.title("🔍 Custom Analysis")
db = get_db_connection(str(DB_PATH))
# Your custom analysis here
```

### Custom Components

Add reusable components to `components/`:

**Example - Custom Chart:**
```python
from dashboards.components.charts import create_bar_chart

fig = create_bar_chart(
    data=my_data,
    x="category",
    y="value",
    title="My Chart"
)
st.plotly_chart(fig, use_container_width=True)
```

**Example - Custom Filter:**
```python
from dashboards.components.filters import create_state_filter

selected_states = create_state_filter(
    states=db.get_states(),
    key="my_filter"
)
```

**Example - Data Export:**
```python
from dashboards.components.export import create_export_section

data = db.get_master_health_record()
create_export_section(data)
```

---

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Easiest)
- **Pros:** Free, zero config, auto-deploy
- **Cons:** Public only (free tier), limited resources
- **Setup:** Connect GitHub → Deploy
- **URL:** `https://your-app.streamlit.app`

### Option 2: Docker (Recommended)
- **Pros:** Full control, reproducible, portable
- **Cons:** Requires Docker knowledge
- **Setup:** `docker-compose up -d`
- **URL:** `http://localhost:8501`

### Option 3: Cloud Platforms
- **AWS ECS:** Container orchestration, auto-scaling
- **GCP Cloud Run:** Serverless containers, pay-per-use
- **Azure ACI:** Simple container deployment
- **Setup:** See [DEPLOYMENT.md](DEPLOYMENT.md)

### Option 4: On-Premise Server
- **Pros:** Full control, custom security
- **Cons:** Requires server management
- **Setup:** Install Python, run `streamlit run app.py`

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

---

## 🔒 Security

### Authentication

Enable Streamlit authentication:
```bash
streamlit run app.py --server.enableXsrfProtection=true
```

Or use secrets management:
```toml
# .streamlit/secrets.toml
[authentication]
username = "admin"
password = "secure-password"
```

### Best Practices
1. **Use HTTPS** - Deploy behind SSL/TLS
2. **Read-only DB** - Database connection is read-only
3. **Validate inputs** - All user inputs are validated
4. **Secrets management** - Never commit secrets to git
5. **Audit logging** - Track user actions (future feature)

---

## 📊 Performance

### Metrics
- **Page Load:** < 2 seconds (target)
- **Query Response:** < 1 second (cached)
- **Cache TTL:** 5 minutes (configurable)
- **Concurrent Users:** 100+ supported

### Optimization Tips
1. **Increase cache TTL** - Edit `config.py`
2. **Limit query results** - Use `LIMIT` in SQL
3. **Use Polars** - Faster than Pandas
4. **Add indexes** - Optimize DuckDB queries
5. **Deploy with more resources** - 4GB+ RAM recommended

---

## 🐛 Troubleshooting

### Database Not Found
```
❌ Database Not Found
Expected location: /path/to/ahgd.db
```
**Solution:** Run ETL pipeline first or check DB path in `config.py`

### Import Errors
```
ModuleNotFoundError: No module named 'streamlit'
```
**Solution:** `pip install -r requirements.txt`

### Slow Performance
**Solutions:**
- Increase `CACHE_TTL` in config
- Deploy with more resources (4GB+ RAM)
- Use data sampling for large visualizations
- Check database query performance

### Port Already in Use
```
Error: Port 8501 is already in use
```
**Solution:** `streamlit run app.py --server.port 8502`

---

## 📝 Development

### Code Style
- British English conventions (optimise, standardise)
- Polars for data manipulation (faster than Pandas)
- Type hints where appropriate
- Docstrings for all functions
- Follow existing patterns

### Testing
```bash
# Syntax check all Python files
python3 -m py_compile dashboards/**/*.py

# Run dashboard locally
streamlit run app.py

# Test with sample data
# (create sample ahgd.db first)
```

### Adding New Metrics
1. Define in `config.py` under `METRICS_CONFIG`
2. Add SQL query to `utils/database.py`
3. Create visualization in page file
4. Add to filters if needed

### Version Control
```bash
# Create feature branch
git checkout -b feature/new-analysis

# Make changes and commit
git add dashboards/
git commit -m "Add new analysis feature"

# Push and create PR
git push origin feature/new-analysis
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📄 License

MIT License - See LICENSE file in project root

---

## 🔗 Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [DuckDB Documentation](https://duckdb.org/docs)
- [Plotly Documentation](https://plotly.com/python)
- [Polars Documentation](https://pola-rs.github.io/polars/py-polars/html/reference/)
- [AHGD Project](../README.md)
- [Deployment Guide](DEPLOYMENT.md)

---

## 📞 Support

For issues or questions:
1. Check [Troubleshooting](#-troubleshooting) section
2. Review [DEPLOYMENT.md](DEPLOYMENT.md) for deployment issues
3. Check logs: `streamlit run app.py --logger.level=debug`
4. Open an issue on GitHub
5. Contact AHGD development team

---

## 🎯 Roadmap

**Completed:**
- ✅ Overview dashboard with KPIs
- ✅ Geographic analysis with interactive maps
- ✅ Health indicators deep dive
- ✅ Socioeconomic impact analysis
- ✅ Climate & environment correlations
- ✅ Data quality monitoring
- ✅ Export functionality (CSV, Excel, Parquet)
- ✅ Docker deployment
- ✅ Reusable components

**Future Enhancements:**
- [ ] Real-time auto-refresh when pipeline runs
- [ ] Anomaly detection alerts
- [ ] Predictive analytics integration
- [ ] ML-based health risk scoring
- [ ] User authentication and roles
- [ ] Custom report builder
- [ ] Email alerts for data quality issues
- [ ] Mobile app version

---

**Built with ❤️ using Streamlit + DuckDB + Polars + Plotly**

*Last updated: November 2025*
