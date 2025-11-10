# AHGD Dashboard

Interactive dashboard for Australian Health Geography Data, providing visualizations and analysis of health indicators, demographics, and socioeconomic factors across Australian Statistical Areas.

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

## 📊 Dashboard Features

### Page 1: Overview Dashboard
- **KPI Cards:** Total SA2 regions, average mortality rate, utilisation rate, population coverage
- **Health Indicator Distribution:** Histograms and scatter plots
- **Top/Bottom Regions:** Best and worst performing SA2s by composite health index
- **Remoteness Analysis:** Health metrics by remoteness category

### Page 2: Geographic Analysis *(Coming Soon)*
- Interactive choropleth maps of Australia
- Click SA2 regions to view details
- State-by-state comparisons
- Spatial health clustering

### Page 3: Health Indicators Deep Dive *(Coming Soon)*
- Mortality rate breakdowns
- Medicare utilisation analysis
- Correlation matrix
- Risk factor identification

### Page 4: Socioeconomic Impact *(Coming Soon)*
- SEIFA analysis
- Income vs health outcomes
- Unemployment impact
- Population demographics

### Page 5: Climate & Environment *(Coming Soon)*
- Climate correlations with health
- Environmental health factors
- Seasonal patterns

### Page 6: Data Quality *(Coming Soon)*
- Pipeline status and last run
- Data freshness indicators
- Coverage statistics
- dbt test results

## 🛠️ Architecture

```
dashboards/
├── app.py                 # Main Streamlit app
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── pages/                 # Multi-page dashboard pages
│   └── 1_📊_Overview.py   # Overview dashboard
├── components/            # Reusable UI components
└── utils/                 # Utility modules
    └── database.py        # DuckDB connection management
```

## 🎨 Customization

### Configuration

Edit `config.py` to customize:
- Database path
- Color schemes
- Map settings
- Cache durations
- Metrics thresholds

### Adding New Pages

1. Create a new file in `pages/` with format: `N_icon_PageName.py`
2. Example: `7_🔍_CustomAnalysis.py`
3. Streamlit will automatically add it to the sidebar navigation

### Custom Visualizations

Add reusable components to `components/` directory:
- `maps.py` - Map components
- `charts.py` - Chart templates
- `filters.py` - Filter widgets

## 📈 Data Connection

The dashboard connects to DuckDB database at:
```
../ahgd.db  (relative to dashboards directory)
```

Ensure the database exists and contains these tables:
- `master_health_record` - Main health and demographic data
- `derived_health_indicators` - Calculated health metrics

## 🔄 Data Updates

The dashboard caches query results for 5 minutes for performance. Data updates automatically when:
- Cache expires (5 minutes)
- User clicks "Refresh Data" button
- Dashboard is restarted

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Easiest)

1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repository
4. Deploy!

### Option 2: Docker

```bash
# Build Docker image
docker build -t ahgd-dashboard .

# Run container
docker run -p 8501:8501 -v /path/to/ahgd.db:/app/ahgd.db ahgd-dashboard
```

### Option 3: Cloud Platform

Deploy to AWS ECS, GCP Cloud Run, or Azure Container Instances using the provided Dockerfile.

## 🔒 Security

For production deployment:

1. **Enable authentication:**
   ```bash
   streamlit run app.py --server.enableXsrfProtection=true
   ```

2. **Use secrets management:**
   Create `.streamlit/secrets.toml`:
   ```toml
   [database]
   path = "/secure/path/to/ahgd.db"
   ```

3. **Configure HTTPS:**
   Deploy behind a reverse proxy (nginx, Caddy) with SSL

## 📊 Performance

- Page load time: < 2 seconds
- Query response time: < 1 second
- Cached queries: 5-minute TTL
- Supports 100+ concurrent users

## 🐛 Troubleshooting

### Database Not Found

```
❌ Database Not Found
Expected location: /path/to/ahgd.db
```

**Solution:** Ensure the ETL pipeline has run and `ahgd.db` exists in the project root.

### Import Errors

```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Slow Performance

**Solutions:**
- Increase cache TTL in `config.py`
- Limit data queries with `LIMIT` clauses
- Use Polars instead of Pandas for large datasets
- Deploy to a server with more resources

## 📝 Development

### Adding New Metrics

1. Define metric in `config.py` under `METRICS_CONFIG`
2. Add SQL query to `utils/database.py`
3. Create visualization in page file

### Code Style

- Follow British English conventions (optimise, not optimize)
- Use Polars for data manipulation (faster than Pandas)
- Cache expensive operations with `@st.cache_data`
- Document functions with docstrings

## 🤝 Contributing

1. Create feature branch
2. Implement changes
3. Test locally
4. Submit pull request

## 📄 License

MIT License - See LICENSE file in project root

## 🔗 Links

- [Streamlit Documentation](https://docs.streamlit.io)
- [DuckDB Documentation](https://duckdb.org/docs)
- [Plotly Documentation](https://plotly.com/python)
- [AHGD Project](../README.md)

## 📞 Support

For issues or questions:
- Check the troubleshooting section above
- Review Streamlit logs
- Check DuckDB database contents
- Contact the AHGD development team
