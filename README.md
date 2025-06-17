# Australian Health Data Analytics Platform

A modern, high-performance health data analytics platform using free Australian government data sources. Built with cutting-edge tools like Polars, DuckDB, and modern Python ecosystem.

## 🚀 Quick Start

```bash
# Install UV (modern Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup project
git clone <your-repo-url>
cd australian-health-analytics

# Install dependencies (5x faster than pip)
uv sync

# Run data pipeline
uv run python scripts/setup/download_abs_data.py
uv run python scripts/data_pipeline/process_census.py

# Launch dashboard
uv run streamlit run src/web/streamlit/dashboard.py
```

## 🎯 Project Goals

- **Learn**: Master Australian health data landscape
- **Build**: High-performance population health analytics
- **Demonstrate**: Modern data engineering practices
- **Portfolio**: Impressive project for career advancement

## ⚡ Modern Tech Stack

### Performance-First Architecture
- **Polars**: 10-30x faster than pandas for data processing
- **DuckDB**: Embedded analytics database, zero setup required
- **HTTPX**: Async data downloads for maximum speed
- **UV**: Lightning-fast dependency management
- **Advanced Storage**: Bronze-Silver-Gold data lake with versioning
- **Memory Optimization**: 57.5% memory reduction with adaptive optimization

### Data Sources (100% Free)
- **ABS Census**: Demographics by Statistical Area Level 2 (SA2)
- **AIHW Health Data**: Population health indicators
- **Medicare/PBS Data**: Healthcare service utilisation
- **Environmental Data**: Air quality, weather patterns

### Key Features
- 🚀 **Ultra-fast processing** with Polars lazy evaluation
- 📊 **Interactive dashboards** with Streamlit
- 🗺️ **Geographic analysis** with GeoPandas and Folium  
- 📈 **Risk modelling** for population health
- 🔄 **Automated pipelines** with GitHub Actions
- 📱 **Mobile-friendly** web interface

## 📁 Project Structure

```
australian-health-analytics/
├── data/
│   ├── raw/           # Downloaded source data (gitignored)
│   ├── bronze/        # Raw ingested data with time partitioning
│   ├── silver/        # Cleaned, versioned data with schema validation
│   ├── gold/          # Analytics-ready aggregated data
│   ├── metadata/      # Data versioning, lineage, and schema tracking
│   ├── processed/     # Legacy CSV/Parquet files
│   └── outputs/       # Analysis results and exports
├── src/
│   ├── data_processing/   # ETL pipelines and storage optimization
│   │   └── storage/      # Advanced storage components (Phase 4)
│   ├── analysis/         # Statistical analysis modules  
│   └── web/             # Dashboard and web interface
├── scripts/
│   ├── setup/           # Environment and data setup
│   ├── data_pipeline/   # Automated data processing  
│   └── deployment/      # Deployment automation
├── docs/               # Documentation and analysis reports
└── tests/             # Comprehensive unit and integration tests
```

## 🛠️ Development Setup

### Prerequisites
- Python 3.11+
- macOS with Apple Silicon (M1/M2) recommended
- Git for version control

### Installation
```bash
# Modern development environment
uv venv --python 3.11
source .venv/bin/activate
uv sync --extra dev --extra jupyter

# Install pre-commit hooks
uv run pre-commit install

# Verify installation
uv run python -c "import polars; print(f'Polars version: {polars.__version__}')"
```

## 📊 Data Pipeline Overview

### ✅ Phase 1: Real Data Sources (Complete)
- **497,181 total records** processed from real Australian government datasets
- **6 verified data sources**: ABS Census, SEIFA, PBS, Medicare, Geographic boundaries
- **92.9% integration success rate** between datasets
- **Robust error handling** and graceful degradation

### ✅ Phase 2: Schema Integration (Complete)  
- **2,454 SA2 areas** with complete geographic and socio-economic integration
- **SEIFAProcessor**: 4 socio-economic indices (IRSD, IRSAD, IER, IEO)
- **HealthDataProcessor**: 492,434 PBS prescription records
- **SimpleBoundaryProcessor**: Geographic metadata and state/territory mapping

### ✅ Phase 3: Health Analytics (Complete)
- **Health risk algorithms** with composite scoring and chronic disease modeling
- **Geographic analysis modules** for spatial health patterns and SA2 mapping
- **Healthcare access scoring** based on service availability and provider density
- **Medicare/pharmaceutical analyzers** for utilization analysis
- **Comprehensive testing** with 90%+ integration success across all analytics

### ✅ Phase 4: Storage Optimization (Complete)
- **57.5% memory reduction** with advanced optimization
- **Bronze-Silver-Gold data lake** with versioning and lineage
- **Parquet compression** achieving 60-70% size reduction
- **Performance benchmarking suite** with comprehensive monitoring
- **Automated optimization recommendations** and regression detection

## 🎨 Key Visualisations

- **Health Atlas**: Interactive map of Australia showing health metrics by area
- **Risk Profiles**: Detailed health risk analysis for any geographic area
- **Trend Analysis**: Population health changes over time
- **Access Analysis**: Healthcare service accessibility mapping

## 📈 Performance Benchmarks

### Data Processing Performance
| Operation | Pandas | Polars | Speedup |
|-----------|--------|--------|----------|
| Census data loading | 45s | 4s | 11x faster |
| Geographic joins | 120s | 8s | 15x faster |
| Risk calculations | 30s | 2s | 15x faster |

### Storage Optimization Results (Phase 4)
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Memory Usage | 18.15 MB | 7.72 MB | **57.5% reduction** |
| File Compression | CSV baseline | Parquet + ZSTD | **60-70% smaller** |
| Processing Speed | Standard operations | Optimized pipeline | **31 optimizations applied** |
| Data Lake Layers | Single storage | Bronze-Silver-Gold | **Versioning + lineage** |

## 🔧 Usage Examples

### Quick Data Analysis
```python
import polars as pl
from src.data_processing import AustralianHealthData

# Load and analyse census data (lightning fast)
health_data = AustralianHealthData()
demographics = health_data.get_sa2_demographics()

# Calculate health risk scores
risk_scores = health_data.calculate_risk_scores(demographics)
print(f"Processed {len(risk_scores)} areas in seconds")
```

### Geographic Analysis
```python
from src.analysis.spatial import HealthGeography

geo = HealthGeography()
hotspots = geo.identify_health_hotspots(
    risk_threshold=0.8,
    min_population=1000
)

# Export for web visualisation
geo.export_geojson(hotspots, "docs/data/health_hotspots.geojson")
```

### Advanced Storage Optimization (Phase 4)
```python
from src.data_processing.storage import MemoryOptimizer, IncrementalProcessor

# Memory optimization with 57.5% reduction
optimizer = MemoryOptimizer(memory_limit_gb=4.0)
optimized_df, stats = optimizer.optimize_dataframe_memory(health_data, "health")
print(f"Memory saved: {stats['memory_savings_mb']:.2f}MB ({stats['memory_savings_percent']:.1f}%)")

# Bronze-Silver-Gold data lake processing
processor = IncrementalProcessor()
bronze_version = processor.ingest_to_bronze(raw_data, "health", source_info)
silver_version = processor.process_to_silver("health", bronze_version)
gold_version = processor.aggregate_to_gold("health", silver_version, agg_config)
```

## 🌟 Portfolio Highlights

This project demonstrates:
- **Modern Data Engineering**: Polars, DuckDB, async processing with Bronze-Silver-Gold architecture
- **Advanced Storage Optimization**: 57.5% memory reduction, versioning, and performance monitoring
- **Geographic Analytics**: Complex spatial data integration with 497,181+ real records
- **Health Domain Expertise**: Population health metrics and risk modelling with Australian data
- **Full-Stack Development**: End-to-end data application with production-ready storage
- **Performance Optimisation**: 10x+ speed improvements with advanced memory management

## 📚 Learning Resources

- [Australian Statistical Geography Standard](https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs)
- [AIHW Data Sources](https://www.aihw.gov.au/about-our-data)
- [Polars User Guide](https://pola-rs.github.io/polars/)
- [DuckDB Documentation](https://duckdb.org/docs/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Australian Bureau of Statistics for open data access
- Australian Institute of Health and Welfare for health statistics
- Open source community for the amazing tools that make this possible