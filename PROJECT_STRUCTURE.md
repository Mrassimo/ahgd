# Australian Health Analytics - Project Structure

## 📁 Directory Organisation

### Core Application (`src/`)
```
src/
├── __init__.py                    # Package initialisation
├── cli.py                         # Command-line interface
├── data_processing/               # Data processing pipeline
│   ├── core.py                   # Core orchestration
│   ├── *_processor.py            # Data processors (SEIFA, health, boundary)
│   ├── downloaders/              # Data download utilities
│   ├── storage/                  # Storage optimization (Phase 4)
│   └── validators/               # Data validation utilities
├── analysis/                     # Health analytics modules (Phase 3)
│   ├── health/                   # Health-specific analysis
│   ├── risk/                     # Risk assessment algorithms
│   └── spatial/                  # Geographic analysis
└── web/                          # Web interface components
    ├── streamlit/                # Streamlit dashboard
    └── static/                   # Static web assets
```

### Testing Framework (`tests/`)
```
tests/
├── conftest.py                   # Pytest configuration
├── test_data_processing/         # Unit tests (Phase 5.1)
├── integration/                  # Integration tests (Phase 5.2)
├── data_quality/                 # Data quality tests (Phase 5.3)
├── performance/                  # Performance tests (Phase 5.4)
├── security/                     # Security tests (Phase 5.6)
├── cicd/                         # CI/CD tests (Phase 5.7)
├── fixtures/                     # Test data fixtures
└── utils/                        # Testing utilities
```

### Data Lake Structure (`data/`)
```
data/
├── raw/                          # Raw Australian government data
│   ├── abs/                     # Australian Bureau of Statistics
│   ├── aihw/                    # Australian Institute of Health
│   ├── census/                  # Census 2021 data
│   └── geographic/              # Boundary shapefiles
├── bronze/                       # Raw data with basic structure
├── silver/                       # Cleaned and validated data
├── gold/                         # Analytics-ready aggregated data
├── processed/                    # Legacy processed data
├── outputs/                      # Analysis outputs
├── cache/                        # Temporary cache files
└── metadata/                     # Data lineage and versioning
```

### Documentation (`docs/`)
```
docs/
├── README.md                     # Project overview
├── reports/                      # Phase completion reports
│   ├── PHASE_*_COMPLETION_REPORT.md
│   ├── PORTFOLIO_*.md
│   └── PROJECT_STATUS_SUMMARY.md
├── architecture/                 # Design documents
│   ├── IMPLEMENTATION_PLAN.md
│   └── *_PLAN.md
├── api/                          # API documentation
├── analysis/                     # Data analysis documentation
└── index.html                    # GitHub Pages site
```

### Scripts and Automation (`scripts/`)
```
scripts/
├── setup/                        # Setup and initialisation
│   ├── quick_start.py
│   ├── create_mock_data.py
│   └── download_abs_data.py
├── data_pipeline/                # Data processing scripts
├── deployment/                   # Deployment automation
├── web_export/                   # Web export utilities
├── run_*.py                      # Test execution scripts
└── launch_portfolio.py           # Portfolio launcher
```

### Configuration (`config/` & Root)
```
├── pyproject.toml               # Python project configuration
├── requirements.txt             # Dependencies
├── setup.py                     # Package setup
├── Dockerfile                   # Container configuration
├── CLAUDE.md                    # Development instructions
├── README.md                    # Main project README
└── TODO.md                      # Task tracking
```

## 🏗️ Architecture Overview

### Data Processing Pipeline
1. **Raw Data Ingestion** → ABS/AIHW/Census data download
2. **Bronze Layer** → Raw data with basic structure and partitioning
3. **Silver Layer** → Cleaned, validated, and versioned data
4. **Gold Layer** → Analytics-ready aggregated data
5. **Analysis Engine** → Health risk assessment and geographic analysis

### Storage Optimization (Phase 4)
- **Parquet Storage**: 60-70% compression with column optimization
- **Memory Optimization**: 57.5% memory reduction with lazy loading
- **Incremental Processing**: Version management and change detection
- **Performance Monitoring**: Real-time performance tracking

### Testing Framework (Phase 5)
- **Unit Testing**: 150+ tests with >90% coverage
- **Integration Testing**: End-to-end pipeline validation
- **Data Quality**: Australian health data compliance validation
- **Performance Testing**: 1M+ record stress testing
- **Security Testing**: Australian Privacy Principles compliance
- **CI/CD Testing**: Production deployment validation

### Web Interface
- **Streamlit Dashboard**: Interactive health analytics interface
- **GitHub Pages**: Static documentation and portfolio site
- **Mobile Responsive**: Cross-device compatibility
- **Performance Optimized**: <2 second load times

## 📊 Key Metrics

- **Data Processed**: 497,181+ Australian health records
- **Geographic Coverage**: 2,454 SA2 statistical areas
- **Integration Success**: 92.9% cross-dataset alignment
- **Memory Optimization**: 57.5% reduction achieved
- **Test Coverage**: 85-90% across critical components
- **Performance**: <2 second dashboard load times

## 🔧 Quick Start

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Run initial setup
python scripts/setup/quick_start.py

# Execute full pipeline
python scripts/run_unified_etl.py

# Launch dashboard
python scripts/launch_portfolio.py

# Run comprehensive tests
python scripts/run_integration_tests.py
```

## 📚 Development Guidelines

1. **Follow existing patterns** in `src/data_processing/` for new processors
2. **Add comprehensive tests** in appropriate `tests/` subdirectory
3. **Update documentation** in `docs/` for significant changes
4. **Use data lake structure** for all data storage operations
5. **Follow Australian health data standards** for compliance

## 🎯 Portfolio Highlights

This project demonstrates:
- **Enterprise Data Engineering**: Production-scale Australian health data processing
- **Modern Tech Stack**: Polars, Streamlit, Docker, pytest, CI/CD
- **Data Lake Architecture**: Bronze-Silver-Gold with versioning
- **Comprehensive Testing**: Unit, integration, performance, security testing
- **Australian Compliance**: Privacy Principles and health data standards
- **Performance Engineering**: Memory optimization and storage efficiency
- **DevOps Excellence**: CI/CD pipelines and production deployment