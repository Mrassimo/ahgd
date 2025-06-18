# Project Navigation Guide

Welcome to the Australian Health Data Analytics project! This guide helps you navigate the comprehensive, well-organised project structure.

## 🎯 Start Here

**New to the project?** → [README.md](README.md)  
**Want to run the dashboard?** → `python run_dashboard.py`  
**Need to run tests?** → `python run_tests.py`  
**Complete setup?** → `python setup_and_run.py`

## 📂 Directory Structure Overview

### 🏠 Root Directory
**What's here**: Main entry points and configuration files
- `README.md` - Project overview and quick start
- `main.py` - Core application entry point
- `setup_and_run.py` - Complete setup script
- `run_dashboard.py` - Launch dashboard
- `run_tests.py` - Execute test suite
- `pyproject.toml` - Project configuration

### 📁 Core Directories

#### `src/` - Application Code
**Purpose**: All production application code
- `config.py` - Configuration management
- `dashboard/` - Complete dashboard application
- `performance/` - Performance monitoring system

#### `scripts/` - Utility Scripts
**Purpose**: Data processing, analysis, and utility scripts
- **[Full Script Index](scripts/INDEX.md)** 📋
- `data_processing/` - Data extraction and processing
- `analysis/` - Statistical analysis scripts  
- `dashboard/` - Dashboard demos and utilities
- `utils/` - General utility scripts

#### `tests/` - Testing Framework
**Purpose**: Comprehensive testing suite (95%+ coverage)
- **[Testing Documentation](tests/README.md)** 📋
- `unit/` - Unit tests
- `integration/` - Integration tests
- `fixtures/` - Test data and fixtures

#### `docs/` - Documentation
**Purpose**: All project documentation organised by type
- **[Documentation Index](docs/INDEX.md)** 📋
- `guides/` - User and developer guides
- `reference/` - Technical reference materials
- `api/` - Auto-generated API documentation
- `assets/` - Images and interactive content

#### `reports/` - Analysis Reports
**Purpose**: Analysis results, assessments, and reports
- **[Reports Index](reports/INDEX.md)** 📋
- `analysis/` - Data analysis reports
- `testing/` - Test results and coverage
- `deployment/` - Production readiness assessments
- `health/` - System health reports
- `coverage/` - Test coverage reports

#### `data/` - Data Storage
**Purpose**: Raw and processed data (1.2GB total)
- `health_analytics.db` - Main SQLite database (5.5MB)
- `raw/` - Downloaded government data
- `processed/` - Cleaned and transformed data

#### `logs/` - Application Logs
**Purpose**: Application and system logs
- `ahgd.log` - Main application log
- `data_download.log` - Data processing log

## 🎭 Navigation by Role

### 👤 End Users
**Goal**: Use the dashboard and understand the analysis

1. **Start** → [README.md](README.md) for project overview
2. **Launch** → `python run_dashboard.py` to start dashboard
3. **Learn** → [Dashboard User Guide](docs/guides/dashboard_user_guide.md)
4. **Explore** → [Analysis Reports](reports/INDEX.md) for insights

### 💻 Developers
**Goal**: Understand, modify, or extend the codebase

1. **Overview** → [README.md](README.md) for project context
2. **Setup** → `python setup_and_run.py` for development environment
3. **Code** → [API Documentation](docs/api/) for code reference
4. **Test** → [Testing Documentation](tests/README.md) for testing approach
5. **Deploy** → [CI/CD Guide](docs/guides/CI_CD_GUIDE.md) for deployment

### 📊 Data Analysts
**Goal**: Understand data sources and analysis methods

1. **Data Sources** → [Data Sources](docs/reference/REAL_DATA_SOURCES.md)
2. **Analysis Methods** → [Methodology](docs/reference/health_risk_methodology.md)
3. **Scripts** → [Scripts Index](scripts/INDEX.md) for analysis tools
4. **Results** → [Analysis Reports](reports/analysis/) for findings

### 🔧 DevOps/SysAdmins
**Goal**: Deploy and maintain the system

1. **Deployment** → [Deployment Guide](docs/guides/DEPLOYMENT_GUIDE.md)
2. **Operations** → [Operational Runbooks](docs/guides/OPERATIONAL_RUNBOOKS.md)
3. **Monitoring** → [Performance Monitoring](docs/guides/PERFORMANCE_MONITORING_GUIDE.md)
4. **Health** → [System Health Reports](reports/health/)

### 📋 Project Managers
**Goal**: Understand project status and capabilities

1. **Overview** → [README.md](README.md) for project summary
2. **Status** → [Production Assessment](reports/deployment/FINAL_PRODUCTION_ASSESSMENT.md)
3. **Progress** → [Analysis Reports](reports/INDEX.md) for achievements
4. **Planning** → [Next Steps](docs/reference/IMMEDIATE_NEXT_STEPS.md)

## 🔍 Finding Specific Information

### Configuration
- **Application Config** → `src/config.py`
- **Project Config** → `pyproject.toml`
- **Environment** → `.env.template` (example)

### Data
- **Database** → `data/health_analytics.db`
- **Raw Data** → `data/raw/`
- **Processed Data** → `data/processed/`
- **Data Documentation** → `docs/reference/REAL_DATA_SOURCES.md`

### Testing
- **Run Tests** → `python run_tests.py`
- **Test Code** → `tests/`
- **Coverage Report** → `reports/coverage/htmlcov/index.html`
- **Test Documentation** → `tests/README.md`

### Performance
- **Performance Dashboard** → `python src/performance/performance_dashboard.py`
- **Monitoring** → `src/performance/`
- **Performance Guide** → `docs/guides/PERFORMANCE_MONITORING_GUIDE.md`

### Development
- **Scripts** → `scripts/` (organised by purpose)
- **Source Code** → `src/`
- **API Docs** → `docs/api/`
- **Developer Guide** → `docs/guides/` (multiple guides)

## 🚀 Common Tasks

### Getting Started
```bash
# Complete setup and launch
python setup_and_run.py

# Or step by step:
python run_dashboard.py                        # Launch dashboard
python run_tests.py                            # Run tests
python src/performance/performance_dashboard.py # Monitor performance
```

### Development Workflow
```bash
# Run specific script categories
python scripts/data_processing/process_data.py
python scripts/analysis/health_correlation_analysis.py
python scripts/utils/health_check.py
```

### Documentation
```bash
# Build API documentation
python scripts/utils/build_docs.py

# View built docs
open docs/build/html/index.html
```

## 📞 Support and Maintenance

### Issue Tracking
- **Current Tasks** → `docs/reference/todo.md`
- **Next Steps** → `docs/reference/IMMEDIATE_NEXT_STEPS.md`

### Health Monitoring
- **System Health** → `python scripts/utils/health_check.py`
- **Performance** → `python performance_dashboard.py`
- **Logs** → `logs/` directory

### Updates and Maintenance
- **Operational Runbooks** → `docs/guides/OPERATIONAL_RUNBOOKS.md`
- **Health Reports** → `reports/health/`

---

**This guide is maintained alongside the project structure. All links are verified and current.**