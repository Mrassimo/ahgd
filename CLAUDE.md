# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Install dependencies with UV (5x faster than pip)
uv sync

# Development dependencies  
uv sync --extra dev --extra jupyter --extra docs

# Quick project initialization
python scripts/setup/quick_start.py --states nsw,vic
```

### Core Application Commands
```bash
# Download Australian health data
uv run health-analytics download --states nsw,vic,qld

# Process data with Polars + DuckDB pipeline
uv run health-analytics process

# Launch interactive dashboard
uv run health-analytics dashboard --port 8501

# Check data and processing status
uv run health-analytics status
```

### Code Quality & Testing
```bash
# Linting and formatting
uv run black src/
uv run isort src/
uv run flake8 src/

# Type checking
uv run mypy src/

# Run tests with coverage
uv run pytest tests/ --cov=src --cov-report=html

# Run single test
uv run pytest tests/test_specific.py::test_function_name -v
```

### Docker Operations
```bash
# Build container
docker build -t australian-health-analytics .

# Run dashboard in container
docker run -p 8501:8501 australian-health-analytics

# Run with data volume
docker run -v $(pwd)/data:/app/data -p 8501:8501 australian-health-analytics
```

## Architecture Overview

### Modern Data Stack Philosophy
This project implements a performance-first data architecture using cutting-edge tools:

- **Polars**: Primary data engine for 10-30x faster processing than pandas
- **DuckDB**: Embedded analytics database requiring zero configuration
- **Async Processing**: Parallel downloads and transformations throughout
- **Lazy Evaluation**: Memory-efficient processing of large Australian datasets

### Data Processing Pipeline Architecture

The pipeline follows a three-layer architecture:

1. **Acquisition Layer** (`src/data_processing/downloaders/`): Async downloads from Australian Bureau of Statistics with automatic retry and progress tracking

2. **Processing Layer** (`src/data_processing/`): 
   - `AustralianHealthData`: Main orchestrator with DuckDB workspace management
   - `CensusProcessor`: Specialised census data transformations with SEIFA integration
   - Polars lazy evaluation for memory-efficient processing

3. **Analysis Layer** (`src/analysis/`): Risk modelling, demographic analysis, and spatial calculations

### CLI and Web Interface Integration

The `src/cli.py` provides the primary interface using Typer with Rich formatting. The CLI orchestrates:
- Data acquisition workflows
- Processing pipeline execution
- Dashboard launching with environment setup
- Status monitoring and validation

The Streamlit dashboard (`src/web/streamlit/dashboard.py`) loads processed data using `@st.cache_data` and provides interactive visualizations with Altair charts and Folium maps.

### Data Flow and Storage Strategy

```
ABS Data Sources → Async Downloads → Polars Processing → DuckDB Analytics → Web Exports
```

Storage follows Git-native principles:
- `data/raw/`: Source downloads (gitignored)
- `data/processed/`: Analysis-ready CSV/Parquet files (version controlled for smaller datasets)
- `data/outputs/`: JSON/GeoJSON for web consumption
- DuckDB file: Local analytics workspace (not committed)

### Performance Optimizations

Key architectural decisions for speed:
- Polars lazy evaluation with `.scan_csv()` and `.collect()` pattern
- DuckDB spatial extensions for geographic operations
- HTTPX async client with connection pooling for downloads
- Streamlit caching for dashboard responsiveness

### Australian Health Data Specifics

The platform is designed around Australian statistical geography:
- **SA2 (Statistical Area Level 2)**: Primary geographic unit
- **SEIFA Indices**: Socio-economic disadvantage measures
- **ABS Census DataPacks**: Demographics by state
- **Health Risk Scoring**: Composite metrics combining SEIFA, demographics, and population density

### CI/CD Pipeline

GitHub Actions workflow implements:
- Automated testing with performance benchmarks comparing Polars vs pandas
- Weekly data updates triggered by schedule
- Docker image builds with health checks
- Security scanning with Trivy
- Documentation deployment to GitHub Pages

### Error Handling and Validation

The codebase implements robust error handling:
- Async download retry with exponential backoff
- Data validation using Great Expectations patterns
- DuckDB connection management with automatic cleanup
- Rich console output for user feedback

### Development Patterns

When extending the platform:
- Use Polars expressions for data transformations (avoid pandas patterns)
- Implement async methods for I/O operations
- Add type hints throughout (enforced by mypy)
- Use Rich console for user feedback
- Follow the three-layer architecture (acquisition → processing → analysis)

The CLI is the primary interface - the dashboard is for visualization only. All data operations should be accessible via `health-analytics` commands.