# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Initial setup (development mode with all tools)
./setup_env.sh --dev

# Production setup
./setup_env.sh

# Activate environment
source venv/bin/activate
```

### Testing
```bash
# Run all tests
python -m pytest

# Run unit tests only
python -m pytest tests/unit/

# Run integration tests only
python -m pytest tests/integration/

# Run with coverage
python -m pytest --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/unit/test_extractors.py

# Run tests with specific markers
python -m pytest -m "not slow"
python -m pytest -m integration
```

### Code Quality
```bash
# Format code with Black
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Run all pre-commit hooks
pre-commit run --all-files

# Install pre-commit hooks
pre-commit install
```

### Data Pipeline (DVC)
```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro extract_features

# Show pipeline status
dvc status

# Show pipeline DAG
dvc dag

# Track data changes
dvc add data_raw/
dvc push
```

### CLI Tools
```bash
# ETL pipeline commands (defined in pyproject.toml)
ahgd-extract --source aihw --output data_raw/
ahgd-transform --input data_raw/ --output data_processed/
ahgd-validate --input data_processed/ --rules schemas/
ahgd-pipeline --config configs/production.yaml
```

## Architecture Overview

### ETL Framework Design
The codebase implements a modular ETL (Extract, Transform, Load) pipeline for Australian health and geographic data. The architecture follows these patterns:

#### Base Classes (`src/*/base.py`)
- **BaseExtractor**: Handles data extraction with retry logic, progress tracking, and checksum validation
- **BaseTransformer**: Manages data transformation with schema enforcement and audit trails  
- **BaseValidator**: Implements comprehensive validation (schema, business rules, statistical, geographic)
- **BaseLoader**: Provides multi-format export with compression and partitioning strategies

All base classes implement standardised interfaces defined in `src/utils/interfaces.py` with common metadata structures and error handling.

#### Configuration System (`src/utils/config.py`)
Environment-aware configuration management supporting:
- Multi-format loading (YAML/JSON/ENV)
- Environment-specific overrides (development.yaml, production.yaml, testing.yaml)
- Hot-reloading for development
- Secrets management with multiple providers
- Configuration validation using JSON Schema

#### Logging Architecture (`src/utils/logging.py`)
Structured logging framework combining loguru and structlog:
- Environment-specific log levels and formats
- Separate log files by component (ETL, validation, errors, performance)  
- Data lineage tracking with operation context
- Performance monitoring with decorators
- Thread-safe context management

### Data Schema Management (`schemas/`)
Pydantic-based schema system with:
- Base schemas with versioning support
- Domain-specific schemas (SA2 geographic, health indicators, SEIFA)
- Migration utilities for schema evolution
- Comprehensive validation rules

### Data Flow
1. **Extract**: Source-specific extractors download and validate raw data
2. **Transform**: Standardise column names, handle missing values, enforce schemas
3. **Validate**: Multi-layer validation (schema, business rules, statistical checks)
4. **Load**: Export to multiple formats with optimisation and partitioning

## Key Implementation Patterns

### British English Conventions
All code follows British English spelling:
- `optimise` not `optimize`
- `standardise` not `standardize`  
- `initialise` not `initialize`

### Error Handling
Use the exception hierarchy from `src/utils/interfaces.py`:
- `AHGDError` (base)
- `ExtractionError`, `TransformationError`, `ValidationError`, `LoadingError`

### Configuration Access
```python
from src.utils.config import get_config, get_config_manager

# Simple access
database_url = get_config("database.url")

# Type-safe access
max_workers = get_config_manager().get_typed("system.max_workers", int)

# Environment detection
from src.utils.config import is_development, is_production
```

### Logging Usage
```python
from src.utils.logging import get_logger, monitor_performance, track_lineage

logger = get_logger(__name__)

# Performance monitoring
@monitor_performance("data_extraction")
def extract_data():
    pass

# Operation context
with logger.operation_context("data_processing"):
    logger.log.info("Processing commenced", records=5000)

# Data lineage
track_lineage("raw_health_data", "processed_health_data", "standardisation")
```

### Adding New Data Sources
1. Create extractor class inheriting from `BaseExtractor`
2. Define schema in `schemas/` directory
3. Add transformation logic inheriting from `BaseTransformer`
4. Configure validation rules in validator
5. Update DVC pipeline in `dvc.yaml`

## Environment-Specific Behaviour

### Development
- Debug logging enabled
- Hot-reload configuration
- Mock external services
- SQLite for local development
- Reduced resource limits

### Production  
- Optimised logging levels
- Real external service integration
- PostgreSQL database
- Comprehensive monitoring
- Security hardening

### Testing
- Minimal logging
- In-memory databases
- Mock all external dependencies
- Fast execution optimisations

## Data Integrity Principles

The codebase enforces these data integrity measures:
1. **Immutable Pipeline**: All transformations are version controlled
2. **Audit Trails**: Complete logging of data modifications
3. **Validation Gates**: Multi-layer validation before data proceeds
4. **Reproducibility**: Deterministic processes with dependency tracking
5. **Rollback Capability**: Version management for data and configuration

## Development Strategies

### Test-Driven Development
- Test driven development for the ETL, we have an idea about how the data should look.. lets conform to meet those standards