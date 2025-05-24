# Australian Healthcare Geographic Database (AHGD) ETL Pipeline

## Overview

The Australian Healthcare Geographic Database (AHGD) ETL Pipeline is a Python-based data processing system that transforms Australian Bureau of Statistics (ABS) Census and Geographic data into a queryable format for healthcare planning and analysis.

This pipeline enables evidence-based healthcare decision making by integrating geographic, demographic, and health data from across Australia.

## Key Features

- **Modular Design**: Separated processing logic for each Census table (G01, G17, G18, G19, G20, G21, G25)
- **Dimensional Modelling**: Star schema with surrogate keys for optimal analytical querying
- **Data Quality Validation**: Comprehensive validation checks for record counts, nulls, and referential integrity
- **Configurable Processing**: Configuration-driven approach with centralised settings
- **Comprehensive Testing**: Unit tests for all core components
- **Robust Logging**: Detailed logging for debugging and auditability
- **Flexible Execution**: Run specific processing steps or the entire pipeline

## Project Structure

```
AHGD/
├── run_etl_enhanced.py     # Main entry point for running ETL pipeline
├── verify_surrogate_keys.py # Validation for surrogate key relationships
├── data/                   # Data input files (raw Census, shapefiles)
├── output/                 # Generated Parquet files
├── logs/                   # Log files
├── ahgd_etl/               # Core ETL package
│   ├── config/             # Configuration system
│   │   ├── yaml/           # YAML configuration files
│   │   │   ├── schemas.yaml          # Schema definitions
│   │   │   ├── column_mappings.yaml  # Input-to-output mappings
│   │   │   └── data_sources.yaml     # External data source URLs
│   │   └── settings.py     # Configuration manager
│   ├── core/               # Core processing logic
│   │   └── temp_fix/       # Temporary fix logic (migrating to standard flows)
│   ├── loaders/            # Data loading with schema enforcement
│   ├── models/             # Dimension and fact table models
│   │   ├── dimensions.py   # Dimension table creation
│   │   └── time_dimension.py # Time dimension generation
│   ├── transformers/       # Data transformation logic
│   │   ├── census/         # Census table transformers
│   │   │   ├── g01_population.py
│   │   │   ├── g17_income.py
│   │   │   ├── g18_assistance_needed.py
│   │   │   ├── g19_health_conditions.py
│   │   │   ├── g20_selected_conditions.py
│   │   │   ├── g21_conditions_by_characteristics.py
│   │   │   └── g25_unpaid_assistance.py
│   │   └── geo/            # Geographic data transformers
│   │       └── geography.py # Geo data processing
│   ├── validators/         # Data validation
│   │   └── data_quality.py # Data quality validation
│   └── utils.py            # Utility functions
├── scripts/                # Utility scripts
├── docs/                   # Developer documentation
│   └── tooling/            # Tool-specific configuration templates
├── tests/                  # Test suite
│   ├── test_utils.py
│   ├── test_validation.py
│   └── test_data/          # Test data fixtures
└── documentation/          # Project documentation
    ├── etl_data_model_diagram.md  # Data model visualization
    ├── etl_outputs.md             # Output file documentation
    ├── configuration_guide.md     # Configuration system guide
    └── hardcoded_values_audit.md  # Audit of hardcoded values
```

The legacy `etl_logic/` module is maintained for backward compatibility as the code is migrated to the new `ahgd_etl/` package structure.

## Data Model

The pipeline generates a star schema consisting of:

### Dimension Tables
- **geo_dimension.parquet**: Geographic entities with surrogate keys and centroids (lat/lon)
- **dim_time.parquet**: Time dimension with date attributes
- **dim_health_condition.parquet**: Health conditions from ABS Census
- **dim_demographic.parquet**: Demographic information (age groups, sex)
- **dim_person_characteristic.parquet**: Person characteristics (income, employment, etc.)

### Fact Tables
- **fact_population.parquet**: Population counts by geography (G01)
- **fact_income.parquet**: Income statistics by geography (G17)
- **fact_assistance_needed.parquet**: Assistance need by geography (G18)
- **fact_health_conditions.parquet**: Health condition prevalence (G19)
- **fact_health_conditions_refined.parquet**: Refined health conditions with dimension keys
- **fact_health_conditions_by_characteristic_refined.parquet**: Health conditions by characteristic (G21)
- **fact_no_assistance.parquet**: Unpaid assistance statistics (G25)

## Setup & Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/ahgd.git
cd ahgd
```

2. Create a virtual environment and activate it
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Create a `.env` file for environment-specific settings
```bash
echo "BASE_DIR=$(pwd)" > .env
```

## Running the Pipeline

### Unified Pipeline (NEW - Use This!)
The project now provides a single unified entry point that includes automatic data quality fixes:

```bash
python run_unified_etl.py
```

This replaces all legacy runners and fix scripts. See [QUICK_START.md](QUICK_START.md) for detailed usage.

### Quick Examples
```bash
# Full pipeline with automatic fixes
python run_unified_etl.py

# Process specific steps
python run_unified_etl.py --steps geo time dimensions

# Export to Snowflake
python run_unified_etl.py --mode export --snowflake-config snowflake/config.json

# Validation only
python run_unified_etl.py --mode validate
```

### Legacy Entry Points (Being Phased Out)
- `run_etl.py` - Original pipeline
- `run_etl_enhanced.py` - Enhanced pipeline
- `fix_all.py` - Separate fix script (no longer needed)

### Available Steps
- `download`: Download required Census and geographic data
- `geo`: Process geographic boundaries
- `time`: Create time dimension
- `dimensions`: Create dimension tables (includes unknown members)
- `g01`: Process G01 (Population) Census data
- `g17`: Process G17 (Income) Census data
- `g18`: Process G18 (Assistance Needed) Census data
- `g19`: Process G19 (Health Conditions) Census data
- `g20`: Process G20 (Selected Health Conditions) Census data
- `g21`: Process G21 (Health Conditions by Characteristics) Census data
- `g25`: Process G25 (Unpaid Assistance) Census data
- `validate`: Run data quality validation checks

### Additional Options
- `--force-download`: Force re-download of data files even if they exist
- `--skip-validation`: Skip validation steps (not recommended for production)
- `--stop-on-error`: Stop pipeline execution on first error

## Testing

Run all tests with pytest:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=etl_logic
```

Run specific test modules:
```bash
# Test utility functions
pytest tests/test_utils.py

# Test validation logic
pytest tests/test_validation.py

# Test specific table processing
pytest tests/test_g21_processing.py
```

## Configuration

Configuration has been moved to a structured configuration system in the `ahgd_etl/config/` directory:

- **YAML Configuration Files**: Located in `ahgd_etl/config/yaml/`
  - `schemas.yaml`: Defines schemas for all dimension and fact tables
  - `column_mappings.yaml`: Maps source data columns to target dimensions and facts
  - `data_sources.yaml`: Defines URLs and metadata for external data sources

- **Settings Manager**: `ahgd_etl/config/settings.py` provides a unified interface for accessing all configuration

Environment-specific settings should be configured in the `.env` file (see `.env.example` for a template):
```
RAW_DATA_DIR=./data/raw
OUTPUT_DIR=./output
VALIDATE_DATA=True
LOG_LEVEL=INFO
```

For a comprehensive guide to configuration, see [Configuration Guide](documentation/configuration_guide.md).

## Adding New Census Tables

1. Create a new processor in `etl_logic/tables/` (use existing files as templates)
2. Add table-specific column mappings in `etl_logic/config.py`
3. Add a new step function in `run_etl.py`
4. Add the step to the `ETL_STEPS` dictionary in `run_etl.py`
5. Add the step to relevant step groups (e.g., "census" or "all")
6. Create tests for the new processor

## Validation Framework

The pipeline includes a comprehensive validation framework to ensure data quality:

1. **Record Count Validation**: Ensures expected number of records in each table
2. **Null Value Checks**: Validates that required fields don't contain nulls
3. **Referential Integrity**: Verifies foreign key relationships between fact and dimension tables
4. **Surrogate Key Verification**: Confirms surrogate keys are correctly generated and linked
5. **Cross-Table Consistency**: Validates consistency between related tables

Run validation separately with:
```bash
python run_etl.py --steps validate
```

Or verify surrogate keys specifically:
```bash
python verify_surrogate_keys.py
```

## Google Colab Integration

The pipeline can be run in Google Colab using:

1. **colab_runner.ipynb**: Interactive notebook for running ETL steps
2. **ahgd_etl_notebook.py**: Python script equivalent for non-interactive execution

To use in Colab:
1. Upload the project files to your Google Drive
2. Open `colab_runner.ipynb` in Colab
3. Mount your Google Drive
4. Set the correct `BASE_DIR` in the notebook
5. Run the desired ETL steps

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:

1. Project structure and organization
2. Development workflow and standards
3. Code style and patterns
4. Testing requirements
5. Configuration standards
6. Documentation requirements

All contributors should follow these guidelines to ensure consistency and maintainability.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 