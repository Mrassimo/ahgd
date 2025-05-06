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
├── config.py               # Global configuration settings
├── run_etl.py              # Main entry point for running ETL pipeline
├── verify_surrogate_keys.py # Validation for surrogate key relationships
├── data/                   # Data input files (raw Census, shapefiles)
├── output/                 # Generated Parquet files
├── logs/                   # Log files
├── etl_logic/              # Core ETL processing modules
│   ├── config.py           # Detailed configuration settings
│   ├── census.py           # Census data processing functions
│   ├── geography.py        # Geographic data processing
│   ├── dimensions.py       # Dimension table creation
│   ├── time_dimension.py   # Time dimension generation
│   ├── utils.py            # Utility functions
│   ├── validation.py       # Data quality validation
│   └── tables/             # Table-specific processing
│       ├── g01_population.py
│       ├── g17_income.py
│       ├── g18_assistance_needed.py
│       ├── g19_health_conditions.py
│       ├── g20_selected_conditions.py
│       ├── g21_conditions_by_characteristics.py
│       └── g25_unpaid_assistance.py
├── scripts/                # Utility scripts
├── tests/                  # Test suite
│   ├── test_utils.py
│   ├── test_validation.py
│   ├── test_config.py
│   ├── test_dimensions.py
│   ├── test_geography.py
│   ├── test_census.py
│   ├── test_time_dimension.py
│   ├── test_g21_processing.py
│   └── test_data/          # Test data fixtures
└── documentation/          # Project documentation
    ├── planning.md         # Architecture and design decisions
    └── datadicttext.md     # Data dictionary and schema details
```

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

### Full Pipeline
```bash
python run_etl.py --steps all
```

### Specific Steps
```bash
# Download data only
python run_etl.py --steps download

# Process geographic data only
python run_etl.py --steps geo

# Process specific Census tables
python run_etl.py --steps g01 g17 g18

# Run only validation
python run_etl.py --steps validate
```

### Available Steps
- `download`: Download required Census and geographic data
- `geo`: Process geographic boundaries
- `time`: Create time dimension
- `g01`: Process G01 (Population) Census data
- `g17`: Process G17 (Income) Census data
- `g18`: Process G18 (Assistance Needed) Census data
- `g19`: Process G19 (Health Conditions) Census data
- `g20`: Process G20 (Selected Health Conditions) Census data
- `g21`: Process G21 (Health Conditions by Characteristics) Census data
- `g25`: Process G25 (Unpaid Assistance) Census data
- `dimensions`: Create dimension tables
- `validate`: Run data validation checks

### Additional Options
- `--force-download`: Force re-download of data files even if they exist
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

The main configuration is in `etl_logic/config.py` which includes:

- Paths for data, outputs, and logs
- Data source URLs
- Geographic levels to process
- Census table patterns
- Column mappings for Census tables
- Data schemas for dimension and fact tables

Environment-specific settings can be configured in the `.env` file:
```
BASE_DIR=/path/to/project
DATA_DIR=/custom/data/location  # Optional
OUTPUT_DIR=/custom/output/location  # Optional
```

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

1. Ensure all tests pass before submitting pull requests
2. Follow the established coding patterns and organisation
3. Add appropriate tests for new features
4. Update configuration and documentation as needed

## License

This project is licensed under the MIT License - see the LICENSE file for details. 