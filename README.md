# Australian Healthcare Geographic Database (AHGD) ETL Pipeline

A robust, automated ETL pipeline for processing Australian Bureau of Statistics (ABS) geographic and Census data into a dimensional data warehouse optimized for healthcare analytics.

## Overview

This pipeline transforms ABS ASGS (Australian Statistical Geography Standard) boundary files and Census 2021 data into a star schema data warehouse suitable for healthcare planning, research, and analysis.

## Features

- **Automated Data Processing**: Downloads and processes ABS geographic boundaries and Census data
- **Dimensional Modeling**: Creates a proper star schema with fact and dimension tables
- **Data Quality**: Built-in validation, unknown member handling, and referential integrity
- **Flexible Configuration**: YAML-based configuration for easy customization
- **Storage Efficient**: Outputs to compressed Parquet format
- **Comprehensive Logging**: Detailed logging for monitoring and debugging

## Architecture

### Dimensional Model

**Dimensions:**
- `geo_dimension` - Geographic hierarchy (SA1→SA2→SA3→SA4→STE)
- `dim_time` - Daily granularity with Australian financial years
- `dim_health_condition` - Health conditions from Census
- `dim_demographic` - Age groups and sex categories
- `dim_person_characteristic` - Income, employment, assistance status

**Facts:** (To be implemented)
- `fact_population` - Population counts (G01)
- `fact_income` - Income statistics (G17)
- `fact_assistance_needed` - Assistance needs (G18)
- `fact_health_conditions` - Health prevalence (G19/G20)
- `fact_unpaid_assistance` - Unpaid care provision (G25)

## Quick Start

### Prerequisites

- Python 3.9+
- 10GB free disk space (for full dataset)
- ABS data access (for real data)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ahgd-etl.git
cd ahgd-etl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Run full pipeline
python run_unified_etl.py --steps all

# Run specific steps
python run_unified_etl.py --steps geo,time,dimensions
python run_unified_etl.py --steps facts,validate

# Run with mock data for testing
python create_mock_data.py
python run_unified_etl.py --steps all

# Download helper
python download_abs_data.py  # Interactive guide for ABS downloads
```

### Using GitHub Codespaces (Recommended)

For users with limited local storage:

1. Open repository in GitHub Codespaces
2. Run `python create_mock_data.py` or upload real ABS data
3. Run `python run_unified_etl.py --steps all`
4. Download results or commit to repository

See [CODESPACES_GUIDE.md](CODESPACES_GUIDE.md) for detailed instructions.

## Configuration

All configuration is managed through YAML files in `ahgd_etl/config/yaml/`:

- `data_sources.yaml` - URLs for ABS data sources
- `schemas.yaml` - Table structure definitions
- `column_mappings.yaml` - Source to target mappings

## Data Sources

### Geographic Data
- ASGS 2021 Digital Boundary Files from ABS
- Levels: SA1, SA2, SA3, SA4, STE, POA

### Census Data
- 2021 Census General Community Profile (GCP)
- Tables: G01, G17, G18, G19, G20, G21, G25

**Note**: ABS data requires authentication. Visit the [ABS website](https://www.abs.gov.au) to register and download data files.

## Project Structure

```
ahgd_etl/
├── ahgd_etl/
│   ├── config/          # Configuration management
│   ├── extractors/      # Data download modules
│   ├── transformers/    # Data transformation logic
│   ├── models/          # Dimension builders
│   ├── loaders/         # Data output modules
│   └── validators/      # Data quality checks
├── data/
│   └── raw/            # Downloaded source data
├── output/             # Generated Parquet files
├── logs/               # Processing logs
└── tests/              # Unit and integration tests
```

## Development Status

### Completed ✅
- Core configuration system
- Data acquisition framework
- Geographic dimension processor
- Time dimension generator
- Core dimension builders (health, demographic, characteristics)

### In Progress 🚧
- Census data transformers for fact tables
- Data validation framework
- Unified CLI and orchestration
- Comprehensive test coverage

### Planned 📋
- Snowflake integration
- Incremental loading
- SCD Type 2 for geographic changes
- Additional Census tables

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ahgd_etl

# Run specific test module
pytest tests/test_dimensions.py -v
```

## Storage Requirements

- **Full Dataset**: ~6-7GB
  - Geographic boundaries: ~500MB
  - Census data: ~3GB extracted
  - Processing space: ~2GB
  
- **Test Dataset**: <200MB
  - Mock data or subset

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

[License details to be added]

## Acknowledgments

Data sourced from the Australian Bureau of Statistics under [appropriate license].