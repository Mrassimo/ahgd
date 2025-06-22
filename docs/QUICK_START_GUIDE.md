# AHGD Quick Start Guide

Welcome to the Australian Health and Geographic Data (AHGD) repository! This guide will help you get started with the system in minutes.

## What is AHGD?

AHGD is a production-grade data platform that combines Australian health, environmental, and socio-economic indicators at Statistical Area Level 2 (SA2) boundaries. It provides:

- **Automated ETL Pipeline**: Extract, transform, validate, and load data from official Australian sources
- **Quality Assurance**: Multi-layered validation ensuring data integrity and compliance
- **Geographic Integration**: Spatial validation and SA2-level aggregation
- **Export Flexibility**: Multiple output formats (CSV, Parquet, JSON, SQLite)

## Prerequisites

- **Python 3.8+** (recommended: Python 3.10 or 3.11)
- **Git** for cloning the repository
- **4GB+ RAM** for processing larger datasets
- **Internet connection** for downloading source data

## Quick Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/ahgd.git
cd ahgd
```

### 2. Set Up Environment (Automated)

The easiest way to get started:

```bash
# Run the automated setup script
./setup_env.sh

# Activate the virtual environment
source venv/bin/activate
```

This script will:
- Create a Python virtual environment
- Install all dependencies
- Set up configuration files
- Validate the installation

### 3. Verify Installation

```bash
# Check that CLI tools are working
ahgd-extract --help
ahgd-transform --help
ahgd-validate --help

# Run system health check
python -c "from src.utils.config import get_config; print('Configuration loaded successfully')"
```

## Your First Pipeline Run

Let's extract and process some Australian health data!

### Step 1: Extract Data

Start with a small dataset - SA2-level population data from the Australian Bureau of Statistics:

```bash
# Extract SA2 population data
ahgd-extract --source abs --dataset sa2_population --output data_raw/

# Check what was downloaded
ls -la data_raw/
```

**Expected output:**
```
data_raw/
├── abs_sa2_population_2021.csv
├── metadata.json
└── extraction_log.txt
```

### Step 2: Transform Data

Standardise the raw data into our unified schema:

```bash
# Transform the extracted data
ahgd-transform --input data_raw/ --output data_processed/ --schema sa2_basic

# View the transformed data structure
head data_processed/sa2_population_transformed.csv
```

**What this does:**
- Standardises column names to British English conventions
- Handles missing values according to best practices
- Adds data quality indicators
- Creates audit trail information

### Step 3: Validate Data

Run comprehensive validation checks:

```bash
# Validate the transformed data
ahgd-validate --input data_processed/ --rules schemas/sa2_basic.yaml --output validation_results/

# View validation summary
cat validation_results/summary.json
```

**Validation includes:**
- Schema compliance (data types, required fields)
- Geographic validation (valid SA2 codes)
- Statistical outlier detection
- Business rule compliance

### Step 4: Export Data

Generate final outputs in your preferred format:

```bash
# Export to multiple formats
ahgd-loader --input data_processed/ --output final_data/ --formats csv,parquet,json

# View final outputs
ls -la final_data/
```

## Understanding Your Results

### Data Structure

Your processed data follows the standardised AHGD schema:

```csv
sa2_code,sa2_name,state,population_total,population_male,population_female,data_quality_score,extraction_date
101021007,Braidwood,NSW,2547,1289,1258,0.95,2024-01-15
101021008,Bungendore,NSW,4821,2398,2423,0.98,2024-01-15
```

### Quality Indicators

- **data_quality_score**: Ranges from 0.0-1.0, indicating data completeness and reliability
- **extraction_date**: When the data was sourced from the official provider
- **validation_status**: Pass/Warning/Fail status from quality checks

### Key Files Generated

| File | Description |
|------|-------------|
| `{dataset}_processed.csv` | Main data file with standardised schema |
| `metadata.json` | Dataset information, source details, processing steps |
| `quality_report.html` | Visual quality assessment report |
| `lineage.json` | Complete audit trail of transformations |

## Next Steps

Congratulations! You've successfully run your first AHGD pipeline. Here's what to explore next:

### Explore More Data Sources

```bash
# List available data sources
ahgd-extract --list-sources

# Extract health indicator data
ahgd-extract --source aihw --dataset health_indicators --output data_raw/

# Extract environmental data
ahgd-extract --source bom --dataset climate_zones --output data_raw/
```

### Advanced Configuration

Customise your pipeline behaviour:

```bash
# Copy example configuration
cp configs/examples/development.yaml configs/custom.yaml

# Edit configuration (use your favourite editor)
nano configs/custom.yaml

# Run with custom configuration
export AHGD_CONFIG=configs/custom.yaml
ahgd-transform --input data_raw/ --output data_processed/
```

### Integration with Analysis Tools

The AHGD data works seamlessly with popular analysis tools:

```python
# Load into pandas
import pandas as pd
df = pd.read_csv('final_data/sa2_population_processed.csv')

# Load into R
# data <- read.csv('final_data/sa2_population_processed.csv')

# Load into QGIS (with spatial boundaries)
# Use the generated shapefiles in final_data/spatial/
```

## Common Workflows

### Daily Data Updates

Set up automated daily processing:

```bash
# Create a simple update script
cat > daily_update.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
ahgd-extract --source all --incremental --output data_raw/
ahgd-transform --input data_raw/ --output data_processed/ --incremental
ahgd-validate --input data_processed/ --rules schemas/ --output validation_results/
ahgd-loader --input data_processed/ --output final_data/ --formats csv,parquet
EOF

chmod +x daily_update.sh

# Test the update script
./daily_update.sh
```

### Working with Large Datasets

For processing large datasets efficiently:

```bash
# Enable parallel processing
export AHGD_MAX_WORKERS=4

# Use chunked processing for memory efficiency
ahgd-transform --input data_raw/ --output data_processed/ --chunk-size 10000

# Enable compression for storage efficiency
ahgd-loader --input data_processed/ --output final_data/ --compress --partition-by state
```

## Troubleshooting

### Common Issues

**Problem**: `ModuleNotFoundError: No module named 'src'`
**Solution**: Make sure you're in the project directory and the virtual environment is activated:
```bash
cd /path/to/ahgd
source venv/bin/activate
```

**Problem**: `ConnectionError` during data extraction
**Solution**: Check your internet connection and try with retry:
```bash
ahgd-extract --source abs --dataset sa2_population --output data_raw/ --retry 3
```

**Problem**: Validation warnings about geographic codes
**Solution**: Update your SA2 boundary data:
```bash
ahgd-extract --source abs --dataset sa2_boundaries --output data_raw/boundaries/
```

**Problem**: Out of memory errors
**Solution**: Reduce chunk size and enable memory monitoring:
```bash
export AHGD_CHUNK_SIZE=5000
export AHGD_MEMORY_LIMIT=2048
ahgd-transform --input data_raw/ --output data_processed/ --monitor-memory
```

### Getting Help

- **Documentation**: Browse `docs/` directory for detailed guides
- **Configuration**: Check `configs/examples/` for sample configurations
- **Logs**: Review logs in `logs/` directory for detailed error information
- **API Reference**: See `docs/api/API_DOCUMENTATION.md` for programmatic usage

### Performance Tips

- **Use SSD storage** for better I/O performance
- **Increase memory** if processing large datasets (8GB+ recommended)
- **Enable parallel processing** with `AHGD_MAX_WORKERS` environment variable
- **Use compressed formats** (Parquet) for faster loading

## What's Next?

Now that you've mastered the basics, explore these advanced guides:

- **[Data Analyst Tutorial](DATA_ANALYST_TUTORIAL.md)**: Working with Australian health data for analysis
- **[Researcher Guide](RESEARCHER_GUIDE.md)**: Advanced analysis workflows and statistical methods
- **[Developer Tutorial](DEVELOPER_TUTORIAL.md)**: Extending the pipeline and adding new data sources

Welcome to the AHGD community! 🇦🇺