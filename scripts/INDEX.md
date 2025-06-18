# Scripts Index

This directory contains all utility scripts organised by function and purpose.

## Quick Navigation

- [**Data Processing**](data_processing/) - Data extraction, processing, and loading
- [**Analysis**](analysis/) - Data analysis and statistical processing
- [**Dashboard**](dashboard/) - Dashboard and visualisation scripts
- [**Utilities**](utils/) - General utility and maintenance scripts

## Script Categories

### Data Processing (`data_processing/`)
Scripts for data acquisition, processing, and database operations:

- `aihw_data_extraction.py` - Extract data from AIHW sources
- `download_data.py` - Download raw data from various sources
- `process_data.py` - Process and clean raw data
- `simple_aihw_extraction.py` - Simplified AIHW data extraction
- `populate_analysis_database.py` - Load processed data into analysis database
- `generate_sample_data.py` - Generate sample data for testing

**Usage Examples:**
```bash
# Download all required data
python scripts/data_processing/download_data.py

# Process downloaded data
python scripts/data_processing/process_data.py

# Populate analysis database
python scripts/data_processing/populate_analysis_database.py
```

### Analysis (`analysis/`)
Scripts for statistical analysis and data exploration:

- `analyse_aihw_data.py` - Comprehensive AIHW data analysis
- `analysis_summary.py` - Generate analysis summaries
- `health_correlation_analysis.py` - Health data correlation analysis
- `examine_data.py` - Data examination and validation

**Usage Examples:**
```bash
# Run comprehensive analysis
python scripts/analysis/analyse_aihw_data.py

# Generate correlation analysis
python scripts/analysis/health_correlation_analysis.py
```

### Dashboard (`dashboard/`)
Scripts for dashboard development and demonstration:

- `demo_dashboard_features.py` - Demonstrate dashboard capabilities
- `demo_geographic_mapping.py` - Geographic mapping demonstrations
- `streamlit_dashboard.py` - Main Streamlit dashboard application

**Usage Examples:**
```bash
# Run dashboard demo
python scripts/dashboard/demo_dashboard_features.py

# Start Streamlit dashboard
streamlit run scripts/dashboard/streamlit_dashboard.py
```

### Utilities (`utils/`)
General utility scripts for maintenance and development:

- `build_docs.py` - Build Sphinx documentation
- `health_check.py` - System health verification
- `geographic_mapping.py` - Geographic mapping utilities
- `test_geographic_mapping.py` - Geographic mapping tests
- `verify_data.py` - Data verification and validation
- `verify_phase3_completion.py` - Phase 3 completion verification

**Usage Examples:**
```bash
# Build documentation
python scripts/utils/build_docs.py

# Verify system health
python scripts/utils/health_check.py

# Verify data integrity
python scripts/utils/verify_data.py
```

## Script Execution Guidelines

### Prerequisites
Ensure you have the project environment activated:
```bash
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows
```

### Common Workflow
1. **Data Setup**: Run data processing scripts in order
2. **Analysis**: Execute analysis scripts for insights
3. **Dashboard**: Use dashboard scripts for visualisation
4. **Utilities**: Run utility scripts for maintenance

### Error Handling
- All scripts include comprehensive error handling
- Check logs in the `logs/` directory for detailed error information
- Use `--help` flag with scripts that support it for usage information

### Development
When adding new scripts:
1. Place in the appropriate category directory
2. Follow existing naming conventions
3. Include comprehensive docstrings
4. Add error handling and logging
5. Update this index file

## Integration with Main Applications

These scripts integrate with the main applications:
- `main.py` - Core application entry point
- `run_dashboard.py` - Dashboard launcher
- `setup_and_run.py` - Complete setup and launch
- `showcase_dashboard.py` - Demonstration dashboard

## Automation and CI/CD

Many scripts are integrated into:
- Pre-commit hooks
- CI/CD pipelines
- Automated testing
- Data processing workflows