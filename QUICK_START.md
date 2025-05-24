# AHGD ETL Quick Start Guide

## Prerequisites

1. Python 3.8+ installed
2. Virtual environment activated
3. Dependencies installed (`pip install -r requirements.txt`)

## Running the ETL Pipeline

### 1. Full Pipeline (Recommended)

Run the complete ETL pipeline with automatic data quality fixes:

```bash
python run_unified_etl.py
```

This will:
- Download Census and geographic data
- Process geographic boundaries
- Generate time and other dimensions (with unknown members)
- Process all Census tables (G01-G25)
- Apply inline data quality fixes
- Validate the output

### 2. Specific Steps Only

Run only the steps you need:

```bash
# Download data only
python run_unified_etl.py --steps download

# Process dimensions only
python run_unified_etl.py --steps geo time dimensions

# Process specific Census tables
python run_unified_etl.py --steps g01 g17 g18

# Run validation only
python run_unified_etl.py --mode validate
```

### 3. Export to Snowflake

First, create a Snowflake configuration file:

```bash
cp snowflake/config.example.json snowflake/config.json
# Edit config.json with your Snowflake credentials
```

Then export:

```bash
python run_unified_etl.py --mode export --snowflake-config snowflake/config.json
```

## Common Scenarios

### First Time Setup

```bash
# 1. Run full pipeline to generate all data
python run_unified_etl.py

# 2. Validate the output
python run_unified_etl.py --mode validate

# 3. Check the output directory
ls -la output/
```

### Updating Specific Tables

```bash
# Re-process just health condition tables
python run_unified_etl.py --steps g19 g20 g21

# Re-run with forced download
python run_unified_etl.py --steps g01 --force-download
```

### Debugging Issues

```bash
# Run with debug logging
python run_unified_etl.py --log-level DEBUG

# Stop on first error
python run_unified_etl.py --stop-on-error

# Check logs
tail -f output/logs/etl_*.log
```

## Output Files

All processed data is saved to the `output/` directory:

### Dimension Tables
- `geo_dimension.parquet` - Geographic boundaries
- `dim_time.parquet` - Time dimension
- `dim_health_condition.parquet` - Health conditions
- `dim_demographic.parquet` - Age groups and sex
- `dim_person_characteristic.parquet` - Person characteristics

### Fact Tables
- `fact_population.parquet` - Population counts (G01)
- `fact_income.parquet` - Income statistics (G17)
- `fact_assistance_needed.parquet` - Assistance needs (G18)
- `fact_health_conditions_refined.parquet` - Health conditions (G19/G20)
- `fact_unpaid_assistance.parquet` - Unpaid care (G25)

## Troubleshooting

### "Module not found" errors
```bash
# Ensure you're in the project root
cd /path/to/AHGD
# Install in development mode
pip install -e .
```

### Data quality validation failures
- Check logs for specific issues
- The unified pipeline includes automatic fixes for:
  - Missing dimension references (uses unknown member)
  - Duplicate keys (aggregates measures)
  - Schema mismatches (coerces types)

### Snowflake connection issues
- Verify credentials in config.json
- Ensure your IP is whitelisted
- Check warehouse is running

## Next Steps

1. Review validation results in logs
2. Set up Snowflake tables using `snowflake/create_all_tables.sql`
3. Schedule regular ETL runs
4. Monitor data quality metrics

## Getting Help

- Check `CLAUDE.md` for detailed architecture information
- Review `IMPROVEMENTS_SUMMARY.md` for recent changes
- See `documentation/` for detailed guides