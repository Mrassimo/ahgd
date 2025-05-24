# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Australian Healthcare Geographic Database (AHGD) ETL Pipeline transforms Australian Bureau of Statistics (ABS) Census and Geographic data into a dimensional data warehouse for healthcare analytics. The system has been refactored to use a unified pipeline with integrated data quality fixes.

## Key Commands

### Running the ETL Pipeline

```bash
# NEW UNIFIED PIPELINE (use this!)
python run_unified_etl.py              # Run full pipeline with inline fixes
python run_unified_etl.py --steps geo time dimensions  # Run specific steps
python run_unified_etl.py --mode validate  # Validation only
python run_unified_etl.py --mode export --snowflake-config snowflake/config.json

# Legacy pipelines (being phased out)
python run_etl_enhanced.py --steps all
python run_etl.py --steps all --stop-on-error
python fix_all.py --output-dir output --steps all  # No longer needed with unified pipeline

# Verify surrogate keys
python verify_surrogate_keys.py

# Or use the module directly
python -m ahgd_etl.cli.main --mode full
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=etl_logic --cov=ahgd_etl

# Run specific test files
pytest tests/test_validation.py
pytest tests/etl_logic/tables/test_g21_conditions_by_characteristics.py

# Run single test
pytest tests/test_utils.py::test_specific_function -v
```

### Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "BASE_DIR=$(pwd)" > .env
```

## Architecture Overview

### Package Structure Issue
The codebase has **two parallel package structures** that need consolidation:
- `etl_logic/` - Legacy package (being phased out)
- `ahgd_etl/` - New modular package (target structure)

Multiple ETL entry points exist (15+), causing confusion. Focus on:
- `run_etl_enhanced.py` - Primary entry point using new structure
- `run_etl.py` - Legacy entry point still required for some operations

### Data Flow Architecture

1. **Extract**: Download Census/Geographic data from ABS
2. **Transform**: Process through dimension and fact transformers
3. **Load**: Save as Parquet with schema enforcement
4. **Fix**: Post-processing fixes for data quality issues

### Configuration System

All configuration centralized in `ahgd_etl/config/yaml/`:
- `schemas.yaml` - Table structure definitions using Polars types
- `column_mappings.yaml` - Source-to-target field mappings
- `data_sources.yaml` - External data URLs

Access via: `from ahgd_etl.config import settings`

### Key Data Quality Issues

1. **Missing Referential Integrity**: Fact tables reference non-existent dimension keys
2. **Duplicate Keys**: Fact tables have grain violations
3. **Missing Unknown Members**: Dimensions lack "unknown" records for null handling
4. **Schema Mismatches**: Columns don't match expected Polars data types
5. **Hardcoded Census Date**: Fixed to 2021-08-10 in multiple places

### Fix Process Architecture

The `fix_all.py` orchestrates three fix modules:
1. `schema_fix.py` - Enforces correct columns and data types
2. `dimension_fix.py` - Adds unknown members, fixes surrogate keys
3. `grain_fix.py` - Resolves duplicate key issues in facts

### Dimensional Model

**Dimensions**:
- `geo_dimension` - Geographic hierarchy (SA1→SA2→SA3→SA4)
- `dim_time` - Time dimension with Census flags
- `dim_health_condition` - Health conditions
- `dim_demographic` - Age groups and sex
- `dim_person_characteristic` - Income, employment, etc.

**Facts**:
- `fact_population` - Population counts (G01)
- `fact_income` - Income statistics (G17)
- `fact_assistance_needed` - Assistance needs (G18)
- `fact_health_conditions` - Health prevalence (G19)
- `fact_health_conditions_refined` - With proper dimension keys
- `fact_unpaid_assistance` - Unpaid care (G25)

### Critical Files to Understand

1. **Entry Points**: `run_etl.py`, `run_etl_enhanced.py`
2. **Configuration**: `ahgd_etl/config/settings.py`
3. **Fix Logic**: `ahgd_etl/core/temp_fix/*.py`
4. **Transformers**: `ahgd_etl/transformers/census/*.py`
5. **Models**: `ahgd_etl/models/dimensions.py`

### Known Issues Requiring Attention

1. **Package Migration**: Incomplete migration from `etl_logic` to `ahgd_etl`
2. **Multiple Runners**: Too many ETL entry points cause confusion
3. **Temporary Fixes**: Fix logic should be integrated into main pipeline
4. **Time Dimension Lookup**: Complex logic to find Census date
5. **Geographic Code Variations**: Multiple column names for same geo codes
6. **Test Coverage**: Limited tests for complex transformations

### Recent Improvements (2025)

Major refactoring has addressed the key issues:

1. **Unified CLI Entry Point** (`run_unified_etl.py`)
   - Single entry point replacing 15+ legacy runners
   - Integrated data quality fixes (no separate fix step needed)
   - Support for Snowflake export

2. **Enhanced Data Quality**
   - Automatic unknown member generation in dimensions
   - Inline deduplication for fact tables
   - Referential integrity fixes during processing
   - Schema enforcement with type coercion

3. **Snowflake Integration**
   - Native loader with proper type mapping
   - Optimized DDL with clustering keys
   - SCD Type 2 support for geography dimension
   - Control tables for monitoring

4. **Key New Components**
   - `ahgd_etl/cli/main.py` - Unified CLI interface
   - `ahgd_etl/core/orchestrator.py` - Pipeline orchestration
   - `ahgd_etl/core/fix_manager.py` - Integrated data fixes
   - `ahgd_etl/loaders/snowflake.py` - Snowflake loader
   - `snowflake/create_all_tables.sql` - Production DDL

### Development Priorities

When working on this codebase:
1. **USE** `run_unified_etl.py` for all ETL operations
2. **ALWAYS** run validation after any data processing changes
3. **AVOID** creating new fix scripts - integrate fixes into transformers
4. **USE** the new `ahgd_etl` package structure for new code
5. **COMPLETE** migration from `etl_logic/` to `ahgd_etl/`