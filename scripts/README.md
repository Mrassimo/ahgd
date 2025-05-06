# Scripts Directory

This directory contains various utility scripts for the AHGD ETL pipeline. These scripts are organized into categories based on their purpose.

## Directory Structure

```
scripts/
├── analysis/               # Data analysis and exploration scripts
│   ├── extract_g21_metadata.py           # G21 table structure analysis
│   ├── extract_all_table_metadata.py     # Extract metadata for all tables
│   ├── explore_metadata.py               # General metadata exploration
│   └── ...                               # Other analysis scripts
├── test_utilities/         # Testing and sample data generation
│   └── ...                               # Test scripts and data generators
├── generate_profiling_reports.py         # Generate data profiling reports
├── output_schema_extractor.py            # Extract schemas from Parquet files
├── run_data_documentation.py             # Generate data documentation
└── README.md                             # This file
```

## Script Categories

### Analysis Scripts (in `analysis/` subdirectory)

Scripts that explore and analyze data structure but are not part of the core ETL pipeline:

- **extract_g21_metadata.py**: Analyzes the structure of G21 table data in ABS Census files
- **explore_metadata.py**: General-purpose metadata exploration tool
- **extract_all_table_metadata.py**: Extract metadata from all Census tables
- **extract_g17_g18_g19_metadata.py**: Extract metadata for specific tables
- **extract_future_tables.py**: Analyzes potential future table additions for health demographics. All paths now use configuration from config.py.
- **extract_future_tables_metadata.py**: Extracts metadata for future tables, identifying health-related data. Hardcoded paths removed.
- **extract_g18_g19_info.py**: Inspects G18 and G19 table structures and column details. Uses config for paths.
- **extract_g17_g18_g19_metadata.py**: Extracts metadata for G17, G18, and G19 tables, including column patterns. Hardcoded paths replaced.
- **extract_metadata.py**: General metadata exploration tool for searching terms in Excel files. Now uses config.PATHS.
- **inspect_census.py**: Analyzes Census data files for column structures and geographic data. Hardcoded paths addressed.
- **metadata_g21.py**: Detailed analysis of G21 files, including column patterns and sample data. Paths now from config.
- **extract_future_tables_metadata.py**: Analyzes metadata for future tables, identifies potential health-related data for expansion.
- **extract_g18_g19_info.py**: Inspects G18 and G19 table structures and column details for analysis.
- **extract_g17_g18_g19_metadata.py**: Extracts metadata for G17, G18, and G19 tables, including column patterns and recommendations.
- **extract_metadata.py**: General metadata exploration tool for searching specific terms in Excel files.
- **inspect_census.py**: Analyzes Census data files for column structures and geographic data.
- **metadata_g21.py**: Detailed analysis of G21 files, including column patterns and sample data.

### Documentation Tools

Scripts that generate documentation about the data:

- **output_schema_extractor.py**: Extracts schemas from Parquet files and generates Mermaid ERD diagrams
- **run_data_documentation.py**: Generates comprehensive data documentation
- **generate_profiling_reports.py**: Generates data profiling reports with statistics

### Test Utilities (in `test_utilities/` subdirectory)

Scripts for testing and generating sample data:

- **download_sample_data.py**: Creates sample data for testing
- **test_g17_g18_processing.py**: Tests the G17 and G18 table processing logic

## Usage Notes

1. **Configuration**: All scripts now use paths from the central configuration module (`config.py`). Do not hardcode paths in scripts.

2. **Adding New Scripts**: When adding new scripts:
   - Place in the appropriate subdirectory based on purpose
   - Use configuration from `config.py` for paths
   - Add documentation in this README

3. **Running Analysis Scripts**: Analysis scripts are intended for one-off exploration and are not part of the main ETL pipeline. Run them as needed:
   ```bash
   python -m scripts.analysis.extract_g21_metadata
   ```

4. **Documentation Generation**: Generate documentation with:
   ```bash
   python -m scripts.output_schema_extractor
   python -m scripts.run_data_documentation
   ``` 