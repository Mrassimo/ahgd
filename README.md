# AHGD ETL Pipeline

This package contains the Extract, Transform, Load (ETL) pipeline for processing Australian Bureau of Statistics (ABS) data, specifically:

1. Geographic boundaries from ASGS (Australian Statistical Geography Standard)
2. Population data from Census G01 (Selected Person Characteristics)
3. Assistance need data from Census G17 (Core Activity Need for Assistance)
4. Unpaid care data from Census G18 (Unpaid Domestic Work, Unpaid Care & Voluntary Work)
5. Health condition data from Census G19 (Long-Term Health Conditions)
6. Detailed health condition data from Census G20 (Selected Long-Term Health Conditions by Age by Sex)
7. Health condition data by characteristics from Census G21 (Type of Long-Term Health Condition by Selected Person Characteristics) - placeholder implementation

## Project Structure

```
ahgd-etl-pipeline/
├── etl_logic/
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration settings
│   ├── utils.py                 # Utility functions
│   ├── geography.py             # Geographic data processing
│   ├── census.py                # Census data processing
│   ├── dimensions.py            # Dimension table creation
│   └── time_dimension.py        # Time dimension generation
├── run_etl.py                   # CLI orchestrator for local execution
├── colab_runner.ipynb           # Colab execution notebook
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── tests/                       # Test suite
└── README.md                    # This file
```

## Features

- Downloads and processes ABS ASGS geographic boundary files (SA1-SA4, STATE)
- Downloads and processes ABS Census G01 population data
- Downloads and processes ABS Census G17 assistance need data
- Downloads and processes ABS Census G18 unpaid care data
- Downloads and processes ABS Census G19 health condition data
- Downloads and processes ABS Census G20 detailed health condition data
- Downloads and processes ABS Census G21 health condition data by characteristics
- Generates time dimension with comprehensive date attributes
- Creates health condition and demographic dimension tables
- Implements proper dimensional model with surrogate keys
- Validates and links Census data with geographic boundaries
- Outputs standardized Parquet files for downstream analysis
- Command-line interface for local execution
- Comprehensive test suite
- Extensive data quality checks at each processing stage

## Output Files

1. `geo_dimension.parquet`: Geographic boundaries and attributes
   - Contains: geo_sk (surrogate key), geo_code, geo_level, geometry (WKT)
   - Levels: SA1, SA2, SA3, SA4, STATE

2. `dim_time.parquet`: Time dimension with various date attributes
   - Contains: time_sk (surrogate key), full_date, year, quarter, month, month_name, day_of_month, 
     day_of_week, day_name, financial_year, is_weekday, is_census_year, etl_load_ts
   - Spans 2011-2031 by default, covering past and future Census years
   - Includes Australian financial year (July-June format)
   - Flags for Census years (2011, 2016, 2021, etc.)

3. `fact_population.parquet`: Population statistics (G01)
   - Contains: geo_sk (foreign key), time_sk (foreign key), total_persons, total_male, total_female, total_indigenous
   - Joined with geo_dimension on geo_code
   - Linked to time dimension for temporal analysis
   - Represents the G01 Census table data

4. `fact_assistance_need.parquet`: Assistance need statistics (G17)
   - Contains: geo_sk (foreign key), time_sk (foreign key), assistance_needed_count, no_assistance_needed_count, assistance_not_stated_count
   - Joined with geo_dimension on geo_code
   - Linked to time dimension for temporal analysis
   - Represents the G17 Census table data

5. `fact_unpaid_care.parquet`: Unpaid care statistics (G18)
   - Contains: geo_sk (foreign key), time_sk (foreign key), provided_care_count, no_care_provided_count, care_not_stated_count
   - Joined with geo_dimension on geo_code
   - Linked to time dimension for temporal analysis
   - Represents the G18 Census table data

6. `fact_health_condition.parquet`: Long-term health condition statistics (G19)
   - Contains: geo_sk (foreign key), time_sk (foreign key), has_condition_count, no_condition_count, condition_not_stated_count
   - Joined with geo_dimension on geo_code
   - Linked to time dimension for temporal analysis
   - Represents the G19 Census table data

7. `dim_health_condition.parquet`: Health condition dimension
   - Contains: condition_sk (surrogate key), condition, condition_name, condition_category, etl_load_ts
   - Conditions include: Arthritis, Asthma, Diabetes, Cancer, Heart Disease, etc.
   - Categories include: Physical, Mental

8. `dim_demographic.parquet`: Demographic dimension
   - Contains: demographic_sk (surrogate key), age_group, sex, sex_name, age_min, age_max, is_total, etl_load_ts
   - Age groups follow Census breakdowns (e.g., 0-14, 15-24, etc.)
   - Sex values: M (Male), F (Female), P (Total Persons)
   - Calculated age ranges allow numeric analysis

9. `fact_health_conditions_detailed.parquet`: Initial detailed health condition statistics (G20)
   - Contains: geo_sk (foreign key), time_sk (foreign key), condition, age_group, sex, count, etl_load_ts
   - Uses an unpivoted structure for flexible analysis
   - Joined with geo_dimension on geo_code
   - Linked to time dimension for temporal analysis

10. `fact_health_conditions_refined.parquet`: Refined detailed health condition statistics (G20)
   - Contains: geo_sk, time_sk, condition_sk, demographic_sk, count, etl_load_ts
   - Fully implements star schema design with surrogate keys
   - Links to dim_health_condition via condition_sk
   - Links to dim_demographic via demographic_sk
   - Provides optimized structure for analytical queries
   - Enables hierarchical aggregations using dimension attributes

11. `fact_health_condition_characteristics.parquet`: Health condition statistics by person characteristics (G21) - placeholder
   - Contains: geo_sk (foreign key), time_sk (foreign key), placeholder_total_count, etl_load_ts
   - Joined with geo_dimension on geo_code
   - Linked to time dimension for temporal analysis
   - Note: Currently uses placeholder logic and requires full implementation based on actual G21 structure

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt

## Installation

1. Clone this repository
2. Create a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Usage

### Local Execution

The package provides a command-line interface for local execution:

```bash
# Run all steps
python run_etl.py --step all

# Run specific steps
python run_etl.py --step download geo
python run_etl.py --step time
python run_etl.py --step g01
python run_etl.py --step g17
python run_etl.py --step g18
python run_etl.py --step g19
python run_etl.py --step g20
python run_etl.py --step g21

# Create dimension tables
python run_etl.py --step health_dim
python run_etl.py --step demo_dim

# Create refined G20 fact table with proper dimension links
python run_etl.py --step refine_g20

# Force redownload of data files
python run_etl.py --step download --force-download
```

Environment variables can be configured in a `.env` file at the project root:

```
BASE_DIR=/path/to/your/project
```

### Colab Execution

1. Open `colab_runner.ipynb` in Google Colab
2. Mount your Google Drive
3. Set your project base path
4. Run the cells sequentially

## Configuration

Edit `etl_logic/config.py` to modify:
- Data source URLs
- Geographic levels to process
- Census table patterns
- Dimension schemas (GEO_DIMENSION_SCHEMA, TIME_DIMENSION_SCHEMA, HEALTH_CONDITION_SCHEMA, DEMOGRAPHIC_SCHEMA)
- File paths (relative to base directory)

## Testing

The package includes a comprehensive test suite using pytest:

```bash
# Run all tests
python -m pytest

# Run specific test files
python -m pytest tests/test_census.py
python -m pytest tests/test_time_dimension.py
python -m pytest tests/test_dimensions.py

# Run specific test cases
python -m pytest tests/test_census.py::test_process_g19_census_data_success
python -m pytest tests/test_time_dimension.py::test_create_time_dimension
python -m pytest tests/test_dimensions.py::test_refined_g20_processing
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Australian Bureau of Statistics (ABS) for providing the data
- Contributors to the open-source libraries used in this project 

## Recent Improvements

- Enhanced metadata extraction for all census tables using the `extract_all_table_metadata.py` script
- Fixed G17 (Need for Assistance) and G18 (Unpaid Care) processing to handle multiple column naming patterns
- Improved geographic code column detection across different file formats
- Added test scripts and sample data generators for validation
- Standardized error handling and logging across all processors
- Implemented G21 refinement with dimension integration (fact_health_conditions_by_characteristic_refined.parquet)
- Added G25 (Unpaid Assistance) processing to support disability and old age care analysis
- Updated ETL pipeline to include all new processors in the full pipeline run

## Project Structure

```
AHGD3/
├── etl_logic/               # Core ETL processing logic
│   ├── census.py            # Census table processing functions
│   ├── dimensions.py        # Dimension table generation
│   └── utils.py             # Utility functions
├── scripts/                 # Scripts for analysis and testing
│   ├── analysis/            # Analysis scripts
│   │   └── extract_all_table_metadata.py  # Metadata extraction tool
│   ├── test_g17_g18_processing.py  # Test script for G17/G18
│   └── download_sample_data.py     # Sample data generator
├── data/                    # Data directories
│   ├── raw/                 # Raw input data
│   │   ├── Census/          # Census CSV files
│   │   └── Metadata/        # Metadata files
│   └── processed/           # Processed output data
├── memory-bank/             # Project documentation
└── README.md                # Project overview
```

## Usage

### Setting Up Sample Data

To generate sample data for testing:

```bash
python scripts/download_sample_data.py
```

### Testing G17 and G18 Processing

To test the G17 and G18 processing functions:

```bash
python scripts/test_g17_g18_processing.py
```

### Extracting Metadata

To extract metadata from all census tables:

```bash
python scripts/analysis/extract_all_table_metadata.py
```

### Running the ETL Pipeline

To run the full ETL pipeline:

```bash
python run_etl.py --step all
```

To run specific steps:

```bash
python run_etl.py --step g25  # Run only G25 processing
python run_etl.py --step refine_g21  # Run only G21 refinement
```

## Implementation Details

### G17 - Need for Assistance Processing

The G17 processor handles Census data related to need for assistance, with these key features:
- Flexible geographic code column detection (SA1, SA2, SA3, SA4, LGA, STE)
- Support for multiple column naming patterns
- Consistent error handling and logging
- Safe conversion of count columns to integers

### G18 - Unpaid Care Processing

The G18 processor handles Census data related to unpaid care provision, with these key features:
- Flexible geographic code column detection
- Support for multiple column naming patterns
- Consistent error handling and logging
- Safe conversion of count columns to integers

### G21 - Health Conditions by Characteristics

The G21 processor handles health condition data by person characteristics:
- Processes data on health conditions by country of birth, income, and other characteristics
- Creates a dimensional model with health condition and person characteristic dimensions
- Supports surrogate key integration for efficient analysis

### G25 - Unpaid Assistance Processing

The G25 processor handles Census data related to unpaid assistance provided to people with disabilities or due to old age:
- Extracts data on assistance provision status by geographic area
- Handles multiple column naming patterns
- Maintains consistency with other processors through standardized patterns
- Integrates with the dimensional model

## Next Steps

1. Complete end-to-end pipeline testing
2. Enhance data validation framework
3. Implement additional G-tables as needed
4. Create data visualization dashboards

## Development Environment

- Python 3.9+
- Polars for data processing
- SQLite for local testing
- PostgreSQL for production database
- pytest for test framework 

## Known Issues

During end-to-end pipeline execution, the following issues were identified:

1. **Warning: MapWithoutReturnDtypeWarning** - Multiple warnings appear during G21 processing about calling `map_elements` without specifying `return_dtype`. This should be fixed in the code to prevent unpredictable results.

2. **G25 Processing Failure** - The G25 processing step fails with the error: `process_census_table() got an unexpected keyword argument 'table_name'`. The function signature for `process_census_table()` needs to be updated to handle the 'table_name' parameter or the calling code needs to be modified to match the expected parameters.

3. **Geographic Code Mismatches** - During G20 and G21 processing, warnings indicate some rows (0.04%-0.05%) were lost during join to geo_dimension because geo_codes were not found.

4. **Duplicate Geographic Codes** - Warnings show duplicate geo_code values in both G20 and G21 data (62922 and 62941 respectively). This may need investigation to ensure data integrity.

These issues should be addressed to ensure the ETL pipeline runs successfully from end to end. 