# Active Context: AHGD ETL Refactoring

*This file tracks the most recent work focus, changes, and active decisions regarding the Australian Healthcare Geographic Database (AHGD) ETL pipeline refactoring.*

## Current Focus

We are currently implementing and improving the ETL pipeline for ABS Census 2021 data tables (G-tables). 

Recent activity:
1. Added geographic centroids with proper CRS projections to all boundary levels
2. Successfully processed and validated all major Census tables:
   - G18: 1.4M rows of assistance needed data
   - G19: 7.4M rows of health condition data
   - G21: 11.9M rows of health conditions by characteristics
   - G25: 1.5M rows of unpaid assistance data (99.97% match rate)
3. Fixed G18 and G19 processing with improved column detection and unpivoting logic
4. Enhanced filter_special_geo_codes utility to handle special geographic codes correctly
5. Successfully processed G20 and dimension tables
6. Fixed G17 and G18 processing by improving column detection and handling multiple column naming patterns
7. Enhanced metadata extraction for census tables
8. Tested G17 and G18 processing with sample data
9. Implemented G21 refinement with dimension integration
10. Created backup of older data files for archival purposes
11. Identified need for comprehensive data validation framework
12. Migrated project to a new directory structure with clear separation of data and code

## Current Design Decisions

- We're using Polars for processing the large CSV files for better performance
- Standardizing on a geographic code detection mechanism for all G-tables
- Implementing standard error handling with detailed logging
- Using safe conversion functions for numeric data to handle inconsistencies
- Extracting detailed metadata to better understand table structures
- Following a consistent dimension-based model with surrogate keys for efficient joins
- Implemented a clean project directory structure separating data files from code

## Focus Areas for Current Sprint

- ✅ Enhanced metadata extraction for all tables (completed)
- ✅ Fixed G17 and G18 processing by improving column detection (completed)
- ✅ Implemented G21 refinement with dimension integration (completed)
- ✅ Implemented G25 processing for unpaid assistance data (completed)
- ✅ Archived older versions of data files (completed)
- ✅ Migrated project to a clean directory structure (completed)
- ✅ Fixed G18 and G19 processing with robust column pattern recognition (completed)
- ✅ Enhanced filter_special_geo_codes to avoid backreference regex patterns (completed)
- 🔄 Develop comprehensive data validation framework
- 🔄 Implement cross-table validation checks against data dictionary
- 🔄 Complete end-to-end pipeline testing

## Recently Modified Files

- `etl_logic/census.py` - Fixed G18 and G19 processors for column detection and unpivoting
- `etl_logic/utils.py` - Enhanced filter_special_geo_codes function to avoid backreference patterns
- Project structure migration to new directory
- `scripts/analysis/extract_all_table_metadata.py`
- `etl_logic/census.py` - Updated G17 and G18 processing functions, added G25 processing
- `scripts/test_g17_g18_processing.py` - Created test script for G17/G18
- `scripts/download_sample_data.py` - Created script to generate sample data
- `run_etl.py` - Added G21 refinement and G25 processing functions
- `tests/test_census.py` - Added tests for G25 processing
- `etl_logic/config.py` - Added G25 pattern

## Current Challenges

- Some G-tables have inconsistent column naming patterns that require flexible processing
- Handling different aggregation levels (SA1, SA2, SA3, SA4, LGA, STE) in a unified way
- Ensuring appropriate detection of geographic code columns across different file formats
- Managing the increasing complexity of the dimensional model as more tables are added

## Next Actions

1. Design and implement data validation framework:
   - Create validation rules based on data dictionary specifications
   - Implement cross-table consistency checks
   - Develop automated validation reporting
   - Set up threshold alerts for data quality metrics
2. Add additional tests for end-to-end pipeline verification
3. Create data visualization dashboards to explore the processed data
4. Optimize performance for large-scale data processing

## Most Recent Changes

- **Geographic Centroids and Census Processing Completion**: (2025-04-07)
  - Added centroid calculations (longitude/latitude) to all boundary levels
  - Ensured proper CRS projections for Australian geography
  - Successfully processed all major Census tables:
    - G18: 1.4M rows processed with full schema compliance
    - G19: 7.4M rows of health condition data processed
    - G21: 11.9M rows processed with improved characteristic parsing
    - G25: 1.5M rows processed with 99.97% geographic match rate
  - Validated all outputs against target schemas
  - Confirmed proper time dimension linkage across all tables
  - Verified centroid calculations in geographic dimension

- **G18 and G19 Processing Fixes and Utility Enhancement**: (2025-04-06)
  - Fixed critical issues in G18 Census table processing with robust column name detection
  - Enhanced G19 processing with improved unpivoting logic for detailed health conditions
  - Updated `filter_special_geo_codes` utility function to avoid regex backreferences
  - Replaced problematic backreference patterns with explicit digit patterns (1-9, 0)
  - Added checks for codes longer than 11 characters and specific patterns
  - Improved logging to track filtering of special geographic codes
  - Confirmed successful processing of postcode (POA) data
  - Validated full end-to-end ETL pipeline with these changes
  - Successfully generated all required fact and dimension tables

- **ETL Performance Optimization and Code Improvements**: (2025-04-05)
  - Fixed performance bottleneck in `run_census_g25_processing` by replacing row-by-row filtering with vectorised operations
  - Improved logging configuration to eliminate duplicate timestamps and module names
  - Added `logger.propagate = False` to prevent duplicate log messages
  - Implemented proper temporary file cleanup in `process_census_table` using try-finally blocks
  - Removed unused legacy function `_process_single_census_csv_legacy` from `census.py`
  - Improved code readability by breaking up long lines and adding proper formatting
  - Updated codebase to use Australian English spelling consistently (changed "optimize" to "optimise", "standardize" to "standardise", etc.)
  - Added backward compatibility for renamed functions
  - Enhanced error handling for temporary file cleanup
  - Fixed inconsistent function documentation for better maintainability

- **Project Structure Migration**: (2025-04-04)
  - Migrated the project to a new directory structure with clear separation of concerns
  - Created two main directories: 'data_files' for input/output data and 'app' for code and documentation
  - Ensured all necessary code components and configuration were migrated
  - Preserved project-related documentation for reference
  - Data files directory structure maintained with raw/ and output/ subdirectories
  - Code organized into appropriate subdirectories (etl_logic/, scripts/, tests/, etc.)
  - Memory bank project documentation moved to app/documentation/project/

- **Census Processing Refactoring Completed**: (2025-04-03)
  - Completed refactoring of all process_gXX_file functions (g01, g17, g18, g19, g20, g21, g25)
  - Converted all functions to thin wrappers that call _process_single_census_csv
  - Each function now properly defines table_code, geo_column_options, measure_column_map, and required_target_columns
  - Updated process_census_table to directly call process_file_function instead of _process_single_census_csv_legacy
  - Verified parameter naming consistency across all functions
  - This refactoring improves maintainability, reduces code duplication, and ensures a consistent approach to Census table processing

- **Census Table Processing Refactoring**: (2025-04-02)
  - Created new _process_single_census_csv function to standardize CSV processing
  - Refactored process_g01_file, process_g17_file, process_g18_file, and process_g25_file
  - Each process function now defines geo_column_options, measure_column_map, and required_target_columns
  - Used standardized column mappings from config.CENSUS_COLUMN_MAPPINGS
  - Renamed original function to _process_single_census_csv_legacy for compatibility

- **Staging File Handling for G20 and G21 Tables**: (2025-04-01)
  - Modified process_g20_census_data to use TEMP_DIR for staging files
  - Changed output filename pattern to staging_{table_code}_detailed.parquet
  - Ensured temp files are explicitly deleted after processing
  - Modified process_g21_census_data to write to TEMP_DIR 
  - Updated run_refined_g21_processing to read from the temporary location
  - Added explicit deletion of staging files 
  - Updated validation checks to verify only refined final output files
  - Modified log messages to clarify final output statuses

- **Housekeeping & Readability Improvements from Senior Architect Review**: (2025-04-01)
  - Enhanced docstrings across all major functions in the ETL pipeline
  - Added detailed Google-style docstrings with comprehensive parameter and return value descriptions
  - Standardized logging messages with consistent formatting and appropriate log levels
  - Added table code prefixes to all log messages for better context and filtering
  - Improved error handling with traceback information for critical errors
  - Added more detailed data quality information in log messages

- **Code Structure & Refactoring from Senior Architect Review**: (2025-03-31)
  - Consolidated dimension logic by moving `generate_health_condition_dimension` from `health_condition.py` into `dimensions.py`
  - Created new helper function `_process_single_census_csv` to reduce code duplication
  - Standardized census column mappings by creating a `CENSUS_COLUMN_MAPPINGS` dictionary in `config.py`
  - Fixed G21 `map_elements` warnings by properly specifying return data types
  - Removed the now redundant `health_condition.py` file
  - Updated all imports to maintain functionality

- **Data Model Refinements from Senior Architect Review**: (2025-03-30)
  - Clarified purpose of _DETAILED vs _REFINED fact tables in documentation
  - Removed redundant `is_total` flag from DEMOGRAPHIC_DIMENSION (can derive from `age_group == 'Tot'` or `sex == 'P'`)
  - Standardized health condition categories to use Title Case (Physical, Mental, Respiratory, etc.)
  - Standardized timestamp column name to `etl_processed_at` across all dimension and fact tables
  - Updated all documentation and diagrams to reflect these changes

- **G19 Detailed Processing Fixes**: (2025-03-29)
  - Implemented a new `extract_files_from_zip` utility function to allow selective extraction of files based on regex patterns
  - Fixed file detection glob pattern for G19 files to handle directories with spaces
  - Enhanced geographic code column detection to support all regional code variants (SA1_CODE_2021, LGA_CODE_2021, etc.)
  - Improved column parsing for health conditions and age groups
  - Added consistent data type enforcement before DataFrame concatenation
  - Successfully processed all G19 detailed files from across multiple geographic levels
  - Saved comprehensive health condition fact table with 10.4 million records

- **G20 Refinement**: (2025-03-28)
  - Implemented dimensional model for G20 data
  - Created health condition dimension (dim_health_condition.parquet)
  - Created demographic dimension (dim_demographic.parquet)
  - Refined G20 fact table with surrogate keys

- **Metadata Analysis**: (2025-03-27)
  - Developed scripts to analyze Census metadata
  - Created comprehensive analysis of G17, G18, G19, G20, and G21 file structures
  - Generated documentation of column patterns and naming conventions

- **G21 Analysis**: (2025-03-26)
  - Analyzed structure of G21 (Type of Condition by Characteristics)
  - Documented characteristic types and their values
  - Added to dataStructures.md

## Key Decisions in Progress

1. **Geographic Code Handling**:
   - Different Census files use different naming conventions for geographic codes
   - Need flexible detection of these variations
   - Solution applied in G19 detailed processing: Enhanced geographic code column detection with a comprehensive list of possible column names

2. **Person Characteristic Dimension**:
   - Required for G21 implementation
   - Will model different types of characteristics (CountryOfBirth, LabourForceStatus, Income)
   - Need to extract codes and values from G21 data
   - Will create a dimension table with characteristic_sk, characteristic_type, characteristic_code, characteristic_name, characteristic_category

3. **Data Type Consistency**:
   - Census data often has inconsistent data types (string vs integer)
   - Need to ensure consistent types before concatenation
   - Solution applied in G19: Added explicit type casting for key columns

## Problems to Solve

1. **Remaining G17/G18 Processing Issues**:
   - Apply the same fixes developed for G19 detailed processing
   - Enhance column detection and mapping
   - Ensure consistent data types

2. **G21 Implementation**:
   - Create person characteristic dimension
   - Implement extraction and transformation of G21 data
   - Integrate with existing dimension tables

3. **Performance Optimization**:
   - Processing large files (especially SA1 level) requires significant memory
   - Need chunked processing or parallelization

## Architecture Decisions

### ETL Pipeline Structure

The ETL pipeline follows a modular structure with clear separation of concerns:

1. **Extraction**: Download and extract data from ABS sources
   - Uses `etl_logic/utils.py` for download and extraction utilities
   - Enhanced with targeted file extraction capabilities

2. **Transformation**: Clean, standardize, and process data
   - Geographic data processing in `etl_logic/geography.py`
   - Census data processing in `etl_logic/census.py`
   - Dimension table creation in `etl_logic/dimensions.py`
   - Time dimension processing in `etl_logic/time_dim.py`

3. **Loading**: Save processed data as Parquet files
   - Dimension tables: `geo_dimension.parquet`, `time_dimension.parquet`, etc.
   - Fact tables: `fact_population.parquet`, `fact_health_condition.parquet`, etc.

### Dimensional Model

The data warehouse follows a star schema design:

- **Dimension Tables**:
  - geo_dimension.parquet - Contains geographic boundaries and attributes
    - Core columns: geo_sk, geo_code, geo_name, geo_type, geo_category, state_code, state_name, geometry
  - time_dimension.parquet - Contains time dimension with day-level granularity
    - Core columns: time_sk, full_date, year, quarter, month, month_name, day_of_month, day_of_week, day_name, financial_year, is_weekday, is_census_year
  - dim_health_condition.parquet - Contains health condition dimension
    - Core columns: condition_sk, condition, condition_name, condition_category
  - dim_demographic.parquet - Contains demographic dimension
    - Core columns: demographic_sk, age_group, sex, sex_name, age_min, age_max, is_total
  - dim_person_characteristic.parquet - Will contain person characteristic dimension for G21
    - Core columns: characteristic_sk, characteristic_type, characteristic_code, characteristic_name, characteristic_category

- **Fact Tables**:
  - fact_population.parquet - Contains population statistics from G01 Census data
    - Core columns: geo_sk, time_sk, total_persons, total_male, total_female, total_indigenous
    - Joined with geo_dimension on geo_code
  - fact_assistance_need.parquet - Contains assistance need statistics from G17 Census data
    - Core columns: geo_sk, time_sk, assistance_needed_count, no_assistance_needed_count, assistance_not_stated_count
    - Joined with geo_dimension on geo_code
  - fact_unpaid_care.parquet - Contains unpaid care statistics from G18 Census data
    - Core columns: geo_sk, time_sk, provided_care_count, no_care_provided_count, care_not_stated_count
    - Joined with geo_dimension on geo_code
  - fact_health_condition.parquet - Contains long-term health condition statistics from G19 Census data
    - Core columns: geo_sk, time_sk, has_condition_count, no_condition_count, condition_not_stated_count
    - Joined with geo_dimension on geo_code
  - fact_health_conditions_detailed.parquet - Contains detailed health condition statistics from G19 Census data
    - Core columns: geo_sk, time_sk, condition, age_group, sex, count
    - Joined with geo_dimension on geo_code
  - fact_health_conditions_detailed.parquet - Contains detailed health condition statistics from G20 Census data
    - Core columns: geo_sk, time_sk, condition, age_group, sex, count
    - Joined with geo_dimension on geo_code
  - fact_health_conditions_refined.parquet - Contains refined detailed health condition statistics from G20 Census data
    - Core columns: geo_sk, time_sk, condition_sk, demographic_sk, count
    - Joined with dimension tables using surrogate keys
  - fact_health_conditions_by_characteristic.parquet - Will contain health condition statistics by person characteristics from G21 Census data
    - Core columns: geo_sk, time_sk, condition_sk, characteristic_sk, count
    - Will join with dimension tables using surrogate keys

## Future Focus Areas

Priority areas for development will be:

1. **G17/G18 Processing**
   - Apply G19 detailed processing fixes to G17 and G18
   - Ensure correct handling of column variations
   - Implement consistent data types

2. **G21 Implementation**
   - Create person characteristic dimension
   - Implement extraction and transformation of G21 data
   - Integrate with existing dimension tables

3. **Data Quality Framework**
   - Enhance existing data quality validations
   - Create data quality reporting mechanisms
   - Develop exception handling for data anomalies

4. **Performance Optimization**
   - Enhance processing efficiency for large files
   - Implement parallel processing where appropriate
   - Optimize memory usage for geographic processing

## Last Updated

Date: 2025-03-29