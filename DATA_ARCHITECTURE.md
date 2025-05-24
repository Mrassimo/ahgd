# AHGD Data Architecture

## Overview

The Australian Healthcare Geographic Database (AHGD) employs a dimensional data warehouse architecture to enable efficient analysis of population health data across geographical regions. This document outlines the key architectural principles, components, and design decisions.

## Core Architectural Principles

### 1. Star Schema Design

The AHGD uses a dimensional model with a star schema design:

- **Dimension Tables**: Contains descriptive attributes for analysis (geography, time, health conditions, etc.)
- **Fact Tables**: Contains metrics and foreign keys to dimensions

This structure allows for intuitive data navigation, optimized query performance, and flexible analytical capabilities.

### 2. Surrogate Key Management

Surrogate keys are essential for data warehouse integrity:

- **Generation Method**: Surrogate keys are generated using MD5 hashes of business keys, ensuring determinism and uniqueness
- **Unknown Members**: Each dimension includes "unknown" members for handling missing or invalid references in fact tables
- **Referential Integrity**: All fact tables maintain strict referential integrity to dimension tables

### 3. Consistent Schema Definition

Schemas are centrally defined and enforced:

- **YAML Configuration**: All table schemas are defined in YAML configuration files
- **Type Checking**: Column data types are explicitly defined and enforced during ETL
- **Validation Layer**: Automated schema validation ensures consistency across all tables

### 4. Fact Table Grain Management

Each fact table adheres to a specific grain (level of detail):

- **Explicit Grain Definition**: The grain of each fact table is clearly documented
- **Duplicate Detection**: Automated validation prevents duplicate key combinations
- **Aggregation Rules**: Predefined rules for handling duplicate values through proper aggregation

### 5. ETL Process Order

Processing follows a clear, dependency-aware sequence:

1. **Dimension Tables**: Processed first to establish lookup values
2. **Basic Fact Tables**: Tables with direct source-to-target mappings
3. **Refined Fact Tables**: Tables requiring complex transformations or combinations

## Data Model

### Dimension Tables

| Table | Description | Key Business Attributes |
|-------|-------------|-------------------------|
| **geo_dimension** | Geographic locations | geo_level, geo_name, state_code, latitude, longitude |
| **dim_time** | Time hierarchy | year, quarter, month, day, financial_year |
| **dim_health_condition** | Health conditions | condition_code, condition_name, category |
| **dim_demographic** | Demographic groups | age_group, sex |
| **dim_person_characteristic** | Person characteristics | characteristic_type, characteristic_value |

### Fact Tables

| Table | Description | Grain | Measures |
|-------|-------------|-------|----------|
| **fact_population** | Population counts | geo_sk + time_sk | total_population, male_population, female_population |
| **fact_income** | Income statistics | geo_sk + time_sk + demographic_sk + characteristic_sk | median_income, mean_income, count_persons |
| **fact_assistance_needed** | Assistance need | geo_sk + time_sk + demographic_sk | count_persons |
| **fact_health_conditions** | Health condition prevalence | geo_sk + time_sk + condition_sk + demographic_sk | count_persons |
| **fact_health_conditions_refined** | Refined health conditions | geo_sk + time_sk + condition_sk + demographic_sk + characteristic_sk | count_persons |
| **fact_no_assistance** | Unpaid assistance statistics | geo_sk + time_sk + demographic_sk | count_persons |

## Implementation Components

### Configuration Management

The configuration is centralized in YAML files:

- **schemas.yaml**: Defines table structures and data types
- **column_mappings.yaml**: Maps source columns to target columns
- **data_sources.yaml**: Contains URLs for data sources

These files are loaded and managed by the `ahgd_etl.config.settings` module.

### Dimension Handling

Dimensions are managed through:

- **Models**: Strong typing with `ahgd_etl.models.dimensions` classes
- **Handler**: The `DimensionHandler` in `ahgd_etl.core.temp_fix.dimension_fix` ensures consistent dimension management
- **Unknown Members**: Each dimension has a special "unknown" member for handling missing references

### Schema Validation

Schema validation is implemented through:

- **SchemaValidator**: The `ahgd_etl.core.temp_fix.schema_fix` module enforces schema consistency
- **Type Coercion**: Automatic conversion to correct data types
- **Missing Column Handling**: Addition of missing columns with default values

### Fact Table Grain Management

Fact table grain is managed through:

- **GrainHandler**: The `ahgd_etl.core.temp_fix.grain_fix` module detects and resolves duplicate key issues
- **Aggregation Rules**: Predefined rules for aggregating measures in case of duplicates
- **Validation**: Automated checks ensure fact tables adhere to their defined grain

## Fix Process

The fix process is orchestrated by `fix_all.py` and follows these steps:

1. **Schema Validation**: Ensures all tables have the correct structure and data types
2. **Dimension Handling**: Fixes dimension tables and adds unknown members if needed
3. **Fact Table Grain**: Resolves duplicate key issues in fact tables

## Future Improvements

1. **Package Structure**: Complete migration from `etl_logic` to modular `ahgd_etl` package
2. **Metadata Management**: Enhanced tracking of data lineage and transformations
3. **Incremental Processing**: Support for incremental updates to fact tables
4. **Data Quality Rules**: Additional data quality validation beyond schema checks
5. **Performance Optimization**: Tune fact table partitioning for query performance

## Conclusion

The AHGD data architecture provides a robust foundation for analyzing healthcare data across geographic dimensions. By following dimensional modeling best practices and implementing strong data governance, the system ensures data consistency, integrity, and analytical flexibility.