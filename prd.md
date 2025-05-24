Okay, this is a great point to define a clear Product Requirements Document (PRD). Starting with a well-defined PRD will set a strong foundation for the rebuilt project.

Here's a PRD tailored for the AGHD ETL Pipeline, incorporating the features and objectives we've distilled:

// AGHD_ETL_Pipeline_PRD.md
# Product Requirements Document: Australian Healthcare Geographic Database (AGHD) ETL Pipeline

## 1. Introduction

**1.1. Project Name:** Australian Healthcare Geographic Database (AGHD) ETL Pipeline

**1.2. Purpose:**
To design, develop, and deploy a robust, automated, and maintainable Extract, Transform, Load (ETL) pipeline. This pipeline will process publicly available Australian Bureau of Statistics (ABS) geographic (ASGS) and Census (GCP) data into a queryable dimensional data warehouse.

**1.3. Vision:**
To empower healthcare planners, public health researchers, data analysts, and policymakers with high-quality, integrated, and easily accessible Australian demographic, socioeconomic, and health-related data at various geographic granularities. This will enable evidence-based decision-making, targeted interventions, and insightful research.

**1.4. Goals:**
*   **Develop a Unified ETL Pipeline:** Create a single, coherent, and automated system for data processing.
*   **Ensure Data Quality & Integrity:** Implement comprehensive validation and cleansing processes.
*   **Produce a Dimensional Model:** Build a star schema data warehouse optimized for analytical querying.
*   **Facilitate Data Access:** Generate outputs in standard, usable formats (Parquet, optionally CSVs, Snowflake).
*   **Maintainable & Extensible:** Design a modular system that is well-documented and easy to update or extend.

## 2. Target Users

*   **Public Health Researchers & Epidemiologists:** For studying disease prevalence, health disparities, and service utilization.
*   **Healthcare Planners & Policymakers (Government & NGOs):** For resource allocation, service planning, and policy development.
*   **Data Analysts & Scientists (Health Sector, Academia):** For custom analytics, modeling, and insights generation.
*   **Private Health Insurers:** For market analysis, risk assessment, and product development.
*   **General Data Consumers (via Snowflake Marketplace or other distribution):** For broader access to curated ABS data.

## 3. Functional Requirements

### FR1: Data Acquisition
*   **FR1.1 Download Geographic Data:** The system shall download specified ASGS digital boundary files (Shapefiles for SA1, SA2, SA3, SA4, STE, POA) from official ABS URLs.
*   **FR1.2 Download Census Data:** The system shall download specified ABS Census 2021 General Community Profile (GCP) DataPacks (ZIP archives containing CSVs for G01, G17, G18, G19, G20, G21, G25) from official ABS URLs.
*   **FR1.3 File Extraction:** The system shall extract relevant files (Shapefiles, CSVs) from downloaded ZIP archives.
*   **FR1.4 Download Configuration:** Download behavior (e.g., force re-download) shall be configurable.
*   **FR1.5 URL Management:** Data source URLs shall be externally configurable.

### FR2: Geographic Data Processing (Dimension: `geo_dimension`)
*   **FR2.1 Shapefile Processing:** Process ASGS Shapefiles for all configured geographic levels (SA1, SA2, SA3, SA4, STE, POA).
*   **FR2.2 Code & Name Standardization:** Clean and standardize geographic codes (e.g., SA1_CODE_2021) and names.
*   **FR2.3 Geometry Validation:** Validate and repair invalid geometries (e.g., using `shapely.validation.make_valid`).
*   **FR2.4 Centroid Calculation:** Calculate and store geometric centroids (latitude, longitude) for each geographic area.
*   **FR2.5 Geometry Storage:** Convert processed geometries to Well-Known Text (WKT) format for storage.
*   **FR2.6 Surrogate Key Generation:** Assign a unique, persistent surrogate key (`geo_sk`) to each geographic entity.
*   **FR2.7 "Unknown" Member:** Include a predefined "unknown" member in the geographic dimension to handle missing or unmappable geographic references.
*   **FR2.8 Output:** Produce `geo_dimension.parquet`.

### FR3: Time Dimension Generation (Dimension: `dim_time`)
*   **FR3.1 Daily Granularity:** Generate a time dimension table with one row per day for a configurable date range (e.g., 2010-2030).
*   **FR3.2 Attributes:** Include attributes such as `time_sk` (PK, YYYYMMDD format), `full_date`, `year`, `quarter`, `month`, `month_name`, `day_of_month`, `day_of_week`, `day_name`, `is_weekday`, `financial_year`, and `is_census_year`.
*   **FR3.3 "Unknown" Member:** Include a predefined "unknown" member for data with unspecified dates.
*   **FR3.4 Output:** Produce `dim_time.parquet`.

### FR4: Core Dimension Generation (Other Dimensions)
*   **FR4.1 `dim_health_condition`:**
    *   Create from a combination of predefined condition lists (based on G19/G20/G21 metadata) and dynamically discovered conditions if necessary.
    *   Attributes: `condition_sk` (PK), `condition_code`, `condition_name`, `condition_category`.
    *   Must include an "unknown" member.
    *   Output: `dim_health_condition.parquet`.
*   **FR4.2 `dim_demographic`:**
    *   Define standard age groups (e.g., 5-year bands, 0-14, 85+) and sex categories (Male, Female, Persons).
    *   Attributes: `demographic_sk` (PK), `age_group`, `sex`.
    *   Must include an "unknown" member for age and sex respectively, and for their combination.
    *   Output: `dim_demographic.parquet`.
*   **FR4.3 `dim_person_characteristic`:**
    *   Define categories for characteristics like income bands, employment status, education level, country of birth (derived from G21, G17 etc. metadata).
    *   Attributes: `characteristic_sk` (PK), `characteristic_type` (e.g., "Income", "Employment"), `characteristic_value` (e.g., "$500-$999", "Employed Full-time"), `characteristic_category`.
    *   Must include an "unknown" member.
    *   Output: `dim_person_characteristic.parquet`.
*   **FR4.4 Surrogate Keys:** All dimension surrogate keys must be unique and persistent. An `is_unknown` boolean flag should identify unknown members.
*   **FR4.5 ETL Timestamp:** All dimension tables must include an `etl_processed_at` timestamp.

### FR5: Census Data Transformation & Fact Table Creation
For each specified Census table (G01, G17, G18, G19, G20, G21, G25):
*   **FR5.x.1 File Identification:** Identify and parse relevant CSV files from the downloaded DataPacks based on configurable patterns.
*   **FR5.x.2 Header Parsing:** Parse complex, multi-row, or coded headers typical of ABS Census tables.
*   **FR5.x.3 Data Unpivoting:** Transform wide-format source data into a long, normalized format suitable for fact tables.
*   **FR5.x.4 Value Mapping:** Map raw data values (codes, textual categories) to standardized dimension codes or descriptive values based on external configuration (e.g., `column_mappings.yaml`).
*   **FR5.x.5 Dimension Linking:** Join transformed data with relevant dimension tables (`geo_dimension`, `dim_time`, `dim_demographic`, `dim_health_condition`, `dim_person_characteristic`) using their surrogate keys.
*   **FR5.x.6 Unknown Key Handling:** If a dimension lookup fails (a value in the source data does not match any known dimension member), the corresponding foreign key in the fact table must be set to the "unknown" member's surrogate key from that dimension.
*   **FR5.x.7 Grain & Aggregation:** Ensure data is aggregated to the correct, predefined grain for each fact table. Resolve duplicate records (at the defined grain) by summing measure columns.
*   **FR5.x.8 Measure Calculation:** Calculate or extract quantitative measures (e.g., population counts, person counts for conditions).
*   **FR5.x.9 ETL Timestamp:** All fact tables must include an `etl_processed_at` timestamp.
*   **FR5.x.10 Output:** Produce `fact_*.parquet` for each processed G-table (e.g., `fact_population.parquet`, `fact_health_conditions_refined.parquet`).

### FR6: Data Quality and Validation
*   **FR6.1 Integrated Validation:** Data validation checks shall be an integral part of the ETL pipeline.
*   **FR6.2 Checks:**
    *   **Record Counts:** Verify tables are not empty and, where applicable, match expected row counts or ranges.
    *   **Null Values:** Ensure key columns (primary, foreign) and critical measure columns are not null.
    *   **Range Checks:** Validate numeric measures fall within expected ranges (e.g., counts are non-negative).
    *   **Uniqueness:** Primary keys in dimension tables must be unique. Composite primary keys in fact tables (defined by the grain) must be unique.
    *   **Referential Integrity:** All foreign keys in fact tables must exist as primary keys in their respective dimension tables.
*   **FR6.3 Logging:** All validation results (passes and failures with details) must be clearly logged.
*   **FR6.4 Error Handling:** The pipeline shall offer configurable behavior on validation failure (e.g., stop pipeline, or log error and continue).

### FR7: Configuration Management
*   **FR7.1 External Configuration:** All data source URLs, file path structures (relative to a configurable base directory), column mappings, parsing rules for Census tables, and output table schemas must be externally configurable via YAML files.
*   **FR7.2 Central Access:** A Python module (`ahgd_etl.config.settings`) shall provide a unified interface for accessing all configurations.
*   **FR7.3 Environment Variables:** Support for overriding specific configurations (e.g., base paths, database credentials) via environment variables or a `.env` file.

### FR8: ETL Orchestration & Command-Line Interface (CLI)
*   **FR8.1 Unified CLI:** A single CLI script (`run_unified_etl.py` or similar) shall be the entry point for all ETL operations.
*   **FR8.2 Step Execution:** The CLI must allow execution of the full pipeline or specific, named steps (e.g., `download`, `geo`, `g01`, `validate`).
*   **FR8.3 CLI Options:** The CLI must support common options such as `--force-download`, `--stop-on-error`, `--log-level`.
*   **FR8.4 Orchestrator Module:** A core orchestrator module (`ahgd_etl.core.orchestrator`) shall manage step execution order, dependencies between steps, and error handling logic.
*   **FR8.5 Pipeline Definition:** A module (`ahgd_etl.core.pipeline`) shall define the available ETL steps and their dependencies.

### FR9: Logging and Monitoring
*   **FR9.1 Comprehensive Logging:** Implement logging across all modules (INFO, WARNING, ERROR, DEBUG levels).
*   **FR9.2 Structured Logs:** Logs must include timestamps, module/function names, log levels, and clear, informative messages.
*   **FR9.3 Log Output:** Logs shall be written to both the console and to timestamped files in a designated `logs/` directory.

### FR10: Output Formats & Deliverables
*   **FR10.1 Primary Format:** All dimension and fact tables shall be output as Apache Parquet files.
*   **FR10.2 Schema Enforcement:** Output Parquet files must strictly adhere to their predefined schemas (column names, order, data types).
*   **FR10.3 (Optional) CSV Export:** The system may provide functionality to export final tables to CSV format.
*   **FR10.4 (Optional) Snowflake Deployment:** The system may provide functionality to load the generated Parquet files into a Snowflake data warehouse, including DDL scripts for table creation.

## 4. Data Architecture & Schema

### 4.1. Overall Architecture
*   **Dimensional Model:** The target data warehouse will employ a star schema design.
*   **Conformed Dimensions:** Dimensions like `geo_dimension` and `dim_time` will be conformed, usable across multiple fact tables.
*   **Surrogate Keys:** All dimension tables will use system-generated surrogate keys (e.g., `geo_sk`, `time_sk`) as primary keys. Fact tables will use these surrogate keys as foreign keys. Integer SKs are preferred for performance where applicable (e.g., geo_sk, time_sk). Hashed string SKs can be used for dimensions derived from complex business keys.
*   **"Unknown" Member Handling:** Each dimension table will include a predefined "unknown" member (e.g., `geo_sk = -1` or specific hash, `is_unknown = True`). This record will be used to link fact table rows where the original data has missing or unmappable dimensional attributes, ensuring referential integrity.

### 4.2. Dimension Table Schemas (Polars Data Types)

*   **`geo_dimension`**
    *   `geo_sk`: `pl.Int64` (PK) - Surrogate key
    *   `geo_id`: `pl.Utf8` (NK) - Official ABS geographic code (e.g., SA1_CODE_2021)
    *   `geo_level`: `pl.Categorical` - Level of geography (SA1, SA2, POA, STE, etc.)
    *   `geo_name`: `pl.Utf8` - Official name of the geographic area
    *   `state_code`: `pl.Utf8` - State/Territory code
    *   `state_name`: `pl.Utf8` - State/Territory name
    *   `latitude`: `pl.Float64` - Latitude of centroid
    *   `longitude`: `pl.Float64` - Longitude of centroid
    *   `geom`: `pl.Utf8` - Geometry in WKT format
    *   `parent_geo_sk`: `pl.Int64` (FK to `geo_dimension.geo_sk`) - Link to parent geography
    *   `is_unknown`: `pl.Boolean` - Flag for the unknown member
    *   `etl_processed_at`: `pl.Datetime` - Timestamp of ETL processing

*   **`dim_time`**
    *   `time_sk`: `pl.Int64` (PK, YYYYMMDD format) - Surrogate key
    *   `full_date`: `pl.Date` (NK) - The specific date
    *   `year`: `pl.Int32` - Calendar year
    *   `quarter`: `pl.Int8` - Calendar quarter (1-4)
    *   `month`: `pl.Int8` - Calendar month (1-12)
    *   `month_name`: `pl.Categorical` - Full name of the month
    *   `day_of_month`: `pl.Int8` - Day number within the month (1-31)
    *   `day_of_week`: `pl.Int8` - Day number within the week (e.g., 0=Mon, 6=Sun)
    *   `day_name`: `pl.Categorical` - Full name of the day
    *   `is_weekday`: `pl.Boolean` - True if Monday-Friday
    *   `financial_year`: `pl.Utf8` - Australian Financial Year (e.g., "2021/22")
    *   `is_census_year`: `pl.Boolean` - True if the year is an ABS Census year
    *   `is_unknown`: `pl.Boolean` - Flag for the unknown member
    *   `etl_processed_at`: `pl.Datetime`

*   **`dim_health_condition`**
    *   `condition_sk`: `pl.Utf8` (PK, e.g., MD5 hash) - Surrogate key
    *   `condition_code`: `pl.Utf8` (NK) - Standardized code for the condition (e.g., "ARTHRITIS")
    *   `condition_name`: `pl.Utf8` - Descriptive name of the condition (e.g., "Arthritis")
    *   `condition_category`: `pl.Categorical` - Broader grouping (e.g., "Musculoskeletal", "Mental Health")
    *   `is_unknown`: `pl.Boolean` - Flag for the unknown member
    *   `etl_processed_at`: `pl.Datetime`

*   **`dim_demographic`**
    *   `demographic_sk`: `pl.Utf8` (PK, e.g., MD5 hash) - Surrogate key
    *   `age_group`: `pl.Categorical` (NK Part) - Standardized age band (e.g., "0-4", "15-19", "85+")
    *   `sex`: `pl.Categorical` (NK Part) - Standardized sex category (e.g., "Male", "Female", "Persons")
    *   `is_unknown`: `pl.Boolean` - Flag for the unknown member
    *   `etl_processed_at`: `pl.Datetime`

*   **`dim_person_characteristic`**
    *   `characteristic_sk`: `pl.Utf8` (PK, e.g., MD5 hash) - Surrogate key
    *   `characteristic_type`: `pl.Categorical` (NK Part) - Type of characteristic (e.g., "Income Bracket", "Employment Status", "Education Level", "Country of Birth")
    *   `characteristic_value`: `pl.Utf8` (NK Part) - Specific value of the characteristic (e.g., "$1000-$1249", "Employed Full-Time", "Bachelor Degree", "Australia")
    *   `characteristic_category`: `pl.Categorical` - Higher-level grouping for the characteristic type (e.g., "Economic", "Social")
    *   `is_unknown`: `pl.Boolean` - Flag for the unknown member
    *   `etl_processed_at`: `pl.Datetime`

### 4.3. Fact Table Schemas (Polars Data Types)

*   **`fact_population`** (From G01)
    *   `geo_sk`: `pl.Int64` (FK to `geo_dimension.geo_sk`)
    *   `time_sk`: `pl.Int64` (FK to `dim_time.time_sk`)
    *   `demographic_sk`: `pl.Utf8` (FK to `dim_demographic.demographic_sk`) - For 'Persons' total, or specific sex if broken down.
    *   `total_persons`: `pl.Int64`
    *   `male_persons`: `pl.Int64`
    *   `female_persons`: `pl.Int64`
    *   `indigenous_persons`: `pl.Int64` (optional, if directly from G01)
    *   `etl_processed_at`: `pl.Datetime`

*   **`fact_income`** (From G17)
    *   `geo_sk`: `pl.Int64` (FK)
    *   `time_sk`: `pl.Int64` (FK)
    *   `demographic_sk`: `pl.Utf8` (FK to `dim_demographic.demographic_sk` - representing specific age/sex combination for income)
    *   `characteristic_sk`: `pl.Utf8` (FK to `dim_person_characteristic.characteristic_sk` - where type is "Income Bracket")
    *   `person_count`: `pl.Int64` - Number of persons in that geo/time/demo/income_bracket.
    *   `median_income_weekly`: `pl.Float64` (Optional, if available directly or calculated)
    *   `mean_income_weekly`: `pl.Float64` (Optional, if available directly or calculated)
    *   `etl_processed_at`: `pl.Datetime`

*   **`fact_assistance_needed`** (From G18)
    *   `geo_sk`: `pl.Int64` (FK)
    *   `time_sk`: `pl.Int64` (FK)
    *   `demographic_sk`: `pl.Utf8` (FK to `dim_demographic.demographic_sk`)
    *   `characteristic_sk`: `pl.Utf8` (FK to `dim_person_characteristic.characteristic_sk` - where type is "Assistance Need Status" and value is "Needs Assistance", "No Need", "Not Stated")
    *   `person_count`: `pl.Int64`
    *   `etl_processed_at`: `pl.Datetime`

*   **`fact_health_conditions_refined`** (From G19, G20)
    *   `geo_sk`: `pl.Int64` (FK)
    *   `time_sk`: `pl.Int64` (FK)
    *   `demographic_sk`: `pl.Utf8` (FK to `dim_demographic.demographic_sk`)
    *   `condition_sk`: `pl.Utf8` (FK to `dim_health_condition.condition_sk`)
    *   `characteristic_sk`: `pl.Utf8` (FK to `dim_person_characteristic.characteristic_sk`, optional - e.g., for 'Number of Conditions' from G20)
    *   `person_count`: `pl.Int64`
    *   `etl_processed_at`: `pl.Datetime`

*   **`fact_health_conditions_by_characteristic_refined`** (From G21)
    *   `geo_sk`: `pl.Int64` (FK)
    *   `time_sk`: `pl.Int64` (FK)
    *   `demographic_sk`: `pl.Utf8` (FK to `dim_demographic.demographic_sk` - G21 often doesn't have age/sex breakdown *with* other chars, this might be overall 'Persons')
    *   `condition_sk`: `pl.Utf8` (FK to `dim_health_condition.condition_sk`)
    *   `characteristic_sk`: `pl.Utf8` (FK to `dim_person_characteristic.characteristic_sk` - for Country of Birth, Labour Force, etc.)
    *   `person_count`: `pl.Int64`
    *   `etl_processed_at`: `pl.Datetime`

*   **`fact_unpaid_assistance`** (From G25)
    *   `geo_sk`: `pl.Int64` (FK)
    *   `time_sk`: `pl.Int64` (FK)
    *   `demographic_sk`: `pl.Utf8` (FK to `dim_demographic.demographic_sk`)
    *   `characteristic_sk`: `pl.Utf8` (FK to `dim_person_characteristic.characteristic_sk` - where type is "Unpaid Assistance Provision" and value is "Provided", "Not Provided", "Not Stated")
    *   `person_count`: `pl.Int64`
    *   `etl_processed_at`: `pl.Datetime`

## 5. Technical Architecture & Engineering

*   **5.1. Language & Core Libraries:**
    *   Python 3.9+
    *   Polars: Primary data manipulation library.
    *   GeoPandas & Shapely: For geospatial operations (reading Shapefiles, geometry validation, centroid calculation).
    *   Requests: For HTTP downloads.
    *   PyArrow: Underlying engine for Parquet I/O with Polars.
    *   python-dotenv: For managing environment variables.
*   **5.2. Project Structure:**
    *   A modular Python package named `ahgd_etl`.
    *   Sub-packages for `cli`, `config`, `core`, `loaders`, `models`, `transformers`, `validators`.
    *   Clear separation of concerns between modules.
*   **5.3. Configuration Management:**
    *   YAML files in `ahgd_etl/config/yaml/` for:
        *   `data_sources.yaml`: URLs.
        *   `column_mappings.yaml`: Source-to-target column maps, unpivoting rules, value mappings.
        *   `schemas.yaml`: Polars schemas for all target dimension and fact tables.
    *   `ahgd_etl/config/settings.py` to load and provide access to all configurations.
*   **5.4. Logging & Monitoring:**
    *   Standard Python `logging` module.
    *   Configurable log levels (DEBUG, INFO, WARNING, ERROR).
    *   Output to console and timestamped log files in `logs/`.
*   **5.5. Error Handling:**
    *   Robust error handling in each ETL step.
    *   Clear logging of errors with tracebacks.
    *   Option to stop pipeline on critical errors or continue with warnings.
*   **5.6. Testing Strategy:**
    *   Pytest framework for unit and integration tests.
    *   Aim for >80% code coverage.
    *   Tests for individual transformers, dimension builders, validation rules, and overall pipeline orchestration.
    *   Use of fixtures and mocking for isolating components.

## 6. Non-Functional Requirements

*   **NFR1: Performance:** Full pipeline execution (all specified ASGS levels and Census tables for Australia-wide data) should complete within 4-8 hours on a standard cloud VM (e.g., 4 vCPU, 16GB RAM). Individual steps should be optimized for memory and CPU usage.
*   **NFR2: Reliability:** The pipeline must be idempotent where possible. It should handle transient errors (e.g., network issues during download) with retries. Critical errors should result in a clean failure state with informative logs.
*   **NFR3: Maintainability:** Code must be well-documented (docstrings, comments), adhere to PEP 8, and be organized logically. Configurations must be easy to understand and update.
*   **NFR4: Scalability:** The design should accommodate future additions of new Census tables or other ABS datasets with minimal architectural changes. Polars is chosen for its ability to handle larger-than-memory datasets if needed, though initial processing will aim to be memory-efficient.
*   **NFR5: Usability (Pipeline Operation):** The CLI must be intuitive. Logs must provide sufficient detail for debugging and monitoring.
*   **NFR6: Data Accuracy:** Output data must be an accurate and verifiable representation of the source ABS data, transformed according to the defined business rules. Validation checks are crucial for this.
*   **NFR7: Security:** If handling sensitive (non-public) data in the future, appropriate security measures for data at rest and in transit must be considered (currently out of scope with public ABS data).

## 7. Data Sources (Summary)

*   **Geographic:** ABS ASGS 2021 Digital Boundary Files (Shapefiles for SA1, SA2, SA3, SA4, STE, POA). URLs as per `ahgd_etl/config/yaml/data_sources.yaml`.
*   **Census:** ABS Census 2021 General Community Profile (GCP) DataPacks (G01, G17, G18, G19, G20, G21, G25). URL for `2021_GCP_all_for_AUS_short-header.zip` as per `ahgd_etl/config/yaml/data_sources.yaml`.

## 8. Output & Deliverables

*   **Primary:** A set of dimension and fact tables in Apache Parquet format, stored in the `output/` directory, forming a coherent star schema.
*   **Code:** A Python package (`ahgd_etl`) containing all ETL logic.
*   **CLI:** A unified command-line interface (`run_unified_etl.py`) for pipeline execution.
*   **Configuration:** YAML files for all configurations.
*   **Documentation:**
    *   This PRD.
    *   `README.md` with setup and usage instructions.
    *   Data dictionary describing all output tables and columns.
    *   Architecture overview document.
*   **Tests:** A suite of unit and integration tests.
*   **(Optional) Snowflake Deployment:** DDL scripts (`snowflake/create_all_tables.sql`) and loading scripts/functionality for Snowflake.
*   **(Optional) CSV Exports:** Scripts or CLI options to export final tables to CSV.
*   **GitHub Repository:** Containing all code, tests, configurations, and documentation.

## 9. Success Metrics

*   **SM1:** Full ETL pipeline completes successfully without critical errors for all specified geographic levels and Census tables.
*   **SM2:** All automated data validation checks pass (record counts, nulls, referential integrity, uniqueness, range checks).
*   **SM3:** Output Parquet files are generated and conform to the defined schemas in `schemas.yaml`.
*   **SM4:** The dimensional model can be successfully queried to answer predefined analytical questions (e.g., population density by SA2, health condition prevalence by age group and state).
*   **SM5 (If Snowflake target):** Data is successfully loaded into Snowflake, tables are queryable, and performance meets basic analytical needs.
*   **SM6:** Code coverage by tests exceeds 80%.
*   **SM7:** Pipeline execution time is within the defined NFR.

## 10. Future Considerations / Roadmap

*   **Incremental Loading:** Design for efficient updates with new data releases (e.g., for dimensions that change over time).
*   **SCD Handling:** Implement Slowly Changing Dimension (SCD Type 2 or other appropriate types) for dimensions like geography if historical versions are required.
*   **Additional Datasets:** Plan for integration of other ABS datasets (e.g., SEIFA, Remoteness Areas, other Census tables).
*   **Data Profiling:** Automated generation of data profiling reports for outputs.
*   **Parameterization:** Allow more runtime parameterization of the pipeline (e.g., specific geographic regions to process).
*   **Containerization:** Package the ETL pipeline in Docker for easier deployment and portability.

## 11. Out of Scope (Initial Version)

*   Real-time data processing or streaming.
*   Building advanced statistical models or machine learning applications (the data warehouse *enables* these).
*   A graphical user interface (GUI) for pipeline execution or data exploration.
*   Direct integration with specific BI visualization tools (though outputs should be compatible).
*   Processing of non-ABS data sources.
