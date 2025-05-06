# PLANNING.md: AHGD ETL Project Technical Plan

**Version:** 1.0
**Date:** 2025-04-16

## 1. Project Vision

To Extract, Transform, and Load (ETL) Australian Bureau of Statistics (ABS) geographic boundary data (ASGS 2021) and Census 2021 General Community Profile (GCP) data into a structured dimensional model (star schema). The goal is to create a queryable data warehouse stored as Parquet files, suitable for healthcare planning, demographic analysis, and related research within the Australian Health Geography context.

## 2. Architecture Overview

The project follows a standard ETL pattern:

*   **Extract:** Download ASGS Shapefiles and Census GCP DataPacks (CSV format) from specified ABS URLs. Extract data from downloaded ZIP archives.
*   **Transform:**
    *   **Geographic Data:** Process ASGS Shapefiles using GeoPandas to create a unified geographic dimension, calculating centroids and relevant attributes.
    *   **Census Data:** Process individual Census CSV tables using Polars. This involves:
        *   Reading raw CSVs.
        *   Applying standardised column mappings (from `config.py`).
        *   Unpivoting/reshaping data from ABS formats into a long format suitable for fact tables.
        *   Cleaning and type casting data.
        *   Generating surrogate keys for dimension linkage.
        *   Joining with dimension tables (Geo, Time, Demographic, Health Condition, etc.) to create refined fact tables.
    *   **Dimension Generation:** Create dedicated dimension tables (Time, Health Condition, Demographic, Person Characteristic) with surrogate keys based on distinct values found in the source data or predefined structures. Ensure dimensions are created *before* facts that depend on them.
*   **Load:** Save the final dimension and fact tables as Parquet files in the `output/` directory. Enforce predefined schemas during write.
*   **Validation:** Implement data quality checks (counts, nulls, uniqueness, referential integrity) post-load using functions in `etl_logic/validation.py`.

**Core Components:**
*   `run_etl.py`: Main orchestration script, executes ETL steps sequentially.
*   `etl_logic/`: Python package containing all core logic.
    *   `config.py`: Central configuration hub (paths, URLs, mappings, schemas). **Needs completion.**
    *   `utils.py`: Common helper functions (download, logging, generic parsing).
    *   `geography.py`: ASGS processing.
    *   `census.py`: Generic Census file handling framework. **Needs refactoring (move specific logic).**
    *   `dimensions.py`: Dimension table creation logic. **Needs review for SK generation and completeness.**
    *   `time_dimension.py`: Time dimension generation.
    *   `validation.py`: Data quality checks. **Needs testing.**
    *   `tables/`: Modules for specific Census table (Gxx) processing logic. **Target location for refactored logic.**
*   `scripts/`: Utility, analysis, and testing support scripts. **Needs review/cleanup.**
*   `tests/`: Pytest-based unit and integration tests. **Needs expansion.**
*   `output/`: Target directory for Parquet files and reports.
*   `data/`: Storage for raw downloaded data (transient, gitignored).
*   `logs/`: Storage for execution logs.
*   `documentation/`: Project documentation (README, data dictionary, diagrams).

**Data Flow:**
ABS Website (ZIPs) -> `data/` directory -> Python (`Requests`, `zipfile`) -> Polars/GeoPandas DataFrames (Transformation in `etl_logic/`) -> `output/` directory (Parquet files). Validation checks run against output files.

## 3. Technology Stack

*   **Language:** Python 3.x (Targeting 3.9+)
*   **Core Data Processing:** Polars (Primary for tabular data manipulation due to performance and expressive API)
*   **Geospatial Processing:** GeoPandas, Shapely
*   **Utilities:** Pandas (Minimal use, for compatibility or specific utility tasks if needed), Requests (downloads), python-dotenv (environment variables), zipfile (extraction)
*   **Testing:** pytest, pytest-mock, pytest-cov (for coverage reporting)
*   **Configuration:** Python modules (`config.py`), `.env` file for environment-specific variables (like `BASE_DIR`).
*   **Output Format:** Apache Parquet (Columnar, efficient storage and query)
*   **Environment:** Local execution (primary), Google Colab (secondary/experimental via notebooks - **needs alignment check**)
*   **Linting/Formatting:** Black (code formatting), Flake8 (linting) - **Recommended for consistency.**
*   **Version Control:** Git

## 4. Data Models

The target data model is a star schema.

**Dimensions:**
*   `geo_dimension.parquet`: Geographic information (SA1, SA2, SA3, SA4, STE, POA levels) with surrogate keys (`geo_sk`) and centroids. Derived from ASGS Shapefiles.
    *   Key Columns: `geo_sk` (PK), `SA1_CODE_2021`, `SA2_CODE_2021`, ..., `geometry`, `centroid_lat`, `centroid_lon`.
*   `dim_time.parquet`: Time dimension (Year). Surrogate key (`time_sk`). Currently static (2021).
    *   Key Columns: `time_sk` (PK), `year`.
*   `dim_health_condition.parquet`: Health conditions mentioned in Census tables (e.g., G19, G20, G21). Surrogate key (`condition_sk`).
    *   Key Columns: `condition_sk` (PK), `condition_code`, `condition_label`. **Needs schema review/correction (ensure `condition_sk` exists and is populated correctly).**
*   `dim_demographic.parquet`: Demographic categories (e.g., Age, Sex). Surrogate key (`demographic_sk`).
    *   Key Columns: `demographic_sk` (PK), `age_group`, `sex`. **Needs review based on G-table requirements.**
*   `dim_person_characteristic.parquet`: Other person characteristics (e.g., Indigenous status, language spoken). Surrogate key (`characteristic_sk`).
    *   Key Columns: `characteristic_sk` (PK), `characteristic_code`, `characteristic_label`. **Needs review based on G-table requirements (e.g., G21).**

**Facts (Examples - based on `apply.md`):**
*   `fact_population.parquet` (from G01): Population counts.
    *   FKs: `geo_sk`, `time_sk`, `demographic_sk`
    *   Measures: `population_count`
*   `fact_income.parquet` (from G17): Income distribution.
    *   FKs: `geo_sk`, `time_sk`, `demographic_sk`, `income_bracket_sk` (Requires `dim_income_bracket`)
    *   Measures: `person_count`
*   `fact_assistance_needed.parquet` (from G18): Persons needing assistance.
    *   FKs: `geo_sk`, `time_sk`, `demographic_sk`
    *   Measures: `needs_assistance_count`
*   `fact_health_conditions_refined.parquet` (from G19, G20, G21): Counts of persons by health condition, demographics, characteristics.
    *   FKs: `geo_sk`, `time_sk`, `condition_sk`, `demographic_sk`, `characteristic_sk` (as applicable)
    *   Measures: `person_count`
    *   **Needs review for grain and key uniqueness (identified validation issue).**
*   `fact_unpaid_care.parquet` (from G25): Persons providing unpaid care.
    *   FKs: `geo_sk`, `time_sk`, `demographic_sk`
    *   Measures: `provides_care_count`, `receives_care_count`
*   `fact_no_assistance.parquet` (from G25): Persons needing assistance but not receiving it.
    *   FKs: `geo_sk`, `time_sk` (**NULL FK issue identified**), `demographic_sk`
    *   Measures: `no_assistance_count`

**Schema Design Principles:**
*   Use surrogate keys (integer type, generated sequentially or via hashing) for all dimension primary keys (`_sk` suffix).
*   Fact tables contain foreign keys referencing dimension surrogate keys.
*   Use descriptive column names.
*   Define explicit data types for all columns using Polars dtypes in `config.py` and enforce them on write.
*   Aim for granular fact tables where possible, allowing aggregation. Review grain carefully (e.g., `fact_health_conditions_refined`).

## 5. API Specifications

Not applicable. This is a batch ETL process, not an API-driven service. Data access is via the output Parquet files.

## 6. File Structure

The current file structure (as listed in `apply.md`) is generally logical. Key areas for refinement:

*   **Consolidate Logic:** Ensure all table-specific processing logic resides within `etl_logic/tables/gXX_*.py`.
*   **Clean `etl_logic/census.py`:** Remove table-specific functions, retain only generic framework/helper functions.
*   **Review `scripts/`:** Categorise scripts (analysis, utility, testing), ensure they use central config, and potentially integrate useful ones into the main ETL flow or testing suite.
*   **Standardise Output:** Ensure all Parquet files land in `output/`. Profiling reports should go to `output/profiling_reports/`. Schema diagrams to `output/`.

AHGD3/
├── .env
├── .gitignore
├── apply.md                 # (Input for this plan)
├── PLANNING.md              # (This file)
├── README.md                # Project overview, setup, usage
├── requirements.txt         # Python dependencies
├── setup.py                 # Basic package setup for etl_logic
├── config.py                # Root config (imports from etl_logic.config)
├── run_etl.py               # Main CLI entry point
├── verify_surrogate_keys.py # FK validation script
├── surrogate_key_verification.json # Output of validation
│
├── data/                    # Raw downloaded data (transient, .gitignored)
│
├── documentation/           # Documentation files
│   ├── datadicttext.md      # Data dictionary (needs update)
│   └── etl_data_model_diagram.md # Diagram (needs update)
│
├── etl_logic/               # Core ETL Python package
│   ├── init.py
│   ├── config.py            # Central configuration (paths, URLs, mappings, schemas)
│   ├── utils.py             # Download, logging, extraction, generic helpers
│   ├── geography.py         # ASGS Shapefile processing -> geo_dimension
│   ├── census.py            # Generic Census processing framework (refactoring needed)
│   ├── dimensions.py        # Dimension table generation logic
│   ├── time_dimension.py    # Time dimension generation
│   ├── validation.py        # Data quality check functions
│   │
│   └── tables/              # Table-specific processing modules
│       ├── init.py
│       ├── g01_population.py
│       ├── g17_income.py
│       ├── ... (other G-tables) ...
│       └── g25_unpaid_assistance.py
│
├── logs/                    # Execution logs
│
├── output/                  # Generated output files
│   ├── .parquet            # Dimension and Fact tables
│   ├── profiling_reports/   # HTML profiling reports
│   └── data_schema.mmd      # Mermaid schema diagram
│
├── scripts/                 # Utility and analysis scripts
│   ├── analysis/            # Data exploration scripts (review needed)
│   ├── test_utilities/      # Scripts supporting tests
│   ├── generate_profiling_reports.py
│   └── run_data_documentation.py
│
└── tests/                   # Pytest tests
├── test_.py            # Unit and integration tests
└── test_data/           # Sample data for testing


## 7. Coding Standards

*   **Language:** Python 3.9+
*   **Style Guide:** PEP 8.
*   **Formatting:** Use `black` for automated code formatting. Run `black .` before committing.
*   **Linting:** Use `flake8` for identifying potential errors and style issues. Configure `.flake8` if needed. Run `flake8 .` before committing.
*   **Naming Conventions:**
    *   Modules: `lowercase_with_underscores.py`
    *   Packages: `lowercase`
    *   Classes: `CamelCase`
    *   Functions/Methods: `lowercase_with_underscores()`
    *   Variables: `lowercase_with_underscores`
    *   Constants: `UPPERCASE_WITH_UNDERSCORES`
    *   Dimension Tables/Files: `dim_*.parquet`
    *   Fact Tables/Files: `fact_*.parquet`
    *   Surrogate Keys: `*_sk`
*   **Docstrings:** Use Google-style docstrings for all modules, classes, and functions. Explain purpose, arguments, and return values.
*   **Type Hinting:** Use Python type hints for function signatures and variables where practical to improve clarity and enable static analysis.
*   **Configuration:** Avoid hardcoding values (paths, URLs, column names, magic numbers). Use `etl_logic/config.py` and `.env`.
*   **Logging:** Use the standard `logging` module configured in `utils.py`. Log key events, transformations, errors, and summary statistics.
*   **Error Handling:** Implement robust error handling (e.g., `try...except` blocks) for file operations, downloads, and data processing steps. Log errors clearly.
*   **Modularity:** Keep functions focused on a single task. Break down complex processing into smaller, testable units. Favour composition over inheritance where appropriate.
*   **Comments:** Use comments (`#`) to explain complex logic or non-obvious decisions, but prefer clear code and good naming over excessive comments.

## 8. Testing Strategy

*   **Framework:** `pytest`
*   **Types of Tests:**
    *   **Unit Tests:** Test individual functions in isolation (e.g., helper functions in `utils.py`, specific parsing logic in `etl_logic/tables/gXX_*.py`, dimension generation logic, validation functions). Use `pytest-mock` to mock external dependencies (downloads, file system access).
    *   **Integration Tests:** Test the interaction between components (e.g., `process_census_table` verifying file finding, processing, joining; `run_etl.py` steps). Use small, representative sample data stored in `tests/test_data/`.
*   **Test Coverage:** Aim for high test coverage (>80%). Use `pytest-cov` to measure coverage and identify gaps. Focus coverage on complex transformation logic and validation rules.
*   **Test Data:** Maintain small, curated sample input files (CSVs, Shapefiles) in `tests/test_data/` that cover common cases, edge cases, and potential errors (e.g., different column headers, missing values).
*   **Assertions:** Use specific assertions (`assert result == expected`, `assert key in dataframe.columns`, `pytest.raises(...)`) rather than generic `assert True`.
*   **Execution:** Tests should be runnable via a simple `pytest` command from the root directory. Integrate testing into the development workflow (run tests before committing/pushing).
*   **Validation Tests:** Add specific tests for the functions within `etl_logic/validation.py` to ensure they correctly identify data quality issues.

**Priority Areas for Test Expansion (from `apply.md`):**
1.  Unit tests for each `process_gXX_file` function in `etl_logic/tables/`.
2.  Integration tests for `process_census_table` and the refinement steps involving dimension joins.
3.  Tests for `validation.py` functions.

## 9. Deployment Plan

*   **Target Environment:** Local execution initially. Potential future deployment could involve containerisation (Docker) or cloud-based execution (e.g., Cloud Functions, scheduled jobs), but this is out of scope for the current phase.
*   **Deployment Artifacts:** The primary outputs are the Parquet files in the `output/` directory. The codebase itself (`etl_logic/`, `scripts/`, `run_etl.py`) is the execution artifact.
*   **Process:**
    1.  Ensure the target machine has Python 3.9+ and required dependencies installed (`pip install -r requirements.txt`).
    2.  Set up the `.env` file with the correct `BASE_DIR`.
    3.  Execute the pipeline via `python run_etl.py --steps all` (or specific steps).
    4.  Verify successful completion by checking logs and running validation (`python run_etl.py --steps validate`, `python verify_surrogate_keys.py`).
    5.  Access the output Parquet files in the `output/` directory.
*   **Automation:** For repeated runs, consider simple shell scripts or task schedulers (like `cron` on Linux/macOS or Task Scheduler on Windows) to automate the execution of `run_etl.py`.

## 10. Constraints and Assumptions

*   **Data Sources:** Assumes ABS continues to provide ASGS and Census GCP data in the expected formats (Shapefile, CSV within ZIP) at stable URLs (defined in `config.py`). Changes in format or URL structure will require code updates.
*   **Data Volume:** Assumes data volumes are manageable for local processing with Polars/GeoPandas on a reasonably modern machine. Very large datasets might require optimisation or distributed processing (out of scope).
*   **Schema Stability:** Assumes the structure of the input Census tables (column names, meanings) is relatively stable for the 2021 Census. Significant variations might require adjustments to mappings in `config.py`.
*   **Network Access:** Requires internet connectivity to download data from ABS.
*   **Software Dependencies:** Relies on the availability and stability of external libraries (Polars, GeoPandas, etc.).
*   **Execution Environment:** Primarily designed for local execution. Colab compatibility is secondary and needs verification.
*   **Scope:** Focused on the specified G-tables (G01, G17, G18, G19, G20, G21, G25) and ASGS levels. Adding new tables or geographic levels will require extending the `etl_logic/tables/` modules and potentially `config.py`.
*   **Surrogate Key Generation:** Assumes the current approach (likely based on hashing or unique value mapping) is sufficient. Needs review for collision potential if data scales significantly.

## 11. Action Plan (Derived from apply.md Section 3)

This plan outlines the immediate priorities to stabilise and complete the ETL pipeline. Tasks should be addressed roughly in this order.

1.  **Fix Validation Failures (Highest Priority):**
    *   **Task 1.1:** Debug duplicate keys in `fact_health_conditions_refined.parquet`. Review grain (Geo, Time, Condition, Demo, Characteristic SKs) and join logic during refinement. Ensure uniqueness constraint is met.
    *   **Task 1.2:** Investigate and fix NULL `time_sk` in `fact_no_assistance.parquet`. Trace `time_sk` propagation for G25 processing.
    *   **Task 1.3:** Resolve all referential integrity errors identified by `validation.py` and `verify_surrogate_keys.py`.
        *   Ensure dimensions (`dim_health_condition`, `dim_demographic`, `dim_person_characteristic`) are generated *before* dependent facts. Adjust `run_etl.py` step order if needed.
        *   Verify/correct surrogate key generation in `dimensions.py`.
        *   Verify/correct join logic linking facts to dimension keys.
    *   **Task 1.4:** Correct schema for `dim_health_condition` (ensure `condition_sk` is present, correctly named, and populated).

2.  **Complete Configuration (`etl_logic/config.py`):**
    *   **Task 2.1:** Consolidate and complete `CENSUS_COLUMN_MAPPINGS` for all processed G-tables (G01, G17-G21, G25). Remove examples from `census.py`.
    *   **Task 2.2:** Define `SCHEMAS` using Polars dtypes for all target dimension and fact tables. Enforce these schemas during the "Load" phase (writing Parquet).
    *   **Task 2.3:** Audit codebase (especially `scripts/`) for any remaining hardcoded paths or values; move them to `config.py` or `.env`.

3.  **Refactor Code Structure:**
    *   **Task 3.1:** Move all remaining table-specific processing logic (`process_gXX_file` functions) from `etl_logic/census.py` to their respective `etl_logic/tables/gXX_*.py` modules.
    *   **Task 3.2:** Update `run_etl.py` imports and function calls to use the refactored locations in `etl_logic.tables`.
    *   **Task 3.3:** Refine generic helper functions remaining in `census.py`. Evaluate moving generic parsing/cleaning logic applicable across tables to `utils.py`.

4.  **Refactor Orchestration (`run_etl.py`):**
    *   **Task 4.1:** (Optional Enhancement) Consider making step execution data-driven (e.g., using a list/dict defining steps, functions, dependencies).
    *   **Task 4.2:** Improve error handling within the orchestration loop. Provide a clear summary report (success/failure, steps completed, validation status) at the end.
    *   **Task 4.3:** Ensure the `validate` step runs reliably after all load steps and its failure prevents reporting overall success.

5.  **Enhance Testing:**
    *   **Task 5.1:** Write specific unit tests for each `process_gXX_file` function in `etl_logic/tables/`, covering different input variations and edge cases.
    *   **Task 5.2:** Add integration tests for the end-to-end processing of at least one simple and one complex Census table, verifying joins and output structure.
    *   **Task 5.3:** Add unit tests for the functions in `etl_logic/validation.py`.
    *   **Task 5.4:** Implement `pytest-cov` and establish a baseline coverage report. Incrementally improve coverage.

6.  **Review Analysis Scripts (`scripts/analysis/`):**
    *   **Task 6.1:** Evaluate each script. Decide whether to: integrate into ETL, keep as separate tool (cleaned, using config), or remove.

7.  **Update Documentation:**
    *   **Task 7.1:** Revise `README.md` (structure, setup, usage).
    *   **Task 7.2:** Update `documentation/datadicttext.md` based on final schemas in `config.py`.
    *   **Task 7.3:** Add/improve docstrings throughout `etl_logic/`.

8.  **Review Colab Environment:**
    *   **Task 8.1:** Ensure `colab_runner.ipynb` and `ahgd_etl_notebook.py` align with the refactored codebase and run correctly.