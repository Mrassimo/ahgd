# Story File: Task 1.1 - Debug Duplicate Keys in fact_health_conditions_refined.parquet

## Goal
Debug and resolve duplicate keys in the `fact_health_conditions_refined.parquet` table to ensure data integrity. The grain of this fact table is defined as (geo_sk, time_sk, condition_sk, demo_sk, characteristic_sk), meaning each combination should be unique. This task addresses validation failures where duplicates were found, likely due to issues in join logic or refinement processes.

## Acceptance Criteria
- No duplicate rows exist in `fact_health_conditions_refined.parquet` based on the grain (geo_sk, time_sk, condition_sk, demo_sk, characteristic_sk).
- All surrogate keys are correctly generated and referenced.
- Validation checks in `validation.py` pass without errors for uniqueness.
- The ETL process runs successfully without duplicate key warnings.
- Code changes are documented and tested.

## Microtask List

### Microtask 1: Review and Analyze Current Code for Duplicate Key Issues
- **Description:** Examine the existing code in `etl_logic/tables/g21_conditions_by_characteristics.py` and related modules to identify where duplicates might be introduced. Focus on join logic, data filtering, and surrogate key assignments. Embed context: The fact table is refined from G21 data, involving joins with dimensions like `dim_geography`, `dim_time`, `dim_health_condition`, `dim_demographic`, and `dim_person_characteristic`. Check for multiple rows with the same business key combinations before surrogate key application.
- **Acceptance Criteria:** Identification of potential sources of duplicates (e.g., incorrect filters or join conditions). Code review notes documented in comments or a local log.
- **Estimated Effort:** M (Medium, as it involves code analysis and may require debugging).
- **Dependencies:** None. This is an initial investigative step.

### Microtask 2: Implement Fixes to Join Logic and Ensure Uniqueness
- **Description:** Modify the join logic in `etl_logic/tables/g21_conditions_by_characteristics.py` to enforce the correct grain. Use Polars to group by business keys and handle any duplicates before assigning surrogate keys. Embed context: Surrogate keys are generated in `dimensions.py` using a hashing function (e.g., MD5 on combined business keys). Ensure that during refinement, data is aggregated or filtered to maintain uniqueness. Example fix: Add a groupby operation on business keys and select the first row or sum aggregates as appropriate.
- **Acceptance Criteria:** After changes, running the ETL process results in no duplicate rows. Uniqueness can be checked with Polars query like `df.select(['geo_sk', 'time_sk', 'condition_sk', 'demo_sk', 'characteristic_sk']).unique().shape[0] == df.shape[0]`.
- **Estimated Effort:** L (Large, as it may involve significant code changes and testing).
- **Dependencies:** Microtask 1 must be completed first to identify issues.

### Microtask 3: Update Validation Checks in validation.py
- **Description:** Enhance the uniqueness validation in `validation.py` to specifically check for duplicates in `fact_health_conditions_refined`. Add or modify functions to log and report any violations. Embed context: Validation standards from PLANNING.md require enforcing grain-level uniqueness. Use Polars for efficient checks, e.g., `df.group_by(['geo_sk', 'time_sk', 'condition_sk', 'demo_sk', 'characteristic_sk']).agg(pl.count().alias('count')).filter(pl.col('count') > 1)`.
- **Acceptance Criteria:** Validation function returns no errors for duplicates. Test cases added to ensure the check works.
- **Estimated Effort:** S (Small, as it builds on existing validation framework).
- **Dependencies:** Microtask 2, as fixes must be in place before updating validation.

### Microtask 4: Test and Verify Fixes
- **Description:** Run the ETL process with sample data and verify that no duplicates are present. Use `run_etl.py` to execute the pipeline and check outputs. Embed context: ETL steps include extract, transform, load, and validate. Ensure that dimension tables are generated correctly in `dimensions.py` before fact table processing.
- **Acceptance Criteria:** ETL run completes successfully, and validation reports confirm no duplicates. Document any changes in code comments.
- **Estimated Effort:** M (Medium, involving testing and potential iterations).
- **Dependencies:** All previous microtasks in this story.

## Additional Context
This task is part of Epic 1: Fix Validation Failures in the AHGD ETL project. The dimensional model includes fact and dimension tables stored in Parquet files. Surrogate keys are generated using hashing in `dimensions.py` to ensure referential integrity. The ETL process uses Polars for data manipulation, and validation is handled in `validation.py`. Specific to this task, `fact_health_conditions_refined` is based on G21 Census data, with a grain of (geo_sk, time_sk, condition_sk, demo_sk, characteristic_sk). Refer to PLANNING.md for data model details, but all necessary information is embedded here: dimensions are generated before facts, and uniqueness must be enforced during transformation. File paths are relative to the project root: outputs in `output/`, ETL logic in `etl_logic/`. Follow PEP 8 coding standards and add Google-style docstrings to any modified code.