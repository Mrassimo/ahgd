# Story File: Task 1.4 - Correct Schema for dim_health_condition

## Goal
Correct the schema of the `dim_health_condition` dimension table to ensure the `condition_sk` column is present, properly named, and populated with unique surrogate keys. This addresses schema inconsistencies that could affect data integrity and downstream processes in the ETL pipeline.

## Acceptance Criteria
- The `dim_health_condition.parquet` table includes a `condition_sk` column that is non-NULL, unique, and correctly generated.
- Schema matches the defined structure in `config.py` after updates.
- No schema-related errors in validation or ETL runs.
- All references to `condition_sk` in fact tables (e.g., `fact_health_conditions_refined`) work correctly.
- Changes are tested and documented.

## Microtask List

### Microtask 1: Review and Document Current Schema in dimensions.py
- **Description:** Examine the code in `dimensions.py` and the generated `dim_health_condition.parquet` to identify issues with the `condition_sk` column, such as missing definitions or incorrect naming. Embed context: From PLANNING.md, dimension tables use surrogate keys generated via hashing (e.g., MD5 on health condition codes). The schema should include columns like `condition_sk`, `condition_code`, and `condition_description`, with `condition_sk` as the primary key.
- **Acceptance Criteria:** Current schema documented, including any discrepancies (e.g., missing columns or data types). No changes made yet, just analysis.
- **Estimated Effort:** S (Small, focused on review and documentation).
- **Dependencies:** None. This is an initial step.

### Microtask 2: Implement Schema Corrections in dimensions.py and config.py
- **Description:** Update `dimensions.py` to ensure `condition_sk` is generated and included in the schema. Also, define or correct the schema in `config.py` for enforcement during ETL. Embed context: Use Polars data types (e.g., `pl.Utf8` for strings) and the surrogate key generation function. Example: In `config.py`, define `SCHEMAS['dim_health_condition'] = {'condition_sk': pl.Utf8, 'condition_code': pl.Utf8, 'condition_description': pl.Utf8}`. Ensure uniqueness during creation.
- **Acceptance Criteria:** Schema is corrected, and `dim_health_condition` output has all required columns with correct data types and no NULLs in `condition_sk`.
- **Estimated Effort:** M (Medium, involving code and configuration changes).
- **Dependencies:** Microtask 1 to identify specific issues.

### Microtask 3: Update Validation and ETL Logic to Enforce Correct Schema
- **Description:** Modify `validation.py` to include checks for the `dim_health_condition` schema, such as presence and uniqueness of `condition_sk`. Ensure ETL logic in `run_etl.py` and `etl_logic/tables/` enforces the schema. Embed context: Validation standards require data type conformance and NULL checks, using Polars functions like `df.schema` and `filter` for verification.
- **Acceptance Criteria:** Validation passes for schema correctness, and ETL process enforces the updated schema without errors.
- **Estimated Effort:** M (Medium, as it may affect multiple parts of the pipeline).
- **Dependencies:** Microtask 2, as schema must be fixed first.

### Microtask 4: Test and Verify Schema Corrections
- **Description:** Run the ETL process and validate the `dim_health_condition` table to confirm the schema is correct and impacts no other tables negatively. Use sample data to test. Embed context: Fact tables like `fact_health_conditions_refined` depend on this dimension, so ensure joins work after changes.
- **Acceptance Criteria:** No schema errors, all tests pass, and referential integrity is maintained with dependent fact tables.
- **Estimated Effort:** M (Medium, including comprehensive testing).
- **Dependencies:** All previous microtasks in this story.

## Additional Context
This task is part of Epic 1: Fix Validation Failures in the AHGD ETL project. The `dim_health_condition` dimension table supports fact tables by providing surrogate keys for health conditions. Surrogate keys are generated in `dimensions.py` using hashing for uniqueness, and schemas are defined in `config.py` for enforcement. Validation in `validation.py` checks for NULLs and data types. All necessary details, including ETL steps, file paths (e.g., `etl_logic/`, `output/`), and best practices, are embedded here to make this Story File self-contained. Adhere to PEP 8 and add docstrings as needed.