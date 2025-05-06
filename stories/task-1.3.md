# Story File: Task 1.3 - Resolve All Referential Integrity Errors

## Goal
Resolve referential integrity errors across all fact tables to ensure that every foreign key (surrogate key references to dimension tables) exists and is correctly linked. This involves verifying the order of dimension and fact table generation, correcting surrogate key generation, and fixing join logic in the ETL process. Referential integrity is crucial for data accuracy in the dimensional model.

## Acceptance Criteria
- All fact tables (e.g., `fact_health_conditions_refined`, `fact_no_assistance`, etc.) have valid foreign keys that reference existing records in dimension tables (e.g., `dim_time`, `dim_geography`, etc.).
- No referential integrity errors are reported by `validation.py` or `verify_surrogate_keys.py`.
- The ETL process runs successfully with dimensions generated before facts.
- Code changes ensure consistent surrogate key usage and proper join conditions.
- Validation and testing confirm that all links are intact.

## Microtask List

### Microtask 1: Verify and Correct Dimension Generation Order in run_etl.py
- **Description:** Review and modify `run_etl.py` to ensure all dimension tables are generated before any fact tables. Embed context: The ETL process steps in PLANNING.md specify that dimensions must be created first to avoid key mismatches. Check the sequence of function calls, ensuring modules like `dimensions.py` are executed prior to fact table processing in files such as `etl_logic/tables/gXX_*.py`. Add explicit ordering or dependencies in the orchestration logic.
- **Acceptance Criteria:** ETL execution order is updated so that dimension tables are always processed first. Run `run_etl.py` and confirm no order-related errors.
- **Estimated Effort:** M (Medium, as it involves reviewing and potentially restructuring the ETL flow).
- **Dependencies:** None. This is a foundational step.

### Microtask 2: Verify and Correct Surrogate Key Generation in dimensions.py
- **Description:** Examine `dimensions.py` to ensure surrogate keys are generated correctly and uniquely for all dimension tables (e.g., `time_sk`, `geo_sk`, `condition_sk`). Fix any issues with hashing or key creation that could lead to missing or invalid keys. Embed context: Surrogate keys are generated using a hashing function (e.g., MD5 on business keys) as per PLANNING.md. Example code snippet: `def generate_surrogate_key(*args): return md5_hash('_'.join(map(str, args)))`. Ensure keys are non-NULL and unique, and handle any edge cases in data.
- **Acceptance Criteria:** All dimension tables have complete, unique surrogate keys. Tests or manual checks confirm no missing keys.
- **Estimated Effort:** L (Large, due to potential comprehensive changes across multiple dimension tables).
- **Dependencies:** Microtask 1, to ensure dimensions are generated correctly before further checks.

### Microtask 3: Fix Join Logic in Fact Table Processing
- **Description:** Review and correct join logic in fact table modules (e.g., `etl_logic/tables/g21_conditions_by_characteristics.py`, `etl_logic/tables/g25_unpaid_assistance.py`) to ensure proper linking to dimension surrogate keys. Embed context: Fact tables join on business keys during transformation, then map to surrogate keys. Use Polars for joins, e.g., `df.join(dim_table, on='business_key', how='left')`. Address any mismatches or orphaned records by adding filters or handling NULL values.
- **Acceptance Criteria:** All joins in fact tables correctly reference dimension surrogate keys, with no dangling references. Validation passes for referential integrity.
- **Estimated Effort:** L (Large, as multiple fact tables may need updates).
- **Dependencies:** Microtasks 1 and 2, as dimension order and keys must be fixed first.

### Microtask 4: Update and Enhance Validation in validation.py and verify_surrogate_keys.py
- **Description:** Modify `validation.py` and `verify_surrogate_keys.py` to include comprehensive referential integrity checks. Add functions to verify that all foreign keys in fact tables exist in their respective dimension tables. Embed context: Validation standards from PLANNING.md include checks for referential integrity using Polars, e.g., `fact_df.join(dim_df, on='sk_column', how='left').filter(pl.col('dim_sk').is_null()).is_empty()`. Ensure errors are logged and reported.
- **Acceptance Criteria:** Validation scripts detect and report any referential integrity issues, and after fixes, they pass cleanly.
- **Estimated Effort:** M (Medium, building on existing validation code).
- **Dependencies:** All previous microtasks, as fixes must be in place.

### Microtask 5: Test and Verify Referential Integrity Fixes
- **Description:** Run the full ETL process with sample data and use validation tools to confirm no referential integrity errors. Document changes and add automated tests if possible. Embed context: ETL steps involve extract, transform, load, and validate phases. Use outputs in `output/` for manual inspection if needed.
- **Acceptance Criteria:** ETL run completes with all referential integrity checks passing. No errors in `surrogate_key_verification.json` or validation logs.
- **Estimated Effort:** M (Medium, including testing across the pipeline).
- **Dependencies:** All prior microtasks in this story.

## Additional Context
This task is part of Epic 1: Fix Validation Failures in the AHGD ETL project. The data model includes dimensions (e.g., `dim_time`, `dim_geography`) and facts (e.g., `fact_health_conditions_refined`), with surrogate keys generated in `dimensions.py` for referential integrity. Validation is handled in `validation.py` and `verify_surrogate_keys.py`, checking for NULLs, uniqueness, and key existence. The ETL orchestration in `run_etl.py` must ensure dimensions are processed before facts. All necessary details are embedded here, including file paths (e.g., `etl_logic/`, `output/`) and best practices like using Polars and adhering to PEP 8. This avoids the need for external references to PLANNING.md or other files.