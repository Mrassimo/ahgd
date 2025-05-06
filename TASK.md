# TASK.md: AHGD ETL Project Tasks

This document outlines the epics and tasks required to complete the AHGD ETL project, based on Section 11 of `PLANNING.md`. Tasks are prioritized based on the plan.

## Epic 1: Fix Validation Failures

**Description:** Address critical data validation errors identified in the initial runs to ensure data integrity and correctness. This is the highest priority.
**Status:** To Do

### Tasks

*   **Task 1.1:** Debug duplicate keys in `fact_health_conditions_refined.parquet`.
    *   **Description:** Review grain (Geo, Time, Condition, Demo, Characteristic SKs) and join logic during refinement. Ensure uniqueness constraint is met.
    *   **Dependencies:** None within this epic.
    *   **Status:** To Do
*   **Task 1.2:** Fix NULL `time_sk` in `fact_no_assistance.parquet`.
    *   **Description:** Trace `time_sk` propagation for G25 processing.
    *   **Dependencies:** None within this epic.
    *   **Status:** To Do
*   **Task 1.3:** Resolve all referential integrity errors.
    *   **Description:** Ensure dimensions are generated before dependent facts, verify/correct surrogate key generation in `dimensions.py`, and verify/correct join logic linking facts to dimension keys. Check `validation.py` and `verify_surrogate_keys.py` outputs.
    *   **Dependencies:** None within this epic (but dimension generation order is key).
    *   **Status:** To Do
*   **Task 1.4:** Correct schema for `dim_health_condition`.
    *   **Description:** Ensure `condition_sk` is present, correctly named, and populated.
    *   **Dependencies:** None within this epic.
    *   **Status:** To Do

## Epic 2: Complete Configuration (`etl_logic/config.py`)

**Description:** Centralise and finalise all configuration settings, including column mappings and table schemas.
**Status:** To Do
**Dependencies:** Epic 1 (Findings might influence config)

### Tasks

*   **Task 2.1:** Consolidate and complete `CENSUS_COLUMN_MAPPINGS`.
    *   **Description:** Finalise mappings for all processed G-tables (G01, G17-G21, G25) in `config.py`. Remove examples from `census.py`.
    *   **Dependencies:** None within this epic.
    *   **Status:** Done (YYYY-MM-DD)
*   **Task 2.2:** Define and enforce `SCHEMAS`.
    *   **Description:** Define Polars dtypes for all target dimension and fact tables in `config.py`. Enforce these schemas during Parquet writing.
    *   **Dependencies:** Task 1.4 (Schema correction for dim_health_condition).
    *   **Status:** Done (YYYY-MM-DD)
*   **Task 2.3:** Audit and remove hardcoded values.
    *   **Description:** Search codebase (especially `scripts/`) for hardcoded paths, URLs, or values and move them to `config.py` or `.env`.
    *   **Dependencies:** None within this epic.
    *   **Status:** Done (YYYY-MM-DD)

## Epic 3: Refactor Code Structure

**Description:** Improve code organisation by moving table-specific logic and refining shared utilities.
**Status:** Done (YYYY-MM-DD)
**Dependencies:** Epic 2

### Tasks

*   **Task 3.1:** Move table-specific logic to `etl_logic/tables/`.
    *   **Description:** Relocate all `process_gXX_file` functions from `etl_logic/census.py` to their respective `etl_logic/tables/gXX_*.py` modules.
    *   **Dependencies:** None within this epic.
    *   **Status:** Done (YYYY-MM-DD)
*   **Task 3.2:** Update `run_etl.py` imports/calls.
    *   **Description:** Modify `run_etl.py` to reflect the new locations of table processing functions.
    *   **Dependencies:** Task 3.1.
    *   **Status:** Done (YYYY-MM-DD)
*   **Task 3.3:** Refine generic helper functions.
    *   **Description:** Review functions remaining in `census.py`. Move generic parsing/cleaning logic applicable across tables to `utils.py`.
    *   **Dependencies:** Task 3.1.
    *   **Status:** Done (YYYY-MM-DD)

## Epic 4: Refactor Orchestration (`run_etl.py`)

**Description:** Enhance the main ETL execution script for better control, error handling, and reporting.
**Status:** To Do
**Dependencies:** Epic 3

### Tasks

*   **Task 4.1:** (Optional) Make step execution data-driven.
    *   **Description:** Consider refactoring `run_etl.py` to use a configuration (list/dict) defining steps, functions, and dependencies.
    *   **Dependencies:** None within this epic.
    *   **Status:** To Do
*   **Task 4.2:** Improve error handling and summary reporting.
    *   **Description:** Implement better try/except blocks in the main loop and provide a clear summary report at the end (success/failure, steps completed, validation status).
    *   **Dependencies:** None within this epic.
    *   **Status:** Done (2024-07-17)
*   **Task 4.3:** Ensure reliable `validate` step execution.
    *   **Description:** Confirm the `validate` step runs after all load steps and its failure prevents reporting overall success.
    *   **Dependencies:** Task 1.3 (Validation logic itself).
    *   **Status:** Done (2024-07-17)

## Epic 5: Enhance Testing

**Description:** Increase test coverage and robustness for core ETL logic and validation.
**Status:** To Do
**Dependencies:** Epic 3

### Tasks

*   **Task 5.1:** Write unit tests for `process_gXX_file` functions.
    *   **Description:** Create tests for each table processing function in `etl_logic/tables/`, covering various inputs and edge cases. Use mocking where appropriate.
    *   **Dependencies:** Task 3.1.
    *   **Status:** To Do
*   **Task 5.2:** Add integration tests for Census table processing.
    *   **Description:** Test the end-to-end flow for at least one simple and one complex Census table, verifying dimension joins and output structure using sample data.
    *   **Dependencies:** Task 3.1, Task 1.3 (Correct joins).
    *   **Status:** To Do
*   **Task 5.3:** Add unit tests for `validation.py`.
    *   **Description:** Write tests to ensure validation functions correctly identify data quality issues.
    *   **Dependencies:** Task 1.3 (Correct validation logic).
    *   **Status:** To Do
*   **Task 5.4:** Implement and improve test coverage.
    *   **Description:** Integrate `pytest-cov`, establish a baseline report, and incrementally increase coverage.
    *   **Dependencies:** Tasks 5.1, 5.2, 5.3.
    *   **Status:** To Do

## Epic 6: Review Analysis Scripts (`scripts/analysis/`)

**Description:** Assess the utility scripts in the analysis folder and integrate, clean, or remove them.
**Status:** To Do
**Dependencies:** Epic 2

### Tasks

*   **Task 6.1:** Evaluate analysis scripts.
    *   **Description:** Review each script in `scripts/analysis/`. Decide whether to integrate into the main ETL, clean up as a standalone tool (using `config.py`), or remove if redundant/obsolete.
    *   **Dependencies:** None within this epic.
    *   **Status:** To Do

## Epic 7: Update Documentation

**Description:** Ensure project documentation is accurate, complete, and reflects the final codebase.
**Status:** To Do
**Dependencies:** Epic 2, Epic 3

### Tasks

*   **Task 7.1:** Revise `README.md`.
    *   **Description:** Update the main README with correct project structure, setup instructions, and usage examples.
    *   **Dependencies:** Epic 3.
    *   **Status:** To Do
*   **Task 7.2:** Update data dictionary.
    *   **Description:** Update `documentation/datadicttext.md` based on the final schemas defined in `config.py`.
    *   **Dependencies:** Task 2.2.
    *   **Status:** To Do
*   **Task 7.3:** Add/improve docstrings.
    *   **Description:** Ensure all modules, classes, and functions in `etl_logic/` have clear Google-style docstrings.
    *   **Dependencies:** Epic 3.
    *   **Status:** To Do

## Epic 8: Review Colab Environment

**Description:** Verify that the Google Colab notebooks align with the refactored codebase.
**Status:** To Do
**Dependencies:** Epic 3

### Tasks

*   **Task 8.1:** Align Colab notebooks.
    *   **Description:** Check and update `colab_runner.ipynb` and `ahgd_etl_notebook.py` to ensure they work correctly with the refactored code structure and configuration.
    *   **Dependencies:** None within this epic.
    *   **Status:** To Do