# Story File: Task 2.1 - Consolidate and Complete CENSUS_COLUMN_MAPPINGS

## Goal
Finalise the column mappings for all processed G-tables (G01, G17, G18, G19, G20, G21, G25) in `etl_logic/config.py`. This involves creating a comprehensive mapping dictionary or structure that defines how source Census columns map to target model columns, and removing any redundant mapping definitions from other files like `etl_logic/census.py`. The objective is to centralise configuration for easier maintenance and to eliminate duplication.

## Acceptance Criteria
- All column mappings for G01, G17, G18, G19, G20, G21, and G25 are defined in `etl_logic/config.py` as a dictionary or DataFrame.
- Mappings are accurate and complete, based on the source Census data structures.
- Redundant mapping code is removed from `etl_logic/census.py` and other relevant files.
- The ETL process continues to function correctly after changes, with no loss of functionality.
- Configuration is loaded dynamically at runtime, allowing for easy updates without code modifications.
- Code adheres to PEP 8 standards, includes type hints, and has Google-style docstrings.

## Microtask List

### Microtask 2.1.1: Define Column Mappings for G01 in config.py
- **Description:** Create a mapping dictionary in `etl_logic/config.py` for G01 (Population) columns, mapping source names (e.g., 'AGE', 'SEX') to target names (e.g., 'age_group', 'gender'). Ensure the mapping includes all relevant columns as per the Census data.
- **Acceptance Criteria:** Mapping is complete and accurate; can be used in ETL logic to transform G01 data. Mapping is tested by running a sample ETL for G01 and verifying output columns.
- **Estimated Effort:** S (Small, estimated 1-2 hours)
- **Dependencies:** None

### Microtask 2.1.2: Define Column Mappings for G17 in config.py
- **Description:** Add a mapping dictionary in `etl_logic/config.py` for G17 (Income) columns, ensuring source columns are mapped to target columns (e.g., 'INCOME' to 'income_amount'). Include any necessary transformations or derivations in the mapping logic.
- **Acceptance Criteria:** Mapping covers all G17 columns; ETL for G17 produces correct output when using the new mapping. No hardcoded mappings remain in other files.
- **Estimated Effort:** S (Small, estimated 1-2 hours)
- **Dependencies:** None (can be done independently, but should be consistent with overall config structure)

### Microtask 2.1.3: Define Column Mappings for G18, G19, G20, G21 in config.py
- **Description:** Define mappings for G18 (Assistance Needed), G19 (Health Conditions), G20 (Selected Conditions), and G21 (Conditions by Characteristics) in `etl_logic/config.py`. Ensure mappings align with dimension keys (e.g., mapping to `condition_sk`, `demo_sk`).
- **Acceptance Criteria:** All mappings are defined and integrated; ETL runs successfully for these tables with the new config. Mappings handle any specific data types or transformations required.
- **Estimated Effort:** M (Medium, estimated 2-4 hours, due to multiple tables)
- **Dependencies:** Microtask 2.1.1 and 2.1.2 (to ensure consistent config format across tables)

### Microtask 2.1.4: Define Column Mappings for G25 in config.py
- **Description:** Add mapping for G25 (Unpaid Assistance) columns in `etl_logic/config.py`, mapping source fields to target fields and ensuring compatibility with fact and dimension tables.
- **Acceptance Criteria:** Mapping is complete; ETL for G25 works correctly. All mappings are centralised in one config object.
- **Estimated Effort:** S (Small, estimated 1-2 hours)
- **Dependencies:** None (independent, but part of the sequence for completeness)

### Microtask 2.1.5: Remove Redundant Mappings from etl_logic/census.py
- **Description:** Search for and delete any existing column mapping code in `etl_logic/census.py` or other files, replacing references with imports from `config.py`. Update any code that relies on these mappings to use the centralised config.
- **Acceptance Criteria:** No duplicate mapping logic exists; all ETL code references `config.py` for mappings. Run ETL to confirm no errors.
- **Estimated Effort:** M (Medium, estimated 2-4 hours, as it involves code search and refactoring)
- **Dependencies:** All mapping definitions in Microtasks 2.1.1-2.1.4 must be complete first

## Additional Context
This task is part of Epic 2, which focuses on completing configuration in `etl_logic/config.py`. Configuration management is centralised to improve maintainability, as outlined in the project planning. All column mappings should be defined as dictionaries or DataFrames in `config.py`, with keys for each G-table (e.g., `CENSUS_COLUMN_MAPPINGS = {'G01': {...}, 'G17': {...}}`). 

From the planning document:
- **Column Mappings Context:** Mappings should handle transformations, such as mapping 'AGE' to derived 'age_group' categories. Ensure mappings align with the data model, where source columns are remapped to standardised names for dimensions and facts.
- **File Paths:** Work within `etl_logic/config.py`. Reference standard directory structure: raw data in `data/`, ETL logic in `etl_logic/`.
- **Coding Standards:** Follow PEP 8, use type hints (e.g., `Dict[str, Dict[str, str]]` for mappings), and include docstrings. For example, docstring for `CENSUS_COLUMN_MAPPINGS` should describe its structure and usage.
- **Data Types and Schemas:** While this task focuses on mappings, ensure mappings are compatible with schemas defined in Task 2.2. For instance, map to columns that will have specific Polars dtypes (e.g., categorical for low-cardinality fields).
- **General Guidelines:** Use Polars for data handling in ETL. After changes, validate the ETL flow to ensure data integrity. If any hardcoded values are found during this process, note them for Task 2.3.

This Story File contains all necessary information for implementation, drawn directly from the planning context. No external references are needed.