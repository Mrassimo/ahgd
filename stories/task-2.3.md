# Story File: Task 2.3 - Audit and Remove Hardcoded Values

## Goal
Audit the codebase for any hardcoded paths, URLs, constants, or other configuration-like values and relocate them to `etl_logic/config.py` or the `.env` file. This centralises configuration, reduces errors, and improves maintainability by making settings easier to update.

## Acceptance Criteria
- All identified hardcoded values (e.g., file paths like 'data/G01.csv', URLs, constants) are moved to `etl_logic/config.py` or `.env` as appropriate.
- Code is updated to reference the centralised configuration instead of hardcodes.
- ETL process runs successfully without errors after changes, with no loss of functionality.
- A log or report of changes is documented in the code or a comment for traceability.
- Code adheres to PEP 8 standards, includes type hints, and has updated docstrings where config references are changed.

## Microtask List

### Microtask 2.3.1: Identify Hardcoded Values in Codebase
- **Description:** Search through the codebase (focusing on `scripts/analysis/*.py`, `etl_logic/*.py`, and other relevant files) for hardcoded strings such as file paths, URLs, or constants. Use regex patterns to find candidates (e.g., strings containing 'data/', '.csv', or specific values).
- **Acceptance Criteria:** A list of all hardcoded values is compiled, categorised (e.g., paths, constants), and documented. No false positives; ensure the search is thorough but targeted.
- **Estimated Effort:** M (Medium, estimated 2-4 hours, due to code review)
- **Dependencies:** None

### Microtask 2.3.2: Move Hardcoded Paths and URLs to config.py or .env
- **Description:** Relocate identified hardcoded file paths and URLs to `etl_logic/config.py` (e.g., define `DATA_DIR` as a constant) or `.env` for environment-specific settings. Update code to import and use these config values.
- **Acceptance Criteria:** All paths and URLs are removed from code and referenced via config; changes are tested by running ETL to confirm correct behaviour.
- **Estimated Effort:** M (Medium, estimated 2-4 hours)
- **Dependencies:** Microtask 2.3.1 (identification must be complete first)

### Microtask 2.3.3: Move Hardcoded Constants to config.py
- **Description:** Identify and move any hardcoded constants (e.g., string literals for column names, thresholds) to `etl_logic/config.py`. Ensure they are defined as variables or in a config dictionary for easy access.
- **Acceptance Criteria:** Constants are centralised; code updates ensure dynamic referencing. Validate that changes do not introduce bugs.
- **Estimated Effort:** S (Small, estimated 1-2 hours)
- **Dependencies:** Microtask 2.3.1

### Microtask 2.3.4: Update Code and Test Changes
- **Description:** Refactor code to use the new config values, removing all hardcodes. Run the ETL process and tests to verify functionality.
- **Acceptance Criteria:** No hardcodes remain in the codebase; ETL and validation pass successfully. Add comments or logs to document the changes made.
- **Estimated Effort:** L (Large, estimated 4+ hours, as it may involve widespread updates)
- **Dependencies:** Microtasks 2.3.2 and 2.3.3 (relocation must be done first)

## Additional Context
This task enhances configuration management by eliminating hardcoding, as emphasised in the planning document. From `PLANNING.md`:
- **Audit Context:** Use regex or code search to find patterns like hardcoded strings in files such as `scripts/analysis/`. Move values to `etl_logic/config.py` for internal settings or `.env` for sensitive/environment-specific ones (e.g., `DATA_DIR`, `OUTPUT_DIR`).
- **File Paths:** Standard structure includes `data/` for inputs, `output/` for results; ensure config references these dynamically. For example, define `DATA_DIR = 'data/'` in config.py.
- **Coding Standards:** Follow PEP 8, use type hints (e.g., `str` for paths), and include docstrings. Load environment variables using `python-dotenv` for `.env` file integration.
- **General Guidelines:** After moving hardcodes, test the entire ETL flow to ensure no breaks. This task supports the project's maintainability by making configurations configurable without code changes, aligning with best practices for ETL systems.

This Story File includes all required context for implementation, ensuring developers can work independently without external references.