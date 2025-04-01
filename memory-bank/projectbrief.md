# Project Brief: AHGD ETL Refactoring

*This is the foundational document shaping all others for this specific refactoring task. It defines core requirements and goals, acting as the project's scope anchor.*

## Project Vision

To refactor the existing AHGD ETL pipeline (currently processing ASGS Geography and Census G01 data) into a robust, maintainable, and testable system that can be executed efficiently both locally and in Google Colab environments. This serves as a foundational step towards building the larger Australian Healthcare Geographic Database (AHGD).

## Core Requirements

1.  **Local Execution:** Enable the ETL pipeline to be run reliably from a local development environment via a command-line interface.
2.  **Testability:** Implement a testing framework (`pytest`) with unit and integration tests covering the core ETL logic.
3.  **Configuration:** Manage environment-specific configurations (especially base paths) cleanly using `.env` files for local runs.
4.  **Modularity:** Ensure the core ETL logic within the `etl_logic` package is well-structured and reusable.
5.  **Colab Compatibility:** Maintain the ability to run the core ETL logic within a Google Colab notebook.
6.  **Output:** Ensure the refactored pipeline continues to correctly produce `geo_dimension.parquet` and `population_dimension.parquet`.

## Goals

*   **Goal 1:** Achieve efficient and reliable local execution of the existing ETL process.
*   **Goal 2:** Improve code quality, maintainability, and confidence through automated testing.
*   **Goal 3:** Establish a solid, refactored foundation for future extensions of the AHGD ETL pipeline (e.g., adding more Census tables, dimensions, facts).

## Scope

### In Scope (This Refactoring Task)

*   Refactoring existing code in `etl_logic/` (config, utils, geography, census).
*   Creating `run_etl.py` CLI orchestrator.
*   Implementing `.env` configuration for local paths.
*   Setting up `pytest` framework and writing initial tests for existing logic.
*   Updating `colab_runner.ipynb` to use refactored components.
*   Populating this Memory Bank.
*   Updating `README.md` with setup/usage instructions.

### Out of Scope (Future Work)

*   Adding ETL logic for new Census tables beyond G01.
*   Implementing processing for additional dimensions (time, health, socioeconomic, etc.).
*   Implementing processing for fact tables.
*   Implementing geographic correspondence processing.
*   Performance optimization beyond the use of Polars.
*   Building the final data warehouse (e.g., in DuckDB) or analysis layers on top of the Parquet files.