# Product Context: AHGD ETL Refactoring

*This file explains why this specific ETL project exists, the problems it addresses within the broader AHGD vision, and outlines its intended functionality and user experience goals.*

## Problem Statement

The broader goal is to create an Australian Healthcare Geographic Database (AHGD) integrating geographic, demographic, and health data for evidence-based healthcare planning (as detailed in the AHGD2 context).

This specific project focuses on building the foundational ETL (Extract, Transform, Load) pipeline for this database. The immediate problems are:
1.  **Data Acquisition & Processing:** Ingesting and standardising core ABS data sources (ASGS geographic boundaries and Census population data like G01) into a usable format (Parquet).
2.  **Local Execution & Efficiency:** The existing codebase (AHGD3) is primarily designed for Colab and lacks a clear, efficient, and testable local execution pathway.
3.  **Maintainability & Testability:** The current structure needs refactoring to improve modularity, introduce automated testing, and make configuration more robust for different environments.

The goal is to refactor the existing AHGD3 ETL logic to run efficiently locally, be easily testable, maintain Colab compatibility, and produce the initial dimension tables (`geo_dimension.parquet`, `population_dimension.parquet`) as part of a planned star schema (detailed in `datadicttext.md`).

## Target Users (of this ETL Pipeline & its Output)

While the ultimate AHGD database serves policymakers, researchers, and providers, this *ETL pipeline* and its immediate outputs primarily target:

1.  **Data Engineers/Analysts (Internal):** Responsible for running, maintaining, and extending the ETL process.
2.  **Data Scientists/Analysts (PHI Focus):** Initial consumers of the generated Parquet files (geo, population dimensions) for analysis, potentially within a Private Health Insurance context (as suggested by `datadicttext.md`).
3.  **Future Developers:** Who will build upon this refactored, testable foundation.

## Functional Overview (ETL Pipeline)

1.  **Configuration:** Manage paths and parameters via `.env` for local runs and environment variables/arguments for Colab.
2.  **Download:** Fetch required ABS ASGS shapefile ZIPs and Census data pack ZIPs from URLs specified in `direct_urls.py`. Validate downloads.
3.  **Geographic Processing:**
    *   Extract ASGS shapefiles (SA1, SA2, SA3, SA4, STATE).
    *   Read, clean, and validate geometries using GeoPandas/Shapely.
    *   Convert to Polars DataFrame, standardise columns (geo\_code, geo\_level, geometry\_wkt).
    *   Combine levels and write `output/geo_dimension.parquet`.
4.  **Census Processing (G01):**
    *   Find and extract G01 CSV files for relevant geographic levels (SA1, SA2) from Census data packs.
    *   Read, clean, and select relevant columns (geo\_code, population counts) using Polars.
    *   Validate `geo_code` against `geo_dimension.parquet`.
    *   Combine data and write `output/population_dimension.parquet`.
5.  **Execution:** Provide a CLI entry point (`run_etl.py`) to orchestrate these steps locally. Maintain ability for Colab notebooks to call the core logic.

## User Experience Goals (for Developers/Users of the ETL)

*   **Local Execution:** The ETL process should be runnable locally with a single, clear command (e.g., `python run_etl.py`).
*   **Configuration:** Setup should be straightforward via a `.env` file locally.
*   **Testability:** Core logic components should have automated tests (`pytest`).
*   **Efficiency:** Leverage Polars for performant data processing.
*   **Clarity:** Codebase should be well-structured and understandable.
*   **Maintainability:** Refactored structure should be easier to maintain and extend.
*   **Colab Compatibility:** The core logic should remain usable within a Google Colab environment.