# Gemini Code Assistant Context

This document provides context for the Gemini code assistant to understand the AHGD (Australian Health Geography Data) project.

## Project Overview

The AHGD project is a modern, production-grade ETL (Extract, Transform, Load) pipeline written in Python. Its primary purpose is to integrate health, environmental, and socio-economic data from various Australian government sources into a unified, analysis-ready dataset at the Statistical Area Level 2 (SA2) geographic level.

This project has been significantly upgraded to a V2 architecture, leveraging modern data tools for enhanced performance, scalability, and maintainability. Key technologies include:

*   **Polars**: For high-performance data manipulation, replacing pandas.
*   **DuckDB**: An in-process analytical database for efficient intermediate data storage and querying.
*   **dbt (data build tool)**: For robust, version-controlled, and testable data transformation and modeling.
*   **Apache Airflow**: For scalable and observable workflow orchestration, replacing custom CLI and DVC-based orchestration.

The project is designed to be robust and maintainable, with a strong emphasis on code quality, testing, and documentation. It uses a modular architecture, with separate components for data extraction, transformation, validation, and loading.

### Key Technologies

*   **Programming Language:** Python 3.8+
*   **Core Libraries:**
    *   `pandas`: For data manipulation and analysis.
    *   `pydantic`: For data validation and schema enforcement.
    *   `click`: For creating the command-line interface.
    *   `requests`: For fetching data from web sources.
*   **Data Versioning:** `dvc` (Data Version Control) is used to manage and version large data files.
*   **Development Tools:**
    *   `pytest`: For unit and integration testing.
    *   `black`: For code formatting.
    *   `isort`: For import sorting.
    *   `mypy`: For static type checking.
    *   `pre-commit`: For running checks before committing code.

### Architecture

The ETL pipeline is orchestrated through a command-line interface (CLI) built with `click`. The main components of the pipeline are:

*   **Extractors:** Responsible for fetching raw data from various sources (e.g., AIHW, ABS, BOM, Medicare/PBS).
*   **Transformers:** Standardize, clean, and integrate the raw data into a master dataset.
*   **Validators:** Perform multi-layered validation checks (e.g., schema, geographic, statistical, temporal) to ensure data quality.
*   **Loaders:** Export the final, processed data into multiple formats (e.g., Parquet, CSV, GeoJSON).

The pipeline is configured through YAML files located in the `configs` directory. `dvc` is used to define and manage the stages of the ETL pipeline, as specified in `dvc.yaml`.

## Building and Running (V2 - Airflow Orchestrated)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Mrassimo/ahgd.git
    cd ahgd
    ```

2.  **Build and launch the Airflow environment:**
    This will build the Docker images and start the Airflow webserver, scheduler, and a PostgreSQL database.
    ```bash
    docker-compose build
    docker-compose up -d
    ```

3.  **Access Airflow UI:**
    Open your browser and navigate to `http://localhost:8080`. Log in with username `admin` and password `admin`.

4.  **Unpause the `ahgd_etl_v2` DAG:**
    In the Airflow UI, find the `ahgd_etl_v2` DAG and toggle it to "On" (unpause).

### Running the ETL Pipeline

To trigger the complete ETL pipeline, manually trigger the `ahgd_etl_v2` DAG in the Airflow UI. You can monitor the progress of the pipeline directly in the Airflow UI, observing the status of each task (extraction, loading to DuckDB, dbt build, dbt test, export).

### Testing

The project has a comprehensive test suite using `pytest`.

*   **Run all tests:**
    ```bash
    pytest
    ```

*   **Run tests with coverage:**
    ```bash
    pytest --cov=src
    ```

## Development Conventions

*   **Code Style:** The project follows the `black` code style with a line length of 88 characters. `isort` is used for sorting imports.
*   **Type Hinting:** All code should be fully type-hinted and pass `mypy` static analysis with the strict rules defined in `pyproject.toml`.
*   **Testing:** All new code should be accompanied by unit tests. The project aims for a high test coverage (80%+).
*   **Commits:** Commits should follow conventional commit standards. `pre-commit` hooks are used to enforce code quality before committing.
*   **Documentation:** All new features should be documented. The project uses Sphinx for generating documentation.
*   **dbt Development:** For data transformation logic, dbt models should be developed following dbt best practices (staging, intermediate, marts layers). All dbt models should have tests defined in `.yml` files.
*   **Airflow Development:** New DAGs should be developed following Airflow best practices, with clear task dependencies and appropriate operators. DAGs should be idempotent and retryable.

## Project Structure

*   `src/`: Contains the main source code for the ETL pipeline.
    *   `api/`: Source code for the project's API.
    *   `cli/`: The command-line interface for the project.
    *   `documentation/`: Code for generating the project's documentation.
    *   `extractors/`: Modules for extracting data from different sources.
    *   `loaders/`: Modules for loading data into different formats.
    *   `pipelines/`: ETL pipeline definitions and orchestration.
    *   `transformers/`: Modules for data transformation and cleaning.
    *   `validators/`: Modules for data validation and quality checks.
*   `tests/`: Contains the test suite for the project.
*   `configs/`: Configuration files for the ETL pipeline, including settings for different environments (development, testing, production).
*   `schemas/`: Pydantic schemas for data validation.
*   `dags/`: Airflow DAGs for pipeline orchestration.
*   `ahgd_dbt/`: dbt project for data transformation and modeling.
    *   `models/`: dbt models (staging, intermediate, marts).
    *   `analyses/`: dbt analyses.
    *   `macros/`: dbt macros.
    *   `seeds/`: dbt seeds.
    *   `tests/`: dbt tests.
*   `docs/`: Project documentation, including the data dictionary, API documentation, and technical guides.
*   `data/`: (Managed by DVC) Raw, processed, and final data files.
    *   `data_raw/`: Raw data extracted from the sources.
    *   `data_processed/`: Intermediate, processed data.
    *   `data_final/`: The final, analysis-ready dataset.
*   `examples/`: Example scripts and notebooks demonstrating how to use the data.

## Data Sources and Schemas

The project integrates data from the following sources:

*   **Australian Institute of Health and Welfare (AIHW)**
*   **Australian Bureau of Statistics (ABS)**
*   **Bureau of Meteorology (BOM)**
*   **Medicare/PBS**

The data schemas for each source and for the integrated dataset are defined using Pydantic models in the `schemas/` directory. These schemas are used for data validation throughout the ETL process.

## Configuration

The ETL pipeline is highly configurable using YAML files in the `configs/` directory. Key configuration files include:

*   `default.yaml`: Default configuration settings.
*   `development.yaml`, `testing.yaml`, `production.yaml`: Environment-specific configurations.
*   `logging_config.yaml`: Configuration for logging.
*   `extractors/`: Configurations for each data source extractor.
*   `pipelines/`: Configurations for the different ETL pipelines.

## Documentation

The project includes comprehensive documentation in the `docs/` directory:

*   **Data Dictionary:** A detailed description of all the fields in the final dataset.
*   **API Documentation:** Documentation for the project's API.
*   **Technical Documentation:** In-depth guides on the ETL process, data lineage, and architecture.
*   **Tutorials:** A `DATA_ANALYST_TUTORIAL.md` is available to guide users on how to analyze the data.
