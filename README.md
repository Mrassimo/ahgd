# Australian Health Geography Data (AHGD) Repository

## About the Project

The Australian Health Geography Data (AHGD) repository is a modern, production-grade ETL pipeline that integrates health, environmental, and socio-economic data from multiple Australian government sources into a unified, analysis-ready dataset at the Statistical Area Level 2 (SA2) geographic level.

This repository has been upgraded to a V2 architecture, leveraging modern data tools like Polars for high-performance data manipulation, DuckDB for efficient in-process analytical processing, dbt (data build tool) for robust data transformation and modeling, and Apache Airflow for scalable workflow orchestration.

This repository provides researchers, policymakers, and healthcare professionals with comprehensive, quality-assured data covering all 2,473 SA2 areas across Australia, enabling evidence-based decision-making and advanced health analytics.

## 🌟 Key Features

- **Comprehensive Integration**: Combines data from AIHW, ABS, BOM, and Medicare/PBS
- **SA2 Geographic Granularity**: Standardised to Australian Statistical Geography Standard
- **Modern Data Stack**: Leverages Polars for data processing, DuckDB for analytical queries, dbt for data modeling, and Apache Airflow for orchestration.
- **Production-Grade ETL**: Robust pipeline with retry logic, checkpointing, and monitoring
- **Multi-Format Export**: Parquet, CSV, GeoJSON, and JSON formats
- **Quality Assured**: Multi-layered validation including statistical, geographic, and temporal checks
- **British English Compliance**: Consistent spelling and terminology throughout
- **Fully Documented**: Comprehensive API docs, tutorials, and data dictionary

## 📊 Data Sources

The AHGD integrates authoritative data from:

- **Australian Institute of Health and Welfare (AIHW)**: Disease prevalence, mortality statistics, healthcare utilisation
- **Australian Bureau of Statistics (ABS)**: Demographics, SEIFA indexes, geographic boundaries
- **Bureau of Meteorology (BOM)**: Climate and environmental health indicators
- **Department of Health**: Medicare statistics, PBS data, immunisation rates

## 🚀 Quick Start (V2 - Airflow Orchestrated)

### Prerequisites

*   Docker and Docker Compose installed
*   Python 3.9+ (for local development/testing outside Docker)

### Installation & Setup

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

To trigger the complete ETL pipeline:

1.  **Manually trigger the DAG:**
    In the Airflow UI, click the "Play" button for the `ahgd_etl_v2` DAG. This will start a new DAG run.

2.  **Monitor progress:**
    You can monitor the progress of the pipeline directly in the Airflow UI, observing the status of each task (extraction, loading to DuckDB, dbt build, dbt test, export).

### Example Usage (Post-ETL)

Once the Airflow pipeline has completed successfully, the processed data will be available in the `ahgd.db` DuckDB database (mounted as `./ahgd.db` in your project root). You can query it directly using DuckDB or connect from Python:

```python
import duckdb
import polars as pl

# Connect to the DuckDB database
con = duckdb.connect(database='./ahgd.db', read_only=True)

# Query the master health record
master_df = con.sql("SELECT * FROM master_health_record").pl()
print("Successfully loaded the master health record.")
print(f"Dataset shape: {master_df.shape}")

# Example Analysis: Top 5 most disadvantaged areas in NSW (using Polars)
print("\nTop 5 most disadvantaged SA2 areas in New South Wales (by SEIFA IRSD score):")
nsw_data = master_df.filter(pl.col('state_name') == 'New South Wales')
top_5_disadvantaged = nsw_data.sort('seifa_irsd_score').head(5)
print(top_5_disadvantaged.select(['sa2_name', 'seifa_irsd_score', 'total_population']))

# Close the connection
con.close()
```


## 📁 Project Structure

```
ahgd/
├── src/                    # Source code
│   ├── extractors/         # Data source extractors (AIHW, ABS, BOM, Medicare)
│   ├── transformers/       # Data transformation and standardisation
│   ├── validators/         # Multi-layer validation framework
│   ├── loaders/           # Multi-format export utilities
│   ├── pipelines/         # Pipeline orchestration
│   ├── documentation/     # Documentation generation tools
│   └── utils/             # Common utilities and interfaces
├── tests/                 # Comprehensive test suite
├── configs/              # Environment-specific configurations
├── schemas/              # Pydantic v2 data schemas
├── dags/                   # Airflow DAGs for pipeline orchestration
├── ahgd_dbt/               # dbt project for data transformation and modeling
│   ├── models/             # dbt models (staging, intermediate, marts)
│   ├── analyses/           # dbt analyses
│   ├── macros/             # dbt macros
│   ├── seeds/              # dbt seeds
│   └── tests/              # dbt tests
├── docs/                 # Documentation
│   ├── api/              # API documentation
│   ├── technical/        # Technical documentation
│   ├── diagrams/         # Data lineage diagrams
│   └── data_dictionary/  # Complete field descriptions
└── examples/             # Usage examples and tutorials
```

## 📖 Documentation

### Core Documentation

- **[Quick Start Guide](./docs/QUICK_START_GUIDE.md)**: Get started in 10 minutes
- **[Data Analyst Tutorial](./docs/DATA_ANALYST_TUTORIAL.md)**: Comprehensive guide for data analysis
- **[API Documentation](./docs/api/API_DOCUMENTATION.md)**: Programmatic access and integration

### Technical Documentation

- **[ETL Process Documentation](./docs/technical/ETL_PROCESS_DOCUMENTATION.md)**: Complete technical guide
- **[Data Lineage Diagrams](./docs/diagrams/README.md)**: Visual representation of data flow
- **[Known Issues & Limitations](./docs/KNOWN_ISSUES_AND_LIMITATIONS.md)**: Transparency about constraints

## 📋 Data Dictionary

A summary of key fields is provided below. For a complete and detailed data dictionary for all tables and fields, please see the **[Full Data Dictionary here](./docs/data_dictionary/data_dictionary.md)**.

### Master Health Record Schema

| Field | Type | Description |
|-------|------|-------------|
| `sa2_code` | string | Statistical Area Level 2 code (2021 edition) |
| `sa2_name` | string | Official SA2 name |
| `state_code` | string | State/Territory code |
| `population_total` | integer | Total population |
| `seifa_irsd_score` | integer | Index of Relative Socio-economic Disadvantage |
| `diabetes_prevalence` | float | Age-standardised diabetes prevalence rate |
| `mental_health_issues_rate` | float | Mental health service utilisation rate |
| `gp_visits_per_capita` | float | Average GP visits per person per year |
| `remoteness_category` | string | ABS remoteness classification |

## 🔍 Data Quality and Validation

The AHGD implements a comprehensive validation framework to ensure data quality:

### Validation Layers

1. **Schema Compliance**: All data validated against strict Pydantic v2 schemas
2. **Geographic Validation**: 
   - SA2 boundary topology checks
   - Complete coverage of all 2,473 official SA2 areas
   - GDA2020 coordinate system compliance
3. **Statistical Validation**:
   - Range checks for all health indicators
   - Outlier detection using multiple methods (IQR, Z-score, Isolation Forest)
   - Cross-dataset consistency validation
4. **Temporal Validation**:
   - Time series consistency checks
   - Data freshness validation
   - Trend analysis for anomaly detection

### Quality Metrics

Every record includes a quality score (0-1) based on:
- Completeness: Percentage of non-null values
- Accuracy: Conformance to business rules
- Consistency: Cross-dataset agreement
- Timeliness: Data currency

## 🛠️ ETL Pipeline (V2 - Airflow, dbt, Polars, DuckDB)

The AHGD ETL pipeline has been re-architected to leverage a modern data stack for improved performance, scalability, and maintainability.

### Pipeline Stages & Technologies

1.  **Extract (Python/Polars)**: Automated data retrieval from government APIs and portals, processed efficiently with Polars.
2.  **Load to DuckDB (Python/Polars)**: Raw data is loaded into a local DuckDB database for high-performance analytical processing.
3.  **Transform (dbt/SQL)**: Data transformation, standardization, geographic harmonization, and integration are managed declaratively using dbt models (SQL). dbt builds and tests are executed against the DuckDB database.
4.  **Validate (dbt Tests)**: Multi-layer quality assurance and data validation are integrated into the dbt models as tests.
5.  **Load/Export (Python/Polars)**: Final, processed data is exported from DuckDB to multiple formats (e.g., Parquet, CSV, GeoJSON) for consumption.

### Orchestration

The entire pipeline is orchestrated using **Apache Airflow**. The `ahgd_etl_v2` DAG defines the end-to-end workflow, managing dependencies, retries, and monitoring of each stage.

### Running the Pipeline

To run the pipeline, ensure your Docker environment is set up (as per Quick Start) and trigger the `ahgd_etl_v2` DAG in the Airflow UI.

## 🔒 Security

The AHGD project maintains industry-leading security standards for handling Australian health and geographic data.

### Security Highlights

- **Zero Critical Vulnerabilities**: 100% elimination of critical and high-severity vulnerabilities
- **Comprehensive Security Framework**: Multi-layered security controls and monitoring
- **Australian Compliance**: Full compliance with Privacy Act 1988 and healthcare security frameworks
- **Regular Security Audits**: Monthly vulnerability assessments and quarterly security reviews

### Security Resources

- **[Security Policy](SECURITY.md)**: Vulnerability reporting and response procedures
- **[Security Guidelines](docs/security/SECURITY_GUIDELINES.md)**: Ongoing security practices and procedures
- **[Security Checklist](docs/security/SECURITY_CHECKLIST.md)**: Pre-release and maintenance security verification
- **[Security Fix Reports](docs/security/)**: Detailed vulnerability remediation documentation

### Reporting Security Issues

**Do not create public GitHub issues for security vulnerabilities.** Instead:

- **Email**: security@ahgd-project.org
- **Emergency**: security-emergency@ahgd-project.org
- **Response Time**: Within 24 hours for critical issues

For more information, see our [Security Policy](SECURITY.md).

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

For local development and testing of individual components (extractors, transformers, etc.) outside of the Airflow environment:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest

# Run linting
black src/ tests/
isort src/ tests/
mypy src/
```

For dbt development, navigate to the `ahgd_dbt` directory and use `dbt` CLI commands (e.g., `dbt run`, `dbt test`, `dbt docs generate`).

For Airflow development, refer to the `dags/ahgd_etl_v2.py` file and the Docker Compose setup.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Data sources retain their original licensing terms. Please refer to individual source documentation for specific usage restrictions.

## 🙏 Acknowledgments

- Australian Institute of Health and Welfare (AIHW)
- Australian Bureau of Statistics (ABS)
- Bureau of Meteorology (BOM)
- Department of Health
- All contributors and the open-source community

## 📞 Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/Mrassimo/ahgd/issues)
- **Discussions**: [Community discussions](https://github.com/Mrassimo/ahgd/discussions)

---

**Version**: 1.0.0  
**Last Updated**: June 2025  
**Status**: Production Ready (Phases 1-4 Complete)