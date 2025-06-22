# Australian Health Geography Data (AHGD) Repository

## About the Project

The Australian Health Geography Data (AHGD) repository is a production-grade ETL pipeline that integrates health, environmental, and socio-economic data from multiple Australian government sources into a unified, analysis-ready dataset at the Statistical Area Level 2 (SA2) geographic level.

This repository provides researchers, policymakers, and healthcare professionals with comprehensive, quality-assured data covering all 2,473 SA2 areas across Australia, enabling evidence-based decision-making and advanced health analytics.

## 🌟 Key Features

- **Comprehensive Integration**: Combines data from AIHW, ABS, BOM, and Medicare/PBS
- **SA2 Geographic Granularity**: Standardised to Australian Statistical Geography Standard
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

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Mrassimo/ahgd.git
cd ahgd

# Set up the environment
./setup_env.sh --dev

# Activate virtual environment
source venv/bin/activate
```

### Example Usage

```python
import pandas as pd

# Load the master dataset from Hugging Face Hub
# Note: Install dependencies first: pip install pandas pyarrow fsspec
try:
    df = pd.read_parquet("hf://datasets/massomo/ahgd/data_processed/ahgd_master_dataset.parquet")
    print("Successfully loaded the dataset.")
    print(f"Dataset shape: {df.shape}")
    print(f"SA2 areas covered: {df['sa2_code'].nunique()}")
    
    # --- Example Analysis: Top 5 most disadvantaged areas in NSW ---
    print("\nTop 5 most disadvantaged SA2 areas in New South Wales (by SEIFA IRSD score):")
    nsw_data = df[df['state_name'] == 'New South Wales']
    top_5_disadvantaged = nsw_data.nsmallest(5, 'seifa_irsd_score')
    print(top_5_disadvantaged[['sa2_name', 'seifa_irsd_score', 'population_total']])
    
    # --- Example: Health indicators by remoteness ---
    print("\nAverage health indicators by remoteness category:")
    health_by_remoteness = df.groupby('remoteness_category').agg({
        'diabetes_prevalence': 'mean',
        'mental_health_issues_rate': 'mean',
        'gp_visits_per_capita': 'mean'
    }).round(2)
    print(health_by_remoteness)
    
except Exception as e:
    print(f"Failed to load dataset. Error: {e}")
    print("Please ensure you have run 'pip install pandas pyarrow fsspec'.")
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

## 🛠️ ETL Pipeline

### Pipeline Stages

1. **Extract**: Automated data retrieval from government APIs and portals
2. **Transform**: Standardisation, geographic harmonisation, and integration
3. **Validate**: Multi-layer quality assurance
4. **Load**: Export to multiple formats with optimisation

### Running the Pipeline

```bash
# Run complete pipeline
ahgd-pipeline --config configs/production.yaml

# Run individual stages
ahgd-extract --source aihw --output data_raw/
ahgd-transform --input data_raw/ --output data_processed/
ahgd-validate --input data_processed/ --rules schemas/
ahgd-load --input data_processed/ --output data_final/ --formats parquet,csv
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

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