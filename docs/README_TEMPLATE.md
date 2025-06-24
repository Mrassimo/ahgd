# Australian Healthcare Geographic Database (AHGD) 

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset Size](https://img.shields.io/badge/Dataset-2,472%20records-blue.svg)](https://huggingface.co/datasets/ahgd)

## Overview

The Australian Healthcare Geographic Database (AHGD) is a comprehensive, production-ready dataset that integrates demographic, socioeconomic, health, and environmental data for all 2,472 Statistical Area Level 2 (SA2) regions across Australia. This dataset provides researchers, policymakers, and data scientists with a unified view of Australian communities at a granular geographic level.

## Dataset Description

This dataset combines data from multiple authoritative Australian government sources:

- **Australian Bureau of Statistics (ABS)**: 2021 Census demographic data
- **Australian Institute of Health and Welfare (AIHW)**: Health indicators and outcomes
- **Bureau of Meteorology (BOM)**: Climate and environmental data
- **Department of Health**: Healthcare service utilization data

### Key Features

- **Complete Coverage**: All 2,472 SA2 regions across Australia
- **Multi-dimensional**: Demographics, health, environment, and socioeconomic indicators
- **Production Quality**: Real government data sources with validation pipelines
- **Research Ready**: Cleaned, standardized, and integrated for immediate analysis

## Data Fields

The dataset contains the following key categories of information:

### Geographic Identifiers
- `sa2_code_2021`: Unique SA2 identifier (2021 boundaries)
- `sa2_name_2021`: SA2 region name
- `state_code_2021`: State/territory code
- `gcc_code_2021`: Greater Capital City Statistical Area code

### Demographic Data
- `total_population_2021`: Total population count
- `median_age_2021`: Median age of residents
- `indigenous_population_pct`: Percentage of Indigenous population
- `overseas_born_pct`: Percentage born overseas
- `unemployment_rate_2021`: Unemployment rate

### Socioeconomic Indicators
- `median_household_income_weekly`: Median weekly household income
- `seifa_advantage_disadvantage_score`: SEIFA disadvantage index
- `education_university_pct`: Percentage with university education

### Health & Environmental
- `healthcare_services_per_1000`: Healthcare services per 1000 residents
- `air_quality_index_avg`: Average air quality index
- `climate_zone`: Köppen climate classification

## Usage Example

```python
import pandas as pd

# Load the dataset
df = pd.read_parquet('ahgd_master_dataset_real.parquet')

# Basic exploration
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"SA2 regions covered: {df['sa2_code_2021'].nunique()}")

# Example analysis: Top 10 SA2s by population
top_population = df.nlargest(10, 'total_population_2021')[
    ['sa2_name_2021', 'state_code_2021', 'total_population_2021']
]
print("Top 10 most populous SA2 regions:")
print(top_population)
```

## Data Sources & Methodology

### Primary Data Sources
1. **ABS 2021 Census DataPacks**: Comprehensive demographic data at SA2 level
2. **ASGS Digital Boundary Files**: Geographic boundaries and classifications
3. **AIHW Health Data**: Health outcomes and healthcare utilization
4. **BOM Climate Data**: Temperature, rainfall, and environmental indicators

### Data Processing Pipeline
- Automated extraction from government APIs and data portals
- Standardization using common geographic identifiers (SA2_CODE_2021)
- Quality validation with statistical and business rule checks
- Integration using spatial and temporal alignment procedures

## File Formats

- **Parquet**: `ahgd_master_dataset_real.parquet` (optimised for analytics)
- **CSV**: `ahgd_master_dataset_real.csv` (universal compatibility)
- **Metadata**: `dataset_metadata.json` (schema and provenance information)

## Data Quality

- **Completeness**: 100% geographic coverage of Australian SA2 regions
- **Accuracy**: Validated against official ABS population totals
- **Timeliness**: Based on most recent available data (2021 Census)
- **Consistency**: Standardized field names and data types across sources

## Licensing & Attribution

This dataset is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

### Required Citation

```
Australian Healthcare Geographic Database (AHGD) v2.0.0 (2024). 
Integrated demographic, health, and environmental data for Australian SA2 regions.
Dataset derived from Australian Bureau of Statistics, Australian Institute of Health and Welfare,
and Bureau of Meteorology official data sources.
```

### Data Source Attribution
- Australian Bureau of Statistics (ABS) © Commonwealth of Australia
- Australian Institute of Health and Welfare (AIHW) © Commonwealth of Australia  
- Bureau of Meteorology (BOM) © Commonwealth of Australia

## Version Information

- **Version**: 2.0.0
- **Release Date**: June 2024
- **Records**: 2,472 SA2 regions
- **Reference Period**: 2021 (primary data year)

## Contact & Support

For questions, issues, or suggestions regarding this dataset, please open an issue in the repository or contact the maintainers.

---

*This dataset supports research into Australian healthcare accessibility, demographic patterns, and regional development planning.*