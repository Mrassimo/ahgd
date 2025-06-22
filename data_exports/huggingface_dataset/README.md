---
license: cc-by-4.0
task_categories:
- other
language:
- en
tags:
- australia
- health
- geography
- sa2
- demographics
- climate
pretty_name: Australian Health and Geographic Data (AHGD)
size_categories:
- n<1K
---

# Australian Health and Geographic Data (AHGD)

## Dataset Description

The Australian Health and Geographic Data (AHGD) dataset provides comprehensive health, demographic, and environmental indicators at the Statistical Area Level 2 (SA2) geography across Australia. This dataset integrates multiple authoritative Australian data sources to enable health geography research, policy analysis, and machine learning applications.

### Dataset Summary

- **Geographic Coverage**: Australian SA2 statistical areas
- **Temporal Coverage**: 2021 reference year
- **Data Sources**: Australian Institute of Health and Welfare (AIHW), Australian Bureau of Statistics (ABS), Bureau of Meteorology (BOM)
- **Total Records**: 12
- **Format Availability**: parquet, csv, json, geojson

### Supported Tasks

- Health geography analysis
- Spatial epidemiology research
- Environmental health studies
- Social determinants of health research
- Machine learning for health prediction
- Policy impact assessment

### Languages

English (Australian spelling and terminology)

## Dataset Structure

### Data Instances

Each record represents a Statistical Area Level 2 (SA2) with associated health, demographic, and environmental indicators.

### Data Fields

Key data fields include:

- **Geographic identifiers**: SA2, SA3, SA4 codes and names
- **Health indicators**: Life expectancy, chronic disease prevalence, health service utilisation
- **Environmental data**: Temperature, rainfall, air quality measures
- **Socioeconomic indicators**: SEIFA indices, employment rates
- **Demographic characteristics**: Population, age structure, cultural diversity

### Data Splits

This dataset does not have predefined train/validation/test splits as it represents cross-sectional geographic data.

## Dataset Creation

### Curation Rationale

This dataset was created to support health geography research and policy analysis in Australia by providing integrated, high-quality data at meaningful geographic scales.

### Source Data

#### Initial Data Collection and Normalisation

Data is sourced from:

1. **Australian Institute of Health and Welfare (AIHW)**: Health indicators and outcomes
2. **Australian Bureau of Statistics (ABS)**: Geographic boundaries and demographic data
3. **Bureau of Meteorology (BOM)**: Climate and environmental data

#### Who are the source language producers?

Australian government agencies producing official statistics and health data.

### Annotations

#### Annotation process

Data undergoes comprehensive validation including:
- Geographic boundary verification
- Statistical outlier detection
- Data quality scoring
- Cross-source consistency checking

#### Who are the annotators?

Automated validation systems with expert review by health geography researchers.

### Personal and Sensitive Information

This dataset contains only aggregated, de-identified data at the SA2 level. No individual-level information is included.

## Considerations for Using the Data

### Social Impact of Dataset

This dataset can support:
- Evidence-based health policy development
- Resource allocation decisions
- Health inequity identification
- Environmental health research

### Discussion of Biases

Users should be aware of:
- Potential underrepresentation in remote areas
- Temporal lag between data collection and availability
- Varying data quality across geographic regions

### Other Known Limitations

- Data currency varies by indicator
- Some regional areas may have suppressed values due to privacy requirements
- Climate data interpolation may introduce spatial uncertainty

## Additional Information

### Dataset Curators

Australian Health and Geographic Data (AHGD) Project Team

### Licensing Information

Creative Commons Attribution 4.0 International (CC BY 4.0)

### Citation Information

```
@dataset{ahgd2024,
  title={Australian Health and Geographic Data (AHGD)},
  author={AHGD Project Team},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/ahgd/australian-health-geographic-data}
}
```

### Contributions

Thanks to the Australian Institute of Health and Welfare, Australian Bureau of Statistics, and Bureau of Meteorology for providing the underlying data sources.
