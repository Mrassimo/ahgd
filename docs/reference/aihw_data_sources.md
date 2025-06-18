# AIHW Health Indicator Data Sources

## Overview
This document describes the Australian Institute of Health and Welfare (AIHW) and related health indicator data sources identified for geographic health analysis. These datasets provide comprehensive health outcomes, chronic disease prevalence, and health service utilisation data at various geographic levels compatible with our SA2-based framework.

## Primary Data Sources

### 1. AIHW MORT Books (Mortality Over Regions and Time)
**Source**: data.gov.au  
**URL**: https://data.gov.au/data/dataset/mort-books  
**Format**: CSV files  
**Geographic Levels**: LGA, Greater Capital City Areas, Primary Health Networks, Remoteness Areas, SA3, SA4, States/Territories  
**Time Period**: 2019-2023  

**Key Indicators**:
- Death counts and rates
- Median age at death
- Premature deaths
- Potential years of life lost
- Potentially avoidable deaths
- Leading causes of death by sex

**Data Files**:
- MORT Table 1: General mortality indicators (2.0MB)
- MORT Table 2: Detailed demographic breakdowns (8.4MB)

### 2. AIHW GRIM Books (General Record of Incidence of Mortality)
**Source**: data.gov.au  
**URL**: https://data.gov.au/data/dataset/grim-books  
**Format**: CSV file  
**Geographic Levels**: National  
**Time Period**: 1907-2023 (historical coverage)  

**Key Indicators**:
- All causes of death
- 55 specific cause of death groupings
- Cancer, cardiovascular disease, COPD, dementia
- Chronic diseases, infectious diseases, suicide

**Data Files**:
- GRIM CSV: Complete historical mortality data

### 3. PHIDU Social Health Atlas of Australia
**Source**: Public Health Information Development Unit, Torrens University  
**URL**: https://phidu.torrens.edu.au/social-health-atlases/data  
**Format**: Excel workbooks (.xlsx, .xls)  
**Geographic Levels**: Population Health Areas (PHA), Local Government Areas (LGA), SA2 codes included  

**Key Indicators**:
- **Chronic Disease Prevalence**:
  - Diabetes
  - Heart disease/cardiovascular conditions
  - Cancer (multiple types)
  - Mental health conditions (depression, anxiety)
  - Respiratory conditions (asthma, COPD)
  - Arthritis, kidney disease, stroke
  - Dementia and Alzheimer's disease

- **Health Service Utilisation**:
  - Medicare-subsidised mental health services
  - Cancer screening participation (bowel, breast, cervical)
  - Preventive health service access

- **Health Risk Factors**:
  - Obesity prevalence
  - High blood pressure
  - Psychological distress levels

**Data Files**:
- Australia PHA data: Complete chronic disease indicators by Population Health Area
- Australia LGA data: Local Government Area level indicators

### 4. Australian Atlas of Healthcare Variation
**Source**: Australian Commission on Safety and Quality in Health Care  
**URL**: https://www.safetyandquality.gov.au/our-work/healthcare-variation/australian-atlas-healthcare-variation-data-sheets  
**Format**: Excel spreadsheets  
**Geographic Levels**: Primary Health Networks, SA3, SA4  
**Time Period**: 2014-2021 (varies by indicator)  

**Key Indicators**:
- **Chronic Disease Hospitalisations**:
  - COPD hospitalisations (2014-15 to 2017-18)
  - Heart failure hospitalisations
  - Diabetes complications hospitalisations
  - Kidney/urinary tract infection hospitalisations
  - Cellulitis hospitalisations

- **Healthcare Utilisation**:
  - Early planned births without medical indication
  - Caesarean section rates
  - Children's ENT surgeries
  - Lumbar spinal surgeries
  - Gastrointestinal investigations
  - Medicine use in older people

## Geographic Compatibility

### SA2 Level Data
- **PHIDU PHA Data**: Population Health Areas include SA2 code mappings
- **Geographic Correspondence**: PHAs are built from SA2 areas, allowing aggregation/disaggregation

### Other Geographic Levels
- **LGA**: Available in MORT, PHIDU datasets
- **SA3/SA4**: Available in MORT, Atlas datasets
- **Primary Health Networks**: Available in Atlas datasets

## Data Processing Approach

### 1. Direct Downloads
All identified datasets provide direct download URLs for automated extraction:
- CSV files: MORT, GRIM data (data.gov.au)
- Excel files: PHIDU data (torrens.edu.au)
- Excel sheets: Atlas data (safetyandquality.gov.au)

### 2. Data Extraction Pipeline
Implemented in `scripts/aihw_data_extraction.py`:
- Automated download with size and error checking
- Multi-format processing (CSV, Excel)
- Geographic code standardisation
- Database integration with `health_analytics.db`

### 3. Quality Validation
- Data completeness checks
- Geographic coverage validation
- Temporal alignment verification
- Cross-dataset consistency checks

## Priority Health Indicators

### High-Value Chronic Disease Indicators
1. **Diabetes Prevalence** (PHIDU PHA data)
   - Geographic Level: SA2-compatible PHAs
   - Time Period: Current
   - Coverage: National

2. **Cardiovascular Disease Hospitalisations** (Atlas data)
   - Geographic Level: PHN, SA3, SA4
   - Time Period: 2014-2018
   - Coverage: National

3. **Mental Health Service Utilisation** (PHIDU data)
   - Geographic Level: SA2-compatible PHAs
   - Time Period: Current
   - Coverage: National

4. **Mortality Rates by Cause** (MORT/GRIM data)
   - Geographic Level: LGA, SA3, SA4
   - Time Period: 2019-2023 (MORT), Historical (GRIM)
   - Coverage: National

## Data Integration Strategy

### Database Schema
```sql
-- Chronic disease prevalence (from PHIDU)
CREATE TABLE phidu_chronic_disease (
    geography_code TEXT,
    geography_name TEXT,
    sa2_codes TEXT,
    indicator_name TEXT,
    indicator_value REAL,
    year TEXT,
    data_source TEXT
);

-- Mortality indicators (from MORT/GRIM)
CREATE TABLE aihw_mort_data (
    geography_code TEXT,
    geography_name TEXT,
    indicator_type TEXT,
    cause_of_death TEXT,
    value REAL,
    year INTEGER,
    source_table TEXT
);
```

### Geographic Harmonisation
1. **SA2 Mapping**: Use PHA-to-SA2 correspondence files
2. **LGA Concordance**: Apply ABS correspondence tables
3. **Aggregation Rules**: Population-weighted averages for area aggregation

## Implementation Status

### ✅ Completed
- Data source identification and verification
- Download URL validation
- Extraction script development
- Database schema design
- Documentation creation

### 🔄 In Progress
- Data download and extraction
- Quality validation
- Geographic harmonisation
- Database population

### 📋 Next Steps
1. Execute `scripts/aihw_data_extraction.py`
2. Validate data quality and coverage
3. Create geographic correspondence tables
4. Integrate with existing SA2 boundaries
5. Develop health indicator analysis functions

## Data Usage Notes

### Limitations
- **PHIDU Data**: Population Health Areas don't perfectly align with SA2 boundaries
- **Atlas Data**: Limited to PHN/SA3/SA4 levels, requires aggregation for SA2 analysis
- **Temporal Variation**: Different datasets cover different time periods

### Strengths
- **Comprehensive Coverage**: Chronic disease, mortality, and service utilisation
- **High Quality**: Authoritative government and academic sources
- **Regular Updates**: Most datasets updated annually
- **Detailed Geographic Coverage**: Multiple levels available

### Recommended Usage
1. **Primary Analysis**: Use PHIDU PHA data for SA2-level chronic disease analysis
2. **Validation**: Cross-reference with MORT data for mortality patterns
3. **Temporal Analysis**: Use GRIM data for historical trends
4. **Service Analysis**: Use Atlas data for healthcare utilisation patterns

## Contact and Support

### Data Providers
- **AIHW**: https://www.aihw.gov.au/
- **PHIDU**: https://phidu.torrens.edu.au/
- **Safety and Quality Commission**: https://www.safetyandquality.gov.au/

### Technical Support
- **data.gov.au**: https://data.gov.au/
- **ABS Geographic Products**: https://www.abs.gov.au/geography

---

**Last Updated**: 2025-06-17  
**Next Review**: 2025-12-17  
**Status**: Ready for Implementation