# VERIFIED REAL DATA SOURCES

## ✅ CONFIRMED WORKING URLS

### **ABS Digital Boundaries**
- **SA2 2021 GDA2020**: https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_SHP_GDA2020.zip
  - Format: ZIP (95.97 MB)
  - Contains: Shapefile format SA2 boundaries
  
- **SA2 2021 GDA94**: https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_SHP_GDA94.zip
  - Format: ZIP (47.46 MB)
  - Contains: Shapefile format SA2 boundaries

### **SEIFA 2021 Data**
- **SA2 Level SEIFA**: https://www.abs.gov.au/statistics/people/people-and-communities/socio-economic-indexes-areas-seifa-australia/2021/Statistical%20Area%20Level%202%2C%20Indexes%2C%20SEIFA%202021.xlsx
  - Format: Excel (1.26 MB)
  - Contains: All four SEIFA indexes by SA2
  - Includes: Rankings, deciles, percentiles

### **Medicare Benefits Schedule (data.gov.au)**
- **MBS Patient Demographics Historical**: https://data.gov.au/data/dataset/8a19a28f-35b0-4035-8cd5-5b611b3cfa6f/resource/519b55ab-8f81-47d1-a483-8495668e38d8/download/mbs-demographics-historical-1993-2015.zip
  - Format: ZIP with CSV files
  - Contains: Patient demographics and service usage 1993-2015

### **Pharmaceutical Benefits Scheme (data.gov.au)**
- **PBS Current Year**: https://data.gov.au/data/dataset/14b536d4-eb6a-485d-bf87-2e6e77ddbac1/resource/08eda5ab-01c0-4c94-8b1a-157bcffe80d3/download/pbs-item-2016csvjuly.csv
  - Format: CSV
  - Contains: Current year pharmaceutical usage
  
- **PBS Historical**: https://data.gov.au/data/dataset/14b536d4-eb6a-485d-bf87-2e6e77ddbac1/resource/56f87bbb-a7cb-4cbf-a723-7aec22996eee/download/csv-pbs-item-historical-1992-2014.zip
  - Format: ZIP with CSV files
  - Contains: Historical pharmaceutical data 1992-2014

## ✅ AIHW HEALTH INDICATOR DATA (VERIFIED)

### **AIHW MORT Books (Mortality Over Regions and Time)**
- **MORT Table 1**: https://data.gov.au/data/dataset/a84a6e8e-dd8f-4bae-a79d-77a5e32877ad/resource/a5de4e7e-d062-4356-9d1b-39f44b1961dc/download/aihw-phe-229-mort-table1-data-gov-au-2025.csv
  - Format: CSV (2.0 MB)
  - Geographic Coverage: SA3, SA4, LGA, PHN, States/Territories
  - Time Period: 2019-2023
  - Contains: Death counts, rates, premature mortality, avoidable deaths

- **MORT Table 2**: https://data.gov.au/data/dataset/a84a6e8e-dd8f-4bae-a79d-77a5e32877ad/resource/3b7d81af-943f-447d-9d64-9ce220be35e7/download/aihw-phe-229-mort-table2-data-gov-au-2025.csv
  - Format: CSV (8.1 MB) 
  - Note: Encoding issues detected, may require special handling

### **AIHW GRIM Books (General Record of Incidence of Mortality)**
- **GRIM Data**: https://data.gov.au/data/dataset/488ef6d4-c763-4b24-b8fb-9c15b67ece19/resource/edcbc14c-ba7c-44ae-9d4f-2622ad3fafe0/download/aihw-phe-229-grim-data-gov-au-2025.csv
  - Format: CSV (24.0 MB)
  - Geographic Coverage: National
  - Time Period: 1907-2023 (comprehensive historical data)
  - Contains: 12 major chronic disease categories, cancer types, mental health

### **PHIDU Social Health Atlas**
- **PHA Data**: https://phidu.torrens.edu.au/current/data/sha-aust/pha/phidu_data_pha_aust.xlsx
  - Format: Excel (73.7 MB)
  - Geographic Coverage: Population Health Areas (PHA) with SA2 mappings
  - Contains: Chronic disease prevalence, health service utilisation
  - Note: Complex multi-sheet structure, requires specialized extraction

- **LGA Data**: https://phidu.torrens.edu.au/current/data/sha-aust/lga/phidu_data_lga_aust.xls
  - Format: Excel (40.0 MB)
  - Geographic Coverage: Local Government Areas
  - Contains: Health indicators, chronic conditions, service access

## 🔍 NEXT TO VERIFY
- Australian Atlas of Healthcare Variation data sheets (Excel format)
- Bureau of Meteorology weather data
- State health department datasets

## 📊 DATA CHARACTERISTICS
- **Total Download Size**: ~285 MB for all datasets (original ~145 MB + 140 MB AIHW data)
- **Geographic Coverage**: 
  - SA2 areas (~2,300+ areas) via PHIDU PHA mappings
  - SA3 areas (912 unique areas in MORT data)
  - LGA, PHN, State/Territory levels
- **Time Coverage**: Historical data back to 1907 (GRIM), 1992 (PBS/MBS), 2019-2023 (MORT)
- **Health Indicators**: 15 chronic disease categories, mortality rates, service utilisation
- **Update Frequency**: Annual (ABS, AIHW), Quarterly (MBS/PBS data)

## 🔧 TECHNICAL REQUIREMENTS
- ZIP file extraction capability
- Excel file processing (.xlsx)
- CSV parsing with proper encoding
- Shapefile to GeoJSON conversion
- Large file handling (100MB+ downloads)

**Status**: Ready for implementation in real downloader
**Last Verified**: Current session
**Next Step**: Implement test-driven download validation