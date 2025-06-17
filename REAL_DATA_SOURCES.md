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

## 🔍 NEXT TO VERIFY
- Additional MBS current year data (quarterly files)
- AIHW Atlas health indicators
- Bureau of Meteorology weather data
- State health department datasets

## 📊 DATA CHARACTERISTICS
- **Total Download Size**: ~145 MB for core datasets
- **Geographic Coverage**: All Australian SA2 areas (~2,300+ areas)
- **Time Coverage**: Historical data back to 1992
- **Update Frequency**: Annual (ABS), Quarterly (health data)

## 🔧 TECHNICAL REQUIREMENTS
- ZIP file extraction capability
- Excel file processing (.xlsx)
- CSV parsing with proper encoding
- Shapefile to GeoJSON conversion
- Large file handling (100MB+ downloads)

**Status**: Ready for implementation in real downloader
**Last Verified**: Current session
**Next Step**: Implement test-driven download validation