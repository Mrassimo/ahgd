# Australian Health Data Analytics - Data Download Summary

**Project**: Australian Health Data Analytics  
**Date**: 2025-06-17  
**Focus**: NSW data for manageable initial analysis  
**Total Downloaded**: 1.17 GB (863.23 MB + 311.87 MB)  

## ✅ Successfully Downloaded Data Sources

### 1. **Census 2021 Demographic Data**
- **Source**: Australian Bureau of Statistics (ABS) Census DataPacks
- **Files Downloaded**:
  - `2021_GCP_NSW_SA2.zip` (182.88 MB) - NSW Statistical Area Level 2 data
  - `2021_GCP_AUS_SA2.zip` (583.30 MB) - Complete Australia SA2 data (for reference)
- **Content**: 
  - NSW: 1,914 files including CSV data tables and metadata
  - Australia: 2,033 files with complete national coverage
  - **Key Variables**: Population, age, income, education, employment, housing characteristics
  - **Geography**: Statistical Area Level 2 (SA2) - optimal for health analysis
  - **License**: Creative Commons Attribution 4.0 International

### 2. **SEIFA 2021 Socio-Economic Indexes**
- **Source**: Australian Bureau of Statistics (ABS)
- **File**: `SEIFA_2021_SA2.xlsx` (1.26 MB)
- **Content**: 
  - **Four Indexes**: IRSAD, IRSD, IER, IEO (socio-economic advantage/disadvantage)
  - **Geographic Coverage**: All Australian SA2s (~2,300+ areas)
  - **Data Format**: Rankings, deciles, percentiles by SA2
  - **Use Cases**: Social determinant analysis, health equity mapping

### 3. **Geographic Boundaries (Shapefiles)**
- **Source**: Australian Bureau of Statistics (ABS) ASGS Edition 3
- **Files Downloaded**:
  - `SA2_2021_AUST_SHP_GDA2020.zip` (48.33 MB) - Modern coordinate system
  - `SA2_2021_AUST_SHP_GDA94.zip` (47.46 MB) - Legacy coordinate system
- **Content**: 
  - **Format**: ESRI Shapefile (5 files each: .shp, .dbf, .prj, .shx, .xml)
  - **Main Shapefile**: 67.26 MB containing all SA2 polygon geometries
  - **Attribute Table**: 1.47 MB with SA2 codes, names, and metadata
  - **Coverage**: Complete Australian SA2 boundaries for mapping and spatial analysis

### 4. **Health Data**
- **Source**: data.gov.au / Department of Health
- **Files Downloaded**:

#### Medicare Benefits Schedule (MBS) Data
- **File**: `mbs_demographics_historical_1993_2015.zip` (196.35 MB)
- **Content**: 
  - **90 quarterly files** (1993 Q3 - 2015 Q4)
  - **Coverage**: Patient demographics and service usage patterns
  - **Privacy**: Aggregated data only, no individual patient records
  - **File Size**: ~12 MB per quarterly file
  - **Use Cases**: Healthcare utilisation analysis, service demand patterns

#### Pharmaceutical Benefits Scheme (PBS) Data
- **Files**: 
  - `pbs_historical_1992_2014.zip` (89.69 MB) - Historical pharmaceutical data
  - `pbs_current_2016.csv` (25.83 MB) - Current year pharmaceutical usage
- **Content**:
  - **Historical**: 23 annual files (1992-2014), ~25 MB each
  - **Current**: State-level data with item numbers, schemes, patient categories
  - **Variables**: Year, Item_number, State, Scheme, Month, Patient_Category, Services, Benefits ($)
  - **Use Cases**: Pharmaceutical usage patterns, cost analysis, medication access

## 📊 Data Quality Verification

All files have been verified for:
- ✅ **Completeness**: All downloads completed successfully
- ✅ **Integrity**: ZIP files tested and verified as uncorrupted
- ✅ **Size Validation**: File sizes match expected values
- ✅ **Format Verification**: File formats readable and consistent

## 🗂️ Directory Structure

```
/australian-health-analytics/data/raw/
├── demographics/
│   ├── 2021_GCP_NSW_SA2.zip          (182.88 MB)
│   └── 2021_GCP_AUS_SA2.zip          (583.30 MB)
├── socioeconomic/
│   └── SEIFA_2021_SA2.xlsx           (1.26 MB)
├── geographic/
│   ├── SA2_2021_AUST_SHP_GDA2020.zip (48.33 MB)
│   └── SA2_2021_AUST_SHP_GDA94.zip   (47.46 MB)
├── health/
│   ├── mbs_demographics_historical_1993_2015.zip (196.35 MB)
│   ├── pbs_historical_1992_2014.zip  (89.69 MB)
│   └── pbs_current_2016.csv          (25.83 MB)
└── download_report_*.md              (Detailed reports)
```

## 🔍 Data Characteristics

### **Geographic Coverage**
- **Primary Focus**: New South Wales (NSW) for initial analysis
- **Complete Coverage**: All Australian SA2s available for expansion
- **SA2 Areas**: Optimal balance of geographic detail and data availability
- **Population**: SA2s typically contain 3,000-25,000 people

### **Temporal Coverage**
- **Census Data**: 2021 (most recent available)
- **SEIFA Data**: 2021 (aligned with Census)
- **Health Data**: 
  - MBS: 1993-2015 (22 years of quarterly data)
  - PBS: 1992-2016 (24 years of annual + current data)

### **Data Volume**
- **Total Size**: 1.17 GB raw data
- **Estimated Extracted**: ~2.5 GB when all ZIP files are extracted
- **Record Count**: Millions of health service records, thousands of geographic areas

## 📋 Immediate Next Steps

### 1. **Data Extraction and Processing**
```bash
# Extract key files for NSW analysis
cd data/raw/demographics
unzip 2021_GCP_NSW_SA2.zip

cd ../geographic  
unzip SA2_2021_AUST_SHP_GDA2020.zip

cd ../health
unzip mbs_demographics_historical_1993_2015.zip
```

### 2. **Initial Data Exploration**
- Load NSW Census data into Polars/Pandas for exploration
- Examine SEIFA indexes for NSW SA2s
- Load SA2 shapefiles into GeoPandas for mapping
- Sample MBS/PBS data to understand structure

### 3. **Data Integration**
- Create SA2 concordance table linking all datasets
- Join demographic and socio-economic data
- Map health service utilisation to geographic areas
- Identify NSW-specific SA2s for focused analysis

## 🛠️ Technical Implementation

### **Recommended Tools**
- **Data Processing**: Polars (10x faster than Pandas)
- **Database**: DuckDB (embedded analytics database)
- **Geographic**: GeoPandas + Folium for mapping
- **Analysis**: Jupyter notebooks or Quarto documents

### **Performance Considerations**
- NSW Census data: ~300 SA2s (manageable subset)
- Use lazy loading for large files
- Consider parquet format for processed data
- Implement incremental processing for historical health data

## 🔗 Data Sources URLs

### **Census 2021 DataPacks**
- NSW SA2: `https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_all_for_NSW_short-header.zip`
- Australia SA2: `https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_all_for_AUS_short-header.zip`

### **SEIFA 2021**
- SA2 Level: `https://www.abs.gov.au/statistics/people/people-and-communities/socio-economic-indexes-areas-seifa-australia/2021/Statistical%20Area%20Level%202%2C%20Indexes%2C%20SEIFA%202021.xlsx`

### **Geographic Boundaries**
- GDA2020: `https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_SHP_GDA2020.zip`
- GDA94: `https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_SHP_GDA94.zip`

### **Health Data (data.gov.au)**
- MBS Historical: `https://data.gov.au/data/dataset/8a19a28f-35b0-4035-8cd5-5b611b3cfa6f/resource/519b55ab-8f81-47d1-a483-8495668e38d8/download/mbs-demographics-historical-1993-2015.zip`
- PBS Historical: `https://data.gov.au/data/dataset/14b536d4-eb6a-485d-bf87-2e6e77ddbac1/resource/56f87bbb-a7cb-4cbf-a723-7aec22996eee/download/csv-pbs-item-historical-1992-2014.zip`
- PBS Current: `https://data.gov.au/data/dataset/14b536d4-eb6a-485d-bf87-2e6e77ddbac1/resource/08eda5ab-01c0-4c94-8b1a-157bcffe80d3/download/pbs-item-2016csvjuly.csv`

## 📈 Project Status

- ✅ **Research Complete**: All data sources identified and verified
- ✅ **Downloads Complete**: All NSW-focused data successfully downloaded
- ✅ **Quality Verified**: File integrity and structure confirmed
- ✅ **Documentation**: Comprehensive documentation created
- 🔄 **Next Phase**: Data extraction and initial processing
- 🔄 **Development**: Set up data processing pipeline

## 🎯 Analysis Opportunities

### **Population Health Analysis**
- Demographic health risk profiling by SA2
- Socio-economic health disparities mapping
- Healthcare service accessibility analysis

### **Geographic Health Insights**
- Health service utilisation hotspots
- Underserved area identification
- Provider distribution analysis

### **Temporal Health Trends**
- Healthcare utilisation trends (1993-2015)
- Pharmaceutical usage evolution (1992-2016)
- Demographic transition impacts

---

**Last Updated**: 2025-06-17 20:45 AEST  
**Next Review**: After data extraction and initial processing  
**Contact**: Australian Health Data Analytics Project Team