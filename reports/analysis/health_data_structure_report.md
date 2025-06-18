# Health Data Structure Report

## Executive Summary

This report examines the structure and integration possibilities of the Medicare Benefits Schedule (MBS) and Pharmaceutical Benefits Scheme (PBS) health data downloaded for the Australian Health Geographic Demographics (AHGD) project.

**Key Findings:**
- Both datasets use **state-level geographic identifiers only** (no postcode or SA2 level data)
- High data quality with consistent structures across all years
- Significant volume: ~90 quarterly MBS files and 24+ annual PBS files
- Rich demographic segmentation in MBS, extensive pharmaceutical categorisation in PBS
- Data spans 1992-2016, providing substantial temporal coverage

## MBS Data Structure Analysis

### Data Files
- **Location**: `/Users/massimoraso/AHGD/data/raw/health/mbs_demographics_historical_1993_2015.zip`
- **Format**: ZIP archive containing 90 quarterly CSV files
- **Temporal Coverage**: Q3 1993 to Q4 2015 (22+ years)
- **File Size**: 1.6GB compressed, estimated 15-20GB uncompressed

### File Structure
```
MBS Demographics YYYY Qtr# (Month).csv
```

**Example**: `MBS Demographics 2015 Qtr4 (December).csv`

### Data Schema
| Field | Type | Description | Sample Values |
|-------|------|-------------|---------------|
| Year | Integer | Calendar year | 2015 |
| Month | String | Month name | October, November, December |
| Item_Number | Integer | MBS service item code | 3, 4, 20, 23, 24... |
| State | String | Australian state/territory | ACT, NSW, NT, QLD, SA, TAS, VIC, WA |
| Age_Range | String | Patient age category | "0-4", "5-14", "15-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84", ">=85" |
| Gender | Character | Patient gender | F, M |
| Services | Integer | Number of services provided | Numeric count |
| Benefit | Integer | Benefit amount in cents | Numeric amount |

### Sample Data
```csv
Year,Month,Item_Number,State,Age_Range,Gender,Services,Benefit
2015,October,3,ACT,"0-4",F,61,1129
2015,October,3,ACT,"0-4",M,67,1180
2015,October,3,ACT,"5-14",F,43,729
```

### Geographic Coverage
- **8 states/territories**: ACT, NSW, NT, QLD, SA, TAS, VIC, WA
- **No sub-state geography**: Data aggregated at state level only
- **Complete coverage**: All Australian jurisdictions included

### Data Volume
- **Typical quarterly file**: ~500,000-600,000 records
- **Total estimated records**: ~50 million across all files
- **Demographic segments**: 10 age ranges × 2 genders × 8 states = 160 base combinations per item

## PBS Data Structure Analysis

### Data Files
1. **Historical**: `/Users/massimoraso/AHGD/data/raw/health/pbs_historical_1992_2014.zip`
   - 23 annual CSV files (1992-2014)
   - 781MB compressed
2. **Current**: `/Users/massimoraso/AHGD/data/raw/health/pbs_current_2016.csv`
   - Single file for 2016
   - ~492,000 records

### Data Schema
| Field | Type | Description | Sample Values |
|-------|------|-------------|---------------|
| Year | Integer | Calendar year | 2014, 2016 |
| Item_Number | String | PBS item code | "00", "13Q", "1234A" |
| State | String | Australian state/territory | ACT, NSW, NT, QLD, SA, TAS, VIC, WA |
| Scheme | String | Benefit scheme type | PBS, RPBS |
| Month | String | Month name | January, February, etc. |
| Patient_Category | String | Patient benefit category | See categories below |
| Services | Integer | Number of prescriptions | Numeric count |
| Benefits ($) | Integer | Benefit amount in dollars | Numeric amount |

### Patient Categories
- **General - Ordinary**: Standard patients, regular co-payment
- **General - Safety Net**: General patients who've reached safety net threshold
- **Concessional - Ordinary**: Pensioners/low income, reduced co-payment
- **Concessional - Free Safety Net**: Concessional patients, free medicines
- **RPBS - Ordinary**: Repatriation scheme, regular co-payment
- **RPBS - Safety Net**: Repatriation scheme, safety net threshold reached

### Sample Data
```csv
Year,Item_Number,State,Scheme,Month,Patient_Category,Services,Benefits ($)
2016,0,ACT,RPBS,January,RPBS - Ordinary,13,225
2016,13Q,ACT,PBS,January,General - Ordinary,23,895
```

### Geographic Coverage
- **8 states/territories**: Same as MBS (ACT, NSW, NT, QLD, SA, TAS, VIC, WA)
- **No sub-state geography**: Data aggregated at state level only

### Data Volume
- **Typical annual file**: ~500,000-900,000 records
- **2014 file**: 881,171 records
- **2016 file**: 492,435 records
- **Total estimated**: ~15-20 million records across all years

## Data Quality Assessment

### Strengths
1. **Consistent Structure**: Identical schemas maintained across years
2. **Complete Geographic Coverage**: All Australian states/territories included
3. **Rich Categorisation**: 
   - MBS: Detailed age/gender demographics
   - PBS: Comprehensive scheme/patient categories
4. **Long Time Series**: 22+ years of historical data
5. **Clean Data**: No missing values observed in samples

### Limitations
1. **Geographic Resolution**: State-level only, no postcode or SA2 identifiers
2. **Age Granularity**: MBS uses age ranges, not specific ages
3. **Item Code Complexity**: Thousands of unique MBS/PBS codes requiring reference tables
4. **File Fragmentation**: Data split across many files requiring complex joins

## Integration Assessment

### Geographic Integration Challenges
**Critical Limitation**: Both MBS and PBS datasets contain only **state-level geographic identifiers**. This presents significant challenges for integration with:

- **Postcode-level datasets**: Cannot directly link to postcode boundaries
- **SA2-level analysis**: Cannot map to Statistical Area Level 2 boundaries
- **Local health planning**: Unable to support sub-state geographic analysis

### Potential Integration Approaches

#### 1. State-Level Analysis Only
- **Pros**: Direct integration possible
- **Cons**: Very coarse geographic resolution
- **Use Case**: High-level policy analysis, state comparisons

#### 2. Proportional Allocation Method
- **Approach**: Distribute state totals to postcodes/SA2s based on population weights
- **Pros**: Enables sub-state analysis
- **Cons**: Assumes uniform health service utilisation across geographic areas
- **Risk**: May introduce significant spatial bias

#### 3. Statistical Modelling Approach
- **Approach**: Use demographic covariates to model health service distribution
- **Requirements**: Additional datasets (population, socioeconomic indicators)
- **Complexity**: High, requires advanced spatial statistics

### Data Processing Requirements

#### Volume Considerations
- **Total uncompressed size**: Estimated 25-30GB
- **Memory requirements**: 8-16GB RAM for processing
- **Processing time**: Several hours for full dataset load
- **Storage**: Recommend database solution (SQLite/PostgreSQL)

#### Preprocessing Steps Required
1. **File consolidation**: Merge quarterly/annual files
2. **Data standardisation**: Consistent field naming and types
3. **Item code lookup**: Link to MBS/PBS item descriptions
4. **Date standardisation**: Convert string months to dates
5. **Validation**: Check for data completeness and outliers

## Key Health Metrics Available

### MBS Metrics
- **Service utilisation**: Count of services by specialty
- **Benefit expenditure**: Healthcare spending patterns
- **Demographic patterns**: Age/gender health service usage
- **Temporal trends**: Quarterly changes in utilisation

### PBS Metrics
- **Prescription volume**: Medication dispensing patterns
- **Pharmaceutical expenditure**: Drug spending by category
- **Safety net utilisation**: Access to subsidised medicines
- **Scheme effectiveness**: Comparison between PBS/RPBS outcomes

### Combined Analysis Opportunities
- **Healthcare cost burden**: Total MBS + PBS expenditure
- **Population health indicators**: Service + prescription patterns
- **Policy impact assessment**: Changes following reforms
- **Interstate comparisons**: Health system performance

## Recommendations

### Integration Strategy
1. **Accept state-level limitation**: Focus on state-based health analytics
2. **Complement with postcode data**: Seek additional health datasets with finer geographic resolution
3. **Develop allocation models**: If sub-state analysis essential, develop population-weighted distribution models

### Technical Implementation
1. **Database design**: Implement dimensional model with separate MBS/PBS fact tables
2. **ETL pipeline**: Automate extraction and processing of all historical files
3. **Item code reference**: Maintain lookup tables for MBS/PBS item descriptions
4. **Data validation**: Implement checks for temporal consistency and outliers

### Analysis Priorities
1. **State health profiles**: Comprehensive health service utilisation by jurisdiction
2. **Temporal trend analysis**: Changes in health patterns over 22+ years
3. **Demographic health patterns**: Age/gender variations in service use
4. **Policy impact studies**: Effects of health reforms on utilisation

## Conclusion

The MBS and PBS datasets provide rich, high-quality health data spanning over two decades. While the **state-level geographic limitation** prevents direct integration with postcode or SA2-level analysis, the datasets offer significant value for:

- State-level health system analysis
- Long-term trend identification
- Demographic health pattern analysis
- Healthcare expenditure tracking

**Primary Integration Recommendation**: Proceed with state-level health analytics while seeking complementary datasets with finer geographic resolution for comprehensive health geography analysis.

---

*Report generated: 17 June 2025*
*Data sources: Australian Department of Health MBS and PBS administrative data*