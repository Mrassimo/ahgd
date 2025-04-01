# Census Data Structures

This document captures the detailed structure of the ABS Census data tables that we're processing in the ETL pipeline, based on analysis of the metadata files.

## Metadata Sources

We've analysed the following metadata files from the ABS:

1. **Metadata_2021_GCP_DataPack_R1_R2.xlsx**
   - Contains detailed descriptions of all Census tables
   - Includes table numbers, names, and population segments
   - Documents cell descriptors for each table

2. **2021_GCP_Sequential_Template_R2.xlsx**
   - Contains templates for each Census table
   - Shows the exact structure of data for each table
   - Includes explanatory notes and definitions

3. **2021Census_geog_desc_1st_2nd_3rd_release.xlsx**
   - Documents the geographic structures used in the Census
   - Details ASGS (Australian Statistical Geography Standard) codes
   - Provides information about geographic coverage

## Census Tables Structure

### G17: Total Personal Income (Weekly) by Age by Sex

**Official Title:** Total Personal Income (Weekly) by Age by Sex

**Structure:**
- Data is organized by income brackets, age groups, and sex
- Each cell represents a count of persons in that category

**Column Pattern:** `{Gender}_{AgeGroup}_{IncomeGroup}`

**Example Columns:**
- `M_15_19_Neg_Nil` (Males aged 15-19 with negative/nil income)
- `F_25_34_1000_1249` (Females aged 25-34 with $1,000-$1,249 weekly income)
- `P_Tot_Tot` (Total persons across all age groups and income brackets)

**Implementation Notes:**
- Income is reported in weekly amounts
- Age groups follow standard ABS groupings
- We process these into standardized column names with the prefix `g17_`

### G18: Core Activity Need for Assistance by Age by Sex

**Official Title:** Core Activity Need for Assistance by Age by Sex

**Structure:**
- Data shows whether individuals need assistance with core activities
- Categories include: has need, no need, not stated
- Broken down by age groups and sex

**Column Pattern:** `{Gender}_{AgeGroup}_{AssistanceStatus}`

**Example Columns:**
- `M_0_4_Need_for_assistance` (Males aged 0-4 who need assistance)
- `F_25_34_No_need_for_assistnce` (Females aged 25-34 who don't need assistance)
- `P_Tot_Need_for_assistance_ns` (Total persons with need for assistance not stated)

**Implementation Notes:**
- Column names contain typos in the original data (e.g., "assistnce" instead of "assistance")
- Our code handles these inconsistencies through flexible column name matching
- We standardize output column names with the prefix `g18_`

### G19: Type of Long-Term Health Condition by Age by Sex

**Official Title:** Type of Long-Term Health Condition by Age by Sex

**Structure:**
- Data shows counts of various health conditions
- Broken down by condition type, age groups, and sex
- Data is split across multiple files (G19A, G19B, G19C) due to large number of columns

**Column Pattern:** `{Gender}_{ConditionType}_{AgeGroup}`

**Example Columns:**
- `M_Arthritis_25_34` (Males aged 25-34 with arthritis)
- `F_Diabetes_75_84` (Females aged 75-84 with diabetes)
- `P_Mental_health_cond_Tot` (Total persons with mental health conditions)

**Health Conditions:**
- Arthritis
- Asthma
- Cancer
- Dementia
- Diabetes
- Heart disease
- Kidney disease
- Lung conditions
- Mental health conditions
- Stroke
- Other
- None (people with no conditions)
- Not stated

**Implementation Notes:**
- Our code handles the split across multiple files by combining them
- We standardize output column names with the prefix `g19_`

### G21: Type of Long-Term Health Condition by Selected Person Characteristics

**Official Title:** Type of Long-Term Health Condition by Selected Person Characteristics

**Structure:**
- Data shows counts of health conditions across different demographic characteristics
- Instead of age groups (as in G19), this table breaks down by country of birth, income, and labour force status
- Data is split across multiple files (G21A, G21B, etc.) due to large number of columns

**Column Pattern:** `{CharacteristicGroup}_{SubCategory}_{ConditionType}`

**Example Columns:**
- `COB_Aus_Arth` (Australian-born people with arthritis)
- `COB_Bo_SE_Asia_Asth` (Southeast Asia-born people with asthma)
- `LFS_Emp_Stroke` (Employed people who had a stroke)

**Health Conditions:**
- Arthritis (Arth)
- Asthma (Asth)
- Cancer (Can_rem)
- Dementia/Alzheimer's (Dem_Alzh)
- Diabetes (Dia_ges_dia)
- Heart disease (HD_HA_ang)
- Kidney disease (Kid_dis)
- Lung conditions (LC_COPD_emph)
- Mental health conditions (MHC_Dep_anx)
- Stroke
- Other conditions (oth_LTHC)
- No conditions (no_LTHC)
- Not stated (LTHC_NS)

**Person Characteristics:**
- Country of Birth (COB):
  - Australia (Aus)
  - Other Oceania and Antarctica (Bo_Ocea_Ant)
  - United Kingdom, Channel Islands and Isle of Man (Bo_UK_CI_IM)
  - Other North-West Europe (Bo_NW_Eu)
  - Southern and Eastern Europe (Bo_SE_Eu)
  - North Africa and the Middle East (Bo_NA_ME)
  - South-East Asia (Bo_SE_Asia)
  - North-East Asia (Bo_NE_Asia)
  - Southern and Central Asia (Bo_SC_Asia)
  - Americas (Bo_Amer)
  - Sub-Saharan Africa (Bo_SS_Afr)
  - Total overseas born (Bo_Tot_ob)
  - Country of birth not stated (COB_NS)
  - Total (Tot)

- Labour Force Status (LFS):
  - Employed (Emp)
  - Unemployed
  - Not in the labour force
  - Labour force status not stated

- Income brackets (weekly):
  - Negative/Nil
  - $1-$299
  - $300-$649
  - $650-$999
  - $1,000-$1,749
  - $1,750-$2,999
  - $3,000 or more
  - Income not stated

**Implementation Notes:**
- Column abbreviations are extensive and require careful mapping
- Our code standardizes output column names with the prefix `g21_`
- The column name structure combines characteristic groups (e.g., COB), subcategories (e.g., Aus), and condition types (e.g., Arth)

## G19 Detailed Processing

### Overview
The G19 detailed data contains health condition information broken down by geographic region, sex, age group, and specific health condition. The data is split across multiple files (G19A, G19B, G19C) covering different conditions and demographics.

### Input Files
- `2021Census_G19A_*.csv` - Contains male (M_) health condition data 
- `2021Census_G19B_*.csv` - Contains female (F_) health condition data
- `2021Census_G19C_*.csv` - Contains person (P_) not stated (NS_) data

### Geographic Levels
The G19 detailed data is available at multiple geographic levels, each with its own column naming pattern:
- STATE level: `STE_CODE_2021`
- SA1 level: `SA1_CODE_2021`
- SA2 level: `SA2_CODE_2021` 
- SA3 level: `SA3_CODE_2021`
- SA4 level: `SA4_CODE_2021`
- LGA level: `LGA_CODE_2021`
- Australia level: `AUS_CODE_2021`
- And others (POA, SUA, GCCSA, etc.)

### Column Structure
The G19 detailed files have a complex column structure with patterns like:
- `M_Arthritis_0_14` - Male, Arthritis condition, age group 0-14 years
- `F_Asthma_65_74` - Female, Asthma condition, age group 65-74 years
- `P_NS_25_34` - Person, Not Stated, age group 25-34 years

### Data Processing
The processing of G19 detailed data involves:
1. Extracting files from ZIP archives using regex patterns
2. Detecting the appropriate geographic code column for each file
3. Unpivoting the wide-format data to a long-format with columns for:
   - Geographic code
   - Sex (M, F, P_NS)
   - Health condition
   - Age group
   - Count
4. Ensuring consistent data types across all files
5. Concatenating data from multiple files
6. Joining with the geographic dimension to get geo_sk

### Output Structure
The resulting `fact_health_conditions_detailed.parquet` contains:
- geo_sk (Int64) - Surrogate key from geo_dimension
- time_sk (Int64) - Surrogate key from time_dimension
- geo_code (String) - Geographic region code
- sex (String) - M (Male), F (Female), or P_NS (Person Not Stated)
- condition (String) - Health condition name (Arthritis, Asthma, etc.)
- age_group (String) - Age group (0_14, 15_24, etc.)
- count (Int64) - Count of people with the condition

## Future Dimensional Model
The G19 detailed data should be refined into a proper dimensional model similar to G20, with:
- geo_sk (Int64) - From geo_dimension
- time_sk (Int64) - From time_dimension
- condition_sk (Int64) - From dim_health_condition
- demographic_sk (Int64) - From dim_demographic
- count (Int64) - Count value

This refinement would allow for consistent querying across G19 and G20 data using surrogate keys.

## Column Naming Conventions and Challenges

The ABS Census data exhibits inconsistencies in column naming that require special handling:

### Gender Codes
- `M` = Male
- `F` = Female
- `P` = Persons (total)

### Age Group Formats
- Standard format: `0_4`, `5_14`, `15_24`, etc.
- Sometimes with "yrs": `0_4_yrs`, `85_yrs_over`
- Abbreviated in some cases: `85_over`

### Inconsistencies and Typos
- Inconsistent abbreviations (e.g., "assistance" vs "assistnce")
- Spelling errors in some column names
- Different separators or formatting for the same concepts

### Our Approach to Handling These Challenges
- Implement flexible column name matching with alternative patterns
- Check for multiple possible column name formats
- Standardize output column names for consistency
- Document known issues and patterns in the code

## Standardized Output Structure

Our ETL pipeline standardizes the output structure as follows:

### G17 Output
- Column format: `g17_{gender}_{age_group}_{income_bracket}`
- Example: `g17_m_25_34_1000_1249`

### G18 Output
- Column format: `g18_{gender}_{age_group}_{assistance_status}`
- Example: `g18_f_65_74_has_need`

### G19 Output
- Column format: `g19_{gender}_{condition_type}_{age_group}`
- Example: `g19_p_arthritis_45_54`

This standardization ensures consistency across our data warehouse and simplifies downstream analysis. 