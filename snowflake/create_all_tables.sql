-- AHGD Snowflake DDL Scripts
-- Australian Healthcare Geographic Database
-- Optimized for Snowflake with clustering keys and proper data types

CREATE DATABASE IF NOT EXISTS AHGD;
USE DATABASE AHGD;
CREATE SCHEMA IF NOT EXISTS PUBLIC;
USE SCHEMA PUBLIC;

-- ============================================
-- Dimension Tables
-- ============================================

-- Geographic Dimension (SCD Type 2 enabled)
CREATE OR REPLACE TABLE dim_geography (
    geo_sk NUMBER(19,0) PRIMARY KEY,
    geo_id VARCHAR(20) NOT NULL,
    geo_level VARCHAR(10) NOT NULL,
    geo_name VARCHAR(255),
    state_code VARCHAR(3),
    state_name VARCHAR(50),
    latitude FLOAT,
    longitude FLOAT,
    parent_geo_sk NUMBER(19,0),
    geom GEOGRAPHY,  -- Native Snowflake geography type
    valid_from DATE NOT NULL DEFAULT CURRENT_DATE(),
    valid_to DATE,
    is_current BOOLEAN DEFAULT TRUE,
    is_unknown BOOLEAN DEFAULT FALSE,
    etl_batch_id NUMBER(19,0),
    etl_processed_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Create index on natural key for lookups
CREATE INDEX idx_geo_id ON dim_geography(geo_id);
CREATE INDEX idx_geo_level ON dim_geography(geo_level);

-- Time Dimension
CREATE OR REPLACE TABLE dim_time (
    time_sk NUMBER(19,0) PRIMARY KEY,  -- Format: YYYYMMDD
    full_date DATE NOT NULL UNIQUE,
    year NUMBER(4),
    quarter NUMBER(1),
    month NUMBER(2),
    month_name VARCHAR(20),
    week_of_year NUMBER(2),
    day_of_month NUMBER(2),
    day_of_week NUMBER(1),
    day_name VARCHAR(20),
    is_weekday BOOLEAN,
    is_weekend BOOLEAN,
    is_holiday BOOLEAN,
    holiday_name VARCHAR(100),
    fiscal_year NUMBER(4),
    fiscal_quarter NUMBER(1),
    is_census_date BOOLEAN DEFAULT FALSE,
    census_year NUMBER(4),
    is_unknown BOOLEAN DEFAULT FALSE,
    etl_processed_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Create index on date for range queries
CREATE INDEX idx_full_date ON dim_time(full_date);
CREATE INDEX idx_year_month ON dim_time(year, month);

-- Health Condition Dimension
CREATE OR REPLACE TABLE dim_health_condition (
    condition_sk VARCHAR(32) PRIMARY KEY,  -- MD5 hash
    condition_code VARCHAR(50) NOT NULL UNIQUE,
    condition_name VARCHAR(255),
    condition_category VARCHAR(100),
    condition_group VARCHAR(100),
    is_chronic BOOLEAN DEFAULT FALSE,
    is_mental_health BOOLEAN DEFAULT FALSE,
    is_disability BOOLEAN DEFAULT FALSE,
    is_unknown BOOLEAN DEFAULT FALSE,
    etl_processed_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Demographic Dimension (Junk Dimension)
CREATE OR REPLACE TABLE dim_demographic (
    demographic_sk VARCHAR(32) PRIMARY KEY,  -- MD5 hash
    age_group VARCHAR(50),
    age_group_code VARCHAR(10),
    sex VARCHAR(20),
    sex_code VARCHAR(1),
    indigenous_status VARCHAR(50),
    is_unknown BOOLEAN DEFAULT FALSE,
    etl_processed_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Person Characteristic Dimension
CREATE OR REPLACE TABLE dim_person_characteristic (
    characteristic_sk VARCHAR(32) PRIMARY KEY,  -- MD5 hash
    characteristic_type VARCHAR(100),
    characteristic_value VARCHAR(255),
    characteristic_code VARCHAR(50),
    characteristic_category VARCHAR(100),
    display_order NUMBER(10),
    is_unknown BOOLEAN DEFAULT FALSE,
    etl_processed_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ============================================
-- Fact Tables
-- ============================================

-- Population Fact (Snapshot)
CREATE OR REPLACE TABLE fact_population (
    geo_sk NUMBER(19,0) NOT NULL REFERENCES dim_geography(geo_sk),
    time_sk NUMBER(19,0) NOT NULL REFERENCES dim_time(time_sk),
    total_population NUMBER(19,0),
    male_population NUMBER(19,0),
    female_population NUMBER(19,0),
    households NUMBER(19,0),
    average_household_size FLOAT,
    population_density FLOAT,
    median_age FLOAT,
    etl_batch_id NUMBER(19,0),
    etl_processed_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (geo_sk, time_sk)
) CLUSTER BY (time_sk, geo_sk);

-- Income Fact
CREATE OR REPLACE TABLE fact_income (
    geo_sk NUMBER(19,0) NOT NULL REFERENCES dim_geography(geo_sk),
    time_sk NUMBER(19,0) NOT NULL REFERENCES dim_time(time_sk),
    demographic_sk VARCHAR(32) NOT NULL REFERENCES dim_demographic(demographic_sk),
    characteristic_sk VARCHAR(32) NOT NULL REFERENCES dim_person_characteristic(characteristic_sk),
    median_income NUMBER(10,2),
    mean_income NUMBER(10,2),
    income_quartile_1 NUMBER(10,2),
    income_quartile_3 NUMBER(10,2),
    gini_coefficient FLOAT,
    person_count NUMBER(19,0),
    household_count NUMBER(19,0),
    etl_batch_id NUMBER(19,0),
    etl_processed_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (geo_sk, time_sk, demographic_sk, characteristic_sk)
) CLUSTER BY (time_sk, geo_sk);

-- Health Conditions Fact
CREATE OR REPLACE TABLE fact_health_condition (
    geo_sk NUMBER(19,0) NOT NULL REFERENCES dim_geography(geo_sk),
    time_sk NUMBER(19,0) NOT NULL REFERENCES dim_time(time_sk),
    condition_sk VARCHAR(32) NOT NULL REFERENCES dim_health_condition(condition_sk),
    demographic_sk VARCHAR(32) NOT NULL REFERENCES dim_demographic(demographic_sk),
    characteristic_sk VARCHAR(32) REFERENCES dim_person_characteristic(characteristic_sk),
    person_count NUMBER(19,0) NOT NULL,
    prevalence_rate FLOAT,
    age_standardized_rate FLOAT,
    confidence_interval_lower FLOAT,
    confidence_interval_upper FLOAT,
    relative_standard_error FLOAT,
    etl_batch_id NUMBER(19,0),
    etl_processed_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
) CLUSTER BY (time_sk, condition_sk, geo_sk);

-- Assistance Needs Fact
CREATE OR REPLACE TABLE fact_assistance_need (
    geo_sk NUMBER(19,0) NOT NULL REFERENCES dim_geography(geo_sk),
    time_sk NUMBER(19,0) NOT NULL REFERENCES dim_time(time_sk),
    demographic_sk VARCHAR(32) NOT NULL REFERENCES dim_demographic(demographic_sk),
    assistance_type VARCHAR(100) NOT NULL,
    assistance_category VARCHAR(50),
    needs_assistance BOOLEAN,
    person_count NUMBER(19,0) NOT NULL,
    percentage_of_population FLOAT,
    severity_level VARCHAR(20),
    etl_batch_id NUMBER(19,0),
    etl_processed_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
) CLUSTER BY (time_sk, geo_sk, assistance_type);

-- Unpaid Assistance Fact
CREATE OR REPLACE TABLE fact_unpaid_assistance (
    geo_sk NUMBER(19,0) NOT NULL REFERENCES dim_geography(geo_sk),
    time_sk NUMBER(19,0) NOT NULL REFERENCES dim_time(time_sk),
    demographic_sk VARCHAR(32) NOT NULL REFERENCES dim_demographic(demographic_sk),
    care_type VARCHAR(100) NOT NULL,
    hours_per_week_category VARCHAR(50),
    provided_care BOOLEAN,
    person_count NUMBER(19,0) NOT NULL,
    percentage_of_population FLOAT,
    estimated_value NUMBER(10,2),
    etl_batch_id NUMBER(19,0),
    etl_processed_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
) CLUSTER BY (time_sk, geo_sk);

-- ============================================
-- Bridge Tables (for many-to-many relationships)
-- ============================================

-- Bridge for multiple conditions per person
CREATE OR REPLACE TABLE bridge_person_conditions (
    person_group_sk NUMBER(19,0) NOT NULL,
    condition_sk VARCHAR(32) NOT NULL REFERENCES dim_health_condition(condition_sk),
    condition_weight FLOAT DEFAULT 1.0,
    condition_rank NUMBER(10),
    is_primary_condition BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (person_group_sk, condition_sk)
);

-- ============================================
-- Control & Metadata Tables
-- ============================================

-- ETL Batch Control
CREATE OR REPLACE TABLE etl_batch_control (
    batch_id NUMBER(19,0) IDENTITY(1,1) PRIMARY KEY,
    batch_name VARCHAR(100),
    batch_type VARCHAR(50),  -- FULL, INCREMENTAL, DIMENSION_ONLY, FACT_ONLY
    source_system VARCHAR(50),
    start_timestamp TIMESTAMP_NTZ,
    end_timestamp TIMESTAMP_NTZ,
    status VARCHAR(20),  -- RUNNING, SUCCESS, FAILED, WARNING
    records_processed NUMBER(19,0),
    records_failed NUMBER(19,0),
    error_message VARCHAR(4000),
    created_by VARCHAR(100) DEFAULT CURRENT_USER()
);

-- Data Quality Metrics
CREATE OR REPLACE TABLE data_quality_metrics (
    check_id NUMBER(19,0) IDENTITY(1,1) PRIMARY KEY,
    batch_id NUMBER(19,0) REFERENCES etl_batch_control(batch_id),
    table_name VARCHAR(100),
    check_type VARCHAR(50),  -- COMPLETENESS, UNIQUENESS, REFERENTIAL, RANGE, PATTERN
    check_name VARCHAR(200),
    check_result VARCHAR(20),  -- PASSED, FAILED, WARNING
    failed_count NUMBER(19,0),
    total_count NUMBER(19,0),
    failure_percentage FLOAT,
    sample_failed_records VARIANT,  -- JSON array of sample failures
    check_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Source to Target Mapping (for lineage)
CREATE OR REPLACE TABLE source_target_mapping (
    mapping_id NUMBER(19,0) IDENTITY(1,1) PRIMARY KEY,
    source_system VARCHAR(50),
    source_table VARCHAR(100),
    source_column VARCHAR(100),
    target_table VARCHAR(100),
    target_column VARCHAR(100),
    transformation_rule VARCHAR(4000),
    is_active BOOLEAN DEFAULT TRUE,
    created_date DATE DEFAULT CURRENT_DATE(),
    modified_date DATE DEFAULT CURRENT_DATE()
);

-- ============================================
-- Views for Common Queries
-- ============================================

-- Current Geography View (for SCD Type 2)
CREATE OR REPLACE VIEW v_current_geography AS
SELECT * FROM dim_geography
WHERE is_current = TRUE;

-- Population Summary by State
CREATE OR REPLACE VIEW v_population_by_state AS
SELECT 
    g.state_name,
    t.year,
    SUM(f.total_population) as total_population,
    SUM(f.male_population) as male_population,
    SUM(f.female_population) as female_population,
    AVG(f.median_age) as avg_median_age
FROM fact_population f
JOIN dim_geography g ON f.geo_sk = g.geo_sk
JOIN dim_time t ON f.time_sk = t.time_sk
WHERE g.is_current = TRUE
GROUP BY g.state_name, t.year;

-- Health Condition Prevalence
CREATE OR REPLACE VIEW v_health_condition_prevalence AS
SELECT 
    hc.condition_name,
    hc.condition_category,
    g.state_name,
    t.year,
    SUM(f.person_count) as affected_persons,
    AVG(f.prevalence_rate) as avg_prevalence_rate
FROM fact_health_condition f
JOIN dim_health_condition hc ON f.condition_sk = hc.condition_sk
JOIN dim_geography g ON f.geo_sk = g.geo_sk
JOIN dim_time t ON f.time_sk = t.time_sk
WHERE g.is_current = TRUE
  AND hc.is_unknown = FALSE
GROUP BY hc.condition_name, hc.condition_category, g.state_name, t.year;

-- ============================================
-- Security: Row Access Policies (Optional)
-- ============================================

-- Example: Restrict data access by state
CREATE OR REPLACE ROW ACCESS POLICY state_access_policy
AS (state_code VARCHAR) RETURNS BOOLEAN ->
  CASE 
    WHEN CURRENT_ROLE() = 'ADMIN' THEN TRUE
    WHEN CURRENT_ROLE() = 'NSW_ANALYST' THEN state_code = 'NSW'
    WHEN CURRENT_ROLE() = 'VIC_ANALYST' THEN state_code = 'VIC'
    ELSE FALSE
  END;

-- Apply policy to geography dimension
-- ALTER TABLE dim_geography ADD ROW ACCESS POLICY state_access_policy ON (state_code);

-- ============================================
-- Performance: Materialized Views (Optional)
-- ============================================

-- Materialized view for frequently accessed aggregations
CREATE OR REPLACE MATERIALIZED VIEW mv_monthly_health_summary
CLUSTER BY (time_sk, geo_sk)
AS
SELECT 
    f.time_sk,
    f.geo_sk,
    f.condition_sk,
    SUM(f.person_count) as total_persons,
    AVG(f.prevalence_rate) as avg_prevalence,
    COUNT(DISTINCT f.demographic_sk) as demographic_groups
FROM fact_health_condition f
GROUP BY f.time_sk, f.geo_sk, f.condition_sk;