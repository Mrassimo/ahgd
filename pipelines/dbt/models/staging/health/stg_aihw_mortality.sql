{{
  config(
    materialized='table',
    indexes=[
      {'columns': ['sa1_code'], 'type': 'btree'},
      {'columns': ['cause_of_death'], 'type': 'btree'},
      {'columns': ['calendar_year'], 'type': 'btree'},
      {'columns': ['data_source'], 'type': 'btree'}
    ]
  )
}}

WITH source_data AS (
  SELECT * FROM {{ source('health_analytics', 'aihw_mortality') }}
),

validated_mortality AS (
  SELECT
    -- Geographic identifiers
    geographic_code AS sa1_code,
    geographic_name AS sa1_name,
    state_code,

    -- Cause classification
    cause_of_death,
    icd_10_code,
    cause_description,

    -- Demographics
    age_group,
    gender,

    -- Mortality indicators
    death_count,
    crude_death_rate,
    age_standardised_rate,
    premature_death_count,
    years_of_life_lost,
    avoidable_death_count,

    -- Time period
    calendar_year,

    -- Data quality and metadata
    quality_score,
    source_system,
    data_source,
    suppression_flag,
    last_updated,

    -- Data validation flags
    CASE WHEN geographic_code ~ '^[0-9]{11}$' THEN 1 ELSE 0 END AS valid_sa1_code,
    CASE WHEN death_count >= 0 THEN 1 ELSE 0 END AS valid_death_count,
    CASE WHEN crude_death_rate IS NULL OR crude_death_rate >= 0 THEN 1 ELSE 0 END AS valid_crude_rate,
    CASE WHEN age_standardised_rate IS NULL OR age_standardised_rate >= 0 THEN 1 ELSE 0 END AS valid_age_std_rate,
    CASE WHEN calendar_year BETWEEN 1900 AND 2030 THEN 1 ELSE 0 END AS valid_calendar_year,
    CASE WHEN icd_10_code IS NULL OR icd_10_code ~ '^[A-Z][0-9]{2}(\.[0-9])?$' THEN 1 ELSE 0 END AS valid_icd_code,

    -- Cause groupings
    CASE
      WHEN cause_of_death IN ('CANCER') THEN 'NEOPLASMS'
      WHEN cause_of_death IN ('CARDIOVASCULAR') THEN 'CIRCULATORY_DISEASES'
      WHEN cause_of_death IN ('RESPIRATORY', 'COPD') THEN 'RESPIRATORY_DISEASES'
      WHEN cause_of_death IN ('DIABETES') THEN 'ENDOCRINE_METABOLIC'
      WHEN cause_of_death IN ('MENTAL_HEALTH', 'SUICIDE', 'DEMENTIA') THEN 'MENTAL_BEHAVIOURAL'
      WHEN cause_of_death IN ('ACCIDENT') THEN 'EXTERNAL_CAUSES'
      WHEN cause_of_death IN ('KIDNEY_DISEASE') THEN 'GENITOURINARY_DISEASES'
      WHEN cause_of_death IN ('LIVER_DISEASE') THEN 'DIGESTIVE_DISEASES'
      ELSE 'OTHER_CAUSES'
    END AS cause_category,

    -- Age group standardisation
    CASE
      WHEN age_group IN ('INFANT', 'CHILD', 'ADOLESCENT') THEN 'UNDER_18'
      WHEN age_group IN ('YOUNG_ADULT', 'ADULT') THEN 'ADULT_18_64'
      WHEN age_group IN ('MIDDLE_AGE', 'OLDER_ADULT', 'ELDERLY') THEN 'SENIOR_65_PLUS'
      ELSE age_group
    END AS age_group_broad,

    -- Mortality burden indicators
    CASE
      WHEN premature_death_count > 0 AND death_count > 0
      THEN CAST(premature_death_count AS DECIMAL(5,2)) / death_count
      ELSE NULL
    END AS premature_death_ratio,

    CASE
      WHEN avoidable_death_count > 0 AND death_count > 0
      THEN CAST(avoidable_death_count AS DECIMAL(5,2)) / death_count
      ELSE NULL
    END AS avoidable_death_ratio,

    -- Calculate years of life lost per death
    CASE
      WHEN years_of_life_lost > 0 AND premature_death_count > 0
      THEN years_of_life_lost / premature_death_count
      ELSE NULL
    END AS avg_yll_per_premature_death,

    -- Time period groupings
    CASE
      WHEN calendar_year BETWEEN 2019 AND 2023 THEN 'RECENT_2019_2023'
      WHEN calendar_year BETWEEN 2014 AND 2018 THEN 'MEDIUM_2014_2018'
      WHEN calendar_year BETWEEN 2009 AND 2013 THEN 'OLDER_2009_2013'
      ELSE 'HISTORICAL_PRE_2009'
    END AS time_period_group,

    -- High mortality flag (above 75th percentile for cause)
    CASE
      WHEN age_standardised_rate > 0 THEN 'CALCULATED'  -- Will be updated in post-processing
      ELSE 'NOT_AVAILABLE'
    END AS mortality_burden_flag

  FROM source_data
  WHERE quality_score >= 0.7  -- Higher quality threshold for mortality data
    AND (suppression_flag IS NULL OR suppression_flag = FALSE)  -- Exclude suppressed data
),

quality_scored AS (
  SELECT *,
    -- Calculate composite data quality score
    CAST((valid_sa1_code + valid_death_count + valid_crude_rate +
          valid_age_std_rate + valid_calendar_year + valid_icd_code) AS DECIMAL(3,2)) / 6.0 AS data_quality_score

  FROM validated_mortality
)

SELECT * FROM quality_scored
WHERE data_quality_score >= 0.7  -- High quality threshold for mortality data
ORDER BY sa1_code, calendar_year, cause_of_death
