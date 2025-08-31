{{
  config(
    materialized='table',
    indexes=[
      {'columns': ['sa1_code'], 'type': 'btree'},
      {'columns': ['mbs_item_number'], 'type': 'btree'},
      {'columns': ['financial_year'], 'type': 'btree'},
      {'columns': ['service_type'], 'type': 'btree'}
    ]
  )
}}

WITH source_data AS (
  SELECT * FROM {{ source('health_analytics', 'mbs_data') }}
),

validated_mbs AS (
  SELECT
    -- Geographic identifiers
    geographic_code AS sa1_code,
    geographic_name AS sa1_name,
    state_code,

    -- Service identification
    mbs_item_number,
    mbs_item_description,
    service_type,

    -- Demographics
    age_group,
    gender,

    -- Service utilisation metrics
    service_count,
    patient_count,
    benefit_paid,
    services_per_1000_population,
    patients_per_1000_population,
    average_benefit_per_service,

    -- Time period
    financial_year,
    quarter,

    -- Data quality and metadata
    quality_score,
    source_system,
    last_updated,

    -- Derived metrics
    CASE
      WHEN patient_count > 0 AND service_count > 0
      THEN CAST(service_count AS DECIMAL(10,2)) / patient_count
      ELSE NULL
    END AS services_per_patient,

    CASE
      WHEN service_count > 0 AND benefit_paid > 0
      THEN benefit_paid / service_count
      ELSE NULL
    END AS calculated_benefit_per_service,

    -- Data validation flags
    CASE WHEN mbs_item_number ~ '^[0-9]{1,6}$' THEN 1 ELSE 0 END AS valid_item_number,
    CASE WHEN geographic_code ~ '^[0-9]{11}$' THEN 1 ELSE 0 END AS valid_sa1_code,
    CASE WHEN service_count >= 0 THEN 1 ELSE 0 END AS valid_service_count,
    CASE WHEN benefit_paid >= 0 THEN 1 ELSE 0 END AS valid_benefit_paid,
    CASE WHEN financial_year ~ '^20[0-9]{2}-[0-9]{2}$' THEN 1 ELSE 0 END AS valid_financial_year,

    -- Service categorisation
    CASE
      WHEN service_type IN ('MEDICAL', 'SPECIALIST') THEN 'PRIMARY_CARE'
      WHEN service_type IN ('DIAGNOSTIC', 'PATHOLOGY') THEN 'DIAGNOSTIC_SERVICES'
      WHEN service_type = 'SURGICAL' THEN 'SURGICAL_SERVICES'
      WHEN service_type = 'MENTAL_HEALTH' THEN 'MENTAL_HEALTH_SERVICES'
      ELSE 'OTHER_SERVICES'
    END AS service_category,

    -- Age group standardisation
    CASE
      WHEN age_group IN ('INFANT', 'CHILD', 'ADOLESCENT') THEN 'UNDER_18'
      WHEN age_group IN ('YOUNG_ADULT', 'ADULT') THEN 'ADULT_18_64'
      WHEN age_group IN ('MIDDLE_AGE', 'OLDER_ADULT', 'ELDERLY') THEN 'SENIOR_65_PLUS'
      ELSE age_group
    END AS age_group_broad

  FROM source_data
  WHERE quality_score >= 0.5  -- Filter out low-quality records
),

quality_scored AS (
  SELECT *,
    -- Calculate composite data quality score
    CAST((valid_item_number + valid_sa1_code + valid_service_count +
          valid_benefit_paid + valid_financial_year) AS DECIMAL(3,2)) / 5.0 AS data_quality_score

  FROM validated_mbs
)

SELECT * FROM quality_scored
WHERE data_quality_score >= 0.6  -- Only include records with reasonable quality
ORDER BY sa1_code, financial_year, mbs_item_number
