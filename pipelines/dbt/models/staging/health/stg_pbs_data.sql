{{
  config(
    materialized='table',
    indexes=[
      {'columns': ['sa1_code'], 'type': 'btree'},
      {'columns': ['pbs_item_code'], 'type': 'btree'},
      {'columns': ['financial_year'], 'type': 'btree'},
      {'columns': ['atc_code'], 'type': 'btree'}
    ]
  )
}}

WITH source_data AS (
  SELECT * FROM {{ source('health_analytics', 'pbs_data') }}
),

validated_pbs AS (
  SELECT
    -- Geographic identifiers
    geographic_code AS sa1_code,
    geographic_name AS sa1_name,
    state_code,

    -- Medicine identification
    pbs_item_code,
    medicine_name,
    brand_name,
    atc_code,
    therapeutic_group,

    -- Demographics
    age_group,
    gender,

    -- Prescription metrics
    prescription_count,
    patient_count,
    ddd_per_1000_population_per_day,

    -- Cost metrics
    government_benefit,
    patient_contribution,
    total_cost,

    -- Time period
    financial_year,
    month,

    -- Data quality and metadata
    quality_score,
    source_system,
    last_updated,

    -- Derived metrics
    CASE
      WHEN patient_count > 0 AND prescription_count > 0
      THEN CAST(prescription_count AS DECIMAL(10,2)) / patient_count
      ELSE NULL
    END AS prescriptions_per_patient,

    CASE
      WHEN prescription_count > 0 AND government_benefit > 0
      THEN government_benefit / prescription_count
      ELSE NULL
    END AS average_government_benefit_per_prescription,

    CASE
      WHEN prescription_count > 0 AND total_cost > 0
      THEN total_cost / prescription_count
      ELSE NULL
    END AS average_total_cost_per_prescription,

    -- Calculate total cost if missing but components available
    CASE
      WHEN total_cost IS NULL AND government_benefit > 0 AND patient_contribution > 0
      THEN government_benefit + patient_contribution
      WHEN total_cost IS NULL AND government_benefit > 0 AND patient_contribution IS NULL
      THEN government_benefit
      ELSE total_cost
    END AS calculated_total_cost,

    -- Data validation flags
    CASE WHEN pbs_item_code ~ '^[0-9]{4}[A-Z]?$' THEN 1 ELSE 0 END AS valid_item_code,
    CASE WHEN geographic_code ~ '^[0-9]{11}$' THEN 1 ELSE 0 END AS valid_sa1_code,
    CASE WHEN prescription_count >= 0 THEN 1 ELSE 0 END AS valid_prescription_count,
    CASE WHEN government_benefit >= 0 THEN 1 ELSE 0 END AS valid_government_benefit,
    CASE WHEN financial_year ~ '^20[0-9]{2}-[0-9]{2}$' THEN 1 ELSE 0 END AS valid_financial_year,
    CASE WHEN atc_code IS NULL OR atc_code ~ '^[A-Z][0-9]{2}[A-Z]{2}[0-9]{2}$' THEN 1 ELSE 0 END AS valid_atc_code,

    -- Therapeutic categorisation from ATC code
    CASE
      WHEN LEFT(atc_code, 1) = 'A' THEN 'ALIMENTARY_TRACT_METABOLISM'
      WHEN LEFT(atc_code, 1) = 'B' THEN 'BLOOD_BLOOD_FORMING_ORGANS'
      WHEN LEFT(atc_code, 1) = 'C' THEN 'CARDIOVASCULAR_SYSTEM'
      WHEN LEFT(atc_code, 1) = 'D' THEN 'DERMATOLOGICALS'
      WHEN LEFT(atc_code, 1) = 'G' THEN 'GENITO_URINARY_REPRODUCTIVE'
      WHEN LEFT(atc_code, 1) = 'H' THEN 'HORMONAL_PREPARATIONS'
      WHEN LEFT(atc_code, 1) = 'J' THEN 'ANTI_INFECTIVES_SYSTEMIC'
      WHEN LEFT(atc_code, 1) = 'L' THEN 'ANTINEOPLASTIC_IMMUNOMODULATING'
      WHEN LEFT(atc_code, 1) = 'M' THEN 'MUSCULO_SKELETAL_SYSTEM'
      WHEN LEFT(atc_code, 1) = 'N' THEN 'NERVOUS_SYSTEM'
      WHEN LEFT(atc_code, 1) = 'P' THEN 'ANTIPARASITIC_PRODUCTS'
      WHEN LEFT(atc_code, 1) = 'R' THEN 'RESPIRATORY_SYSTEM'
      WHEN LEFT(atc_code, 1) = 'S' THEN 'SENSORY_ORGANS'
      WHEN LEFT(atc_code, 1) = 'V' THEN 'VARIOUS'
      ELSE 'UNKNOWN_THERAPEUTIC_GROUP'
    END AS atc_therapeutic_category,

    -- Age group standardisation
    CASE
      WHEN age_group IN ('INFANT', 'CHILD', 'ADOLESCENT') THEN 'UNDER_18'
      WHEN age_group IN ('YOUNG_ADULT', 'ADULT') THEN 'ADULT_18_64'
      WHEN age_group IN ('MIDDLE_AGE', 'OLDER_ADULT', 'ELDERLY') THEN 'SENIOR_65_PLUS'
      ELSE age_group
    END AS age_group_broad,

    -- High-cost medicines flag (top quartile)
    CASE
      WHEN government_benefit > 0 THEN
        CASE
          WHEN government_benefit / prescription_count > 100 THEN 'HIGH_COST'
          WHEN government_benefit / prescription_count > 50 THEN 'MEDIUM_COST'
          ELSE 'LOW_COST'
        END
      ELSE 'UNKNOWN_COST'
    END AS cost_category

  FROM source_data
  WHERE quality_score >= 0.5  -- Filter out low-quality records
),

quality_scored AS (
  SELECT *,
    -- Calculate composite data quality score
    CAST((valid_item_code + valid_sa1_code + valid_prescription_count +
          valid_government_benefit + valid_financial_year + valid_atc_code) AS DECIMAL(3,2)) / 6.0 AS data_quality_score

  FROM validated_pbs
)

SELECT * FROM quality_scored
WHERE data_quality_score >= 0.6  -- Only include records with reasonable quality
ORDER BY sa1_code, financial_year, pbs_item_code
