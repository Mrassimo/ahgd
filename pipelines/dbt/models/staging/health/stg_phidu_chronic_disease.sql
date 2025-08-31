{{
  config(
    materialized='table',
    indexes=[
      {'columns': ['sa1_code'], 'type': 'btree'},
      {'columns': ['disease_type'], 'type': 'btree'},
      {'columns': ['pha_code'], 'type': 'btree'}
    ]
  )
}}

WITH source_data AS (
  SELECT * FROM {{ source('health_analytics', 'phidu_chronic_disease') }}
),

validated_chronic_disease AS (
  SELECT
    -- Geographic identifiers
    geographic_code AS sa1_code,
    geographic_name AS sa1_name,
    state_code,
    pha_code,
    pha_name,
    sa2_mapping_percentage,
    
    -- Disease classification
    disease_type,
    disease_description,
    
    -- Prevalence indicators
    prevalence_rate,
    prevalence_count,
    age_standardised_prevalence,
    
    -- Demographics  
    age_group,
    gender,
    
    -- Service utilisation
    gp_visits_per_person,
    specialist_visits_per_person,
    hospitalisation_rate,
    
    -- Risk factors
    risk_factor_score,
    modifiable_risk_factors,
    
    -- Population data
    population_total,
    
    -- Data quality and metadata
    quality_score,
    source_system,
    last_updated,
    
    -- Data validation flags
    CASE WHEN geographic_code ~ '^[0-9]{11}$' THEN 1 ELSE 0 END AS valid_sa1_code,
    CASE WHEN prevalence_rate BETWEEN 0 AND 100 THEN 1 ELSE 0 END AS valid_prevalence_rate,
    CASE WHEN age_standardised_prevalence IS NULL OR age_standardised_prevalence BETWEEN 0 AND 100 THEN 1 ELSE 0 END AS valid_age_std_prevalence,
    CASE WHEN sa2_mapping_percentage BETWEEN 0 AND 100 THEN 1 ELSE 0 END AS valid_mapping_percentage,
    CASE WHEN population_total >= 0 THEN 1 ELSE 0 END AS valid_population,
    CASE WHEN risk_factor_score IS NULL OR risk_factor_score BETWEEN 0 AND 1 THEN 1 ELSE 0 END AS valid_risk_score,
    
    -- Disease burden categories
    CASE 
      WHEN prevalence_rate >= 20.0 THEN 'VERY_HIGH_PREVALENCE'
      WHEN prevalence_rate >= 15.0 THEN 'HIGH_PREVALENCE'
      WHEN prevalence_rate >= 10.0 THEN 'MODERATE_PREVALENCE'
      WHEN prevalence_rate >= 5.0 THEN 'LOW_PREVALENCE'
      ELSE 'VERY_LOW_PREVALENCE'
    END AS prevalence_category,
    
    -- Disease group classifications
    CASE 
      WHEN disease_type IN ('DIABETES', 'CARDIOVASCULAR', 'STROKE') THEN 'METABOLIC_CARDIOVASCULAR'
      WHEN disease_type IN ('CANCER') THEN 'NEOPLASMS'
      WHEN disease_type IN ('MENTAL_HEALTH', 'DEMENTIA') THEN 'MENTAL_NEUROLOGICAL'
      WHEN disease_type IN ('RESPIRATORY') THEN 'RESPIRATORY_DISEASES'
      WHEN disease_type IN ('ARTHRITIS', 'OSTEOPOROSIS') THEN 'MUSCULOSKELETAL'
      WHEN disease_type IN ('KIDNEY_DISEASE') THEN 'RENAL_DISEASES'
      ELSE 'OTHER_CHRONIC'
    END AS disease_group,
    
    -- Age group standardisation
    CASE 
      WHEN age_group IN ('INFANT', 'CHILD', 'ADOLESCENT') THEN 'UNDER_18'
      WHEN age_group IN ('YOUNG_ADULT', 'ADULT') THEN 'ADULT_18_64'
      WHEN age_group IN ('MIDDLE_AGE', 'OLDER_ADULT', 'ELDERLY') THEN 'SENIOR_65_PLUS'
      ELSE age_group
    END AS age_group_broad,
    
    -- Service utilisation burden
    CASE 
      WHEN gp_visits_per_person > 10 THEN 'HIGH_GP_UTILISATION'
      WHEN gp_visits_per_person > 5 THEN 'MODERATE_GP_UTILISATION'
      WHEN gp_visits_per_person > 0 THEN 'LOW_GP_UTILISATION'
      ELSE 'NO_DATA'
    END AS gp_utilisation_category,
    
    CASE 
      WHEN specialist_visits_per_person > 5 THEN 'HIGH_SPECIALIST_UTILISATION'
      WHEN specialist_visits_per_person > 2 THEN 'MODERATE_SPECIALIST_UTILISATION'
      WHEN specialist_visits_per_person > 0 THEN 'LOW_SPECIALIST_UTILISATION'
      ELSE 'NO_DATA'
    END AS specialist_utilisation_category,
    
    -- Calculate estimated affected population
    CASE 
      WHEN prevalence_rate > 0 AND population_total > 0 
      THEN ROUND(population_total * prevalence_rate / 100)
      ELSE prevalence_count
    END AS estimated_affected_population,
    
    -- Risk factor availability
    CASE 
      WHEN modifiable_risk_factors IS NOT NULL AND LENGTH(modifiable_risk_factors) > 0 THEN 1
      ELSE 0
    END AS has_risk_factor_data,
    
    -- Data completeness score (proportion of non-null optional fields)
    (CASE WHEN prevalence_count IS NOT NULL THEN 1 ELSE 0 END +
     CASE WHEN age_standardised_prevalence IS NOT NULL THEN 1 ELSE 0 END +
     CASE WHEN gp_visits_per_person IS NOT NULL THEN 1 ELSE 0 END +
     CASE WHEN specialist_visits_per_person IS NOT NULL THEN 1 ELSE 0 END +
     CASE WHEN hospitalisation_rate IS NOT NULL THEN 1 ELSE 0 END +
     CASE WHEN risk_factor_score IS NOT NULL THEN 1 ELSE 0 END) / 6.0 AS data_completeness_score
    
  FROM source_data
  WHERE quality_score >= 0.5  -- Filter out low-quality records
),

quality_scored AS (
  SELECT *,
    -- Calculate composite data quality score
    CAST((valid_sa1_code + valid_prevalence_rate + valid_age_std_prevalence + 
          valid_mapping_percentage + valid_population + valid_risk_score) AS DECIMAL(3,2)) / 6.0 AS data_quality_score
    
  FROM validated_chronic_disease
)

SELECT * FROM quality_scored
WHERE data_quality_score >= 0.6  -- Only include records with reasonable quality
  AND sa2_mapping_percentage >= 5.0  -- Only include mappings with reasonable coverage
ORDER BY sa1_code, disease_type, age_group