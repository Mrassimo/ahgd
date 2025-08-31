{{
  config(
    materialized='table',
    indexes=[
      {'columns': ['sa1_code'], 'unique': true},
      {'columns': ['state_code']},
      {'columns': ['irsd_decile_australia']},
      {'columns': ['disadvantage_category']}
    ]
  )
}}

-- Staging model for SA1-level SEIFA socio-economic data
-- Processes and validates SEIFA indexes for 61,845 SA1 areas

WITH source_data AS (
    SELECT
        -- Geographic identifiers
        sa1_code,
        geographic_name,
        state_code,
        state_name,
        population_total,

        -- IRSD - Index of Relative Socio-economic Disadvantage
        irsd_score,
        irsd_rank_australia,
        irsd_decile_australia,
        irsd_percentile_australia,

        -- IRSAD - Index of Relative Socio-economic Advantage and Disadvantage
        irsad_score,
        irsad_rank_australia,
        irsad_decile_australia,
        irsad_percentile_australia,

        -- IER - Index of Education and Occupation
        ier_score,
        ier_rank_australia,
        ier_decile_australia,
        ier_percentile_australia,

        -- IEO - Index of Economic Resources
        ieo_score,
        ieo_rank_australia,
        ieo_decile_australia,
        ieo_percentile_australia,

        -- Data quality from DLT
        complete_indexes_count,
        primary_index_used,
        disadvantage_category,
        has_missing_data,
        validation_errors,
        quality_score,

        -- DLT metadata
        _dlt_load_id,
        _dlt_id

    FROM {{ source('raw_data', 'seifa_sa1') }}
),

data_quality_checks AS (
    SELECT
        *,

        -- Check for minimum population threshold
        CASE
            WHEN population_total >= {{ var('min_population_threshold', 50) }}
            THEN TRUE
            ELSE FALSE
        END AS meets_population_threshold,

        -- Check for at least one complete index
        CASE
            WHEN complete_indexes_count >= 1
            THEN TRUE
            ELSE FALSE
        END AS has_minimum_indexes,

        -- Validate decile ranges
        CASE
            WHEN (irsd_decile_australia IS NULL OR irsd_decile_australia BETWEEN 1 AND 10)
                AND (irsad_decile_australia IS NULL OR irsad_decile_australia BETWEEN 1 AND 10)
                AND (ier_decile_australia IS NULL OR ier_decile_australia BETWEEN 1 AND 10)
                AND (ieo_decile_australia IS NULL OR ieo_decile_australia BETWEEN 1 AND 10)
            THEN TRUE
            ELSE FALSE
        END AS valid_deciles,

        -- Validate percentile ranges
        CASE
            WHEN (irsd_percentile_australia IS NULL OR irsd_percentile_australia BETWEEN 0 AND 100)
                AND (irsad_percentile_australia IS NULL OR irsad_percentile_australia BETWEEN 0 AND 100)
                AND (ier_percentile_australia IS NULL OR ier_percentile_australia BETWEEN 0 AND 100)
                AND (ieo_percentile_australia IS NULL OR ieo_percentile_australia BETWEEN 0 AND 100)
            THEN TRUE
            ELSE FALSE
        END AS valid_percentiles

    FROM source_data
),

imputed_data AS (
    SELECT
        -- Core fields
        sa1_code,
        TRIM(geographic_name) AS sa1_name_clean,
        state_code,

        -- Standardise state names
        CASE state_code
            WHEN '1' THEN 'NSW'
            WHEN '2' THEN 'VIC'
            WHEN '3' THEN 'QLD'
            WHEN '4' THEN 'SA'
            WHEN '5' THEN 'WA'
            WHEN '6' THEN 'TAS'
            WHEN '7' THEN 'NT'
            WHEN '8' THEN 'ACT'
            ELSE 'Unknown'
        END AS state_name_std,

        population_total AS population_seifa,

        -- IRSD Index (primary disadvantage indicator)
        irsd_score,
        irsd_rank_australia,
        irsd_decile_australia,
        irsd_percentile_australia,

        -- Calculate quintiles for simplified analysis
        CASE
            WHEN irsd_decile_australia IN (1, 2) THEN 1
            WHEN irsd_decile_australia IN (3, 4) THEN 2
            WHEN irsd_decile_australia IN (5, 6) THEN 3
            WHEN irsd_decile_australia IN (7, 8) THEN 4
            WHEN irsd_decile_australia IN (9, 10) THEN 5
            ELSE NULL
        END AS irsd_quintile_australia,

        -- IRSAD Index
        irsad_score,
        irsad_rank_australia,
        irsad_decile_australia,
        irsad_percentile_australia,

        -- IER Index
        ier_score,
        ier_rank_australia,
        ier_decile_australia,
        ier_percentile_australia,

        -- IEO Index
        ieo_score,
        ieo_rank_australia,
        ieo_decile_australia,
        ieo_percentile_australia,

        -- Composite disadvantage scoring
        COALESCE(
            disadvantage_category,
            CASE
                WHEN irsd_decile_australia <= 2 THEN 'very_high'
                WHEN irsd_decile_australia <= 4 THEN 'high'
                WHEN irsd_decile_australia <= 6 THEN 'moderate'
                WHEN irsd_decile_australia <= 8 THEN 'low'
                WHEN irsd_decile_australia >= 9 THEN 'very_low'
                ELSE 'unknown'
            END
        ) AS disadvantage_category,

        -- Calculate composite advantage score (0-1 scale)
        CAST(
            (
                COALESCE(irsd_percentile_australia, 50) * 0.4 +
                COALESCE(irsad_percentile_australia, 50) * 0.3 +
                COALESCE(ier_percentile_australia, 50) * 0.2 +
                COALESCE(ieo_percentile_australia, 50) * 0.1
            ) / 100.0
        AS DECIMAL(5,4)) AS composite_advantage_score,

        -- Data quality
        complete_indexes_count,
        primary_index_used,
        meets_population_threshold,
        has_minimum_indexes,
        valid_deciles,
        valid_percentiles,

        -- Overall quality score
        CAST(
            (meets_population_threshold::INT +
             has_minimum_indexes::INT +
             valid_deciles::INT +
             valid_percentiles::INT +
             (complete_indexes_count / 4.0)) / 5.0
        AS DECIMAL(3,2)) AS data_quality_score,

        -- Metadata
        CURRENT_TIMESTAMP AS dbt_processed_at,
        '{{ var("pipeline_version", "1.0.0") }}' AS pipeline_version,
        _dlt_load_id,
        _dlt_id

    FROM data_quality_checks
)

SELECT
    *,

    -- Additional categorisations
    CASE
        WHEN composite_advantage_score < 0.2 THEN 'very_disadvantaged'
        WHEN composite_advantage_score < 0.4 THEN 'disadvantaged'
        WHEN composite_advantage_score < 0.6 THEN 'moderate'
        WHEN composite_advantage_score < 0.8 THEN 'advantaged'
        ELSE 'very_advantaged'
    END AS advantage_category,

    -- Flag areas needing special attention
    CASE
        WHEN irsd_decile_australia <= 3
            AND population_seifa > 500
        THEN TRUE
        ELSE FALSE
    END AS priority_intervention_area,

    -- Research cohort flags
    CASE
        WHEN complete_indexes_count = 4
            AND population_seifa >= 200
        THEN TRUE
        ELSE FALSE
    END AS suitable_for_research

FROM imputed_data
WHERE has_minimum_indexes = TRUE  -- Must have at least one SEIFA index
