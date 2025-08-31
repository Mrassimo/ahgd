{{
  config(
    materialized='table',
    indexes=[
      {'columns': ['sa1_code'], 'unique': true},
      {'columns': ['sa2_code']},
      {'columns': ['state_code']}
    ]
  )
}}

-- SA1 to SA2 Bridge Table
-- Provides relationship mappings and aggregation logic between 61,845 SA1s and 2,454 SA2s

WITH sa1_data AS (
    SELECT
        sa1_code,
        sa1_name_clean AS sa1_name,
        sa2_code,
        sa3_code,
        sa4_code,
        state_code,
        state_name_std AS state_name,
        area_sqkm AS sa1_area_sqkm,
        centroid_longitude AS sa1_centroid_lon,
        centroid_latitude AS sa1_centroid_lat,
        data_quality_score AS sa1_quality_score
    FROM {{ ref('stg_sa1_boundaries') }}
),

sa1_seifa AS (
    SELECT
        sa1_code,
        population_seifa AS sa1_population,
        irsd_score,
        irsd_decile_australia,
        disadvantage_category,
        composite_advantage_score
    FROM {{ ref('stg_seifa_sa1') }}
),

sa2_aggregates AS (
    SELECT
        sa2_code,
        COUNT(DISTINCT sa1_code) AS sa1_count,
        SUM(sa1_area_sqkm) AS total_area_sqkm,
        AVG(sa1_centroid_lon) AS sa2_centroid_lon,
        AVG(sa1_centroid_lat) AS sa2_centroid_lat,
        MIN(sa1_quality_score) AS min_quality_score,
        MAX(sa1_quality_score) AS max_quality_score,
        AVG(sa1_quality_score) AS avg_quality_score
    FROM sa1_data
    GROUP BY sa2_code
),

sa2_population AS (
    SELECT
        b.sa2_code,
        SUM(s.sa1_population) AS total_population,
        -- Population-weighted SEIFA scores
        SUM(s.irsd_score * s.sa1_population) / NULLIF(SUM(s.sa1_population), 0) AS weighted_irsd_score,
        -- Mode of disadvantage categories
        MODE() WITHIN GROUP (ORDER BY s.disadvantage_category) AS predominant_disadvantage,
        -- Population-weighted advantage score
        SUM(s.composite_advantage_score * s.sa1_population) / NULLIF(SUM(s.sa1_population), 0) AS weighted_advantage_score
    FROM sa1_data b
    LEFT JOIN sa1_seifa s ON b.sa1_code = s.sa1_code
    GROUP BY b.sa2_code
)

SELECT
    -- SA1 identifiers
    b.sa1_code,
    b.sa1_name,

    -- SA2 identifiers
    b.sa2_code,

    -- Higher level geography
    b.sa3_code,
    b.sa4_code,
    b.state_code,
    b.state_name,

    -- SA1 metrics
    b.sa1_area_sqkm,
    COALESCE(s.sa1_population, 0) AS sa1_population,
    b.sa1_centroid_lon,
    b.sa1_centroid_lat,

    -- SA1 SEIFA data
    s.irsd_score AS sa1_irsd_score,
    s.irsd_decile_australia AS sa1_irsd_decile,
    s.disadvantage_category AS sa1_disadvantage_category,
    s.composite_advantage_score AS sa1_advantage_score,

    -- SA2 aggregate metrics
    a.sa1_count AS sa2_sa1_count,
    a.total_area_sqkm AS sa2_total_area_sqkm,
    p.total_population AS sa2_total_population,

    -- Allocation percentages for aggregation
    -- Area-based allocation
    CAST(b.sa1_area_sqkm / NULLIF(a.total_area_sqkm, 0) * 100 AS DECIMAL(5,2)) AS area_allocation_pct,

    -- Population-based allocation (preferred for health metrics)
    CAST(s.sa1_population / NULLIF(p.total_population, 0) * 100 AS DECIMAL(5,2)) AS population_allocation_pct,

    -- SA2 weighted scores (for validation)
    p.weighted_irsd_score AS sa2_weighted_irsd_score,
    p.predominant_disadvantage AS sa2_predominant_disadvantage,
    p.weighted_advantage_score AS sa2_weighted_advantage_score,

    -- Relationship metadata
    'exact' AS relationship_type,  -- SA1s fully contained in SA2s
    b.sa1_quality_score,
    a.avg_quality_score AS sa2_avg_quality_score,

    -- Processing metadata
    CURRENT_TIMESTAMP AS created_at,
    '{{ var("pipeline_version", "1.0.0") }}' AS pipeline_version

FROM sa1_data b
LEFT JOIN sa1_seifa s ON b.sa1_code = s.sa1_code
LEFT JOIN sa2_aggregates a ON b.sa2_code = a.sa2_code
LEFT JOIN sa2_population p ON b.sa2_code = p.sa2_code
