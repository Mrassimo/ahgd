{{
  config(
    materialized='incremental',
    unique_key='sa1_code',
    on_schema_change='fail',
    indexes=[
      {'columns': ['sa1_code'], 'unique': true},
      {'columns': ['sa2_code']},
      {'columns': ['state_code']}
    ]
  )
}}

-- Staging model for SA1 boundary data
-- Cleans and standardises 61,845 SA1 areas from raw DLT-loaded data

WITH source_data AS (
    SELECT
        -- Primary identifiers
        sa1_code,
        sa1_name,
        
        -- Hierarchical relationships
        sa2_code,
        sa3_code,
        sa3_name,
        sa4_code,
        sa4_name,
        
        -- State/territory
        state_code,
        state_name,
        
        -- Geographic measurements
        area_sqkm,
        centroid_longitude,
        centroid_latitude,
        
        -- Geometry (store as WKT for compatibility)
        geometry_wkt,
        
        -- Change tracking
        change_flag,
        change_label,
        
        -- Data quality flags from DLT
        COALESCE(has_missing_data, FALSE) AS has_missing_data,
        validation_errors,
        
        -- DLT metadata
        _dlt_load_id,
        _dlt_id
        
    FROM {{ source('raw_data', 'sa1_boundaries') }}
    
    {% if is_incremental() %}
        -- Only process new or updated records
        WHERE _dlt_load_id > (SELECT MAX(_dlt_load_id) FROM {{ this }})
    {% endif %}
),

data_quality_checks AS (
    SELECT
        *,
        
        -- Validate SA1 code format (11 digits starting with state code 1-8)
        CASE 
            WHEN LENGTH(sa1_code) = 11 
                AND REGEXP_MATCHES(sa1_code, '^[1-8][0-9]{10}$')
            THEN TRUE
            ELSE FALSE
        END AS valid_sa1_code,
        
        -- Validate SA2 parent code matches
        CASE
            WHEN SUBSTR(sa1_code, 1, 9) = sa2_code
            THEN TRUE
            ELSE FALSE
        END AS valid_sa2_relationship,
        
        -- Check for valid area
        CASE
            WHEN area_sqkm > 0 AND area_sqkm < 100000  -- Max reasonable area
            THEN TRUE
            ELSE FALSE
        END AS valid_area,
        
        -- Check for valid coordinates (Australia bounds)
        CASE
            WHEN centroid_longitude BETWEEN 112 AND 154
                AND centroid_latitude BETWEEN -44 AND -10
            THEN TRUE
            ELSE FALSE
        END AS valid_coordinates
        
    FROM source_data
),

cleaned_data AS (
    SELECT
        -- Core identifiers
        sa1_code,
        TRIM(sa1_name) AS sa1_name_clean,
        
        -- Hierarchical codes
        sa2_code,
        sa3_code,
        TRIM(sa3_name) AS sa3_name_clean,
        sa4_code,
        TRIM(sa4_name) AS sa4_name_clean,
        
        -- Standardise state names
        state_code,
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
        
        -- Geographic measurements
        ROUND(area_sqkm, 2) AS area_sqkm,
        ROUND(centroid_longitude, 6) AS centroid_longitude,
        ROUND(centroid_latitude, 6) AS centroid_latitude,
        
        -- Geometry
        geometry_wkt,
        
        -- Change tracking
        change_flag,
        change_label,
        
        -- Data quality scoring
        CAST(
            (valid_sa1_code::INT + 
             valid_sa2_relationship::INT + 
             valid_area::INT + 
             valid_coordinates::INT) / 4.0 
        AS DECIMAL(3,2)) AS data_quality_score,
        
        -- Quality flags
        valid_sa1_code,
        valid_sa2_relationship,
        valid_area,
        valid_coordinates,
        has_missing_data,
        validation_errors,
        
        -- Metadata
        CURRENT_TIMESTAMP AS dbt_processed_at,
        '{{ var("pipeline_version", "1.0.0") }}' AS pipeline_version,
        _dlt_load_id,
        _dlt_id
        
    FROM data_quality_checks
    WHERE valid_sa1_code = TRUE  -- Only keep valid SA1 codes
)

SELECT
    -- All cleaned fields
    *,
    
    -- Additional derived fields
    CASE 
        WHEN data_quality_score >= 0.9 THEN 'excellent'
        WHEN data_quality_score >= 0.7 THEN 'good'
        WHEN data_quality_score >= 0.5 THEN 'fair'
        ELSE 'poor'
    END AS data_quality_category,
    
    -- Flag for simplified geometry needs
    CASE
        WHEN area_sqkm > 1000 THEN TRUE  -- Large rural areas
        ELSE FALSE
    END AS needs_geometry_simplification
    
FROM cleaned_data