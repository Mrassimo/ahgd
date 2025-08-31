-- AHGD V3: Standardized SA1 Demographics from ABS Census
-- Transforms raw ABS demographic data with data quality validation

{{ config(
    materialized='view',
    tags=['abs', 'demographics', 'staging']
) }}

with raw_demographics as (
    select * from {{ source('abs', 'census_sa1_demographic') }}
),

geography_lookup as (
    select * from {{ source('abs', 'geographic_boundaries_sa1') }}
),

demographic_standardized as (
    select
        -- Primary identifiers
        d.sa1_code,
        upper(trim(coalesce(g.sa1_name, 'Unknown'))) as sa1_name,
        d.sa2_code,
        g.state_code,
        
        -- Population metrics (validated and standardized)
        case 
            when d.total_population between 0 and 10000 then d.total_population
            else null 
        end as total_population,
        
        case 
            when d.median_age between 0 and 120 then d.median_age
            else null 
        end as median_age,
        
        case 
            when d.median_income > 0 then d.median_income
            else null 
        end as median_income_weekly,
        
        coalesce(d.indigenous_population, 0) as indigenous_population_count,
        
        -- Calculate population density
        case 
            when g.area_sqkm > 0 and d.total_population > 0 
            then round(d.total_population / g.area_sqkm, 2)
            else null 
        end as population_density_per_sqkm,
        
        -- Data quality scoring
        case 
            when d.total_population is not null 
                 and d.median_age is not null 
                 and d.median_income is not null 
            then 1.0
            when d.total_population is not null and d.median_age is not null 
            then 0.8
            when d.total_population is not null 
            then 0.6
            else 0.3
        end as data_quality_score,
        
        -- Metadata
        current_timestamp as updated_at,
        '{{ var("current_asgs_year") }}' as asgs_version
        
    from raw_demographics d
    left join geography_lookup g 
        on d.sa1_code = g.sa1_code
),

final as (
    select
        *,
        -- Additional derived metrics
        case 
            when total_population > 0 
            then round(100.0 * indigenous_population_count / total_population, 2)
            else null 
        end as indigenous_population_percentage,
        
        -- Population size categories for analysis
        case 
            when total_population is null then 'Unknown'
            when total_population = 0 then 'No usual residents'
            when total_population between 1 and 50 then 'Very small (1-50)'
            when total_population between 51 and 200 then 'Small (51-200)'  
            when total_population between 201 and 500 then 'Medium (201-500)'
            when total_population between 501 and 1000 then 'Large (501-1000)'
            when total_population > 1000 then 'Very large (1000+)'
        end as population_size_category
        
    from demographic_standardized
    where sa1_code is not null
)

select * from final

-- Data quality checks in comments for visibility:
-- Quality score distribution should be monitored
-- Population totals should sum to known state/national totals  
-- Missing SA1 codes indicate boundary/linkage issues