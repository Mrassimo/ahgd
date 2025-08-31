-- AHGD V3: SA1 Comprehensive Health Profile
-- Integrated health, demographic, and socioeconomic analytics model

{{ config(
    materialized='table',
    tags=['health', 'analytics', 'core'],
    post_hook="CREATE INDEX IF NOT EXISTS idx_sa1_health_sa1_code ON {{ this }} (sa1_code)"
) }}

with demographics as (
    select * from {{ ref('stg_abs__sa1_demographics') }}
),

geography as (
    select 
        sa1_code,
        sa1_name,
        sa2_code,
        sa3_code,
        sa4_code,
        state_code,
        state_name,
        remoteness_category,
        centroid_longitude,
        centroid_latitude,
        area_sqkm
    from {{ ref('stg_abs__sa1_geography') }}
),

seifa as (
    select * from {{ ref('stg_abs__seifa_indices') }}
),

health_indicators as (
    select * from {{ ref('stg_aihw__health_indicators') }}
    where indicator_year = (
        select max(indicator_year) from {{ ref('stg_aihw__health_indicators') }}
    )
),

medicare_services as (
    select * from {{ ref('stg_medicare__gp_utilisation') }}
    where service_year = (
        select max(service_year) from {{ ref('stg_medicare__gp_utilisation') }}
    )
),

immunisation as (
    select * from {{ ref('stg_medicare__immunisation_rates') }}
    where assessment_year = (
        select max(assessment_year) from {{ ref('stg_medicare__immunisation_rates') }}
    )
),

climate_summary as (
    select 
        sa1_code,
        avg(avg_temperature_c) as avg_annual_temperature_c,
        sum(total_rainfall_mm) as total_annual_rainfall_mm,
        avg(avg_humidity_percent) as avg_annual_humidity_percent,
        sum(heat_wave_days) as total_heat_wave_days,
        sum(extreme_rainfall_events) as total_extreme_rainfall_events
    from {{ ref('stg_bom__climate_sa1') }}
    where climate_year = (
        select max(climate_year) from {{ ref('stg_bom__climate_sa1') }}
    )
    group by sa1_code
),

integrated_profile as (
    select
        -- Geographic identifiers
        g.sa1_code,
        g.sa1_name,
        g.sa2_code,
        g.sa3_code, 
        g.sa4_code,
        g.state_code,
        g.state_name,
        g.remoteness_category,
        g.centroid_longitude,
        g.centroid_latitude,
        g.area_sqkm,
        
        -- Demographic indicators
        d.total_population,
        d.median_age,
        d.median_income_weekly,
        d.indigenous_population_count,
        d.indigenous_population_percentage,
        d.population_density_per_sqkm,
        d.population_size_category,
        
        -- Socioeconomic indicators
        s.irsd_score,
        s.irsd_decile,
        s.irsad_score,
        s.ier_score,
        s.iec_score,
        s.overall_disadvantage_rank,
        
        -- Health outcome indicators
        h.diabetes_prevalence_rate,
        h.mental_health_service_rate,
        h.cardiovascular_disease_rate,
        h.cancer_incidence_rate,
        h.chronic_disease_burden_index,
        h.mental_health_usage_category,
        
        -- Healthcare access indicators
        m.gp_visits_per_capita_annual,
        m.specialist_referrals_per_capita,
        m.bulk_billing_percentage,
        m.after_hours_visits_per_capita,
        m.telehealth_visits_per_capita,
        
        -- Prevention indicators
        i.fully_immunised_1yr_rate,
        i.fully_immunised_2yr_rate,
        i.fully_immunised_5yr_rate,
        i.hpv_immunisation_rate,
        
        -- Environmental health factors
        c.avg_annual_temperature_c,
        c.total_annual_rainfall_mm,
        c.avg_annual_humidity_percent,
        c.total_heat_wave_days,
        c.total_extreme_rainfall_events,
        
        -- Data quality metadata
        greatest(
            coalesce(d.data_quality_score, 0),
            coalesce(h.health_data_quality_score, 0)
        ) as overall_data_quality_score,
        
        current_timestamp as last_updated
        
    from geography g
    left join demographics d on g.sa1_code = d.sa1_code
    left join seifa s on g.sa1_code = s.sa1_code  
    left join health_indicators h on g.sa1_code = h.sa1_code
    left join medicare_services m on g.sa1_code = m.sa1_code
    left join immunisation i on g.sa1_code = i.sa1_code
    left join climate_summary c on g.sa1_code = c.sa1_code
),

with_derived_analytics as (
    select
        *,
        
        -- Health vulnerability index (0-100, higher = more vulnerable)
        case 
            when diabetes_prevalence_rate is not null 
                 and irsd_decile is not null 
                 and gp_visits_per_capita_annual is not null 
            then round(
                (100 - (irsd_decile * 10)) * 0.4 +  -- Socioeconomic factor (40%)
                coalesce(diabetes_prevalence_rate, 0) * 1.5 +  -- Health outcomes (30%)
                greatest(0, 10 - coalesce(gp_visits_per_capita_annual, 10)) * 3  -- Access factor (30%)
            , 1)
            else null 
        end as health_vulnerability_index,
        
        -- Healthcare access classification
        case 
            when gp_visits_per_capita_annual is null then 'Unknown'
            when remoteness_category in ('Major Cities', 'Inner Regional') 
                 and gp_visits_per_capita_annual >= 4 
                 and bulk_billing_percentage >= 80
            then 'Excellent access'
            when gp_visits_per_capita_annual >= 3 and bulk_billing_percentage >= 60
            then 'Good access'
            when gp_visits_per_capita_annual >= 2 and bulk_billing_percentage >= 40  
            then 'Moderate access'
            when gp_visits_per_capita_annual >= 1
            then 'Limited access'
            else 'Poor access'
        end as healthcare_access_category,
        
        -- Climate health risk level
        case 
            when total_heat_wave_days is null then 'Unknown'
            when total_heat_wave_days = 0 then 'Low risk'
            when total_heat_wave_days between 1 and 5 then 'Moderate risk'
            when total_heat_wave_days between 6 and 15 then 'High risk'
            when total_heat_wave_days > 15 then 'Very high risk'
        end as climate_health_risk_level
        
    from integrated_profile
)

select * from with_derived_analytics
where sa1_code is not null

-- Post-processing notes:
-- This mart enables cross-domain analytics linking health outcomes to social determinants
-- Health vulnerability index weights can be adjusted based on domain expertise  
-- Missing data patterns should be monitored for systematic coverage gaps