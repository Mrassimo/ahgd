-- AHGD V3: Standardized Health Indicators from AIHW
-- Clean and validate health outcome data with statistical checks

{{ config(
    materialized='view',
    tags=['aihw', 'health', 'staging']
) }}

with raw_health as (
    select * from {{ source('aihw', 'health_indicators_sa1') }}
),

health_standardized as (
    select
        -- Identifiers
        sa1_code,
        data_year as indicator_year,

        -- Diabetes prevalence (age-standardised rate per 100)
        case
            when diabetes_prevalence between 0 and 50 then diabetes_prevalence
            when diabetes_prevalence > 50 then null -- Statistical outlier, likely error
            else null
        end as diabetes_prevalence_rate,

        -- Mental health service utilisation (rate per 1000)
        case
            when mental_health_rate >= 0 then mental_health_rate
            else null
        end as mental_health_service_rate,

        -- Cardiovascular disease prevalence
        case
            when cardiovascular_disease_rate between 0 and 100 then cardiovascular_disease_rate
            else null
        end as cardiovascular_disease_rate,

        -- Cancer incidence (age-standardised rate per 100,000)
        case
            when cancer_incidence_rate between 0 and 2000 then cancer_incidence_rate
            else null
        end as cancer_incidence_rate,

        -- Data quality indicators
        95.0 as data_confidence_level, -- AIHW standard confidence level

        case
            when diabetes_prevalence = -1 or mental_health_rate = -1 or cardiovascular_disease_rate = -1
            then true
            else false
        end as data_suppression_flag

    from raw_health
    where sa1_code is not null
      and data_year between {{ var("start_date")[:4] }} and {{ var("end_date")[:4] }}
),

with_derived_metrics as (
    select
        *,

        -- Combined chronic disease burden indicator
        case
            when diabetes_prevalence_rate is not null
                 and cardiovascular_disease_rate is not null
            then (diabetes_prevalence_rate + cardiovascular_disease_rate) / 2.0
            else null
        end as chronic_disease_burden_index,

        -- Health service utilisation categories
        case
            when mental_health_service_rate is null then 'Unknown'
            when mental_health_service_rate = 0 then 'No recorded usage'
            when mental_health_service_rate between 0.1 and 20 then 'Low usage'
            when mental_health_service_rate between 20.1 and 50 then 'Moderate usage'
            when mental_health_service_rate between 50.1 and 100 then 'High usage'
            when mental_health_service_rate > 100 then 'Very high usage'
        end as mental_health_usage_category,

        -- Overall health indicator quality score
        case
            when diabetes_prevalence_rate is not null
                 and mental_health_service_rate is not null
                 and cardiovascular_disease_rate is not null
                 and cancer_incidence_rate is not null
            then 1.0
            when diabetes_prevalence_rate is not null
                 and mental_health_service_rate is not null
                 and cardiovascular_disease_rate is not null
            then 0.8
            when diabetes_prevalence_rate is not null
                 and mental_health_service_rate is not null
            then 0.6
            when diabetes_prevalence_rate is not null
            then 0.4
            else 0.2
        end as health_data_quality_score

    from health_standardized
)

select
    *,
    current_timestamp as updated_at
from with_derived_metrics

-- Data validation notes:
-- Rates suppressed for small areas (n<5) show as -1 in source
-- Age-standardised rates use Australian standard population
-- Mental health rates include all MBS-funded services
-- Cancer rates are 3-year averages to ensure statistical reliability
