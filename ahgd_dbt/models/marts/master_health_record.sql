with health_profile as (

    select * from {{ ref('int_sa2_health_profile') }}

),

socioeconomic_profile as (

    select * from {{ ref('int_sa2_socioeconomic_profile') }}

),

final as (

    select
        health_profile.sa2_code,
        health_profile.mortality_rate,
        health_profile.utilisation_rate,
        health_profile.bulk_billed_percentage,
        health_profile.temperature_max_celsius,
        health_profile.rainfall_mm,
        socioeconomic_profile.total_population,
        socioeconomic_profile.median_age,
        socioeconomic_profile.median_household_income,
        socioeconomic_profile.unemployment_rate,
        socioeconomic_profile.seifa_irsad_score,
        socioeconomic_profile.seifa_irsd_decile,
        socioeconomic_profile.area_square_km,
        socioeconomic_profile.remoteness_category

    from health_profile
    inner join socioeconomic_profile on health_profile.sa2_code = socioeconomic_profile.sa2_code

)

select * from final
