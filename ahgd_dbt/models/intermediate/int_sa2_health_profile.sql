with mortality as (

    select * from {{ ref('stg_aihw_mortality') }}

),

medicare as (

    select * from {{ ref('stg_medicare_utilisation') }}

),

climate as (

    select * from {{ ref('stg_bom_climate') }}

),

final as (

    select
        mortality.sa2_code,
        mortality.mortality_rate,
        medicare.utilisation_rate,
        medicare.bulk_billed_percentage,
        climate.temperature_max_celsius,
        climate.rainfall_mm

    from mortality
    left join medicare on mortality.sa2_code = medicare.sa2_code
    left join climate on mortality.sa2_code = climate.station_id -- This join condition is likely incorrect and needs to be revisited

)

select * from final
