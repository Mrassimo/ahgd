with source as (

    select * from {{ source('raw', 'raw_bom_climate') }}

),

renamed as (

    select
        station_id,
        station_name,
        measurement_date,
        latitude,
        longitude,
        temperature_max_celsius,
        temperature_min_celsius,
        rainfall_mm,
        relative_humidity_9am_percent,
        relative_humidity_3pm_percent,
        wind_speed_kmh,
        solar_exposure_mj_per_m2,
        heat_stress_indicator

    from source

)

select * from renamed
