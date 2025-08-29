with census as (

    select * from {{ ref('stg_abs_census') }}

),

seifa as (

    select * from {{ ref('stg_abs_seifa') }}

),

geographic as (

    select * from {{ ref('stg_abs_geographic') }}

),

final as (

    select
        census.sa2_code,
        census.total_population,
        census.median_age,
        census.median_household_income,
        census.unemployment_rate,
        seifa.seifa_irsad_score,
        seifa.seifa_irsd_decile,
        geographic.area_square_km,
        geographic.remoteness_category

    from census
    left join seifa on census.sa2_code = seifa.sa2_code
    left join geographic on census.sa2_code = geographic.sa2_code

)

select * from final
