with source as (

    select * from {{ source('raw', 'raw_abs_census') }}

),

renamed as (

    select
        sa2_code as sa2_code,
        total_population as total_population,
        male_population as male_population,
        female_population as female_population,
        median_age as median_age,
        median_household_income as median_household_income,
        unemployment_rate as unemployment_rate,
        indigenous_population as indigenous_population

    from source

)

select * from renamed
