with source as (

    select * from {{ source('raw', 'raw_aihw_mortality') }}

),

renamed as (

    select
        geographic_id as sa2_code,
        indicator_name,
        indicator_code,
        value as mortality_rate,
        unit,
        reference_year,
        age_group,
        sex,
        cause_of_death,
        icd10_code,
        deaths_count

    from source

)

select * from renamed
