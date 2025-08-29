with source as (

    select * from {{ source('raw', 'raw_medicare_utilisation') }}

),

renamed as (

    select
        geographic_id as sa2_code,
        indicator_name,
        indicator_code,
        value as utilisation,
        unit,
        reference_year,
        reference_quarter,
        service_type,
        service_category,
        visits_count,
        utilisation_rate,
        bulk_billed_percentage,
        total_benefits_paid,
        unique_patients,
        data_suppressed

    from source

)

select * from renamed
