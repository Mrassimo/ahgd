with source as (

    select * from {{ source('raw', 'raw_abs_geographic') }}

),

renamed as (

    select
        geographic_id as sa2_code,
        geographic_level,
        geographic_name as sa2_name,
        area_square_km,
        coordinate_system,
        boundary_geometry,
        geographic_hierarchy,
        urbanisation,
        remoteness_category

    from source

)

select * from renamed
