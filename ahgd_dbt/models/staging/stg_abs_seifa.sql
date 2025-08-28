with source as (

    select * from {{ source('raw', 'raw_abs_seifa') }}

),

renamed as (

    select
        sa2_code,
        seifa_irsad_score,
        seifa_irsd_decile,
        seifa_ier_score,
        seifa_ier_decile,
        seifa_ieo_score,
        seifa_ieo_decile,
        seifa_ersad_score,
        seifa_ersad_decile

    from source

)

select * from renamed
