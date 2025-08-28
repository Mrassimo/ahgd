with master_health_record as (

    select * from {{ ref('master_health_record') }}

),

final as (

    select
        *,
        (mortality_rate * 0.5) + (utilisation_rate * 0.3) + (bulk_billed_percentage * 0.2) as composite_health_index

    from master_health_record

)

select * from final
