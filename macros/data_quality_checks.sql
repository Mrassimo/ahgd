-- AHGD V3: Data Quality Check Macros
-- Standardized data validation functions for health analytics

-- Calculate data completeness percentage
{% macro calculate_completeness(column_name) %}
    round(
        100.0 * count({{ column_name }}) / count(*), 
        2
    ) as {{ column_name }}_completeness_pct
{% endmacro %}

-- Generate data quality score based on completeness
{% macro data_quality_score(required_columns) %}
    case 
    {% for column in required_columns %}
        when {{ column }} is not null 
        {% if not loop.last %} and {% endif %}
    {% endfor %}
        then 1.0
    {% for i in range(required_columns|length - 1, 0, -1) %}
        when {% for j in range(i) %}{{ required_columns[j] }} is not null{% if not loop.last %} and {% endif %}{% endfor %}
        then {{ "%.1f"|format(i / required_columns|length) }}
    {% endfor %}
        else 0.0
    end as data_quality_score
{% endmacro %}

-- Validate SA1 code format (11 digits, starts with valid state code)
{% macro validate_sa1_code(sa1_code_column) %}
    case 
        when length({{ sa1_code_column }}) = 11 
             and {{ sa1_code_column }} ~ '^[1-9][0-9]{10}$'
             and left({{ sa1_code_column }}, 1) in ('1', '2', '3', '4', '5', '6', '7', '8', '9')
        then true
        else false 
    end as {{ sa1_code_column }}_valid
{% endmacro %}

-- Generate statistical outlier flags using IQR method
{% macro flag_outliers_iqr(column_name, multiplier=1.5) %}
    case 
        when {{ column_name }} < (
            percentile_cont(0.25) within group (order by {{ column_name }}) - 
            {{ multiplier }} * (
                percentile_cont(0.75) within group (order by {{ column_name }}) - 
                percentile_cont(0.25) within group (order by {{ column_name }})
            )
        ) then 'Low outlier'
        when {{ column_name }} > (
            percentile_cont(0.75) within group (order by {{ column_name }}) + 
            {{ multiplier }} * (
                percentile_cont(0.75) within group (order by {{ column_name }}) - 
                percentile_cont(0.25) within group (order by {{ column_name }})
            )
        ) then 'High outlier'
        else 'Normal'
    end as {{ column_name }}_outlier_flag
{% endmacro %}

-- Age-standardise rates using Australian standard population
{% macro age_standardise_rate(numerator, denominator, age_group_col) %}
    -- Simplified age standardisation (full implementation would use ABS standard population weights)
    sum({{ numerator }}) / sum({{ denominator }}) * 100000 as {{ numerator }}_age_std_rate
{% endmacro %}

-- Generate remoteness category from coordinates (simplified)
{% macro assign_remoteness_category(longitude, latitude) %}
    case 
        -- Major cities (simplified - based on proximity to major urban centres)
        when ({{ longitude }} between 150.5 and 151.5 and {{ latitude }} between -34.2 and -33.5)  -- Sydney
             or ({{ longitude }} between 144.5 and 145.5 and {{ latitude }} between -38.2 and -37.5)  -- Melbourne  
             or ({{ longitude }} between 152.5 and 153.5 and {{ latitude }} between -27.8 and -27.0)  -- Brisbane
             or ({{ longitude }} between 138.3 and 139.0 and {{ latitude }} between -35.2 and -34.5)  -- Adelaide
             or ({{ longitude }} between 115.5 and 116.5 and {{ latitude }} between -32.2 and -31.5)  -- Perth
        then 'Major Cities'
        -- Inner Regional (within 200km of major cities - simplified)  
        when ({{ longitude }} between 149.5 and 152.5 and {{ latitude }} between -35.2 and -32.5)
             or ({{ longitude }} between 143.5 and 146.5 and {{ latitude }} between -39.2 and -36.5)
        then 'Inner Regional'
        -- Outer Regional
        when ({{ longitude }} between 140.0 and 155.0 and {{ latitude }} between -40.0 and -28.0)
        then 'Outer Regional'  
        -- Remote and Very Remote (simplified)
        else 'Remote/Very Remote'
    end as remoteness_category_derived
{% endmacro %}