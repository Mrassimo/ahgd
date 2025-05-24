# Claude MCP Configuration Template

This template shows how to configure Claude MCP for the AHGD ETL project. Copy this to `.claudemcpconfig` in the project root and customize as needed.

```json
{
    "api_key": "<YOUR_API_KEY>",
    "model": "claude-3-opus-20240229",
    "max_tokens": 4096,
    "temperature": 0.7,
    "context": {
        "project_description": "Australian Health Geospatial Data (AHGD) ETL Pipeline processes Australian Bureau of Statistics Census data into a dimensional data warehouse.",
        "primary_language": "Python",
        "package_name": "ahgd_etl",
        "main_script": "run_etl_enhanced.py",
        "data_sources": "ABS Census 2021, GCP Tables G01, G17, G18, G19, G20, G21, G25",
        "core_technologies": ["Python", "Polars", "Parquet", "YAML", "Shapely", "GeoJSON"]
    },
    "snippets": {
        "dimensional_model": "The project uses a star schema with dimension tables (geography, time, demographic, health condition, person characteristic) and fact tables for various health metrics."
    }
}
```

## Important Configuration Notes

1. Do not commit your actual `.claudemcpconfig` file, especially with real API keys
2. The `context` section should be customized for your specific focus areas
3. Set `temperature` lower (0.2-0.5) for more deterministic responses
4. Set `max_tokens` based on your typical interaction needs