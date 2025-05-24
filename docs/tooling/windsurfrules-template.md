# WindSurf Rules Template

This template shows how to configure WindSurf code quality checks for the AHGD ETL project. Copy this to `.windsurfrules` in the project root and customize as needed.

```json
{
    "python": {
        "formatting": {
            "use_black": true,
            "line_length": 88,
            "use_isort": true
        },
        "linting": {
            "use_flake8": true,
            "ignore_errors": ["E203", "W503"],
            "max_complexity": 10
        },
        "typing": {
            "use_mypy": true,
            "disallow_untyped_defs": true,
            "disallow_incomplete_defs": true
        },
        "docstrings": {
            "style": "google",
            "require_docstrings": true,
            "check_docstrings": true
        }
    },
    "testing": {
        "use_pytest": true,
        "min_coverage": 80,
        "measure_coverage": true
    },
    "imports": {
        "sort_imports": true,
        "preferred_modules": {
            "dataframes": "polars",
            "file_formats": ["parquet", "yaml", "json"],
            "geo": ["shapely", "geopandas"]
        }
    },
    "project_specific": {
        "enforce_schemas": true,
        "validate_surrogate_keys": true,
        "check_dimension_references": true
    }
}
```

## Important Configuration Notes

1. Do not commit your actual `.windsurfrules` file to the repository
2. Adjust the python version, complexity, and coverage requirements based on your team's standards
3. The `project_specific` section contains custom rules specific to the AHGD ETL project data model
4. Ensure that the preferred modules match what's actually used in the project
5. Consider adding rules specifically for data quality checks if used extensively