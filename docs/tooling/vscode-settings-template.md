# VS Code Settings Template

This template shows recommended VS Code settings for the AHGD ETL project. Create a `.vscode/settings.json` file in the project root with these settings and customize as needed.

```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": [
        "--line-length=88"
    ],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.nosetestsEnabled": false,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.extraPaths": [
        "${workspaceFolder}"
    ],
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/.coverage": true,
        "**/*.egg-info": true
    },
    "files.watcherExclude": {
        "**/data/raw/**": true,
        "**/output/**": true
    },
    "yaml.schemas": {
        "${workspaceFolder}/ahgd_etl/config/yaml/schemas.yaml": ["schemas.yaml"],
        "${workspaceFolder}/ahgd_etl/config/yaml/column_mappings.yaml": ["column_mappings.yaml"],
        "${workspaceFolder}/ahgd_etl/config/yaml/data_sources.yaml": ["data_sources.yaml"]
    }
}
```

## Important Configuration Notes

1. Do not commit your user-specific settings if they contain personal preferences
2. Adjust the linting and formatting settings based on your team's standards
3. Update the `python.analysis.extraPaths` if your project structure changes
4. The `files.exclude` and `files.watcherExclude` help improve VS Code performance by ignoring large data directories
5. The `yaml.schemas` section provides schema validation for the project's YAML configuration files