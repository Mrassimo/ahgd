# AHGD ETL Configuration Guide

This document provides a comprehensive overview of the configuration system used in the AHGD ETL project, including the different types of configuration files, their locations, and their purposes.

## Configuration Types

The AHGD ETL project uses several types of configuration:

1. **Project Configuration** - Core settings for the ETL pipeline
2. **Data Model Configuration** - Schemas, mappings, and data source definitions
3. **Environment Configuration** - Environment-specific settings
4. **Tool Configuration** - Settings for development tools
5. **User Configuration** - Individual developer preferences

## Project Configuration

Project-wide configurations define the core behavior of the ETL pipeline and are committed to the repository.

### Location: `ahgd_etl/config/`

| File | Purpose |
|------|---------|
| `settings.py` | Central configuration manager that loads and processes all config sources |

## Data Model Configuration

These YAML files define the data schemas, transformations, and sources used by the ETL pipeline.

### Location: `ahgd_etl/config/yaml/`

| File | Purpose |
|------|---------|
| `schemas.yaml` | Defines schemas for all dimension and fact tables |
| `column_mappings.yaml` | Maps source data columns to target dimensions and facts |
| `data_sources.yaml` | Defines URLs and metadata for external data sources |

### Best Practices:

1. Use the `settings.py` module to access these configurations
2. Keep schemas in sync with the actual data model
3. Document all mappings with clear comments

## Environment Configuration

These files contain environment-specific settings that vary between deployment environments.

### Location: Project root

| File | Purpose | In Git? |
|------|---------|---------|
| `.env` | Production/development environment variables | No |
| `.env.example` | Template for environment variables | Yes |

### Best Practices:

1. Never commit `.env` files with actual values
2. Keep `.env.example` up-to-date with all required variables
3. Document each variable's purpose in the example file

## Tool Configuration

These files configure development tools used in the project workflow.

### Location: Project root (usually hidden files)

| File | Purpose | In Git? |
|------|---------|---------|
| `.gitignore` | Specifies files Git should ignore | Yes |
| `.claudemcpconfig` | Claude MCP AI assistant configuration | No |
| `.taskmasterconfig` | Task Master project management configuration | No |
| `.windsurfrules` | WindSurf code quality rules | No |
| `.vscode/` | VS Code editor configuration | No |

### Templates: `docs/tooling/`

Templates for tool configurations are provided in the documentation directory:

- `claude-mcp-template.md` - Template for Claude MCP configuration
- `taskmaster-template.md` - Template for Task Master configuration
- `windsurfrules-template.md` - Template for WindSurf rules
- `vscode-settings-template.md` - Template for VS Code settings

## User Configuration

These files contain individual developer preferences and should never be committed.

### Best Practices:

1. Store user-specific configurations in files with `.local` in the name
2. Ensure all user-specific files are in `.gitignore`
3. Document your customizations if they affect project behavior

## Configuration Access

The `settings.py` module provides a unified interface for accessing all configuration:

```python
from ahgd_etl.config.settings import settings

# Access schema definitions
dim_time_schema = settings.get_schema('dim_time')

# Access column mappings
geo_levels = settings.get_column_mapping('geo_levels')

# Access data source URLs
geo_urls = settings.get_data_source('asgs2021_urls')

# Access paths
output_dir = settings.get_path('OUTPUT_DIR')

# Access environment variables
validate_data = settings.get_env('VALIDATE_DATA', default=True)
```

## Adding New Configurations

To add new configurations to the project:

1. Determine the appropriate configuration type
2. Add to the corresponding file
3. Update `settings.py` if needed to expose the new configuration
4. Document the configuration in this guide
5. Update templates if applicable