# AHGD ETL Tooling Configurations

This directory contains configuration templates and documentation for external tools used in the AHGD ETL project.

## Tool Configurations

The AHGD ETL project uses several external tools to improve development workflow. Each tool has its own configuration that should be placed in the appropriate location.

### Configuration Locations

| Tool | Configuration File | Location | In Git? |
|------|-------------------|---------|---------|
| Claude MCP | `.claudemcpconfig` | Project root | No (in .gitignore) |
| Task Master | `.taskmasterconfig` | Project root | No (in .gitignore) |
| WindSurf | `.windsurfrules` | Project root | No (in .gitignore) |
| Roo | `.roo/` | Project root | No (in .gitignore) |
| VS Code | `.vscode/` | Project root | No (in .gitignore) |
| Cursor | `.cursor/` | Project root | No (in .gitignore) |

## Tool Templates

Each subdirectory in this `/docs/tooling/` contains templates for the corresponding tools. These templates are not used directly by the tools but serve as examples for team members to create their own configurations.

### How to Use Templates

1. Find the appropriate template in the corresponding tool directory
2. Copy the template to the correct location as indicated in the table above
3. Customize the configuration for your environment

## Standard Project Rules

While tool configurations are user-specific, some rules should be consistent across all developer environments:

1. **Code Formatting**:
   - All Python code should be formatted with Black
   - Use isort for import sorting
   - Enforce PEP 8 compliance

2. **Git Workflow**:
   - Follow the conventional commit format
   - Keep commits small and focused
   - Squash commits before merging

3. **Documentation**:
   - Use Google-style docstrings
   - Document complex algorithms
   - Keep documentation up-to-date with code

See the `CONTRIBUTING.md` document in the project root for more detailed guidance on project standards and workflows.