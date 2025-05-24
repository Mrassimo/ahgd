# Contributing to AHGD ETL

This document outlines the standards, workflows, and guidelines for contributing to the AHGD ETL project.

## Table of Contents
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing Standards](#testing-standards)
- [Configuration Standards](#configuration-standards)
- [Documentation](#documentation)

## Project Structure

The AHGD ETL project follows a modular package-based structure:

```
ahgd_etl/                  # Main package
  ├── cli/                 # Command-line interface
  ├── config/              # Configuration
  │   ├── yaml/            # YAML configuration files 
  │   └── settings.py      # Settings manager
  ├── core/                # Core ETL functionality
  │   └── temp_fix/        # Temporary fix logic (being migrated to standard flows)
  ├── loaders/             # Data loaders with schema enforcement
  ├── models/              # Dimension and fact table models
  ├── transformers/        # Data transformers
  │   ├── census/          # Census-specific transformers
  │   └── geo/             # Geographic data transformers
  └── validators/          # Data validation modules
```

## Development Workflow

1. **Feature Development**:
   - Create a feature branch from `main`
   - Implement changes following the code standards
   - Add tests for new functionality
   - Update documentation
   - Submit a pull request

2. **Bug Fixes**:
   - Create a bugfix branch from `main`
   - Fix the issue
   - Add a test that would have caught the bug
   - Submit a pull request

3. **Code Review**:
   - All changes require code review
   - Changes must pass all tests
   - Documentation must be updated

## Code Standards

1. **Python Style**:
   - Follow PEP 8 guidelines
   - Use type hints for all functions and methods
   - Use docstrings in Google style format

2. **Object-Oriented Principles**:
   - Use classes to organize related functionality
   - Implement base classes for common behaviors
   - Prefer composition over inheritance where appropriate

3. **Error Handling**:
   - Use specific exception types
   - Log errors with appropriate context
   - Validate input data early

## Testing Standards

1. **Test Coverage**:
   - Aim for 80%+ test coverage
   - Test all public interfaces
   - Include unit, integration, and validation tests

2. **Test Organization**:
   - Maintain parallel test structure matching the code
   - Use pytest fixtures for setup/teardown
   - Use parameterized tests for multiple test cases

3. **Test Data**:
   - Use small, focused datasets for unit tests
   - Store test data in `tests/test_data/`
   - Document the purpose of each test dataset

## Configuration Standards

1. **Configuration Sources**:
   - Store shared configurations in YAML files in `ahgd_etl/config/yaml/`
   - Use environment variables for deployment-specific settings
   - Use settings.py to access all configuration with proper defaults

2. **Schema Definitions**:
   - Define all table schemas in `schemas.yaml`
   - Define column mappings in `column_mappings.yaml`
   - Define data sources in `data_sources.yaml`

3. **Tool-Specific Configurations**:
   - Store tool-specific configurations in the `docs/tooling/` directory
   - Include in .gitignore if the configuration contains sensitive information or is user-specific

## Documentation

1. **Code Documentation**:
   - Document all public modules, classes, and functions
   - Include examples for complex functionality
   - Keep documentation close to the code it documents

2. **Project Documentation**:
   - Maintain high-level documentation in the `documentation/` directory
   - Document data model in `documentation/etl_data_model_diagram.md`
   - Document ETL outputs in `documentation/etl_outputs.md`
   - Document data quality checks in validator modules

3. **Workflow Documentation**:
   - Document ETL workflow steps in README
   - Include diagrams for complex processes
   - Document known issues or limitations