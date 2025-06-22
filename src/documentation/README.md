# AHGD Documentation Generation

This module provides comprehensive documentation generation capabilities for the Australian Health and Geographic Data (AHGD) project.

## Features

### Data Dictionary Generator

The data dictionary generator automatically creates comprehensive documentation from Pydantic schemas with:

- **Multi-format export**: Markdown, HTML, CSV, JSON, PDF, and interactive web documentation
- **Australian health data context**: Includes ABS, AIHW, and health standards context
- **Schema categorisation**: Automatically categorises schemas by domain
- **Field relationships**: Cross-references between schemas and nested relationships
- **Validation rules documentation**: Extracts and documents Pydantic validators
- **Australian examples**: Generates realistic Australian data examples
- **Interactive browser**: Searchable, filterable web interface

## Usage

### Command Line Interface

```bash
# Generate all default formats (markdown, html, csv)
ahgd-generate-docs --type data-dictionary

# Generate specific formats
ahgd-generate-docs --type data-dictionary --formats markdown html interactive

# Custom paths
ahgd-generate-docs --type data-dictionary --schemas-path /path/to/schemas --output-path /path/to/output

# Verbose output
ahgd-generate-docs --type data-dictionary --verbose
```

### Programmatic Usage

```python
from src.documentation import generate_data_dictionary, DocumentationFormat

# Generate with defaults
output_files = generate_data_dictionary()

# Generate specific formats
output_files = generate_data_dictionary(
    formats=[
        DocumentationFormat.MARKDOWN,
        DocumentationFormat.HTML,
        DocumentationFormat.INTERACTIVE
    ]
)

# Custom generator
from src.documentation import DataDictionaryGenerator

generator = DataDictionaryGenerator()
generator.scan_schemas()
generator.generate_schema_documentation()
docs = generator.generate_documentation([DocumentationFormat.HTML])
```

## Output Formats

### Markdown (.md)
- Comprehensive tables with field details
- Australian context sections
- Usage examples
- Cross-references between schemas

### HTML (.html)
- Styled documentation with responsive design
- Table of contents with navigation
- Australian flag and context highlighting
- Print-friendly layout

### CSV (.csv)
- Spreadsheet-compatible format
- All field metadata in tabular form
- Easy filtering and analysis

### JSON (.json)
- Machine-readable format
- Complete metadata preservation
- API integration friendly

### Interactive HTML
- Searchable interface
- Category filtering
- Expandable field details
- Real-time statistics

### PDF (.pdf)
- Professional document format
- Requires `reportlab` package
- Fallback to markdown if unavailable

## Schema Categories

The generator automatically categorises schemas into:

- **Base**: Core schemas and versioning
- **Geographic**: Boundaries, coordinates, SA2/SA3/SA4
- **Health**: Mortality, morbidity, health indicators
- **Demographic**: Population, census, households
- **Socioeconomic**: SEIFA indexes and components
- **Environmental**: Climate, pollution data
- **Integrated**: Master records and aggregates
- **Validation**: Quality rules and checks

## Australian Data Standards

The generator includes context for:

- **ABS**: Australian Bureau of Statistics
- **AIHW**: Australian Institute of Health and Welfare
- **ASGS**: Australian Statistical Geography Standard
- **SEIFA**: Socio-Economic Indexes for Areas
- **ICD-10-AM**: Australian modification of ICD-10
- **Geographic hierarchies**: SA1, SA2, SA3, SA4, LGA

## Field Documentation

For each field, the generator extracts:

- **Type information**: Including Optional, Union, List types
- **Descriptions**: From Pydantic field descriptions
- **Constraints**: Validation rules (min, max, regex, etc.)
- **Examples**: Realistic Australian data examples
- **Australian context**: Geographic codes, health classifications
- **Data sources**: ABS, AIHW, Medicare data attribution
- **Business rules**: Domain-specific validation logic
- **Enum values**: Complete enumeration documentation

## Configuration

The generator can be configured through:

```python
generator = DataDictionaryGenerator(
    schemas_path=Path("custom/schemas"),
    output_path=Path("custom/output")
)
```

## Dependencies

### Required
- `pydantic>=2.0.0` - Schema inspection
- `pathlib` - File operations (built-in)

### Optional
- `reportlab` - PDF generation
- Web browser - Interactive documentation viewing

## Examples

### Health Schema Documentation
Generates Australian health context including:
- ICD-10-AM disease codes
- AIHW health indicators
- Medicare service codes
- ABS mortality classifications

### Geographic Schema Documentation
Includes Australian geographic standards:
- SA2 boundary codes (e.g., 101021007)
- State/territory codes (NSW, VIC, QLD, etc.)
- Postcode examples (2000, 3000, 4000)
- ASGS hierarchy explanations

### SEIFA Schema Documentation
Provides socio-economic context:
- Index methodology explanations
- Decile and quintile interpretations
- Census data source attribution
- Advantage/disadvantage classifications

## Integration

The documentation generator integrates with:

- **CI/CD pipelines**: Auto-generate on schema changes
- **Version control**: Track documentation changes
- **API documentation**: JSON export for API tools
- **Data catalogues**: CSV export for data management tools

## Development

To extend the generator:

1. Add new format handlers in `_generate_<format>()`
2. Extend Australian context in constructor dictionaries
3. Add field example generators in `_generate_field_examples()`
4. Implement custom schema categorisation in `_determine_schema_category()`

## British English

All generated documentation follows British English conventions:
- "optimise" not "optimize"
- "standardise" not "standardize"
- "categorise" not "categorize"