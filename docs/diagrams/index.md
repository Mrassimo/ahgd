# AHGD Data Lineage Diagrams Index

This directory contains comprehensive visual documentation of the Australian Health and Geographic Data (AHGD) ETL pipeline data lineage and architecture.

## Complete Diagram Set

### Data Lineage Diagrams (Mermaid Format)

| Diagram | File | Description | Audience |
|---------|------|-------------|----------|
| **High-Level Overview** | `ahgd_data_lineage_overview.mmd` | Complete pipeline from sources to outputs | All stakeholders |
| **Master Record Flow** | `master_health_record_detailed_flow.mmd` | Detailed MasterHealthRecord creation process | Developers, Analysts |
| **Technical Pipeline** | `etl_pipeline_technical_flow.mmd` | DVC orchestration and CLI commands | Developers |
| **Source Extraction** | `data_sources_extraction_detail.mmd` | Field mappings and schema transformations | Developers, Data Engineers |
| **Validation & Quality** | `validation_quality_flow.mmd` | Quality assurance and validation processes | QA Engineers, Analysts |

### Existing Documentation

| Document | File | Description |
|----------|------|-------------|
| **Data Lineage Overview** | `data_lineage_overview.md` | Text-based data lineage documentation |
| **Master Health Record Flow** | `master_health_record_flow.md` | Text-based record creation documentation |

## Quick Start Guide

### For New Team Members
1. Start with `ahgd_data_lineage_overview.mmd` for overall understanding
2. Review `master_health_record_detailed_flow.mmd` for data structure
3. Study `etl_pipeline_technical_flow.mmd` for implementation details

### For Developers
1. Focus on `etl_pipeline_technical_flow.mmd` for DVC commands
2. Use `data_sources_extraction_detail.mmd` for field mappings
3. Reference `validation_quality_flow.mmd` for quality implementation

### For Data Analysts
1. Begin with `ahgd_data_lineage_overview.mmd` for data provenance
2. Explore `master_health_record_detailed_flow.mmd` for available data
3. Understand `validation_quality_flow.mmd` for data quality context

## Viewing the Diagrams

### Online Rendering
- **GitHub**: Native Mermaid support in markdown files
- **Mermaid Live**: https://mermaid.live (paste diagram code)
- **VS Code**: Install Mermaid preview extensions

### Local Rendering
```bash
# Install mermaid CLI
npm install -g @mermaid-js/mermaid-cli

# Render to PNG
mmdc -i diagram.mmd -o diagram.png

# Render to SVG
mmdc -i diagram.mmd -o diagram.svg -f svg
```

## Architecture Overview

```
AHGD ETL Pipeline Data Flow:

External Sources → Extractors → Raw Data → Transformers → Validators → Loaders → Outputs
      ↓              ↓           ↓            ↓            ↓           ↓         ↓
   • AIHW         • Source     data_raw/   • Geographic  • Schema   • Multi-  • Parquet
   • ABS          • Specific   • CSV       • Integrator  • Business   format  • CSV  
   • BOM          • Field      • JSON      • Derived     • Statistical         • GeoJSON
   • Medicare     • Mapping    • Geo       • Indicators  • Geographic          • Reports
                                           • Quality     • Quality
```

## Data Quality Assurance

The pipeline implements comprehensive quality assurance:
- **Pre-extraction**: Source validation and availability checks
- **Post-extraction**: Schema compliance and format validation  
- **Post-transformation**: Business rules and geographic validation
- **Post-integration**: Statistical validation and cross-source consistency
- **Final quality gate**: Complete validation before export

## Key Integration Points

### SA2 Geographic Level
All data is integrated at the Statistical Area Level 2 (SA2) geographic unit:
- Primary key: 9-digit SA2 code
- Complete geographic hierarchy (SA1 → SA2 → SA3 → SA4 → State)
- Spatial boundary data and area calculations
- Urban/rural and remoteness classifications

### MasterHealthRecord Structure
The final integrated record contains:
- **Geographic Dimensions**: Boundaries, hierarchy, classifications
- **Demographic Profile**: Population, age/sex, households, income
- **Health Profile**: Mortality, morbidity, healthcare utilisation, risk factors
- **Socioeconomic Profile**: SEIFA indices, education, employment
- **Environmental Profile**: Climate, air quality, environmental health
- **Quality Metadata**: Data sources, quality scores, audit trail

## Maintenance and Version Control

These diagrams are maintained as:
- **Living Documentation**: Updated with code changes
- **Version Controlled**: Tracked in Git with the codebase
- **Automated Testing**: Validated during CI/CD pipeline
- **Stakeholder Communication**: Used in reviews and presentations

For questions or updates to these diagrams, please refer to the development team or create an issue in the project repository.