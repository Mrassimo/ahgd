# AHGD ETL Pipeline Data Lineage Diagrams

This directory contains comprehensive Mermaid diagrams that visualise the data lineage and flow through the Australian Health and Geographic Data (AHGD) ETL pipeline. These diagrams document how data moves from external sources through extraction, transformation, validation, and loading stages to produce the final MasterHealthRecord output.

## Diagram Overview

### 1. High-Level Data Lineage Overview
**File:** `ahgd_data_lineage_overview.mmd`

This diagram provides a bird's-eye view of the entire AHGD ETL pipeline, showing:
- **External Data Sources**: AIHW, ABS, BOM, and Medicare PBS
- **Extraction Layer**: Source-specific extractors for each data provider
- **Transformation Layer**: Geographic standardisation, data integration, and derived indicators
- **Validation Layer**: Multi-stage validation including schema, geographic, and statistical checks
- **Loading Layer**: Production loader with multi-format export capabilities
- **Final Outputs**: Parquet, CSV, GeoJSON, JSON formats and quality reports

**Key Features:**
- Colour-coded components by function (sources, extractors, transformers, validators, loaders)
- Clear data flow dependencies
- Comprehensive coverage of all pipeline stages

### 2. Detailed MasterHealthRecord Creation Flow
**File:** `master_health_record_detailed_flow.mmd`

This diagram shows the specific data elements and processing stages involved in creating the MasterHealthRecord:
- **Source Data Components**: Detailed breakdown of what data comes from each source
- **Processing Stages**: SA2 aggregation, health indicator calculation, demographic profiling
- **MasterHealthRecord Structure**: Complete structure showing all data dimensions
- **Integration Logic**: How different data sources are combined and harmonised

**Key Features:**
- Detailed data element mapping
- Processing component responsibilities
- Complete MasterHealthRecord structure
- Quality metadata integration

### 3. Technical Pipeline Flow
**File:** `etl_pipeline_technical_flow.mmd`

This diagram focuses on the technical implementation using DVC (Data Version Control):
- **DVC Stages**: The four main pipeline stages orchestrated by DVC
- **CLI Commands**: Actual command-line interface commands executed
- **Data Dependencies**: How each stage depends on previous outputs
- **Technical Components**: Specific transformation and validation components

**Key Features:**
- DVC stage orchestration
- Command-line interface mapping
- Technical component details
- Data dependency chain

### 4. Data Sources and Extraction Detail
**File:** `data_sources_extraction_detail.mmd`

This diagram provides detailed mapping of data sources to target schemas:
- **Source Specifications**: Detailed breakdown of each external data source
- **Extraction Mapping**: How source fields map to target schema fields
- **Schema Targets**: Target schema components for each data type
- **Field Transformations**: Examples of key field mappings and transformations

**Key Features:**
- Comprehensive source data inventory
- Field-level mapping examples
- Schema target organisation
- Transformation specifications

### 5. Validation and Quality Flow
**File:** `validation_quality_flow.mmd`

This diagram illustrates the comprehensive data validation and quality assurance process:
- **Validation Stages**: Pre-extraction, post-extraction, post-transformation, post-integration
- **Quality Gates**: Decision points that determine data processing continuation
- **Quality Metrics**: Completeness, accuracy, and consistency measurements
- **Outcomes**: Pass/fail/warning results and corresponding actions

**Key Features:**
- Multi-stage validation process
- Quality gate decision logic
- Comprehensive quality metrics
- Clear outcome pathways

## Using These Diagrams

### For Developers
- Use the **Technical Pipeline Flow** to understand DVC orchestration and CLI commands
- Refer to **Data Sources and Extraction Detail** for field mapping specifications
- Consult **Validation and Quality Flow** for implementing quality checks

### For Data Analysts
- Start with **High-Level Data Lineage Overview** for understanding data provenance
- Use **Detailed MasterHealthRecord Creation Flow** to understand available data dimensions
- Reference **Validation and Quality Flow** for data quality interpretation

### For Project Managers
- **High-Level Data Lineage Overview** provides complete pipeline understanding
- **Validation and Quality Flow** shows quality assurance processes
- All diagrams demonstrate comprehensive data governance

### For Stakeholders
- **High-Level Data Lineage Overview** shows data sources and final outputs
- **Detailed MasterHealthRecord Creation Flow** demonstrates data integration capabilities
- Quality assurance processes are clearly documented

## Rendering the Diagrams

These Mermaid diagrams can be rendered using:

### 1. GitHub/GitLab
Both platforms natively support Mermaid diagrams in Markdown files.

### 2. Mermaid Live Editor
Visit [mermaid.live](https://mermaid.live) and paste the diagram code for interactive viewing.

### 3. VS Code Extensions
- Install the "Mermaid Markdown Syntax Highlighting" extension
- Use "Markdown Preview Enhanced" for rendering

### 4. Command Line Tools
```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Render diagrams to images
mmdc -i ahgd_data_lineage_overview.mmd -o ahgd_data_lineage_overview.png
mmdc -i master_health_record_detailed_flow.mmd -o master_health_record_detailed_flow.svg
```

### 5. Documentation Tools
- Sphinx with mermaid extension
- GitBook with mermaid plugin
- Confluence with mermaid macros

## Data Lineage Key Concepts

### Source Systems
- **AIHW**: Health and healthcare data (mortality, hospitalisation, indicators)
- **ABS**: Geographic, demographic, and socioeconomic data
- **BOM**: Environmental and climate data
- **Medicare PBS**: Pharmaceutical and medical service data

### Processing Stages
1. **Extraction**: Pull data from external sources with validation
2. **Transformation**: Standardise, integrate, and derive indicators
3. **Validation**: Multi-layer quality assurance and compliance checking
4. **Loading**: Export to multiple optimised formats

### Quality Assurance
- **Schema Validation**: Ensure data conforms to expected structure
- **Business Rules**: Apply domain-specific validation logic
- **Statistical Validation**: Detect outliers and ensure statistical validity
- **Geographic Validation**: Verify spatial relationships and accuracy

### Output Formats
- **Parquet**: Optimised for analytics and big data processing
- **CSV**: Human-readable format for Excel and general use
- **GeoJSON**: Spatial data for web mapping applications
- **JSON**: API consumption and web applications
- **Quality Reports**: Comprehensive validation and audit documentation

## Maintenance and Updates

These diagrams should be updated when:
- New data sources are added to the pipeline
- Processing logic is modified
- Validation rules are changed
- Output formats are added or modified
- Quality requirements are updated

The diagrams serve as living documentation and should be maintained alongside code changes to ensure accuracy and usefulness for all stakeholders.

## Integration with Development Workflow

Consider integrating these diagrams into:
- **Pull Request Templates**: Reference relevant diagrams for changes
- **Documentation Generation**: Automated rendering in CI/CD pipelines
- **Architecture Reviews**: Use as reference for system design discussions
- **Training Materials**: Comprehensive onboarding documentation
- **Stakeholder Communications**: Visual communication of system capabilities