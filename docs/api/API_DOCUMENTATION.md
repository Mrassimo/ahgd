# AHGD API Documentation

Comprehensive API documentation for programmatic access to Australian Health and Geographic Data (AHGD) ETL pipeline.

## Table of Contents

1. [Introduction](#introduction)
2. [Python API Reference](#python-api-reference)
3. [CLI Command Reference](#cli-command-reference)
4. [Configuration API](#configuration-api)
5. [Data Access Patterns](#data-access-patterns)
6. [Integration Examples](#integration-examples)
7. [Error Handling](#error-handling)
8. [Troubleshooting](#troubleshooting)

## Introduction

The AHGD system provides both programmatic Python APIs and command-line interfaces for extracting, transforming, validating, and loading Australian health and geographic data. This documentation covers all programmatic access methods.

### Key Features

- **Modular ETL Pipeline**: Extract, transform, validate, and load data independently
- **Comprehensive Validation**: Schema, business rules, geographic, and statistical validation
- **British English**: All interfaces use British spelling (optimise, standardise, etc.)
- **Production Ready**: Retry logic, checkpointing, performance monitoring
- **Data Quality**: Built-in quality assessment and reporting
- **Geographic Support**: Australian statistical areas (SA2, SA3, LGA) with spatial validation

### Architecture Overview

```
Raw Data Sources → Extractors → Transformers → Validators → Loaders → Processed Data
                      ↓             ↓            ↓          ↓
                  Checkpoints   Audit Trail  Quality    Export
                               & Lineage    Reports    Formats
```

## Python API Reference

### Core Modules

#### Extractors Module (`src.extractors`)

Extract data from various Australian health and geographic sources.

##### Base Extractor

```python
from src.extractors.base import BaseExtractor
from src.utils.interfaces import SourceMetadata, DataBatch

class MyCustomExtractor(BaseExtractor):
    def __init__(self, extractor_id: str, config: Dict[str, Any]):
        super().__init__(extractor_id, config)
    
    def extract(self, source: Union[str, Path, Dict], **kwargs) -> Iterator[DataBatch]:
        """Extract data from source"""
        # Implementation here
        yield batch
    
    def get_source_metadata(self, source) -> SourceMetadata:
        """Get metadata about the source"""
        return SourceMetadata(
            source_id=self.extractor_id,
            source_type="api",
            source_url=str(source)
        )
    
    def validate_source(self, source) -> bool:
        """Validate source accessibility"""
        return True
```

##### Extractor Registry

```python
from src.extractors.extractor_registry import ExtractorRegistry

# Initialise registry
registry = ExtractorRegistry()

# List available extractors
extractors = registry.list_extractors()
for extractor_id, info in extractors.items():
    print(f"{extractor_id}: {info['description']}")

# Get specific extractor
abs_extractor = registry.get_extractor("abs_census")

# Extract data with retry logic
for batch in abs_extractor.extract_with_retry(
    source="https://api.abs.gov.au/data",
    progress_callback=lambda current, total, msg: print(f"{current}/{total}: {msg}")
):
    print(f"Extracted {len(batch)} records")
```

##### Built-in Extractors

```python
# ABS Census Extractor
from src.extractors.abs_census_extractor import ABSCensusExtractor

config = {
    'api_key': 'your_abs_api_key',
    'batch_size': 1000,
    'validation_enabled': True,
    'quality_threshold': 95.0
}

extractor = ABSCensusExtractor("abs_census", config)
census_data = list(extractor.extract_with_retry({
    'dataset': 'census_2021',
    'geographic_level': 'sa2',
    'variables': ['population', 'median_age']
}))

# AIHW Health Data Extractor
from src.extractors.aihw_extractor import AIHWExtractor

aihw_extractor = AIHWExtractor("aihw_health", config)
health_data = list(aihw_extractor.extract_with_retry({
    'indicators': ['mortality_rate', 'diabetes_prevalence'],
    'year': 2022
}))
```

#### Transformers Module (`src.transformers`)

Transform and standardise extracted data.

##### Base Transformer

```python
from src.transformers.base import BaseTransformer
import pandas as pd

class CustomTransformer(BaseTransformer):
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data to standard format"""
        # Standardise column names to snake_case
        data.columns = data.columns.str.lower().str.replace(' ', '_')
        
        # Apply data type conversions
        if 'population' in data.columns:
            data['population'] = pd.to_numeric(data['population'], errors='coerce')
        
        return data
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate input data"""
        required_cols = ['sa2_code', 'sa2_name']
        return all(col in data.columns for col in required_cols)
```

##### Geographic Transformer

```python
from src.transformers.geographic_transformer import GeographicTransformer

geo_transformer = GeographicTransformer({
    'target_crs': 'EPSG:7844',  # GDA2020
    'simplify_tolerance': 0.001,
    'validate_geometry': True
})

# Transform geographic data
transformed_df = geo_transformer.transform(raw_geodata)

# Check transformation results
print(f"Transformed {len(transformed_df)} geographic features")
print(f"CRS: {geo_transformer.get_crs(transformed_df)}")
```

##### Health Data Transformer

```python
from src.transformers.health_transformer import HealthTransformer

health_transformer = HealthTransformer({
    'standardise_indicators': True,
    'calculate_confidence_intervals': True,
    'quality_threshold': 85.0
})

# Transform health indicators
health_df = health_transformer.transform(raw_health_data)

# Add derived indicators
health_df = health_transformer.calculate_derived_indicators(health_df)
```

#### Validators Module (`src.validators`)

Comprehensive data validation framework.

##### Validation Orchestrator

```python
from src.validators.validation_orchestrator import ValidationOrchestrator
from src.validators.geographic_validator import EnhancedGeographicValidator

# Configure validation
config = {
    'validation_types': ['schema', 'business', 'geographic', 'statistical'],
    'severity_threshold': 'medium',
    'quality_thresholds': {
        'overall_quality': 85.0,
        'completeness': 90.0,
        'accuracy': 95.0
    }
}

orchestrator = ValidationOrchestrator(config=config)

# Validate dataset
validation_results = orchestrator.validate_dataset(
    input_path="data_processed/master_health_record.parquet",
    rules_path="schemas/"
)

# Process results
for validation_type, result in validation_results.items():
    print(f"{validation_type}: Quality {result['quality_score']:.1f}%")
    if result['errors']:
        print(f"  Errors: {len(result['errors'])}")
```

##### Custom Validators

```python
from src.validators.base import BaseValidator
from src.utils.interfaces import ValidationResult, ValidationSeverity

class CustomBusinessValidator(BaseValidator):
    def validate(self, data: pd.DataFrame) -> List[ValidationResult]:
        results = []
        
        # Business rule: Population density should be reasonable
        if 'population_density' in data.columns:
            extreme_density = data['population_density'] > 50000  # per sq km
            
            if extreme_density.any():
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    rule_id="BR001",
                    message="Extreme population density detected",
                    affected_records=data[extreme_density].index.tolist()
                ))
        
        return results
```

##### Geographic Validation

```python
from src.validators.geographic_validator import EnhancedGeographicValidator

# Configure geographic validator
geo_validator = EnhancedGeographicValidator({
    'target_crs': 'EPSG:7844',
    'boundary_files': {
        'sa2': 'boundaries/sa2_2021.gpkg',
        'sa3': 'boundaries/sa3_2021.gpkg'
    },
    'tolerance_meters': 100
})

# Validate geographic data
geo_results = geo_validator.validate_australian_boundaries(geodata_df)
print(f"Geographic validation: {geo_results['quality_score']:.1f}% quality")
```

#### Loaders Module (`src.loaders`)

Export data to multiple formats with optimisation.

##### Production Loader

```python
from src.loaders.production_loader import ProductionLoader
from src.utils.interfaces import DataFormat

# Configure loader
loader_config = {
    'compression': {
        'algorithm': 'gzip',
        'level': 6
    },
    'partitioning': {
        'strategy': 'geographic',
        'partition_by': 'state'
    },
    'web_optimisation': True
}

loader = ProductionLoader(config=loader_config)

# Load data to multiple formats
export_result = loader.load(
    data=processed_df,
    output_path=Path("data_exports"),
    formats=['parquet', 'csv', 'json', 'geojson'],
    compress=True,
    partition=True
)

# Check export results
for format_name, format_info in export_result['formats'].items():
    print(f"{format_name}: {len(format_info['files'])} files, "
          f"{format_info['total_size_bytes'] / 1024 / 1024:.1f} MB")
```

##### Custom Export Options

```python
# Export with custom partitioning
partitioned_export = loader.load(
    data=health_data,
    output_path=Path("exports/health"),
    formats=['parquet'],
    partition_strategy='temporal',
    partition_columns=['reference_year', 'state']
)

# Web-optimised export
web_export = loader.load(
    data=geographic_data,
    output_path=Path("exports/web"),
    formats=['geojson', 'json'],
    optimise_for_web=True,
    reduce_precision=True,
    generate_cache_headers=True
)
```

#### Pipeline Module (`src.pipelines`)

End-to-end pipeline orchestration.

##### Master ETL Pipeline

```python
from src.pipelines.master_etl_pipeline import MasterETLPipeline
from src.utils.interfaces import DataIntegrationLevel

# Configure pipeline
pipeline_config = {
    'integration_level': DataIntegrationLevel.COMPREHENSIVE,
    'quality_config': {
        'enable_validation': True,
        'quality_threshold': 85.0
    },
    'processing_config': {
        'max_workers': 4,
        'chunk_size': 10000
    },
    'output_config': {
        'format': 'parquet',
        'compress': True
    }
}

pipeline = MasterETLPipeline(config=pipeline_config)

# Execute pipeline
execution_result = pipeline.execute(
    input_path="data_raw",
    output_path="data_processed/master_health_record.parquet"
)

if execution_result['success']:
    print(f"Pipeline completed: {execution_result['records_processed']} records")
    print(f"Quality score: {execution_result['overall_quality_score']:.1f}%")
```

## CLI Command Reference

### Data Extraction

#### ahgd-extract

Extract data from Australian health and geographic sources.

**Basic Usage:**
```bash
# Extract all available sources
ahgd-extract --all --output data_raw

# Extract specific sources
ahgd-extract --sources abs_census aihw_mortality --output data_raw

# List available sources
ahgd-extract --list-sources
```

**Advanced Options:**
```bash
# Extract with custom configuration
ahgd-extract \
  --sources abs_census \
  --output data_raw \
  --format parquet \
  --compress \
  --max-workers 4 \
  --retry-attempts 5 \
  --verbose

# Dry run to check what would be extracted
ahgd-extract --sources aihw_health --dry-run

# Force re-extraction of existing data
ahgd-extract --sources abs_census --force --output data_raw
```

**Configuration File:**
```bash
# Use custom configuration
ahgd-extract --config configs/production.yaml --all --output data_raw
```

### Data Transformation

#### ahgd-transform

Transform raw data into standardised format.

**Basic Usage:**
```bash
# Transform data using master pipeline
ahgd-transform \
  --input data_raw \
  --output data_processed/master_health_record.parquet

# Transform with specific integration level
ahgd-transform \
  --input data_raw \
  --output data_processed \
  --integration-level comprehensive
```

**Advanced Options:**
```bash
# Transform with custom settings
ahgd-transform \
  --input data_raw \
  --output data_processed \
  --pipeline master_integration_pipeline \
  --quality-threshold 90.0 \
  --max-workers 8 \
  --chunk-size 20000 \
  --target-crs EPSG:7844 \
  --compress \
  --reports-dir reports

# Resume from checkpoint
ahgd-transform \
  --input data_raw \
  --output data_processed \
  --resume-from checkpoint_20240115_143022.json
```

### Data Validation

#### ahgd-validate

Validate processed data for quality and compliance.

**Basic Usage:**
```bash
# Validate processed data
ahgd-validate \
  --input data_processed/master_health_record.parquet \
  --rules schemas/ \
  --report reports/validation_report.html

# Validate with specific validation types
ahgd-validate \
  --input data_processed \
  --validation-types schema business geographic \
  --severity-threshold high
```

**Advanced Options:**
```bash
# Comprehensive validation with custom thresholds
ahgd-validate \
  --input data_processed/master_health_record.parquet \
  --rules schemas/ \
  --report reports/validation_report.html \
  --format html \
  --validation-types schema business geographic statistical quality \
  --quality-threshold 90.0 \
  --completeness-threshold 95.0 \
  --accuracy-threshold 98.0 \
  --fail-on-errors \
  --parallel \
  --max-workers 4

# Geographic-specific validation
ahgd-validate \
  --input geodata.parquet \
  --geographic-boundaries boundaries/ \
  --coordinate-system EPSG:7844 \
  --enable-outlier-detection \
  --outlier-method iqr
```

### Data Loading/Export

#### ahgd-loader

Export processed data to multiple formats.

**Basic Usage:**
```bash
# Export to multiple formats
ahgd-loader \
  --input data_processed/master_health_record.parquet \
  --output data_exports \
  --formats parquet csv json

# List supported formats
ahgd-loader --list-formats
```

**Advanced Options:**
```bash
# Export with optimisation
ahgd-loader \
  --input data_processed/master_health_record.parquet \
  --output data_exports \
  --formats parquet csv json geojson \
  --compress \
  --compression-level 9 \
  --partition \
  --partition-by state \
  --optimise-web \
  --reduce-precision \
  --generate-metadata \
  --include-lineage

# Export with custom naming
ahgd-loader \
  --input data.parquet \
  --output exports/ \
  --formats parquet \
  --prefix "ahgd_" \
  --suffix "_v2" \
  --timestamp \
  --validate-exports
```

### Documentation Generation

#### ahgd-generate-docs

Generate comprehensive documentation.

**Basic Usage:**
```bash
# Generate all documentation
ahgd-generate-docs --output docs/

# Generate specific documentation types
ahgd-generate-docs \
  --types data_dictionary api_docs \
  --output docs/ \
  --format html markdown
```

### Pipeline Orchestration

Common workflow patterns using CLI commands:

**Complete ETL Pipeline:**
```bash
#!/bin/bash
# complete_etl_pipeline.sh

# Extract data
ahgd-extract --all --output data_raw --verbose

# Transform data
ahgd-transform \
  --input data_raw \
  --output data_processed/master_health_record.parquet \
  --integration-level comprehensive \
  --quality-threshold 85.0

# Validate processed data
ahgd-validate \
  --input data_processed/master_health_record.parquet \
  --rules schemas/ \
  --report reports/validation_report.html \
  --fail-on-errors

# Export to multiple formats
ahgd-loader \
  --input data_processed/master_health_record.parquet \
  --output data_exports \
  --formats parquet csv json geojson \
  --compress \
  --optimise-web
```

## Configuration API

### Configuration Management

```python
from src.utils.config import get_config, get_config_manager, ConfigurationManager

# Get configuration values
database_url = get_config("database.url")
api_key = get_config("apis.abs.api_key", default="default_key")

# Type-safe configuration access
max_workers = get_config_manager().get_typed("system.max_workers", int)

# Environment detection
from src.utils.config import is_development, is_production, Environment

if is_development():
    debug_logging = True
elif is_production():
    enable_monitoring = True
```

### Configuration Schema

The configuration system supports YAML, JSON, and environment variables:

**Base Configuration Structure:**
```yaml
# config/base.yaml
system:
  max_workers: 4
  memory_limit: "8GB"
  enable_monitoring: true

database:
  url: "postgresql://localhost:5432/ahgd"
  pool_size: 10
  timeout: 30

apis:
  abs:
    base_url: "https://api.abs.gov.au"
    api_key: "${ABS_API_KEY}"
    timeout: 60
  aihw:
    base_url: "https://api.aihw.gov.au"
    api_key: "${AIHW_API_KEY}"

validation:
  default_quality_threshold: 85.0
  enable_geographic_validation: true
  enable_statistical_validation: true
  geographic_tolerance_meters: 100

processing:
  chunk_size: 10000
  enable_checkpoints: true
  checkpoint_interval: 1000

exports:
  compression:
    algorithm: "gzip"
    level: 6
  partitioning:
    default_strategy: "geographic"
  web_optimisation:
    enabled: true
    reduce_precision: true
```

**Environment-Specific Overrides:**

```yaml
# config/development.yaml
system:
  max_workers: 2
  enable_monitoring: false

database:
  url: "sqlite:///dev.db"

validation:
  default_quality_threshold: 70.0
```

```yaml
# config/production.yaml
system:
  max_workers: 16
  memory_limit: "32GB"
  enable_monitoring: true

processing:
  chunk_size: 50000

validation:
  default_quality_threshold: 95.0
  halt_on_validation_failure: true
```

### Configuration Validation

```python
from src.utils.config import ValidationRule, ConfigurationManager

# Define validation rules
rules = [
    ValidationRule(
        path="system.max_workers",
        required=True,
        type_check=int,
        validator=lambda x: 1 <= x <= 32,
        description="Number of worker processes"
    ),
    ValidationRule(
        path="database.url",
        required=True,
        type_check=str,
        validator=lambda x: x.startswith(("postgresql://", "sqlite://")),
        description="Database connection URL"
    )
]

# Validate configuration
config_manager = ConfigurationManager()
validation_results = config_manager.validate_configuration(rules)

if not validation_results['is_valid']:
    for error in validation_results['errors']:
        print(f"Config error: {error}")
```

### Hot-Reloading Configuration

```python
from src.utils.config import ConfigurationManager

# Enable configuration hot-reloading (development)
config_manager = ConfigurationManager(enable_hot_reload=True)

# Register callback for configuration changes
def on_config_change(changed_keys):
    print(f"Configuration changed: {changed_keys}")
    # Reinitialise components that depend on changed config

config_manager.add_change_callback(on_config_change)
```

## Data Access Patterns

### Pattern 1: Complete ETL Workflow

```python
from src.extractors.extractor_registry import ExtractorRegistry
from src.transformers.master_transformer import MasterTransformer
from src.validators.validation_orchestrator import ValidationOrchestrator
from src.loaders.production_loader import ProductionLoader

# Configure components
extractor_config = {'batch_size': 5000, 'validation_enabled': True}
transformer_config = {'integration_level': 'comprehensive'}
validator_config = {'quality_threshold': 90.0}
loader_config = {'optimise_for_web': True}

# Extract data
registry = ExtractorRegistry()
abs_extractor = registry.get_extractor("abs_census")

extracted_data = []
for batch in abs_extractor.extract_with_retry("census_2021"):
    extracted_data.extend(batch)

# Transform data
transformer = MasterTransformer(transformer_config)
df = pd.DataFrame(extracted_data)
transformed_df = transformer.transform(df)

# Validate data
orchestrator = ValidationOrchestrator(validator_config)
validation_results = orchestrator.validate_data(transformed_df)

if validation_results['overall_quality_score'] >= 90.0:
    # Load data
    loader = ProductionLoader(loader_config)
    export_result = loader.load(
        data=transformed_df,
        output_path=Path("exports"),
        formats=['parquet', 'csv', 'json']
    )
    print(f"Export successful: {export_result['success']}")
else:
    print(f"Data quality insufficient: {validation_results['overall_quality_score']:.1f}%")
```

### Pattern 2: Incremental Data Processing

```python
from datetime import datetime, timedelta
import pandas as pd

class IncrementalProcessor:
    def __init__(self, extractor, transformer, validator, loader):
        self.extractor = extractor
        self.transformer = transformer
        self.validator = validator
        self.loader = loader
        self.last_processed = self.get_last_processed_timestamp()
    
    def process_incremental_data(self):
        # Extract only new/updated data
        incremental_data = self.extractor.extract({
            'modified_since': self.last_processed,
            'batch_size': 1000
        })
        
        for batch in incremental_data:
            # Transform batch
            df = pd.DataFrame(batch)
            transformed_df = self.transformer.transform(df)
            
            # Validate batch
            validation_result = self.validator.validate_data(transformed_df)
            
            if validation_result['overall_quality_score'] >= 85.0:
                # Append to existing data
                self.loader.append_data(
                    data=transformed_df,
                    output_path="data_processed/incremental",
                    partition_by=['date', 'source']
                )
                
                # Update checkpoint
                self.update_last_processed_timestamp()
            else:
                # Handle validation failure
                self.handle_validation_failure(batch, validation_result)
```

### Pattern 3: Geographic Data Processing

```python
from src.validators.geographic_validator import EnhancedGeographicValidator
import geopandas as gpd

class GeographicDataProcessor:
    def __init__(self):
        self.geo_validator = EnhancedGeographicValidator({
            'target_crs': 'EPSG:7844',
            'boundary_files': {
                'sa2': 'boundaries/sa2_2021.gpkg'
            }
        })
    
    def process_geographic_data(self, health_data_path, boundaries_path):
        # Load health data
        health_df = pd.read_parquet(health_data_path)
        
        # Load geographic boundaries
        boundaries_gdf = gpd.read_file(boundaries_path)
        
        # Validate geographic components
        geo_validation = self.geo_validator.validate_australian_boundaries(boundaries_gdf)
        
        if geo_validation['quality_score'] >= 95.0:
            # Join health data with boundaries
            merged_gdf = boundaries_gdf.merge(
                health_df, 
                left_on='SA2_CODE21', 
                right_on='sa2_code',
                how='inner'
            )
            
            # Spatial operations
            merged_gdf['area_sq_km'] = merged_gdf.geometry.area / 1e6
            merged_gdf['population_density'] = merged_gdf['population'] / merged_gdf['area_sq_km']
            
            return merged_gdf
        else:
            raise ValueError(f"Geographic validation failed: {geo_validation['quality_score']:.1f}%")
```

### Pattern 4: Quality Assessment Workflow

```python
from src.validators.quality_checker import QualityChecker

class QualityAssessmentWorkflow:
    def __init__(self):
        self.quality_checker = QualityChecker()
    
    def assess_data_quality(self, data_path):
        # Load data
        df = pd.read_parquet(data_path)
        
        # Multi-dimensional quality assessment
        quality_report = {
            'completeness': self.assess_completeness(df),
            'accuracy': self.assess_accuracy(df),
            'consistency': self.assess_consistency(df),
            'validity': self.assess_validity(df),
            'uniqueness': self.assess_uniqueness(df)
        }
        
        # Calculate overall quality score
        weights = {
            'completeness': 0.25,
            'accuracy': 0.30,
            'consistency': 0.20,
            'validity': 0.15,
            'uniqueness': 0.10
        }
        
        overall_score = sum(
            quality_report[dimension] * weight 
            for dimension, weight in weights.items()
        )
        
        quality_report['overall_score'] = overall_score
        quality_report['quality_grade'] = self.get_quality_grade(overall_score)
        
        return quality_report
    
    def assess_completeness(self, df):
        total_cells = df.size
        non_null_cells = df.count().sum()
        return (non_null_cells / total_cells) * 100
    
    def assess_accuracy(self, df):
        # Implement accuracy checks based on business rules
        accuracy_checks = []
        
        # Check for reasonable value ranges
        if 'population' in df.columns:
            valid_population = (df['population'] >= 0) & (df['population'] <= 100000)
            accuracy_checks.append(valid_population.mean())
        
        if 'mortality_rate' in df.columns:
            valid_mortality = (df['mortality_rate'] >= 0) & (df['mortality_rate'] <= 100)
            accuracy_checks.append(valid_mortality.mean())
        
        return (sum(accuracy_checks) / len(accuracy_checks)) * 100 if accuracy_checks else 100
    
    def get_quality_grade(self, score):
        if score >= 95: return "A+"
        elif score >= 90: return "A"
        elif score >= 85: return "B+"
        elif score >= 80: return "B"
        elif score >= 75: return "C+"
        elif score >= 70: return "C"
        else: return "F"
```

## Integration Examples

### Example 1: Research Data Pipeline

```python
"""
Research pipeline for health outcomes analysis
"""
from pathlib import Path
import pandas as pd
from src.extractors.extractor_registry import ExtractorRegistry
from src.pipelines.master_etl_pipeline import MasterETLPipeline

class HealthResearchPipeline:
    def __init__(self, research_config):
        self.config = research_config
        self.output_dir = Path(research_config['output_directory'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_research_data(self):
        """Extract data for health research"""
        registry = ExtractorRegistry()
        
        datasets = {
            'health_indicators': registry.get_extractor('aihw_health'),
            'census_data': registry.get_extractor('abs_census'),
            'geographic_boundaries': registry.get_extractor('abs_boundaries')
        }
        
        extracted_data = {}
        for dataset_name, extractor in datasets.items():
            print(f"Extracting {dataset_name}...")
            
            data_batches = []
            for batch in extractor.extract_with_retry(self.config['sources'][dataset_name]):
                data_batches.extend(batch)
            
            extracted_data[dataset_name] = pd.DataFrame(data_batches)
            print(f"Extracted {len(extracted_data[dataset_name])} records")
        
        return extracted_data
    
    def prepare_research_dataset(self, extracted_data):
        """Prepare integrated research dataset"""
        pipeline_config = {
            'integration_level': 'comprehensive',
            'quality_threshold': 90.0,
            'geographic_validation': True
        }
        
        pipeline = MasterETLPipeline(pipeline_config)
        
        # Process each dataset through the pipeline
        processed_datasets = {}
        for dataset_name, raw_data in extracted_data.items():
            processed_datasets[dataset_name] = pipeline.process_dataset(
                data=raw_data,
                dataset_type=dataset_name
            )
        
        # Create master research dataset
        master_dataset = self.merge_research_datasets(processed_datasets)
        
        return master_dataset
    
    def merge_research_datasets(self, datasets):
        """Merge datasets for research analysis"""
        # Start with health indicators as base
        master_df = datasets['health_indicators'].copy()
        
        # Merge with census data on SA2 code
        master_df = master_df.merge(
            datasets['census_data'],
            on='sa2_code',
            how='left',
            suffixes=('', '_census')
        )
        
        # Add geographic information
        master_df = master_df.merge(
            datasets['geographic_boundaries'][['sa2_code', 'area_sq_km', 'centroid_lat', 'centroid_lon']],
            on='sa2_code',
            how='left'
        )
        
        return master_df
    
    def generate_research_outputs(self, master_dataset):
        """Generate research-ready outputs"""
        from src.loaders.production_loader import ProductionLoader
        
        loader = ProductionLoader({
            'optimise_for_analysis': True,
            'include_metadata': True
        })
        
        # Export research dataset
        research_export = loader.load(
            data=master_dataset,
            output_path=self.output_dir / "research_dataset",
            formats=['parquet', 'csv', 'stata'],
            metadata={
                'purpose': 'health_research',
                'creation_date': pd.Timestamp.now().isoformat(),
                'variables': list(master_dataset.columns)
            }
        )
        
        # Generate data dictionary
        data_dict = self.generate_data_dictionary(master_dataset)
        data_dict.to_csv(self.output_dir / "data_dictionary.csv")
        
        # Generate summary statistics
        summary_stats = master_dataset.describe()
        summary_stats.to_csv(self.output_dir / "summary_statistics.csv")
        
        return research_export

# Usage
research_config = {
    'output_directory': 'research_outputs/health_study_2024',
    'sources': {
        'health_indicators': {'indicators': ['mortality', 'diabetes', 'obesity'], 'year': 2022},
        'census_data': {'year': 2021, 'variables': ['population', 'age', 'income']},
        'geographic_boundaries': {'level': 'sa2', 'include_geometry': False}
    }
}

pipeline = HealthResearchPipeline(research_config)
extracted_data = pipeline.extract_research_data()
research_dataset = pipeline.prepare_research_dataset(extracted_data)
outputs = pipeline.generate_research_outputs(research_dataset)

print(f"Research pipeline completed. Outputs in: {research_config['output_directory']}")
```

### Example 2: Production Data Service

```python
"""
Production data service for web applications
"""
from fastapi import FastAPI, HTTPException
from typing import List, Optional
import pandas as pd
from src.extractors.extractor_registry import ExtractorRegistry
from src.validators.validation_orchestrator import ValidationOrchestrator

app = FastAPI(title="AHGD Data Service", version="1.0.0")

class AHGDDataService:
    def __init__(self):
        self.registry = ExtractorRegistry()
        self.validator = ValidationOrchestrator()
        self.cached_data = {}
        
    def get_health_indicators(
        self, 
        sa2_codes: Optional[List[str]] = None,
        indicator_types: Optional[List[str]] = None,
        year: Optional[int] = None
    ) -> pd.DataFrame:
        """Get health indicators with caching"""
        cache_key = f"health_{sa2_codes}_{indicator_types}_{year}"
        
        if cache_key not in self.cached_data:
            extractor = self.registry.get_extractor('aihw_health')
            
            query_params = {}
            if sa2_codes:
                query_params['sa2_codes'] = sa2_codes
            if indicator_types:
                query_params['indicator_types'] = indicator_types
            if year:
                query_params['year'] = year
            
            # Extract data
            data_batches = list(extractor.extract_with_retry(query_params))
            df = pd.DataFrame([record for batch in data_batches for record in batch])
            
            # Validate data quality
            validation_result = self.validator.validate_data(df)
            if validation_result['overall_quality_score'] < 85.0:
                raise HTTPException(
                    status_code=503, 
                    detail=f"Data quality insufficient: {validation_result['overall_quality_score']:.1f}%"
                )
            
            self.cached_data[cache_key] = df
        
        return self.cached_data[cache_key]

# Create service instance
data_service = AHGDDataService()

@app.get("/health/indicators")
async def get_health_indicators(
    sa2_codes: Optional[str] = None,
    indicator_types: Optional[str] = None,
    year: Optional[int] = None,
    format: str = "json"
):
    """Get health indicators endpoint"""
    try:
        # Parse comma-separated parameters
        sa2_list = sa2_codes.split(',') if sa2_codes else None
        indicator_list = indicator_types.split(',') if indicator_types else None
        
        # Get data
        df = data_service.get_health_indicators(sa2_list, indicator_list, year)
        
        # Format response
        if format == "json":
            return df.to_dict('records')
        elif format == "csv":
            return df.to_csv(index=False)
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/geographic/boundaries/{boundary_type}")
async def get_geographic_boundaries(
    boundary_type: str,
    state: Optional[str] = None,
    include_geometry: bool = False
):
    """Get geographic boundaries endpoint"""
    try:
        extractor = data_service.registry.get_extractor('abs_boundaries')
        
        query_params = {
            'boundary_type': boundary_type,
            'include_geometry': include_geometry
        }
        if state:
            query_params['state'] = state
        
        # Extract boundaries
        boundary_batches = list(extractor.extract_with_retry(query_params))
        df = pd.DataFrame([record for batch in boundary_batches for record in batch])
        
        return df.to_dict('records')
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Example 3: Automated Reporting System

```python
"""
Automated reporting system for health authorities
"""
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from src.pipelines.master_etl_pipeline import MasterETLPipeline
from src.validators.validation_orchestrator import ValidationOrchestrator
from src.loaders.production_loader import ProductionLoader

class AutomatedReportingSystem:
    def __init__(self, config):
        self.config = config
        self.pipeline = MasterETLPipeline(config['pipeline'])
        self.validator = ValidationOrchestrator(config['validation'])
        self.loader = ProductionLoader(config['export'])
        self.reports_dir = Path(config['reports_directory'])
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def daily_health_report(self):
        """Generate daily health indicators report"""
        print(f"Starting daily health report: {datetime.now()}")
        
        try:
            # Execute ETL pipeline
            execution_result = self.pipeline.execute(
                input_path=self.config['data_sources']['health'],
                output_path=self.reports_dir / "daily" / f"health_data_{datetime.now().strftime('%Y%m%d')}.parquet"
            )
            
            if execution_result['success']:
                # Load processed data
                df = pd.read_parquet(execution_result['output_path'])
                
                # Generate summary report
                summary_report = self.generate_health_summary(df)
                
                # Export report
                report_exports = self.loader.load(
                    data=summary_report,
                    output_path=self.reports_dir / "daily" / f"summary_{datetime.now().strftime('%Y%m%d')}",
                    formats=['xlsx', 'csv', 'html'],
                    metadata={
                        'report_type': 'daily_health_summary',
                        'generation_time': datetime.now().isoformat(),
                        'data_quality_score': execution_result['overall_quality_score']
                    }
                )
                
                # Send notifications
                self.send_report_notification(report_exports, 'daily')
                
                print(f"Daily health report completed successfully")
            else:
                self.handle_pipeline_failure(execution_result)
                
        except Exception as e:
            print(f"Daily health report failed: {str(e)}")
            self.send_error_notification(str(e))
    
    def weekly_comprehensive_report(self):
        """Generate weekly comprehensive report"""
        print(f"Starting weekly comprehensive report: {datetime.now()}")
        
        try:
            # Get data for the past week
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            # Execute comprehensive pipeline
            execution_result = self.pipeline.execute(
                input_path=self.config['data_sources']['comprehensive'],
                output_path=self.reports_dir / "weekly" / f"comprehensive_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet",
                date_range=(start_date, end_date)
            )
            
            if execution_result['success']:
                df = pd.read_parquet(execution_result['output_path'])
                
                # Generate comprehensive analytics
                analytics_report = self.generate_comprehensive_analytics(df, start_date, end_date)
                
                # Create visualisations
                charts = self.generate_charts(df)
                
                # Export comprehensive report
                self.export_comprehensive_report(analytics_report, charts)
                
                print(f"Weekly comprehensive report completed successfully")
            else:
                self.handle_pipeline_failure(execution_result)
                
        except Exception as e:
            print(f"Weekly comprehensive report failed: {str(e)}")
            self.send_error_notification(str(e))
    
    def generate_health_summary(self, df):
        """Generate health indicators summary"""
        summary = df.groupby(['state', 'indicator_type']).agg({
            'value': ['mean', 'std', 'min', 'max', 'count'],
            'data_quality_score': 'mean'
        }).round(2)
        
        summary.columns = ['_'.join(col).strip() for col in summary.columns]
        summary = summary.reset_index()
        
        return summary
    
    def generate_comprehensive_analytics(self, df, start_date, end_date):
        """Generate comprehensive analytics"""
        analytics = {
            'summary_statistics': df.describe(),
            'quality_metrics': {
                'overall_quality': df['data_quality_score'].mean(),
                'completeness': (1 - df.isnull().sum().sum() / df.size) * 100,
                'record_count': len(df)
            },
            'trend_analysis': self.calculate_trends(df),
            'geographic_distribution': df.groupby('state')['value'].agg(['mean', 'count']),
            'reporting_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            }
        }
        
        return analytics
    
    def schedule_reports(self):
        """Schedule automated reports"""
        # Daily report at 6 AM
        schedule.every().day.at("06:00").do(self.daily_health_report)
        
        # Weekly report on Mondays at 7 AM
        schedule.every().monday.at("07:00").do(self.weekly_comprehensive_report)
        
        # Data quality check every 4 hours
        schedule.every(4).hours.do(self.quality_monitoring_check)
        
        print("Report scheduler started. Running...")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

# Configuration for automated reporting
reporting_config = {
    'pipeline': {
        'integration_level': 'comprehensive',
        'quality_threshold': 85.0
    },
    'validation': {
        'quality_threshold': 85.0,
        'validation_types': ['schema', 'business', 'quality']
    },
    'export': {
        'optimise_for_web': True,
        'include_metadata': True
    },
    'reports_directory': 'automated_reports',
    'data_sources': {
        'health': 'data_raw/health',
        'comprehensive': 'data_raw'
    }
}

# Usage
if __name__ == "__main__":
    reporting_system = AutomatedReportingSystem(reporting_config)
    reporting_system.schedule_reports()
```

## Error Handling

### Exception Hierarchy

The AHGD system uses a structured exception hierarchy for error handling:

```python
from src.utils.interfaces import (
    AHGDException,
    ExtractionError,
    TransformationError,
    ValidationError,
    LoadingError,
    ConfigurationError,
    DataQualityError,
    GeographicValidationError
)

try:
    # Extract data
    extracted_data = extractor.extract_with_retry(source)
except ExtractionError as e:
    logger.error(f"Data extraction failed: {e}")
    # Handle extraction-specific error
except ValidationError as e:
    logger.error(f"Data validation failed: {e}")
    # Handle validation-specific error
except AHGDException as e:
    logger.error(f"AHGD system error: {e}")
    # Handle general AHGD error
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Handle unexpected errors
```

### Retry Logic Implementation

```python
import time
import random
from functools import wraps

def retry_with_backoff(max_retries=3, base_delay=1.0, backoff_factor=2.0, jitter=True):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (ExtractionError, ValidationError, LoadingError) as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    # Calculate delay with exponential backoff
                    delay = base_delay * (backoff_factor ** attempt)
                    if jitter:
                        delay *= (0.5 + random.random() * 0.5)  # Add jitter
                    
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator

# Usage
@retry_with_backoff(max_retries=5, base_delay=2.0)
def extract_with_retries(extractor, source):
    return extractor.extract(source)
```

### Error Recovery Patterns

```python
from pathlib import Path
import json

class ErrorRecoveryManager:
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(self, operation_id, state):
        """Save checkpoint for recovery"""
        checkpoint_file = self.checkpoint_dir / f"{operation_id}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'operation_id': operation_id,
                'state': state
            }, f, indent=2)
    
    def load_checkpoint(self, operation_id):
        """Load checkpoint for recovery"""
        checkpoint_file = self.checkpoint_dir / f"{operation_id}.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        return None
    
    def recover_from_failure(self, operation_id, recovery_function):
        """Attempt to recover from failure using checkpoint"""
        checkpoint = self.load_checkpoint(operation_id)
        if checkpoint:
            print(f"Recovering operation {operation_id} from {checkpoint['timestamp']}")
            return recovery_function(checkpoint['state'])
        else:
            raise ValueError(f"No checkpoint found for operation {operation_id}")

# Usage
recovery_manager = ErrorRecoveryManager()

def resilient_etl_pipeline(input_path, output_path):
    operation_id = f"etl_{int(time.time())}"
    
    try:
        # Save initial state
        recovery_manager.save_checkpoint(operation_id, {
            'stage': 'extraction',
            'input_path': str(input_path),
            'output_path': str(output_path),
            'progress': 0
        })
        
        # Execute pipeline stages with checkpoints
        extracted_data = extract_data(input_path)
        
        recovery_manager.save_checkpoint(operation_id, {
            'stage': 'transformation',
            'extracted_records': len(extracted_data),
            'progress': 33
        })
        
        transformed_data = transform_data(extracted_data)
        
        recovery_manager.save_checkpoint(operation_id, {
            'stage': 'validation',
            'transformed_records': len(transformed_data),
            'progress': 66
        })
        
        validation_result = validate_data(transformed_data)
        
        recovery_manager.save_checkpoint(operation_id, {
            'stage': 'loading',
            'validation_score': validation_result['quality_score'],
            'progress': 90
        })
        
        load_result = load_data(transformed_data, output_path)
        
        # Clean up checkpoint on success
        checkpoint_file = recovery_manager.checkpoint_dir / f"{operation_id}.json"
        checkpoint_file.unlink(missing_ok=True)
        
        return load_result
        
    except Exception as e:
        print(f"Pipeline failed at operation {operation_id}: {e}")
        print("Checkpoint saved for recovery")
        raise
```

### Validation Error Handling

```python
from src.validators.validation_orchestrator import ValidationOrchestrator

class ValidationErrorHandler:
    def __init__(self, config):
        self.config = config
        self.orchestrator = ValidationOrchestrator(config)
    
    def handle_validation_results(self, validation_results):
        """Handle validation results with appropriate actions"""
        actions_taken = []
        
        for validation_type, result in validation_results.items():
            if result['errors']:
                action = self.handle_validation_errors(validation_type, result['errors'])
                actions_taken.append(action)
            
            if result['warnings']:
                self.handle_validation_warnings(validation_type, result['warnings'])
        
        return actions_taken
    
    def handle_validation_errors(self, validation_type, errors):
        """Handle validation errors based on type and severity"""
        critical_errors = [e for e in errors if e.severity == 'critical']
        
        if critical_errors:
            if validation_type == 'schema':
                # Schema errors are usually unrecoverable
                raise ValidationError(f"Critical schema validation errors: {critical_errors}")
            
            elif validation_type == 'geographic':
                # Try to fix geographic errors
                return self.attempt_geographic_fix(critical_errors)
            
            elif validation_type == 'business':
                # Log business rule violations but continue
                return self.log_business_violations(critical_errors)
        
        return f"Handled {len(errors)} {validation_type} validation errors"
    
    def attempt_geographic_fix(self, geo_errors):
        """Attempt to fix geographic validation errors"""
        fixes_applied = []
        
        for error in geo_errors:
            if 'invalid_geometry' in error.details:
                # Attempt geometry repair
                fixes_applied.append("geometry_repair")
            elif 'coordinate_out_of_bounds' in error.details:
                # Attempt coordinate correction
                fixes_applied.append("coordinate_correction")
        
        return f"Applied geographic fixes: {fixes_applied}"
    
    def log_business_violations(self, business_errors):
        """Log business rule violations for review"""
        violations_log = Path("logs/business_violations.json")
        violations_log.parent.mkdir(exist_ok=True)
        
        violations = []
        for error in business_errors:
            violations.append({
                'timestamp': datetime.now().isoformat(),
                'rule_id': error.rule_id,
                'message': error.message,
                'affected_records': error.affected_records
            })
        
        # Append to violations log
        existing_violations = []
        if violations_log.exists():
            with open(violations_log, 'r') as f:
                existing_violations = json.load(f)
        
        existing_violations.extend(violations)
        
        with open(violations_log, 'w') as f:
            json.dump(existing_violations, f, indent=2)
        
        return f"Logged {len(violations)} business rule violations"
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Memory Issues with Large Datasets

**Problem**: Out of memory errors when processing large datasets.

**Solution**: Use chunked processing and streaming:

```python
from src.extractors.base import BaseExtractor

# Configure smaller batch sizes
config = {
    'batch_size': 1000,  # Reduce from default 5000
    'memory_limit': '4GB',
    'enable_streaming': True
}

# Use chunked processing
def process_large_dataset(data_path, chunk_size=10000):
    for chunk in pd.read_parquet(data_path, chunksize=chunk_size):
        processed_chunk = process_data_chunk(chunk)
        yield processed_chunk

# Memory-efficient pipeline
def memory_efficient_pipeline(input_path, output_path):
    writer = None
    
    for chunk in process_large_dataset(input_path):
        if writer is None:
            # Initialise parquet writer
            writer = pd.ExcelWriter(output_path, engine='parquet')
        
        # Write chunk
        chunk.to_parquet(writer, append=True)
    
    if writer:
        writer.close()
```

#### 2. Geographic Validation Failures

**Problem**: Geographic data fails validation due to coordinate system issues.

**Solution**: Proper CRS handling and transformation:

```python
import geopandas as gpd
from src.validators.geographic_validator import EnhancedGeographicValidator

def fix_geographic_issues(geodata_path):
    # Load with automatic CRS detection
    gdf = gpd.read_file(geodata_path)
    
    # Check current CRS
    print(f"Current CRS: {gdf.crs}")
    
    # Transform to Australian standard (GDA2020)
    if gdf.crs != 'EPSG:7844':
        print("Transforming to GDA2020...")
        gdf = gdf.to_crs('EPSG:7844')
    
    # Fix invalid geometries
    invalid_geoms = ~gdf.geometry.is_valid
    if invalid_geoms.any():
        print(f"Fixing {invalid_geoms.sum()} invalid geometries...")
        gdf.loc[invalid_geoms, 'geometry'] = gdf.loc[invalid_geoms, 'geometry'].buffer(0)
    
    # Remove duplicate geometries
    duplicates = gdf.geometry.duplicated()
    if duplicates.any():
        print(f"Removing {duplicates.sum()} duplicate geometries...")
        gdf = gdf[~duplicates]
    
    return gdf

# Usage
try:
    validator = EnhancedGeographicValidator()
    validation_result = validator.validate_australian_boundaries(geodata)
except GeographicValidationError as e:
    print(f"Geographic validation failed: {e}")
    fixed_geodata = fix_geographic_issues(geodata_path)
    validation_result = validator.validate_australian_boundaries(fixed_geodata)
```

#### 3. API Rate Limiting Issues

**Problem**: External API rate limits cause extraction failures.

**Solution**: Implement rate limiting and caching:

```python
import time
from functools import wraps
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, calls_per_minute=60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    def wait_if_needed(self):
        now = datetime.now()
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls 
                     if now - call_time < timedelta(minutes=1)]
        
        if len(self.calls) >= self.calls_per_minute:
            # Wait until the oldest call is more than 1 minute old
            oldest_call = min(self.calls)
            wait_time = 61 - (now - oldest_call).seconds
            if wait_time > 0:
                print(f"Rate limit reached. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
        
        self.calls.append(now)

def rate_limited_extraction(rate_limiter):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            rate_limiter.wait_if_needed()
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Usage
abs_rate_limiter = RateLimiter(calls_per_minute=30)  # ABS API limit

@rate_limited_extraction(abs_rate_limiter)
def extract_abs_data(endpoint):
    # API call implementation
    pass
```

#### 4. Configuration Issues

**Problem**: Configuration errors prevent system startup.

**Solution**: Configuration validation and fallbacks:

```python
from src.utils.config import ConfigurationManager, ValidationRule

def validate_and_fix_config():
    """Validate configuration and apply fixes"""
    config_manager = ConfigurationManager()
    
    # Define validation rules
    rules = [
        ValidationRule(
            path="database.url",
            required=True,
            validator=lambda x: x.startswith(("postgresql://", "sqlite://")),
            default="sqlite:///fallback.db"
        ),
        ValidationRule(
            path="system.max_workers",
            required=True,
            type_check=int,
            validator=lambda x: 1 <= x <= 32,
            default=4
        )
    ]
    
    # Validate configuration
    validation_result = config_manager.validate_configuration(rules)
    
    if not validation_result['is_valid']:
        print("Configuration issues found:")
        for error in validation_result['errors']:
            print(f"  - {error}")
        
        # Apply fallback values
        for rule in rules:
            if rule.default is not None:
                try:
                    current_value = config_manager.get(rule.path)
                    if current_value is None:
                        config_manager.set(rule.path, rule.default)
                        print(f"Applied default for {rule.path}: {rule.default}")
                except:
                    config_manager.set(rule.path, rule.default)
                    print(f"Set fallback for {rule.path}: {rule.default}")
    
    return config_manager

# Usage
try:
    config = get_config_manager()
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    config = validate_and_fix_config()
```

#### 5. Data Quality Issues

**Problem**: Data fails quality thresholds consistently.

**Solution**: Adaptive quality thresholds and data cleaning:

```python
from src.validators.quality_checker import QualityChecker

class AdaptiveQualityManager:
    def __init__(self, base_threshold=85.0):
        self.base_threshold = base_threshold
        self.quality_history = []
        self.quality_checker = QualityChecker()
    
    def assess_with_adaptive_threshold(self, data):
        """Assess quality with adaptive thresholds"""
        quality_result = self.quality_checker.assess_quality(data)
        current_score = quality_result['overall_score']
        
        # Track quality history
        self.quality_history.append(current_score)
        if len(self.quality_history) > 10:
            self.quality_history.pop(0)
        
        # Calculate adaptive threshold
        if len(self.quality_history) >= 5:
            avg_quality = sum(self.quality_history) / len(self.quality_history)
            adaptive_threshold = min(self.base_threshold, avg_quality * 0.9)
        else:
            adaptive_threshold = self.base_threshold
        
        # Apply data cleaning if below threshold
        if current_score < adaptive_threshold:
            cleaned_data = self.apply_data_cleaning(data, quality_result)
            cleaned_score = self.quality_checker.assess_quality(cleaned_data)['overall_score']
            
            if cleaned_score >= adaptive_threshold:
                return cleaned_data, cleaned_score
            else:
                print(f"Warning: Data quality {cleaned_score:.1f}% below adaptive threshold {adaptive_threshold:.1f}%")
        
        return data, current_score
    
    def apply_data_cleaning(self, data, quality_result):
        """Apply automated data cleaning"""
        cleaned_data = data.copy()
        
        # Remove rows with excessive missing values
        missing_threshold = 0.5  # 50% missing values
        row_completeness = 1 - cleaned_data.isnull().sum(axis=1) / len(cleaned_data.columns)
        cleaned_data = cleaned_data[row_completeness >= missing_threshold]
        
        # Fill missing values with appropriate strategies
        for col in cleaned_data.columns:
            if cleaned_data[col].dtype in ['int64', 'float64']:
                # Numeric: fill with median
                cleaned_data[col].fillna(cleaned_data[col].median(), inplace=True)
            else:
                # Categorical: fill with mode
                mode_value = cleaned_data[col].mode()
                if len(mode_value) > 0:
                    cleaned_data[col].fillna(mode_value[0], inplace=True)
        
        # Remove obvious outliers (beyond 3 standard deviations)
        numeric_cols = cleaned_data.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            mean_val = cleaned_data[col].mean()
            std_val = cleaned_data[col].std()
            outlier_mask = abs(cleaned_data[col] - mean_val) > 3 * std_val
            cleaned_data = cleaned_data[~outlier_mask]
        
        return cleaned_data

# Usage
quality_manager = AdaptiveQualityManager(base_threshold=85.0)

def robust_data_processing(raw_data):
    try:
        # Process with adaptive quality management
        processed_data, quality_score = quality_manager.assess_with_adaptive_threshold(raw_data)
        
        print(f"Data processing completed. Quality score: {quality_score:.1f}%")
        return processed_data
        
    except DataQualityError as e:
        print(f"Data quality error: {e}")
        # Apply emergency fallback processing
        return apply_minimal_processing(raw_data)
```

### Debugging Tools

#### Enable Debug Logging

```python
import logging
from src.utils.logging import get_logger

# Enable debug logging for specific modules
logging.getLogger('src.extractors').setLevel(logging.DEBUG)
logging.getLogger('src.validators').setLevel(logging.DEBUG)

# Use structured logging
logger = get_logger(__name__)
logger.debug("Starting data extraction", source_id="abs_census", batch_size=1000)
```

#### Performance Monitoring

```python
from src.utils.logging import monitor_performance
import time

@monitor_performance("data_processing")
def debug_data_processing(data):
    with logger.operation_context("data_processing"):
        logger.info("Processing started", records=len(data))
        
        start_time = time.time()
        result = process_data(data)
        processing_time = time.time() - start_time
        
        logger.info("Processing completed", 
                   duration=f"{processing_time:.2f}s",
                   output_records=len(result))
        
        return result
```

This comprehensive API documentation provides developers with all the necessary information to integrate with the AHGD system programmatically, from basic usage patterns to advanced integration scenarios and troubleshooting.