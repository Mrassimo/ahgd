# AHGD ETL Process Documentation

## Executive Summary

The Australian Health Geography Data (AHGD) ETL pipeline is a production-grade data processing framework designed to extract, transform, validate, and load health, geographic, and socio-economic data from multiple Australian government sources into a unified, standardised dataset at the Statistical Area Level 2 (SA2) granularity.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Extract Phase](#extract-phase)
3. [Transform Phase](#transform-phase)
4. [Validate Phase](#validate-phase)
5. [Load Phase](#load-phase)
6. [Pipeline Orchestration](#pipeline-orchestration)
7. [Configuration Management](#configuration-management)
8. [Error Handling and Recovery](#error-handling-and-recovery)
9. [Performance Optimisation](#performance-optimisation)
10. [Quality Assurance](#quality-assurance)
11. [Deployment and Monitoring](#deployment-and-monitoring)

## Architecture Overview

### Design Principles

The AHGD ETL pipeline follows these core architectural principles:

- **Modularity**: Each component is independently testable and replaceable
- **Immutability**: All transformations are versioned and traceable
- **Resilience**: Comprehensive error handling with automatic retry mechanisms
- **Performance**: Optimised for processing 2,473 SA2 areas efficiently
- **Compliance**: Adheres to Australian health data standards and British English conventions

### Core Components

```
AHGD ETL Pipeline Architecture
├── Extractors/          # Source-specific data extraction
│   ├── BaseExtractor    # Abstract base class with common functionality
│   ├── AIHWExtractor    # Australian Institute of Health and Welfare
│   ├── ABSExtractor     # Australian Bureau of Statistics
│   ├── BOMExtractor     # Bureau of Meteorology
│   └── MedicarePBSExtractor # Medicare and PBS data
├── Transformers/        # Data transformation and standardisation
│   ├── BaseTransformer  # Abstract base with audit trail
│   ├── GeographicStandardiser # SA2 code validation and mapping
│   ├── DataIntegrator   # Multi-source data joining
│   └── Denormaliser     # MasterHealthRecord creation
├── Validators/          # Multi-layered validation framework
│   ├── BaseValidator    # Quality assessment foundation
│   ├── BusinessRules    # Australian health data rules
│   ├── StatisticalValidator # Range and distribution checks
│   └── GeographicValidator  # SA2 boundary validation
└── Loaders/            # Multi-format export capabilities
    ├── BaseLoader      # Export interface
    ├── ProductionLoader # Optimised batch export
    └── FormatExporters # Parquet, CSV, GeoJSON, JSON
```

## Extract Phase

### Overview

The extraction phase retrieves data from multiple Australian government sources, implementing robust error handling, progress tracking, and data validation at source.

### Base Extractor Architecture

All extractors inherit from `BaseExtractor` which provides:

```python
class BaseExtractor(ABC):
    """Abstract base class for data extractors."""
    
    def __init__(self, extractor_id: str, config: Dict[str, Any]):
        self.extractor_id = extractor_id
        self.config = config
        self.audit_trail = AuditTrail()
        self.progress_tracker = ProgressTracker()
    
    @abstractmethod
    def extract(self) -> Iterator[DataBatch]:
        """Extract data from source."""
        pass
    
    def validate_source(self) -> bool:
        """Validate source availability and credentials."""
        pass
    
    def get_metadata(self) -> SourceMetadata:
        """Retrieve source metadata."""
        pass
```

### Extractor Implementations

#### AIHW Extractor (`aihw_extractor.py`)
- **Purpose**: Extracts health indicators, mortality data, and disease prevalence
- **Sources**: AIHW data portal APIs and downloadable datasets
- **Output**: Health indicators standardised to Australian health data standards
- **Key Features**:
  - Automatic retry on API failures
  - Incremental update detection
  - Data lineage tracking

#### ABS Extractor (`abs_extractor.py`)
- **Purpose**: Extracts demographic, census, and geographic boundary data
- **Sources**: ABS Data API, Census TableBuilder, Geographic boundaries
- **Output**: SA2-level demographic and boundary data
- **Key Features**:
  - SA2 correspondence file management
  - Population weighting calculations
  - GDA2020 coordinate system compliance

#### BOM Extractor (`bom_extractor.py`)
- **Purpose**: Extracts climate and environmental data
- **Sources**: Bureau of Meteorology APIs and weather station data
- **Output**: Climate statistics and environmental health indices
- **Key Features**:
  - Weather station to SA2 mapping
  - Statistical aggregation from point data
  - Environmental health risk calculation

#### Medicare/PBS Extractor (`medicare_pbs_extractor.py`)
- **Purpose**: Extracts healthcare utilisation and pharmaceutical data
- **Sources**: Medicare Benefits Schedule, Pharmaceutical Benefits Scheme
- **Output**: Healthcare utilisation metrics by SA2
- **Key Features**:
  - Privacy-compliant aggregation (small area suppression)
  - Service classification standardisation
  - Temporal consistency checks

### Extraction Process Flow

1. **Source Validation**: Verify data source availability and credentials
2. **Metadata Retrieval**: Collect source metadata and version information
3. **Incremental Detection**: Identify new or updated data since last extraction
4. **Data Retrieval**: Extract data with progress tracking and error handling
5. **Initial Validation**: Perform basic schema and format validation
6. **Checksum Verification**: Validate data integrity
7. **Staging**: Save raw data to staging area with metadata

### Configuration Example

```yaml
extractors:
  aihw:
    api_base_url: "https://api.aihw.gov.au"
    rate_limit: 100  # requests per minute
    timeout: 30      # seconds
    retry_attempts: 3
    retry_delay: 5   # seconds
    
  abs:
    api_key: "${ABS_API_KEY}"
    base_url: "https://api.data.abs.gov.au"
    census_year: 2021
    geographic_level: "SA2"
```

## Transform Phase

### Overview

The transformation phase standardises data from multiple sources into a common schema, performs geographic harmonisation, and creates derived indicators while maintaining complete audit trails.

### Base Transformer Architecture

```python
class BaseTransformer(ABC):
    """Abstract base class for data transformers."""
    
    def __init__(self, transformer_id: str, config: Dict[str, Any]):
        self.transformer_id = transformer_id
        self.config = config
        self.audit_trail = AuditTrail()
        
    @abstractmethod
    def transform(self, data: DataBatch) -> DataBatch:
        """Transform input data batch."""
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Get expected output schema."""
        pass
```

### Key Transformation Components

#### Geographic Standardiser (`geographic_standardiser.py`)
- **Purpose**: Standardise all geographic references to SA2 codes
- **Functions**:
  - Postcode to SA2 mapping using correspondence files
  - LGA to SA2 conversion with population weighting
  - Coordinate validation against GDA2020 standards
  - Geographic boundary validation

#### Data Integrator (`data_integrator.py`)
- **Purpose**: Join data from multiple sources at SA2 level
- **Functions**:
  - Multi-source joins with conflict resolution
  - Temporal alignment across datasets
  - Missing value imputation strategies
  - Cross-dataset consistency validation

#### Denormaliser (`denormaliser.py`)
- **Purpose**: Create denormalised MasterHealthRecord instances
- **Functions**:
  - Schema compliance validation
  - Derived indicator calculation
  - Data quality scoring
  - Performance optimisation for query access

### Transformation Process Flow

1. **Schema Validation**: Ensure input data conforms to expected schema
2. **Geographic Standardisation**: Convert all geographic references to SA2
3. **Data Type Enforcement**: Standardise data types and formats
4. **Unit Conversion**: Harmonise units across sources (e.g., percentages, rates)
5. **Missing Value Handling**: Apply appropriate imputation strategies
6. **Derived Indicators**: Calculate composite health and risk indicators
7. **Quality Assessment**: Generate data quality metrics
8. **Output Validation**: Ensure transformed data meets target schema

### British English Compliance

All transformations maintain British English spelling conventions:
- `standardise` not `standardize`
- `optimise` not `optimize`
- `colour` not `color`
- Date formats: DD/MM/YYYY

## Validate Phase

### Overview

The validation phase implements comprehensive quality assurance through multiple validation layers, ensuring data integrity, consistency, and compliance with Australian health data standards.

### Validation Architecture

```python
class ValidationOrchestrator:
    """Orchestrates multi-layered validation process."""
    
    def __init__(self, config: Dict[str, Any]):
        self.validators = [
            SchemaValidator(),
            BusinessRulesValidator(),
            StatisticalValidator(),
            GeographicValidator(),
            TemporalValidator()
        ]
    
    def validate(self, data: DataBatch) -> ValidationResult:
        """Execute all validation layers."""
        pass
```

### Validation Layers

#### 1. Schema Validation
- **Purpose**: Ensure data conforms to Pydantic schema definitions
- **Checks**: Data types, required fields, format constraints
- **Implementation**: Pydantic v2 with Australian health data models

#### 2. Business Rules Validation
- **Purpose**: Enforce Australian health data standards and domain rules
- **Rules**:
  - Population counts must be non-negative integers
  - Percentages must be between 0 and 100
  - SA2 codes must exist in official ABS list (2,473 valid codes)
  - Health indicators must fall within expected ranges

#### 3. Statistical Validation
- **Purpose**: Detect outliers and statistical anomalies
- **Methods**:
  - Interquartile range (IQR) outlier detection
  - Z-score analysis for normal distributions
  - Correlation analysis between related indicators
  - Temporal consistency checks

#### 4. Geographic Validation
- **Purpose**: Ensure geographic data integrity
- **Checks**:
  - SA2 boundary topology validation
  - Coordinate system compliance (GDA2020)
  - Coverage completeness across all SA2 areas
  - Population-weighted aggregation validation

#### 5. Temporal Validation
- **Purpose**: Maintain temporal consistency
- **Checks**:
  - Time series continuity
  - Reference period alignment
  - Trend analysis for anomaly detection
  - Data freshness validation

### Quality Scoring

Each record receives a comprehensive quality score:

```python
@dataclass
class QualityScore:
    completeness: float    # 0-100, percentage of non-null values
    accuracy: float        # 0-100, conformance to business rules
    consistency: float     # 0-100, cross-dataset consistency
    timeliness: float      # 0-100, data freshness score
    overall: float         # Weighted average of components
```

## Load Phase

### Overview

The load phase exports validated data to multiple formats optimised for different use cases, from analytical processing to web delivery.

### Export Formats

#### 1. Parquet Format
- **Purpose**: Analytical processing and data science workflows
- **Features**: Columnar storage, efficient compression, schema evolution
- **Partitioning**: By state and SA2 for optimal query performance

#### 2. CSV Format
- **Purpose**: Spreadsheet analysis and legacy system integration
- **Features**: UTF-8 encoding, British English headers, standardised date formats

#### 3. GeoJSON Format
- **Purpose**: Geographic information systems and mapping applications
- **Features**: CRS metadata, simplified geometries, web-optimised

#### 4. JSON Format
- **Purpose**: API responses and web applications
- **Features**: Nested structure, metadata inclusion, compression

### Performance Optimisation

#### Compression Strategies
- **Parquet**: Snappy compression for balance of speed and size
- **CSV**: Gzip compression for text-based data
- **JSON**: Brotli compression for web delivery

#### Partitioning
- **Geographic**: By state and region for localised queries
- **Temporal**: By reference year for time-series analysis
- **Thematic**: By data category for domain-specific access

## Pipeline Orchestration

### DVC Integration

The pipeline uses Data Version Control (DVC) for orchestration:

```yaml
# dvc.yaml
stages:
  extract:
    cmd: python -m src.pipelines.cli extract --config configs/production.yaml
    deps:
    - src/extractors/
    - configs/production.yaml
    outs:
    - data_raw/
    
  transform:
    cmd: python -m src.pipelines.cli transform --input data_raw/ --output data_processed/
    deps:
    - src/transformers/
    - data_raw/
    outs:
    - data_processed/
    
  validate:
    cmd: python -m src.pipelines.cli validate --input data_processed/
    deps:
    - src/validators/
    - data_processed/
    outs:
    - validation_reports/
    
  load:
    cmd: python -m src.pipelines.cli load --input data_processed/ --output data_final/
    deps:
    - src/loaders/
    - data_processed/
    outs:
    - data_final/
```

### CLI Commands

```bash
# Individual stage execution
ahgd-extract --source aihw --output data_raw/
ahgd-transform --input data_raw/ --output data_processed/
ahgd-validate --input data_processed/ --rules schemas/
ahgd-load --input data_processed/ --output data_final/ --format parquet

# Full pipeline execution
ahgd-pipeline --config configs/production.yaml

# DVC orchestration
dvc repro          # Run entire pipeline
dvc repro extract  # Run specific stage
dvc status         # Check pipeline status
```

## Configuration Management

### Environment-Specific Configuration

The pipeline supports multiple environments with hierarchical configuration:

```
configs/
├── base.yaml          # Common configuration
├── development.yaml   # Development overrides
├── production.yaml    # Production settings
└── testing.yaml       # Test environment
```

### Configuration Loading

```python
from src.utils.config import get_config, get_config_manager

# Simple access
database_url = get_config("database.url")

# Type-safe access
max_workers = get_config_manager().get_typed("system.max_workers", int)

# Environment detection
from src.utils.config import is_development, is_production
```

### Secrets Management

Sensitive configuration uses environment variables:

```yaml
# production.yaml
database:
  url: "${DATABASE_URL}"
  username: "${DB_USERNAME}"
  password: "${DB_PASSWORD}"

apis:
  aihw_key: "${AIHW_API_KEY}"
  abs_key: "${ABS_API_KEY}"
```

## Error Handling and Recovery

### Exception Hierarchy

```python
class AHGDError(Exception):
    """Base exception for AHGD pipeline."""
    pass

class ExtractionError(AHGDError):
    """Raised during data extraction."""
    pass

class TransformationError(AHGDError):
    """Raised during data transformation."""
    pass

class ValidationError(AHGDError):
    """Raised during data validation."""
    pass

class LoadingError(AHGDError):
    """Raised during data loading."""
    pass
```

### Retry Mechanisms

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(requests.RequestException)
)
def extract_data_with_retry(url: str) -> Dict[str, Any]:
    """Extract data with automatic retry on failure."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()
```

### Checkpointing

The pipeline supports checkpointing for recovery:

```python
class PipelineCheckpoint:
    """Manages pipeline execution checkpoints."""
    
    def save_checkpoint(self, stage: str, data: Any) -> None:
        """Save pipeline state for recovery."""
        pass
        
    def load_checkpoint(self, stage: str) -> Optional[Any]:
        """Load pipeline state from checkpoint."""
        pass
        
    def resume_from_checkpoint(self, stage: str) -> None:
        """Resume pipeline execution from checkpoint."""
        pass
```

## Performance Optimisation

### Memory Management

- **Streaming Processing**: Process data in configurable batch sizes
- **Memory Monitoring**: Track memory usage and trigger garbage collection
- **Chunking**: Split large datasets into manageable chunks
- **Lazy Loading**: Load data only when needed

### Processing Optimisation

- **Parallel Processing**: Utilise multiple CPU cores for independent operations
- **Vectorisation**: Use NumPy and Pandas for efficient array operations  
- **Caching**: Cache frequently accessed data and computations
- **Database Optimisation**: Use appropriate indexes and query optimisation

### Performance Monitoring

```python
from src.performance.monitoring import monitor_performance

@monitor_performance("data_extraction")
def extract_health_data():
    """Extract health data with performance monitoring."""
    pass

# Performance metrics automatically logged:
# - Execution time
# - Memory usage
# - CPU utilisation
# - I/O operations
```

## Quality Assurance

### Testing Strategy

#### Unit Tests
- **Coverage**: >90% code coverage for all components
- **Focus**: Individual function and method validation
- **Tools**: pytest, hypothesis for property-based testing

#### Integration Tests
- **Coverage**: End-to-end pipeline validation
- **Focus**: Component interaction and data flow
- **Approach**: Test-driven development with target schema validation

#### Performance Tests
- **Coverage**: Processing time and resource utilisation
- **Focus**: Scalability and efficiency validation
- **Benchmarks**: <5 minutes for full pipeline execution

### Continuous Integration

```yaml
# .github/workflows/ci.yml
name: AHGD CI/CD Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -e .
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
```

## Deployment and Monitoring

### Deployment Pipeline

1. **Code Quality**: Automated linting and type checking
2. **Testing**: Comprehensive test suite execution
3. **Build**: Package creation and dependency resolution
4. **Deployment**: Automated deployment to target environment
5. **Validation**: Post-deployment smoke tests
6. **Monitoring**: Real-time performance and error monitoring

### Monitoring and Alerting

```python
from src.utils.monitoring import track_metric, alert_on_threshold

# Track pipeline metrics
track_metric("pipeline.execution_time", execution_time)
track_metric("pipeline.records_processed", record_count)
track_metric("pipeline.quality_score", quality_score)

# Alert on threshold breaches
alert_on_threshold("pipeline.error_rate", error_rate, threshold=0.01)
alert_on_threshold("pipeline.quality_score", quality_score, threshold=0.95)
```

### Production Readiness Checklist

- [✅] Comprehensive error handling and logging
- [✅] Performance monitoring and optimisation
- [✅] Security compliance and data protection
- [✅] Scalability testing and resource planning
- [✅] Documentation and operational procedures
- [✅] Backup and recovery procedures
- [✅] Monitoring and alerting systems

## Best Practices

### Development Guidelines

1. **Code Quality**: Follow PEP 8 with British English spelling
2. **Documentation**: Comprehensive docstrings and README files
3. **Testing**: Write tests before implementation (TDD approach)
4. **Version Control**: Use semantic versioning and conventional commits
5. **Security**: Never commit secrets or sensitive data

### Operational Guidelines

1. **Monitoring**: Monitor all critical metrics and set appropriate alerts
2. **Backup**: Regular backups of configuration and critical data
3. **Updates**: Test all updates in staging before production deployment
4. **Documentation**: Keep operational documentation current
5. **Security**: Regular security audits and dependency updates

### Performance Guidelines

1. **Profiling**: Regular performance profiling and optimisation
2. **Scalability**: Design for 10x data volume growth
3. **Efficiency**: Optimise for both processing speed and resource usage
4. **Monitoring**: Continuous performance monitoring and tuning
5. **Benchmarking**: Regular performance benchmarking and regression testing

---

*This documentation is maintained as part of the AHGD project and updated with each major release. For the latest version, please refer to the project repository.*