# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Install dependencies with UV (5x faster than pip)
uv sync

# Development dependencies  
uv sync --extra dev --extra jupyter --extra docs

# Quick project initialization
python scripts/setup/quick_start.py --states nsw,vic
```

### Core Application Commands
```bash
# Download Australian health data
uv run health-analytics download --states nsw,vic,qld

# Process data with Polars + DuckDB pipeline
uv run health-analytics process

# Launch interactive dashboard
uv run health-analytics dashboard --port 8501

# Check data and processing status
uv run health-analytics status
```

### Code Quality & Testing
```bash
# Linting and formatting
uv run black src/
uv run isort src/
uv run flake8 src/

# Type checking
uv run mypy src/

# Run tests with coverage
uv run pytest tests/ --cov=src --cov-report=html

# Run single test
uv run pytest tests/test_specific.py::test_function_name -v
```

### Docker Operations
```bash
# Build container
docker build -t australian-health-analytics .

# Run dashboard in container
docker run -p 8501:8501 australian-health-analytics

# Run with data volume
docker run -v $(pwd)/data:/app/data -p 8501:8501 australian-health-analytics
```

## Architecture Overview

### Modern Data Stack Philosophy
This project implements a performance-first data architecture using cutting-edge tools:

- **Polars**: Primary data engine for 10-30x faster processing than pandas
- **DuckDB**: Embedded analytics database requiring zero configuration
- **Async Processing**: Parallel downloads and transformations throughout
- **Lazy Evaluation**: Memory-efficient processing of large Australian datasets

### Data Processing Pipeline Architecture

The pipeline follows a three-layer architecture:

1. **Acquisition Layer** (`src/data_processing/downloaders/`): Async downloads from Australian Bureau of Statistics with automatic retry and progress tracking

2. **Processing Layer** (`src/data_processing/`): 
   - `AustralianHealthData`: Main orchestrator with DuckDB workspace management
   - `CensusProcessor`: Specialised census data transformations with SEIFA integration
   - Polars lazy evaluation for memory-efficient processing

3. **Analysis Layer** (`src/analysis/`): Risk modelling, demographic analysis, and spatial calculations

### CLI and Web Interface Integration

The `src/cli.py` provides the primary interface using Typer with Rich formatting. The CLI orchestrates:
- Data acquisition workflows
- Processing pipeline execution
- Dashboard launching with environment setup
- Status monitoring and validation

The Streamlit dashboard (`src/web/streamlit/dashboard.py`) loads processed data using `@st.cache_data` and provides interactive visualizations with Altair charts and Folium maps.

### Data Flow and Storage Strategy

```
ABS Data Sources → Async Downloads → Polars Processing → Storage Optimization → DuckDB Analytics → Web Exports
```

**Production Storage Architecture (Phase 4 Complete):**
- `data/raw/`: Source downloads (gitignored)
- `data/bronze/`: Raw ingested data with time partitioning (Parquet, Snappy compression)
- `data/silver/`: Cleaned, versioned data with schema validation (Parquet, ZSTD compression)
- `data/gold/`: Analytics-ready aggregated data for business intelligence (Parquet, ZSTD compression)
- `data/processed/`: Legacy CSV/Parquet files (version controlled for smaller datasets)
- `data/outputs/`: JSON/GeoJSON for web consumption
- `data/metadata/`: Data versioning, lineage, and schema evolution tracking
- DuckDB file: Local analytics workspace (not committed)

### Performance Optimizations

**Phase 4 Storage Optimization (Complete):**
- **Memory Optimization**: 57.5% memory reduction with adaptive data type optimization
- **Parquet Compression**: 60-70% size reduction with Snappy/ZSTD compression algorithms
- **Lazy Loading**: Memory-efficient processing with Polars lazy evaluation and streaming
- **Incremental Processing**: Bronze-Silver-Gold data lake with versioning and rollback
- **Performance Monitoring**: Real-time metrics, bottleneck detection, and optimization recommendations

**Core Performance Features:**
- Polars lazy evaluation with `.scan_csv()` and `.collect()` pattern
- DuckDB spatial extensions for geographic operations
- HTTPX async client with connection pooling for downloads
- Streamlit caching for dashboard responsiveness
- Adaptive batch sizing based on available system memory
- Australian health data-specific optimizations (SA2 codes, SEIFA indices)

### Australian Health Data Specifics

The platform is designed around Australian statistical geography:
- **SA2 (Statistical Area Level 2)**: Primary geographic unit
- **SEIFA Indices**: Socio-economic disadvantage measures
- **ABS Census DataPacks**: Demographics by state
- **Health Risk Scoring**: Composite metrics combining SEIFA, demographics, and population density

### CI/CD Pipeline

GitHub Actions workflow implements:
- Automated testing with performance benchmarks comparing Polars vs pandas
- Weekly data updates triggered by schedule
- Docker image builds with health checks
- Security scanning with Trivy
- Documentation deployment to GitHub Pages

### Error Handling and Validation

The codebase implements robust error handling:
- Async download retry with exponential backoff
- Data validation using Great Expectations patterns
- DuckDB connection management with automatic cleanup
- Rich console output for user feedback

### Development Patterns

When extending the platform:
- Use Polars expressions for data transformations (avoid pandas patterns)
- Implement async methods for I/O operations
- Add type hints throughout (enforced by mypy)
- Use Rich console for user feedback
- Follow the three-layer architecture (acquisition → processing → analysis)

The CLI is the primary interface - the dashboard is for visualization only. All data operations should be accessible via `health-analytics` commands.

## Phase 2 Implementation Best Practices (Discovered 2025-06-17)

### Real Data Processing Lessons Learned

#### **Excel File Processing**
- Use **position-based mapping** for Excel files with generic column headers like "Score", "Decile"
- Implement **data cleaning during extraction** to handle missing values represented as "-" strings
- Always **stop extraction at non-data rows** (e.g., copyright notices) to prevent schema conflicts
- Use **unique column name generation** to handle duplicate headers in Excel files

#### **Geographic Data Processing**
- **DBF extraction without GeoPandas** provides more reliable processing when facing NumPy compatibility issues
- Use **dbfread library** for lightweight shapefile attribute extraction
- Implement **dual approaches**: full GeoPandas for geometry + simple DBF for attributes-only
- Always **validate SA2 codes** (9-digit format) during boundary processing

#### **Health Data Processing**
- **Standardize column names** using pattern matching and mapping dictionaries
- Implement **fallback to mock data** when real files are unavailable for development
- Use **temporal validation** (reasonable year ranges) for historical health data
- Apply **volume-based validation** for CSV files (e.g., expect 400K+ records for PBS data)

#### **Integration and Validation**
- Achieve **90%+ integration success rates** by careful SA2 code validation across datasets
- Use **comprehensive logging** with loguru for troubleshooting real data issues
- Implement **graceful degradation** when components fail
- Always **measure and report integration statistics** (match rates, record counts)

#### **Testing Strategy for Real Data**
- **Download real data in tests** but skip tests if network fails (pytest.skip)
- **Validate file sizes and formats** before processing to catch corrupted downloads
- Use **temporary directories** for all test processing to avoid conflicts
- Implement **both unit and integration tests** for comprehensive coverage

#### **Performance Optimization**
- Use **Polars lazy evaluation** with `.scan_csv()` for large files
- Implement **async downloads** with progress tracking using Rich
- **Chunk large operations** and provide progress feedback
- **Export both CSV and Parquet** formats for different use cases

### Updated Commands for Phase 2

```bash
# Process SEIFA socio-economic data
uv run python -c "from src.data_processing.seifa_processor import SEIFAProcessor; processor = SEIFAProcessor(); processor.process_complete_pipeline()"

# Process SA2 boundary data
uv run python -c "from src.data_processing.simple_boundary_processor import SimpleBoundaryProcessor; processor = SimpleBoundaryProcessor(); processor.process_complete_pipeline()"

# Process health data
uv run python -c "from src.data_processing.health_processor import HealthDataProcessor; processor = HealthDataProcessor(); processor.process_complete_pipeline()"

# Test complete integration pipeline
uv run python -c "import asyncio; asyncio.run(test_complete_pipeline())"
```

### Data Pipeline Status

#### **Current Capabilities (Phase 2 Complete)**
- **497,181 total records** processed across all real Australian government datasets
- **2,454 SA2 areas** with complete geographic and socio-economic integration
- **92.9% integration success rate** between SEIFA and boundary datasets
- **74.6MB** of real data processed with robust error handling

#### **Implemented Processors**
- **SEIFAProcessor**: 2,293 SA2 areas with 4 socio-economic indices (IRSD, IRSAD, IER, IEO)
- **SimpleBoundaryProcessor**: 2,454 geographic areas with state/territory metadata
- **HealthDataProcessor**: 492,434 PBS prescription records across Australian states
- **RealDataDownloader**: 6 verified Australian government data sources

## Phase 4 Storage Optimization (Complete - All Subphases)

### Advanced Storage Architecture

**Bronze-Silver-Gold Data Lake Implementation:**
```
Raw Data → Bronze Layer → Silver Layer → Gold Layer → Analytics
    ↓         ↓            ↓             ↓
  Ingestion  Cleaning   Aggregation   Business Intelligence
```

**Key Storage Components:**
- **ParquetStorageManager**: Optimized Parquet storage with compression benchmarking
- **IncrementalProcessor**: Data versioning, lineage tracking, and rollback capabilities
- **LazyDataLoader**: Memory-efficient lazy loading with query result caching
- **MemoryOptimizer**: Advanced memory optimization with 57.5% memory reduction
- **StoragePerformanceMonitor**: Real-time performance metrics and bottleneck detection
- **PerformanceBenchmarkingSuite**: Comprehensive benchmarking across all storage components
- **PerformanceDashboard**: Interactive performance visualization and automated reporting

### Storage Optimization Commands

```bash
# Test storage optimization pipeline
uv run python -c "from src.data_processing.storage import ParquetStorageManager; manager = ParquetStorageManager(); print('Storage optimization ready')"

# Run memory optimization demo
uv run python scripts/demo_memory_optimization.py

# Test incremental processing
uv run python -c "from src.data_processing.storage import IncrementalProcessor; processor = IncrementalProcessor(); print('Incremental processing ready')"

# Benchmark storage performance
uv run python -c "from src.data_processing.storage import StoragePerformanceMonitor; monitor = StoragePerformanceMonitor(); results = monitor.benchmark_storage_operations(); print(f'Benchmark completed: {len(results.get(\"write_performance\", {}))} tests')"

# Test lazy data loading
uv run python -c "from src.data_processing.storage import LazyDataLoader; loader = LazyDataLoader(); stats = loader.get_loader_statistics(); print(f'Lazy loader ready: {stats[\"current_memory_usage_gb\"]:.2f}GB memory usage')"

# Phase 4.4: Performance Benchmarking and Monitoring (COMPLETE)
uv run python -c "from src.data_processing.storage import PerformanceBenchmarkingSuite; suite = PerformanceBenchmarkingSuite(); print('✅ Phase 4.4 benchmarking suite ready')"

# Run comprehensive performance benchmark
uv run python -c "import sys; sys.path.append('src'); from src.data_processing.storage.performance_benchmarking_suite import PerformanceBenchmarkingSuite; suite = PerformanceBenchmarkingSuite(); results = suite.run_comprehensive_benchmark()"
```

### Storage Performance Achievements

#### **Memory Optimization Results**
- **57.5% memory reduction** on realistic Australian health datasets (18.15MB → 7.72MB)
- **31 automatic optimizations** applied including data type downcasting and categorical encoding
- **Australian health data patterns** specifically optimized (SA2 codes, SEIFA deciles, state names)
- **Adaptive streaming processing** with memory pressure detection and response

#### **Compression Performance**
- **60-70% file size reduction** with Parquet compression vs CSV
- **Snappy compression** for bronze layer (speed-optimized)
- **ZSTD compression** for silver/gold layers (space-optimized)
- **Compression algorithm benchmarking** for optimal format selection

#### **Incremental Processing Capabilities**
- **Data versioning** with complete lineage tracking and rollback support
- **Schema evolution** handling with automatic compatibility checking
- **Multiple merge strategies**: APPEND, UPSERT, REPLACE, MERGE_BY_DATE
- **Retention policies** with automatic cleanup of old versions
- **Change Data Capture (CDC)** for detecting new/updated records

#### **Lazy Loading Performance**
- **Memory-efficient processing** with Polars lazy evaluation
- **Query result caching** with TTL-based invalidation
- **Batch processing** with adaptive memory-based sizing
- **Query plan optimization** with predicate and projection pushdown

#### **Performance Benchmarking and Monitoring (Phase 4.4 Complete)**
- **Comprehensive benchmarking suite** testing all storage optimization components
- **Performance score calculation** with standardized metrics (0.885 average achieved)
- **Component comparison** across Parquet storage, memory optimization, and lazy loading
- **Automated performance reporting** with optimization recommendations
- **Regression detection** with baseline comparison and trend analysis
- **Interactive dashboards** using Plotly for performance visualization

### Phase 4 Storage Testing

```bash
# Run comprehensive storage tests
uv run pytest tests/test_storage_optimization.py -v
uv run pytest tests/test_incremental_processing.py -v
uv run pytest tests/test_memory_optimization.py -v

# Test storage integration
uv run pytest tests/test_storage_optimization.py::TestStorageIntegration::test_complete_storage_pipeline -v

# Test real-world memory optimization
uv run pytest tests/test_memory_optimization.py::TestMemoryOptimizationIntegration::test_real_world_australian_health_data_optimization -v
```

### Storage Architecture Best Practices

#### **Data Lake Design Patterns**
- **Bronze Layer**: Raw data ingestion with time-based partitioning
- **Silver Layer**: Cleaned, validated data with versioning and schema enforcement
- **Gold Layer**: Business-ready aggregated data for analytics and reporting
- **Metadata Management**: Comprehensive tracking of data lineage, versions, and schemas

#### **Memory Optimization Strategies**
- **Data Type Optimization**: Automatic downcasting (Int64→Int32→Int8, Float64→Float32)
- **Categorical Encoding**: String columns with <50% cardinality converted to categories
- **Australian Health Patterns**: SA2 codes, SEIFA deciles, state names optimized specifically
- **Streaming Processing**: Out-of-core processing for datasets exceeding available memory

#### **Performance Monitoring**
- **Real-time Metrics**: Storage operation tracking with duration, memory usage, and throughput
- **System Monitoring**: CPU, memory, and disk I/O tracking with alert thresholds
- **Optimization Recommendations**: Automated suggestions for performance improvements
- **Benchmarking**: Comprehensive storage performance testing with different algorithms

### Integration with Existing Architecture

The Phase 4 storage optimization seamlessly integrates with the existing data processing pipeline:

1. **Data Acquisition** → Uses existing downloaders and processors
2. **Storage Optimization** → Applies memory optimization and Parquet compression
3. **Incremental Processing** → Manages data versions and schema evolution
4. **Performance Monitoring** → Tracks and optimizes all storage operations
5. **Analysis Layer** → Provides optimized data for existing analytics modules