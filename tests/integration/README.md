# Integration Testing Framework - Phase 5.2

## Overview

This integration testing framework validates the complete Australian Health Analytics platform with real Australian government data patterns and production-scale workloads. It builds on the unit testing framework from Phase 5.1 and provides comprehensive end-to-end validation.

## Test Modules

### 1. Complete Pipeline Integration (`test_complete_pipeline.py`)

Tests the full end-to-end data processing pipeline:

- **Bronze → Silver → Gold** data lake transitions
- **497,181+ record processing** simulation
- **92.9% integration success rate** validation
- **Real Australian data patterns** (2,454 SA2 areas, PBS prescriptions, SEIFA indices)
- **Performance targets**: <5 minutes end-to-end, <2s dashboard load times

#### Key Test Methods:
- `test_complete_health_analytics_pipeline()` - Full pipeline validation
- `test_cross_dataset_sa2_integration()` - SA2 code consistency across datasets
- `test_pipeline_with_incremental_updates()` - Incremental processing validation
- `test_pipeline_performance_at_scale()` - Performance under realistic loads

### 2. Cross-Component Integration (`test_cross_component_integration.py`)

Validates interaction between platform components:

- **Data processor coordination** (SEIFA ↔ Health ↔ Boundary)
- **Risk calculator integration** with all data processors
- **Storage manager compatibility** across all components
- **Memory optimizer effectiveness** (57.5% target reduction)
- **Thread safety** and **error propagation**

#### Key Test Methods:
- `test_seifa_health_boundary_processor_integration()` - Processor coordination
- `test_risk_calculator_data_processor_coordination()` - Risk calculation integration
- `test_storage_manager_processor_integration()` - Storage system integration
- `test_end_to_end_component_workflow()` - Complete workflow validation

### 3. Data Lake Operations (`test_data_lake_operations.py`)

Tests enterprise-grade data lake architecture:

- **Bronze-Silver-Gold** layer transitions with full metadata
- **Data versioning** and **schema evolution**
- **Rollback capabilities** and **data lineage tracking**
- **Concurrent operations** across data lake layers
- **Storage optimization** and **compression efficiency**

#### Key Test Methods:
- `test_bronze_silver_gold_transitions()` - Complete data lake workflow
- `test_data_versioning_and_rollback()` - Version management validation
- `test_concurrent_data_lake_operations()` - Multi-threaded data lake operations

### 4. Real Data Processing (`test_real_data_processing.py`)

Validates processing with authentic Australian government data:

- **ABS Census 2021** data processing (2,454 SA2 areas)
- **SEIFA 2021** socio-economic indices (92.9% success rate target)
- **PBS prescription data** (492,434+ records equivalent)
- **Geographic boundaries** (96MB+ shapefile processing)
- **Australian data compliance** patterns

#### Key Test Methods:
- `test_abs_census_seifa_integration()` - Real ABS/SEIFA data processing
- `test_health_geographic_boundary_integration()` - PBS + geographic integration
- `test_performance_under_real_data_volumes()` - Large-scale real data performance

### 5. Performance Integration (`test_performance_integration.py`)

Tests performance under realistic production loads:

- **End-to-end pipeline performance** at scale
- **Memory optimization effectiveness** (57.5% reduction target)
- **Dashboard response times** (<2 seconds target)
- **Concurrent processing scalability**
- **Storage performance** under heavy load

#### Key Test Methods:
- `test_end_to_end_pipeline_performance_at_scale()` - Complete pipeline performance
- `test_concurrent_processing_scalability()` - Multi-threading scalability
- `test_storage_performance_under_load()` - Storage system performance

### 6. Concurrent Operations (`test_concurrent_operations.py`)

Validates multi-threading and concurrent processing:

- **Concurrent data processing** across multiple datasets
- **Parallel risk assessment** calculations
- **Simultaneous storage operations** with thread safety
- **Error isolation** in multi-threaded environments
- **Performance scaling** with thread count

#### Key Test Methods:
- `test_concurrent_data_processing_multiple_datasets()` - Multi-dataset processing
- `test_parallel_risk_assessment_calculations()` - Parallel risk calculations
- `test_simultaneous_storage_operations()` - Concurrent storage validation
- `test_error_isolation_in_concurrent_environment()` - Error handling in threads

### 7. Error Recovery (`test_error_recovery.py`)

Tests comprehensive error handling and recovery:

- **Data corruption** handling and recovery
- **Memory pressure** and resource exhaustion recovery
- **Pipeline failure** and rollback mechanisms
- **Network timeout** and interruption simulation
- **Graceful degradation** under adverse conditions

#### Key Test Methods:
- `test_data_corruption_handling_and_recovery()` - Corruption scenarios
- `test_memory_pressure_and_resource_exhaustion_recovery()` - Resource management
- `test_pipeline_failure_and_rollback_mechanisms()` - Failure recovery
- `test_network_timeout_and_interruption_simulation()` - Network resilience

## Performance Targets

### Platform Performance Requirements

| Metric | Target | Test Validation |
|--------|--------|-----------------|
| End-to-end pipeline execution | <5 minutes | ✓ Validated across all integration tests |
| Memory optimization effectiveness | ≥57.5% reduction | ✓ Measured in performance integration |
| Dashboard load time | <2 seconds | ✓ Validated in pipeline performance tests |
| Integration success rate | ≥92.9% | ✓ Validated with real data patterns |
| SA2 area processing | 2,454 areas | ✓ Full Australian SA2 coverage tested |
| Health record processing | 497,181+ records | ✓ Scaled simulation validated |
| Concurrent processing efficiency | >500 records/second | ✓ Multi-threading performance validated |

### Data Quality Requirements

| Requirement | Validation Method | Test Coverage |
|-------------|------------------|---------------|
| Australian SA2 code compliance | 9-digit pattern validation | ✓ Cross-component integration |
| SEIFA methodology compliance | 1-10 deciles, 800-1200 scores | ✓ Real data processing tests |
| PBS/ATC code validation | WHO ATC classification | ✓ Health data integration tests |
| Geographic coordinate bounds | Australian lat/lon ranges | ✓ Boundary processing tests |
| Data lineage tracking | Metadata preservation | ✓ Data lake operations tests |

## Running Integration Tests

### Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-benchmark polars numpy psutil

# Ensure data directories exist
mkdir -p data/{raw,processed,parquet}
mkdir -p tests/integration

# Set environment variables
export PYTEST_CURRENT_TEST=integration
export PYTHONPATH=${PYTHONPATH}:$(pwd)
```

### Running All Integration Tests

```bash
# Run complete integration test suite
pytest tests/integration/ -v --tb=short

# Run with coverage reporting
pytest tests/integration/ --cov=src --cov-report=html --cov-report=term

# Run specific test module
pytest tests/integration/test_complete_pipeline.py -v

# Run performance tests only
pytest tests/integration/test_performance_integration.py -v

# Run with detailed logging
pytest tests/integration/ -v -s --log-cli-level=INFO
```

### Running Individual Test Categories

```bash
# Complete pipeline tests
pytest tests/integration/test_complete_pipeline.py::TestCompletePipelineIntegration::test_complete_health_analytics_pipeline -v

# Cross-component integration
pytest tests/integration/test_cross_component_integration.py -v

# Data lake operations
pytest tests/integration/test_data_lake_operations.py -v

# Real data processing
pytest tests/integration/test_real_data_processing.py -v

# Performance integration
pytest tests/integration/test_performance_integration.py -v

# Concurrent operations
pytest tests/integration/test_concurrent_operations.py -v

# Error recovery
pytest tests/integration/test_error_recovery.py -v
```

### Parallel Test Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pip install pytest-xdist
pytest tests/integration/ -n auto

# Run with specific number of workers
pytest tests/integration/ -n 4
```

## Test Data and Fixtures

### Shared Fixtures (from `conftest.py`)

- **`australian_health_config`** - Configuration constants for Australian health data patterns
- **`mock_seifa_data`** - Generate realistic SEIFA data with proper correlations
- **`mock_health_data`** - Generate PBS prescription data with Australian patterns
- **`mock_boundary_data`** - Generate SA2 boundary data with geographic attributes
- **`integration_test_data`** - Comprehensive integrated dataset for cross-component testing
- **`performance_benchmarks`** - Performance targets and thresholds

### Data Volume Scaling

Tests automatically scale data volumes based on available system resources:

- **Small scale**: 100-500 SA2 areas, 1K-5K health records
- **Medium scale**: 1,000-1,500 SA2 areas, 10K-25K health records  
- **Large scale**: 2,454 SA2 areas, 50K-100K health records
- **Production scale**: Full 2,454 SA2 areas, 150K+ health records

## Test Reports and Analytics

### Performance Reports

Each test generates detailed performance reports including:

- **Execution timing** for each pipeline stage
- **Memory usage** patterns and optimization effectiveness
- **Throughput metrics** (records/second, MB/second)
- **Concurrency scaling** analysis
- **Resource utilisation** monitoring

### Data Quality Reports

- **Integration success rates** across datasets
- **SA2 code consistency** validation
- **Data lineage** tracking and verification
- **Schema evolution** compatibility
- **Error handling** effectiveness

### Example Report Output

```python
{
    "total_pipeline_time": 245.7,
    "stage_timings": {
        "bronze_processing": 85.2,
        "silver_optimization": 65.1,
        "gold_analytics": 45.8,
        "storage_operations": 49.6
    },
    "memory_optimization": {
        "reduction_achieved": 61.3,  # % reduction
        "target_met": True
    },
    "integration_success_rate": 94.2,  # % of SA2 areas successfully integrated
    "performance_targets_met": True
}
```

## Continuous Integration

### GitHub Actions Integration

```yaml
# .github/workflows/integration-tests.yml
name: Integration Tests
on: [push, pull_request]

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-xdist
      - name: Run integration tests
        run: |
          pytest tests/integration/ --cov=src --cov-report=xml -n auto
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Performance Regression Detection

Integration tests include performance regression detection:

- **Baseline performance** metrics stored
- **Regression thresholds** (e.g., >10% performance degradation)
- **Automatic alerts** for performance regressions
- **Historical trending** analysis

## Troubleshooting

### Common Issues

1. **Memory limitations**: Reduce dataset sizes in fixtures
2. **Timeout issues**: Increase timeout values for large datasets
3. **File permission errors**: Ensure write permissions to test directories
4. **Import errors**: Check PYTHONPATH includes src directory

### Debug Mode

```bash
# Run with debug logging
pytest tests/integration/ -v -s --log-cli-level=DEBUG

# Run single test with detailed output
pytest tests/integration/test_complete_pipeline.py::TestCompletePipelineIntegration::test_complete_health_analytics_pipeline -v -s

# Enable pytest debugging
pytest tests/integration/ --pdb
```

### Performance Profiling

```bash
# Run with performance profiling
pytest tests/integration/ --profile --profile-svg

# Memory profiling
pytest tests/integration/ --memray

# Benchmark specific tests
pytest tests/integration/test_performance_integration.py --benchmark-only
```

## Contributing

### Adding New Integration Tests

1. **Follow naming convention**: `test_[component]_integration.py`
2. **Use shared fixtures** from `conftest.py`
3. **Include performance validation** with appropriate targets
4. **Add comprehensive error handling** and recovery scenarios
5. **Document test purpose** and validation criteria
6. **Include realistic data volumes** representative of production usage

### Test Categories

- **Smoke Tests**: Basic functionality validation
- **Performance Tests**: Load and stress testing
- **Reliability Tests**: Error handling and recovery
- **Scalability Tests**: Concurrent and parallel processing
- **Data Quality Tests**: Australian health data compliance

### Code Quality

- **Type hints** for all test functions
- **Docstrings** explaining test purpose and validation
- **Logging** for debugging and monitoring
- **Resource cleanup** in test teardown
- **Deterministic behaviour** with fixed random seeds

## Australian Health Data Compliance

### Regulatory Compliance

Integration tests validate compliance with:

- **Australian Statistical Geography Standard (ASGS)** 2021
- **WHO ATC Classification** for pharmaceutical codes  
- **ABS Census methodology** and data structures
- **SEIFA methodology** for socio-economic indices
- **PBS data formats** and prescription patterns

### Data Privacy and Security

- **No real patient data** used in tests
- **Synthetic data patterns** matching real distributions
- **Anonymised identifiers** for all test scenarios
- **Secure test data handling** and cleanup

## Support and Documentation

For detailed information on specific test components:

- **Unit Testing Framework**: See `tests/TEST_FRAMEWORK_DOCUMENTATION.md`
- **Component Architecture**: See `docs/architecture/`
- **Performance Benchmarking**: See `src/data_processing/storage/performance_benchmarking_suite.py`
- **Australian Data Sources**: See `REAL_DATA_SOURCES.md`

## Version History

- **v1.0** (Phase 5.2): Initial integration testing framework
- **Performance targets**: 57.5% memory optimization, <2s dashboard, 92.9% integration rate
- **Scale validation**: 497,181+ records, 2,454 SA2 areas
- **Australian compliance**: Full SEIFA, PBS, Census, geographic boundary integration