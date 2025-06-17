# Australian Health Analytics - Comprehensive Unit Testing Framework

## Overview

This comprehensive unit testing framework provides enterprise-grade testing for the Australian Health Analytics platform, covering all major components with >90% target coverage for critical paths.

## Framework Structure

```
tests/
├── conftest.py                          # Central pytest configuration and fixtures
├── test_data_processing/               # Data processing component tests
│   ├── __init__.py
│   ├── test_seifa_processor.py         # SEIFA processor comprehensive tests
│   ├── test_health_processor.py        # Health data processor tests
│   └── test_simple_boundary_processor.py # Boundary processor tests
├── test_storage/                       # Storage optimization tests
│   ├── __init__.py
│   ├── test_parquet_storage_manager.py # Parquet optimization tests
│   └── test_memory_optimizer.py       # Memory optimization tests
├── test_analysis/                      # Analysis module tests
│   ├── __init__.py
│   └── test_health_risk_calculator.py # Risk assessment tests
├── fixtures/                          # Mock data generators
│   └── __init__.py
├── utils/                             # Testing utilities
│   └── __init__.py
├── test_performance_benchmarks.py     # Performance validation tests
└── test_integration_comprehensive.py  # End-to-end integration tests
```

## Key Features

### 1. Australian Health Data Pattern Fixtures

The framework provides comprehensive mock data generators that simulate real Australian health data patterns:

```python
# SA2 codes (9-digit Australian statistical areas)
@pytest.fixture
def mock_sa2_codes(australian_health_config):
    """Generate realistic 9-digit SA2 codes with valid state prefixes."""
    
# SEIFA data with realistic correlations
@pytest.fixture  
def mock_seifa_data(mock_sa2_codes, australian_health_config):
    """Generate SEIFA data with correlated indices and realistic patterns."""
    
# PBS health data with Australian patterns
@pytest.fixture
def mock_health_data(mock_sa2_codes, australian_health_config):
    """Generate PBS prescription data with Australian health patterns."""
```

### 2. Performance Benchmarking

Built-in performance benchmarks validate optimization targets:

```python
@pytest.fixture
def performance_benchmarks():
    return {
        "parquet_compression": {
            "min_compression_ratio": 0.6,  # 60% compression minimum
            "max_read_time_per_mb": 0.1,   # 0.1s per MB read time
            "max_write_time_per_mb": 0.2   # 0.2s per MB write time
        },
        "memory_optimization": {
            "max_memory_increase": 1.5,    # Max 50% memory increase
            "min_memory_reduction": 0.3,   # Min 30% reduction after optimization
        }
    }
```

### 3. Property-Based Testing

Advanced testing with Hypothesis for edge case discovery:

```python
@given(
    sa2_codes=st.lists(st.text(alphabet="12345678", min_size=9, max_size=9)),
    scores=st.lists(st.integers(min_value=800, max_value=1200))
)
def test_seifa_validation_properties(self, sa2_codes, scores):
    """Property-based test ensuring SEIFA validation maintains invariants."""
```

### 4. Comprehensive Error Simulation

Built-in error simulation for robust testing:

```python
@pytest.fixture
def error_simulation():
    class ErrorSimulator:
        @staticmethod
        def corrupt_sa2_codes(df, corruption_rate=0.1):
            """Introduce invalid SA2 codes for error handling tests."""
            
        @staticmethod
        def introduce_missing_values(df, missing_rate=0.1):
            """Add missing values across columns."""
```

## Test Categories

### 1. Data Processing Tests

**SEIFA Processor Tests** (`test_seifa_processor.py`):
- Excel file validation and structure checking
- Data extraction from Table 1 sheet with real ABS format
- Column standardization and type conversion
- SA2 code validation (9-digit Australian codes)
- SEIFA index validation (scores 800-1200, deciles 1-10)
- Error handling for corrupted/invalid data
- Performance benchmarks for 2000+ SA2 areas

**Health Data Processor Tests** (`test_health_processor.py`):
- PBS and MBS data validation and cleaning
- SA2 code linking and geographic aggregation
- ATC code validation and classification
- Temporal trend analysis
- Error handling for negative values and missing data
- Performance with 50K+ prescription records

**Boundary Processor Tests** (`test_simple_boundary_processor.py`):
- Geographic coordinate validation (Australian bounds)
- Population density calculations
- Remoteness classification (ARIA categories)
- Area calculations and boundary validation
- Performance with 2000+ SA2 boundaries

### 2. Storage Optimization Tests

**Parquet Storage Manager Tests** (`test_parquet_storage_manager.py`):
- Compression benchmarking (60-70% target ratios)
- Column-specific optimizations (categorical encoding)
- Read/write performance validation
- Schema evolution handling
- Large dataset batch processing
- Memory efficiency during I/O operations

**Memory Optimizer Tests** (`test_memory_optimizer.py`):
- Data type optimization for Australian patterns
- Chunked processing for memory efficiency
- Memory leak detection across repeated operations
- Garbage collection strategies
- Adaptive optimization based on data characteristics
- Performance benchmarks with 30%+ memory reduction targets

### 3. Analysis Module Tests

**Health Risk Calculator Tests** (`test_health_risk_calculator.py`):
- SEIFA-based risk score calculations with weighted indices
- Health utilisation risk assessment
- Geographic accessibility risk factors
- Composite risk score validation
- Risk categorisation (Very Low to Very High)
- Population-adjusted risk calculations
- Integration with all data components

### 4. Performance Tests

**Benchmark Validation** (`test_performance_benchmarks.py`):
- End-to-end pipeline performance (complete processing <60s)
- Memory usage validation (<1GB for realistic datasets)
- Concurrent processing safety and scalability
- I/O performance across file formats
- Query performance on large datasets
- Memory leak detection during repeated operations

### 5. Integration Tests

**Comprehensive Integration** (`test_integration_comprehensive.py`):
- End-to-end data processing pipeline
- Cross-component data consistency validation
- Storage optimization integration
- Risk assessment pipeline integration
- Error handling across components
- Data versioning and compatibility

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/test_data_processing/

# Run with coverage report
pytest --cov=src --cov-report=html

# Run performance tests only
pytest tests/test_performance_benchmarks.py

# Run integration tests
pytest tests/test_integration_comprehensive.py
```

### Selective Test Execution

```bash
# Run SEIFA processor tests only
pytest tests/test_data_processing/test_seifa_processor.py

# Run specific test method
pytest tests/test_data_processing/test_seifa_processor.py::TestSEIFAProcessor::test_validate_seifa_file_valid

# Run tests matching pattern
pytest -k "seifa_processor"

# Run tests with specific markers (when configured)
pytest -m "performance"
```

### Debugging and Verbose Output

```bash
# Verbose output with detailed logging
pytest -v -s

# Stop on first failure
pytest -x

# Show local variables in traceback
pytest --tb=long

# Run specific test with debugging
pytest tests/test_data_processing/test_seifa_processor.py::TestSEIFAProcessor::test_validate_seifa_file_valid -v -s
```

## Test Data Patterns

### Australian Health Data Characteristics

The framework generates realistic test data matching Australian health patterns:

**SA2 Codes**: 9-digit codes with valid state prefixes (1-8)
- Format: `{state_digit}{8_area_digits}`
- Examples: `123456789` (NSW), `234567890` (VIC)

**SEIFA Indices**: Socio-economic measures with realistic correlations
- Deciles: 1-10 (1=most disadvantaged, 10=most advantaged)
- Scores: 800-1200 range with normal distribution around 1000
- Correlated indices reflecting real socio-economic patterns

**Health Utilisation**: PBS prescription patterns
- Prescription counts: Poisson distribution (realistic usage patterns)
- Chronic medication rates: ~30% of population
- Cost distributions: Exponential (typical pharmaceutical cost patterns)
- ATC codes: Valid 7-character therapeutic classifications

**Geographic Data**: Australian boundary characteristics
- Coordinate bounds: Latitude -44° to -10°, Longitude 113° to 154°
- Population densities: 0.1 to 5000+ people/km²
- Remoteness categories: Major Cities to Very Remote (ARIA classification)

## Performance Targets

### Processing Performance
- **SEIFA Processing**: <30s for 2000+ SA2 areas
- **Health Data Processing**: <45s for 50K+ prescription records  
- **Boundary Processing**: <20s for 2000+ SA2 boundaries
- **Risk Assessment**: <30s for complete multi-factor analysis

### Memory Efficiency
- **Memory Optimization**: 30%+ reduction in data memory footprint
- **Peak Memory**: <1GB for realistic dataset processing
- **Memory Leaks**: <100MB growth over 20 repeated operations

### Storage Performance
- **Parquet Compression**: 60-70% compression ratio vs CSV
- **Read Performance**: <0.1s per MB file size
- **Write Performance**: <0.2s per MB file size

### Coverage Targets
- **Critical Components**: >90% code coverage
- **Core Data Processing**: >85% coverage
- **Analysis Modules**: >80% coverage
- **Integration Tests**: All major component interactions covered

## Best Practices

### Writing New Tests

1. **Use Existing Fixtures**: Leverage `mock_seifa_data`, `mock_health_data`, etc.
2. **Follow Australian Patterns**: Ensure test data reflects real Australian health characteristics
3. **Test Error Conditions**: Use `error_simulation` fixture for robust error testing
4. **Validate Performance**: Include timing assertions for performance-critical code
5. **Test Data Integrity**: Verify data consistency throughout processing pipelines

### Mock Data Generation

```python
def test_custom_component(mock_seifa_data, australian_health_config):
    """Example test using framework fixtures."""
    # Generate realistic SEIFA data
    seifa_df = mock_seifa_data(num_areas=100, with_missing_data=True)
    
    # Verify Australian patterns
    sa2_codes = seifa_df["sa2_code_2021"].to_list()
    assert all(len(code) == 9 for code in sa2_codes)
    assert all(code[0] in "12345678" for code in sa2_codes)
```

### Performance Testing

```python
def test_component_performance(mock_health_data, performance_benchmarks):
    """Example performance test."""
    large_df = mock_health_data(num_records=10000, num_sa2_areas=500)
    
    start_time = time.time()
    result = process_health_data(large_df)
    processing_time = time.time() - start_time
    
    # Validate against benchmarks
    assert processing_time < 30.0
    assert len(result) > 0
```

### Integration Testing

```python
def test_end_to_end_pipeline(integration_test_data):
    """Example integration test."""
    # Get consistent test dataset
    integrated_data = integration_test_data(num_sa2_areas=50)
    
    # Test component interactions
    seifa_results = process_seifa(integrated_data["seifa"])
    health_results = process_health(integrated_data["health"])
    
    # Verify data consistency
    assert set(seifa_results["sa2_code_2021"]) & set(health_results["sa2_code"])
```

## Coverage Analysis

### Generating Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# View coverage in browser
open htmlcov/index.html

# Terminal coverage summary
pytest --cov=src --cov-report=term-missing

# XML coverage for CI/CD
pytest --cov=src --cov-report=xml
```

### Coverage Interpretation

- **Green**: >90% coverage (excellent)
- **Yellow**: 70-90% coverage (good, aim for improvement)
- **Red**: <70% coverage (needs attention)

Focus coverage improvements on:
1. Critical data processing paths
2. Error handling and validation logic
3. Risk calculation algorithms
4. Storage optimization routines

## Continuous Integration

### GitHub Actions Integration

```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: pytest --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v3
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
```

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Ensure package is installed in development mode
pip install -e .
```

**Fixture Not Found**:
```bash
# Check conftest.py is in tests/ directory
# Verify fixture names match usage
```

**Performance Test Failures**:
```bash
# Run on consistent hardware
# Adjust benchmarks for CI environment
# Check for resource contention
```

**Memory Test Failures**:
```bash
# Run tests individually to isolate memory leaks
# Use memory profiler for detailed analysis
pytest tests/test_storage/test_memory_optimizer.py -v -s
```

### Debugging Test Failures

1. **Increase Verbosity**: Use `-v -s` flags
2. **Isolate Tests**: Run single test methods
3. **Check Fixtures**: Verify mock data generation
4. **Memory Profiling**: Use built-in `memory_profiler` fixture
5. **Performance Analysis**: Check system resources during test runs

## Contributing to Tests

### Adding New Test Files

1. Follow naming convention: `test_{component_name}.py`
2. Include comprehensive docstrings
3. Use existing fixtures where possible
4. Add performance assertions for optimization components
5. Include both success and failure scenarios

### Extending Fixtures

```python
# In conftest.py
@pytest.fixture
def mock_new_component_data(australian_health_config):
    """Generate mock data for new component."""
    def generate_data(param1, param2):
        # Implementation
        return data
    return generate_data
```

### Adding Performance Benchmarks

```python
# In performance_benchmarks fixture
"new_component": {
    "max_processing_time": 10.0,
    "max_memory_usage": 200.0,
    "min_throughput": 1000
}
```

This comprehensive testing framework ensures the Australian Health Analytics platform meets enterprise-grade quality and performance standards while providing extensive coverage of Australian health data processing scenarios.