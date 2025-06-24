# TDD Real Data Extraction Tests

This directory contains comprehensive Test-Driven Development (TDD) tests to verify that AHGD extractors are production-ready and can download real Australian government data.

## Overview

The tests follow TDD methodology:
1. **Write failing tests first** that expect real data behavior
2. **Verify/fix extractors** to make tests pass  
3. **Ensure production readiness** with real data sources

## Test Files

### Core TDD Test Files

1. **`test_real_data_extractors.py`** - Primary TDD tests for real data extraction
   - Tests that extractors download real data, not demo data
   - Validates record counts meet Australian data expectations (2400+ SA2 areas, etc.)
   - Tests production deployment with `--force-real-data` flag
   - Verifies URL accessibility and data format validation

2. **`test_url_accessibility.py`** - URL accessibility and validation tests
   - Tests all configured ABS, AIHW, BOM URLs are accessible
   - Validates data format expectations (ZIP files, CSV, etc.)
   - Tests failover mechanisms and alternative URL discovery
   - Validates HTTPS usage where required

3. **`test_production_readiness.py`** - Production deployment readiness tests
   - Configuration validation for production environments
   - Australian data standards compliance (SA2 codes, GDA2020, etc.)
   - Error handling robustness (timeouts, HTTP errors, corrupted data)
   - Performance requirements and memory efficiency
   - Monitoring and observability features

4. **`test_extractor_production_suite.py`** - Comprehensive production test suite
   - End-to-end validation of all extractors
   - Real data capability testing where possible
   - Error resilience and fallback mechanisms
   - Scalability and performance characteristics

### Test Runner

5. **`test_real_data_runner.py`** - Production readiness test runner
   - Executable script for comprehensive production validation
   - Generates production readiness reports
   - Supports various testing modes

## Running the Tests

### Quick Start - Production Readiness Check

```bash
# Run comprehensive production readiness tests
python tests/test_real_data_runner.py --force-real-data

# Check URL accessibility only
python tests/test_real_data_runner.py --check-urls

# Validate configurations only
python tests/test_real_data_runner.py --validate-config
```

### Running Individual Test Suites

```bash
# Core TDD tests (requires network access)
pytest tests/unit/test_real_data_extractors.py -v -m "integration and network"

# URL accessibility tests
pytest tests/integration/test_url_accessibility.py -v -m "network"

# Production readiness tests  
pytest tests/integration/test_production_readiness.py -v -m "production"

# Comprehensive production suite
pytest tests/integration/test_extractor_production_suite.py -v -m "production"
```

### Test Markers

- `@pytest.mark.integration` - Integration tests requiring full system
- `@pytest.mark.network` - Tests requiring internet connectivity
- `@pytest.mark.production` - Production readiness tests
- `@pytest.mark.slow` - Tests that may take longer to complete

## Expected Data Volumes

The tests validate that extractors return expected data volumes for Australian datasets:

### ABS Data Expectations
- **SA2 Boundaries**: ~2,300+ areas (2021 Census)
- **SA3 Boundaries**: ~350+ areas
- **SA4 Boundaries**: ~100+ areas
- **Census G01 Data**: 2,400+ SA2 records
- **Total Population**: 24-28 million people (2021 Census)

### AIHW Data Expectations
- **Mortality Data**: Variable, depends on cause and geographic level
- **Hospital Data**: Variable, based on service type and area
- **Privacy Thresholds**: Minimum cell sizes enforced (typically 5+)

### BOM Data Expectations
- **Weather Stations**: 750+ active stations
- **Climate Data**: Daily observations for selected stations
- **Air Quality**: Major urban areas covered

## Production Deployment Validation

### URL Accessibility Requirements
- ✅ All ABS URLs accessible (HTTPS required)
- ✅ All AIHW URLs accessible (HTTPS required)  
- ✅ All BOM URLs accessible (HTTP acceptable for legacy system)
- ✅ Alternative URLs available for failover

### Data Quality Requirements
- ✅ SA2 codes follow 9-digit Australian standard
- ✅ Coordinate system uses GDA2020 (Australian standard)
- ✅ Population totals within expected ranges
- ✅ Geographic boundaries cover all of Australia

### Error Handling Requirements
- ✅ Network timeouts handled gracefully
- ✅ HTTP errors trigger fallback mechanisms
- ✅ Corrupted data detected and handled
- ✅ Partial data recovery implemented
- ✅ Demo data fallback when real data unavailable

### Performance Requirements
- ✅ Memory-efficient batch processing
- ✅ Extraction completes within timeout limits
- ✅ Concurrent extraction safety
- ✅ Progress tracking and checkpointing

## Configuration Validation

### Required Configuration Fields
- `timeout_seconds`: 10-600 seconds
- `retry_attempts`: 1-10 attempts  
- `batch_size`: > 0 records
- `coordinate_system`: "GDA2020" for ABS data

### URL Configuration Standards
- ABS URLs: Must use HTTPS and abs.gov.au domain
- AIHW URLs: Must use HTTPS and aihw.gov.au domain
- BOM URLs: HTTP acceptable for bom.gov.au domain
- All URLs must be accessible and return expected content types

## Interpreting Test Results

### Production Readiness Report

The test runner generates a comprehensive report with:

1. **URL Accessibility**: X/Y URLs accessible
2. **Data Extraction**: X/Y extractors successful  
3. **Record Count Validation**: X/Y within expected ranges
4. **Overall Assessment**:
   - ✅ **READY FOR PRODUCTION**: No issues detected
   - ⚠️ **MOSTLY READY**: Minor issues need attention
   - ❌ **NOT READY**: Significant issues detected

### Common Issues and Solutions

#### URL Accessibility Issues
- **Problem**: ABS URLs return 404/500 errors
- **Solution**: Update URLs in `configs/extractors/abs_sources.yaml`
- **Note**: ABS occasionally changes download URLs

#### Record Count Issues  
- **Problem**: Fewer records than expected
- **Solution**: Check if data source has been updated/restructured
- **Note**: Some variance is acceptable due to data updates

#### Network Timeout Issues
- **Problem**: Extractions timeout in network-constrained environments
- **Solution**: Increase timeout values in configuration
- **Note**: Production environments should have stable connectivity

#### Demo Data Fallback Issues
- **Problem**: Extractors always fall back to demo data
- **Solution**: Check URL accessibility and network connectivity
- **Note**: Demo fallback is intentional for development environments

## Continuous Integration Integration

### Pre-Deployment Tests
```bash
# Run before any production deployment
python tests/test_real_data_runner.py --force-real-data --report-file production_readiness.txt

# Check exit code
if [ $? -eq 0 ]; then
    echo "Production deployment approved"
else
    echo "Production deployment blocked - check report"
    exit 1
fi
```

### Monitoring Tests
```bash
# Regular production health checks
python tests/test_real_data_runner.py --check-urls --report-file url_health.txt

# Schedule this to run daily in production
```

## Development Workflow

1. **Write Failing Tests**: Add tests that expect real data behavior
2. **Run Tests**: `pytest tests/unit/test_real_data_extractors.py -v`
3. **Fix Extractors**: Update extractor code to pass tests
4. **Validate URLs**: Update configurations if URLs have changed
5. **Test Production**: Run full production suite before deployment

## Contributing

When adding new extractors or data sources:

1. Add URL accessibility tests to `test_url_accessibility.py`
2. Add real data extraction tests to `test_real_data_extractors.py`
3. Add production readiness tests to `test_production_readiness.py`
4. Update expected data volumes in this README
5. Test with `--force-real-data` flag before submitting

## Troubleshooting

### Test Environment Setup
```bash
# Ensure all dependencies installed
pip install -r requirements.txt

# Ensure configs are available
ls configs/extractors/

# Check network connectivity
curl -I https://www.abs.gov.au
```

### Common Test Failures

1. **"Real data extraction was not attempted"**
   - Extractor is falling back to demo data too quickly
   - Check URL accessibility and network connectivity

2. **"Record count outside expected range"**
   - Data source may have been updated
   - Check if ABS has released new Census/boundary data

3. **"URL not accessible"**
   - Government website may be down or URL changed
   - Check website manually and update configurations

4. **"Invalid data format"**
   - Data source may have changed format
   - Update field mappings in extractor configuration

This TDD approach ensures that extractors are genuinely production-ready and can handle real Australian government data reliably.