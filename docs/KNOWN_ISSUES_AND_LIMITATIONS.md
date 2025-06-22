# AHGD Known Issues and Limitations

This document outlines known issues, limitations, and workarounds for the Australian Health Geography Data (AHGD) ETL pipeline. We maintain transparency about these limitations to help users make informed decisions about data usage.

## Table of Contents

1. [Data Source Limitations](#data-source-limitations)
2. [Geographic Limitations](#geographic-limitations)
3. [Technical Limitations](#technical-limitations)
4. [Statistical Limitations](#statistical-limitations)
5. [Performance Considerations](#performance-considerations)
6. [Known Bugs](#known-bugs)
7. [Workarounds and Solutions](#workarounds-and-solutions)
8. [Future Improvements](#future-improvements)

## Data Source Limitations

### 1. Temporal Inconsistencies

**Issue**: Different data sources update at different frequencies
- AIHW health indicators: Annual updates (12-18 month lag)
- ABS census data: 5-year cycles
- BOM climate data: Real-time to monthly
- Medicare/PBS: Quarterly with 6-month lag

**Impact**: Integrated datasets may combine data from different time periods

**Workaround**: 
```bash
# Use temporal alignment feature
ahgd-transform --temporal-align nearest --reference-year 2021
```

### 2. Geographic Coverage Gaps

**Issue**: Not all datasets cover all 2,473 SA2 areas
- Remote areas often have suppressed data (privacy)
- Some health services not available in all areas
- Climate stations sparse in remote regions

**Impact**: Missing values in integrated datasets

**Current Handling**: 
- Imputation for minor gaps (<5% missing)
- Flag missing data with quality indicators
- Exclude areas with >20% missing data

### 3. Small Area Suppression

**Issue**: Australian privacy regulations require suppression of small counts
- Health data suppressed when n<5
- Some demographic data suppressed in small populations
- Service utilisation data may be aggregated to SA3 level

**Impact**: Loss of granularity in sparsely populated areas

**Mitigation**: 
```python
# Check suppression impact
from src.utils.privacy import assess_suppression_impact
impact_report = assess_suppression_impact(data)
print(f"Suppressed cells: {impact_report['suppressed_count']}")
print(f"Affected SA2s: {impact_report['affected_areas']}")
```

## Geographic Limitations

### 1. Boundary Changes

**Issue**: SA2 boundaries change between census years
- 2016 to 2021 boundary changes affect ~10% of SA2s
- Historical comparisons require correspondence tables
- Some SA2s split or merge

**Current Support**: 
- 2021 boundaries (current default)
- 2016 boundaries (with --legacy flag)
- Correspondence tables for 2016-2021 transition

**Limitation**: Pre-2016 data requires manual mapping

### 2. Coordinate Reference Systems

**Issue**: Mixed CRS usage across data sources
- Official standard: GDA2020 (EPSG:7844)
- Legacy data: GDA94 (EPSG:4283)
- Web mapping: WGS84 (EPSG:4326)

**Impact**: ~1-2 metre discrepancies if not handled correctly

**Automatic Handling**:
```python
# CRS conversion handled automatically
from src.utils.geographic_utils import standardise_crs
standardised_gdf = standardise_crs(geodataframe, target_crs='EPSG:7844')
```

### 3. Offshore Territories

**Issue**: Limited support for external territories
- Christmas Island, Cocos Islands: Limited data availability
- Norfolk Island: Partial coverage only
- Antarctic territories: Not included

**Workaround**: Focus on mainland and Tasmania only
```bash
ahgd-extract --exclude-territories external
```

## Technical Limitations

### 1. Memory Requirements

**Issue**: Full pipeline requires significant memory
- Peak usage: ~16GB for national processing
- SA2 boundary processing: ~8GB
- Climate interpolation: ~12GB

**Minimum Requirements**:
- 8GB RAM: State-level processing
- 16GB RAM: National processing
- 32GB RAM: Historical analysis (5+ years)

**Memory Optimisation**:
```bash
# Enable memory-efficient processing
export AHGD_BATCH_SIZE=100
export AHGD_MAX_WORKERS=2
ahgd-pipeline --low-memory-mode
```

### 2. Processing Time

**Issue**: Large datasets take significant time
- Full national pipeline: 2-4 hours
- Climate interpolation: 30-60 minutes
- Geographic validation: 20-40 minutes

**Performance Tips**:
```bash
# Parallel processing (requires more memory)
export AHGD_MAX_WORKERS=8

# Skip validation for development
ahgd-pipeline --skip-validation

# Process single state
ahgd-pipeline --filter-state NSW
```

### 3. Dependency Conflicts

**Issue**: Some optional dependencies may conflict
- GeoPandas vs Fiona versions
- NumPy compatibility with some statistical packages

**Solution**: Use provided environment files
```bash
# Use exact versions from lock file
pip install -r requirements-lock.txt

# Or use conda environment
conda env create -f environment.yml
```

## Statistical Limitations

### 1. Age Standardisation

**Issue**: Different sources use different standard populations
- AIHW: 2001 Australian standard population
- WHO indicators: World standard population
- Some sources: No standardisation

**Impact**: Direct comparison between sources may be misleading

**Recommendation**: Re-standardise using consistent population
```python
from src.utils.statistical_methods import age_standardise
standardised_rate = age_standardise(
    crude_rate, 
    age_distribution,
    standard_population='Australian_2021'
)
```

### 2. Confidence Intervals

**Issue**: Not all sources provide confidence intervals
- Generated CIs assume normal distribution
- Small areas may have wide intervals
- Composite indicators lack proper CIs

**Limitation**: Statistical significance testing limited

### 3. Ecological Fallacy

**Issue**: SA2-level data cannot infer individual-level relationships
- Aggregated data masks within-area variation
- Correlation ≠ causation at area level
- Simpson's paradox may occur

**Important**: Always acknowledge ecological nature of analyses

## Performance Considerations

### 1. Large File Handling

**Issue**: Some outputs can be very large
- National GeoJSON: >500MB
- Full time series: >2GB per year
- Uncompressed CSV: >1GB

**Recommendations**:
```bash
# Use compressed formats
ahgd-loader --format parquet --compress snappy

# Partition large datasets
ahgd-loader --partition-by state,year

# Stream processing for large files
ahgd-transform --streaming-mode
```

### 2. Network Limitations

**Issue**: Data downloads can be slow/fail
- AIHW rate limits: 100 requests/minute
- Large files may timeout
- Network interruptions cause failures

**Built-in Handling**:
- Automatic retry with exponential backoff
- Resume capability for large downloads
- Local caching of downloaded data

### 3. Disk Space

**Issue**: Full pipeline requires substantial disk space
- Raw data: ~10GB
- Processed data: ~15GB
- Temporary files: ~20GB during processing

**Space Saving**:
```bash
# Clean temporary files
ahgd-pipeline --clean-temp

# Remove raw data after processing
ahgd-pipeline --no-keep-raw

# Use compression
ahgd-loader --compress-all
```

## Known Bugs

### 1. Loguru Timestamp Error

**Status**: Non-critical
**Version**: All versions
**Issue**: `KeyError: '"timestamp"'` in logs
**Impact**: Cosmetic only - doesn't affect data processing
**Workaround**: Ignore or disable formatted logging
```bash
export AHGD_SIMPLE_LOGS=true
```

### 2. Memory Leak in Climate Interpolation

**Status**: Under investigation
**Version**: <1.2.0
**Issue**: Memory usage grows with large climate datasets
**Impact**: May cause OOM errors for national processing
**Workaround**: Process in state batches
```bash
for state in NSW VIC QLD SA WA TAS NT ACT; do
    ahgd-transform --filter-state $state --dataset climate
done
```

### 3. SA2 Name Encoding

**Status**: Fixed in v1.2.1
**Issue**: Special characters in SA2 names cause encoding errors
**Impact**: CSV export may fail for certain areas
**Solution**: Update to latest version or use UTF-8 encoding
```bash
export PYTHONIOENCODING=utf-8
```

## Workarounds and Solutions

### Handling Missing Data

```python
# Option 1: Imputation
from src.utils.imputation import smart_impute
imputed_df = smart_impute(df, method='spatial_average')

# Option 2: Exclusion
complete_df = df.dropna(subset=['critical_field'])

# Option 3: Flag and retain
df['missing_flag'] = df['value'].isna()
```

### Temporal Alignment

```python
# Align multiple datasets to common year
from src.utils.temporal import align_temporal
aligned_data = align_temporal(
    datasets=[health_df, census_df, climate_df],
    target_year=2021,
    method='nearest'  # or 'interpolate'
)
```

### Geographic Matching

```python
# Fuzzy matching for inconsistent SA2 names
from src.utils.geographic_matching import fuzzy_match_sa2
matched_df = fuzzy_match_sa2(
    df, 
    sa2_column='area_name',
    threshold=0.9
)
```

## Future Improvements

### Planned Enhancements

1. **Real-time Data Integration** (v2.0)
   - Streaming updates from source APIs
   - Incremental processing pipeline
   - Live dashboard capabilities

2. **Machine Learning Integration** (v2.1)
   - Automated anomaly detection
   - Predictive health indicators
   - Smart imputation models

3. **Enhanced Geographic Support** (v2.2)
   - Historical boundary support (pre-2016)
   - Mesh block level processing
   - 3D visualisation support

4. **Performance Optimisation** (v2.3)
   - GPU acceleration for spatial operations
   - Distributed processing support
   - Cloud-native architecture

### Community Requests

Track and vote on feature requests:
- GitHub Issues: [AHGD Feature Requests](https://github.com/ahgd/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement)
- Discussions: [AHGD Discussions](https://github.com/ahgd/discussions)

## Reporting Issues

### How to Report

1. Check this document first
2. Search existing issues on GitHub
3. Provide minimal reproducible example
4. Include system information:

```bash
ahgd-diagnose --system-info > diagnosis.txt
```

### What to Include

- AHGD version: `ahgd --version`
- Python version: `python --version`
- Operating system and version
- Full error message and traceback
- Steps to reproduce
- Sample data (if applicable)

## Support

### Community Support
- GitHub Discussions: Questions and answers
- Stack Overflow: Tag with `ahgd` and `australian-health-data`

### Commercial Support
- Contact: support@ahgd.org.au
- Response time: 2 business days
- Priority fixes for critical issues

---

*This document is updated with each release. Last updated: Version 1.3.0*

For the latest version, see: [GitHub - Known Issues](https://github.com/ahgd/blob/main/docs/KNOWN_ISSUES_AND_LIMITATIONS.md)