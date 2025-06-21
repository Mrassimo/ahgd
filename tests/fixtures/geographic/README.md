# Geographic Test Fixtures

This directory contains reference data fixtures for testing the AHGD geographic standardisation pipeline. These files provide known test data with expected mappings and transformations for comprehensive testing of the SA2 standardisation functionality.

## Files Overview

### `sample_sa2_boundaries.geojson`
- **Format**: GeoJSON FeatureCollection
- **CRS**: GDA2020 (EPSG:7844)
- **Content**: 12 sample SA2 boundaries representing major Australian cities and territories
- **Purpose**: Testing boundary processing, spatial indexing, and topology validation
- **Features**:
  - Complete SA hierarchy (SA2 → SA3 → SA4 → State)
  - Realistic area and population data
  - Simplified rectangular geometries for testing
  - Coverage of all Australian states and territories

### `sample_postcode_mappings.csv`
- **Format**: CSV with headers
- **Content**: 35+ postcode to SA2 correspondence records
- **Purpose**: Testing postcode to SA2 mapping with population weighting
- **Key Features**:
  - 1:1 mappings (single SA2 per postcode)
  - 1:many mappings (postcodes spanning multiple SA2s)
  - Population-weighted and area-weighted allocation factors
  - Confidence scores and mapping methods
  - Coverage of major Australian postcodes
  - Error handling test cases

### `sample_coordinate_transformations.csv`
- **Format**: CSV with headers
- **Content**: 50+ coordinate transformation test cases
- **Purpose**: Testing coordinate system transformations to GDA2020
- **Key Features**:
  - Multiple input coordinate systems (WGS84, GDA94, GDA2020)
  - Expected transformation results
  - MGA zone assignments
  - Transformation accuracy estimates
  - Famous Australian landmarks for reference
  - All MGA zones (49-56) represented
  - Edge cases and error conditions

### `sample_lga_mappings.csv`
- **Format**: CSV with headers
- **Content**: 50+ Local Government Area to SA2 mappings
- **Purpose**: Testing LGA to SA2 correspondence with area weighting
- **Key Features**:
  - Major Australian cities and councils
  - Area-weighted allocation factors
  - Complex 1:many relationships
  - Special areas (unincorporated, Aboriginal land, islands)
  - Population and area overlap data
  - Confidence scores by mapping method

## Data Characteristics

### Geographic Coverage
- **States**: All 8 Australian states and territories represented
- **Cities**: Major capital cities and regional centres
- **Areas**: Urban, rural, remote, and special areas
- **Coordinates**: Span all MGA zones (49-56)

### Test Scenarios
1. **Simple Mappings**: Direct 1:1 relationships
2. **Complex Mappings**: 1:many relationships with allocation factors
3. **Edge Cases**: Boundary conditions and invalid data
4. **Error Handling**: Invalid codes and out-of-range coordinates
5. **Performance**: Large datasets for stress testing

### Data Quality
- **Accuracy**: Based on real Australian geography
- **Consistency**: Internally consistent hierarchies and relationships
- **Completeness**: All required fields populated
- **Validation**: Passes geographic validation rules

## Usage in Tests

### Unit Tests
```python
# Load sample data
with open('tests/fixtures/geographic/sample_postcode_mappings.csv') as f:
    test_data = pd.read_csv(f)

# Test specific scenarios
single_mapping = test_data[test_data['POSTCODE'] == '4000']
multi_mapping = test_data[test_data['POSTCODE'] == '2000']
```

### Integration Tests
```python
# Test complete pipeline with fixture data
boundaries = load_fixture('sample_sa2_boundaries.geojson')
postcodes = load_fixture('sample_postcode_mappings.csv')
coordinates = load_fixture('sample_coordinate_transformations.csv')
```

### Performance Tests
```python
# Generate large datasets based on fixture patterns
large_dataset = replicate_fixture_pattern('sample_postcode_mappings.csv', 10000)
```

## Data Sources and Attribution

The test data is derived from publicly available Australian Bureau of Statistics (ABS) data sources:
- **SA2 Boundaries**: Based on ABS Statistical Area Level 2 boundaries (2021)
- **Postcode Correspondences**: Based on ABS Postcode to SA2 correspondence files
- **Coordinate Systems**: Using official Australian coordinate system definitions
- **LGA Mappings**: Based on ABS Local Government Area boundaries

## Maintenance

### Updating Test Data
When updating test fixtures:
1. Maintain internal consistency across all files
2. Preserve test scenario coverage
3. Update documentation if new test cases are added
4. Validate against current ABS standards

### Version Control
- **Current Version**: Based on 2021 Census boundaries
- **Update Frequency**: As needed for new test requirements
- **Backward Compatibility**: Maintain for existing tests

## Notes

- All coordinates are in decimal degrees (GDA2020)
- Population and area figures are simplified for testing
- Some boundaries are simplified rectangular shapes for clarity
- Error test cases are clearly marked with "ERROR" or "Invalid" prefixes
- All allocation factors within each group sum to 1.0 (±0.01 tolerance)

## Contact

For questions about the test fixtures or to request additional test scenarios, please contact the AHGD development team.