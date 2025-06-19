# Data Pipeline Fix Recommendations

## Critical Fixes Required

### 1. ZIP File Extraction Implementation

**File**: `src/data_processing/census_processor.py`

Add ZIP extraction method:
```python
import zipfile
from pathlib import Path

def extract_census_zips(self):
    """Extract census ZIP files before processing"""
    zip_files = [
        "2021_GCP_AUS_SA2.zip",
        "2021_GCP_NSW_SA2.zip", 
        "2021_GCP_VIC_SA2.zip",
        "2021_GCP_QLD_SA2.zip"
    ]
    
    extraction_dir = self.raw_dir / "extracted_census"
    extraction_dir.mkdir(exist_ok=True)
    
    for zip_name in zip_files:
        zip_path = self.raw_dir / zip_name
        if zip_path.exists():
            print(f"Extracting {zip_name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extraction_dir)
            print(f"✅ Extracted {zip_name}")
    
    return extraction_dir
```

**Integration**: Call `extract_census_zips()` before `load_census_datapack()`

### 2. Remove Health Data Artificial Limits

**File**: `src/data_processing/health_processor.py:207`

**Current (BROKEN)**:
```python
for csv_file in csv_files[:3]:  # Limit to first 3 files for performance
```

**Fix**:
```python
for csv_file in csv_files:  # Process ALL files
```

Add progress tracking:
```python
from tqdm import tqdm

for csv_file in tqdm(csv_files, desc="Processing health CSV files"):
    # Process each file with progress bar
```

### 3. Fix SEIFA Data Corruption

**File**: `src/data_processing/seifa_processor.py`

Add data validation:
```python
def validate_seifa_data(self, df):
    """Validate SEIFA data integrity"""
    expected_rows = 2454  # Expected SA2 count
    
    if len(df) < expected_rows * 0.9:  # Allow 10% tolerance
        raise ValueError(f"SEIFA data severely truncated: {len(df)} rows, expected ~{expected_rows}")
    
    # Check for binary corruption
    text_cols = ['SA2_NAME_2021', 'STE_NAME_2021']
    for col in text_cols:
        if col in df.columns:
            binary_count = df[col].str.contains(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', na=False).sum()
            if binary_count > 0:
                raise ValueError(f"Binary corruption detected in {col}: {binary_count} rows")
    
    return True
```

### 4. Update Main ETL Integration

**File**: `scripts/run_unified_etl.py`

Add census processing:
```python
def run_complete_pipeline():
    """Run complete data processing pipeline"""
    
    # 1. Extract ZIP files first
    census_processor = CensusProcessor()
    extraction_dir = census_processor.extract_census_zips()
    
    # 2. Process SEIFA with validation
    seifa_processor = SEIFAProcessor()
    seifa_data = seifa_processor.process_complete_pipeline()
    seifa_processor.validate_seifa_data(seifa_data)
    
    # 3. Process extracted census data
    census_data = census_processor.load_census_datapack(extraction_dir)
    
    # 4. Process ALL health data (remove limits)
    health_processor = HealthDataProcessor()
    health_data = health_processor.process_complete_pipeline()
    
    # 5. Integration and validation
    validate_data_volumes(seifa_data, census_data, health_data)
```

### 5. Add Data Volume Monitoring

**New File**: `src/monitoring/data_volume_monitor.py`

```python
def monitor_pipeline_data_loss():
    """Monitor data loss through pipeline stages"""
    
    # Raw data sizes
    raw_sizes = {
        'census_zips': get_zip_sizes(['2021_GCP_AUS_SA2.zip', '2021_GCP_NSW_SA2.zip']),
        'health_zips': get_zip_sizes(['mbs_demographics_historical_1993_2015.zip']),
        'seifa_files': get_file_sizes(['SEIFA_2021_SA2.xlsx'])
    }
    
    # Processed data sizes  
    processed_sizes = {
        'census_processed': get_parquet_sizes('census_*.parquet'),
        'health_processed': get_parquet_sizes('health_*.parquet'),
        'seifa_processed': get_parquet_sizes('seifa_*.parquet')
    }
    
    # Calculate retention rates
    for data_type in raw_sizes:
        raw_size = raw_sizes[data_type]
        processed_size = processed_sizes.get(f"{data_type.replace('_zips', '_processed').replace('_files', '_processed')}", 0)
        
        retention_rate = (processed_size / raw_size) * 100 if raw_size > 0 else 0
        
        if retention_rate < 50:  # Alert on >50% data loss
            print(f"⚠️  HIGH DATA LOSS in {data_type}: {retention_rate:.1f}% retention")
        else:
            print(f"✅ {data_type}: {retention_rate:.1f}% retention")
```

## Implementation Priority

### Phase 1: Critical Data Recovery (Week 1)
1. ✅ **ZIP Extraction**: Implement census ZIP extraction
2. ✅ **Health Data Limits**: Remove artificial processing limits  
3. ✅ **SEIFA Corruption**: Fix Parquet generation and validation

### Phase 2: Pipeline Integration (Week 2) 
4. ✅ **ETL Integration**: Update main scripts to use all processors
5. ✅ **Data Monitoring**: Add volume monitoring and loss alerts

### Phase 3: Validation & Testing (Week 3)
6. ✅ **End-to-End Testing**: Full pipeline with real data
7. ✅ **Performance Optimization**: Handle large datasets efficiently

## Expected Data Recovery

### Before Fixes
- **Raw Data**: 1.355GB
- **Processed Data**: 67MB  
- **Data Loss**: 95.1%

### After Fixes  
- **Raw Data**: 1.355GB
- **Processed Data**: 1.2GB+ (estimated)
- **Data Loss**: <15% (acceptable level)

### Specific Recoveries
- **Census Demographics**: 0MB → 800MB+ (complete recovery)
- **Health Historical**: 1.3MB → 250MB+ (19,000% improvement)
- **SEIFA Socio-economic**: 60KB → 5MB+ (8,000% improvement)

## Testing Strategy

### 1. Component Testing
```bash
# Test ZIP extraction
uv run python -c "from src.data_processing.census_processor import CensusProcessor; cp = CensusProcessor(); cp.extract_census_zips()"

# Test SEIFA validation  
uv run python -c "from src.data_processing.seifa_processor import SEIFAProcessor; sp = SEIFAProcessor(); sp.process_complete_pipeline()"

# Test health data processing
uv run python -c "from src.data_processing.health_processor import HealthDataProcessor; hp = HealthDataProcessor(); hp.process_complete_pipeline()"
```

### 2. Integration Testing
```bash
# Test complete pipeline
uv run python scripts/run_unified_etl.py

# Monitor data volumes
uv run python -c "from src.monitoring.data_volume_monitor import monitor_pipeline_data_loss; monitor_pipeline_data_loss()"
```

### 3. Validation Checks
```bash
# Validate record counts
uv run python -c "
import polars as pl
seifa = pl.read_parquet('data/processed/seifa_*.parquet')
print(f'SEIFA records: {len(seifa)} (expect ~2,454)')

census = pl.read_parquet('data/processed/census_*.parquet') 
print(f'Census records: {len(census)} (expect 100,000+)')

health = pl.read_parquet('data/processed/health_*.parquet')
print(f'Health records: {len(health)} (expect 500,000+)')
"
```

## Success Metrics

### Data Volume Targets
- **SEIFA**: 2,454 SA2 records (currently 287 corrupted)
- **Census**: 100,000+ demographic records (currently 0)
- **Health**: 500,000+ health records (currently minimal)
- **Overall Retention**: 85%+ (currently 4.9%)

### Quality Targets  
- **Data Corruption**: 0% (currently high in SEIFA)
- **Processing Success**: 95%+ (currently ~25%)
- **Pipeline Completeness**: 90%+ (currently ~30%)

## Risk Mitigation

### 1. Large Dataset Handling
- Implement streaming processing for multi-GB census data
- Use Polars lazy evaluation for memory efficiency
- Add progress bars for long-running operations

### 2. Error Recovery
- Graceful handling of corrupted individual files
- Partial processing success rather than complete failure
- Detailed logging of processing issues

### 3. Performance Optimization
- Process files in parallel where possible
- Use efficient Parquet compression
- Implement incremental processing for updates

---

*Fix recommendations generated: 2025-06-19*  
*Based on comprehensive pipeline analysis and DataPilot investigation*