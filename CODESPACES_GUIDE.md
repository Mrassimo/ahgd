# Running AHGD ETL in GitHub Codespaces

GitHub Codespaces provides a complete cloud development environment with:
- ✅ 32GB storage (plenty for full ABS dataset)
- ✅ 4-core CPU, 8GB RAM
- ✅ Pre-configured Python environment
- ✅ No local storage needed!

## Quick Start

### 1. Open in Codespaces

Click the green "Code" button on GitHub and select "Open with Codespaces"

### 2. Wait for Environment Setup

The container will automatically:
- Install all Python dependencies
- Set up the development environment
- Display a welcome message when ready

### 3. Create Mock Data (for testing)

```bash
# Create comprehensive mock data
python create_mock_data.py
```

### 4. Run the ETL Pipeline

```bash
# Run all dimensions with mock data
python -c "
from ahgd_etl.transformers.geo import GeographyTransformer
from ahgd_etl.models import *
from ahgd_etl.utils import setup_logging

# Setup
logger = setup_logging('codespaces_etl', log_file=False)

# Geographic dimension
print('Processing geographic dimension...')
geo = GeographyTransformer()
geo_df = geo.transform_all()
geo.save_to_parquet(geo_df)

# Time dimension
print('Processing time dimension...')
time = TimeDimensionBuilder(2020, 2025)
time_df = time.build()
time.save_to_parquet(time_df)

# Other dimensions
print('Processing other dimensions...')
health = HealthConditionDimensionBuilder()
health.save_to_parquet(health.build(), 'dim_health_condition.parquet')

demo = DemographicDimensionBuilder()
demo.save_to_parquet(demo.build(), 'dim_demographic.parquet')

char = PersonCharacteristicDimensionBuilder()
char.save_to_parquet(char.build(), 'dim_person_characteristic.parquet')

print('✅ All dimensions processed!')
"
```

### 5. Check Results

```bash
# List output files
ls -lh output/

# View a sample
python -c "
import polars as pl
df = pl.read_parquet('output/geo_dimension.parquet')
print(df.head())
"
```

## Working with Real ABS Data

### Option 1: Upload via Codespaces

1. Download ABS files to your local machine
2. Drag and drop into the Codespaces file explorer
3. Move to appropriate directories:
   ```bash
   mv *.zip data/raw/geographic/
   ```

### Option 2: Direct Download (if you have URLs)

```bash
# Example for geographic data
cd data/raw/geographic
wget -O sa1_2021.zip "YOUR_ABS_URL_HERE"
unzip sa1_2021.zip
```

### Option 3: Use GitHub Releases

1. Create a release with data files
2. Download in Codespaces:
   ```bash
   gh release download v1.0 --dir data/raw/
   ```

## Storage Management

Check available space:
```bash
df -h /workspaces
```

Clean up after processing:
```bash
# Remove raw data after processing
rm -rf data/raw/census/extracted/
# Keep only output files
```

## Persistent Storage

Your output files persist in the Codespace. To save permanently:

```bash
# Create a GitHub release with output
cd output
zip -r ahgd_output.zip *.parquet
gh release create v1.0 ahgd_output.zip --title "AHGD ETL Output"

# Or commit to a data branch
git checkout -b data-output
git add output/*.parquet
git commit -m "Add ETL output files"
git push origin data-output
```

## Tips

1. **Free Tier**: You get 120 core-hours/month free
2. **Auto-stop**: Codespaces stop after 30 min inactivity
3. **Prebuilds**: Enable for faster starts
4. **Secrets**: Store ABS credentials as Codespace secrets

## Full ETL Command (when implemented)

```bash
# Future unified command
python run_unified_etl.py --steps all
```

## Troubleshooting

- **Out of memory**: Process census files in chunks
- **Slow downloads**: Use GitHub releases for data files
- **Connection lost**: Codespaces auto-save your work

---

💡 **Pro tip**: Pin your Codespace to prevent deletion and keep your processed data ready!