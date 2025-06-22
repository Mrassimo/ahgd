# AHGD Dataset Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the Australian Health and Geographic Data (AHGD) dataset to Hugging Face Hub with comprehensive monitoring and analytics.

## Prerequisites

### 1. Environment Setup
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt
pip install datasets

# Verify installation
python -c "import datasets, huggingface_hub; print('Dependencies installed successfully')"
```

### 2. Hugging Face Authentication
```bash
# Login to Hugging Face Hub
huggingface-cli login

# Verify authentication
huggingface-cli whoami
```

### 3. Repository Access
Ensure you have write access to the target repository: `massomo/ahgd`

## Deployment Process

### Step 1: Pre-Deployment Verification
```bash
# Verify all dataset files are ready
ls -la data_exports/huggingface_dataset/

# Expected files:
# - README.md
# - USAGE_GUIDE.md
# - ahgd_data.csv
# - ahgd_data.geojson
# - ahgd_data.json
# - ahgd_data.parquet
# - data_dictionary.json
# - dataset_metadata.json
# - examples/ (directory)
```

### Step 2: Deploy Dataset
```bash
# Run the deployment script
python scripts/deploy_to_huggingface.py --deploy --monitor

# This will:
# 1. Create the repository if it doesn't exist
# 2. Upload all dataset files
# 3. Create enhanced dataset card
# 4. Set up monitoring configuration
```

### Step 3: Verify Deployment
```bash
# Run comprehensive verification
python scripts/verify_deployment.py --detailed

# Check specific aspects
python scripts/verify_deployment.py --repo-id massomo/ahgd --output verification_results.json
```

### Step 4: Initialise Monitoring
```bash
# Set up analytics and monitoring
python -c "
from src.monitoring.analytics import create_monitoring_system
analytics, feedback = create_monitoring_system('massomo/ahgd')
print('Monitoring system initialised')
"
```

## Post-Deployment Checklist

### Immediate Verification
- [ ] Repository exists and is public
- [ ] All file formats are accessible
- [ ] Dataset can be loaded with `datasets.load_dataset('massomo/ahgd')`
- [ ] README displays correctly on Hugging Face
- [ ] Metadata is complete and accurate
- [ ] Example code works as documented

### Long-term Monitoring
- [ ] Usage analytics are being collected
- [ ] Quality metrics are being tracked
- [ ] User feedback system is active
- [ ] Automated alerts are configured

## Usage Examples

### Basic Dataset Loading
```python
from datasets import load_dataset
import pandas as pd

# Load the dataset
dataset = load_dataset("massomo/ahgd")

# Convert to pandas DataFrame
df = dataset['train'].to_pandas()

# Display basic information
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
```

### Format-Specific Loading
```python
import pandas as pd
from huggingface_hub import hf_hub_download

# Download specific format
csv_file = hf_hub_download(
    repo_id="massomo/ahgd",
    filename="ahgd_data.csv",
    repo_type="dataset"
)

# Load with pandas
df = pd.read_csv(csv_file)
```

### GIS Analysis
```python
import geopandas as gpd
from huggingface_hub import hf_hub_download

# Download GeoJSON file
geojson_file = hf_hub_download(
    repo_id="massomo/ahgd",
    filename="ahgd_data.geojson",
    repo_type="dataset"
)

# Load as GeoDataFrame
gdf = gpd.read_file(geojson_file)

# Perform spatial analysis
print(f"Coordinate system: {gdf.crs}")
print(f"Bounding box: {gdf.total_bounds}")
```

## Monitoring and Analytics

### Real-time Monitoring
```python
from src.monitoring.analytics import DatasetAnalytics

# Initialise analytics
analytics = DatasetAnalytics("massomo/ahgd")

# Collect current metrics
metrics = analytics.collect_huggingface_metrics()
print(f"Total downloads: {metrics.get('downloads', 0)}")

# Run quality checks
quality_metrics = analytics.run_quality_checks()
print(f"Quality checks completed: {len(quality_metrics)}")

# Generate usage report
report = analytics.generate_usage_report(days=30)
print(f"Usage report generated for {report['report_period']['days']} days")
```

### Dashboard Creation
```python
# Create monitoring dashboard data
dashboard_data = analytics.create_dashboard_data()

# Dashboard data includes:
# - Overview metrics
# - Current download statistics
# - Quality status
# - Recent activity
# - User feedback summary
```

### User Feedback Collection
```python
from src.monitoring.analytics import FeedbackCollector

# Initialise feedback collector
feedback = FeedbackCollector(analytics)

# Submit feedback (simulated)
feedback.submit_feedback(
    feedback_type="rating",
    rating=5,
    comment="Excellent dataset for health geography research"
)

# Get feedback summary
summary = feedback.get_feedback_summary(days=30)
print(f"Feedback summary: {summary}")
```

## Troubleshooting

### Common Issues

#### Authentication Problems
```bash
# Clear existing credentials
huggingface-cli logout

# Re-authenticate
huggingface-cli login

# Use write token if needed
huggingface-cli login --token YOUR_WRITE_TOKEN
```

#### Repository Access Issues
```bash
# Check repository exists
huggingface-cli repo info massomo/ahgd --repo-type dataset

# Create repository if needed
huggingface-cli repo create massomo/ahgd --type dataset --public
```

#### File Upload Failures
```bash
# Check file sizes
ls -lh data_exports/huggingface_dataset/

# Upload files individually if needed
huggingface-cli upload massomo/ahgd data_exports/huggingface_dataset/README.md --repo-type dataset
```

#### Dataset Loading Issues
```python
# Test with explicit configuration
from datasets import load_dataset, DatasetBuilder

# Check dataset configuration
builder = DatasetBuilder.from_dict({
    "name": "ahgd",
    "description": "Australian Health and Geographic Data"
})

# Load with error handling
try:
    dataset = load_dataset("massomo/ahgd")
    print("Dataset loaded successfully")
except Exception as e:
    print(f"Loading failed: {e}")
```

## Security and Compliance

### Data Privacy
- All data is aggregated to SA2 level (minimum 3,000 population)
- No individual-level information is included
- Geographic boundaries are publicly available
- Health indicators are from official sources

### Licensing
- Dataset: CC-BY-4.0 (Creative Commons Attribution)
- Code: MIT License
- Documentation: CC-BY-4.0

### Attribution Requirements
When using this dataset, please cite:
```
Australian Health and Geographic Data (AHGD) Dataset (2025)
Available at: https://huggingface.co/datasets/massomo/ahgd
License: CC-BY-4.0
```

## Maintenance and Updates

### Regular Maintenance Tasks
1. **Weekly**: Check download statistics and user feedback
2. **Monthly**: Run comprehensive quality checks
3. **Quarterly**: Update data if new sources are available
4. **Annually**: Review and update documentation

### Version Management
- Use semantic versioning (e.g., 1.0.0, 1.1.0)
- Document changes in README.md
- Maintain backwards compatibility
- Archive old versions if needed

### Quality Assurance
- Automated quality checks run daily
- Manual review of user feedback
- Regular validation against source data
- Performance monitoring of download speeds

## Support and Contact

### Getting Help
1. **Documentation**: Check this guide and README.md
2. **Issues**: Create issue on the Hugging Face repository
3. **Discussions**: Use Hugging Face discussions feature
4. **Email**: Contact maintainers for urgent issues

### Contributing
- Feedback and suggestions welcome
- Report data quality issues
- Suggest new features or improvements
- Contribute example code and analyses

---

**Last Updated**: June 22, 2025
**Version**: 1.0.0
**Status**: Production Ready