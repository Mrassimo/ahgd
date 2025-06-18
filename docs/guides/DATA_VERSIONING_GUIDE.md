# Data Versioning with DVC

## Overview

This project uses DVC (Data Version Control) to manage large datasets and create reproducible data processing pipelines for the Australian Health Geography Data (AHGD) analytics project.

## Data Structure

### Raw Data (1.4GB)
- **Health Data**: AIHW mortality, hospital admissions, disease prevalence
- **Geographic Data**: SA2 boundaries, postcode mappings, spatial data
- **Demographic Data**: ABS Census data, population statistics
- **Socioeconomic Data**: SEIFA indices, income, education statistics

### Processed Data
- **Cleaned datasets**: Standardised, validated, analysis-ready
- **Merged datasets**: Integrated data across sources
- **Analysis database**: SQLite database for dashboard queries

## DVC Pipeline

### Stages

1. **download_data**: Download raw data from Australian government sources
2. **process_aihw_data**: Clean and standardise AIHW health data
3. **process_demographics**: Process ABS Census demographic data
4. **process_geographic**: Process SA2 boundary and geographic data
5. **process_socioeconomic**: Process SEIFA socioeconomic indices
6. **create_analysis_database**: Create consolidated SQLite database
7. **health_correlation_analysis**: Perform correlation analysis

### Running the Pipeline

```bash
# Run the entire pipeline
dvc repro

# Run specific stage
dvc repro process_aihw_data

# Check pipeline status
dvc status

# View pipeline DAG
dvc dag
```

## Data Versioning Commands

### Basic Operations

```bash
# Check data status
dvc status

# Push data to remote
dvc push

# Pull data from remote
dvc pull

# Add new data file
dvc add data/new_file.csv
git add data/new_file.csv.dvc
git commit -m "Add new data file"

# Get specific version
git checkout <commit-hash>
dvc checkout
```

### Remote Storage

Currently configured with local remote storage. For production:

```bash
# Add S3 remote
dvc remote add -d s3remote s3://bucket/path

# Add Azure remote
dvc remote add -d azureremote azure://container/path

# Add GCS remote
dvc remote add -d gcsremote gs://bucket/path
```

## Best Practices

### Data Management
1. **Version Control**: Track data changes with meaningful commit messages
2. **Documentation**: Update data dictionaries when schema changes
3. **Validation**: Run data quality checks before committing
4. **Backup**: Regular pushes to remote storage

### Pipeline Management
1. **Atomic Stages**: Each stage should be independently runnable
2. **Dependencies**: Clearly define input/output dependencies
3. **Parameters**: Use params.yaml for configurable settings
4. **Metrics**: Track pipeline performance and data quality metrics

### Collaboration
1. **Shared Remote**: Use cloud storage for team collaboration
2. **Data Lineage**: Document data sources and transformations
3. **Reproducibility**: Pin dependency versions
4. **Testing**: Validate pipeline outputs

## Monitoring and Metrics

### Data Quality Metrics
- **Completeness**: Percentage of non-null values
- **Consistency**: Data format and type validation
- **Accuracy**: Cross-validation against known sources
- **Timeliness**: Data freshness indicators

### Pipeline Metrics
- **Processing Time**: Stage execution duration
- **Memory Usage**: Peak memory consumption
- **Error Rates**: Failed processing attempts
- **Data Volume**: Input/output dataset sizes

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce chunk_size in params.yaml
2. **Slow Processing**: Enable parallel processing
3. **Missing Dependencies**: Check dvc.yaml stage definitions
4. **Corrupted Data**: Validate checksums with `dvc status`

### Recovery Procedures

1. **Reset Pipeline**: `dvc repro --force`
2. **Restore Data**: `dvc checkout --force`
3. **Clean Cache**: `dvc cache dir` and manual cleanup
4. **Rebuild Database**: `dvc repro create_analysis_database --force`

## Configuration Files

- **dvc.yaml**: Pipeline definition
- **params.yaml**: Configuration parameters
- **.dvc/config**: DVC settings
- **data/*.dvc**: Data version files

## Integration with CI/CD

The DVC pipeline integrates with the project's CI/CD system:

1. **Automated Testing**: Pipeline validation on commits
2. **Data Validation**: Quality checks on new data
3. **Performance Monitoring**: Track pipeline metrics
4. **Deployment**: Automated data updates in production

## Security and Compliance

### Data Protection
- **Encryption**: Data encrypted at rest and in transit
- **Access Control**: Role-based permissions
- **Audit Trail**: Complete data lineage tracking
- **Anonymisation**: PII removed from processed datasets

### Compliance
- **Data Retention**: Automated cleanup of old versions
- **Documentation**: Comprehensive metadata
- **Backup**: Regular data backups
- **Disaster Recovery**: Tested recovery procedures

## Future Enhancements

1. **Cloud Integration**: Migration to AWS/Azure/GCP
2. **Real-time Processing**: Streaming data pipeline
3. **ML Integration**: Model training and deployment
4. **Advanced Analytics**: Time series and predictive models
5. **Data Catalogue**: Searchable data inventory