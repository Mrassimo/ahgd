# ETL Pipeline Orchestration Framework

This module provides a comprehensive ETL pipeline orchestration framework with checkpointing, recovery, and monitoring capabilities for the Australian Health and Geographic Data (AHGD) project.

## Features

- **Checkpoint and Recovery**: Automatic state persistence with multiple serialisation formats
- **Pipeline Orchestration**: Complex dependency management with parallel and sequential execution
- **Resource Management**: Intelligent resource allocation and monitoring
- **Comprehensive Monitoring**: Real-time metrics, alerting, and performance tracking
- **Stage Management**: Configurable execution with validation and retry logic
- **Web Integration**: Dashboard-ready metrics and status reporting

## Quick Start

### Basic Pipeline Creation

```python
from src.pipelines import BasePipeline, PipelineContext, StageConfig
from src.pipelines.stage import ExtractorStage, TransformerStage, LoaderStage

class HealthDataPipeline(BasePipeline):
    def define_stages(self):
        return ["extract", "transform", "validate", "load"]
    
    def execute_stage(self, stage_name: str, context: PipelineContext):
        if stage_name == "extract":
            # Extract health data
            return self.extract_health_data()
        elif stage_name == "transform":
            # Transform data
            return self.transform_data(context.get_output("extract"))
        # ... other stages

# Run pipeline
pipeline = HealthDataPipeline(
    name="health_pipeline",
    enable_checkpoints=True,
    parallel_stages=False
)

result = pipeline.run()
```

### Pipeline Orchestration

```python
from src.pipelines import PipelineOrchestrator, PipelineDefinition
from src.pipelines.orchestrator import ExecutionMode

# Create orchestrator
orchestrator = PipelineOrchestrator(
    name="ahgd_orchestrator",
    execution_mode=ExecutionMode.PARALLEL,
    max_workers=4
)

# Register pipelines
orchestrator.register_pipeline(PipelineDefinition(
    name="geographic_pipeline",
    pipeline_class=GeographicPipeline,
    dependencies=[],
    priority=10
))

orchestrator.register_pipeline(PipelineDefinition(
    name="health_pipeline", 
    pipeline_class=HealthDataPipeline,
    dependencies=["geographic_pipeline"],
    priority=8
))

# Run orchestration
result = orchestrator.run()
```

### Configuration-Driven Pipelines

```python
# Load from YAML configuration
orchestrator.register_from_config(Path("pipelines/health_data_pipeline.yaml"))

# Run specific pipelines
result = orchestrator.run(target_pipelines=["health_data_pipeline"])
```

### Monitoring and Alerting

```python
from src.pipelines.monitoring import PipelineMonitor, AlertSeverity

# Create monitor
monitor = PipelineMonitor(
    metrics_dir=Path("metrics"),
    enable_system_monitoring=True
)

# Start monitoring
monitor.start_system_monitoring()
monitor.start_pipeline_monitoring("health_pipeline")

# Get dashboard data
dashboard_data = monitor.get_dashboard_data()

# Export metrics
metrics_json = monitor.export_metrics(format="json")
```

### Checkpoint Management

```python
from src.pipelines.checkpointing import CheckpointManager, CheckpointFormat, RecoveryStrategy

# Create checkpoint manager
checkpoint_mgr = CheckpointManager(
    checkpoint_dir=Path("checkpoints"),
    default_format=CheckpointFormat.COMPRESSED_PICKLE
)

# Create checkpoint
metadata = checkpoint_mgr.create_checkpoint(
    pipeline_name="health_pipeline",
    stage_name="transform",
    data=transformed_data,
    version="v1.2.0"
)

# Load checkpoint
data, metadata = checkpoint_mgr.load_checkpoint(
    pipeline_name="health_pipeline",
    stage_name="transform",
    strategy=RecoveryStrategy.LATEST
)

# Resume pipeline from checkpoint
result = pipeline.run(resume_from="transform")
```

## Architecture

### Core Components

1. **BasePipeline**: Abstract base class for all pipelines
2. **PipelineOrchestrator**: Manages multiple pipeline execution
3. **CheckpointManager**: Handles state persistence and recovery
4. **PipelineStage**: Individual stage execution with monitoring
5. **PipelineMonitor**: Comprehensive monitoring and alerting

### Data Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Extractors    │───▶│   Transformers   │───▶│     Loaders     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Checkpoints   │    │    Validation    │    │   Monitoring    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Configuration

### Pipeline Configuration (YAML)

```yaml
pipeline:
  name: "example_pipeline"
  description: "Example ETL pipeline"
  
orchestration:
  execution_mode: "parallel"
  max_workers: 4
  timeout: "PT2H"

stages:
  - name: "extract_data"
    type: "extractor"
    class: "src.extractors.ExampleExtractor"
    config:
      source_url: "https://example.com/data"
    timeout: "PT30M"
    retry_attempts: 3
    resource_requirements:
      cpu: 1.0
      memory: 2048
```

### Stage Configuration

```python
stage_config = StageConfig(
    name="extract_health_data",
    description="Extract health indicators from AIHW",
    timeout=timedelta(minutes=30),
    retry_attempts=3,
    resource_usage=ResourceUsage.MEDIUM,
    validation_rules={
        "required_columns": ["year", "state", "indicator"],
        "min_records": 1000
    }
)
```

## Best Practices

### 1. Pipeline Design

- Keep stages small and focused
- Use meaningful stage names
- Implement proper error handling
- Design for idempotency

### 2. Checkpointing

- Enable checkpoints for long-running pipelines
- Use appropriate serialisation formats
- Regular checkpoint cleanup
- Test recovery procedures

### 3. Resource Management

- Set realistic resource requirements
- Monitor resource usage
- Use parallel execution judiciously
- Implement timeout handling

### 4. Monitoring

- Enable comprehensive monitoring
- Set appropriate alert thresholds
- Regular monitoring dashboard reviews
- Implement escalation procedures

## Integration with AHGD Components

### Extractors

```python
from src.extractors.aihw import AIHWExtractor

extractor_stage = ExtractorStage(
    config=stage_config,
    extractor_class=AIHWExtractor,
    source_type="mortality_data"
)
```

### Transformers

```python
from src.transformers.health import HealthTransformer

transformer_stage = TransformerStage(
    config=stage_config,
    transformer_class=HealthTransformer,
    standardisation_rules="aihw_standard"
)
```

### Validators

```python
from src.validators.health import HealthDataValidator

validator_stage = ValidatorStage(
    config=stage_config,
    validator_class=HealthDataValidator,
    schema_path="schemas/health_schema.json"
)
```

### Loaders

```python
from src.loaders.warehouse import DataWarehouseLoader

loader_stage = LoaderStage(
    config=stage_config,
    loader_class=DataWarehouseLoader,
    target_database="health_analytics"
)
```

## Performance Considerations

- **Memory**: Geographic data requires substantial memory (8-32GB)
- **CPU**: Parallel processing benefits from multiple cores
- **Disk**: Large intermediate files need adequate storage
- **Network**: Consider bandwidth for data downloads

## Error Handling

The framework provides multiple levels of error handling:

1. **Stage-level**: Retry logic with configurable delays
2. **Pipeline-level**: Checkpoint recovery and resume capability
3. **Orchestrator-level**: Dependency failure handling
4. **System-level**: Resource exhaustion and timeout handling

## Troubleshooting

### Common Issues

1. **Memory Errors**: Increase memory allocation or enable data streaming
2. **Timeout Errors**: Adjust timeout settings or optimise processing
3. **Checkpoint Corruption**: Validate checksums and use backup recovery
4. **Dependency Failures**: Check pipeline execution order and prerequisites

### Debugging

```python
# Enable debug logging
import logging
logging.getLogger("src.pipelines").setLevel(logging.DEBUG)

# Check pipeline status
status = pipeline.get_progress()
print(f"Pipeline state: {status['state']}")
print(f"Completed stages: {status['completed_stages']}")

# Validate checkpoints
result = checkpoint_mgr.validate_checkpoint(checkpoint_id)
if not result.is_valid:
    print(f"Validation errors: {result.errors}")
```

## Testing

The framework includes comprehensive test coverage:

```bash
# Run pipeline tests
pytest tests/unit/test_pipelines.py

# Run integration tests
pytest tests/integration/test_etl_pipeline.py

# Run with coverage
pytest --cov=src.pipelines tests/
```

## Contributing

When extending the pipeline framework:

1. Follow the established patterns and interfaces
2. Add comprehensive logging and monitoring
3. Include proper error handling and validation
4. Write tests for new functionality
5. Update documentation and examples