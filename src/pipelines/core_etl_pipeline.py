"""
Core ETL pipeline for AHGD - Simplified SA1-focused implementation.

This module provides a streamlined ETL pipeline that processes Australian health
and geographic data with SA1 as the core geographic unit. It replaces the complex
master ETL pipeline with a simplified, maintainable approach.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Optional

import duckdb
import polars as pl

from ..extractors import ExtractorRegistry
from ..transformers.sa1_processor import SA1GeographicTransformer
from ..utils.interfaces import ExtractionError
from ..utils.interfaces import LoadingError
from ..utils.interfaces import TransformationError
from ..utils.logging import get_logger
from ..utils.logging import monitor_performance
from ..validators.core_validator import CoreValidator
from .base_pipeline import BasePipeline
from .base_pipeline import PipelineContext

logger = get_logger(__name__)


class PipelineStage(str, Enum):
    """Core pipeline stages."""

    EXTRACT = "extract"
    TRANSFORM = "transform"
    VALIDATE = "validate"
    LOAD = "load"


class PipelineStatus(str, Enum):
    """Pipeline execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class StageResult:
    """Result from pipeline stage execution."""

    stage: PipelineStage
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    records_processed: int = 0
    output_table: Optional[str] = None
    error: Optional[Exception] = None
    metadata: dict[str, Any] = None

    @property
    def duration(self) -> Optional[float]:
        """Calculate stage duration in seconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class CoreETLPipeline(BasePipeline):
    """
    Simplified core ETL pipeline focused on SA1 geographic processing.

    This pipeline implements a straightforward extraction -> transformation ->
    validation -> loading workflow without the complexity of the original
    master pipeline architecture.
    """

    def __init__(
        self,
        name: str = "core_etl_pipeline",
        db_path: str = "ahgd_sa1.db",
        config: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialise core ETL pipeline.

        Args:
            name: Pipeline name
            db_path: Path to DuckDB database
            config: Pipeline configuration
            **kwargs: Additional pipeline arguments
        """
        super().__init__(name, **kwargs)

        # Configuration
        self.config = config or {}
        self.db_path = db_path

        # Logger
        from ..utils.logging import get_logger

        self.logger = get_logger(self.__class__.__name__)

        # Pipeline components
        self.extractor_registry = ExtractorRegistry()
        self.sa1_transformer = SA1GeographicTransformer(self.config.get("transformer", {}))
        self.validator = CoreValidator(self.config.get("validator", {}), self.logger)

        # DuckDB connection
        self.con = duckdb.connect(database=self.db_path, read_only=False)

        # Pipeline state
        self.stage_results: dict[PipelineStage, StageResult] = {}
        self.current_table: Optional[str] = None

        # Configuration
        self.batch_size = self.config.get("batch_size", 1000)
        self.max_memory_gb = self.config.get("max_memory_gb", 4)
        self.parallel_processing = self.config.get("parallel_processing", False)

        logger.info(
            f"Core ETL pipeline initialised: {name}",
            db_path=db_path,
            batch_size=self.batch_size,
        )

    def define_stages(self) -> list[str]:
        """Define pipeline stages in execution order."""
        return [stage.value for stage in PipelineStage]

    @monitor_performance("core_etl_execution")
    def run_complete_etl(
        self,
        source_config: Optional[dict[str, Any]] = None,
        target_config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Execute the complete ETL pipeline from extraction to loading.

        Args:
            source_config: Source data configuration
            target_config: Target output configuration

        Returns:
            Dict containing pipeline execution results
        """
        logger.info("Starting complete ETL pipeline execution")

        try:
            # Create pipeline context
            context = self._create_context()
            if source_config:
                context.metadata["source_config"] = source_config
            if target_config:
                context.metadata["target_config"] = target_config

            # Execute pipeline stages in sequence
            self._execute_extraction_stage(context)
            self._execute_transformation_stage(context)
            self._execute_validation_stage(context)
            self._execute_loading_stage(context)

            # Generate results
            results = self._generate_pipeline_results(context)

            logger.info(
                "Complete ETL pipeline execution finished",
                status=results["status"],
                total_records=results.get("total_records", 0),
                duration=results.get("total_duration", 0),
            )

            return results

        except Exception as e:
            logger.error(f"ETL pipeline execution failed: {e!s}")
            self._mark_pipeline_failed(e)
            raise
        finally:
            self._cleanup()

    @monitor_performance("extraction_stage")
    def _execute_extraction_stage(self, context: PipelineContext) -> None:
        """Execute data extraction stage."""
        stage = PipelineStage.EXTRACT
        result = StageResult(stage=stage, status=PipelineStatus.RUNNING, start_time=datetime.now())

        try:
            logger.info("Executing extraction stage")

            # Get extraction configuration
            source_config = context.metadata.get("source_config", {})
            extractor_type = source_config.get("type", "aihw")

            # Get extractor
            extractor = self.extractor_registry.get_extractor(extractor_type)
            if not extractor:
                raise ExtractionError(f"No extractor found for type: {extractor_type}")

            # Extract data
            extracted_data = []
            batch_count = 0

            for batch in extractor.extract(source_config):
                if isinstance(batch, list):
                    extracted_data.extend(batch)
                else:
                    extracted_data.append(batch)

                batch_count += 1
                if batch_count % 10 == 0:
                    logger.info(f"Processed {batch_count} extraction batches")

            # Convert to DataFrame and store in DuckDB
            if extracted_data:
                df = pl.DataFrame(extracted_data)
                table_name = "extracted_data"
                self.con.register(table_name, df)
                self.current_table = table_name

                result.records_processed = len(df)
                result.output_table = table_name
            else:
                logger.warning("No data extracted")
                result.records_processed = 0

            result.status = PipelineStatus.COMPLETED
            result.end_time = datetime.now()

            logger.info(
                f"Extraction completed: {result.records_processed} records",
                duration=result.duration,
            )

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = e
            result.end_time = datetime.now()
            logger.error(f"Extraction stage failed: {e!s}")
            raise ExtractionError(f"Extraction failed: {e!s}") from e

        finally:
            self.stage_results[stage] = result

    @monitor_performance("transformation_stage")
    def _execute_transformation_stage(self, context: PipelineContext) -> None:
        """Execute SA1 geographic transformation stage."""
        stage = PipelineStage.TRANSFORM
        result = StageResult(stage=stage, status=PipelineStatus.RUNNING, start_time=datetime.now())

        try:
            logger.info("Executing SA1 transformation stage")

            if not self.current_table:
                raise TransformationError("No data available for transformation")

            # Get data from DuckDB
            input_data = self.con.table(self.current_table).pl()

            # Apply SA1 geographic transformation
            transformed_data = self.sa1_transformer.transform(input_data)

            # Store transformed data
            table_name = "transformed_data"
            self.con.register(table_name, transformed_data)
            self.current_table = table_name

            result.records_processed = len(transformed_data)
            result.output_table = table_name
            result.status = PipelineStatus.COMPLETED
            result.end_time = datetime.now()

            logger.info(
                f"SA1 transformation completed: {result.records_processed} records",
                duration=result.duration,
            )

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = e
            result.end_time = datetime.now()
            logger.error(f"Transformation stage failed: {e!s}")
            raise TransformationError(f"SA1 transformation failed: {e!s}") from e

        finally:
            self.stage_results[stage] = result

    @monitor_performance("validation_stage")
    def _execute_validation_stage(self, context: PipelineContext) -> None:
        """Execute data validation stage."""
        stage = PipelineStage.VALIDATE
        result = StageResult(stage=stage, status=PipelineStatus.RUNNING, start_time=datetime.now())

        try:
            logger.info("Executing validation stage")

            if not self.current_table:
                raise Exception("No data available for validation")

            # Get data from DuckDB
            data = self.con.table(self.current_table).pl()

            # Validate data using CoreValidator
            validation_results = self.validator.validate_sa1_data(data)

            # Check if validation passed
            if not validation_results.get("overall_valid", False):
                error_count = validation_results.get("error_count", 0)
                logger.warning(f"Validation found {error_count} errors")

                # Depending on configuration, either fail or continue with warnings
                validation_mode = self.config.get("validation_mode", "warn")
                if validation_mode == "strict" and error_count > 0:
                    raise Exception(f"Strict validation failed with {error_count} errors")

            result.records_processed = len(data)
            result.metadata = validation_results
            result.status = PipelineStatus.COMPLETED
            result.end_time = datetime.now()

            logger.info(
                f"Validation completed: {result.records_processed} records validated",
                validation_score=validation_results.get("quality_score", 0),
                duration=result.duration,
            )

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = e
            result.end_time = datetime.now()
            logger.error(f"Validation stage failed: {e!s}")
            # Don't raise - validation failures can be warnings

        finally:
            self.stage_results[stage] = result

    @monitor_performance("loading_stage")
    def _execute_loading_stage(self, context: PipelineContext) -> None:
        """Execute data loading stage."""
        stage = PipelineStage.LOAD
        result = StageResult(stage=stage, status=PipelineStatus.RUNNING, start_time=datetime.now())

        try:
            logger.info("Executing loading stage")

            if not self.current_table:
                raise LoadingError("No data available for loading")

            # Get data from DuckDB
            final_data = self.con.table(self.current_table).pl()

            # Get target configuration
            target_config = context.metadata.get("target_config", {})
            output_path = target_config.get("output_path", "output/sa1_processed_data.parquet")
            output_format = target_config.get("format", "parquet")

            # Load data to target
            self._load_data_to_target(final_data, output_path, output_format)

            # Store final table in DuckDB for future access
            final_table_name = "final_sa1_data"
            self.con.register(final_table_name, final_data)

            result.records_processed = len(final_data)
            result.output_table = final_table_name
            result.metadata = {"output_path": output_path, "format": output_format}
            result.status = PipelineStatus.COMPLETED
            result.end_time = datetime.now()

            logger.info(
                f"Loading completed: {result.records_processed} records saved to {output_path}",
                duration=result.duration,
            )

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = e
            result.end_time = datetime.now()
            logger.error(f"Loading stage failed: {e!s}")
            raise LoadingError(f"Data loading failed: {e!s}") from e

        finally:
            self.stage_results[stage] = result

    def _load_data_to_target(self, data: pl.DataFrame, output_path: str, format: str) -> None:
        """Load data to target destination."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "parquet":
            data.write_parquet(output_path)
        elif format.lower() == "csv":
            data.write_csv(output_path)
        elif format.lower() == "json":
            data.write_json(output_path)
        else:
            raise LoadingError(f"Unsupported output format: {format}")

    def _generate_pipeline_results(self, context: PipelineContext) -> dict[str, Any]:
        """Generate comprehensive pipeline execution results."""
        total_duration = sum(result.duration or 0 for result in self.stage_results.values())

        total_records = 0
        final_table = None
        overall_status = PipelineStatus.COMPLETED

        for stage, result in self.stage_results.items():
            if result.status == PipelineStatus.FAILED:
                overall_status = PipelineStatus.FAILED
            if result.records_processed:
                total_records = max(total_records, result.records_processed)
            if stage == PipelineStage.LOAD and result.output_table:
                final_table = result.output_table

        return {
            "pipeline_id": context.pipeline_id,
            "run_id": context.run_id,
            "status": overall_status.value,
            "total_records": total_records,
            "total_duration": total_duration,
            "final_table": final_table,
            "stage_results": {
                stage.value: {
                    "status": result.status.value,
                    "records_processed": result.records_processed,
                    "duration": result.duration,
                    "error": str(result.error) if result.error else None,
                }
                for stage, result in self.stage_results.items()
            },
            "execution_summary": self._generate_execution_summary(),
        }

    def _generate_execution_summary(self) -> dict[str, Any]:
        """Generate execution summary statistics."""
        completed_stages = sum(
            1 for result in self.stage_results.values() if result.status == PipelineStatus.COMPLETED
        )
        failed_stages = sum(
            1 for result in self.stage_results.values() if result.status == PipelineStatus.FAILED
        )

        total_stages = len(self.stage_results)
        success_rate = (completed_stages / total_stages * 100) if total_stages > 0 else 0

        return {
            "total_stages": total_stages,
            "completed_stages": completed_stages,
            "failed_stages": failed_stages,
            "success_rate": success_rate,
            "pipeline_efficiency": (
                "high" if success_rate >= 100 else "medium" if success_rate >= 75 else "low"
            ),
        }

    def _mark_pipeline_failed(self, error: Exception) -> None:
        """Mark pipeline as failed due to unrecoverable error."""
        for stage in PipelineStage:
            if stage not in self.stage_results:
                self.stage_results[stage] = StageResult(
                    stage=stage,
                    status=PipelineStatus.FAILED,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    error=error,
                )

    def _cleanup(self) -> None:
        """Clean up pipeline resources."""
        try:
            if self.con:
                self.con.close()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e!s}")

    def get_pipeline_status(self) -> dict[str, Any]:
        """Get current pipeline status and progress."""
        return {
            "pipeline_name": self.name,
            "current_table": self.current_table,
            "stage_results": {
                stage.value: {
                    "status": result.status.value,
                    "records_processed": result.records_processed,
                    "duration": result.duration,
                }
                for stage, result in self.stage_results.items()
            },
        }

    # Implement required abstract methods from BasePipeline
    def execute_stage(self, stage_name: str, context: PipelineContext) -> Any:
        """Execute individual pipeline stage."""
        stage = PipelineStage(stage_name)

        if stage == PipelineStage.EXTRACT:
            self._execute_extraction_stage(context)
        elif stage == PipelineStage.TRANSFORM:
            self._execute_transformation_stage(context)
        elif stage == PipelineStage.VALIDATE:
            self._execute_validation_stage(context)
        elif stage == PipelineStage.LOAD:
            self._execute_loading_stage(context)

        return self.stage_results.get(stage)


# Convenience functions for pipeline execution


def run_sa1_etl_pipeline(
    source_config: Optional[dict[str, Any]] = None,
    target_config: Optional[dict[str, Any]] = None,
    pipeline_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Convenience function to run the complete SA1 ETL pipeline.

    Args:
        source_config: Source data configuration
        target_config: Target output configuration
        pipeline_config: Pipeline execution configuration

    Returns:
        Pipeline execution results
    """
    pipeline = CoreETLPipeline(config=pipeline_config)
    try:
        return pipeline.run_complete_etl(source_config, target_config)
    finally:
        pipeline._cleanup()


def create_sa1_pipeline(name: str = "sa1_etl", **kwargs) -> CoreETLPipeline:
    """
    Create and configure an SA1-focused ETL pipeline.

    Args:
        name: Pipeline name
        **kwargs: Pipeline configuration options

    Returns:
        Configured CoreETLPipeline instance
    """
    return CoreETLPipeline(name=name, **kwargs)
