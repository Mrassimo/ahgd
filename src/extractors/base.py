"""
Abstract base class for data extractors in the AHGD ETL pipeline.

This module provides the BaseExtractor class which defines the standard interface
for all data extraction components.
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
import logging

from ..utils.interfaces import (
    AuditTrail,
    DataBatch,
    DataRecord,
    ExtractionError,
    ProcessingMetadata,
    ProcessingStatus,
    ProgressCallback,
    SourceMetadata,
    ValidationError,
)
from ..validators import ValidationOrchestrator, QualityChecker
from ..pipelines.validation_pipeline import ValidationMode


class BaseExtractor(ABC):
    """
    Abstract base class for data extractors.
    
    This class provides the standard interface and common functionality
    for all data extraction components in the AHGD ETL pipeline.
    """
    
    def __init__(
        self,
        extractor_id: str,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialise the extractor.
        
        Args:
            extractor_id: Unique identifier for this extractor
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.extractor_id = extractor_id
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Retry configuration
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        self.retry_backoff = config.get('retry_backoff', 2.0)
        
        # Batch processing configuration
        self.batch_size = config.get('batch_size', 1000)
        
        # Progress tracking
        self._progress_callback: Optional[ProgressCallback] = None
        self._checkpoint_interval = config.get('checkpoint_interval', 1000)
        self._last_checkpoint: Optional[Dict[str, Any]] = None
        
        # Metadata tracking
        self._source_metadata: Optional[SourceMetadata] = None
        self._processing_metadata: Optional[ProcessingMetadata] = None
        self._audit_trail: Optional[AuditTrail] = None
        
        # Validation configuration
        self.validation_enabled = config.get('validation_enabled', True)
        self.validation_mode = ValidationMode(config.get('validation_mode', 'selective'))
        self.quality_threshold = config.get('quality_threshold', 95.0)
        self.halt_on_validation_failure = config.get('halt_on_validation_failure', False)
        
        # Initialise validation components
        if self.validation_enabled:
            self.validation_orchestrator = ValidationOrchestrator()
            self.quality_checker = QualityChecker()
        else:
            self.validation_orchestrator = None
            self.quality_checker = None
    
    @abstractmethod
    def extract(
        self,
        source: Union[str, Path, Dict[str, Any]],
        **kwargs
    ) -> Iterator[DataBatch]:
        """
        Extract data from the source.
        
        Args:
            source: Source specification (URL, file path, or configuration)
            **kwargs: Additional extraction parameters
            
        Yields:
            DataBatch: Batches of extracted records
            
        Raises:
            ExtractionError: If extraction fails
        """
        pass
    
    @abstractmethod
    def get_source_metadata(
        self,
        source: Union[str, Path, Dict[str, Any]]
    ) -> SourceMetadata:
        """
        Get metadata about the data source.
        
        Args:
            source: Source specification
            
        Returns:
            SourceMetadata: Metadata about the source
            
        Raises:
            ExtractionError: If metadata extraction fails
        """
        pass
    
    @abstractmethod
    def validate_source(
        self,
        source: Union[str, Path, Dict[str, Any]]
    ) -> bool:
        """
        Validate that the source is accessible and valid.
        
        Args:
            source: Source specification
            
        Returns:
            bool: True if source is valid
        """
        pass
    
    def validate_extracted_data(
        self,
        data_batch: DataBatch,
        validation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate extracted data batch.
        
        Args:
            data_batch: Batch of extracted data to validate
            validation_context: Optional validation context
            
        Returns:
            Dict containing validation results
            
        Raises:
            ValidationError: If validation fails critically
        """
        if not self.validation_enabled or not self.validation_orchestrator:
            return {"validation_enabled": False, "passed": True}
        
        try:
            # Convert batch to DataFrame for validation
            import pandas as pd
            if hasattr(data_batch, 'records'):
                df = pd.DataFrame([record.__dict__ if hasattr(record, '__dict__') else record 
                                 for record in data_batch.records])
            else:
                df = pd.DataFrame(data_batch)
            
            # Run validation orchestrator
            validation_result = self.validation_orchestrator.validate_data(
                df,
                context=validation_context or {}
            )
            
            # Check quality threshold
            quality_score = getattr(validation_result, 'overall_quality_score', 100.0)
            passed_threshold = quality_score >= self.quality_threshold
            
            # Prepare validation results
            validation_summary = {
                "validation_enabled": True,
                "quality_score": quality_score,
                "quality_threshold": self.quality_threshold,
                "passed_threshold": passed_threshold,
                "validation_mode": self.validation_mode.value,
                "errors": getattr(validation_result, 'errors', []),
                "warnings": getattr(validation_result, 'warnings', []),
                "validation_timestamp": datetime.now().isoformat()
            }
            
            # Handle validation failures
            if not passed_threshold and self.halt_on_validation_failure:
                error_msg = f"Data validation failed: quality score {quality_score:.2f}% below threshold {self.quality_threshold}%"
                self.logger.error(error_msg)
                raise ValidationError(error_msg)
            elif not passed_threshold:
                self.logger.warning(
                    f"Data validation warning: quality score {quality_score:.2f}% below threshold {self.quality_threshold}%"
                )
            
            self.logger.debug(
                f"Data validation completed: quality score {quality_score:.2f}%, passed: {passed_threshold}"
            )
            
            return validation_summary
            
        except ValidationError:
            raise
        except Exception as e:
            error_msg = f"Validation execution failed: {str(e)}"
            self.logger.error(error_msg)
            if self.halt_on_validation_failure:
                raise ValidationError(error_msg) from e
            else:
                return {
                    "validation_enabled": True,
                    "validation_failed": True,
                    "error": str(e),
                    "passed": False
                }
    
    def validate_source_schema(
        self,
        source: Union[str, Path, Dict[str, Any]],
        expected_schema: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Validate source data schema compatibility.
        
        Args:
            source: Source specification
            expected_schema: Expected schema definition
            
        Returns:
            bool: True if schema is compatible
        """
        if not self.validation_enabled:
            return True
        
        try:
            # Get source metadata
            source_metadata = self.get_source_metadata(source)
            
            # Basic schema validation
            if expected_schema:
                # Check if required fields are available
                if hasattr(source_metadata, 'columns'):
                    available_columns = set(source_metadata.columns or [])
                    required_columns = set(expected_schema.get('required_fields', []))
                    
                    if not required_columns.issubset(available_columns):
                        missing_columns = required_columns - available_columns
                        self.logger.warning(
                            f"Source schema validation warning: missing columns {missing_columns}"
                        )
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Source schema validation failed: {str(e)}")
            return False
    
    def extract_with_retry(
        self,
        source: Union[str, Path, Dict[str, Any]],
        progress_callback: Optional[ProgressCallback] = None,
        **kwargs
    ) -> Iterator[DataBatch]:
        """
        Extract data with retry logic and progress tracking.
        
        Args:
            source: Source specification
            progress_callback: Optional progress callback
            **kwargs: Additional extraction parameters
            
        Yields:
            DataBatch: Batches of extracted records
            
        Raises:
            ExtractionError: If extraction fails after all retries
        """
        self._progress_callback = progress_callback
        
        # Initialize processing metadata
        self._processing_metadata = ProcessingMetadata(
            operation_id=f"{self.extractor_id}_{int(time.time())}",
            operation_type="extraction",
            status=ProcessingStatus.RUNNING,
            start_time=datetime.now()
        )
        
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries:
            try:
                # Get source metadata
                self._source_metadata = self.get_source_metadata(source)
                
                # Validate source
                if not self.validate_source(source):
                    raise ExtractionError(f"Source validation failed: {source}")
                
                # Validate source schema if enabled
                if self.validation_enabled:
                    self.validate_source_schema(source, kwargs.get('expected_schema'))
                
                # Extract data
                total_records = 0
                total_validation_failures = 0
                for batch in self.extract(source, **kwargs):
                    batch_size = len(batch)
                    total_records += batch_size
                    self._processing_metadata.records_processed = total_records
                    
                    # Validate extracted batch if enabled
                    validation_result = None
                    if self.validation_enabled:
                        try:
                            validation_result = self.validate_extracted_data(
                                batch,
                                validation_context={
                                    'extractor_id': self.extractor_id,
                                    'batch_number': total_records // self.batch_size,
                                    'total_records': total_records
                                }
                            )
                            
                            if not validation_result.get('passed_threshold', True):
                                total_validation_failures += 1
                                
                        except ValidationError as ve:
                            if self.halt_on_validation_failure:
                                raise ExtractionError(f"Extraction halted due to validation failure: {str(ve)}") from ve
                            else:
                                self.logger.warning(f"Validation failed for batch but continuing: {str(ve)}")
                                total_validation_failures += 1
                    
                    # Report progress
                    if self._progress_callback:
                        progress_message = f"Extracted {total_records} records"
                        if validation_result:
                            quality_score = validation_result.get('quality_score', 100.0)
                            progress_message += f" (Quality: {quality_score:.1f}%)"
                        
                        self._progress_callback(
                            total_records,
                            self._source_metadata.row_count or 0,
                            progress_message
                        )
                    
                    # Create checkpoint
                    if total_records % self._checkpoint_interval == 0:
                        checkpoint_data = {
                            'total_records': total_records,
                            'validation_failures': total_validation_failures
                        }
                        if validation_result:
                            checkpoint_data['last_validation_result'] = validation_result
                        self._create_checkpoint(total_records, checkpoint_data)
                    
                    yield batch
                
                # Mark as completed
                self._processing_metadata.mark_completed()
                
                # Log completion with validation statistics
                completion_message = f"Extraction completed: {total_records} records processed"
                if self.validation_enabled and total_validation_failures > 0:
                    validation_failure_rate = (total_validation_failures / (total_records // self.batch_size)) * 100 if total_records > 0 else 0
                    completion_message += f", {total_validation_failures} validation failures ({validation_failure_rate:.1f}%)"
                
                self.logger.info(completion_message)
                return
                
            except Exception as e:
                last_error = e
                retry_count += 1
                
                if retry_count <= self.max_retries:
                    delay = self.retry_delay * (self.retry_backoff ** (retry_count - 1))
                    self.logger.warning(
                        f"Extraction failed (attempt {retry_count}/{self.max_retries}): {e}. "
                        f"Retrying in {delay} seconds..."
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(f"Extraction failed after {self.max_retries} retries: {e}")
        
        # Mark as failed
        if self._processing_metadata:
            self._processing_metadata.mark_failed(str(last_error))
        
        raise ExtractionError(f"Extraction failed after {self.max_retries} retries: {last_error}")
    
    def resume_extraction(
        self,
        source: Union[str, Path, Dict[str, Any]],
        checkpoint: Dict[str, Any],
        progress_callback: Optional[ProgressCallback] = None,
        **kwargs
    ) -> Iterator[DataBatch]:
        """
        Resume extraction from a checkpoint.
        
        Args:
            source: Source specification
            checkpoint: Checkpoint data
            progress_callback: Optional progress callback
            **kwargs: Additional extraction parameters
            
        Yields:
            DataBatch: Batches of extracted records
        """
        self._last_checkpoint = checkpoint
        self._progress_callback = progress_callback
        
        self.logger.info(f"Resuming extraction from checkpoint: {checkpoint}")
        
        # Delegate to the concrete implementation
        yield from self._resume_from_checkpoint(source, checkpoint, **kwargs)
    
    def get_checksum(self, file_path: Path) -> str:
        """
        Calculate checksum for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: SHA256 checksum
        """
        if self._source_metadata:
            return self._source_metadata.calculate_checksum(file_path)
        
        # Fallback to create temporary metadata
        temp_metadata = SourceMetadata(
            source_id="temp",
            source_type="file"
        )
        return temp_metadata.calculate_checksum(file_path)
    
    def get_audit_trail(self) -> Optional[AuditTrail]:
        """
        Get the audit trail for the extraction.
        
        Returns:
            AuditTrail: Audit trail information
        """
        if not self._audit_trail and self._source_metadata and self._processing_metadata:
            self._audit_trail = AuditTrail(
                operation_id=self._processing_metadata.operation_id,
                operation_type=self._processing_metadata.operation_type,
                source_metadata=self._source_metadata,
                processing_metadata=self._processing_metadata
            )
        
        return self._audit_trail
    
    def _create_checkpoint(self, records_processed: int, additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a checkpoint for resumability.
        
        Args:
            records_processed: Number of records processed so far
            additional_data: Additional data to include in checkpoint
            
        Returns:
            Dict[str, Any]: Checkpoint data
        """
        checkpoint = {
            'extractor_id': self.extractor_id,
            'records_processed': records_processed,
            'timestamp': datetime.now().isoformat(),
            'source_checksum': self._source_metadata.checksum if self._source_metadata else None
        }
        
        # Add validation data if provided
        if additional_data:
            checkpoint.update(additional_data)
        
        self._last_checkpoint = checkpoint
        self.logger.debug(f"Created checkpoint: {checkpoint}")
        
        return checkpoint
    
    def _resume_from_checkpoint(
        self,
        source: Union[str, Path, Dict[str, Any]],
        checkpoint: Dict[str, Any],
        **kwargs
    ) -> Iterator[DataBatch]:
        """
        Resume extraction from a specific checkpoint.
        
        This method should be implemented by concrete extractors
        to support resumable extraction.
        
        Args:
            source: Source specification
            checkpoint: Checkpoint data
            **kwargs: Additional extraction parameters
            
        Yields:
            DataBatch: Batches of extracted records
        """
        # Default implementation - just extract from the beginning
        self.logger.warning("Resumable extraction not implemented, starting from beginning")
        yield from self.extract(source, **kwargs)


# Import datetime at the top to avoid circular import issues
from datetime import datetime