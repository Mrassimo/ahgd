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
)


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
                
                # Extract data
                total_records = 0
                for batch in self.extract(source, **kwargs):
                    total_records += len(batch)
                    self._processing_metadata.records_processed = total_records
                    
                    # Report progress
                    if self._progress_callback:
                        self._progress_callback(
                            total_records,
                            self._source_metadata.row_count or 0,
                            f"Extracted {total_records} records"
                        )
                    
                    # Create checkpoint
                    if total_records % self._checkpoint_interval == 0:
                        self._create_checkpoint(total_records)
                    
                    yield batch
                
                # Mark as completed
                self._processing_metadata.mark_completed()
                self.logger.info(
                    f"Extraction completed: {total_records} records processed"
                )
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
    
    def _create_checkpoint(self, records_processed: int) -> Dict[str, Any]:
        """
        Create a checkpoint for resumability.
        
        Args:
            records_processed: Number of records processed so far
            
        Returns:
            Dict[str, Any]: Checkpoint data
        """
        checkpoint = {
            'extractor_id': self.extractor_id,
            'records_processed': records_processed,
            'timestamp': datetime.now().isoformat(),
            'source_checksum': self._source_metadata.checksum if self._source_metadata else None
        }
        
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