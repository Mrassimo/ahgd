"""
Abstract base class for data loaders in the AHGD ETL pipeline.

This module provides the BaseLoader class which defines the standard interface
for all data loading and export components.
"""

import gzip
import json
import sqlite3
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
import logging
import pandas as pd

from ..utils.interfaces import (
    AuditTrail,
    DataBatch,
    DataFormat,
    DataPartition,
    LoadingError,
    ProcessingMetadata,
    ProcessingStatus,
    ProgressCallback,
)


class CompressionType:
    """Supported compression types."""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    XZ = "xz"


class PartitionStrategy:
    """Data partitioning strategies."""
    NONE = "none"
    BY_SIZE = "by_size"
    BY_COUNT = "by_count"
    BY_DATE = "by_date"
    BY_COLUMN = "by_column"
    BY_HASH = "by_hash"


class BaseLoader(ABC):
    """
    Abstract base class for data loaders.
    
    This class provides the standard interface and common functionality
    for all data loading and export components in the AHGD ETL pipeline.
    """
    
    def __init__(
        self,
        loader_id: str,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialise the loader.
        
        Args:
            loader_id: Unique identifier for this loader
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.loader_id = loader_id
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Output configuration
        self.output_format = DataFormat(config.get('output_format', DataFormat.CSV.value))
        self.output_path = Path(config.get('output_path', './output'))
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Compression configuration
        self.compression = config.get('compression', CompressionType.NONE)
        self.compression_level = config.get('compression_level', 6)
        
        # Partitioning configuration
        self.partition_strategy = config.get('partition_strategy', PartitionStrategy.NONE)
        self.partition_size = config.get('partition_size', 10000)  # records
        self.partition_column = config.get('partition_column')
        
        # Optimisation configuration
        self.batch_size = config.get('batch_size', 1000)
        self.enable_indexing = config.get('enable_indexing', True)
        self.optimise_for_read = config.get('optimise_for_read', True)
        
        # Version management
        self.enable_versioning = config.get('enable_versioning', True)
        self.version_column = config.get('version_column', 'data_version')
        self.current_version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Processing metadata
        self._processing_metadata: Optional[ProcessingMetadata] = None
        self._audit_trail: Optional[AuditTrail] = None
        self._partitions: List[DataPartition] = []
        
        # Progress tracking
        self._progress_callback: Optional[ProgressCallback] = None
    
    @abstractmethod
    def load(
        self,
        data: Union[DataBatch, Iterator[DataBatch]],
        destination: Optional[Union[str, Path, Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Load data to the destination.
        
        Args:
            data: Data to load (batch or iterator of batches)
            destination: Optional destination override
            **kwargs: Additional loading parameters
            
        Returns:
            Dict[str, Any]: Loading results and metadata
            
        Raises:
            LoadingError: If loading fails
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[DataFormat]:
        """
        Get the list of supported output formats.
        
        Returns:
            List[DataFormat]: Supported formats
        """
        pass
    
    def load_with_optimisation(
        self,
        data: Union[DataBatch, Iterator[DataBatch]],
        destination: Optional[Union[str, Path, Dict[str, Any]]] = None,
        progress_callback: Optional[ProgressCallback] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Load data with full optimisation and monitoring.
        
        Args:
            data: Data to load
            destination: Optional destination override
            progress_callback: Optional progress callback
            **kwargs: Additional loading parameters
            
        Returns:
            Dict[str, Any]: Loading results and metadata
        """
        self._progress_callback = progress_callback
        
        # Initialise processing metadata
        self._processing_metadata = ProcessingMetadata(
            operation_id=f"{self.loader_id}_{int(time.time())}",
            operation_type="loading",
            status=ProcessingStatus.RUNNING,
            start_time=datetime.now(),
            parameters=kwargs
        )
        
        try:
            # Process data based on input type
            if isinstance(data, list):
                # Single batch
                result = self._load_single_batch(data, destination, **kwargs)
            else:
                # Iterator of batches
                result = self._load_batch_iterator(data, destination, **kwargs)
            
            # Apply post-loading optimisations
            if self.optimise_for_read:
                self._optimise_for_read(result.get('output_path'))
            
            # Update metadata
            self._processing_metadata.mark_completed()
            
            # Add metadata to result
            result.update({
                'processing_metadata': self._processing_metadata,
                'partitions': self._partitions,
                'version': self.current_version
            })
            
            self.logger.info(
                f"Loading completed: {self._processing_metadata.records_processed} records"
            )
            
            return result
            
        except Exception as e:
            if self._processing_metadata:
                self._processing_metadata.mark_failed(str(e))
            self.logger.error(f"Loading failed: {e}")
            raise LoadingError(f"Loading failed: {e}") from e
    
    def export_multi_format(
        self,
        data: DataBatch,
        base_filename: str,
        formats: List[DataFormat],
        **kwargs
    ) -> Dict[DataFormat, Dict[str, Any]]:
        """
        Export data to multiple formats simultaneously.
        
        Args:
            data: Data to export
            base_filename: Base filename (without extension)
            formats: List of formats to export to
            **kwargs: Additional export parameters
            
        Returns:
            Dict[DataFormat, Dict[str, Any]]: Export results by format
        """
        results = {}
        
        for format_type in formats:
            try:
                # Temporarily set format
                original_format = self.output_format
                self.output_format = format_type
                
                # Create format-specific filename
                extension = self._get_file_extension(format_type)
                filename = f"{base_filename}.{extension}"
                
                # Export data
                result = self.load(data, filename, **kwargs)
                results[format_type] = result
                
                # Restore original format
                self.output_format = original_format
                
            except Exception as e:
                self.logger.error(f"Export to {format_type.value} failed: {e}")
                results[format_type] = {'error': str(e)}
        
        return results
    
    def partition_data(
        self,
        data: DataBatch,
        strategy: Optional[str] = None
    ) -> List[DataBatch]:
        """
        Partition data according to the configured strategy.
        
        Args:
            data: Data to partition
            strategy: Optional strategy override
            
        Returns:
            List[DataBatch]: List of data partitions
        """
        partition_strategy = strategy or self.partition_strategy
        
        if partition_strategy == PartitionStrategy.NONE:
            return [data]
        
        elif partition_strategy == PartitionStrategy.BY_COUNT:
            return self._partition_by_count(data)
        
        elif partition_strategy == PartitionStrategy.BY_SIZE:
            return self._partition_by_size(data)
        
        elif partition_strategy == PartitionStrategy.BY_DATE:
            return self._partition_by_date(data)
        
        elif partition_strategy == PartitionStrategy.BY_COLUMN:
            return self._partition_by_column(data)
        
        elif partition_strategy == PartitionStrategy.BY_HASH:
            return self._partition_by_hash(data)
        
        else:
            self.logger.warning(f"Unknown partition strategy: {partition_strategy}")
            return [data]
    
    def compress_output(
        self,
        file_path: Path,
        compression_type: Optional[str] = None
    ) -> Path:
        """
        Compress output file.
        
        Args:
            file_path: Path to file to compress
            compression_type: Optional compression type override
            
        Returns:
            Path: Path to compressed file
        """
        compression = compression_type or self.compression
        
        if compression == CompressionType.NONE:
            return file_path
        
        elif compression == CompressionType.GZIP:
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb', compresslevel=self.compression_level) as f_out:
                    f_out.writelines(f_in)
            
            # Remove original file
            file_path.unlink()
            return compressed_path
        
        else:
            self.logger.warning(f"Compression type {compression} not implemented")
            return file_path
    
    def create_version(
        self,
        data: DataBatch,
        version_info: Optional[Dict[str, Any]] = None
    ) -> DataBatch:
        """
        Add version information to data.
        
        Args:
            data: Input data
            version_info: Optional version information
            
        Returns:
            DataBatch: Data with version information
        """
        if not self.enable_versioning:
            return data
        
        versioned_data = []
        version_info = version_info or {
            'version': self.current_version,
            'created_at': datetime.now().isoformat(),
            'loader_id': self.loader_id
        }
        
        for record in data:
            versioned_record = record.copy()
            versioned_record[self.version_column] = version_info
            versioned_data.append(versioned_record)
        
        return versioned_data
    
    def get_audit_trail(self) -> Optional[AuditTrail]:
        """
        Get the audit trail for the loading operation.
        
        Returns:
            AuditTrail: Audit trail information
        """
        if not self._audit_trail and self._processing_metadata:
            self._audit_trail = AuditTrail(
                operation_id=self._processing_metadata.operation_id,
                operation_type=self._processing_metadata.operation_type,
                source_metadata=None,  # Would be provided by the caller
                processing_metadata=self._processing_metadata
            )
            
            # Add loading-specific metadata
            self._audit_trail.output_metadata = {
                'output_format': self.output_format.value,
                'output_path': str(self.output_path),
                'partitions': [
                    {
                        'key': partition.partition_key,
                        'value': partition.partition_value,
                        'records': partition.record_count,
                        'size': partition.file_size
                    }
                    for partition in self._partitions
                ],
                'version': self.current_version
            }
        
        return self._audit_trail
    
    def _load_single_batch(
        self,
        data: DataBatch,
        destination: Optional[Union[str, Path, Dict[str, Any]]],
        **kwargs
    ) -> Dict[str, Any]:
        """Load a single batch of data."""
        # Add version information
        versioned_data = self.create_version(data)
        
        # Partition data if needed
        partitions = self.partition_data(versioned_data)
        
        # Load each partition
        output_files = []
        total_records = 0
        
        for i, partition in enumerate(partitions):
            partition_result = self.load(partition, self._get_partition_destination(destination, i), **kwargs)
            output_files.append(partition_result)
            total_records += len(partition)
            
            # Update progress
            if self._progress_callback:
                self._progress_callback(total_records, len(versioned_data), f"Loaded {total_records} records")
        
        # Update processing metadata
        if self._processing_metadata:
            self._processing_metadata.records_processed = total_records
        
        return {
            'output_files': output_files,
            'total_records': total_records,
            'partitions': len(partitions)
        }
    
    def _load_batch_iterator(
        self,
        data_iterator: Iterator[DataBatch],
        destination: Optional[Union[str, Path, Dict[str, Any]]],
        **kwargs
    ) -> Dict[str, Any]:
        """Load from an iterator of data batches."""
        output_files = []
        total_records = 0
        batch_count = 0
        
        for batch in data_iterator:
            batch_result = self._load_single_batch(batch, destination, **kwargs)
            output_files.extend(batch_result['output_files'])
            total_records += batch_result['total_records']
            batch_count += 1
            
            # Update progress
            if self._progress_callback:
                self._progress_callback(total_records, 0, f"Processed {batch_count} batches")
        
        # Update processing metadata
        if self._processing_metadata:
            self._processing_metadata.records_processed = total_records
        
        return {
            'output_files': output_files,
            'total_records': total_records,
            'batches_processed': batch_count
        }
    
    def _partition_by_count(self, data: DataBatch) -> List[DataBatch]:
        """Partition data by record count."""
        partitions = []
        
        for i in range(0, len(data), self.partition_size):
            partition = data[i:i + self.partition_size]
            partitions.append(partition)
            
            # Track partition metadata
            self._partitions.append(DataPartition(
                partition_key="count",
                partition_value=str(i // self.partition_size),
                file_path=Path(f"partition_{i // self.partition_size}"),
                record_count=len(partition),
                file_size=0  # Will be updated after writing
            ))
        
        return partitions
    
    def _partition_by_size(self, data: DataBatch) -> List[DataBatch]:
        """Partition data by approximate memory size."""
        # Simplified implementation - would need actual size calculation
        return self._partition_by_count(data)
    
    def _partition_by_date(self, data: DataBatch) -> List[DataBatch]:
        """Partition data by date column."""
        if not self.partition_column:
            return [data]
        
        partitions_dict = {}
        
        for record in data:
            date_value = record.get(self.partition_column)
            if date_value:
                # Extract date part (assuming datetime or date string)
                date_key = str(date_value)[:10]  # YYYY-MM-DD
                
                if date_key not in partitions_dict:
                    partitions_dict[date_key] = []
                
                partitions_dict[date_key].append(record)
        
        return list(partitions_dict.values())
    
    def _partition_by_column(self, data: DataBatch) -> List[DataBatch]:
        """Partition data by column value."""
        if not self.partition_column:
            return [data]
        
        partitions_dict = {}
        
        for record in data:
            column_value = record.get(self.partition_column, 'null')
            
            if column_value not in partitions_dict:
                partitions_dict[column_value] = []
            
            partitions_dict[column_value].append(record)
        
        return list(partitions_dict.values())
    
    def _partition_by_hash(self, data: DataBatch) -> List[DataBatch]:
        """Partition data by hash of partition column."""
        if not self.partition_column:
            return [data]
        
        num_partitions = max(1, len(data) // self.partition_size)
        partitions = [[] for _ in range(num_partitions)]
        
        for record in data:
            column_value = str(record.get(self.partition_column, ''))
            partition_index = hash(column_value) % num_partitions
            partitions[partition_index].append(record)
        
        # Filter out empty partitions
        return [partition for partition in partitions if partition]
    
    def _get_partition_destination(
        self,
        base_destination: Optional[Union[str, Path, Dict[str, Any]]],
        partition_index: int
    ) -> Optional[Union[str, Path, Dict[str, Any]]]:
        """Get destination for a specific partition."""
        if isinstance(base_destination, (str, Path)):
            path = Path(base_destination)
            return path.parent / f"{path.stem}_part_{partition_index}{path.suffix}"
        
        return base_destination
    
    def _get_file_extension(self, format_type: DataFormat) -> str:
        """Get file extension for a data format."""
        extensions = {
            DataFormat.CSV: 'csv',
            DataFormat.JSON: 'json',
            DataFormat.PARQUET: 'parquet',
            DataFormat.XLSX: 'xlsx',
            DataFormat.GEOJSON: 'geojson',
            DataFormat.SQLITE: 'db'
        }
        
        return extensions.get(format_type, 'dat')
    
    def _optimise_for_read(self, output_path: Optional[str]) -> None:
        """Apply read optimisations to output files."""
        if not output_path or not self.enable_indexing:
            return
        
        path = Path(output_path)
        
        # Add optimisations based on file type
        if path.suffix.lower() == '.db':
            self._optimise_sqlite(path)
        elif path.suffix.lower() == '.parquet':
            self._optimise_parquet(path)
    
    def _optimise_sqlite(self, db_path: Path) -> None:
        """Optimise SQLite database for read performance."""
        try:
            with sqlite3.connect(db_path) as conn:
                # Analyze tables for better query planning
                conn.execute("ANALYZE")
                
                # Set pragmas for better performance
                conn.execute("PRAGMA optimize")
                
        except Exception as e:
            self.logger.warning(f"SQLite optimisation failed: {e}")
    
    def _optimise_parquet(self, parquet_path: Path) -> None:
        """Optimise Parquet file for read performance."""
        try:
            # Read and rewrite with optimised settings
            df = pd.read_parquet(parquet_path)
            df.to_parquet(
                parquet_path,
                engine='pyarrow',
                compression='snappy',
                index=False
            )
            
        except Exception as e:
            self.logger.warning(f"Parquet optimisation failed: {e}")