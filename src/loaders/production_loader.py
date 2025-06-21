"""Production loader for multi-format exports with compression and partitioning.

This module provides production-ready data export capabilities with support for
multiple formats, compression optimisation, and data partitioning strategies.

British English spelling is used throughout (optimise, standardise, etc.).
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import gzip
import brotli
import lz4.frame
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from .base import BaseLoader
from ..utils.interfaces import LoadingError, DataFormat
from ..utils.config import get_config
from ..utils.logging import get_logger, monitor_performance, track_lineage


class CompressionManager:
    """Manages compression optimisation for different export formats."""
    
    COMPRESSION_ALGORITHMS = {
        'gzip': {'module': gzip, 'extension': '.gz', 'level_range': (1, 9)},
        'brotli': {'module': brotli, 'extension': '.br', 'level_range': (1, 11)},
        'lz4': {'module': lz4.frame, 'extension': '.lz4', 'level_range': (1, 12)},
        'snappy': {'extension': '.snappy', 'level_range': (1, 1)}  # Snappy has no levels
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        try:
            self.config = config or get_config('exports.compression', {})
        except Exception:
            # Fallback to basic config if configuration loading fails
            self.config = config or {}
        self.logger = get_logger(__name__)
        
    def get_optimal_compression(self, 
                              data: pd.DataFrame, 
                              format_type: str) -> Tuple[str, int]:
        """Analyse data to recommend optimal compression algorithm and level.
        
        Args:
            data: DataFrame to analyse
            format_type: Target export format (parquet, csv, json, etc.)
            
        Returns:
            Tuple of (algorithm, compression_level)
        """
        data_size = data.memory_usage(deep=True).sum()
        row_count = len(data)
        column_count = len(data.columns)
        
        # Default compression based on format and data characteristics
        if format_type == 'parquet':
            if data_size > 100_000_000:  # > 100MB
                return 'snappy', 1  # Fast compression for large data
            else:
                return 'gzip', 6  # Better compression for smaller data
                
        elif format_type == 'csv':
            if row_count > 1_000_000:  # > 1M rows
                return 'lz4', 3  # Fast compression
            else:
                return 'gzip', 6  # Better compression
                
        elif format_type in ['json', 'geojson']:
            return 'brotli', 6  # Excellent for text-based formats
            
        else:
            return 'gzip', 6  # Safe default
    
    def compress_file(self, 
                     input_path: Path, 
                     algorithm: str, 
                     level: int = 6) -> Path:
        """Compress a file using specified algorithm and level.
        
        Args:
            input_path: Path to input file
            algorithm: Compression algorithm to use
            level: Compression level
            
        Returns:
            Path to compressed file
        """
        if algorithm not in self.COMPRESSION_ALGORITHMS:
            raise LoadingError(f"Unsupported compression algorithm: {algorithm}")
            
        compression_info = self.COMPRESSION_ALGORITHMS[algorithm]
        output_path = input_path.with_suffix(input_path.suffix + compression_info['extension'])
        
        try:
            with open(input_path, 'rb') as input_file:
                data = input_file.read()
                
            if algorithm == 'gzip':
                compressed_data = gzip.compress(data, compresslevel=level)
            elif algorithm == 'brotli':
                compressed_data = brotli.compress(data, quality=level)
            elif algorithm == 'lz4':
                compressed_data = lz4.frame.compress(data, compression_level=level)
            else:
                raise LoadingError(f"Compression method not implemented: {algorithm}")
                
            with open(output_path, 'wb') as output_file:
                output_file.write(compressed_data)
                
            # Log compression ratio
            original_size = input_path.stat().st_size
            compressed_size = output_path.stat().st_size
            ratio = compressed_size / original_size if original_size > 0 else 0
            
            self.logger.info(f"Compressed {input_path.name}", 
                           algorithm=algorithm, level=level,
                           original_size=original_size, compressed_size=compressed_size,
                           compression_ratio=f"{ratio:.2%}")
            
            return output_path
            
        except Exception as e:
            raise LoadingError(f"Compression failed for {input_path}: {str(e)}")


class PartitionManager:
    """Implements data partitioning strategies for large datasets."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        try:
            self.config = config or get_config('exports.partitioning', {})
        except Exception:
            # Fallback to basic config if configuration loading fails
            self.config = config or {}
        self.logger = get_logger(__name__)
        
    def get_partition_strategy(self, 
                             data: pd.DataFrame, 
                             partition_type: str = 'auto') -> Dict[str, Any]:
        """Determine optimal partitioning strategy for data.
        
        Args:
            data: DataFrame to partition
            partition_type: Partitioning strategy (auto, state, sa3, temporal, size)
            
        Returns:
            Dictionary describing partition strategy
        """
        row_count = len(data)
        data_size = data.memory_usage(deep=True).sum()
        
        if partition_type == 'auto':
            # Auto-select based on data characteristics
            if 'state_code' in data.columns and row_count > 10000:
                return {'type': 'state', 'column': 'state_code', 'max_size': 50_000_000}
            elif 'sa3_code' in data.columns and row_count > 50000:
                return {'type': 'sa3', 'column': 'sa3_code', 'max_size': 20_000_000}
            elif data_size > 100_000_000:  # > 100MB
                return {'type': 'size', 'max_rows': 100000}
            else:
                return {'type': 'none'}
                
        elif partition_type == 'state':
            return {'type': 'state', 'column': 'state_code', 'max_size': 50_000_000}
            
        elif partition_type == 'sa3':
            return {'type': 'sa3', 'column': 'sa3_code', 'max_size': 20_000_000}
            
        elif partition_type == 'temporal':
            # Look for date columns
            date_columns = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                return {'type': 'temporal', 'column': date_columns[0], 'period': 'month'}
            else:
                return {'type': 'none'}
                
        elif partition_type == 'size':
            return {'type': 'size', 'max_rows': 100000}
            
        else:
            return {'type': 'none'}
    
    def partition_data(self, 
                      data: pd.DataFrame, 
                      strategy: Dict[str, Any]) -> List[Tuple[str, pd.DataFrame]]:
        """Partition data according to strategy.
        
        Args:
            data: DataFrame to partition
            strategy: Partitioning strategy from get_partition_strategy
            
        Returns:
            List of (partition_name, partition_data) tuples
        """
        if strategy['type'] == 'none':
            return [('all', data)]
            
        elif strategy['type'] in ['state', 'sa3']:
            column = strategy['column']
            if column not in data.columns:
                self.logger.warning(f"Partition column {column} not found, using no partitioning")
                return [('all', data)]
                
            partitions = []
            for value in data[column].unique():
                partition_data = data[data[column] == value].copy()
                partition_name = f"{column}_{value}"
                partitions.append((partition_name, partition_data))
                
            return partitions
            
        elif strategy['type'] == 'temporal':
            column = strategy['column']
            if column not in data.columns:
                self.logger.warning(f"Temporal column {column} not found, using no partitioning")
                return [('all', data)]
                
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(data[column]):
                data[column] = pd.to_datetime(data[column], errors='coerce')
                
            partitions = []
            period = strategy.get('period', 'month')
            
            if period == 'month':
                for period_group, group_data in data.groupby(data[column].dt.to_period('M')):
                    partition_name = f"period_{period_group}"
                    partitions.append((partition_name, group_data))
                    
            return partitions
            
        elif strategy['type'] == 'size':
            max_rows = strategy['max_rows']
            partitions = []
            
            for i in range(0, len(data), max_rows):
                partition_data = data.iloc[i:i + max_rows].copy()
                partition_name = f"chunk_{i // max_rows + 1:04d}"
                partitions.append((partition_name, partition_data))
                
            return partitions
            
        else:
            return [('all', data)]


class ExportOptimizer:
    """Optimises exports for web delivery and performance."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config('exports.optimisation', {})
        self.logger = get_logger(__name__)
        
    def optimise_for_web(self, data: pd.DataFrame, format_type: str) -> pd.DataFrame:
        """Optimise data structure for web delivery.
        
        Args:
            data: DataFrame to optimise
            format_type: Target format for optimisation
            
        Returns:
            Optimised DataFrame
        """
        optimised_data = data.copy()
        
        # Reduce precision for numeric columns where appropriate
        for column in optimised_data.select_dtypes(include=['float64']).columns:
            # Check if column contains coordinates (likely need higher precision)
            if any(coord_term in column.lower() for coord_term in ['lat', 'lon', 'lng', 'x', 'y']):
                optimised_data[column] = optimised_data[column].astype('float32')
            else:
                # For non-coordinate numeric data, round to reasonable precision
                optimised_data[column] = optimised_data[column].round(3).astype('float32')
                
        # Optimise string columns
        for column in optimised_data.select_dtypes(include=['object']).columns:
            if optimised_data[column].dtype == 'object':
                # Try to convert to category if it has few unique values
                unique_ratio = optimised_data[column].nunique() / len(optimised_data)
                if unique_ratio < 0.1:  # Less than 10% unique values
                    optimised_data[column] = optimised_data[column].astype('category')
                    
        # Sort data for better compression
        if 'sa2_code' in optimised_data.columns:
            optimised_data = optimised_data.sort_values('sa2_code')
        elif 'postcode' in optimised_data.columns:
            optimised_data = optimised_data.sort_values('postcode')
            
        return optimised_data
    
    def generate_cache_headers(self, 
                             data_freshness: datetime,
                             format_type: str) -> Dict[str, str]:
        """Generate appropriate cache headers for exported data.
        
        Args:
            data_freshness: When the source data was last updated
            format_type: Export format type
            
        Returns:
            Dictionary of cache headers
        """
        # Calculate cache duration based on data type and format
        if format_type in ['parquet', 'csv']:
            # Raw data formats - longer cache
            cache_duration = 86400  # 24 hours
        elif format_type in ['json', 'geojson']:
            # API formats - shorter cache
            cache_duration = 3600  # 1 hour
        else:
            cache_duration = 7200  # 2 hours default
            
        headers = {
            'Cache-Control': f'public, max-age={cache_duration}',
            'ETag': f'"{hash(str(data_freshness))}"',
            'Last-Modified': data_freshness.strftime('%a, %d %b %Y %H:%M:%S GMT'),
            'Expires': (datetime.now() + pd.Timedelta(seconds=cache_duration)).strftime('%a, %d %b %Y %H:%M:%S GMT')
        }
        
        return headers


class ProductionLoader(BaseLoader):
    """Main production loader for multi-format exports with optimisation."""
    
    def __init__(self, 
                 loader_id: str = "production_loader",
                 config: Optional[Dict[str, Any]] = None,
                 logger: Optional[Any] = None):
        loader_config = config or {}
        super().__init__(
            loader_id=loader_id,
            config=loader_config,
            logger=logger or get_logger(__name__)
        )
        self.compression_manager = CompressionManager(config)
        self.partition_manager = PartitionManager(config)
        self.export_optimiser = ExportOptimizer(config)
        self.logger = get_logger(__name__)
        
        # Supported formats (only those in DataFormat enum)
        self.supported_formats = {
            'parquet', 'csv', 'json', 'geojson', 'xlsx'
        }
    
    def get_supported_formats(self) -> List[DataFormat]:
        """
        Get the list of supported output formats.
        
        Returns:
            List[DataFormat]: Supported formats
        """
        # Map string format names to DataFormat enum values
        format_mapping = {
            'parquet': DataFormat.PARQUET,
            'csv': DataFormat.CSV,
            'json': DataFormat.JSON,
            'geojson': DataFormat.GEOJSON,
            'xlsx': DataFormat.XLSX,
        }
        
        return [format_mapping[fmt] for fmt in self.supported_formats if fmt in format_mapping]
        
    @monitor_performance("production_export")
    def load(self, 
             data: pd.DataFrame, 
             output_path: Union[str, Path],
             formats: Optional[List[str]] = None,
             compress: bool = True,
             partition: bool = True,
             optimise_for_web: bool = True,
             **kwargs) -> Dict[str, Any]:
        """Load data in multiple optimised formats for production.
        
        Args:
            data: DataFrame to export
            output_path: Base output directory
            formats: List of formats to export (default: all supported)
            compress: Whether to compress exports
            partition: Whether to partition large datasets
            optimise_for_web: Whether to optimise for web delivery
            **kwargs: Additional export parameters
            
        Returns:
            Dictionary with export results and metadata
        """
        if formats is None:
            formats = list(self.supported_formats)
            
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Track export lineage
        track_lineage("processed_data", str(output_path), "production_export")
        
        export_results = {
            'formats': {},
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'total_rows': len(data),
                'total_columns': len(data.columns),
                'data_size_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
            }
        }
        
        try:
            # Optimise data if requested
            if optimise_for_web:
                data = self.export_optimiser.optimise_for_web(data, formats[0])
                
            # Determine partitioning strategy
            partition_strategy = {'type': 'none'}
            if partition:
                partition_strategy = self.partition_manager.get_partition_strategy(data)
                
            # Partition data if needed
            partitions = self.partition_manager.partition_data(data, partition_strategy)
            
            # Export each format
            for format_type in formats:
                if format_type not in self.supported_formats:
                    self.logger.warning(f"Unsupported format: {format_type}")
                    continue
                    
                # Remove conflicting parameters from kwargs
                clean_kwargs = {k: v for k, v in kwargs.items() 
                               if k not in ['partitions', 'format_type', 'output_path', 
                                          'compress', 'partition_strategy']}
                
                format_results = self._export_format(
                    partitions, format_type, output_path, 
                    compress, partition_strategy, **clean_kwargs
                )
                export_results['formats'][format_type] = format_results
                
            # Generate export metadata
            metadata_path = output_path / 'export_metadata.json'
            self._generate_export_metadata(export_results, metadata_path)
            
            self.logger.info("Production export completed successfully",
                           formats=formats, partitions=len(partitions),
                           output_path=str(output_path))
            
            return export_results
            
        except Exception as e:
            raise LoadingError(f"Production export failed: {str(e)}")
    
    def _export_format(self, 
                      partitions: List[Tuple[str, pd.DataFrame]],
                      format_type: str,
                      output_path: Path,
                      compress: bool,
                      partition_strategy: Dict[str, Any],
                      **kwargs) -> Dict[str, Any]:
        """Export data in specific format."""
        format_dir = output_path / format_type
        format_dir.mkdir(exist_ok=True)
        
        format_results = {
            'files': [],
            'total_size_bytes': 0,
            'compression_info': {},
            'partition_info': partition_strategy
        }
        
        for partition_name, partition_data in partitions:
            # Determine compression
            compression_algo, compression_level = None, None
            if compress:
                compression_algo, compression_level = self.compression_manager.get_optimal_compression(
                    partition_data, format_type
                )
                
            # Export partition
            file_result = self._export_partition(
                partition_data, format_type, format_dir, 
                partition_name, compression_algo, compression_level, **kwargs
            )
            
            format_results['files'].append(file_result)
            format_results['total_size_bytes'] += file_result['size_bytes']
            
        return format_results
    
    def _export_partition(self, 
                         data: pd.DataFrame,
                         format_type: str,
                         output_dir: Path,
                         partition_name: str,
                         compression_algo: Optional[str],
                         compression_level: Optional[int],
                         **kwargs) -> Dict[str, Any]:
        """Export a single partition in specified format."""
        # This will be implemented by format-specific exporters
        # For now, provide a basic implementation
        
        if format_type == 'parquet':
            filename = f"{partition_name}.parquet"
            filepath = output_dir / filename
            
            # Use compression in parquet directly
            compression = compression_algo if compression_algo in ['snappy', 'gzip', 'brotli'] else 'snappy'
            data.to_parquet(filepath, compression=compression, index=False)
            
        elif format_type == 'csv':
            filename = f"{partition_name}.csv"
            filepath = output_dir / filename
            data.to_csv(filepath, index=False, encoding='utf-8')
            
            # Compress separately if requested
            if compression_algo:
                compressed_path = self.compression_manager.compress_file(
                    filepath, compression_algo, compression_level
                )
                filepath.unlink()  # Remove uncompressed file
                filepath = compressed_path
                
        else:
            # Basic implementation for other formats
            filename = f"{partition_name}.{format_type}"
            filepath = output_dir / filename
            
            if format_type == 'json':
                data.to_json(filepath, orient='records', date_format='iso')
            elif format_type == 'xlsx':
                data.to_excel(filepath, index=False)
                
        file_info = {
            'filename': filepath.name,
            'path': str(filepath),
            'size_bytes': filepath.stat().st_size,
            'rows': len(data),
            'compression': compression_algo
        }
        
        return file_info
    
    def _generate_export_metadata(self, 
                                 export_results: Dict[str, Any], 
                                 metadata_path: Path) -> None:
        """Generate comprehensive export metadata."""
        import json
        
        metadata = {
            'export_info': export_results['metadata'],
            'formats': export_results['formats'],
            'schema_version': '1.0',
            'data_lineage': {
                'source': 'AHGD ETL Pipeline',
                'transformation': 'Production Export',
                'quality_assured': True
            }
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
    def validate_export(self, export_results: Dict[str, Any]) -> bool:
        """Validate that export completed successfully."""
        try:
            # Check that all requested formats were exported
            if not export_results['formats']:
                return False
                
            # Check that files exist and have reasonable sizes
            for format_type, format_info in export_results['formats'].items():
                if not format_info['files']:
                    return False
                    
                for file_info in format_info['files']:
                    filepath = Path(file_info['path'])
                    if not filepath.exists() or filepath.stat().st_size == 0:
                        return False
                        
            return True
            
        except Exception as e:
            self.logger.error(f"Export validation failed: {str(e)}")
            return False
