"""Compression analysis and optimisation utilities.

This module provides comprehensive compression analysis and optimisation
capabilities for Australian health and geographic data exports.

British English spelling is used throughout (optimise, analyse, etc.).
"""

import os
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from loguru import logger

from .interfaces import AHGDError
from .config import get_config
from .logging import get_logger, monitor_performance


class CompressionAlgorithm(Enum):
    """Supported compression algorithms."""
    GZIP = "gzip"
    BROTLI = "brotli"
    LZ4 = "lz4"
    SNAPPY = "snappy"
    ZSTD = "zstd"


@dataclass
class CompressionResult:
    """Results of compression analysis."""
    algorithm: CompressionAlgorithm
    level: int
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time: float
    decompression_time: float
    memory_usage: int
    
    @property
    def size_reduction_percent(self) -> float:
        """Percentage reduction in file size."""
        return (1 - self.compression_ratio) * 100
    
    @property
    def efficiency_score(self) -> float:
        """Combined efficiency score (compression ratio vs time)."""
        # Higher score is better
        ratio_score = (1 - self.compression_ratio) * 100  # 0-100
        time_score = max(0, 100 - self.compression_time)  # Penalise slow compression
        return (ratio_score * 0.7) + (time_score * 0.3)


class CompressionAnalyzer:
    """Analyses data to recommend optimal compression strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config('compression', {})
        self.logger = get_logger(__name__)
        
        # Default algorithm preferences for different data types
        self.algorithm_preferences = {
            'text_heavy': [CompressionAlgorithm.BROTLI, CompressionAlgorithm.GZIP],
            'numeric_heavy': [CompressionAlgorithm.SNAPPY, CompressionAlgorithm.LZ4],
            'mixed': [CompressionAlgorithm.GZIP, CompressionAlgorithm.BROTLI],
            'geographic': [CompressionAlgorithm.GZIP, CompressionAlgorithm.BROTLI],
            'large_files': [CompressionAlgorithm.LZ4, CompressionAlgorithm.SNAPPY]
        }
        
    @monitor_performance("compression_analysis")
    def analyse_data_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyse data characteristics to inform compression decisions.
        
        Args:
            data: DataFrame to analyse
            
        Returns:
            Dictionary of data characteristics
        """
        characteristics = {
            'total_size_bytes': data.memory_usage(deep=True).sum(),
            'row_count': len(data),
            'column_count': len(data.columns),
            'data_types': {}
        }
        
        # Analyse column types
        numeric_columns = 0
        text_columns = 0
        datetime_columns = 0
        categorical_columns = 0
        
        for column in data.columns:
            dtype = data[column].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                numeric_columns += 1
                characteristics['data_types'][column] = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                datetime_columns += 1
                characteristics['data_types'][column] = 'datetime'
            elif pd.api.types.is_categorical_dtype(dtype):
                categorical_columns += 1
                characteristics['data_types'][column] = 'categorical'
            else:
                text_columns += 1
                characteristics['data_types'][column] = 'text'
                
        # Calculate proportions
        total_cols = len(data.columns)
        characteristics.update({
            'numeric_ratio': numeric_columns / total_cols,
            'text_ratio': text_columns / total_cols,
            'datetime_ratio': datetime_columns / total_cols,
            'categorical_ratio': categorical_columns / total_cols
        })
        
        # Estimate entropy (data randomness)
        characteristics['entropy_score'] = self._estimate_entropy(data)
        
        # Detect special patterns
        characteristics['patterns'] = self._detect_patterns(data)
        
        # Classify data type
        characteristics['data_classification'] = self._classify_data_type(characteristics)
        
        return characteristics
    
    def recommend_compression(self, 
                            data: pd.DataFrame, 
                            target_format: str,
                            priority: str = 'balanced') -> Tuple[CompressionAlgorithm, int]:
        """Recommend optimal compression algorithm and level.
        
        Args:
            data: DataFrame to compress
            target_format: Target export format (parquet, csv, json, etc.)
            priority: Optimisation priority (speed, size, balanced)
            
        Returns:
            Tuple of (algorithm, compression_level)
        """
        characteristics = self.analyse_data_characteristics(data)
        
        # Get algorithm candidates based on data classification
        data_class = characteristics['data_classification']
        candidates = self.algorithm_preferences.get(data_class, 
                                                   self.algorithm_preferences['mixed'])
        
        # Adjust based on target format
        if target_format == 'parquet':
            # Parquet benefits from column-oriented compression
            if CompressionAlgorithm.SNAPPY not in candidates:
                candidates.insert(0, CompressionAlgorithm.SNAPPY)
        elif target_format in ['json', 'geojson']:
            # Text-based formats benefit from better text compression
            candidates = [CompressionAlgorithm.BROTLI, CompressionAlgorithm.GZIP]
        elif target_format == 'csv':
            # CSV benefits from fast compression for large files
            if characteristics['total_size_bytes'] > 100_000_000:  # > 100MB
                candidates = [CompressionAlgorithm.LZ4, CompressionAlgorithm.SNAPPY]
                
        # Select algorithm based on priority
        selected_algorithm = candidates[0]  # Default to first candidate
        
        if priority == 'speed':
            speed_priority = [CompressionAlgorithm.LZ4, CompressionAlgorithm.SNAPPY]
            for algo in speed_priority:
                if algo in candidates:
                    selected_algorithm = algo
                    break
        elif priority == 'size':
            size_priority = [CompressionAlgorithm.BROTLI, CompressionAlgorithm.GZIP]
            for algo in size_priority:
                if algo in candidates:
                    selected_algorithm = algo
                    break
                    
        # Determine compression level
        compression_level = self._get_optimal_level(
            selected_algorithm, characteristics, priority
        )
        
        self.logger.info(f"Compression recommendation",
                        algorithm=selected_algorithm.value,
                        level=compression_level,
                        data_class=data_class,
                        priority=priority)
        
        return selected_algorithm, compression_level
    
    def _estimate_entropy(self, data: pd.DataFrame) -> float:
        """Estimate data entropy (randomness) to predict compressibility."""
        # Sample data for entropy estimation
        sample_size = min(10000, len(data))
        sample_data = data.sample(n=sample_size, random_state=42)
        
        total_entropy = 0
        
        for column in sample_data.columns:
            if sample_data[column].dtype == 'object':
                # For text columns, estimate character-level entropy
                text_data = sample_data[column].astype(str).str.cat()
                if text_data:
                    char_counts = pd.Series(list(text_data)).value_counts()
                    probabilities = char_counts / len(text_data)
                    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
                    total_entropy += entropy
            else:
                # For numeric columns, estimate value distribution entropy
                value_counts = sample_data[column].value_counts()
                if len(value_counts) > 1:
                    probabilities = value_counts / len(sample_data)
                    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
                    total_entropy += entropy
                    
        # Normalise by number of columns
        return total_entropy / len(data.columns) if len(data.columns) > 0 else 0
    
    def _detect_patterns(self, data: pd.DataFrame) -> Dict[str, bool]:
        """Detect special patterns in data that affect compression."""
        patterns = {
            'has_coordinates': False,
            'has_postcodes': False,
            'has_repeated_values': False,
            'has_nulls': False,
            'has_long_strings': False
        }
        
        # Check for coordinate columns
        coord_indicators = ['lat', 'lon', 'lng', 'latitude', 'longitude', '_x', '_y']
        patterns['has_coordinates'] = any(
            any(indicator in col.lower() for indicator in coord_indicators)
            for col in data.columns
        )
        
        # Check for postcodes
        postcode_indicators = ['postcode', 'zip', 'postal']
        patterns['has_postcodes'] = any(
            any(indicator in col.lower() for indicator in postcode_indicators)
            for col in data.columns
        )
        
        # Check for repeated values (good for compression)
        for column in data.columns:
            unique_ratio = data[column].nunique() / len(data)
            if unique_ratio < 0.1:  # Less than 10% unique values
                patterns['has_repeated_values'] = True
                break
                
        # Check for null values
        patterns['has_nulls'] = data.isnull().any().any()
        
        # Check for long strings
        for column in data.select_dtypes(include=['object']).columns:
            if data[column].astype(str).str.len().max() > 1000:
                patterns['has_long_strings'] = True
                break
                
        return patterns
    
    def _classify_data_type(self, characteristics: Dict[str, Any]) -> str:
        """Classify data type based on characteristics."""
        # Determine primary data classification
        if characteristics['text_ratio'] > 0.6:
            return 'text_heavy'
        elif characteristics['numeric_ratio'] > 0.7:
            return 'numeric_heavy'
        elif characteristics['total_size_bytes'] > 100_000_000:  # > 100MB
            return 'large_files'
        elif characteristics['patterns']['has_coordinates']:
            return 'geographic'
        else:
            return 'mixed'
    
    def _get_optimal_level(self, 
                          algorithm: CompressionAlgorithm,
                          characteristics: Dict[str, Any],
                          priority: str) -> int:
        """Get optimal compression level for algorithm."""
        # Default levels for each algorithm
        default_levels = {
            CompressionAlgorithm.GZIP: 6,
            CompressionAlgorithm.BROTLI: 6,
            CompressionAlgorithm.LZ4: 3,
            CompressionAlgorithm.SNAPPY: 1,  # Snappy doesn't have levels
            CompressionAlgorithm.ZSTD: 3
        }
        
        base_level = default_levels.get(algorithm, 3)
        
        # Adjust based on data size and priority
        if priority == 'speed':
            # Lower compression levels for speed
            if algorithm == CompressionAlgorithm.GZIP:
                return max(1, base_level - 2)
            elif algorithm == CompressionAlgorithm.BROTLI:
                return max(1, base_level - 3)
        elif priority == 'size':
            # Higher compression levels for better compression
            if algorithm == CompressionAlgorithm.GZIP:
                return min(9, base_level + 2)
            elif algorithm == CompressionAlgorithm.BROTLI:
                return min(11, base_level + 3)
                
        # Adjust for large files (prefer speed)
        if characteristics['total_size_bytes'] > 500_000_000:  # > 500MB
            if algorithm == CompressionAlgorithm.GZIP:
                return max(1, base_level - 1)
            elif algorithm == CompressionAlgorithm.BROTLI:
                return max(1, base_level - 2)
                
        return base_level


class FormatOptimizer:
    """Optimises data structure for each export format."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config('format_optimisation', {})
        self.logger = get_logger(__name__)
        
    @monitor_performance("format_optimisation")
    def optimise_for_format(self, 
                           data: pd.DataFrame, 
                           target_format: str) -> pd.DataFrame:
        """Optimise DataFrame structure for specific export format.
        
        Args:
            data: DataFrame to optimise
            target_format: Target export format
            
        Returns:
            Optimised DataFrame
        """
        optimised_data = data.copy()
        
        if target_format == 'parquet':
            optimised_data = self._optimise_for_parquet(optimised_data)
        elif target_format == 'csv':
            optimised_data = self._optimise_for_csv(optimised_data)
        elif target_format in ['json', 'geojson']:
            optimised_data = self._optimise_for_json(optimised_data)
        elif target_format == 'xlsx':
            optimised_data = self._optimise_for_excel(optimised_data)
            
        self.logger.info(f"Format optimisation completed",
                        format=target_format,
                        original_size=data.memory_usage(deep=True).sum(),
                        optimised_size=optimised_data.memory_usage(deep=True).sum())
        
        return optimised_data
    
    def _optimise_for_parquet(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimise for Parquet format."""
        # Convert to most efficient types for Parquet
        for column in data.columns:
            if data[column].dtype == 'object':
                # Try to convert to categorical for repeated values
                unique_ratio = data[column].nunique() / len(data)
                if unique_ratio < 0.1:  # Less than 10% unique values
                    data[column] = data[column].astype('category')
            elif pd.api.types.is_integer_dtype(data[column]):
                # Downcast integers
                data[column] = pd.to_numeric(data[column], downcast='integer')
            elif pd.api.types.is_float_dtype(data[column]):
                # Downcast floats, but preserve coordinate precision
                if not any(coord in column.lower() for coord in ['lat', 'lon', 'lng', 'x', 'y']):
                    data[column] = pd.to_numeric(data[column], downcast='float')
                    
        return data
    
    def _optimise_for_csv(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimise for CSV format."""
        # Clean string data for CSV
        for column in data.select_dtypes(include=['object']).columns:
            if data[column].dtype == 'object':
                # Remove problematic characters
                data[column] = data[column].astype(str)
                data[column] = data[column].str.replace('\n', ' ')
                data[column] = data[column].str.replace('\r', ' ')
                data[column] = data[column].str.replace('"', "'")
                
        # Format datetime columns consistently
        for column in data.select_dtypes(include=['datetime64']).columns:
            data[column] = data[column].dt.strftime('%Y-%m-%d %H:%M:%S')
            
        return data
    
    def _optimise_for_json(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimise for JSON formats."""
        # Convert to JSON-safe types
        for column in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[column]):
                data[column] = data[column].dt.strftime('%Y-%m-%dT%H:%M:%S')
            elif pd.api.types.is_integer_dtype(data[column]):
                # Ensure integers are JSON-safe
                data[column] = data[column].astype('Int64')  # Nullable integer
                
        # Handle NaN values
        data = data.fillna('')
        
        return data
    
    def _optimise_for_excel(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimise for Excel format."""
        # Excel has limitations on string length and sheet size
        max_rows = 1_048_576  # Excel row limit
        max_string_length = 32_767  # Excel cell character limit
        
        if len(data) > max_rows:
            self.logger.warning(f"Data exceeds Excel row limit, truncating to {max_rows} rows")
            data = data.head(max_rows)
            
        # Truncate long strings
        for column in data.select_dtypes(include=['object']).columns:
            if data[column].dtype == 'object':
                mask = data[column].astype(str).str.len() > max_string_length
                if mask.any():
                    data.loc[mask, column] = data.loc[mask, column].astype(str).str[:max_string_length-3] + '...'
                    
        return data


class SizeCalculator:
    """Calculates export sizes and performance metrics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config('size_calculation', {})
        self.logger = get_logger(__name__)
        
    def estimate_export_size(self, 
                            data: pd.DataFrame, 
                            target_format: str,
                            compression: Optional[CompressionAlgorithm] = None) -> Dict[str, Any]:
        """Estimate export file size and performance metrics.
        
        Args:
            data: DataFrame to analyse
            target_format: Target export format
            compression: Compression algorithm (if any)
            
        Returns:
            Dictionary with size estimates and metrics
        """
        # Calculate base memory usage
        memory_usage = data.memory_usage(deep=True).sum()
        
        # Format-specific size multipliers (rough estimates)
        format_multipliers = {
            'parquet': 0.3,  # Very efficient binary format
            'csv': 1.2,      # Text format with some overhead
            'json': 1.8,     # JSON overhead
            'geojson': 2.0,  # Geographic JSON overhead
            'xlsx': 0.8,     # Binary format with compression
            'feather': 0.4   # Efficient binary format
        }
        
        base_size = memory_usage * format_multipliers.get(target_format, 1.0)
        
        # Apply compression estimates
        if compression:
            compression_ratios = {
                CompressionAlgorithm.GZIP: 0.3,
                CompressionAlgorithm.BROTLI: 0.25,
                CompressionAlgorithm.LZ4: 0.5,
                CompressionAlgorithm.SNAPPY: 0.4,
                CompressionAlgorithm.ZSTD: 0.28
            }
            compressed_size = base_size * compression_ratios.get(compression, 0.5)
        else:
            compressed_size = base_size
            
        # Estimate export time based on data size and format
        export_time_seconds = self._estimate_export_time(data, target_format, compression)
        
        return {
            'memory_usage_bytes': memory_usage,
            'memory_usage_mb': memory_usage / 1024 / 1024,
            'estimated_size_bytes': int(compressed_size),
            'estimated_size_mb': compressed_size / 1024 / 1024,
            'compression_ratio': compressed_size / base_size if compression else 1.0,
            'estimated_export_time_seconds': export_time_seconds,
            'format': target_format,
            'compression': compression.value if compression else None
        }
    
    def _estimate_export_time(self, 
                             data: pd.DataFrame, 
                             target_format: str,
                             compression: Optional[CompressionAlgorithm]) -> float:
        """Estimate export time based on data characteristics."""
        # Base time estimates (seconds per MB of memory)
        base_times = {
            'parquet': 0.1,
            'csv': 0.2,
            'json': 0.3,
            'geojson': 0.4,
            'xlsx': 0.5,
            'feather': 0.05
        }
        
        memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
        base_time = memory_mb * base_times.get(target_format, 0.2)
        
        # Add compression overhead
        if compression:
            compression_overhead = {
                CompressionAlgorithm.GZIP: 1.5,
                CompressionAlgorithm.BROTLI: 3.0,
                CompressionAlgorithm.LZ4: 1.1,
                CompressionAlgorithm.SNAPPY: 1.05,
                CompressionAlgorithm.ZSTD: 2.0
            }
            base_time *= compression_overhead.get(compression, 1.5)
            
        return base_time


class CacheManager:
    """Manages export caching for performance optimisation."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / '.ahgd_cache' / 'exports'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
        
    def get_cache_key(self, 
                     data: pd.DataFrame, 
                     export_params: Dict[str, Any]) -> str:
        """Generate cache key for export parameters.
        
        Args:
            data: DataFrame being exported
            export_params: Export parameters
            
        Returns:
            Cache key string
        """
        # Create hash of data content and parameters
        data_hash = hashlib.md5(str(data.values.tobytes()).encode()).hexdigest()[:16]
        params_hash = hashlib.md5(str(sorted(export_params.items())).encode()).hexdigest()[:16]
        
        return f"{data_hash}_{params_hash}"
    
    def is_cached(self, cache_key: str, max_age: timedelta = timedelta(hours=24)) -> bool:
        """Check if export is cached and still valid.
        
        Args:
            cache_key: Cache key to check
            max_age: Maximum age for cache validity
            
        Returns:
            True if cached and valid
        """
        cache_file = self.cache_dir / f"{cache_key}.cache"
        
        if not cache_file.exists():
            return False
            
        # Check age
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age < max_age
    
    def get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached export result.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached result or None if not found
        """
        cache_file = self.cache_dir / f"{cache_key}.cache"
        
        try:
            if cache_file.exists():
                import pickle
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load cache: {str(e)}")
            
        return None
    
    def cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache export result.
        
        Args:
            cache_key: Cache key
            result: Export result to cache
        """
        cache_file = self.cache_dir / f"{cache_key}.cache"
        
        try:
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
                
            self.logger.debug(f"Cached export result", cache_key=cache_key)
            
        except Exception as e:
            self.logger.warning(f"Failed to cache result: {str(e)}")
    
    def clear_cache(self, max_age: Optional[timedelta] = None) -> int:
        """Clear cached exports.
        
        Args:
            max_age: Only clear cache files older than this age
            
        Returns:
            Number of files cleared
        """
        cleared_count = 0
        
        for cache_file in self.cache_dir.glob('*.cache'):
            should_clear = True
            
            if max_age:
                file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                should_clear = file_age > max_age
                
            if should_clear:
                try:
                    cache_file.unlink()
                    cleared_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to clear cache file {cache_file}: {str(e)}")
                    
        self.logger.info(f"Cleared {cleared_count} cache files")
        return cleared_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_files = list(self.cache_dir.glob('*.cache'))
        
        if not cache_files:
            return {
                'total_files': 0,
                'total_size_bytes': 0,
                'total_size_mb': 0,
                'oldest_file': None,
                'newest_file': None
            }
            
        total_size = sum(f.stat().st_size for f in cache_files)
        file_times = [datetime.fromtimestamp(f.stat().st_mtime) for f in cache_files]
        
        return {
            'total_files': len(cache_files),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / 1024 / 1024,
            'oldest_file': min(file_times).isoformat(),
            'newest_file': max(file_times).isoformat()
        }
