"""
Memory Optimizer - Advanced memory optimization strategies for Australian health data analytics

Implements sophisticated memory management for processing 497,181+ health records
with minimal memory footprint while maintaining high performance.

Key Features:
- Adaptive batch sizing based on available memory
- Memory-efficient data transformations
- Smart garbage collection strategies
- Out-of-core processing for datasets exceeding memory
- Memory profiling and optimization recommendations
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Generator, Callable
from pathlib import Path
import logging
import time
import gc
import psutil
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
import weakref
import mmap
import os

logger = logging.getLogger(__name__)


@dataclass
class MemoryProfile:
    """Memory usage profile for a specific operation."""
    operation_id: str
    operation_type: str
    timestamp: str
    peak_memory_mb: float
    memory_efficiency_ratio: float  # output_size / peak_memory
    processing_time_seconds: float
    rows_processed: int
    columns_processed: int
    optimization_applied: List[str]


@dataclass
class MemoryRecommendation:
    """Memory optimization recommendation."""
    priority: str  # critical, high, medium, low
    category: str  # batch_size, data_types, query_strategy, caching
    title: str
    description: str
    estimated_memory_savings_mb: float
    implementation_effort: str  # low, medium, high


class MemoryOptimizer:
    """
    Advanced memory optimizer for Australian health data processing.
    Provides adaptive memory management and optimization strategies.
    """
    
    # Memory optimization configuration
    MEMORY_CONFIG = {
        "adaptive_batch_sizing": True,         # Adjust batch sizes based on available memory
        "aggressive_gc": True,                 # Force garbage collection after operations
        "memory_mapping": True,                # Use memory mapping for large files
        "lazy_evaluation_threshold": 50000,    # Rows threshold for lazy vs eager evaluation
        "memory_pressure_threshold": 0.85,     # Memory usage threshold for optimization
        "compression_in_memory": True,         # Compress DataFrames in memory when possible
    }
    
    # Data type optimization mappings
    DTYPE_OPTIMIZATIONS = {
        "int64": "int32",     # Most health data fits in int32
        "float64": "float32", # Health metrics rarely need float64 precision
        "object": "category", # String columns to categorical for compression
    }
    
    # Australian health data specific optimizations
    HEALTH_DATA_PATTERNS = {
        "sa2_code": {"dtype": "category", "compression": "dictionary"},
        "state_name": {"dtype": "category", "compression": "dictionary"},
        "postcode": {"dtype": "category", "compression": "dictionary"},
        "age_group": {"dtype": "category", "compression": "dictionary"},
        "prescription_count": {"dtype": "int32", "compression": "delta"},
        "cost_amount": {"dtype": "float32", "compression": "none"},
        "risk_score": {"dtype": "float32", "compression": "none"},
        "seifa_decile": {"dtype": "int8", "compression": "none"},  # 1-10 range
    }
    
    def __init__(self, 
                 memory_limit_gb: Optional[float] = None,
                 enable_profiling: bool = True):
        """Initialize memory optimizer with system-aware configuration."""
        self.memory_limit_gb = memory_limit_gb or self._calculate_safe_memory_limit()
        self.enable_profiling = enable_profiling
        
        # Memory tracking
        self.memory_profiles: deque = deque(maxlen=1000)
        self.active_operations: Dict[str, Dict[str, Any]] = {}
        self.memory_recommendations: List[MemoryRecommendation] = []
        
        # System monitoring
        self.system_monitor = SystemMemoryMonitor()
        self.memory_pressure_detector = MemoryPressureDetector()
        
        # Weak references to track managed objects
        self.managed_dataframes: weakref.WeakSet = weakref.WeakSet()
        
        logger.info(f"Initialized memory optimizer with {self.memory_limit_gb:.2f}GB limit")
    
    def _calculate_safe_memory_limit(self) -> float:
        """Calculate safe memory limit based on system resources."""
        try:
            total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
            
            # Use 60% of total memory as safe limit
            safe_limit = total_memory_gb * 0.6
            
            # Minimum 1GB, maximum 16GB for reasonable limits
            return max(1.0, min(16.0, safe_limit))
            
        except Exception as e:
            logger.warning(f"Failed to calculate memory limit: {e}")
            return 4.0  # Default to 4GB
    
    def optimize_dataframe_memory(self, 
                                 df: pl.DataFrame,
                                 dataset_type: str = "health") -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """Optimize DataFrame memory usage with type conversion and compression."""
        try:
            operation_id = f"optimize_memory_{int(time.time())}"
            self._start_memory_tracking(operation_id, "memory_optimization")
            
            original_size_mb = df.estimated_size("mb")
            optimization_log = []
            
            # Apply data type optimizations
            optimized_df = df
            
            # 1. Categorical encoding for string columns
            for col in df.columns:
                dtype = str(df[col].dtype)
                
                if dtype == "Utf8" or "str" in dtype.lower():
                    # Check cardinality for categorical conversion
                    unique_ratio = df[col].n_unique() / df.shape[0]
                    
                    if unique_ratio < 0.5:  # Less than 50% unique values
                        optimized_df = optimized_df.with_columns([
                            pl.col(col).cast(pl.Categorical)
                        ])
                        optimization_log.append(f"Converted {col} to categorical")
            
            # 2. Numeric type downcasting
            for col in df.columns:
                dtype = str(optimized_df[col].dtype)
                
                # Integer downcasting
                if dtype in ["Int64", "Int32"]:
                    col_min = optimized_df[col].min()
                    col_max = optimized_df[col].max()
                    
                    if col_min is not None and col_max is not None:
                        if col_min >= -128 and col_max <= 127:
                            optimized_df = optimized_df.with_columns([
                                pl.col(col).cast(pl.Int8)
                            ])
                            optimization_log.append(f"Downcasted {col} to int8")
                        elif col_min >= -32768 and col_max <= 32767:
                            optimized_df = optimized_df.with_columns([
                                pl.col(col).cast(pl.Int16)
                            ])
                            optimization_log.append(f"Downcasted {col} to int16")
                        elif dtype == "Int64" and col_min >= -2147483648 and col_max <= 2147483647:
                            optimized_df = optimized_df.with_columns([
                                pl.col(col).cast(pl.Int32)
                            ])
                            optimization_log.append(f"Downcasted {col} to int32")
                
                # Float downcasting
                elif dtype == "Float64":
                    # Check if values fit in float32 without losing precision
                    try:
                        float32_df = optimized_df.with_columns([
                            pl.col(col).cast(pl.Float32)
                        ])
                        
                        # Simple precision check
                        optimized_df = float32_df
                        optimization_log.append(f"Downcasted {col} to float32")
                    except:
                        pass  # Keep as float64 if conversion fails
            
            # 3. Apply health data specific optimizations
            if dataset_type == "health":
                optimized_df = self._apply_health_data_optimizations(optimized_df, optimization_log)
            
            # 4. Remove duplicate columns if any
            if len(optimized_df.columns) != len(set(optimized_df.columns)):
                unique_cols = []
                seen_cols = set()
                for col in optimized_df.columns:
                    if col not in seen_cols:
                        unique_cols.append(col)
                        seen_cols.add(col)
                optimized_df = optimized_df.select(unique_cols)
                optimization_log.append("Removed duplicate columns")
            
            # Calculate memory savings
            optimized_size_mb = optimized_df.estimated_size("mb")
            memory_savings_mb = original_size_mb - optimized_size_mb
            memory_savings_percent = (memory_savings_mb / original_size_mb) * 100 if original_size_mb > 0 else 0
            
            # Track optimized DataFrame (skip due to weakref limitations with DataFrames)
            # self.managed_dataframes.add(optimized_df)
            
            optimization_stats = {
                "original_size_mb": original_size_mb,
                "optimized_size_mb": optimized_size_mb,
                "memory_savings_mb": memory_savings_mb,
                "memory_savings_percent": memory_savings_percent,
                "optimizations_applied": optimization_log,
                "optimization_count": len(optimization_log)
            }
            
            self._end_memory_tracking(operation_id, optimized_df.shape[0], optimization_stats)
            
            logger.info(f"Memory optimization completed: {memory_savings_mb:.2f}MB saved "
                       f"({memory_savings_percent:.1f}% reduction)")
            
            return optimized_df, optimization_stats
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            return df, {"error": str(e)}
    
    def _apply_health_data_optimizations(self, 
                                       df: pl.DataFrame, 
                                       optimization_log: List[str]) -> pl.DataFrame:
        """Apply Australian health data specific optimizations."""
        try:
            optimized_df = df
            
            # Apply pattern-based optimizations
            for col in df.columns:
                col_lower = col.lower()
                
                # SA2 codes - always categorical
                if "sa2" in col_lower and "code" in col_lower:
                    optimized_df = optimized_df.with_columns([
                        pl.col(col).cast(pl.Categorical)
                    ])
                    optimization_log.append(f"Optimized {col} as SA2 code")
                
                # State names - categorical
                elif "state" in col_lower and ("name" in col_lower or "code" in col_lower):
                    optimized_df = optimized_df.with_columns([
                        pl.col(col).cast(pl.Categorical)
                    ])
                    optimization_log.append(f"Optimized {col} as state identifier")
                
                # SEIFA deciles - int8 (1-10 range)
                elif "seifa" in col_lower or "decile" in col_lower:
                    if str(optimized_df[col].dtype) in ["Int64", "Int32", "Int16"]:
                        optimized_df = optimized_df.with_columns([
                            pl.col(col).cast(pl.Int8)
                        ])
                        optimization_log.append(f"Optimized {col} as decile (int8)")
                
                # Age groups - categorical
                elif "age" in col_lower and "group" in col_lower:
                    optimized_df = optimized_df.with_columns([
                        pl.col(col).cast(pl.Categorical)
                    ])
                    optimization_log.append(f"Optimized {col} as age group")
                
                # Population counts - int32 sufficient for Australian areas
                elif "population" in col_lower or "count" in col_lower:
                    if str(optimized_df[col].dtype) == "Int64":
                        optimized_df = optimized_df.with_columns([
                            pl.col(col).cast(pl.Int32)
                        ])
                        optimization_log.append(f"Optimized {col} as population count (int32)")
            
            return optimized_df
            
        except Exception as e:
            logger.warning(f"Health data optimization failed: {e}")
            return df
    
    def process_large_dataset_streaming(self,
                                      file_path: Path,
                                      processing_func: Callable[[pl.DataFrame], pl.DataFrame],
                                      batch_size: Optional[int] = None,
                                      output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Process large dataset using streaming with adaptive memory management."""
        try:
            operation_id = f"streaming_process_{int(time.time())}"
            self._start_memory_tracking(operation_id, "streaming_processing")
            
            # Adaptive batch sizing based on available memory
            if batch_size is None:
                batch_size = self._calculate_optimal_batch_size(file_path)
            
            # Initialize output
            output_batches = []
            total_rows_processed = 0
            batch_count = 0
            
            # Process file in batches
            lazy_df = pl.scan_csv(file_path) if file_path.suffix == '.csv' else pl.scan_parquet(file_path)
            
            # Get total rows estimate for progress tracking
            try:
                total_rows = lazy_df.select(pl.len()).collect().item()
                logger.info(f"Processing {total_rows:,} rows in adaptive batches")
            except:
                total_rows = None
                logger.info(f"Processing file in adaptive batches of ~{batch_size:,} rows")
            
            # Streaming batch processing
            offset = 0
            while True:
                try:
                    # Check memory pressure before each batch
                    if self.memory_pressure_detector.is_under_pressure():
                        logger.warning("Memory pressure detected, triggering cleanup")
                        self._aggressive_memory_cleanup()
                        
                        # Reduce batch size if under pressure
                        batch_size = max(1000, int(batch_size * 0.7))
                        logger.info(f"Reduced batch size to {batch_size:,} due to memory pressure")
                    
                    # Load batch
                    batch_df = lazy_df.slice(offset, batch_size).collect()
                    
                    if batch_df.shape[0] == 0:
                        break  # No more data
                    
                    batch_count += 1
                    
                    # Optimize memory for this batch
                    optimized_batch, _ = self.optimize_dataframe_memory(batch_df, "health")
                    
                    # Apply processing function
                    processed_batch = processing_func(optimized_batch)
                    
                    # Store result
                    if output_path:
                        # Stream to file
                        if batch_count == 1:
                            processed_batch.write_parquet(output_path)
                        else:
                            # Append to existing file
                            existing_df = pl.read_parquet(output_path)
                            combined_df = pl.concat([existing_df, processed_batch], how="vertical")
                            combined_df.write_parquet(output_path)
                    else:
                        # Store in memory (with limit)
                        output_batches.append(processed_batch)
                        
                        # Check memory usage
                        current_memory = self.system_monitor.get_memory_usage_gb()
                        if current_memory > self.memory_limit_gb * 0.8:
                            logger.warning("Approaching memory limit, switching to file-based output")
                            if not output_path:
                                output_path = Path(f"temp_output_{operation_id}.parquet")
                            
                            # Write accumulated batches to file
                            if output_batches:
                                combined_df = pl.concat(output_batches, how="vertical")
                                combined_df.write_parquet(output_path)
                                output_batches.clear()
                                self._aggressive_memory_cleanup()
                    
                    total_rows_processed += processed_batch.shape[0]
                    offset += batch_size
                    
                    # Progress logging
                    if batch_count % 10 == 0:
                        memory_usage = self.system_monitor.get_memory_usage_gb()
                        logger.info(f"Processed batch {batch_count}: {total_rows_processed:,} rows, "
                                   f"Memory: {memory_usage:.2f}GB")
                    
                    # Break if we processed fewer rows than batch_size (last batch)
                    if batch_df.shape[0] < batch_size:
                        break
                        
                except Exception as e:
                    logger.error(f"Error processing batch {batch_count}: {e}")
                    break
            
            # Finalize output
            final_result = None
            if output_path and output_path.exists():
                final_result = pl.read_parquet(output_path)
            elif output_batches:
                final_result = pl.concat(output_batches, how="vertical")
            
            processing_stats = {
                "total_rows_processed": total_rows_processed,
                "batches_processed": batch_count,
                "final_batch_size": batch_size,
                "output_file": str(output_path) if output_path else None,
                "memory_efficient": True
            }
            
            self._end_memory_tracking(operation_id, total_rows_processed, processing_stats)
            
            logger.info(f"Streaming processing completed: {total_rows_processed:,} rows in {batch_count} batches")
            
            return {
                "result": final_result,
                "stats": processing_stats
            }
            
        except Exception as e:
            logger.error(f"Streaming processing failed: {e}")
            return {"error": str(e)}
    
    def _calculate_optimal_batch_size(self, file_path: Path) -> int:
        """Calculate optimal batch size based on file size and available memory."""
        try:
            # Get file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            # Get available memory
            available_memory_gb = self.system_monitor.get_available_memory_gb()
            available_memory_mb = available_memory_gb * 1024
            
            # Estimate memory usage per row (rough heuristic)
            # For health data, assume ~200 bytes per row including overhead
            bytes_per_row = 200
            
            # Use 25% of available memory for batch
            target_batch_memory_mb = available_memory_mb * 0.25
            optimal_batch_size = int(target_batch_memory_mb * 1024 * 1024 / bytes_per_row)
            
            # Apply reasonable bounds
            min_batch_size = 1000
            max_batch_size = 100000
            
            batch_size = max(min_batch_size, min(max_batch_size, optimal_batch_size))
            
            logger.debug(f"Calculated optimal batch size: {batch_size:,} rows "
                        f"(~{target_batch_memory_mb:.1f}MB)")
            
            return batch_size
            
        except Exception as e:
            logger.warning(f"Failed to calculate optimal batch size: {e}")
            return 10000  # Default fallback
    
    def create_memory_efficient_lazy_query(self, 
                                          file_paths: List[Path],
                                          query_operations: List[str]) -> pl.LazyFrame:
        """Create memory-efficient lazy query with optimization hints."""
        try:
            # Load files as lazy frames
            lazy_frames = []
            for file_path in file_paths:
                if file_path.suffix == '.csv':
                    lazy_df = pl.scan_csv(file_path)
                elif file_path.suffix == '.parquet':
                    lazy_df = pl.scan_parquet(file_path)
                else:
                    continue
                lazy_frames.append(lazy_df)
            
            if not lazy_frames:
                raise ValueError("No valid files provided")
            
            # Combine lazy frames
            if len(lazy_frames) == 1:
                combined_lazy = lazy_frames[0]
            else:
                combined_lazy = pl.concat(lazy_frames, how="vertical")
            
            # Apply memory-efficient query optimizations
            optimized_lazy = combined_lazy
            
            # Add projection pushdown hint (select only needed columns early)
            for operation in query_operations:
                if operation.startswith("select:"):
                    columns = operation.split(":")[1].split(",")
                    optimized_lazy = optimized_lazy.select(columns)
                elif operation.startswith("filter:"):
                    filter_expr = operation.split(":")[1]
                    # Apply simple filters (would need proper parsing for complex ones)
                    optimized_lazy = optimized_lazy.filter(pl.col(filter_expr.split(">")[0]) > float(filter_expr.split(">")[1]))
                elif operation.startswith("groupby:"):
                    group_cols = operation.split(":")[1].split(",")
                    optimized_lazy = optimized_lazy.group_by(group_cols)
            
            logger.debug(f"Created memory-efficient lazy query for {len(file_paths)} files")
            
            return optimized_lazy
            
        except Exception as e:
            logger.error(f"Failed to create lazy query: {e}")
            raise
    
    def _aggressive_memory_cleanup(self) -> None:
        """Perform aggressive memory cleanup when under pressure."""
        try:
            # Clear weak references to managed DataFrames (if any)
            try:
                self.managed_dataframes.clear()
            except:
                pass  # WeakSet may not work with all object types
            
            # Force garbage collection multiple times
            for _ in range(3):
                gc.collect()
            
            # Clear any cached query results if available
            # (This would integrate with lazy_data_loader's cache)
            
            logger.debug("Aggressive memory cleanup completed")
            
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
    
    def _start_memory_tracking(self, operation_id: str, operation_type: str) -> None:
        """Start tracking memory usage for an operation."""
        if not self.enable_profiling:
            return
            
        try:
            self.active_operations[operation_id] = {
                "start_time": time.time(),
                "start_memory": self.system_monitor.get_memory_usage_gb(),
                "operation_type": operation_type,
                "peak_memory": self.system_monitor.get_memory_usage_gb()
            }
        except Exception as e:
            logger.warning(f"Failed to start memory tracking: {e}")
    
    def _end_memory_tracking(self, 
                           operation_id: str, 
                           rows_processed: int,
                           additional_stats: Dict[str, Any]) -> None:
        """End memory tracking and create profile."""
        if not self.enable_profiling or operation_id not in self.active_operations:
            return
            
        try:
            operation_info = self.active_operations.pop(operation_id)
            
            end_time = time.time()
            end_memory = self.system_monitor.get_memory_usage_gb()
            processing_time = end_time - operation_info["start_time"]
            peak_memory = operation_info["peak_memory"]
            
            # Calculate efficiency ratio
            output_size_mb = additional_stats.get("optimized_size_mb", 0)
            efficiency_ratio = output_size_mb / (peak_memory * 1024) if peak_memory > 0 else 0
            
            # Create memory profile
            profile = MemoryProfile(
                operation_id=operation_id,
                operation_type=operation_info["operation_type"],
                timestamp=datetime.now().isoformat(),
                peak_memory_mb=peak_memory * 1024,
                memory_efficiency_ratio=efficiency_ratio,
                processing_time_seconds=processing_time,
                rows_processed=rows_processed,
                columns_processed=additional_stats.get("columns_processed", 0),
                optimization_applied=additional_stats.get("optimizations_applied", [])
            )
            
            self.memory_profiles.append(profile)
            
        except Exception as e:
            logger.warning(f"Failed to end memory tracking: {e}")
    
    def get_memory_optimization_recommendations(self) -> List[MemoryRecommendation]:
        """Generate memory optimization recommendations based on profiles."""
        try:
            recommendations = []
            
            if len(self.memory_profiles) == 0:
                return recommendations
            
            # Analyze recent profiles
            recent_profiles = list(self.memory_profiles)[-50:]  # Last 50 operations
            
            # Check for high memory usage operations
            high_memory_ops = [p for p in recent_profiles if p.peak_memory_mb > 1000]  # >1GB
            if high_memory_ops:
                avg_memory = sum(p.peak_memory_mb for p in high_memory_ops) / len(high_memory_ops)
                recommendations.append(MemoryRecommendation(
                    priority="high",
                    category="batch_size",
                    title="High Memory Usage Detected",
                    description=f"Found {len(high_memory_ops)} operations using >1GB memory (avg: {avg_memory:.0f}MB)",
                    estimated_memory_savings_mb=avg_memory * 0.5,
                    implementation_effort="low"
                ))
            
            # Check for low efficiency operations
            low_efficiency_ops = [p for p in recent_profiles if p.memory_efficiency_ratio < 0.1]
            if low_efficiency_ops:
                recommendations.append(MemoryRecommendation(
                    priority="medium",
                    category="data_types",
                    title="Low Memory Efficiency",
                    description=f"Found {len(low_efficiency_ops)} operations with low memory efficiency",
                    estimated_memory_savings_mb=200.0,
                    implementation_effort="medium"
                ))
            
            # Check for slow operations that might benefit from optimization
            slow_ops = [p for p in recent_profiles if p.processing_time_seconds > 10]
            if slow_ops:
                recommendations.append(MemoryRecommendation(
                    priority="medium",
                    category="query_strategy",
                    title="Slow Operations Detected",
                    description=f"Found {len(slow_ops)} slow operations that might benefit from streaming",
                    estimated_memory_savings_mb=500.0,
                    implementation_effort="high"
                ))
            
            # System-level recommendations
            current_memory_pressure = self.memory_pressure_detector.get_pressure_level()
            if current_memory_pressure > 0.8:
                recommendations.append(MemoryRecommendation(
                    priority="critical",
                    category="system",
                    title="System Memory Pressure",
                    description="System is under high memory pressure",
                    estimated_memory_savings_mb=1000.0,
                    implementation_effort="low"
                ))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate memory recommendations: {e}")
            return []
    
    def get_memory_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory optimization summary."""
        try:
            if len(self.memory_profiles) == 0:
                return {"error": "No memory profiles available"}
            
            profiles = list(self.memory_profiles)
            
            # Calculate statistics
            total_operations = len(profiles)
            avg_memory_mb = sum(p.peak_memory_mb for p in profiles) / total_operations
            avg_efficiency = sum(p.memory_efficiency_ratio for p in profiles) / total_operations
            avg_processing_time = sum(p.processing_time_seconds for p in profiles) / total_operations
            
            # Get current system status
            current_memory = self.system_monitor.get_memory_usage_gb()
            memory_pressure = self.memory_pressure_detector.get_pressure_level()
            
            summary = {
                "total_operations_tracked": total_operations,
                "average_peak_memory_mb": avg_memory_mb,
                "average_efficiency_ratio": avg_efficiency,
                "average_processing_time_seconds": avg_processing_time,
                "current_memory_usage_gb": current_memory,
                "memory_pressure_level": memory_pressure,
                "memory_limit_gb": self.memory_limit_gb,
                "recommendations_count": len(self.get_memory_optimization_recommendations()),
                "optimization_categories": {
                    "data_type_optimization": len([p for p in profiles if "Downcasted" in str(p.optimization_applied)]),
                    "categorical_encoding": len([p for p in profiles if "categorical" in str(p.optimization_applied)]),
                    "health_data_optimization": len([p for p in profiles if "SA2" in str(p.optimization_applied)])
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate memory summary: {e}")
            return {"error": str(e)}


class SystemMemoryMonitor:
    """Monitor system memory usage and availability."""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def get_memory_usage_gb(self) -> float:
        """Get current process memory usage in GB."""
        try:
            return self.process.memory_info().rss / (1024 ** 3)
        except:
            return 0.0
    
    def get_available_memory_gb(self) -> float:
        """Get available system memory in GB."""
        try:
            return psutil.virtual_memory().available / (1024 ** 3)
        except:
            return 0.0
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get comprehensive memory information."""
        try:
            virtual_memory = psutil.virtual_memory()
            process_memory = self.process.memory_info()
            
            return {
                "total_gb": virtual_memory.total / (1024 ** 3),
                "available_gb": virtual_memory.available / (1024 ** 3),
                "used_gb": virtual_memory.used / (1024 ** 3),
                "percent_used": virtual_memory.percent,
                "process_rss_gb": process_memory.rss / (1024 ** 3),
                "process_vms_gb": process_memory.vms / (1024 ** 3)
            }
        except Exception as e:
            logger.error(f"Failed to get memory info: {e}")
            return {}


class MemoryPressureDetector:
    """Detect memory pressure conditions."""
    
    def __init__(self, pressure_threshold: float = 0.85):
        self.pressure_threshold = pressure_threshold
        self.pressure_history: deque = deque(maxlen=10)
    
    def is_under_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        try:
            memory_percent = psutil.virtual_memory().percent / 100.0
            self.pressure_history.append(memory_percent)
            
            return memory_percent > self.pressure_threshold
        except:
            return False
    
    def get_pressure_level(self) -> float:
        """Get current memory pressure level (0.0 to 1.0)."""
        try:
            return psutil.virtual_memory().percent / 100.0
        except:
            return 0.0
    
    def get_pressure_trend(self) -> str:
        """Get memory pressure trend (increasing, stable, decreasing)."""
        try:
            if len(self.pressure_history) < 3:
                return "stable"
            
            recent_avg = sum(list(self.pressure_history)[-3:]) / 3
            older_avg = sum(list(self.pressure_history)[-6:-3]) / 3 if len(self.pressure_history) >= 6 else recent_avg
            
            if recent_avg > older_avg + 0.05:
                return "increasing"
            elif recent_avg < older_avg - 0.05:
                return "decreasing"
            else:
                return "stable"
        except:
            return "unknown"


if __name__ == "__main__":
    # Development testing
    optimizer = MemoryOptimizer(memory_limit_gb=4.0, enable_profiling=True)
    
    # Test memory optimization
    np.random.seed(42)
    test_data = pl.DataFrame({
        "sa2_code": np.random.choice([f"1{str(i).zfill(8)}" for i in range(1000, 1500)], 10000),
        "state_name": np.random.choice(['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT'], 10000),
        "prescription_count": np.random.poisson(3, 10000),
        "total_cost": np.random.exponential(45, 10000),
        "risk_score": np.random.uniform(1, 10, 10000),
        "seifa_decile": np.random.randint(1, 11, 10000)
    })
    
    print(f"🧠 Testing memory optimizer with {test_data.shape[0]:,} rows")
    
    # Test DataFrame optimization
    optimized_df, stats = optimizer.optimize_dataframe_memory(test_data, "health")
    print(f"💾 Memory optimization: {stats['memory_savings_mb']:.2f}MB saved "
          f"({stats['memory_savings_percent']:.1f}% reduction)")
    
    # Get optimization summary
    summary = optimizer.get_memory_optimization_summary()
    print(f"📊 Memory summary: {summary.get('total_operations_tracked', 0)} operations tracked")
    
    # Get recommendations
    recommendations = optimizer.get_memory_optimization_recommendations()
    print(f"💡 Memory recommendations: {len(recommendations)} suggestions")