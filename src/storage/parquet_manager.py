"""
AHGD V3: Parquet-First Data Storage Manager
High-performance Parquet storage with optimized partitioning and compression.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Optional

import polars as pl

from src.utils.config import get_config

logger = logging.getLogger(__name__)


class ParquetStorageManager:
    """
    High-performance Parquet storage manager for AHGD data.

    Provides:
    - Optimized partitioning by geographic and temporal dimensions
    - Compression tuned for health analytics
    - Fast query capabilities
    - Automatic schema evolution
    """

    def __init__(self, base_path: str = "./data/parquet_store"):
        self.base_path = Path(base_path)
        self.config = get_config()

        # Create storage structure
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        self.cache_path = self.base_path / "cache"
        self.exports_path = self.base_path / "exports"

        # Create directories
        for path in [self.raw_path, self.processed_path, self.cache_path, self.exports_path]:
            path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized Parquet storage at {self.base_path}")

    def store_raw_data(
        self, df: pl.DataFrame, source: str, dataset: str, partition_by: Optional[list[str]] = None
    ) -> Path:
        """
        Store raw extracted data with optimal partitioning.

        Args:
            df: Polars DataFrame to store
            source: Data source (aihw, abs, bom, phidu)
            dataset: Dataset name (mortality, census, climate, etc.)
            partition_by: Optional partition columns

        Returns:
            Path to stored Parquet file/directory
        """
        storage_path = self.raw_path / source / dataset
        storage_path.mkdir(parents=True, exist_ok=True)

        # Add metadata columns
        df_with_meta = df.with_columns(
            [
                pl.lit(source).alias("_source"),
                pl.lit(dataset).alias("_dataset"),
                pl.lit(datetime.now()).alias("_extracted_at"),
                pl.lit("raw").alias("_stage"),
            ]
        )

        if partition_by:
            # Partitioned storage for large datasets
            parquet_path = storage_path / "partitioned"
            df_with_meta.write_parquet(
                parquet_path,
                compression="snappy",  # Balanced compression/speed
                statistics=True,
                row_group_size=50000,
                partition_by=partition_by,
            )
        else:
            # Single file storage
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            parquet_path = storage_path / f"{dataset}_{timestamp}.parquet"
            df_with_meta.write_parquet(
                parquet_path, compression="snappy", statistics=True, row_group_size=50000
            )

        logger.info(f"Stored raw data: {source}/{dataset} -> {parquet_path}")
        return parquet_path

    def store_processed_data(
        self,
        df: pl.DataFrame,
        table_name: str,
        geographic_level: str = "sa1",
        partition_by_state: bool = True,
    ) -> Path:
        """
        Store processed health analytics data with geographic partitioning.

        Args:
            df: Processed DataFrame
            table_name: Analytics table name
            geographic_level: Geographic granularity (sa1, sa2, lga)
            partition_by_state: Whether to partition by state

        Returns:
            Path to stored data
        """
        storage_path = self.processed_path / geographic_level / table_name
        storage_path.mkdir(parents=True, exist_ok=True)

        # Add processing metadata
        df_with_meta = df.with_columns(
            [
                pl.lit(table_name).alias("_table"),
                pl.lit(geographic_level).alias("_geographic_level"),
                pl.lit(datetime.now()).alias("_processed_at"),
                pl.lit("processed").alias("_stage"),
            ]
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if partition_by_state and "state_code" in df_with_meta.columns:
            # Partition by state for efficient regional queries
            parquet_path = storage_path / f"partitioned_{timestamp}"
            df_with_meta.write_parquet(
                parquet_path,
                compression="zstd",  # Higher compression for processed data
                statistics=True,
                row_group_size=100000,
                partition_by=["state_code"],
            )
        else:
            parquet_path = storage_path / f"{table_name}_{timestamp}.parquet"
            df_with_meta.write_parquet(
                parquet_path, compression="zstd", statistics=True, row_group_size=100000
            )

        # Also store latest version without timestamp
        latest_path = storage_path / "latest.parquet"
        df_with_meta.write_parquet(latest_path, compression="zstd")

        logger.info(f"Stored processed data: {table_name} -> {parquet_path}")
        return parquet_path

    def cache_intermediate_result(
        self, df: pl.DataFrame, cache_key: str, ttl_hours: int = 24
    ) -> Path:
        """
        Cache intermediate processing results with TTL.

        Args:
            df: DataFrame to cache
            cache_key: Unique cache identifier
            ttl_hours: Time-to-live in hours

        Returns:
            Path to cached file
        """
        cache_file = self.cache_path / f"{cache_key}.parquet"

        # Add cache metadata
        df_with_cache = df.with_columns(
            [
                pl.lit(cache_key).alias("_cache_key"),
                pl.lit(datetime.now()).alias("_cached_at"),
                pl.lit(ttl_hours).alias("_ttl_hours"),
                pl.lit("cache").alias("_stage"),
            ]
        )

        df_with_cache.write_parquet(
            cache_file,
            compression="lz4",  # Fastest compression for cache
            statistics=False,  # Skip stats for cache files
            row_group_size=25000,
        )

        logger.debug(f"Cached intermediate result: {cache_key}")
        return cache_file

    def load_raw_data(
        self, source: str, dataset: str, filters: Optional[dict[str, Any]] = None
    ) -> Optional[pl.LazyFrame]:
        """
        Load raw data with optional filtering.

        Args:
            source: Data source name
            dataset: Dataset name
            filters: Optional filters to apply

        Returns:
            LazyFrame for efficient processing
        """
        source_path = self.raw_path / source / dataset

        if not source_path.exists():
            logger.warning(f"Raw data not found: {source}/{dataset}")
            return None

        # Find latest data file/directory
        parquet_files = list(source_path.glob("*.parquet"))
        partition_dirs = [d for d in source_path.iterdir() if d.is_dir()]

        if partition_dirs:
            # Load from partitioned storage
            latest_partition = max(partition_dirs, key=lambda p: p.stat().st_mtime)
            lf = pl.scan_parquet(latest_partition)
        elif parquet_files:
            # Load from single file
            latest_file = max(parquet_files, key=lambda p: p.stat().st_mtime)
            lf = pl.scan_parquet(latest_file)
        else:
            logger.warning(f"No Parquet files found in {source_path}")
            return None

        # Apply filters if provided
        if filters:
            for col, value in filters.items():
                if isinstance(value, (list, tuple)):
                    lf = lf.filter(pl.col(col).is_in(value))
                else:
                    lf = lf.filter(pl.col(col) == value)

        logger.debug(f"Loaded raw data: {source}/{dataset}")
        return lf

    def load_processed_data(
        self, table_name: str, geographic_level: str = "sa1", use_latest: bool = True
    ) -> Optional[pl.LazyFrame]:
        """
        Load processed analytics data.

        Args:
            table_name: Analytics table name
            geographic_level: Geographic level
            use_latest: Whether to use latest version

        Returns:
            LazyFrame for efficient querying
        """
        table_path = self.processed_path / geographic_level / table_name

        if not table_path.exists():
            logger.warning(f"Processed table not found: {table_name}")
            return None

        if use_latest:
            latest_file = table_path / "latest.parquet"
            if latest_file.exists():
                lf = pl.scan_parquet(latest_file)
                logger.debug(f"Loaded latest processed data: {table_name}")
                return lf

        # Find most recent file
        parquet_files = list(table_path.glob("*.parquet"))
        if parquet_files:
            latest_file = max(parquet_files, key=lambda p: p.stat().st_mtime)
            lf = pl.scan_parquet(latest_file)
            logger.debug(f"Loaded processed data: {table_name}")
            return lf

        logger.warning(f"No processed data found for: {table_name}")
        return None

    def get_cache(self, cache_key: str) -> Optional[pl.LazyFrame]:
        """
        Retrieve cached data if still valid.

        Args:
            cache_key: Cache identifier

        Returns:
            LazyFrame if cache hit, None if miss/expired
        """
        cache_file = self.cache_path / f"{cache_key}.parquet"

        if not cache_file.exists():
            return None

        # Check TTL (simplified - in production use proper metadata table)
        cache_age_hours = (
            datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        ).total_seconds() / 3600

        if cache_age_hours > 24:  # Default TTL
            cache_file.unlink()  # Remove expired cache
            return None

        lf = pl.scan_parquet(cache_file)
        logger.debug(f"Cache hit: {cache_key}")
        return lf

    def export_for_analysis(
        self,
        df: pl.DataFrame,
        export_name: str,
        format: str = "parquet",
        optimize_for: str = "analytics",
    ) -> Path:
        """
        Export data optimized for specific analysis workflows.

        Args:
            df: DataFrame to export
            export_name: Export filename
            format: Export format (parquet, csv, json)
            optimize_for: Optimization target (analytics, web, ml)

        Returns:
            Path to exported file
        """
        export_path = self.exports_path / export_name
        export_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "parquet":
            if optimize_for == "analytics":
                # Heavy compression, column stats
                file_path = export_path / f"{export_name}_analytics_{timestamp}.parquet"
                df.write_parquet(
                    file_path,
                    compression="zstd",  # Maximum compression
                    statistics=True,
                    row_group_size=200000,
                )
            elif optimize_for == "web":
                # Balanced compression, smaller row groups
                file_path = export_path / f"{export_name}_web_{timestamp}.parquet"
                df.write_parquet(
                    file_path, compression="snappy", statistics=False, row_group_size=10000
                )
            else:  # ml
                # Optimized for ML workflows
                file_path = export_path / f"{export_name}_ml_{timestamp}.parquet"
                df.write_parquet(
                    file_path, compression="lz4", statistics=False, row_group_size=50000
                )
        elif format == "csv":
            file_path = export_path / f"{export_name}_{timestamp}.csv"
            df.write_csv(file_path)
        elif format == "json":
            file_path = export_path / f"{export_name}_{timestamp}.json"
            df.write_ndjson(file_path)

        logger.info(f"Exported {format} file: {file_path}")
        return file_path

    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics and health metrics."""

        def get_dir_size(path: Path) -> int:
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

        def count_files(path: Path, pattern: str = "*.parquet") -> int:
            return len(list(path.rglob(pattern)))

        stats = {
            "storage_path": str(self.base_path),
            "total_size_mb": get_dir_size(self.base_path) / (1024 * 1024),
            "raw_data": {
                "size_mb": get_dir_size(self.raw_path) / (1024 * 1024),
                "files": count_files(self.raw_path),
            },
            "processed_data": {
                "size_mb": get_dir_size(self.processed_path) / (1024 * 1024),
                "files": count_files(self.processed_path),
            },
            "cache": {
                "size_mb": get_dir_size(self.cache_path) / (1024 * 1024),
                "files": count_files(self.cache_path),
            },
            "exports": {
                "size_mb": get_dir_size(self.exports_path) / (1024 * 1024),
                "files": count_files(self.exports_path),
            },
        }

        return stats
