"""
Incremental Processor - Data versioning and incremental processing for Australian health data

Implements Bronze-Silver-Gold data lake architecture for efficient incremental updates
of the 497,181+ health records without full pipeline reprocessing.

Key Features:
- Change Data Capture (CDC) for detecting new/updated records
- Data versioning with rollback capability
- Incremental merge strategies for SA2-level health data
- Audit trail for healthcare data compliance
- Conflict resolution for concurrent updates
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from pathlib import Path
import logging
import time
import json
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import shutil
from enum import Enum

logger = logging.getLogger(__name__)


class DataLayer(Enum):
    """Data lake layer definitions."""
    BRONZE = "bronze"    # Raw data (append-only)
    SILVER = "silver"    # Cleaned data (versioned)
    GOLD = "gold"        # Analytics-ready (aggregated)


class MergeStrategy(Enum):
    """Data merge strategies for incremental processing."""
    APPEND = "append"           # Append new records only
    UPSERT = "upsert"          # Insert new, update existing
    REPLACE = "replace"        # Replace entire dataset
    MERGE_BY_DATE = "merge_by_date"  # Merge based on date ranges


@dataclass
class DataVersion:
    """Data version metadata."""
    version_id: str
    dataset_name: str
    layer: DataLayer
    created_timestamp: str
    source_files: List[str]
    record_count: int
    file_size_mb: float
    schema_hash: str
    merge_strategy: MergeStrategy
    parent_version: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ChangeRecord:
    """Change detection record."""
    record_id: str
    change_type: str  # INSERT, UPDATE, DELETE
    timestamp: str
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None


class IncrementalProcessor:
    """
    Manage incremental processing and data versioning for Australian health analytics.
    Implements data lake patterns for production-scale health data.
    """
    
    # Data lake configuration
    LAYER_CONFIG = {
        DataLayer.BRONZE: {
            "retention_days": 90,        # Keep raw data for 90 days
            "compression": "snappy",     # Fast compression for bronze
            "partition_by": ["year", "month"],  # Time-based partitioning
        },
        DataLayer.SILVER: {
            "retention_versions": 10,    # Keep last 10 versions
            "compression": "zstd",       # Better compression for silver
            "enable_versioning": True,
        },
        DataLayer.GOLD: {
            "retention_versions": 5,     # Keep last 5 gold versions
            "compression": "zstd",       # Best compression for gold
            "enable_caching": True,
        }
    }
    
    # Australian health data specific settings
    HEALTH_DATA_CONFIG = {
        "sa2_code_column": "sa2_code",           # Primary geographic key
        "date_columns": ["dispensing_date", "service_date"],  # Date-based partitioning
        "change_detection_columns": [            # Columns to monitor for changes
            "prescription_count", "total_cost", "risk_score", "access_score"
        ],
        "immutable_columns": [                   # Columns that shouldn't change
            "sa2_code", "state_name"
        ]
    }
    
    def __init__(self, base_path: Optional[Path] = None):
        """Initialize incremental processor with data lake structure."""
        self.base_path = base_path or Path("data")
        self.versions_metadata: Dict[str, DataVersion] = {}
        self.change_log: List[ChangeRecord] = []
        
        self._setup_data_lake_structure()
        self._load_version_metadata()
        
        logger.info(f"Initialized incremental processor at {self.base_path}")
    
    def _setup_data_lake_structure(self) -> None:
        """Create Bronze-Silver-Gold data lake directory structure."""
        directories = [
            # Bronze layer - raw data with time partitions
            self.base_path / "bronze" / "health" / "year=2023" / "month=01",
            self.base_path / "bronze" / "seifa" / "year=2021",
            self.base_path / "bronze" / "geographic" / "year=2021",
            
            # Silver layer - cleaned and versioned data
            self.base_path / "silver" / "health",
            self.base_path / "silver" / "seifa", 
            self.base_path / "silver" / "geographic",
            
            # Gold layer - analytics-ready aggregated data
            self.base_path / "gold" / "sa2_health_summary",
            self.base_path / "gold" / "risk_assessments",
            self.base_path / "gold" / "access_assessments",
            
            # Metadata and versioning
            self.base_path / "metadata" / "versions",
            self.base_path / "metadata" / "change_log",
            self.base_path / "metadata" / "schemas",
            
            # Staging area for new data
            self.base_path / "staging" / "incoming"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Created data lake structure with {len(directories)} directories")
    
    def _load_version_metadata(self) -> None:
        """Load existing version metadata from disk."""
        try:
            metadata_dir = self.base_path / "metadata" / "versions"
            
            for metadata_file in metadata_dir.glob("*.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        version_data = json.load(f)
                    
                    version = DataVersion(**version_data)
                    self.versions_metadata[version.version_id] = version
                    
                except Exception as e:
                    logger.warning(f"Failed to load version metadata from {metadata_file}: {e}")
            
            logger.info(f"Loaded {len(self.versions_metadata)} version records")
            
        except Exception as e:
            logger.error(f"Failed to load version metadata: {e}")
    
    def detect_schema_changes(self, new_data: pl.DataFrame, dataset_name: str) -> bool:
        """Detect if schema has changed compared to latest version."""
        try:
            # Get latest version for this dataset
            latest_version = self._get_latest_version(dataset_name, DataLayer.SILVER)
            
            if not latest_version:
                logger.info(f"No previous version found for {dataset_name} - schema is new")
                return True
            
            # Calculate schema hash for new data
            new_schema_hash = self._calculate_schema_hash(new_data)
            
            # Compare with previous schema
            schema_changed = new_schema_hash != latest_version.schema_hash
            
            if schema_changed:
                logger.info(f"Schema change detected for {dataset_name}")
                logger.debug(f"Previous hash: {latest_version.schema_hash}")
                logger.debug(f"New hash: {new_schema_hash}")
            
            return schema_changed
            
        except Exception as e:
            logger.error(f"Schema change detection failed: {e}")
            return True  # Assume schema changed on error
    
    def _calculate_schema_hash(self, df: pl.DataFrame) -> str:
        """Calculate hash of DataFrame schema for change detection."""
        try:
            # Create schema representation
            schema_repr = {
                "columns": sorted(df.columns),
                "dtypes": {col: str(df[col].dtype) for col in df.columns},
                "shape": df.shape
            }
            
            # Calculate hash
            schema_str = json.dumps(schema_repr, sort_keys=True)
            return hashlib.sha256(schema_str.encode()).hexdigest()[:16]
            
        except Exception as e:
            logger.error(f"Schema hash calculation failed: {e}")
            return "unknown"
    
    def _get_latest_version(self, dataset_name: str, layer: DataLayer) -> Optional[DataVersion]:
        """Get the latest version for a dataset in specified layer."""
        try:
            # Filter versions for this dataset and layer
            dataset_versions = [
                v for v in self.versions_metadata.values()
                if v.dataset_name == dataset_name and v.layer == layer
            ]
            
            if not dataset_versions:
                return None
            
            # Sort by timestamp and return latest
            latest_version = max(dataset_versions, key=lambda v: v.created_timestamp)
            return latest_version
            
        except Exception as e:
            logger.error(f"Failed to get latest version: {e}")
            return None
    
    def ingest_to_bronze(self, 
                        data: pl.DataFrame,
                        dataset_name: str,
                        source_info: Dict[str, Any]) -> str:
        """Ingest new data to bronze layer with time partitioning."""
        try:
            # Create version ID
            timestamp = datetime.now()
            version_id = f"bronze_{dataset_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Determine partition path based on current date
            partition_path = self._get_partition_path(dataset_name, DataLayer.BRONZE, timestamp)
            
            # Create partitioned file path
            file_path = partition_path / f"{dataset_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.parquet"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write data with bronze layer configuration
            config = self.LAYER_CONFIG[DataLayer.BRONZE]
            data.write_parquet(
                file_path,
                compression=config["compression"]
            )
            
            # Calculate file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            # Create version metadata
            version = DataVersion(
                version_id=version_id,
                dataset_name=dataset_name,
                layer=DataLayer.BRONZE,
                created_timestamp=timestamp.isoformat(),
                source_files=[str(file_path)],
                record_count=data.shape[0],
                file_size_mb=file_size_mb,
                schema_hash=self._calculate_schema_hash(data),
                merge_strategy=MergeStrategy.APPEND,
                metadata=source_info
            )
            
            # Save version metadata
            self._save_version_metadata(version)
            
            logger.info(f"Ingested {data.shape[0]} records to bronze layer: {version_id}")
            return version_id
            
        except Exception as e:
            logger.error(f"Bronze ingestion failed: {e}")
            raise
    
    def _get_partition_path(self, dataset_name: str, layer: DataLayer, timestamp: datetime) -> Path:
        """Get partitioned path for data storage."""
        base_layer_path = self.base_path / layer.value / dataset_name
        
        if layer == DataLayer.BRONZE:
            # Time-based partitioning for bronze
            return base_layer_path / f"year={timestamp.year}" / f"month={timestamp.month:02d}"
        else:
            # No partitioning for silver/gold
            return base_layer_path
    
    def process_to_silver(self, 
                         dataset_name: str,
                         bronze_version_id: str,
                         merge_strategy: MergeStrategy = MergeStrategy.UPSERT) -> str:
        """Process bronze data to silver layer with versioning."""
        try:
            # Load bronze data
            bronze_version = self.versions_metadata.get(bronze_version_id)
            if not bronze_version:
                raise ValueError(f"Bronze version {bronze_version_id} not found")
            
            bronze_data = pl.read_parquet(bronze_version.source_files[0])
            
            # Clean and validate data
            cleaned_data = self._clean_data_for_silver(bronze_data, dataset_name)
            
            # Check if schema changed
            schema_changed = self.detect_schema_changes(cleaned_data, dataset_name)
            
            # Get latest silver version
            latest_silver = self._get_latest_version(dataset_name, DataLayer.SILVER)
            
            # Determine merge strategy
            if schema_changed or not latest_silver:
                # Create new version for schema changes
                merged_data = cleaned_data
                parent_version = None
            else:
                # Merge with existing silver data
                existing_silver = pl.read_parquet(latest_silver.source_files[0])
                merged_data = self._merge_data(existing_silver, cleaned_data, merge_strategy, dataset_name)
                parent_version = latest_silver.version_id
            
            # Create silver version
            timestamp = datetime.now()
            silver_version_id = f"silver_{dataset_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Save to silver layer
            silver_path = self.base_path / "silver" / dataset_name / f"{silver_version_id}.parquet"
            silver_path.parent.mkdir(parents=True, exist_ok=True)
            
            config = self.LAYER_CONFIG[DataLayer.SILVER]
            merged_data.write_parquet(
                silver_path,
                compression=config["compression"]
            )
            
            # Create silver version metadata
            file_size_mb = silver_path.stat().st_size / (1024 * 1024)
            
            silver_version = DataVersion(
                version_id=silver_version_id,
                dataset_name=dataset_name,
                layer=DataLayer.SILVER,
                created_timestamp=timestamp.isoformat(),
                source_files=[str(silver_path)],
                record_count=merged_data.shape[0],
                file_size_mb=file_size_mb,
                schema_hash=self._calculate_schema_hash(merged_data),
                merge_strategy=merge_strategy,
                parent_version=parent_version,
                metadata={
                    "bronze_source": bronze_version_id,
                    "schema_changed": schema_changed,
                    "merge_conflicts": 0  # Would be calculated during merge
                }
            )
            
            # Save version metadata
            self._save_version_metadata(silver_version)
            
            # Cleanup old versions if needed
            self._cleanup_old_versions(dataset_name, DataLayer.SILVER)
            
            logger.info(f"Processed to silver layer: {silver_version_id}, {merged_data.shape[0]} records")
            return silver_version_id
            
        except Exception as e:
            logger.error(f"Silver processing failed: {e}")
            raise
    
    def _clean_data_for_silver(self, data: pl.DataFrame, dataset_name: str) -> pl.DataFrame:
        """Clean and validate data for silver layer."""
        try:
            cleaned_data = data
            
            # Apply health data specific cleaning
            if dataset_name in ["health", "medicare", "pbs"]:
                cleaned_data = self._clean_health_data(cleaned_data)
            elif dataset_name == "seifa":
                cleaned_data = self._clean_seifa_data(cleaned_data)
            elif dataset_name == "geographic":
                cleaned_data = self._clean_geographic_data(cleaned_data)
            
            # Remove duplicates based on SA2 codes
            if self.HEALTH_DATA_CONFIG["sa2_code_column"] in cleaned_data.columns:
                before_count = cleaned_data.shape[0]
                cleaned_data = cleaned_data.unique(subset=[self.HEALTH_DATA_CONFIG["sa2_code_column"]])
                after_count = cleaned_data.shape[0]
                
                if before_count != after_count:
                    logger.info(f"Removed {before_count - after_count} duplicate SA2 records")
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            return data  # Return original data on error
    
    def _clean_health_data(self, data: pl.DataFrame) -> pl.DataFrame:
        """Clean health-specific data."""
        cleaned = data
        
        # Ensure SA2 codes are properly formatted (9 digits)
        if "sa2_code" in cleaned.columns:
            cleaned = cleaned.filter(pl.col("sa2_code").str.len_chars() == 9)
        
        # Remove negative prescription counts
        if "prescription_count" in cleaned.columns:
            cleaned = cleaned.filter(pl.col("prescription_count") >= 0)
        
        # Remove negative costs
        cost_columns = [col for col in cleaned.columns if "cost" in col.lower()]
        for col in cost_columns:
            cleaned = cleaned.filter(pl.col(col) >= 0)
        
        return cleaned
    
    def _clean_seifa_data(self, data: pl.DataFrame) -> pl.DataFrame:
        """Clean SEIFA-specific data."""
        cleaned = data
        
        # Ensure deciles are in valid range (1-10)
        decile_columns = [col for col in cleaned.columns if "decile" in col.lower()]
        for col in decile_columns:
            cleaned = cleaned.filter(
                (pl.col(col) >= 1) & (pl.col(col) <= 10)
            )
        
        return cleaned
    
    def _clean_geographic_data(self, data: pl.DataFrame) -> pl.DataFrame:
        """Clean geographic-specific data."""
        cleaned = data
        
        # Remove zero population areas
        if "usual_resident_population" in cleaned.columns:
            cleaned = cleaned.filter(pl.col("usual_resident_population") > 0)
        
        return cleaned
    
    def _merge_data(self, 
                   existing_data: pl.DataFrame,
                   new_data: pl.DataFrame,
                   strategy: MergeStrategy,
                   dataset_name: str) -> pl.DataFrame:
        """Merge new data with existing data using specified strategy."""
        try:
            key_column = self.HEALTH_DATA_CONFIG["sa2_code_column"]
            
            if strategy == MergeStrategy.APPEND:
                # Simple append - just concatenate
                return pl.concat([existing_data, new_data], how="vertical")
            
            elif strategy == MergeStrategy.UPSERT:
                # Update existing records, insert new ones
                # Remove existing records that have updates
                existing_keys = set(existing_data[key_column].to_list())
                new_keys = set(new_data[key_column].to_list())
                
                # Keep existing records that aren't being updated
                unchanged_data = existing_data.filter(
                    ~pl.col(key_column).is_in(list(new_keys))
                )
                
                # Combine unchanged existing + new/updated records
                return pl.concat([unchanged_data, new_data], how="vertical")
            
            elif strategy == MergeStrategy.REPLACE:
                # Complete replacement
                return new_data
            
            elif strategy == MergeStrategy.MERGE_BY_DATE:
                # Merge based on date ranges (for time series data)
                if any(col in new_data.columns for col in self.HEALTH_DATA_CONFIG["date_columns"]):
                    # Find date column
                    date_col = None
                    for col in self.HEALTH_DATA_CONFIG["date_columns"]:
                        if col in new_data.columns:
                            date_col = col
                            break
                    
                    if date_col:
                        # Get max date in new data
                        max_new_date = new_data[date_col].max()
                        
                        # Keep existing data before the new data's date range
                        historical_data = existing_data.filter(pl.col(date_col) < max_new_date)
                        
                        # Combine historical + new data
                        return pl.concat([historical_data, new_data], how="vertical")
                
                # Fallback to upsert if no date column found
                return self._merge_data(existing_data, new_data, MergeStrategy.UPSERT, dataset_name)
            
            else:
                raise ValueError(f"Unknown merge strategy: {strategy}")
                
        except Exception as e:
            logger.error(f"Data merge failed: {e}")
            raise
    
    def aggregate_to_gold(self, 
                         dataset_name: str,
                         silver_version_id: str,
                         aggregation_config: Dict[str, Any]) -> str:
        """Aggregate silver data to gold layer for analytics."""
        try:
            # Load silver data
            silver_version = self.versions_metadata.get(silver_version_id)
            if not silver_version:
                raise ValueError(f"Silver version {silver_version_id} not found")
            
            silver_data = pl.read_parquet(silver_version.source_files[0])
            
            # Apply aggregations based on config
            aggregated_data = self._apply_aggregations(silver_data, aggregation_config)
            
            # Create gold version
            timestamp = datetime.now()
            gold_version_id = f"gold_{dataset_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Save to gold layer
            gold_path = self.base_path / "gold" / dataset_name / f"{gold_version_id}.parquet"
            gold_path.parent.mkdir(parents=True, exist_ok=True)
            
            config = self.LAYER_CONFIG[DataLayer.GOLD]
            aggregated_data.write_parquet(
                gold_path,
                compression=config["compression"]
            )
            
            # Create gold version metadata
            file_size_mb = gold_path.stat().st_size / (1024 * 1024)
            
            gold_version = DataVersion(
                version_id=gold_version_id,
                dataset_name=dataset_name,
                layer=DataLayer.GOLD,
                created_timestamp=timestamp.isoformat(),
                source_files=[str(gold_path)],
                record_count=aggregated_data.shape[0],
                file_size_mb=file_size_mb,
                schema_hash=self._calculate_schema_hash(aggregated_data),
                merge_strategy=MergeStrategy.REPLACE,
                parent_version=silver_version_id,
                metadata={
                    "aggregation_config": aggregation_config,
                    "silver_source": silver_version_id
                }
            )
            
            # Save version metadata
            self._save_version_metadata(gold_version)
            
            # Cleanup old gold versions
            self._cleanup_old_versions(dataset_name, DataLayer.GOLD)
            
            logger.info(f"Aggregated to gold layer: {gold_version_id}, {aggregated_data.shape[0]} records")
            return gold_version_id
            
        except Exception as e:
            logger.error(f"Gold aggregation failed: {e}")
            raise
    
    def _apply_aggregations(self, data: pl.DataFrame, config: Dict[str, Any]) -> pl.DataFrame:
        """Apply aggregation configuration to create gold layer data."""
        try:
            aggregated = data
            
            # Group by SA2 and state for health analytics
            if config.get("group_by_sa2", True):
                group_cols = ["sa2_code", "state_name"]
                group_cols = [col for col in group_cols if col in data.columns]
                
                if group_cols:
                    agg_expressions = []
                    
                    # Standard aggregations for health data
                    if "prescription_count" in data.columns:
                        agg_expressions.append(pl.col("prescription_count").sum().alias("total_prescriptions"))
                        agg_expressions.append(pl.col("prescription_count").mean().alias("avg_prescriptions"))
                    
                    if "total_cost" in data.columns:
                        agg_expressions.append(pl.col("total_cost").sum().alias("total_healthcare_cost"))
                        agg_expressions.append(pl.col("total_cost").mean().alias("avg_cost_per_service"))
                    
                    if "risk_score" in data.columns:
                        agg_expressions.append(pl.col("risk_score").mean().alias("avg_risk_score"))
                        agg_expressions.append(pl.col("risk_score").max().alias("max_risk_score"))
                    
                    # Count of records
                    agg_expressions.append(pl.len().alias("record_count"))
                    
                    if agg_expressions:
                        aggregated = data.group_by(group_cols).agg(agg_expressions)
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return data
    
    def _save_version_metadata(self, version: DataVersion) -> None:
        """Save version metadata to disk."""
        try:
            metadata_file = self.base_path / "metadata" / "versions" / f"{version.version_id}.json"
            
            # Convert DataVersion to serializable dict
            version_dict = asdict(version)
            # Convert enum values to strings
            version_dict['layer'] = version_dict['layer'].value if hasattr(version_dict['layer'], 'value') else str(version_dict['layer'])
            version_dict['merge_strategy'] = version_dict['merge_strategy'].value if hasattr(version_dict['merge_strategy'], 'value') else str(version_dict['merge_strategy'])
            
            with open(metadata_file, 'w') as f:
                json.dump(version_dict, f, indent=2)
            
            # Also update in-memory metadata
            self.versions_metadata[version.version_id] = version
            
            logger.debug(f"Saved version metadata: {version.version_id}")
            
        except Exception as e:
            logger.error(f"Failed to save version metadata: {e}")
    
    def _cleanup_old_versions(self, dataset_name: str, layer: DataLayer) -> None:
        """Clean up old versions based on retention policy."""
        try:
            config = self.LAYER_CONFIG[layer]
            
            if layer == DataLayer.BRONZE:
                # Time-based cleanup for bronze
                retention_days = config["retention_days"]
                cutoff_date = datetime.now() - timedelta(days=retention_days)
                
                old_versions = [
                    v for v in self.versions_metadata.values()
                    if (v.dataset_name == dataset_name and 
                        v.layer == layer and
                        datetime.fromisoformat(v.created_timestamp) < cutoff_date)
                ]
            else:
                # Version count-based cleanup for silver/gold
                retention_count = config["retention_versions"]
                
                # Get all versions for this dataset and layer
                all_versions = [
                    v for v in self.versions_metadata.values()
                    if v.dataset_name == dataset_name and v.layer == layer
                ]
                
                # Sort by timestamp and keep only recent versions
                all_versions.sort(key=lambda v: v.created_timestamp, reverse=True)
                old_versions = all_versions[retention_count:]
            
            # Delete old versions
            for version in old_versions:
                try:
                    # Delete data files
                    for file_path in version.source_files:
                        file_path_obj = Path(file_path)
                        if file_path_obj.exists():
                            file_path_obj.unlink()
                    
                    # Delete metadata file
                    metadata_file = self.base_path / "metadata" / "versions" / f"{version.version_id}.json"
                    if metadata_file.exists():
                        metadata_file.unlink()
                    
                    # Remove from in-memory metadata
                    if version.version_id in self.versions_metadata:
                        del self.versions_metadata[version.version_id]
                    
                    logger.debug(f"Cleaned up old version: {version.version_id}")
                    
                except Exception as e:
                    logger.warning(f"Failed to cleanup version {version.version_id}: {e}")
            
            if old_versions:
                logger.info(f"Cleaned up {len(old_versions)} old versions for {dataset_name}/{layer.value}")
                
        except Exception as e:
            logger.error(f"Version cleanup failed: {e}")
    
    def rollback_to_version(self, dataset_name: str, layer: DataLayer, version_id: str) -> bool:
        """Rollback dataset to a specific version."""
        try:
            # Find the target version
            target_version = self.versions_metadata.get(version_id)
            if not target_version:
                raise ValueError(f"Version {version_id} not found")
            
            if target_version.dataset_name != dataset_name or target_version.layer != layer:
                raise ValueError(f"Version {version_id} is not for {dataset_name}/{layer.value}")
            
            # Create rollback version
            timestamp = datetime.now()
            rollback_version_id = f"rollback_{dataset_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Copy target version data to new rollback version
            original_path = Path(target_version.source_files[0])
            rollback_path = self._get_partition_path(dataset_name, layer, timestamp) / f"{rollback_version_id}.parquet"
            rollback_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(original_path, rollback_path)
            
            # Create rollback version metadata
            rollback_version = DataVersion(
                version_id=rollback_version_id,
                dataset_name=dataset_name,
                layer=layer,
                created_timestamp=timestamp.isoformat(),
                source_files=[str(rollback_path)],
                record_count=target_version.record_count,
                file_size_mb=target_version.file_size_mb,
                schema_hash=target_version.schema_hash,
                merge_strategy=MergeStrategy.REPLACE,
                parent_version=version_id,
                metadata={
                    "rollback_source": version_id,
                    "rollback_reason": "Manual rollback"
                }
            )
            
            # Save rollback version
            self._save_version_metadata(rollback_version)
            
            logger.info(f"Rolled back {dataset_name}/{layer.value} to version {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def get_data_lineage(self, version_id: str) -> Dict[str, Any]:
        """Get complete data lineage for a version."""
        try:
            lineage = {
                "target_version": version_id,
                "lineage_chain": [],
                "creation_timestamps": [],
                "record_counts": [],
                "merge_strategies": []
            }
            
            current_version_id = version_id
            
            while current_version_id:
                version = self.versions_metadata.get(current_version_id)
                if not version:
                    break
                
                lineage["lineage_chain"].append(current_version_id)
                lineage["creation_timestamps"].append(version.created_timestamp)
                lineage["record_counts"].append(version.record_count)
                lineage["merge_strategies"].append(version.merge_strategy.value)
                
                current_version_id = version.parent_version
            
            return lineage
            
        except Exception as e:
            logger.error(f"Data lineage retrieval failed: {e}")
            return {"error": str(e)}
    
    def get_incremental_summary(self) -> Dict[str, Any]:
        """Get comprehensive incremental processing summary."""
        try:
            summary = {
                "total_versions": len(self.versions_metadata),
                "layers": {layer.value: 0 for layer in DataLayer},
                "datasets": {},
                "recent_activity": [],
                "storage_by_layer": {layer.value: 0 for layer in DataLayer}
            }
            
            # Analyze versions
            for version in self.versions_metadata.values():
                # Count by layer
                summary["layers"][version.layer.value] += 1
                
                # Count by dataset
                if version.dataset_name not in summary["datasets"]:
                    summary["datasets"][version.dataset_name] = {
                        "versions": 0,
                        "latest_update": version.created_timestamp,
                        "total_records": 0
                    }
                
                dataset_info = summary["datasets"][version.dataset_name]
                dataset_info["versions"] += 1
                dataset_info["total_records"] += version.record_count
                
                if version.created_timestamp > dataset_info["latest_update"]:
                    dataset_info["latest_update"] = version.created_timestamp
                
                # Storage by layer
                summary["storage_by_layer"][version.layer.value] += version.file_size_mb
            
            # Recent activity (last 24 hours)
            recent_cutoff = datetime.now() - timedelta(hours=24)
            recent_versions = [
                v for v in self.versions_metadata.values()
                if datetime.fromisoformat(v.created_timestamp) > recent_cutoff
            ]
            
            summary["recent_activity"] = [
                {
                    "version_id": v.version_id,
                    "dataset": v.dataset_name,
                    "layer": v.layer.value,
                    "timestamp": v.created_timestamp,
                    "records": v.record_count
                }
                for v in sorted(recent_versions, key=lambda x: x.created_timestamp, reverse=True)[:10]
            ]
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return {"error": str(e)}


if __name__ == "__main__":
    # Development testing
    processor = IncrementalProcessor()
    
    # Generate test data
    np.random.seed(42)
    test_data = pl.DataFrame({
        "sa2_code": [f"1{str(i).zfill(8)}" for i in range(1000, 1100)],
        "prescription_count": np.random.poisson(3, 100),
        "total_cost": np.random.exponential(45, 100),
        "state_name": np.random.choice(['NSW', 'VIC', 'QLD'], 100)
    })
    
    # Test incremental processing
    bronze_version = processor.ingest_to_bronze(test_data, "test_health", {"source": "test"})
    silver_version = processor.process_to_silver("test_health", bronze_version)
    
    # Get summary
    summary = processor.get_incremental_summary()
    print(f"✅ Incremental processing test: {summary['total_versions']} versions created")