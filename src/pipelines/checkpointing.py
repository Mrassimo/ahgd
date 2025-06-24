"""
Checkpoint management system for pipeline state persistence.

This module provides comprehensive checkpointing capabilities including
state serialisation, versioning, and recovery strategies.
"""

import gzip
import hashlib
import json
import pickle
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union, Tuple
import threading
from collections import OrderedDict

from ..utils.logging import get_logger
from ..utils.interfaces import AHGDException

logger = get_logger(__name__)


class CheckpointFormat(Enum):
    """Supported checkpoint formats."""
    PICKLE = "pickle"
    JSON = "json"
    PARQUET = "parquet"
    COMPRESSED_PICKLE = "pickle.gz"


class RecoveryStrategy(Enum):
    """Checkpoint recovery strategies."""
    LATEST = "latest"  # Use most recent checkpoint
    STABLE = "stable"  # Use last stable checkpoint
    VERSIONED = "versioned"  # Use specific version
    INCREMENTAL = "incremental"  # Apply incremental updates


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    checkpoint_id: str
    pipeline_name: str
    stage_name: str
    version: str
    created_at: datetime
    size_bytes: int
    checksum: str
    format: CheckpointFormat
    is_stable: bool = True
    tags: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class CheckpointValidationResult:
    """Result of checkpoint validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Optional[CheckpointMetadata] = None


class CheckpointSerializer(ABC):
    """Abstract base class for checkpoint serializers."""
    
    @abstractmethod
    def serialize(self, data: Any, path: Path) -> None:
        """Serialise data to file."""
        pass
    
    @abstractmethod
    def deserialize(self, path: Path) -> Any:
        """Deserialise data from file."""
        pass
    
    @abstractmethod
    def get_format(self) -> CheckpointFormat:
        """Get serializer format."""
        pass


class PickleSerializer(CheckpointSerializer):
    """Pickle-based serializer."""
    
    def serialize(self, data: Any, path: Path) -> None:
        """Serialise using pickle."""
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def deserialize(self, path: Path) -> Any:
        """Deserialise from pickle."""
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def get_format(self) -> CheckpointFormat:
        """Get format type."""
        return CheckpointFormat.PICKLE


class CompressedPickleSerializer(CheckpointSerializer):
    """Compressed pickle serializer."""
    
    def serialize(self, data: Any, path: Path) -> None:
        """Serialise using compressed pickle."""
        with gzip.open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def deserialize(self, path: Path) -> Any:
        """Deserialise from compressed pickle."""
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    
    def get_format(self) -> CheckpointFormat:
        """Get format type."""
        return CheckpointFormat.COMPRESSED_PICKLE


class JsonSerializer(CheckpointSerializer):
    """JSON-based serializer."""
    
    def serialize(self, data: Any, path: Path) -> None:
        """Serialise using JSON."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def deserialize(self, path: Path) -> Any:
        """Deserialise from JSON."""
        with open(path, "r") as f:
            return json.load(f)
    
    def get_format(self) -> CheckpointFormat:
        """Get format type."""
        return CheckpointFormat.JSON


class ParquetSerializer(CheckpointSerializer):
    """Parquet-based serializer for dataframes."""
    
    def serialize(self, data: Any, path: Path) -> None:
        """Serialise using Parquet."""
        import pandas as pd
        
        if isinstance(data, pd.DataFrame):
            data.to_parquet(path, engine="pyarrow", compression="snappy")
        else:
            raise TypeError("ParquetSerializer only supports pandas DataFrames")
    
    def deserialize(self, path: Path) -> Any:
        """Deserialise from Parquet."""
        import pandas as pd
        return pd.read_parquet(path, engine="pyarrow")
    
    def get_format(self) -> CheckpointFormat:
        """Get format type."""
        return CheckpointFormat.PARQUET


class CheckpointManager:
    """
    Manages checkpoint creation, storage, and recovery.
    
    Features:
    - Multiple serialisation formats
    - Checkpoint versioning
    - Automatic cleanup
    - Validation and integrity checks
    - Recovery strategies
    """
    
    # Serializer registry
    SERIALIZERS: Dict[CheckpointFormat, Type[CheckpointSerializer]] = {
        CheckpointFormat.PICKLE: PickleSerializer,
        CheckpointFormat.COMPRESSED_PICKLE: CompressedPickleSerializer,
        CheckpointFormat.JSON: JsonSerializer,
        CheckpointFormat.PARQUET: ParquetSerializer,
    }
    
    def __init__(
        self,
        checkpoint_dir: Path,
        default_format: CheckpointFormat = CheckpointFormat.COMPRESSED_PICKLE,
        max_checkpoints: int = 10,
        retention_days: int = 7,
        enable_versioning: bool = True
    ):
        """
        Initialise checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for storing checkpoints
            default_format: Default serialisation format
            max_checkpoints: Maximum checkpoints to retain per stage
            retention_days: Days to retain checkpoints
            enable_versioning: Whether to enable checkpoint versioning
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.default_format = default_format
        self.max_checkpoints = max_checkpoints
        self.retention_days = retention_days
        self.enable_versioning = enable_versioning
        
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.metadata_cache: Dict[str, CheckpointMetadata] = {}
        self._lock = threading.RLock()
        
        # Load existing metadata
        self._load_metadata()
        
        logger.info(
            "Checkpoint manager initialised",
            checkpoint_dir=str(checkpoint_dir),
            default_format=default_format.value
        )
    
    def create_checkpoint(
        self,
        pipeline_name: str,
        stage_name: str,
        data: Any,
        version: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        format: Optional[CheckpointFormat] = None,
        is_stable: bool = True
    ) -> CheckpointMetadata:
        """
        Create a new checkpoint.
        
        Args:
            pipeline_name: Pipeline name
            stage_name: Stage name
            data: Data to checkpoint
            version: Optional version string
            tags: Optional metadata tags
            format: Serialisation format (defaults to manager default)
            is_stable: Whether checkpoint is stable
            
        Returns:
            Checkpoint metadata
        """
        with self._lock:
            try:
                # Generate checkpoint ID
                checkpoint_id = self._generate_checkpoint_id(
                    pipeline_name,
                    stage_name,
                    version
                )
                
                # Determine format and serializer
                checkpoint_format = format or self.default_format
                serializer = self._get_serializer(checkpoint_format)
                
                # Determine file path
                file_ext = self._get_file_extension(checkpoint_format)
                checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.{file_ext}"
                
                # Serialise data
                logger.debug(
                    "Creating checkpoint",
                    checkpoint_id=checkpoint_id,
                    format=checkpoint_format.value
                )
                
                serializer.serialize(data, checkpoint_path)
                
                # Calculate checksum
                checksum = self._calculate_checksum(checkpoint_path)
                
                # Create metadata
                metadata = CheckpointMetadata(
                    checkpoint_id=checkpoint_id,
                    pipeline_name=pipeline_name,
                    stage_name=stage_name,
                    version=version or self._get_next_version(pipeline_name, stage_name),
                    created_at=datetime.now(),
                    size_bytes=checkpoint_path.stat().st_size,
                    checksum=checksum,
                    format=checkpoint_format,
                    is_stable=is_stable,
                    tags=tags or {}
                )
                
                # Store metadata
                self._save_metadata(metadata)
                
                # Cleanup old checkpoints
                self._cleanup_old_checkpoints(pipeline_name, stage_name)
                
                logger.info(
                    "Checkpoint created",
                    checkpoint_id=checkpoint_id,
                    size_mb=metadata.size_bytes / 1024 / 1024
                )
                
                return metadata
                
            except Exception as e:
                logger.error(
                    "Failed to create checkpoint",
                    pipeline=pipeline_name,
                    stage=stage_name,
                    error=str(e)
                )
                raise CheckpointError(f"Checkpoint creation failed: {str(e)}") from e
    
    def load_checkpoint(
        self,
        pipeline_name: str,
        stage_name: str,
        strategy: RecoveryStrategy = RecoveryStrategy.LATEST,
        version: Optional[str] = None,
        validate: bool = True
    ) -> Tuple[Any, CheckpointMetadata]:
        """
        Load a checkpoint using specified strategy.
        
        Args:
            pipeline_name: Pipeline name
            stage_name: Stage name
            strategy: Recovery strategy
            version: Specific version (for VERSIONED strategy)
            validate: Whether to validate checkpoint
            
        Returns:
            Tuple of (data, metadata)
        """
        with self._lock:
            try:
                # Find checkpoint based on strategy
                metadata = self._find_checkpoint(
                    pipeline_name,
                    stage_name,
                    strategy,
                    version
                )
                
                if not metadata:
                    raise CheckpointError(
                        f"No checkpoint found for {pipeline_name}/{stage_name}"
                    )
                
                # Validate if requested
                if validate:
                    validation_result = self.validate_checkpoint(metadata.checkpoint_id)
                    if not validation_result.is_valid:
                        raise CheckpointError(
                            f"Checkpoint validation failed: {validation_result.errors}"
                        )
                
                # Load checkpoint
                checkpoint_path = self._get_checkpoint_path(metadata)
                serializer = self._get_serializer(metadata.format)
                
                logger.debug(
                    "Loading checkpoint",
                    checkpoint_id=metadata.checkpoint_id,
                    format=metadata.format.value
                )
                
                data = serializer.deserialize(checkpoint_path)
                
                logger.info(
                    "Checkpoint loaded",
                    checkpoint_id=metadata.checkpoint_id,
                    age_hours=(datetime.now() - metadata.created_at).total_seconds() / 3600
                )
                
                return data, metadata
                
            except Exception as e:
                logger.error(
                    "Failed to load checkpoint",
                    pipeline=pipeline_name,
                    stage=stage_name,
                    error=str(e)
                )
                raise CheckpointError(f"Checkpoint loading failed: {str(e)}") from e
    
    def list_checkpoints(
        self,
        pipeline_name: Optional[str] = None,
        stage_name: Optional[str] = None,
        stable_only: bool = False
    ) -> List[CheckpointMetadata]:
        """
        List available checkpoints.
        
        Args:
            pipeline_name: Filter by pipeline
            stage_name: Filter by stage
            stable_only: Only return stable checkpoints
            
        Returns:
            List of checkpoint metadata
        """
        with self._lock:
            checkpoints = list(self.metadata_cache.values())
            
            # Apply filters
            if pipeline_name:
                checkpoints = [
                    c for c in checkpoints
                    if c.pipeline_name == pipeline_name
                ]
            
            if stage_name:
                checkpoints = [
                    c for c in checkpoints
                    if c.stage_name == stage_name
                ]
            
            if stable_only:
                checkpoints = [
                    c for c in checkpoints
                    if c.is_stable
                ]
            
            # Sort by creation time (newest first)
            checkpoints.sort(key=lambda c: c.created_at, reverse=True)
            
            return checkpoints
    
    def validate_checkpoint(self, checkpoint_id: str) -> CheckpointValidationResult:
        """
        Validate a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint to validate
            
        Returns:
            Validation result
        """
        result = CheckpointValidationResult(is_valid=True)
        
        try:
            # Get metadata
            metadata = self.metadata_cache.get(checkpoint_id)
            if not metadata:
                result.is_valid = False
                result.errors.append(f"Checkpoint not found: {checkpoint_id}")
                return result
            
            result.metadata = metadata
            
            # Check file exists
            checkpoint_path = self._get_checkpoint_path(metadata)
            if not checkpoint_path.exists():
                result.is_valid = False
                result.errors.append("Checkpoint file missing")
                return result
            
            # Verify checksum
            current_checksum = self._calculate_checksum(checkpoint_path)
            if current_checksum != metadata.checksum:
                result.is_valid = False
                result.errors.append("Checksum mismatch - checkpoint corrupted")
                return result
            
            # Check size
            current_size = checkpoint_path.stat().st_size
            if current_size != metadata.size_bytes:
                result.warnings.append(
                    f"Size mismatch: expected {metadata.size_bytes}, got {current_size}"
                )
            
            # Check age
            age_days = (datetime.now() - metadata.created_at).days
            if age_days > self.retention_days:
                result.warnings.append(
                    f"Checkpoint is {age_days} days old (retention: {self.retention_days} days)"
                )
            
            logger.debug(
                "Checkpoint validated",
                checkpoint_id=checkpoint_id,
                is_valid=result.is_valid
            )
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Validation error: {str(e)}")
        
        return result
    
    def delete_checkpoint(self, checkpoint_id: str) -> None:
        """Delete a specific checkpoint."""
        with self._lock:
            metadata = self.metadata_cache.get(checkpoint_id)
            if not metadata:
                logger.warning(
                    "Checkpoint not found for deletion",
                    checkpoint_id=checkpoint_id
                )
                return
            
            # Delete file
            checkpoint_path = self._get_checkpoint_path(metadata)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            
            # Remove from metadata
            del self.metadata_cache[checkpoint_id]
            self._persist_metadata()
            
            logger.info(
                "Checkpoint deleted",
                checkpoint_id=checkpoint_id
            )
    
    def cleanup(
        self,
        older_than_days: Optional[int] = None,
        keep_stable: bool = True
    ) -> int:
        """
        Clean up old checkpoints.
        
        Args:
            older_than_days: Delete checkpoints older than this
            keep_stable: Whether to keep stable checkpoints
            
        Returns:
            Number of checkpoints deleted
        """
        with self._lock:
            cutoff_days = older_than_days or self.retention_days
            cutoff_date = datetime.now() - timedelta(days=cutoff_days)
            
            to_delete = []
            
            for checkpoint_id, metadata in self.metadata_cache.items():
                # Skip stable checkpoints if requested
                if keep_stable and metadata.is_stable:
                    continue
                
                # Check age
                if metadata.created_at < cutoff_date:
                    to_delete.append(checkpoint_id)
            
            # Delete checkpoints
            for checkpoint_id in to_delete:
                self.delete_checkpoint(checkpoint_id)
            
            logger.info(
                "Checkpoint cleanup completed",
                deleted=len(to_delete),
                cutoff_days=cutoff_days
            )
            
            return len(to_delete)
    
    def create_incremental_checkpoint(
        self,
        pipeline_name: str,
        stage_name: str,
        delta_data: Any,
        base_version: str,
        tags: Optional[Dict[str, Any]] = None
    ) -> CheckpointMetadata:
        """
        Create an incremental checkpoint.
        
        Args:
            pipeline_name: Pipeline name
            stage_name: Stage name
            delta_data: Incremental changes
            base_version: Base checkpoint version
            tags: Optional metadata tags
            
        Returns:
            Checkpoint metadata
        """
        # Find base checkpoint
        base_metadata = self._find_checkpoint(
            pipeline_name,
            stage_name,
            RecoveryStrategy.VERSIONED,
            base_version
        )
        
        if not base_metadata:
            raise CheckpointError(f"Base checkpoint not found: {base_version}")
        
        # Create incremental checkpoint
        version = f"{base_version}_delta_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        tags = tags or {}
        tags["base_version"] = base_version
        tags["checkpoint_type"] = "incremental"
        
        metadata = self.create_checkpoint(
            pipeline_name,
            stage_name,
            delta_data,
            version=version,
            tags=tags,
            is_stable=False
        )
        
        # Update dependencies
        metadata.dependencies = [base_metadata.checkpoint_id]
        self._save_metadata(metadata)
        
        return metadata
    
    def apply_incremental_checkpoints(
        self,
        pipeline_name: str,
        stage_name: str,
        base_version: str
    ) -> Any:
        """
        Load and apply incremental checkpoints.
        
        Args:
            pipeline_name: Pipeline name
            stage_name: Stage name
            base_version: Base version to start from
            
        Returns:
            Merged data
        """
        # Load base checkpoint
        base_data, base_metadata = self.load_checkpoint(
            pipeline_name,
            stage_name,
            RecoveryStrategy.VERSIONED,
            base_version
        )
        
        # Find incremental checkpoints
        incremental_checkpoints = [
            m for m in self.list_checkpoints(pipeline_name, stage_name)
            if m.tags.get("checkpoint_type") == "incremental"
            and m.tags.get("base_version") == base_version
        ]
        
        # Sort by creation time
        incremental_checkpoints.sort(key=lambda m: m.created_at)
        
        # Apply incremental updates
        result = base_data
        for metadata in incremental_checkpoints:
            delta_data, _ = self.load_checkpoint(
                pipeline_name,
                stage_name,
                RecoveryStrategy.VERSIONED,
                metadata.version
            )
            
            # Merge logic (override in subclasses for specific merge strategies)
            if isinstance(result, dict) and isinstance(delta_data, dict):
                result.update(delta_data)
            else:
                logger.warning(
                    "Cannot merge non-dict incremental checkpoint",
                    checkpoint_id=metadata.checkpoint_id
                )
        
        return result
    
    def _generate_checkpoint_id(
        self,
        pipeline_name: str,
        stage_name: str,
        version: Optional[str] = None
    ) -> str:
        """Generate unique checkpoint ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        if version and self.enable_versioning:
            return f"{pipeline_name}_{stage_name}_{version}_{timestamp}"
        else:
            return f"{pipeline_name}_{stage_name}_{timestamp}"
    
    def _get_next_version(self, pipeline_name: str, stage_name: str) -> str:
        """Get next version number for a stage."""
        if not self.enable_versioning:
            return "latest"
        
        # Find existing versions
        existing = self.list_checkpoints(pipeline_name, stage_name)
        
        if not existing:
            return "v1.0.0"
        
        # Extract version numbers
        versions = []
        for metadata in existing:
            if metadata.version.startswith("v"):
                try:
                    parts = metadata.version[1:].split(".")
                    versions.append(tuple(int(p) for p in parts))
                except (ValueError, IndexError):
                    continue
        
        if not versions:
            return "v1.0.0"
        
        # Increment patch version
        latest = max(versions)
        return f"v{latest[0]}.{latest[1]}.{latest[2] + 1}"
    
    def _get_serializer(self, format: CheckpointFormat) -> CheckpointSerializer:
        """Get serializer for format."""
        serializer_class = self.SERIALIZERS.get(format)
        if not serializer_class:
            raise ValueError(f"Unsupported checkpoint format: {format}")
        return serializer_class()
    
    def _get_file_extension(self, format: CheckpointFormat) -> str:
        """Get file extension for format."""
        extensions = {
            CheckpointFormat.PICKLE: "pkl",
            CheckpointFormat.COMPRESSED_PICKLE: "pkl.gz",
            CheckpointFormat.JSON: "json",
            CheckpointFormat.PARQUET: "parquet"
        }
        return extensions.get(format, "checkpoint")
    
    def _get_checkpoint_path(self, metadata: CheckpointMetadata) -> Path:
        """Get checkpoint file path."""
        file_ext = self._get_file_extension(metadata.format)
        return self.checkpoint_dir / f"{metadata.checkpoint_id}.{file_ext}"
    
    def _calculate_checksum(self, path: Path) -> str:
        """Calculate file checksum."""
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _find_checkpoint(
        self,
        pipeline_name: str,
        stage_name: str,
        strategy: RecoveryStrategy,
        version: Optional[str] = None
    ) -> Optional[CheckpointMetadata]:
        """Find checkpoint based on recovery strategy."""
        checkpoints = self.list_checkpoints(pipeline_name, stage_name)
        
        if not checkpoints:
            return None
        
        if strategy == RecoveryStrategy.LATEST:
            return checkpoints[0]  # Already sorted by creation time
        
        elif strategy == RecoveryStrategy.STABLE:
            stable_checkpoints = [c for c in checkpoints if c.is_stable]
            return stable_checkpoints[0] if stable_checkpoints else None
        
        elif strategy == RecoveryStrategy.VERSIONED:
            if not version:
                raise ValueError("Version required for VERSIONED strategy")
            
            for checkpoint in checkpoints:
                if checkpoint.version == version:
                    return checkpoint
            return None
        
        else:
            raise ValueError(f"Unknown recovery strategy: {strategy}")
    
    def _cleanup_old_checkpoints(self, pipeline_name: str, stage_name: str) -> None:
        """Clean up old checkpoints for a stage."""
        checkpoints = self.list_checkpoints(pipeline_name, stage_name)
        
        # Keep only max_checkpoints
        if len(checkpoints) > self.max_checkpoints:
            # Sort by creation time (oldest first)
            checkpoints.sort(key=lambda c: c.created_at)
            
            # Keep stable checkpoints and most recent ones
            stable_checkpoints = [c for c in checkpoints if c.is_stable]
            unstable_checkpoints = [c for c in checkpoints if not c.is_stable]
            
            # Determine how many to delete
            to_keep = set()
            
            # Always keep stable checkpoints
            for checkpoint in stable_checkpoints:
                to_keep.add(checkpoint.checkpoint_id)
            
            # Keep most recent unstable checkpoints
            remaining_slots = self.max_checkpoints - len(stable_checkpoints)
            if remaining_slots > 0:
                for checkpoint in unstable_checkpoints[-remaining_slots:]:
                    to_keep.add(checkpoint.checkpoint_id)
            
            # Delete others
            for checkpoint in checkpoints:
                if checkpoint.checkpoint_id not in to_keep:
                    self.delete_checkpoint(checkpoint.checkpoint_id)
    
    def _load_metadata(self) -> None:
        """Load metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    data = json.load(f)
                
                for checkpoint_data in data.get("checkpoints", []):
                    metadata = CheckpointMetadata(
                        checkpoint_id=checkpoint_data["checkpoint_id"],
                        pipeline_name=checkpoint_data["pipeline_name"],
                        stage_name=checkpoint_data["stage_name"],
                        version=checkpoint_data["version"],
                        created_at=datetime.fromisoformat(checkpoint_data["created_at"]),
                        size_bytes=checkpoint_data["size_bytes"],
                        checksum=checkpoint_data["checksum"],
                        format=CheckpointFormat(checkpoint_data["format"]),
                        is_stable=checkpoint_data.get("is_stable", True),
                        tags=checkpoint_data.get("tags", {}),
                        dependencies=checkpoint_data.get("dependencies", [])
                    )
                    self.metadata_cache[metadata.checkpoint_id] = metadata
                
                logger.debug(
                    "Metadata loaded",
                    checkpoints=len(self.metadata_cache)
                )
                
            except Exception as e:
                logger.error(
                    "Failed to load metadata",
                    error=str(e)
                )
    
    def _save_metadata(self, metadata: CheckpointMetadata) -> None:
        """Save metadata to cache and file."""
        self.metadata_cache[metadata.checkpoint_id] = metadata
        self._persist_metadata()
    
    def _persist_metadata(self) -> None:
        """Persist metadata cache to file."""
        try:
            data = {
                "checkpoints": [
                    {
                        "checkpoint_id": m.checkpoint_id,
                        "pipeline_name": m.pipeline_name,
                        "stage_name": m.stage_name,
                        "version": m.version,
                        "created_at": m.created_at.isoformat(),
                        "size_bytes": m.size_bytes,
                        "checksum": m.checksum,
                        "format": m.format.value,
                        "is_stable": m.is_stable,
                        "tags": m.tags,
                        "dependencies": m.dependencies
                    }
                    for m in self.metadata_cache.values()
                ]
            }
            
            with open(self.metadata_file, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(
                "Failed to persist metadata",
                error=str(e)
            )