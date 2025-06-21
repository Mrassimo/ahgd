"""
Abstract base class for data transformers in the AHGD ETL pipeline.

This module provides the BaseTransformer class which defines the standard interface
for all data transformation components.
"""

import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import logging
import pandas as pd

from ..utils.interfaces import (
    AuditTrail,
    ColumnMapping,
    DataBatch,
    DataRecord,
    ProcessingMetadata,
    ProcessingStatus,
    ProgressCallback,
    TransformationError,
    ValidationResult,
    ValidationSeverity,
)


class MissingValueStrategy:
    """Strategies for handling missing values."""
    DROP = "drop"
    FILL_ZERO = "fill_zero"
    FILL_MEAN = "fill_mean"
    FILL_MEDIAN = "fill_median"
    FILL_MODE = "fill_mode"
    FILL_FORWARD = "fill_forward"
    FILL_BACKWARD = "fill_backward"
    FILL_INTERPOLATE = "fill_interpolate"
    FILL_CUSTOM = "fill_custom"


class BaseTransformer(ABC):
    """
    Abstract base class for data transformers.
    
    This class provides the standard interface and common functionality
    for all data transformation components in the AHGD ETL pipeline.
    """
    
    def __init__(
        self,
        transformer_id: str,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialise the transformer.
        
        Args:
            transformer_id: Unique identifier for this transformer
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.transformer_id = transformer_id
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Column mappings
        self.column_mappings: List[ColumnMapping] = []
        self._load_column_mappings()
        
        # Schema enforcement
        self.enforce_schema = config.get('enforce_schema', True)
        self.target_schema: Optional[Dict[str, str]] = config.get('target_schema')
        
        # Missing value handling
        self.missing_value_strategy = config.get(
            'missing_value_strategy', 
            MissingValueStrategy.DROP
        )
        self.missing_value_config = config.get('missing_value_config', {})
        
        # Audit trail
        self._processing_metadata: Optional[ProcessingMetadata] = None
        self._audit_trail: Optional[AuditTrail] = None
        self._transformation_log: List[Dict[str, Any]] = []
        
        # Progress tracking
        self._progress_callback: Optional[ProgressCallback] = None
    
    @abstractmethod
    def transform(
        self,
        data: DataBatch,
        **kwargs
    ) -> DataBatch:
        """
        Transform a batch of data records.
        
        Args:
            data: Batch of data records to transform
            **kwargs: Additional transformation parameters
            
        Returns:
            DataBatch: Transformed data records
            
        Raises:
            TransformationError: If transformation fails
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, str]:
        """
        Get the expected output schema.
        
        Returns:
            Dict[str, str]: Schema definition (column_name -> data_type)
        """
        pass
    
    def transform_with_audit(
        self,
        data: DataBatch,
        progress_callback: Optional[ProgressCallback] = None,
        **kwargs
    ) -> DataBatch:
        """
        Transform data with full audit trail generation.
        
        Args:
            data: Batch of data records to transform
            progress_callback: Optional progress callback
            **kwargs: Additional transformation parameters
            
        Returns:
            DataBatch: Transformed data records
        """
        self._progress_callback = progress_callback
        
        # Initialize processing metadata
        self._processing_metadata = ProcessingMetadata(
            operation_id=f"{self.transformer_id}_{int(time.time())}",
            operation_type="transformation",
            status=ProcessingStatus.RUNNING,
            start_time=datetime.now(),
            parameters=kwargs
        )
        
        try:
            # Report initial progress
            if self._progress_callback:
                self._progress_callback(0, len(data), "Starting transformation")
            
            # Perform transformation
            transformed_data = self.transform(data, **kwargs)
            
            # Validate output schema if required
            if self.enforce_schema:
                schema_violations = self._validate_output_schema(transformed_data)
                if schema_violations:
                    self.logger.warning(f"Schema violations found: {len(schema_violations)}")
                    for violation in schema_violations:
                        self._transformation_log.append({
                            'type': 'schema_violation',
                            'details': violation,
                            'timestamp': datetime.now()
                        })
            
            # Apply missing value handling
            transformed_data = self._handle_missing_values(transformed_data)
            
            # Update metadata
            self._processing_metadata.records_processed = len(data)
            self._processing_metadata.mark_completed()
            
            # Report completion
            if self._progress_callback:
                self._progress_callback(len(data), len(data), "Transformation completed")
            
            self.logger.info(
                f"Transformation completed: {len(data)} records processed"
            )
            
            return transformed_data
            
        except Exception as e:
            self._processing_metadata.mark_failed(str(e))
            self.logger.error(f"Transformation failed: {e}")
            raise TransformationError(f"Transformation failed: {e}") from e
    
    def apply_column_mappings(self, data: DataBatch) -> DataBatch:
        """
        Apply column mappings to transform column names and types.
        
        Args:
            data: Input data batch
            
        Returns:
            DataBatch: Data with column mappings applied
        """
        if not self.column_mappings:
            return data
        
        transformed_data = []
        
        for record in data:
            transformed_record = {}
            
            for mapping in self.column_mappings:
                source_value = record.get(mapping.source_column)
                
                # Handle missing required columns
                if source_value is None and mapping.is_required:
                    if mapping.default_value is not None:
                        source_value = mapping.default_value
                    else:
                        self.logger.warning(
                            f"Required column '{mapping.source_column}' is missing"
                        )
                        continue
                
                # Apply transformation if specified
                if mapping.transformation and source_value is not None:
                    try:
                        source_value = self._apply_transformation(
                            source_value, 
                            mapping.transformation
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Transformation failed for column '{mapping.source_column}': {e}"
                        )
                        continue
                
                # Convert data type
                if source_value is not None:
                    try:
                        source_value = self._convert_data_type(
                            source_value, 
                            mapping.data_type
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Data type conversion failed for column '{mapping.source_column}': {e}"
                        )
                        continue
                
                transformed_record[mapping.target_column] = source_value
            
            transformed_data.append(transformed_record)
        
        # Log transformation
        self._transformation_log.append({
            'type': 'column_mapping',
            'records_processed': len(data),
            'mappings_applied': len(self.column_mappings),
            'timestamp': datetime.now()
        })
        
        return transformed_data
    
    def standardise_columns(self, data: DataBatch) -> DataBatch:
        """
        Standardise column names and formats.
        
        Args:
            data: Input data batch
            
        Returns:
            DataBatch: Data with standardised columns
        """
        standardised_data = []
        
        for record in data:
            standardised_record = {}
            
            for key, value in record.items():
                # Standardise column name
                standardised_key = self._standardise_column_name(key)
                
                # Standardise value format
                standardised_value = self._standardise_value(value)
                
                standardised_record[standardised_key] = standardised_value
            
            standardised_data.append(standardised_record)
        
        return standardised_data
    
    def get_audit_trail(self) -> Optional[AuditTrail]:
        """
        Get the audit trail for the transformation.
        
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
            
            # Add transformation log to audit trail
            self._audit_trail.output_metadata = {
                'transformation_log': self._transformation_log,
                'column_mappings': [
                    {
                        'source': mapping.source_column,
                        'target': mapping.target_column,
                        'type': mapping.data_type
                    }
                    for mapping in self.column_mappings
                ]
            }
        
        return self._audit_trail
    
    def _load_column_mappings(self) -> None:
        """Load column mappings from configuration."""
        mappings_config = self.config.get('column_mappings', [])
        
        for mapping_config in mappings_config:
            mapping = ColumnMapping(
                source_column=mapping_config['source_column'],
                target_column=mapping_config['target_column'],
                data_type=mapping_config['data_type'],
                transformation=mapping_config.get('transformation'),
                validation_rules=mapping_config.get('validation_rules', []),
                is_required=mapping_config.get('is_required', True),
                default_value=mapping_config.get('default_value')
            )
            self.column_mappings.append(mapping)
    
    def _validate_output_schema(self, data: DataBatch) -> List[str]:
        """
        Validate that the output data conforms to the expected schema.
        
        Args:
            data: Transformed data batch
            
        Returns:
            List[str]: List of schema violations
        """
        if not self.target_schema or not data:
            return []
        
        violations = []
        sample_record = data[0]
        
        # Check for missing required columns
        for column, data_type in self.target_schema.items():
            if column not in sample_record:
                violations.append(f"Missing required column: {column}")
        
        # Check for unexpected columns
        for column in sample_record.keys():
            if column not in self.target_schema:
                violations.append(f"Unexpected column: {column}")
        
        return violations
    
    def _handle_missing_values(self, data: DataBatch) -> DataBatch:
        """
        Handle missing values according to the configured strategy.
        
        Args:
            data: Input data batch
            
        Returns:
            DataBatch: Data with missing values handled
        """
        if self.missing_value_strategy == MissingValueStrategy.DROP:
            return [record for record in data if all(v is not None for v in record.values())]
        
        # For other strategies, we'd typically work with pandas DataFrame
        df = pd.DataFrame(data)
        
        if self.missing_value_strategy == MissingValueStrategy.FILL_ZERO:
            df = df.fillna(0)
        elif self.missing_value_strategy == MissingValueStrategy.FILL_MEAN:
            df = df.fillna(df.mean())
        elif self.missing_value_strategy == MissingValueStrategy.FILL_MEDIAN:
            df = df.fillna(df.median())
        elif self.missing_value_strategy == MissingValueStrategy.FILL_MODE:
            df = df.fillna(df.mode().iloc[0])
        elif self.missing_value_strategy == MissingValueStrategy.FILL_FORWARD:
            df = df.fillna(method='ffill')
        elif self.missing_value_strategy == MissingValueStrategy.FILL_BACKWARD:
            df = df.fillna(method='bfill')
        elif self.missing_value_strategy == MissingValueStrategy.FILL_INTERPOLATE:
            df = df.interpolate()
        elif self.missing_value_strategy == MissingValueStrategy.FILL_CUSTOM:
            fill_values = self.missing_value_config.get('fill_values', {})
            df = df.fillna(value=fill_values)
        
        return df.to_dict('records')
    
    def _apply_transformation(self, value: Any, transformation: str) -> Any:
        """
        Apply a specific transformation to a value.
        
        Args:
            value: Input value
            transformation: Transformation to apply
            
        Returns:
            Any: Transformed value
        """
        # This is a simplified example - real implementations would be more comprehensive
        if transformation == "upper":
            return str(value).upper()
        elif transformation == "lower":
            return str(value).lower()
        elif transformation == "strip":
            return str(value).strip()
        elif transformation == "int":
            return int(value)
        elif transformation == "float":
            return float(value)
        else:
            self.logger.warning(f"Unknown transformation: {transformation}")
            return value
    
    def _convert_data_type(self, value: Any, target_type: str) -> Any:
        """
        Convert a value to the target data type.
        
        Args:
            value: Input value
            target_type: Target data type
            
        Returns:
            Any: Converted value
        """
        if target_type == "string":
            return str(value)
        elif target_type == "integer":
            return int(value)
        elif target_type == "float":
            return float(value)
        elif target_type == "boolean":
            return bool(value)
        else:
            return value
    
    def _standardise_column_name(self, column_name: str) -> str:
        """
        Standardise a column name.
        
        Args:
            column_name: Original column name
            
        Returns:
            str: Standardised column name
        """
        # Convert to lowercase, replace spaces with underscores
        return column_name.lower().replace(' ', '_').replace('-', '_')
    
    def _standardise_value(self, value: Any) -> Any:
        """
        Standardise a value format.
        
        Args:
            value: Original value
            
        Returns:
            Any: Standardised value
        """
        # Basic standardisation - trim whitespace for strings
        if isinstance(value, str):
            return value.strip()
        return value