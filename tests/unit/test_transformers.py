"""
Unit tests for AHGD data transformers.

Tests transformation logic, schema enforcement, data mapping,
and error handling for all transformer components.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime
from typing import Iterator, Dict, Any, List

from src.transformers.base import BaseTransformer
from src.utils.interfaces import (
    DataBatch,
    DataRecord,
    TransformationError,
    ColumnMapping,
    ProcessingMetadata,
    ProcessingStatus,
    ValidationResult,
    ValidationSeverity,
)


class ConcreteTransformer(BaseTransformer):
    """Concrete transformer implementation for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._transform_calls = 0
        self._should_fail = False
        self._failure_count = 0
    
    def transform(self, data: DataBatch, **kwargs) -> Iterator[DataBatch]:
        """Mock transform implementation."""
        self._transform_calls += 1
        
        if self._should_fail and self._failure_count < 2:
            self._failure_count += 1
            raise TransformationError(f"Simulated transformation failure #{self._failure_count}")
        
        # Apply basic transformations
        transformed_batch = []
        for record in data:
            transformed_record = record.copy()
            
            # Example transformations
            if 'sa2_code' in transformed_record:
                # Standardise SA2 code format
                transformed_record['sa2_code'] = str(transformed_record['sa2_code']).zfill(9)
            
            if 'value' in transformed_record:
                # Round numeric values
                if isinstance(transformed_record['value'], (int, float)):
                    transformed_record['value'] = round(float(transformed_record['value']), 2)
            
            if 'year' in transformed_record:
                # Ensure year is integer
                transformed_record['year'] = int(transformed_record['year'])
            
            # Add transformation metadata
            transformed_record['_transformed_at'] = datetime.now().isoformat()
            transformed_record['_transformer_id'] = self.transformer_id
            
            transformed_batch.append(transformed_record)
        
        yield transformed_batch
    
    def get_schema(self) -> Dict[str, Any]:
        """Mock schema retrieval implementation."""
        return {
            "type": "object",
            "properties": {
                "sa2_code": {"type": "string", "pattern": "^[0-9]{9}$"},
                "value": {"type": "number", "minimum": 0},
                "year": {"type": "integer", "minimum": 2000, "maximum": 2030},
                "_transformed_at": {"type": "string", "format": "date-time"},
                "_transformer_id": {"type": "string"}
            },
            "required": ["sa2_code", "value", "year"]
        }
    
    def validate_schema(self, data: DataBatch) -> bool:
        """Mock schema validation implementation."""
        schema = self.get_schema()
        required_fields = schema.get("required", [])
        
        for record in data:
            # Check required fields
            for field in required_fields:
                if field not in record or record[field] is None:
                    return False
            
            # Basic type checking
            if 'sa2_code' in record and not isinstance(record['sa2_code'], str):
                return False
            if 'value' in record and not isinstance(record['value'], (int, float)):
                return False
            if 'year' in record and not isinstance(record['year'], int):
                return False
        
        return True
    
    def get_column_mappings(self) -> List[ColumnMapping]:
        """Mock column mappings implementation."""
        return [
            ColumnMapping(
                source_column="sa2_code",
                target_column="statistical_area_2",
                data_type="string",
                transformation="standardize_sa2_code",
                is_required=True
            ),
            ColumnMapping(
                source_column="value",
                target_column="indicator_value",
                data_type="float",
                transformation="round_2dp",
                is_required=True
            ),
            ColumnMapping(
                source_column="year",
                target_column="reference_year",
                data_type="integer",
                is_required=True
            )
        ]
    
    def set_failure_mode(self, should_fail: bool):
        """Set the transformer to fail for testing error handling."""
        self._should_fail = should_fail
        self._failure_count = 0


@pytest.mark.unit
class TestBaseTransformer:
    """Test cases for BaseTransformer."""
    
    def test_transformer_initialisation(self, sample_config, mock_logger):
        """Test transformer initialisation with configuration."""
        transformer_id = "test_transformer"
        config = sample_config["transformers"]["sa2_transformer"]
        
        transformer = ConcreteTransformer(transformer_id, config, mock_logger)
        
        assert transformer.transformer_id == transformer_id
        assert transformer.config == config
        assert transformer.logger == mock_logger
        assert transformer.batch_size == 200
        assert transformer.output_format == "parquet"
    
    def test_transformer_default_configuration(self):
        """Test transformer initialisation with default configuration."""
        transformer = ConcreteTransformer("test", {})
        
        assert transformer.batch_size == 1000
        assert transformer.output_format == "csv"
        assert transformer.max_retries == 3
    
    def test_successful_transformation(self, sample_config, mock_logger, sample_data_batch):
        """Test successful data transformation."""
        transformer = ConcreteTransformer("test", sample_config["transformers"]["sa2_transformer"], mock_logger)
        
        transformed_batches = list(transformer.transform_with_validation(sample_data_batch))
        
        assert len(transformed_batches) == 1
        transformed_batch = transformed_batches[0]
        assert len(transformed_batch) == len(sample_data_batch)
        
        # Verify transformations applied
        for record in transformed_batch:
            assert len(record['sa2_code']) == 9  # Standardised format
            assert isinstance(record['value'], float)
            assert isinstance(record['year'], int)
            assert '_transformed_at' in record
            assert '_transformer_id' in record
    
    def test_transformation_with_validation(self, sample_config, mock_logger, sample_data_batch):
        """Test transformation with schema validation."""
        transformer = ConcreteTransformer("test", sample_config["transformers"]["sa2_transformer"], mock_logger)
        
        transformed_batches = list(transformer.transform_with_validation(sample_data_batch))
        
        # All records should pass validation
        assert len(transformed_batches) == 1
        
        # Verify schema validation was applied
        for record in transformed_batches[0]:
            assert transformer.validate_schema([record])
    
    def test_transformation_schema_validation_failure(self, sample_config, mock_logger):
        """Test transformation with schema validation failure."""
        transformer = ConcreteTransformer("test", sample_config["transformers"]["sa2_transformer"], mock_logger)
        
        # Create invalid data
        invalid_data = [
            {"sa2_code": None, "value": "invalid", "year": "not_a_year"},  # Multiple validation failures
            {"incomplete": "record"}  # Missing required fields
        ]
        
        with pytest.raises(TransformationError, match="Schema validation failed"):
            list(transformer.transform_with_validation(invalid_data))
    
    def test_transformation_retry_logic(self, sample_config, mock_logger, sample_data_batch):
        """Test retry logic on transformation failure."""
        config = sample_config["transformers"]["sa2_transformer"].copy()
        config["max_retries"] = 2
        config["retry_delay"] = 0.1
        
        transformer = ConcreteTransformer("test", config, mock_logger)
        transformer.set_failure_mode(True)  # Will fail first 2 attempts
        
        with patch('time.sleep') as mock_sleep:
            transformed_batches = list(transformer.transform_with_validation(sample_data_batch))
        
        # Should succeed after 2 failures
        assert len(transformed_batches) == 1
        assert transformer._transform_calls == 3  # 2 failures + 1 success
        
        # Verify sleep was called for retries
        assert mock_sleep.call_count == 2
    
    def test_column_mapping_application(self, sample_config, mock_logger, sample_data_batch):
        """Test column mapping application."""
        transformer = ConcreteTransformer("test", sample_config["transformers"]["sa2_transformer"], mock_logger)
        
        mappings = transformer.get_column_mappings()
        mapped_data = transformer.apply_column_mappings(sample_data_batch, mappings)
        
        # Verify column mappings were applied
        for record in mapped_data:
            assert 'statistical_area_2' in record
            assert 'indicator_value' in record
            assert 'reference_year' in record
            
            # Original columns should still be present (unless configured otherwise)
            assert 'sa2_code' in record
            assert 'value' in record
            assert 'year' in record
    
    def test_data_type_enforcement(self, sample_config, mock_logger):
        """Test data type enforcement during transformation."""
        transformer = ConcreteTransformer("test", sample_config["transformers"]["sa2_transformer"], mock_logger)
        
        # Create data with mixed types
        mixed_type_data = [
            {"sa2_code": 101011001, "value": "25.5", "year": "2021"},  # Numbers as strings/integers
            {"sa2_code": "101011002", "value": 30, "year": 2021.0}     # Mixed types
        ]
        
        transformed_batches = list(transformer.transform_with_validation(mixed_type_data))
        transformed_batch = transformed_batches[0]
        
        # Verify type enforcement
        for record in transformed_batch:
            assert isinstance(record['sa2_code'], str)
            assert isinstance(record['value'], float)
            assert isinstance(record['year'], int)
    
    def test_progress_tracking(self, sample_config, mock_logger, mock_progress_callback, large_data_batch):
        """Test progress tracking during transformation."""
        transformer = ConcreteTransformer("test", sample_config["transformers"]["sa2_transformer"], mock_logger)
        
        list(transformer.transform_with_validation(large_data_batch, progress_callback=mock_progress_callback))
        
        # Progress callback should be called
        assert mock_progress_callback.call_count >= 1
        
        # Verify callback parameters
        calls = mock_progress_callback.call_args_list
        for call in calls:
            assert len(call[0]) >= 2  # Should have current and total parameters
    
    def test_processing_metadata_tracking(self, sample_config, mock_logger, sample_data_batch):
        """Test processing metadata tracking."""
        transformer = ConcreteTransformer("test", sample_config["transformers"]["sa2_transformer"], mock_logger)
        
        list(transformer.transform_with_validation(sample_data_batch))
        
        metadata = transformer._processing_metadata
        assert metadata is not None
        assert metadata.operation_type == "transformation"
        assert metadata.status == ProcessingStatus.COMPLETED
        assert metadata.records_processed == len(sample_data_batch)
        assert metadata.start_time is not None
        assert metadata.end_time is not None
    
    def test_transformation_error_handling(self, sample_config, mock_logger, sample_data_batch):
        """Test transformation error handling and metadata update."""
        config = sample_config["transformers"]["sa2_transformer"].copy()
        config["max_retries"] = 0  # No retries
        
        transformer = ConcreteTransformer("test", config, mock_logger)
        transformer.set_failure_mode(True)
        
        with pytest.raises(TransformationError):
            list(transformer.transform_with_validation(sample_data_batch))
        
        # Check that processing metadata reflects failure
        metadata = transformer._processing_metadata
        assert metadata.status == ProcessingStatus.FAILED
        assert metadata.error_message is not None
        assert "Simulated transformation failure" in metadata.error_message
    
    def test_schema_retrieval(self, sample_config, mock_logger):
        """Test schema retrieval functionality."""
        transformer = ConcreteTransformer("test", sample_config["transformers"]["sa2_transformer"], mock_logger)
        
        schema = transformer.get_schema()
        
        assert isinstance(schema, dict)
        assert "type" in schema
        assert "properties" in schema
        assert "required" in schema
        
        # Verify expected properties
        properties = schema["properties"]
        assert "sa2_code" in properties
        assert "value" in properties
        assert "year" in properties
    
    def test_empty_data_handling(self, sample_config, mock_logger):
        """Test handling of empty data batches."""
        transformer = ConcreteTransformer("test", sample_config["transformers"]["sa2_transformer"], mock_logger)
        
        empty_data = []
        transformed_batches = list(transformer.transform_with_validation(empty_data))
        
        assert len(transformed_batches) == 1
        assert len(transformed_batches[0]) == 0
        assert transformer._processing_metadata.records_processed == 0


@pytest.mark.unit
@pytest.mark.parametrize("output_format", ["csv", "json", "parquet", "xlsx"])
class TestTransformerOutputFormats:
    """Test transformer handling of different output formats."""
    
    def test_output_format_configuration(self, sample_config, mock_logger, output_format):
        """Test configuration of different output formats."""
        config = sample_config["transformers"]["sa2_transformer"].copy()
        config["output_format"] = output_format
        
        transformer = ConcreteTransformer("test", config, mock_logger)
        
        assert transformer.output_format == output_format


@pytest.mark.unit
class TestTransformerValidation:
    """Test transformer validation functionality."""
    
    def test_schema_validation_success(self, sample_config, mock_logger):
        """Test successful schema validation."""
        transformer = ConcreteTransformer("test", sample_config["transformers"]["sa2_transformer"], mock_logger)
        
        valid_data = [
            {"sa2_code": "101011001", "value": 25.5, "year": 2021},
            {"sa2_code": "101011002", "value": 30.2, "year": 2021}
        ]
        
        is_valid = transformer.validate_schema(valid_data)
        assert is_valid is True
    
    def test_schema_validation_failure_missing_required(self, sample_config, mock_logger):
        """Test schema validation failure for missing required fields."""
        transformer = ConcreteTransformer("test", sample_config["transformers"]["sa2_transformer"], mock_logger)
        
        invalid_data = [
            {"sa2_code": "101011001", "value": 25.5},  # Missing year
            {"value": 30.2, "year": 2021}              # Missing sa2_code
        ]
        
        is_valid = transformer.validate_schema(invalid_data)
        assert is_valid is False
    
    def test_schema_validation_failure_wrong_types(self, sample_config, mock_logger):
        """Test schema validation failure for wrong data types."""
        transformer = ConcreteTransformer("test", sample_config["transformers"]["sa2_transformer"], mock_logger)
        
        invalid_data = [
            {"sa2_code": 101011001, "value": 25.5, "year": 2021},    # sa2_code should be string
            {"sa2_code": "101011002", "value": "invalid", "year": 2021}  # value should be number
        ]
        
        is_valid = transformer.validate_schema(invalid_data)
        assert is_valid is False
    
    def test_custom_validation_rules(self, sample_config, mock_logger):
        """Test custom validation rules implementation."""
        class CustomValidationTransformer(ConcreteTransformer):
            def validate_schema(self, data: DataBatch) -> bool:
                """Custom validation with business rules."""
                base_valid = super().validate_schema(data)
                
                if not base_valid:
                    return False
                
                # Custom business rule: SA2 codes must start with '10'
                for record in data:
                    if not record.get('sa2_code', '').startswith('10'):
                        return False
                
                return True
        
        transformer = CustomValidationTransformer("test", sample_config["transformers"]["sa2_transformer"], mock_logger)
        
        # Valid data (SA2 codes start with '10')
        valid_data = [{"sa2_code": "101011001", "value": 25.5, "year": 2021}]
        assert transformer.validate_schema(valid_data) is True
        
        # Invalid data (SA2 code starts with '20')
        invalid_data = [{"sa2_code": "201011001", "value": 25.5, "year": 2021}]
        assert transformer.validate_schema(invalid_data) is False


@pytest.mark.unit
class TestTransformerColumnMapping:
    """Test transformer column mapping functionality."""
    
    def test_get_column_mappings(self, sample_config, mock_logger):
        """Test retrieval of column mappings."""
        transformer = ConcreteTransformer("test", sample_config["transformers"]["sa2_transformer"], mock_logger)
        
        mappings = transformer.get_column_mappings()
        
        assert isinstance(mappings, list)
        assert len(mappings) == 3
        
        # Verify mapping structure
        for mapping in mappings:
            assert isinstance(mapping, ColumnMapping)
            assert mapping.source_column is not None
            assert mapping.target_column is not None
            assert mapping.data_type is not None
    
    def test_apply_column_mappings(self, sample_config, mock_logger, sample_data_batch):
        """Test application of column mappings."""
        transformer = ConcreteTransformer("test", sample_config["transformers"]["sa2_transformer"], mock_logger)
        
        mappings = transformer.get_column_mappings()
        mapped_data = transformer.apply_column_mappings(sample_data_batch, mappings)
        
        # Verify mappings were applied
        for record in mapped_data:
            assert 'statistical_area_2' in record
            assert 'indicator_value' in record
            assert 'reference_year' in record
    
    def test_column_mapping_with_defaults(self, sample_config, mock_logger):
        """Test column mapping with default values."""
        transformer = ConcreteTransformer("test", sample_config["transformers"]["sa2_transformer"], mock_logger)
        
        # Create mapping with default value
        mappings = [
            ColumnMapping(
                source_column="optional_field",
                target_column="optional_target",
                data_type="string",
                is_required=False,
                default_value="default_value"
            )
        ]
        
        # Data without the optional field
        data = [{"sa2_code": "101011001", "value": 25.5}]
        
        mapped_data = transformer.apply_column_mappings(data, mappings)
        
        # Should have default value
        assert mapped_data[0]['optional_target'] == "default_value"
    
    def test_column_mapping_transformations(self, sample_config, mock_logger):
        """Test column mapping with transformations."""
        class TransformationTransformer(ConcreteTransformer):
            def apply_transformation(self, value: Any, transformation: str) -> Any:
                """Apply specific transformation to value."""
                if transformation == "uppercase":
                    return str(value).upper()
                elif transformation == "round_2dp":
                    return round(float(value), 2)
                elif transformation == "add_prefix":
                    return f"PREFIX_{value}"
                return value
        
        transformer = TransformationTransformer("test", sample_config["transformers"]["sa2_transformer"], mock_logger)
        
        # Create mappings with transformations
        mappings = [
            ColumnMapping(
                source_column="text_field",
                target_column="text_upper",
                data_type="string",
                transformation="uppercase"
            ),
            ColumnMapping(
                source_column="number_field",
                target_column="number_rounded",
                data_type="float",
                transformation="round_2dp"
            )
        ]
        
        data = [{"text_field": "hello", "number_field": 3.14159}]
        mapped_data = transformer.apply_column_mappings(data, mappings)
        
        assert mapped_data[0]['text_upper'] == "HELLO"
        assert mapped_data[0]['number_rounded'] == 3.14


@pytest.mark.unit
@pytest.mark.slow
class TestTransformerPerformance:
    """Performance-related tests for transformers."""
    
    def test_large_dataset_transformation(self, sample_config, mock_logger, performance_data_medium):
        """Test transformation performance with medium dataset."""
        transformer = ConcreteTransformer("test", sample_config["transformers"]["sa2_transformer"], mock_logger)
        
        start_time = datetime.now()
        
        transformed_batches = list(transformer.transform_with_validation(performance_data_medium))
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        assert len(transformed_batches) == 1
        assert len(transformed_batches[0]) == len(performance_data_medium)
        assert duration < 10.0  # Should complete within 10 seconds
        
        # Verify all records were processed
        assert transformer._processing_metadata.records_processed == len(performance_data_medium)
    
    def test_memory_efficient_transformation(self, sample_config, mock_logger, memory_intensive_data):
        """Test memory-efficient transformation of large records."""
        transformer = ConcreteTransformer("test", sample_config["transformers"]["sa2_transformer"], mock_logger)
        
        # Transform should handle large records without memory issues
        transformed_batches = list(transformer.transform_with_validation(memory_intensive_data))
        
        assert len(transformed_batches) == 1
        assert len(transformed_batches[0]) == len(memory_intensive_data)
        
        # Verify transformation metadata was added
        for record in transformed_batches[0]:
            assert '_transformed_at' in record
            assert '_transformer_id' in record


@pytest.mark.unit
class TestTransformerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_none_data_handling(self, sample_config, mock_logger):
        """Test handling of None data."""
        transformer = ConcreteTransformer("test", sample_config["transformers"]["sa2_transformer"], mock_logger)
        
        with pytest.raises(TypeError):
            list(transformer.transform_with_validation(None))
    
    def test_malformed_records_handling(self, sample_config, mock_logger):
        """Test handling of malformed records."""
        transformer = ConcreteTransformer("test", sample_config["transformers"]["sa2_transformer"], mock_logger)
        
        malformed_data = [
            None,  # None record
            {},    # Empty record
            {"nested": {"deep": {"value": 1}}},  # Deeply nested
            {"binary_data": b"binary"},  # Binary data
            {"datetime": datetime.now()}  # Complex objects
        ]
        
        # Should handle gracefully or raise appropriate error
        try:
            transformed_batches = list(transformer.transform_with_validation(malformed_data))
            # If successful, verify structure
            assert len(transformed_batches) == 1
        except (TransformationError, TypeError, ValueError):
            # Expected for malformed data
            pass
    
    def test_concurrent_transformation_safety(self, sample_config, mock_logger, sample_data_batch):
        """Test that transformers handle concurrent access safely."""
        import threading
        
        transformer = ConcreteTransformer("test", sample_config["transformers"]["sa2_transformer"], mock_logger)
        results = []
        errors = []
        
        def transform_data(batch_id):
            try:
                # Create unique data for each thread
                thread_data = []
                for record in sample_data_batch:
                    thread_record = record.copy()
                    thread_record['batch_id'] = batch_id
                    thread_data.append(thread_record)
                
                transformed_batches = list(transformer.transform_with_validation(thread_data))
                results.append((batch_id, len(transformed_batches[0])))
            except Exception as e:
                errors.append((batch_id, str(e)))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=transform_data, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All transformations should succeed
        assert len(errors) == 0
        assert len(results) == 5
        
        # Each transformation should process all records
        for batch_id, record_count in results:
            assert record_count == len(sample_data_batch)
    
    def test_transformer_with_no_logger(self, sample_config):
        """Test transformer initialisation without logger."""
        transformer = ConcreteTransformer("test", sample_config["transformers"]["sa2_transformer"])
        
        # Should create default logger
        assert transformer.logger is not None
        assert transformer.logger.name == "ConcreteTransformer"