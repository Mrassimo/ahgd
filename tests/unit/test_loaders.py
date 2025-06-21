"""
Unit tests for AHGD data loaders.

Tests loading mechanisms, export formats, output validation,
and error handling for all loader components.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
from datetime import datetime
from typing import Iterator, Dict, Any, List
import json
import sqlite3

from src.loaders.base import BaseLoader
from src.utils.interfaces import (
    DataBatch,
    DataRecord,
    LoadingError,
    DataFormat,
    ProcessingMetadata,
    ProcessingStatus,
    ValidationResult,
    ValidationSeverity,
)


class ConcreteLoader(BaseLoader):
    """Concrete loader implementation for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_calls = 0
        self._should_fail = False
        self._failure_count = 0
        self._loaded_data = []
    
    def load(self, data: DataBatch, destination: str, **kwargs) -> Dict[str, Any]:
        """Mock load implementation."""
        self._load_calls += 1
        
        if self._should_fail and self._failure_count < 2:
            self._failure_count += 1
            raise LoadingError(f"Simulated loading failure #{self._failure_count}")
        
        # Store loaded data for verification
        self._loaded_data.extend(data)
        
        # Simulate different output formats
        output_metadata = {
            "destination": destination,
            "format": self.output_format,
            "records_loaded": len(data),
            "timestamp": datetime.now().isoformat(),
            "loader_id": self.loader_id
        }
        
        # Format-specific metadata
        if self.output_format == "csv":
            output_metadata["file_size"] = len(data) * 100  # Estimated
            output_metadata["delimiter"] = ","
        elif self.output_format == "json":
            output_metadata["file_size"] = len(data) * 150  # Estimated
            output_metadata["encoding"] = "utf-8"
        elif self.output_format == "parquet":
            output_metadata["file_size"] = len(data) * 80   # More compressed
            output_metadata["compression"] = "snappy"
        elif self.output_format == "sqlite":
            output_metadata["table_name"] = kwargs.get("table_name", "data")
            output_metadata["database_size"] = len(data) * 120
        
        return output_metadata
    
    def get_output_metadata(self, destination: str) -> Dict[str, Any]:
        """Mock output metadata implementation."""
        return {
            "destination": destination,
            "format": self.output_format,
            "exists": True,
            "size": 1024,
            "created_at": datetime.now().isoformat(),
            "schema": {
                "columns": ["sa2_code", "value", "year"],
                "types": {"sa2_code": "string", "value": "float", "year": "integer"}
            }
        }
    
    def validate_output(self, destination: str) -> bool:
        """Mock output validation implementation."""
        if destination == "invalid_destination":
            return False
        return True
    
    def get_supported_formats(self) -> List[DataFormat]:
        """Mock supported formats implementation."""
        return [
            DataFormat.CSV,
            DataFormat.JSON,
            DataFormat.PARQUET,
            DataFormat.SQLITE
        ]
    
    def set_failure_mode(self, should_fail: bool):
        """Set the loader to fail for testing."""
        self._should_fail = should_fail
        self._failure_count = 0
    
    def get_loaded_data(self) -> List[DataRecord]:
        """Get the data that was loaded (for testing verification)."""
        return self._loaded_data.copy()


@pytest.mark.unit
class TestBaseLoader:
    """Test cases for BaseLoader."""
    
    def test_loader_initialisation(self, sample_config, mock_logger):
        """Test loader initialisation with configuration."""
        loader_id = "test_loader"
        config = sample_config["loaders"]["sqlite_loader"]
        
        loader = ConcreteLoader(loader_id, config, mock_logger)
        
        assert loader.loader_id == loader_id
        assert loader.config == config
        assert loader.logger == mock_logger
        assert loader.batch_size == 1000
        assert loader.output_format == "sqlite"
    
    def test_loader_default_configuration(self):
        """Test loader initialisation with default configuration."""
        loader = ConcreteLoader("test", {})
        
        assert loader.batch_size == 1000
        assert loader.output_format == "csv"
        assert loader.max_retries == 3
        assert loader.validate_on_load is True
    
    def test_successful_loading(self, sample_config, mock_logger, sample_data_batch):
        """Test successful data loading."""
        loader = ConcreteLoader("test", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        result = loader.load_with_validation(sample_data_batch, "test_destination")
        
        assert isinstance(result, dict)
        assert result["records_loaded"] == len(sample_data_batch)
        assert result["destination"] == "test_destination"
        assert result["format"] == "sqlite"
        
        # Verify data was actually loaded
        loaded_data = loader.get_loaded_data()
        assert len(loaded_data) == len(sample_data_batch)
        assert loaded_data == sample_data_batch
    
    def test_loading_with_validation_success(self, sample_config, mock_logger, sample_data_batch):
        """Test loading with successful output validation."""
        loader = ConcreteLoader("test", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        result = loader.load_with_validation(sample_data_batch, "valid_destination")
        
        assert result["records_loaded"] == len(sample_data_batch)
        
        # Verify processing metadata
        metadata = loader._processing_metadata
        assert metadata.status == ProcessingStatus.COMPLETED
        assert metadata.records_processed == len(sample_data_batch)
    
    def test_loading_with_validation_failure(self, sample_config, mock_logger, sample_data_batch):
        """Test loading with output validation failure."""
        loader = ConcreteLoader("test", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        with pytest.raises(LoadingError, match="Output validation failed"):
            loader.load_with_validation(sample_data_batch, "invalid_destination")
    
    def test_loading_retry_logic(self, sample_config, mock_logger, sample_data_batch):
        """Test retry logic on loading failure."""
        config = sample_config["loaders"]["sqlite_loader"].copy()
        config["max_retries"] = 2
        config["retry_delay"] = 0.1
        
        loader = ConcreteLoader("test", config, mock_logger)
        loader.set_failure_mode(True)  # Will fail first 2 attempts
        
        with patch('time.sleep') as mock_sleep:
            result = loader.load_with_validation(sample_data_batch, "test_destination")
        
        # Should succeed after 2 failures
        assert result["records_loaded"] == len(sample_data_batch)
        assert loader._load_calls == 3  # 2 failures + 1 success
        
        # Verify sleep was called for retries
        assert mock_sleep.call_count == 2
    
    def test_loading_max_retries_exceeded(self, sample_config, mock_logger, sample_data_batch):
        """Test loading failure after max retries exceeded."""
        config = sample_config["loaders"]["sqlite_loader"].copy()
        config["max_retries"] = 1
        config["retry_delay"] = 0.1
        
        loader = ConcreteLoader("test", config, mock_logger)
        
        # Make loader always fail
        loader.set_failure_mode(True)
        loader._failure_count = 0  # Reset to always fail
        
        with patch('time.sleep'):
            with pytest.raises(LoadingError, match="failed after 1 retries"):
                loader.load_with_validation(sample_data_batch, "test_destination")
    
    def test_progress_tracking(self, sample_config, mock_logger, mock_progress_callback, large_data_batch):
        """Test progress tracking during loading."""
        loader = ConcreteLoader("test", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        loader.load_with_validation(large_data_batch, "test_destination", progress_callback=mock_progress_callback)
        
        # Progress callback should be called
        assert mock_progress_callback.call_count >= 1
        
        # Verify callback parameters
        calls = mock_progress_callback.call_args_list
        for call in calls:
            assert len(call[0]) >= 2  # Should have current and total parameters
    
    def test_batch_processing(self, sample_config, mock_logger, large_data_batch):
        """Test batch processing functionality."""
        config = sample_config["loaders"]["sqlite_loader"].copy()
        config["batch_size"] = 100  # Process in smaller batches
        
        loader = ConcreteLoader("test", config, mock_logger)
        
        result = loader.load_with_validation(large_data_batch, "test_destination")
        
        assert result["records_loaded"] == len(large_data_batch)
        
        # Verify all data was loaded despite batching
        loaded_data = loader.get_loaded_data()
        assert len(loaded_data) == len(large_data_batch)
    
    def test_output_metadata_retrieval(self, sample_config, mock_logger):
        """Test output metadata retrieval."""
        loader = ConcreteLoader("test", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        metadata = loader.get_output_metadata("test_destination")
        
        assert isinstance(metadata, dict)
        assert "destination" in metadata
        assert "format" in metadata
        assert "schema" in metadata
        assert metadata["destination"] == "test_destination"
    
    def test_supported_formats_retrieval(self, sample_config, mock_logger):
        """Test supported formats retrieval."""
        loader = ConcreteLoader("test", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        formats = loader.get_supported_formats()
        
        assert isinstance(formats, list)
        assert DataFormat.CSV in formats
        assert DataFormat.JSON in formats
        assert DataFormat.PARQUET in formats
        assert DataFormat.SQLITE in formats
    
    def test_processing_metadata_tracking(self, sample_config, mock_logger, sample_data_batch):
        """Test processing metadata tracking."""
        loader = ConcreteLoader("test", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        loader.load_with_validation(sample_data_batch, "test_destination")
        
        metadata = loader._processing_metadata
        assert metadata is not None
        assert metadata.operation_type == "loading"
        assert metadata.status == ProcessingStatus.COMPLETED
        assert metadata.records_processed == len(sample_data_batch)
        assert metadata.start_time is not None
        assert metadata.end_time is not None
        assert metadata.duration_seconds is not None
    
    def test_loading_error_handling(self, sample_config, mock_logger, sample_data_batch):
        """Test loading error handling and metadata update."""
        config = sample_config["loaders"]["sqlite_loader"].copy()
        config["max_retries"] = 0  # No retries
        
        loader = ConcreteLoader("test", config, mock_logger)
        loader.set_failure_mode(True)
        
        with pytest.raises(LoadingError):
            loader.load_with_validation(sample_data_batch, "test_destination")
        
        # Check that processing metadata reflects failure
        metadata = loader._processing_metadata
        assert metadata.status == ProcessingStatus.FAILED
        assert metadata.error_message is not None
        assert "Simulated loading failure" in metadata.error_message
    
    def test_empty_data_handling(self, sample_config, mock_logger):
        """Test handling of empty data batches."""
        loader = ConcreteLoader("test", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        empty_data = []
        result = loader.load_with_validation(empty_data, "test_destination")
        
        assert result["records_loaded"] == 0
        assert loader._processing_metadata.records_processed == 0


@pytest.mark.unit
@pytest.mark.parametrize("output_format", ["csv", "json", "parquet", "sqlite"])
class TestLoaderOutputFormats:
    """Test loader handling of different output formats."""
    
    def test_output_format_configuration(self, sample_config, mock_logger, output_format, sample_data_batch):
        """Test configuration of different output formats."""
        config = sample_config["loaders"]["sqlite_loader"].copy()
        config["output_format"] = output_format
        
        loader = ConcreteLoader("test", config, mock_logger)
        
        assert loader.output_format == output_format
        
        # Test loading with specific format
        result = loader.load_with_validation(sample_data_batch, "test_destination")
        assert result["format"] == output_format
    
    def test_format_specific_metadata(self, sample_config, mock_logger, output_format, sample_data_batch):
        """Test format-specific metadata generation."""
        config = sample_config["loaders"]["sqlite_loader"].copy()
        config["output_format"] = output_format
        
        loader = ConcreteLoader("test", config, mock_logger)
        
        result = loader.load_with_validation(sample_data_batch, "test_destination")
        
        # Verify format-specific metadata
        if output_format == "csv":
            assert "delimiter" in result
        elif output_format == "json":
            assert "encoding" in result
        elif output_format == "parquet":
            assert "compression" in result
        elif output_format == "sqlite":
            assert "table_name" in result


@pytest.mark.unit
class TestLoaderValidation:
    """Test loader validation functionality."""
    
    def test_output_validation_success(self, sample_config, mock_logger):
        """Test successful output validation."""
        loader = ConcreteLoader("test", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        is_valid = loader.validate_output("valid_destination")
        assert is_valid is True
    
    def test_output_validation_failure(self, sample_config, mock_logger):
        """Test output validation failure."""
        loader = ConcreteLoader("test", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        is_valid = loader.validate_output("invalid_destination")
        assert is_valid is False
    
    def test_validation_disabled(self, sample_config, mock_logger, sample_data_batch):
        """Test loading with validation disabled."""
        config = sample_config["loaders"]["sqlite_loader"].copy()
        config["validate_on_load"] = False
        
        loader = ConcreteLoader("test", config, mock_logger)
        
        # Should succeed even with invalid destination when validation is disabled
        result = loader.load_with_validation(sample_data_batch, "invalid_destination")
        assert result["records_loaded"] == len(sample_data_batch)


@pytest.mark.unit
class TestLoaderFileOperations:
    """Test loader file operations."""
    
    def test_csv_loading_simulation(self, sample_config, mock_logger, sample_data_batch, temp_dir):
        """Test CSV loading simulation."""
        config = sample_config["loaders"]["sqlite_loader"].copy()
        config["output_format"] = "csv"
        
        loader = ConcreteLoader("test", config, mock_logger)
        
        csv_file = temp_dir / "test_output.csv"
        
        with patch('builtins.open', mock_open()) as mock_file:
            result = loader.load_with_validation(sample_data_batch, str(csv_file))
        
        assert result["format"] == "csv"
        assert result["records_loaded"] == len(sample_data_batch)
    
    def test_json_loading_simulation(self, sample_config, mock_logger, sample_data_batch, temp_dir):
        """Test JSON loading simulation."""
        config = sample_config["loaders"]["sqlite_loader"].copy()
        config["output_format"] = "json"
        
        loader = ConcreteLoader("test", config, mock_logger)
        
        json_file = temp_dir / "test_output.json"
        
        with patch('builtins.open', mock_open()) as mock_file:
            result = loader.load_with_validation(sample_data_batch, str(json_file))
        
        assert result["format"] == "json"
        assert result["records_loaded"] == len(sample_data_batch)
    
    def test_sqlite_loading_simulation(self, sample_config, mock_logger, sample_data_batch, sqlite_db):
        """Test SQLite loading simulation."""
        config = sample_config["loaders"]["sqlite_loader"].copy()
        config["output_format"] = "sqlite"
        
        loader = ConcreteLoader("test", config, mock_logger)
        
        # Mock SQLite operations
        with patch('sqlite3.connect') as mock_connect:
            mock_connect.return_value = sqlite_db
            
            result = loader.load_with_validation(
                sample_data_batch, 
                "test.db", 
                table_name="test_table"
            )
        
        assert result["format"] == "sqlite"
        assert result["records_loaded"] == len(sample_data_batch)
        assert result["table_name"] == "test_table"


@pytest.mark.unit
@pytest.mark.slow
class TestLoaderPerformance:
    """Performance-related tests for loaders."""
    
    def test_large_dataset_loading(self, sample_config, mock_logger, performance_data_large):
        """Test loading performance with large dataset."""
        loader = ConcreteLoader("test", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        start_time = datetime.now()
        
        result = loader.load_with_validation(performance_data_large, "large_destination")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        assert result["records_loaded"] == len(performance_data_large)
        assert duration < 15.0  # Should complete within 15 seconds
        
        # Verify all records were processed
        assert loader._processing_metadata.records_processed == len(performance_data_large)
    
    def test_memory_efficient_loading(self, sample_config, mock_logger, memory_intensive_data):
        """Test memory-efficient loading of large records."""
        config = sample_config["loaders"]["sqlite_loader"].copy()
        config["batch_size"] = 100  # Smaller batches for memory efficiency
        
        loader = ConcreteLoader("test", config, mock_logger)
        
        # Should handle large records without memory issues
        result = loader.load_with_validation(memory_intensive_data, "memory_test_destination")
        
        assert result["records_loaded"] == len(memory_intensive_data)
        
        # Verify all data was loaded
        loaded_data = loader.get_loaded_data()
        assert len(loaded_data) == len(memory_intensive_data)
    
    def test_concurrent_loading_safety(self, sample_config, mock_logger, sample_data_batch):
        """Test that loaders handle concurrent access safely."""
        import threading
        
        loader = ConcreteLoader("test", sample_config["loaders"]["sqlite_loader"], mock_logger)
        results = []
        errors = []
        
        def load_data(batch_id):
            try:
                # Create unique data for each thread
                thread_data = []
                for record in sample_data_batch:
                    thread_record = record.copy()
                    thread_record['batch_id'] = batch_id
                    thread_data.append(thread_record)
                
                result = loader.load_with_validation(thread_data, f"destination_{batch_id}")
                results.append((batch_id, result["records_loaded"]))
            except Exception as e:
                errors.append((batch_id, str(e)))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=load_data, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All loads should succeed
        assert len(errors) == 0
        assert len(results) == 5
        
        # Each load should process all records
        for batch_id, record_count in results:
            assert record_count == len(sample_data_batch)


@pytest.mark.unit
class TestLoaderEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_none_data_handling(self, sample_config, mock_logger):
        """Test handling of None data."""
        loader = ConcreteLoader("test", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        with pytest.raises(TypeError):
            loader.load_with_validation(None, "test_destination")
    
    def test_none_destination_handling(self, sample_config, mock_logger, sample_data_batch):
        """Test handling of None destination."""
        loader = ConcreteLoader("test", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        with pytest.raises(ValueError):
            loader.load_with_validation(sample_data_batch, None)
    
    def test_empty_destination_handling(self, sample_config, mock_logger, sample_data_batch):
        """Test handling of empty destination."""
        loader = ConcreteLoader("test", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        with pytest.raises(ValueError):
            loader.load_with_validation(sample_data_batch, "")
    
    def test_malformed_records_handling(self, sample_config, mock_logger):
        """Test handling of malformed records."""
        loader = ConcreteLoader("test", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        malformed_data = [
            None,  # None record
            {},    # Empty record
            {"nested": {"deep": {"value": 1}}},  # Deeply nested
            {"binary_data": b"binary"},  # Binary data
            {"datetime": datetime.now()}  # Complex objects
        ]
        
        # Should handle gracefully or raise appropriate error
        try:
            result = loader.load_with_validation(malformed_data, "malformed_destination")
            # If successful, verify structure
            assert result["records_loaded"] == len(malformed_data)
        except (LoadingError, TypeError, ValueError):
            # Expected for malformed data
            pass
    
    def test_invalid_output_format(self, sample_config, mock_logger, sample_data_batch):
        """Test handling of invalid output format."""
        config = sample_config["loaders"]["sqlite_loader"].copy()
        config["output_format"] = "invalid_format"
        
        loader = ConcreteLoader("test", config, mock_logger)
        
        # Should either handle gracefully or raise appropriate error
        try:
            result = loader.load_with_validation(sample_data_batch, "test_destination")
            # If successful, should have some default behavior
            assert result["records_loaded"] == len(sample_data_batch)
        except (LoadingError, ValueError):
            # Expected for invalid format
            pass
    
    def test_loader_with_no_logger(self, sample_config):
        """Test loader initialisation without logger."""
        loader = ConcreteLoader("test", sample_config["loaders"]["sqlite_loader"])
        
        # Should create default logger
        assert loader.logger is not None
        assert loader.logger.name == "ConcreteLoader"
    
    def test_loading_interruption_handling(self, sample_config, mock_logger, large_data_batch):
        """Test handling of loading interruption."""
        class InterruptibleLoader(ConcreteLoader):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._should_interrupt = False
            
            def load(self, data: DataBatch, destination: str, **kwargs) -> Dict[str, Any]:
                if self._should_interrupt:
                    raise KeyboardInterrupt("Loading interrupted")
                return super().load(data, destination, **kwargs)
            
            def interrupt(self):
                self._should_interrupt = True
        
        loader = InterruptibleLoader("test", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        # Start loading in thread and interrupt it
        import threading
        
        def load_with_interrupt():
            import time
            time.sleep(0.1)  # Let loading start
            loader.interrupt()
        
        interrupt_thread = threading.Thread(target=load_with_interrupt)
        interrupt_thread.start()
        
        with pytest.raises(KeyboardInterrupt):
            loader.load_with_validation(large_data_batch, "test_destination")
        
        interrupt_thread.join()
    
    def test_destination_path_validation(self, sample_config, mock_logger, sample_data_batch):
        """Test destination path validation."""
        loader = ConcreteLoader("test", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        # Test various destination formats
        valid_destinations = [
            "/tmp/test.db",
            "relative/path/test.csv",
            "C:\\Windows\\temp\\test.json",  # Windows path
            "s3://bucket/path/test.parquet",  # S3 path
            "memory://test_table"  # Memory destination
        ]
        
        for destination in valid_destinations:
            try:
                result = loader.load_with_validation(sample_data_batch, destination)
                assert result["destination"] == destination
            except (LoadingError, ValueError):
                # Some destinations might not be supported
                pass