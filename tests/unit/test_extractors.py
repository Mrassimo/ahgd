"""
Unit tests for AHGD data extractors.

Tests all extractor base classes and implementations including error handling,
retry logic, progress tracking, and data validation.
"""

import pytest
from unittest.mock import Mock, patch, call, MagicMock
from pathlib import Path
from datetime import datetime, timedelta
from typing import Iterator, List

from src.extractors.base import BaseExtractor
from src.utils.interfaces import (
    DataBatch,
    DataRecord,
    ExtractionError,
    ProcessingMetadata,
    ProcessingStatus,
    SourceMetadata,
    ValidationResult,
    ValidationSeverity,
)


class ConcreteExtractor(BaseExtractor):
    """Concrete extractor implementation for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._extract_calls = 0
        self._should_fail = False
        self._failure_count = 0
    
    def extract(self, source, **kwargs) -> Iterator[DataBatch]:
        """Mock extract implementation."""
        self._extract_calls += 1
        
        if self._should_fail and self._failure_count < 2:
            self._failure_count += 1
            raise ExtractionError(f"Simulated failure #{self._failure_count}")
        
        # Return sample data batches
        batch1 = [
            {"sa2_code": "101011001", "value": 25.5, "year": 2021},
            {"sa2_code": "101011002", "value": 30.2, "year": 2021}
        ]
        batch2 = [
            {"sa2_code": "101021003", "value": 22.8, "year": 2021},
            {"sa2_code": "102011004", "value": 28.1, "year": 2021}
        ]
        
        yield batch1
        yield batch2
    
    def get_source_metadata(self, source) -> SourceMetadata:
        """Mock source metadata implementation."""
        return SourceMetadata(
            source_id="test_source",
            source_type="csv",
            source_file=Path(str(source)) if isinstance(source, (str, Path)) else None,
            row_count=4,
            column_count=3,
            headers=["sa2_code", "value", "year"]
        )
    
    def validate_source(self, source) -> bool:
        """Mock source validation implementation."""
        if isinstance(source, str) and source == "invalid_source":
            return False
        return True
    
    def set_failure_mode(self, should_fail: bool):
        """Set the extractor to fail for testing retry logic."""
        self._should_fail = should_fail
        self._failure_count = 0


@pytest.mark.unit
class TestBaseExtractor:
    """Test cases for BaseExtractor."""
    
    def test_extractor_initialisation(self, sample_config, mock_logger):
        """Test extractor initialisation with configuration."""
        extractor_id = "test_extractor"
        config = sample_config["extractors"]["csv_extractor"]
        
        extractor = ConcreteExtractor(extractor_id, config, mock_logger)
        
        assert extractor.extractor_id == extractor_id
        assert extractor.config == config
        assert extractor.logger == mock_logger
        assert extractor.max_retries == 3
        assert extractor.retry_delay == 1.0
        assert extractor.batch_size == 100
    
    def test_extractor_default_configuration(self):
        """Test extractor initialisation with default configuration."""
        extractor = ConcreteExtractor("test", {})
        
        assert extractor.max_retries == 3
        assert extractor.retry_delay == 1.0
        assert extractor.retry_backoff == 2.0
        assert extractor.batch_size == 1000
        assert extractor._checkpoint_interval == 1000
    
    def test_successful_extraction(self, sample_config, mock_logger):
        """Test successful data extraction."""
        extractor = ConcreteExtractor("test", sample_config["extractors"]["csv_extractor"], mock_logger)
        
        batches = list(extractor.extract_with_retry("test_source"))
        
        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert batches[0][0]["sa2_code"] == "101011001"
        assert batches[1][0]["sa2_code"] == "101021003"
    
    def test_source_validation_failure(self, sample_config, mock_logger):
        """Test extraction failure due to invalid source."""
        extractor = ConcreteExtractor("test", sample_config["extractors"]["csv_extractor"], mock_logger)
        
        with pytest.raises(ExtractionError, match="Source validation failed"):
            list(extractor.extract_with_retry("invalid_source"))
    
    def test_extraction_retry_logic(self, sample_config, mock_logger):
        """Test retry logic on extraction failure."""
        config = sample_config["extractors"]["csv_extractor"].copy()
        config["max_retries"] = 2
        config["retry_delay"] = 0.1  # Fast retry for testing
        
        extractor = ConcreteExtractor("test", config, mock_logger)
        extractor.set_failure_mode(True)  # Will fail first 2 attempts
        
        with patch('time.sleep') as mock_sleep:
            batches = list(extractor.extract_with_retry("test_source"))
        
        # Should succeed after 2 failures
        assert len(batches) == 2
        assert extractor._extract_calls == 3  # 2 failures + 1 success
        
        # Verify sleep was called for retries
        assert mock_sleep.call_count == 2
    
    def test_extraction_max_retries_exceeded(self, sample_config, mock_logger):
        """Test extraction failure after max retries exceeded."""
        config = sample_config["extractors"]["csv_extractor"].copy()
        config["max_retries"] = 1
        config["retry_delay"] = 0.1
        
        extractor = ConcreteExtractor("test", config, mock_logger)
        
        # Make extractor always fail
        extractor.set_failure_mode(True)
        extractor._failure_count = 0  # Reset failure count to always fail
        
        with patch('time.sleep'):
            with pytest.raises(ExtractionError, match="failed after 1 retries"):
                list(extractor.extract_with_retry("test_source"))
    
    def test_progress_callback(self, sample_config, mock_logger, mock_progress_callback):
        """Test progress callback functionality."""
        extractor = ConcreteExtractor("test", sample_config["extractors"]["csv_extractor"], mock_logger)
        
        list(extractor.extract_with_retry("test_source", progress_callback=mock_progress_callback))
        
        # Should be called for each batch
        assert mock_progress_callback.call_count >= 2
        
        # Verify callback parameters
        calls = mock_progress_callback.call_args_list
        assert calls[0][0][0] == 2  # First batch has 2 records
        assert calls[1][0][0] == 4  # Total after second batch is 4 records
    
    def test_checkpoint_creation(self, sample_config, mock_logger):
        """Test checkpoint creation during extraction."""
        config = sample_config["extractors"]["csv_extractor"].copy()
        config["checkpoint_interval"] = 2  # Create checkpoint every 2 records
        
        extractor = ConcreteExtractor("test", config, mock_logger)
        
        list(extractor.extract_with_retry("test_source"))
        
        # Should have created checkpoints
        assert extractor._last_checkpoint is not None
        assert "records_processed" in extractor._last_checkpoint
        assert extractor._last_checkpoint["records_processed"] == 4
    
    def test_processing_metadata_tracking(self, sample_config, mock_logger):
        """Test processing metadata tracking."""
        extractor = ConcreteExtractor("test", sample_config["extractors"]["csv_extractor"], mock_logger)
        
        list(extractor.extract_with_retry("test_source"))
        
        metadata = extractor._processing_metadata
        assert metadata is not None
        assert metadata.operation_type == "extraction"
        assert metadata.status == ProcessingStatus.COMPLETED
        assert metadata.records_processed == 4
        assert metadata.start_time is not None
        assert metadata.end_time is not None
        assert metadata.duration_seconds is not None
    
    def test_source_metadata_retrieval(self, sample_config, mock_logger):
        """Test source metadata retrieval."""
        extractor = ConcreteExtractor("test", sample_config["extractors"]["csv_extractor"], mock_logger)
        
        list(extractor.extract_with_retry("test_source"))
        
        metadata = extractor._source_metadata
        assert metadata is not None
        assert metadata.source_id == "test_source"
        assert metadata.source_type == "csv"
        assert metadata.row_count == 4
        assert metadata.column_count == 3
    
    def test_audit_trail_generation(self, sample_config, mock_logger):
        """Test audit trail generation."""
        extractor = ConcreteExtractor("test", sample_config["extractors"]["csv_extractor"], mock_logger)
        
        list(extractor.extract_with_retry("test_source"))
        
        audit_trail = extractor.get_audit_trail()
        assert audit_trail is not None
        assert audit_trail.operation_type == "extraction"
        assert audit_trail.source_metadata is not None
        assert audit_trail.processing_metadata is not None
    
    def test_checksum_calculation(self, sample_config, mock_logger, temp_dir):
        """Test file checksum calculation."""
        extractor = ConcreteExtractor("test", sample_config["extractors"]["csv_extractor"], mock_logger)
        
        # Create a test file
        test_file = temp_dir / "test_checksum.txt"
        test_file.write_text("test content for checksum")
        
        checksum = extractor.get_checksum(test_file)
        
        assert checksum is not None
        assert len(checksum) == 64  # SHA256 hex digest length
        assert isinstance(checksum, str)
    
    def test_resume_extraction_default(self, sample_config, mock_logger):
        """Test default resume extraction implementation."""
        extractor = ConcreteExtractor("test", sample_config["extractors"]["csv_extractor"], mock_logger)
        
        checkpoint = {
            "extractor_id": "test",
            "records_processed": 100,
            "timestamp": datetime.now().isoformat()
        }
        
        with patch.object(extractor, 'extract') as mock_extract:
            mock_extract.return_value = iter([])
            
            list(extractor.resume_extraction("test_source", checkpoint))
            
            # Default implementation should call extract
            mock_extract.assert_called_once_with("test_source")
    
    def test_extraction_with_empty_source(self, sample_config, mock_logger):
        """Test extraction handling of empty source."""
        class EmptyExtractor(BaseExtractor):
            def extract(self, source, **kwargs):
                return iter([])  # Empty iterator
            
            def get_source_metadata(self, source):
                return SourceMetadata(
                    source_id="empty_source",
                    source_type="csv",
                    row_count=0
                )
            
            def validate_source(self, source):
                return True
        
        extractor = EmptyExtractor("test", sample_config["extractors"]["csv_extractor"], mock_logger)
        
        batches = list(extractor.extract_with_retry("empty_source"))
        
        assert batches == []
        assert extractor._processing_metadata.records_processed == 0
    
    def test_extraction_error_handling(self, sample_config, mock_logger):
        """Test extraction error handling and metadata update."""
        config = sample_config["extractors"]["csv_extractor"].copy()
        config["max_retries"] = 0  # No retries
        
        extractor = ConcreteExtractor("test", config, mock_logger)
        extractor.set_failure_mode(True)
        
        with pytest.raises(ExtractionError):
            list(extractor.extract_with_retry("test_source"))
        
        # Check that processing metadata reflects failure
        metadata = extractor._processing_metadata
        assert metadata.status == ProcessingStatus.FAILED
        assert metadata.error_message is not None
        assert "Simulated failure" in metadata.error_message
    
    def test_batch_size_configuration(self, sample_config, mock_logger):
        """Test batch size configuration affects processing."""
        config = sample_config["extractors"]["csv_extractor"].copy()
        config["batch_size"] = 50
        
        extractor = ConcreteExtractor("test", config, mock_logger)
        
        assert extractor.batch_size == 50
    
    def test_retry_backoff_calculation(self, sample_config, mock_logger):
        """Test retry delay backoff calculation."""
        config = sample_config["extractors"]["csv_extractor"].copy()
        config["retry_delay"] = 1.0
        config["retry_backoff"] = 2.0
        config["max_retries"] = 3
        
        extractor = ConcreteExtractor("test", config, mock_logger)
        extractor.set_failure_mode(True)
        
        with patch('time.sleep') as mock_sleep:
            with pytest.raises(ExtractionError):
                list(extractor.extract_with_retry("test_source"))
        
        # Verify exponential backoff: 1.0, 2.0, 4.0
        expected_delays = [1.0, 2.0, 4.0]
        actual_delays = [call[0][0] for call in mock_sleep.call_args_list]
        
        assert actual_delays == expected_delays


@pytest.mark.unit
@pytest.mark.parametrize("source_type", ["csv", "json", "excel", "database"])
class TestExtractorSourceHandling:
    """Test extractor handling of different source types."""
    
    def test_source_type_handling(self, sample_config, mock_logger, source_type):
        """Test handling of different source types."""
        extractor = ConcreteExtractor("test", sample_config["extractors"]["csv_extractor"], mock_logger)
        
        # Mock the source metadata to return different source types
        with patch.object(extractor, 'get_source_metadata') as mock_metadata:
            mock_metadata.return_value = SourceMetadata(
                source_id="test",
                source_type=source_type,
                row_count=100
            )
            
            batches = list(extractor.extract_with_retry(f"test_source.{source_type}"))
            
            assert len(batches) == 2
            mock_metadata.assert_called_once()


@pytest.mark.unit
@pytest.mark.slow
class TestExtractorPerformance:
    """Performance-related tests for extractors."""
    
    def test_large_dataset_extraction(self, sample_config, mock_logger, performance_data_medium):
        """Test extraction performance with medium dataset."""
        class LargeDataExtractor(BaseExtractor):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._data = performance_data_medium
            
            def extract(self, source, **kwargs):
                # Yield data in chunks based on batch_size
                for i in range(0, len(self._data), self.batch_size):
                    yield self._data[i:i + self.batch_size]
            
            def get_source_metadata(self, source):
                return SourceMetadata(
                    source_id="large_source",
                    source_type="csv",
                    row_count=len(self._data)
                )
            
            def validate_source(self, source):
                return True
        
        extractor = LargeDataExtractor("test", sample_config["extractors"]["csv_extractor"], mock_logger)
        
        start_time = datetime.now()
        total_records = 0
        
        for batch in extractor.extract_with_retry("large_source"):
            total_records += len(batch)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        assert total_records == len(performance_data_medium)
        assert duration < 5.0  # Should complete within 5 seconds
        
        # Verify processing metadata
        assert extractor._processing_metadata.records_processed == total_records
    
    def test_concurrent_extraction_safety(self, sample_config, mock_logger):
        """Test that extractors handle concurrent access safely."""
        import threading
        
        extractor = ConcreteExtractor("test", sample_config["extractors"]["csv_extractor"], mock_logger)
        results = []
        errors = []
        
        def extract_data(source_id):
            try:
                batches = list(extractor.extract_with_retry(f"source_{source_id}"))
                results.append((source_id, len(batches)))
            except Exception as e:
                errors.append((source_id, str(e)))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=extract_data, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All extractions should succeed
        assert len(errors) == 0
        assert len(results) == 5
        
        # Each extraction should return 2 batches
        for source_id, batch_count in results:
            assert batch_count == 2


@pytest.mark.unit
class TestExtractorEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_none_source_handling(self, sample_config, mock_logger):
        """Test handling of None source."""
        extractor = ConcreteExtractor("test", sample_config["extractors"]["csv_extractor"], mock_logger)
        
        with pytest.raises(Exception):  # Should raise some exception
            list(extractor.extract_with_retry(None))
    
    def test_empty_config_handling(self, mock_logger):
        """Test handling of empty configuration."""
        extractor = ConcreteExtractor("test", {}, mock_logger)
        
        # Should use default values
        assert extractor.max_retries == 3
        assert extractor.batch_size == 1000
    
    def test_invalid_config_values(self, mock_logger):
        """Test handling of invalid configuration values."""
        config = {
            "max_retries": -1,  # Invalid
            "retry_delay": "invalid",  # Invalid type
            "batch_size": 0  # Invalid
        }
        
        # Should handle gracefully or raise appropriate error
        try:
            extractor = ConcreteExtractor("test", config, mock_logger)
            # If no exception, verify fallback to defaults
            assert extractor.max_retries >= 0
            assert isinstance(extractor.retry_delay, (int, float))
            assert extractor.batch_size > 0
        except (ValueError, TypeError):
            # Expected for invalid configuration
            pass
    
    def test_extractor_with_no_logger(self, sample_config):
        """Test extractor initialisation without logger."""
        extractor = ConcreteExtractor("test", sample_config["extractors"]["csv_extractor"])
        
        # Should create default logger
        assert extractor.logger is not None
        assert extractor.logger.name == "ConcreteExtractor"
    
    def test_extraction_interruption_handling(self, sample_config, mock_logger):
        """Test handling of extraction interruption."""
        class InterruptibleExtractor(BaseExtractor):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._should_interrupt = False
            
            def extract(self, source, **kwargs):
                for i in range(5):
                    if self._should_interrupt:
                        raise KeyboardInterrupt("Extraction interrupted")
                    yield [{"record": i}]
            
            def get_source_metadata(self, source):
                return SourceMetadata(source_id="test", source_type="csv")
            
            def validate_source(self, source):
                return True
            
            def interrupt(self):
                self._should_interrupt = True
        
        extractor = InterruptibleExtractor("test", sample_config["extractors"]["csv_extractor"], mock_logger)
        
        # Start extraction in thread and interrupt it
        import threading
        
        def extract_with_interrupt():
            import time
            time.sleep(0.1)  # Let extraction start
            extractor.interrupt()
        
        interrupt_thread = threading.Thread(target=extract_with_interrupt)
        interrupt_thread.start()
        
        with pytest.raises(KeyboardInterrupt):
            list(extractor.extract_with_retry("test_source"))
        
        interrupt_thread.join()