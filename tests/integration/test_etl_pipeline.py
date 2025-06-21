"""
Integration tests for AHGD ETL pipeline.

Tests end-to-end ETL pipeline functionality including extraction,
transformation, validation, and loading with real data flows.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime
import tempfile
import sqlite3
import json
import csv

from src.extractors.base import BaseExtractor
from src.transformers.base import BaseTransformer
from src.validators.base import BaseValidator
from src.loaders.base import BaseLoader
from src.utils.interfaces import (
    DataBatch,
    DataRecord,
    ProcessingStatus,
    ValidationSeverity,
    ValidationResult,
    ExtractionError,
    TransformationError,
    ValidationError,
    LoadingError
)


class TestExtractor(BaseExtractor):
    """Test extractor for integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._test_data = []
    
    def set_test_data(self, data: DataBatch):
        """Set test data to be extracted."""
        self._test_data = data
    
    def extract(self, source, **kwargs):
        """Extract test data."""
        # Simulate chunked extraction
        chunk_size = self.batch_size
        for i in range(0, len(self._test_data), chunk_size):
            yield self._test_data[i:i + chunk_size]
    
    def get_source_metadata(self, source):
        """Get test source metadata."""
        from src.utils.interfaces import SourceMetadata
        return SourceMetadata(
            source_id="test_source",
            source_type="test",
            row_count=len(self._test_data),
            column_count=len(self._test_data[0]) if self._test_data else 0
        )
    
    def validate_source(self, source):
        """Validate test source."""
        return True


class TestTransformer(BaseTransformer):
    """Test transformer for integration testing."""
    
    def transform(self, data: DataBatch, **kwargs):
        """Transform test data."""
        transformed_batch = []
        for record in data:
            transformed_record = record.copy()
            
            # Apply test transformations
            if 'sa2_code' in transformed_record:
                # Standardise SA2 code
                transformed_record['sa2_code'] = str(transformed_record['sa2_code']).zfill(9)
            
            if 'value' in transformed_record:
                # Round numeric values
                transformed_record['value'] = round(float(transformed_record['value']), 2)
            
            # Add transformation metadata
            transformed_record['_transformed'] = True
            transformed_record['_transformer'] = self.transformer_id
            
            transformed_batch.append(transformed_record)
        
        yield transformed_batch
    
    def get_schema(self):
        """Get test schema."""
        return {
            "type": "object",
            "properties": {
                "sa2_code": {"type": "string"},
                "value": {"type": "number"},
                "_transformed": {"type": "boolean"}
            }
        }
    
    def validate_schema(self, data: DataBatch):
        """Validate test schema."""
        for record in data:
            if 'sa2_code' not in record or 'value' not in record:
                return False
        return True


class TestValidator(BaseValidator):
    """Test validator for integration testing."""
    
    def validate(self, data: DataBatch):
        """Validate test data."""
        results = []
        
        for record_idx, record in enumerate(data):
            # Check for test validation rules
            if record.get('value', 0) < 0:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="negative_value_check",
                    message=f"Negative value found: {record['value']}",
                    affected_records=[record_idx]
                ))
            
            if len(record.get('sa2_code', '')) != 9:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="sa2_code_format",
                    message=f"Invalid SA2 code format: {record.get('sa2_code')}",
                    affected_records=[record_idx]
                ))
        
        return results
    
    def get_validation_rules(self):
        """Get test validation rules."""
        return ["negative_value_check", "sa2_code_format"]


class TestLoader(BaseLoader):
    """Test loader for integration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loaded_data = []
    
    def load(self, data: DataBatch, destination: str, **kwargs):
        """Load test data."""
        self._loaded_data.extend(data)
        
        return {
            "destination": destination,
            "records_loaded": len(data),
            "format": self.output_format,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_supported_formats(self):
        """Get supported formats for test loader."""
        from src.utils.interfaces import DataFormat
        return [DataFormat.CSV, DataFormat.JSON, DataFormat.PARQUET]
    
    def get_output_metadata(self, destination: str):
        """Get test output metadata."""
        return {
            "destination": destination,
            "records": len(self._loaded_data),
            "format": self.output_format
        }
    
    def validate_output(self, destination: str):
        """Validate test output."""
        return len(self._loaded_data) > 0
    
    def get_loaded_data(self):
        """Get loaded data for verification."""
        return self._loaded_data.copy()


@pytest.mark.integration
class TestETLPipelineIntegration:
    """Integration tests for complete ETL pipeline."""
    
    def test_simple_etl_pipeline_success(self, sample_config, mock_logger):
        """Test successful end-to-end ETL pipeline execution."""
        # Setup components
        extractor = TestExtractor("test_extractor", sample_config["extractors"]["csv_extractor"], mock_logger)
        transformer = TestTransformer("test_transformer", sample_config["transformers"]["sa2_transformer"], mock_logger)
        validator = TestValidator("test_validator", sample_config["validators"]["schema_validator"], mock_logger)
        loader = TestLoader("test_loader", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        # Setup test data
        source_data = [
            {"sa2_code": "101011001", "value": 25.5, "year": 2021},
            {"sa2_code": "101011002", "value": 30.2, "year": 2021},
            {"sa2_code": "101021003", "value": 22.8, "year": 2021}
        ]
        extractor.set_test_data(source_data)
        
        # Execute ETL pipeline
        total_records_processed = 0
        
        # Extract
        for extracted_batch in extractor.extract_with_retry("test_source"):
            
            # Transform
            for transformed_batch in transformer.transform_with_validation(extracted_batch):
                
                # Validate
                validation_results = validator.validate_comprehensive(transformed_batch)
                
                # Check for validation errors
                errors = [r for r in validation_results if r.severity == ValidationSeverity.ERROR]
                if errors:
                    raise ValidationError(f"Validation failed with {len(errors)} errors")
                
                # Load
                load_result = loader.load_with_validation(transformed_batch, "test_destination")
                total_records_processed += load_result["records_loaded"]
        
        # Verify pipeline execution
        assert total_records_processed == len(source_data)
        
        # Verify data transformations
        loaded_data = loader.get_loaded_data()
        assert len(loaded_data) == len(source_data)
        
        for record in loaded_data:
            assert record['_transformed'] is True
            assert record['_transformer'] == "test_transformer"
            assert len(record['sa2_code']) == 9  # Standardised format
            assert isinstance(record['value'], float)
    
    def test_etl_pipeline_with_validation_errors(self, sample_config, mock_logger):
        """Test ETL pipeline handling of validation errors."""
        # Setup components
        extractor = TestExtractor("test_extractor", sample_config["extractors"]["csv_extractor"], mock_logger)
        transformer = TestTransformer("test_transformer", sample_config["transformers"]["sa2_transformer"], mock_logger)
        validator = TestValidator("test_validator", sample_config["validators"]["schema_validator"], mock_logger)
        loader = TestLoader("test_loader", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        # Setup test data with validation issues
        source_data = [
            {"sa2_code": "101011001", "value": 25.5, "year": 2021},  # Valid
            {"sa2_code": "invalid", "value": -10.0, "year": 2021},    # Invalid SA2 code and negative value
            {"sa2_code": "101021003", "value": 22.8, "year": 2021}    # Valid
        ]
        extractor.set_test_data(source_data)
        
        # Execute ETL pipeline with error handling
        validation_errors = []
        processed_records = []
        
        for extracted_batch in extractor.extract_with_retry("test_source"):
            for transformed_batch in transformer.transform_with_validation(extracted_batch):
                
                # Validate and collect errors
                validation_results = validator.validate_comprehensive(transformed_batch)
                errors = [r for r in validation_results if r.severity == ValidationSeverity.ERROR]
                validation_errors.extend(errors)
                
                # Filter out invalid records
                valid_records = []
                error_record_indices = set()
                for error in errors:
                    error_record_indices.update(error.affected_records)
                
                for idx, record in enumerate(transformed_batch):
                    if idx not in error_record_indices:
                        valid_records.append(record)
                
                # Load only valid records
                if valid_records:
                    loader.load_with_validation(valid_records, "test_destination")
                    processed_records.extend(valid_records)
        
        # Verify error handling
        assert len(validation_errors) == 3  # 2 errors for invalid record + potentially more
        assert len(processed_records) == 2   # Only 2 valid records processed
        
        # Verify error details
        error_rule_ids = [error.rule_id for error in validation_errors]
        assert "sa2_code_format" in error_rule_ids
        assert "negative_value_check" in error_rule_ids
    
    def test_etl_pipeline_with_large_dataset(self, sample_config, mock_logger, performance_data_medium):
        """Test ETL pipeline with larger dataset."""
        # Setup components with smaller batch sizes
        extractor_config = sample_config["extractors"]["csv_extractor"].copy()
        extractor_config["batch_size"] = 100
        
        extractor = TestExtractor("test_extractor", extractor_config, mock_logger)
        transformer = TestTransformer("test_transformer", sample_config["transformers"]["sa2_transformer"], mock_logger)
        validator = TestValidator("test_validator", sample_config["validators"]["schema_validator"], mock_logger)
        loader = TestLoader("test_loader", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        extractor.set_test_data(performance_data_medium)
        
        # Track processing statistics
        batches_processed = 0
        total_records = 0
        validation_results_total = []
        
        start_time = datetime.now()
        
        for extracted_batch in extractor.extract_with_retry("test_source"):
            for transformed_batch in transformer.transform_with_validation(extracted_batch):
                
                validation_results = validator.validate_comprehensive(transformed_batch)
                validation_results_total.extend(validation_results)
                
                # Only load if no critical errors
                critical_errors = [r for r in validation_results if r.severity == ValidationSeverity.ERROR]
                if not critical_errors:
                    loader.load_with_validation(transformed_batch, "test_destination")
                    total_records += len(transformed_batch)
                
                batches_processed += 1
        
        end_time = datetime.now()
        processing_duration = (end_time - start_time).total_seconds()
        
        # Verify performance
        assert total_records == len(performance_data_medium)
        assert batches_processed > 1  # Should process in multiple batches
        assert processing_duration < 60.0  # Should complete within 60 seconds
        
        # Verify data integrity
        loaded_data = loader.get_loaded_data()
        assert len(loaded_data) == len(performance_data_medium)
    
    def test_etl_pipeline_error_recovery(self, sample_config, mock_logger):
        """Test ETL pipeline error recovery mechanisms."""
        # Setup components with retry configuration
        extractor_config = sample_config["extractors"]["csv_extractor"].copy()
        extractor_config["max_retries"] = 2
        
        extractor = TestExtractor("test_extractor", extractor_config, mock_logger)
        transformer = TestTransformer("test_transformer", sample_config["transformers"]["sa2_transformer"], mock_logger)
        validator = TestValidator("test_validator", sample_config["validators"]["schema_validator"], mock_logger)
        loader = TestLoader("test_loader", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        source_data = [
            {"sa2_code": "101011001", "value": 25.5, "year": 2021},
            {"sa2_code": "101011002", "value": 30.2, "year": 2021}
        ]
        extractor.set_test_data(source_data)
        
        # Simulate intermittent failures
        call_count = 0
        original_transform = transformer.transform_with_validation
        
        def failing_transform(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # Fail on first attempt
                raise TransformationError("Simulated transformation failure")
            return original_transform(*args, **kwargs)
        
        transformer.transform_with_validation = failing_transform
        
        # Execute pipeline with error recovery
        total_records = 0
        
        with patch('time.sleep'):  # Speed up retry delays
            for extracted_batch in extractor.extract_with_retry("test_source"):
                for transformed_batch in transformer.transform_with_validation(extracted_batch):
                    validation_results = validator.validate_comprehensive(transformed_batch)
                    
                    errors = [r for r in validation_results if r.severity == ValidationSeverity.ERROR]
                    if not errors:
                        load_result = loader.load_with_validation(transformed_batch, "test_destination")
                        total_records += load_result["records_loaded"]
        
        # Should succeed after retry
        assert total_records == len(source_data)
        assert call_count == 2  # One failure + one success
    
    def test_etl_pipeline_progress_tracking(self, sample_config, mock_logger, mock_progress_callback):
        """Test ETL pipeline progress tracking."""
        extractor = TestExtractor("test_extractor", sample_config["extractors"]["csv_extractor"], mock_logger)
        transformer = TestTransformer("test_transformer", sample_config["transformers"]["sa2_transformer"], mock_logger)
        validator = TestValidator("test_validator", sample_config["validators"]["schema_validator"], mock_logger)
        loader = TestLoader("test_loader", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        source_data = [{"sa2_code": f"10101{i:04d}", "value": i * 0.5, "year": 2021} for i in range(100)]
        extractor.set_test_data(source_data)
        
        # Execute pipeline with progress tracking
        for extracted_batch in extractor.extract_with_retry("test_source", progress_callback=mock_progress_callback):
            for transformed_batch in transformer.transform_with_validation(extracted_batch, progress_callback=mock_progress_callback):
                validation_results = validator.validate_comprehensive(transformed_batch)
                
                if not any(r.severity == ValidationSeverity.ERROR for r in validation_results):
                    loader.load_with_validation(transformed_batch, "test_destination", progress_callback=mock_progress_callback)
        
        # Verify progress callbacks were made
        assert mock_progress_callback.call_count > 0
    
    def test_etl_pipeline_audit_trail(self, sample_config, mock_logger):
        """Test ETL pipeline audit trail generation."""
        extractor = TestExtractor("test_extractor", sample_config["extractors"]["csv_extractor"], mock_logger)
        transformer = TestTransformer("test_transformer", sample_config["transformers"]["sa2_transformer"], mock_logger)
        validator = TestValidator("test_validator", sample_config["validators"]["schema_validator"], mock_logger)
        loader = TestLoader("test_loader", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        source_data = [
            {"sa2_code": "101011001", "value": 25.5, "year": 2021},
            {"sa2_code": "101011002", "value": 30.2, "year": 2021}
        ]
        extractor.set_test_data(source_data)
        
        # Execute pipeline and collect audit information
        audit_data = {
            "pipeline_id": "test_pipeline_001",
            "start_time": datetime.now(),
            "components": []
        }
        
        for extracted_batch in extractor.extract_with_retry("test_source"):
            # Track extraction
            audit_data["components"].append({
                "component": "extractor",
                "component_id": extractor.extractor_id,
                "records_processed": len(extracted_batch),
                "status": "completed"
            })
            
            for transformed_batch in transformer.transform_with_validation(extracted_batch):
                # Track transformation
                audit_data["components"].append({
                    "component": "transformer",
                    "component_id": transformer.transformer_id,
                    "records_processed": len(transformed_batch),
                    "status": "completed"
                })
                
                validation_results = validator.validate_comprehensive(transformed_batch)
                
                # Track validation
                audit_data["components"].append({
                    "component": "validator",
                    "component_id": validator.validator_id,
                    "records_processed": len(transformed_batch),
                    "validation_results": len(validation_results),
                    "status": "completed"
                })
                
                if not any(r.severity == ValidationSeverity.ERROR for r in validation_results):
                    load_result = loader.load_with_validation(transformed_batch, "test_destination")
                    
                    # Track loading
                    audit_data["components"].append({
                        "component": "loader",
                        "component_id": loader.loader_id,
                        "records_loaded": load_result["records_loaded"],
                        "status": "completed"
                    })
        
        audit_data["end_time"] = datetime.now()
        audit_data["duration"] = (audit_data["end_time"] - audit_data["start_time"]).total_seconds()
        
        # Verify audit trail
        assert len(audit_data["components"]) >= 4  # At least one of each component type
        assert all(comp["status"] == "completed" for comp in audit_data["components"])
        assert audit_data["duration"] > 0
        
        # Verify component execution order
        component_types = [comp["component"] for comp in audit_data["components"]]
        assert component_types.index("extractor") < component_types.index("transformer")
        assert component_types.index("transformer") < component_types.index("validator")
        assert component_types.index("validator") < component_types.index("loader")


@pytest.mark.integration
@pytest.mark.database
class TestETLPipelineWithDatabase:
    """Integration tests for ETL pipeline with real database operations."""
    
    def test_etl_pipeline_sqlite_integration(self, sample_config, mock_logger, sqlite_db):
        """Test ETL pipeline with SQLite database integration."""
        extractor = TestExtractor("test_extractor", sample_config["extractors"]["csv_extractor"], mock_logger)
        transformer = TestTransformer("test_transformer", sample_config["transformers"]["sa2_transformer"], mock_logger)
        validator = TestValidator("test_validator", sample_config["validators"]["schema_validator"], mock_logger)
        
        # Database loader
        class SQLiteLoader(TestLoader):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.db_connection = sqlite_db
            
            def load(self, data: DataBatch, destination: str, **kwargs):
                table_name = kwargs.get("table_name", "etl_test_data")
                
                # Create table if not exists
                self.db_connection.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id INTEGER PRIMARY KEY,
                        sa2_code TEXT,
                        value REAL,
                        year INTEGER,
                        _transformed BOOLEAN,
                        _transformer TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Insert data
                for record in data:
                    self.db_connection.execute(f"""
                        INSERT INTO {table_name} (sa2_code, value, year, _transformed, _transformer)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        record.get('sa2_code'),
                        record.get('value'),
                        record.get('year'),
                        record.get('_transformed'),
                        record.get('_transformer')
                    ))
                
                self.db_connection.commit()
                
                return {
                    "destination": table_name,
                    "records_loaded": len(data),
                    "format": "sqlite"
                }
            
            def get_supported_formats(self):
                """Get supported formats for SQLite loader."""
                from src.utils.interfaces import DataFormat
                return [DataFormat.CSV, DataFormat.JSON, DataFormat.SQLITE]
        
        loader = SQLiteLoader("sqlite_loader", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        source_data = [
            {"sa2_code": "101011001", "value": 25.5, "year": 2021},
            {"sa2_code": "101011002", "value": 30.2, "year": 2021},
            {"sa2_code": "101021003", "value": 22.8, "year": 2021}
        ]
        extractor.set_test_data(source_data)
        
        # Execute ETL pipeline
        for extracted_batch in extractor.extract_with_retry("test_source"):
            for transformed_batch in transformer.transform_with_validation(extracted_batch):
                validation_results = validator.validate_comprehensive(transformed_batch)
                
                if not any(r.severity == ValidationSeverity.ERROR for r in validation_results):
                    loader.load_with_validation(transformed_batch, "test_destination", table_name="etl_test_data")
        
        # Verify data in database
        cursor = sqlite_db.execute("SELECT COUNT(*) FROM etl_test_data")
        record_count = cursor.fetchone()[0]
        assert record_count == len(source_data)
        
        # Verify data content
        cursor = sqlite_db.execute("SELECT sa2_code, value, _transformed FROM etl_test_data ORDER BY sa2_code")
        db_records = cursor.fetchall()
        
        assert len(db_records) == len(source_data)
        for record in db_records:
            assert len(record[0]) == 9  # Standardised SA2 code
            assert isinstance(record[1], float)
            assert record[2] == 1  # _transformed = True
    
    def test_etl_pipeline_transaction_handling(self, sample_config, mock_logger, sqlite_db):
        """Test ETL pipeline transaction handling and rollback."""
        extractor = TestExtractor("test_extractor", sample_config["extractors"]["csv_extractor"], mock_logger)
        transformer = TestTransformer("test_transformer", sample_config["transformers"]["sa2_transformer"], mock_logger)
        validator = TestValidator("test_validator", sample_config["validators"]["schema_validator"], mock_logger)
        
        class TransactionalLoader(TestLoader):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.db_connection = sqlite_db
                self._should_fail = False
            
            def load(self, data: DataBatch, destination: str, **kwargs):
                table_name = kwargs.get("table_name", "transaction_test")
                
                # Start transaction
                self.db_connection.execute("BEGIN TRANSACTION")
                
                try:
                    # Create table
                    self.db_connection.execute(f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            id INTEGER PRIMARY KEY,
                            sa2_code TEXT,
                            value REAL
                        )
                    """)
                    
                    # Insert data
                    for i, record in enumerate(data):
                        if self._should_fail and i == len(data) - 1:
                            raise Exception("Simulated database error")
                        
                        self.db_connection.execute(f"""
                            INSERT INTO {table_name} (sa2_code, value)
                            VALUES (?, ?)
                        """, (record.get('sa2_code'), record.get('value')))
                    
                    self.db_connection.commit()
                    return {"records_loaded": len(data)}
                    
                except Exception as e:
                    self.db_connection.rollback()
                    raise LoadingError(f"Database error: {e}")
            
            def get_supported_formats(self):
                """Get supported formats for transactional loader."""
                from src.utils.interfaces import DataFormat
                return [DataFormat.SQLITE]
            
            def set_failure_mode(self, should_fail: bool):
                self._should_fail = should_fail
        
        loader = TransactionalLoader("transactional_loader", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        source_data = [
            {"sa2_code": "101011001", "value": 25.5},
            {"sa2_code": "101011002", "value": 30.2}
        ]
        extractor.set_test_data(source_data)
        
        # First, test successful transaction
        for extracted_batch in extractor.extract_with_retry("test_source"):
            for transformed_batch in transformer.transform_with_validation(extracted_batch):
                validation_results = validator.validate_comprehensive(transformed_batch)
                
                if not any(r.severity == ValidationSeverity.ERROR for r in validation_results):
                    loader.load_with_validation(transformed_batch, "test_destination", table_name="transaction_success")
        
        # Verify successful load
        cursor = sqlite_db.execute("SELECT COUNT(*) FROM transaction_success")
        assert cursor.fetchone()[0] == len(source_data)
        
        # Test failed transaction with rollback
        loader.set_failure_mode(True)
        
        with pytest.raises(LoadingError):
            for extracted_batch in extractor.extract_with_retry("test_source"):
                for transformed_batch in transformer.transform_with_validation(extracted_batch):
                    validation_results = validator.validate_comprehensive(transformed_batch)
                    
                    if not any(r.severity == ValidationSeverity.ERROR for r in validation_results):
                        loader.load_with_validation(transformed_batch, "test_destination", table_name="transaction_failure")
        
        # Verify no data was committed due to rollback
        try:
            cursor = sqlite_db.execute("SELECT COUNT(*) FROM transaction_failure")
            count = cursor.fetchone()[0]
            assert count == 0  # Should be empty due to rollback
        except sqlite3.OperationalError:
            # Table might not exist at all due to rollback
            pass


@pytest.mark.integration
@pytest.mark.slow
class TestETLPipelinePerformance:
    """Performance integration tests for ETL pipeline."""
    
    def test_etl_pipeline_memory_efficiency(self, sample_config, mock_logger, memory_intensive_data):
        """Test ETL pipeline memory efficiency with large records."""
        # Configure for memory efficiency
        extractor_config = sample_config["extractors"]["csv_extractor"].copy()
        extractor_config["batch_size"] = 50  # Smaller batches
        
        extractor = TestExtractor("test_extractor", extractor_config, mock_logger)
        transformer = TestTransformer("test_transformer", sample_config["transformers"]["sa2_transformer"], mock_logger)
        validator = TestValidator("test_validator", sample_config["validators"]["schema_validator"], mock_logger)
        loader = TestLoader("test_loader", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        extractor.set_test_data(memory_intensive_data)
        
        # Monitor memory usage (simplified)
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Execute pipeline
        total_records = 0
        for extracted_batch in extractor.extract_with_retry("test_source"):
            for transformed_batch in transformer.transform_with_validation(extracted_batch):
                validation_results = validator.validate_comprehensive(transformed_batch)
                
                if not any(r.severity == ValidationSeverity.ERROR for r in validation_results):
                    load_result = loader.load_with_validation(transformed_batch, "test_destination")
                    total_records += load_result["records_loaded"]
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Verify processing completed
        assert total_records == len(memory_intensive_data)
        
        # Memory growth should be reasonable (less than 100MB for this test)
        assert memory_growth < 100 * 1024 * 1024
    
    def test_etl_pipeline_concurrent_processing(self, sample_config, mock_logger):
        """Test ETL pipeline thread safety with concurrent processing."""
        import threading
        import queue
        
        extractor = TestExtractor("test_extractor", sample_config["extractors"]["csv_extractor"], mock_logger)
        transformer = TestTransformer("test_transformer", sample_config["transformers"]["sa2_transformer"], mock_logger)
        validator = TestValidator("test_validator", sample_config["validators"]["schema_validator"], mock_logger)
        loader = TestLoader("test_loader", sample_config["loaders"]["sqlite_loader"], mock_logger)
        
        # Create separate datasets for each thread
        base_data = [{"sa2_code": f"10101{i:04d}", "value": i * 0.5, "year": 2021} for i in range(100)]
        
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def process_data(thread_id):
            try:
                # Create thread-specific data
                thread_data = []
                for record in base_data:
                    thread_record = record.copy()
                    thread_record['thread_id'] = thread_id
                    thread_data.append(thread_record)
                
                extractor.set_test_data(thread_data)
                
                processed_count = 0
                for extracted_batch in extractor.extract_with_retry(f"source_{thread_id}"):
                    for transformed_batch in transformer.transform_with_validation(extracted_batch):
                        validation_results = validator.validate_comprehensive(transformed_batch)
                        
                        if not any(r.severity == ValidationSeverity.ERROR for r in validation_results):
                            load_result = loader.load_with_validation(transformed_batch, f"destination_{thread_id}")
                            processed_count += load_result["records_loaded"]
                
                results_queue.put((thread_id, processed_count))
                
            except Exception as e:
                errors_queue.put((thread_id, str(e)))
        
        # Create and start threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=process_data, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert errors_queue.empty(), "No errors should occur in concurrent processing"
        
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        assert len(results) == 3
        for thread_id, processed_count in results:
            assert processed_count == len(base_data)