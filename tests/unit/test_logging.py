"""
Unit tests for AHGD logging framework.

Tests logging configuration, formatters, handlers, and monitoring functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import logging
import sys
from io import StringIO
from pathlib import Path
import json

from src.utils.logging import (
    get_logger,
    configure_logging,
    StructuredFormatter,
    SensitiveDataFilter,
    PerformanceLogger,
    AuditLogger
)


@pytest.mark.unit
class TestBasicLogging:
    """Test basic logging functionality."""
    
    def test_get_logger_default(self):
        """Test getting logger with default configuration."""
        logger = get_logger("test_logger")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
    
    def test_get_logger_with_level(self):
        """Test getting logger with specific level."""
        logger = get_logger("test_logger", level=logging.DEBUG)
        
        assert logger.level == logging.DEBUG
    
    def test_get_logger_hierarchy(self):
        """Test logger hierarchy and inheritance."""
        parent_logger = get_logger("ahgd")
        child_logger = get_logger("ahgd.extractors")
        grandchild_logger = get_logger("ahgd.extractors.csv")
        
        assert parent_logger.name == "ahgd"
        assert child_logger.name == "ahgd.extractors"
        assert grandchild_logger.name == "ahgd.extractors.csv"
        
        # Test hierarchy relationship
        assert child_logger.parent == parent_logger
        assert grandchild_logger.parent == child_logger
    
    def test_logger_singleton_behavior(self):
        """Test that get_logger returns the same instance for the same name."""
        logger1 = get_logger("singleton_test")
        logger2 = get_logger("singleton_test")
        
        assert logger1 is logger2
    
    def test_configure_logging_from_dict(self, temp_dir):
        """Test configuring logging from dictionary configuration."""
        log_file = temp_dir / "test.log"
        
        config = {
            "version": 1,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                }
            },
            "handlers": {
                "file": {
                    "class": "logging.FileHandler",
                    "filename": str(log_file),
                    "formatter": "standard",
                    "level": "INFO"
                }
            },
            "loggers": {
                "ahgd": {
                    "level": "DEBUG",
                    "handlers": ["file"]
                }
            }
        }
        
        configure_logging(config)
        
        logger = get_logger("ahgd.test")
        logger.info("Test message")
        
        # Check that log file was created and contains message
        assert log_file.exists()
        log_content = log_file.read_text()
        assert "Test message" in log_content
    
    def test_configure_logging_from_file(self, temp_dir):
        """Test configuring logging from YAML configuration file."""
        config_file = temp_dir / "logging_config.yaml"
        log_file = temp_dir / "test.log"
        
        config_content = f"""
version: 1
formatters:
  standard:
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
handlers:
  file:
    class: logging.FileHandler
    filename: {log_file}
    formatter: standard
    level: INFO
loggers:
  ahgd:
    level: DEBUG
    handlers: [file]
"""
        
        config_file.write_text(config_content)
        
        configure_logging(str(config_file))
        
        logger = get_logger("ahgd.test")
        logger.info("Test message from file config")
        
        assert log_file.exists()
        log_content = log_file.read_text()
        assert "Test message from file config" in log_content


@pytest.mark.unit
class TestStructuredFormatter:
    """Test structured JSON formatter."""
    
    def test_structured_formatter_basic(self):
        """Test basic structured formatting."""
        formatter = StructuredFormatter()
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        
        assert parsed["logger"] == "test_logger"
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert parsed["pathname"] == "/path/to/file.py"
        assert parsed["lineno"] == 42
        assert "timestamp" in parsed
    
    def test_structured_formatter_with_extra_fields(self):
        """Test structured formatting with extra fields."""
        formatter = StructuredFormatter()
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Add extra fields
        record.user_id = "user123"
        record.request_id = "req456"
        record.operation = "data_extraction"
        
        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        
        assert parsed["user_id"] == "user123"
        assert parsed["request_id"] == "req456"
        assert parsed["operation"] == "data_extraction"
    
    def test_structured_formatter_with_exception(self):
        """Test structured formatting with exception information."""
        formatter = StructuredFormatter()
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="/path/to/file.py",
                lineno=42,
                msg="An error occurred",
                args=(),
                exc_info=sys.exc_info()
            )
        
        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        
        assert parsed["level"] == "ERROR"
        assert parsed["message"] == "An error occurred"
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]
        assert "Test exception" in parsed["exception"]
    
    def test_structured_formatter_custom_fields(self):
        """Test structured formatter with custom field configuration."""
        custom_fields = ["module", "function", "thread"]
        formatter = StructuredFormatter(include_fields=custom_fields)
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        record.thread = 12345
        
        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        
        assert "module" in parsed
        assert "function" in parsed
        assert "thread" in parsed
        assert parsed["module"] == "test_module"
        assert parsed["function"] == "test_function"
        assert parsed["thread"] == 12345


@pytest.mark.unit
class TestSensitiveDataFilter:
    """Test sensitive data filtering."""
    
    def test_sensitive_data_filter_passwords(self):
        """Test filtering of password-related sensitive data."""
        sensitive_patterns = [
            r'password["\']?\s*[:=]\s*["\']?([^"\'\s]+)',
            r'api_key["\']?\s*[:=]\s*["\']?([^"\'\s]+)',
            r'secret["\']?\s*[:=]\s*["\']?([^"\'\s]+)'
        ]
        
        filter_obj = SensitiveDataFilter(patterns=sensitive_patterns)
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="User login with password=secret123 and api_key=abc123xyz",
            args=(),
            exc_info=None
        )
        
        should_log = filter_obj.filter(record)
        
        assert should_log is True  # Record should still be logged
        assert "password=***" in record.getMessage()
        assert "api_key=***" in record.getMessage()
        assert "secret123" not in record.getMessage()
        assert "abc123xyz" not in record.getMessage()
    
    def test_sensitive_data_filter_credit_cards(self):
        """Test filtering of credit card numbers."""
        credit_card_pattern = r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        
        filter_obj = SensitiveDataFilter(patterns=[credit_card_pattern])
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Processing payment for card 1234-5678-9012-3456",
            args=(),
            exc_info=None
        )
        
        filter_obj.filter(record)
        
        assert "1234-5678-9012-3456" not in record.getMessage()
        assert "****-****-****-****" in record.getMessage()
    
    def test_sensitive_data_filter_email_addresses(self):
        """Test filtering of email addresses."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        filter_obj = SensitiveDataFilter(patterns=[email_pattern])
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="User registration: user@example.com and admin@company.org",
            args=(),
            exc_info=None
        )
        
        filter_obj.filter(record)
        
        message = record.getMessage()
        assert "user@example.com" not in message
        assert "admin@company.org" not in message
        assert "***@***.***" in message
    
    def test_sensitive_data_filter_no_match(self):
        """Test filter when no sensitive data is found."""
        filter_obj = SensitiveDataFilter(patterns=[r'password[:=]\s*(\w+)'])
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Normal log message with no sensitive data",
            args=(),
            exc_info=None
        )
        
        original_message = record.getMessage()
        filter_obj.filter(record)
        
        assert record.getMessage() == original_message
    
    def test_sensitive_data_filter_custom_replacement(self):
        """Test custom replacement text for sensitive data."""
        filter_obj = SensitiveDataFilter(
            patterns=[r'token[:=]\s*(\w+)'],
            replacement="[REDACTED]"
        )
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Authentication with token=abc123def456",
            args=(),
            exc_info=None
        )
        
        filter_obj.filter(record)
        
        assert "token=[REDACTED]" in record.getMessage()
        assert "abc123def456" not in record.getMessage()


@pytest.mark.unit
class TestPerformanceLogger:
    """Test performance logging functionality."""
    
    def test_performance_logger_context_manager(self):
        """Test performance logger as context manager."""
        logger = Mock()
        perf_logger = PerformanceLogger(logger)
        
        with perf_logger.time_operation("test_operation"):
            # Simulate some work
            import time
            time.sleep(0.1)
        
        # Verify that performance was logged
        logger.info.assert_called()
        call_args = logger.info.call_args[0][0]
        assert "test_operation" in call_args
        assert "duration" in call_args or "time" in call_args
    
    def test_performance_logger_decorator(self):
        """Test performance logger as decorator."""
        logger = Mock()
        perf_logger = PerformanceLogger(logger)
        
        @perf_logger.log_performance
        def test_function():
            import time
            time.sleep(0.1)
            return "result"
        
        result = test_function()
        
        assert result == "result"
        logger.info.assert_called()
    
    def test_performance_logger_manual_timing(self):
        """Test manual timing with performance logger."""
        logger = Mock()
        perf_logger = PerformanceLogger(logger)
        
        timer = perf_logger.start_timer("manual_operation")
        import time
        time.sleep(0.1)
        perf_logger.end_timer(timer)
        
        logger.info.assert_called()
    
    def test_performance_logger_statistics(self):
        """Test performance statistics collection."""
        logger = Mock()
        perf_logger = PerformanceLogger(logger, collect_stats=True)
        
        # Perform multiple operations
        for i in range(5):
            with perf_logger.time_operation("repeated_operation"):
                import time
                time.sleep(0.05)
        
        stats = perf_logger.get_statistics("repeated_operation")
        
        assert stats["count"] == 5
        assert "min_duration" in stats
        assert "max_duration" in stats
        assert "avg_duration" in stats
        assert "total_duration" in stats
    
    def test_performance_logger_threshold_warning(self):
        """Test performance warning for slow operations."""
        logger = Mock()
        perf_logger = PerformanceLogger(logger, slow_threshold=0.05)
        
        # Fast operation - should not trigger warning
        with perf_logger.time_operation("fast_operation"):
            import time
            time.sleep(0.01)
        
        # Slow operation - should trigger warning
        with perf_logger.time_operation("slow_operation"):
            import time
            time.sleep(0.1)
        
        # Check that warning was logged for slow operation
        warning_calls = [call for call in logger.warning.call_args_list 
                        if "slow_operation" in str(call)]
        assert len(warning_calls) > 0


@pytest.mark.unit
class TestAuditLogger:
    """Test audit logging functionality."""
    
    def test_audit_logger_basic_event(self, temp_dir):
        """Test basic audit event logging."""
        audit_file = temp_dir / "audit.log"
        audit_logger = AuditLogger(str(audit_file))
        
        audit_logger.log_event(
            event_type="data_extraction",
            user_id="user123",
            details={"source": "census_data", "records": 1000}
        )
        
        assert audit_file.exists()
        audit_content = audit_file.read_text()
        assert "data_extraction" in audit_content
        assert "user123" in audit_content
        assert "census_data" in audit_content
    
    def test_audit_logger_structured_format(self, temp_dir):
        """Test audit logger structured JSON format."""
        audit_file = temp_dir / "audit.json"
        audit_logger = AuditLogger(str(audit_file), format="json")
        
        audit_logger.log_event(
            event_type="data_validation",
            user_id="user456",
            resource="health_indicators",
            action="validate",
            details={"errors": 5, "warnings": 12}
        )
        
        audit_content = audit_file.read_text()
        lines = audit_content.strip().split('\n')
        
        # Each line should be valid JSON
        for line in lines:
            audit_entry = json.loads(line)
            assert "timestamp" in audit_entry
            assert "event_type" in audit_entry
            assert "user_id" in audit_entry
    
    def test_audit_logger_access_events(self, temp_dir):
        """Test logging of data access events."""
        audit_file = temp_dir / "audit.log"
        audit_logger = AuditLogger(str(audit_file))
        
        # Log data access
        audit_logger.log_access(
            user_id="researcher001",
            resource="mortality_data",
            action="read",
            ip_address="192.168.1.100"
        )
        
        audit_content = audit_file.read_text()
        assert "data_access" in audit_content
        assert "researcher001" in audit_content
        assert "mortality_data" in audit_content
        assert "192.168.1.100" in audit_content
    
    def test_audit_logger_security_events(self, temp_dir):
        """Test logging of security events."""
        audit_file = temp_dir / "audit.log"
        audit_logger = AuditLogger(str(audit_file))
        
        # Log failed authentication
        audit_logger.log_security_event(
            event_type="authentication_failed",
            user_id="unknown_user",
            ip_address="10.0.0.1",
            details={"reason": "invalid_credentials", "attempts": 3}
        )
        
        audit_content = audit_file.read_text()
        assert "security_event" in audit_content
        assert "authentication_failed" in audit_content
        assert "invalid_credentials" in audit_content
    
    def test_audit_logger_data_modification_events(self, temp_dir):
        """Test logging of data modification events."""
        audit_file = temp_dir / "audit.log"
        audit_logger = AuditLogger(str(audit_file))
        
        # Log data modification
        audit_logger.log_data_modification(
            user_id="admin001",
            table="sa2_boundaries",
            action="update",
            record_id="101011001",
            old_values={"population": 15000},
            new_values={"population": 15500}
        )
        
        audit_content = audit_file.read_text()
        assert "data_modification" in audit_content
        assert "sa2_boundaries" in audit_content
        assert "101011001" in audit_content
    
    def test_audit_logger_retention_policy(self, temp_dir):
        """Test audit log retention policy."""
        audit_file = temp_dir / "audit.log"
        audit_logger = AuditLogger(str(audit_file), retention_days=7)
        
        # Create some old entries (mocked)
        from datetime import datetime, timedelta
        old_date = datetime.now() - timedelta(days=10)
        
        with patch('src.utils.logging.datetime') as mock_datetime:
            mock_datetime.now.return_value = old_date
            
            audit_logger.log_event("old_event", "user123")
        
        # Apply retention policy
        audit_logger.apply_retention_policy()
        
        # Old entries should be removed (implementation dependent)
        # This test would need actual implementation of retention policy


@pytest.mark.unit
class TestLoggingEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_logging_with_none_values(self):
        """Test logging with None values."""
        logger = get_logger("test_none")
        
        # Should handle None message gracefully
        try:
            logger.info(None)
        except Exception:
            pytest.fail("Logger should handle None message gracefully")
    
    def test_logging_with_unicode_characters(self):
        """Test logging with unicode characters."""
        logger = get_logger("test_unicode")
        
        # Should handle unicode characters
        unicode_message = "Processing data for 中国城 and café résumé: αβγδε 🏥📊"
        
        try:
            logger.info(unicode_message)
        except Exception:
            pytest.fail("Logger should handle unicode characters gracefully")
    
    def test_logging_very_long_messages(self):
        """Test logging with very long messages."""
        logger = get_logger("test_long")
        
        # Create very long message
        long_message = "A" * 10000
        
        try:
            logger.info(long_message)
        except Exception:
            pytest.fail("Logger should handle very long messages gracefully")
    
    def test_logging_with_circular_references(self):
        """Test logging with objects containing circular references."""
        logger = get_logger("test_circular")
        
        # Create object with circular reference
        obj1 = {"name": "obj1"}
        obj2 = {"name": "obj2", "ref": obj1}
        obj1["ref"] = obj2
        
        try:
            logger.info("Object with circular reference: %s", obj1)
        except Exception:
            # This might fail, which is acceptable for circular references
            pass
    
    def test_logging_configuration_errors(self, temp_dir):
        """Test handling of logging configuration errors."""
        config_file = temp_dir / "invalid_logging.yaml"
        
        # Create invalid configuration
        invalid_config = """
invalid_yaml_syntax: [
    missing_closing_bracket
"""
        config_file.write_text(invalid_config)
        
        with pytest.raises((ValueError, yaml.YAMLError)):
            configure_logging(str(config_file))
    
    def test_logging_file_permission_errors(self, temp_dir):
        """Test handling of file permission errors."""
        # Create directory with restricted permissions
        restricted_dir = temp_dir / "restricted"
        restricted_dir.mkdir(mode=0o000)  # No permissions
        
        log_file = restricted_dir / "test.log"
        
        try:
            config = {
                "version": 1,
                "handlers": {
                    "file": {
                        "class": "logging.FileHandler",
                        "filename": str(log_file),
                        "level": "INFO"
                    }
                },
                "root": {
                    "level": "INFO",
                    "handlers": ["file"]
                }
            }
            
            configure_logging(config)
            logger = get_logger("permission_test")
            logger.info("This should fail")
            
        except (PermissionError, OSError):
            # Expected for restricted permissions
            pass
        finally:
            # Cleanup - restore permissions
            try:
                restricted_dir.chmod(0o755)
            except OSError:
                pass
    
    def test_concurrent_logging_safety(self):
        """Test thread safety of logging operations."""
        import threading
        import queue
        
        logger = get_logger("concurrent_test")
        messages = queue.Queue()
        errors = queue.Queue()
        
        def log_messages(thread_id):
            try:
                for i in range(100):
                    message = f"Thread {thread_id} message {i}"
                    logger.info(message)
                    messages.put(message)
            except Exception as e:
                errors.put(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=log_messages, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check for errors
        assert errors.empty(), "Logging should be thread-safe"
        
        # Verify messages were logged
        assert messages.qsize() == 500  # 5 threads × 100 messages