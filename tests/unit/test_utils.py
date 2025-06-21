"""
Unit tests for AHGD utility functions.

Tests utility functions, helpers, monitoring, and miscellaneous functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import hashlib
import json

from src.utils.monitoring import HealthMonitor, MetricsCollector, AlertManager
from src.utils.secrets import SecretManager, encrypt_data, decrypt_data
from src.utils.interfaces import ConfigurationError


@pytest.mark.unit
class TestHealthMonitor:
    """Test health monitoring functionality."""
    
    def test_health_monitor_initialisation(self):
        """Test health monitor initialisation."""
        monitor = HealthMonitor()
        
        assert monitor.is_healthy() is True
        assert monitor.get_status() == "healthy"
        assert isinstance(monitor.get_checks(), dict)
    
    def test_health_monitor_add_check(self):
        """Test adding health checks."""
        monitor = HealthMonitor()
        
        def database_check():
            return {"status": "healthy", "response_time": 0.05}
        
        def api_check():
            return {"status": "degraded", "error": "High latency"}
        
        monitor.add_check("database", database_check)
        monitor.add_check("api", api_check)
        
        checks = monitor.get_checks()
        assert "database" in checks
        assert "api" in checks
        assert checks["database"]["status"] == "healthy"
        assert checks["api"]["status"] == "degraded"
    
    def test_health_monitor_overall_status(self):
        """Test overall health status calculation."""
        monitor = HealthMonitor()
        
        def healthy_check():
            return {"status": "healthy"}
        
        def unhealthy_check():
            return {"status": "unhealthy", "error": "Service down"}
        
        # All checks healthy
        monitor.add_check("check1", healthy_check)
        monitor.add_check("check2", healthy_check)
        assert monitor.is_healthy() is True
        assert monitor.get_status() == "healthy"
        
        # One check unhealthy
        monitor.add_check("check3", unhealthy_check)
        assert monitor.is_healthy() is False
        assert monitor.get_status() == "unhealthy"
    
    def test_health_monitor_check_execution_error(self):
        """Test handling of health check execution errors."""
        monitor = HealthMonitor()
        
        def failing_check():
            raise Exception("Check failed")
        
        monitor.add_check("failing", failing_check)
        
        checks = monitor.get_checks()
        assert checks["failing"]["status"] == "error"
        assert "Check failed" in checks["failing"]["error"]
    
    def test_health_monitor_detailed_status(self):
        """Test detailed health status information."""
        monitor = HealthMonitor()
        
        def detailed_check():
            return {
                "status": "healthy",
                "response_time": 0.025,
                "connections": 5,
                "memory_usage": "45%",
                "last_error": None
            }
        
        monitor.add_check("detailed", detailed_check)
        
        status = monitor.get_detailed_status()
        
        assert "overall_status" in status
        assert "checks" in status
        assert "timestamp" in status
        assert status["checks"]["detailed"]["response_time"] == 0.025
        assert status["checks"]["detailed"]["connections"] == 5
    
    def test_health_monitor_history_tracking(self):
        """Test health status history tracking."""
        monitor = HealthMonitor(track_history=True, history_limit=10)
        
        def variable_check():
            import random
            return {"status": "healthy" if random.random() > 0.5 else "degraded"}
        
        monitor.add_check("variable", variable_check)
        
        # Run multiple checks to build history
        for _ in range(15):
            monitor.get_status()
        
        history = monitor.get_history()
        
        assert len(history) <= 10  # Should respect history limit
        assert all("timestamp" in entry for entry in history)
        assert all("status" in entry for entry in history)


@pytest.mark.unit
class TestMetricsCollector:
    """Test metrics collection functionality."""
    
    def test_metrics_collector_counter(self):
        """Test counter metrics."""
        collector = MetricsCollector()
        
        # Increment counter
        collector.increment("requests_total")
        collector.increment("requests_total", value=5)
        collector.increment("errors_total", tags={"type": "validation"})
        
        metrics = collector.get_metrics()
        
        assert metrics["requests_total"]["value"] == 6
        assert metrics["errors_total"]["value"] == 1
        assert metrics["errors_total"]["tags"]["type"] == "validation"
    
    def test_metrics_collector_gauge(self):
        """Test gauge metrics."""
        collector = MetricsCollector()
        
        # Set gauge values
        collector.set_gauge("cpu_usage", 75.5)
        collector.set_gauge("memory_usage", 60.2, tags={"component": "extractor"})
        
        # Update gauge
        collector.set_gauge("cpu_usage", 80.0)
        
        metrics = collector.get_metrics()
        
        assert metrics["cpu_usage"]["value"] == 80.0
        assert metrics["memory_usage"]["value"] == 60.2
        assert metrics["memory_usage"]["tags"]["component"] == "extractor"
    
    def test_metrics_collector_histogram(self):
        """Test histogram metrics for timing data."""
        collector = MetricsCollector()
        
        # Record timing values
        response_times = [0.1, 0.15, 0.12, 0.18, 0.09, 0.25, 0.11]
        
        for time in response_times:
            collector.record_histogram("response_time", time)
        
        metrics = collector.get_metrics()
        histogram = metrics["response_time"]
        
        assert histogram["count"] == len(response_times)
        assert histogram["min"] == min(response_times)
        assert histogram["max"] == max(response_times)
        assert abs(histogram["avg"] - sum(response_times) / len(response_times)) < 0.001
    
    def test_metrics_collector_timer_context(self):
        """Test timer context manager."""
        collector = MetricsCollector()
        
        with collector.timer("operation_duration"):
            import time
            time.sleep(0.1)
        
        metrics = collector.get_metrics()
        
        assert "operation_duration" in metrics
        assert metrics["operation_duration"]["count"] == 1
        assert metrics["operation_duration"]["min"] >= 0.1
    
    def test_metrics_collector_tags_filtering(self):
        """Test filtering metrics by tags."""
        collector = MetricsCollector()
        
        collector.increment("requests", tags={"method": "GET", "endpoint": "/api/v1"})
        collector.increment("requests", tags={"method": "POST", "endpoint": "/api/v1"})
        collector.increment("requests", tags={"method": "GET", "endpoint": "/api/v2"})
        
        # Filter by method
        get_metrics = collector.get_metrics(tags={"method": "GET"})
        post_metrics = collector.get_metrics(tags={"method": "POST"})
        
        assert len(get_metrics) == 2  # Two GET requests
        assert len(post_metrics) == 1  # One POST request
    
    def test_metrics_collector_export_prometheus(self):
        """Test exporting metrics in Prometheus format."""
        collector = MetricsCollector()
        
        collector.increment("http_requests_total", tags={"status": "200"})
        collector.set_gauge("memory_usage_bytes", 1024000)
        collector.record_histogram("request_duration_seconds", 0.15)
        
        prometheus_output = collector.export_prometheus()
        
        assert "http_requests_total" in prometheus_output
        assert "memory_usage_bytes" in prometheus_output
        assert "request_duration_seconds" in prometheus_output
        assert 'status="200"' in prometheus_output
    
    def test_metrics_collector_reset(self):
        """Test resetting metrics."""
        collector = MetricsCollector()
        
        collector.increment("counter")
        collector.set_gauge("gauge", 100)
        
        assert len(collector.get_metrics()) == 2
        
        collector.reset()
        
        assert len(collector.get_metrics()) == 0


@pytest.mark.unit
class TestAlertManager:
    """Test alert management functionality."""
    
    def test_alert_manager_rule_registration(self):
        """Test registering alert rules."""
        alert_manager = AlertManager()
        
        def high_cpu_rule(metrics):
            cpu_usage = metrics.get("cpu_usage", {}).get("value", 0)
            return cpu_usage > 80
        
        def error_rate_rule(metrics):
            errors = metrics.get("errors_total", {}).get("value", 0)
            requests = metrics.get("requests_total", {}).get("value", 1)
            return (errors / requests) > 0.05
        
        alert_manager.register_rule("high_cpu", high_cpu_rule, severity="warning")
        alert_manager.register_rule("high_error_rate", error_rate_rule, severity="critical")
        
        rules = alert_manager.get_rules()
        assert "high_cpu" in rules
        assert "high_error_rate" in rules
        assert rules["high_cpu"]["severity"] == "warning"
        assert rules["high_error_rate"]["severity"] == "critical"
    
    def test_alert_manager_alert_triggering(self):
        """Test alert triggering based on metrics."""
        alert_manager = AlertManager()
        
        def cpu_alert_rule(metrics):
            return metrics.get("cpu_usage", {}).get("value", 0) > 80
        
        alert_manager.register_rule("cpu_alert", cpu_alert_rule)
        
        # Normal CPU usage - no alert
        normal_metrics = {"cpu_usage": {"value": 65}}
        alerts = alert_manager.evaluate_rules(normal_metrics)
        assert len(alerts) == 0
        
        # High CPU usage - should trigger alert
        high_metrics = {"cpu_usage": {"value": 90}}
        alerts = alert_manager.evaluate_rules(high_metrics)
        assert len(alerts) == 1
        assert alerts[0]["rule"] == "cpu_alert"
        assert alerts[0]["triggered"] is True
    
    def test_alert_manager_alert_history(self):
        """Test alert history tracking."""
        alert_manager = AlertManager(track_history=True)
        
        def test_rule(metrics):
            return metrics.get("test_value", 0) > 50
        
        alert_manager.register_rule("test_alert", test_rule)
        
        # Trigger multiple alerts
        alert_manager.evaluate_rules({"test_value": 60})  # Trigger
        alert_manager.evaluate_rules({"test_value": 30})  # No trigger
        alert_manager.evaluate_rules({"test_value": 80})  # Trigger
        
        history = alert_manager.get_alert_history()
        
        assert len(history) >= 2  # At least 2 triggered alerts
        assert all("timestamp" in alert for alert in history)
        assert all("rule" in alert for alert in history)
    
    def test_alert_manager_notification_handlers(self):
        """Test alert notification handlers."""
        alert_manager = AlertManager()
        
        email_notifications = []
        slack_notifications = []
        
        def email_handler(alert):
            email_notifications.append(alert)
        
        def slack_handler(alert):
            slack_notifications.append(alert)
        
        alert_manager.add_notification_handler("email", email_handler)
        alert_manager.add_notification_handler("slack", slack_handler)
        
        def test_rule(metrics):
            return True  # Always trigger
        
        alert_manager.register_rule("test_alert", test_rule, 
                                   notifications=["email", "slack"])
        
        alert_manager.evaluate_rules({"test": "data"})
        
        assert len(email_notifications) == 1
        assert len(slack_notifications) == 1
    
    def test_alert_manager_alert_suppression(self):
        """Test alert suppression to prevent spam."""
        alert_manager = AlertManager(suppression_window=60)  # 60 seconds
        
        def always_trigger_rule(metrics):
            return True
        
        alert_manager.register_rule("spam_alert", always_trigger_rule)
        
        # First evaluation should trigger
        alerts1 = alert_manager.evaluate_rules({"test": "data"})
        assert len(alerts1) == 1
        
        # Immediate second evaluation should be suppressed
        alerts2 = alert_manager.evaluate_rules({"test": "data"})
        assert len(alerts2) == 0  # Suppressed
        
        # Mock time passing
        with patch('time.time', return_value=time.time() + 120):
            alerts3 = alert_manager.evaluate_rules({"test": "data"})
            assert len(alerts3) == 1  # Should trigger again


@pytest.mark.unit
class TestSecretManager:
    """Test secret management functionality."""
    
    def test_secret_manager_basic_operations(self, temp_dir):
        """Test basic secret storage and retrieval."""
        secret_file = temp_dir / "secrets.json"
        secret_manager = SecretManager(str(secret_file))
        
        # Store secrets
        secret_manager.set_secret("api_key", "secret123")
        secret_manager.set_secret("database_password", "db_pass456")
        
        # Retrieve secrets
        assert secret_manager.get_secret("api_key") == "secret123"
        assert secret_manager.get_secret("database_password") == "db_pass456"
        
        # Test non-existent secret
        with pytest.raises(KeyError):
            secret_manager.get_secret("nonexistent")
    
    def test_secret_manager_encryption(self, temp_dir):
        """Test secret encryption and decryption."""
        secret_file = temp_dir / "secrets.enc"
        encryption_key = b'test_key_32_bytes_long_padding!!'
        
        secret_manager = SecretManager(str(secret_file), encryption_key=encryption_key)
        
        secret_manager.set_secret("encrypted_secret", "sensitive_data")
        
        # File should contain encrypted data, not plain text
        file_content = secret_file.read_text()
        assert "sensitive_data" not in file_content
        
        # Should be able to retrieve decrypted secret
        assert secret_manager.get_secret("encrypted_secret") == "sensitive_data"
    
    def test_secret_manager_environment_variables(self, monkeypatch):
        """Test loading secrets from environment variables."""
        monkeypatch.setenv("AHGD_SECRET_API_KEY", "env_api_key")
        monkeypatch.setenv("AHGD_SECRET_DB_PASS", "env_db_pass")
        
        secret_manager = SecretManager(load_from_env=True)
        
        assert secret_manager.get_secret("API_KEY") == "env_api_key"
        assert secret_manager.get_secret("DB_PASS") == "env_db_pass"
    
    def test_secret_manager_rotation(self, temp_dir):
        """Test secret rotation functionality."""
        secret_file = temp_dir / "secrets.json"
        secret_manager = SecretManager(str(secret_file))
        
        # Set initial secret
        secret_manager.set_secret("rotating_key", "old_value")
        assert secret_manager.get_secret("rotating_key") == "old_value"
        
        # Rotate secret
        secret_manager.rotate_secret("rotating_key", "new_value")
        assert secret_manager.get_secret("rotating_key") == "new_value"
        
        # Should be able to get previous value
        previous = secret_manager.get_previous_secret("rotating_key")
        assert previous == "old_value"
    
    def test_secret_manager_expiration(self, temp_dir):
        """Test secret expiration functionality."""
        secret_file = temp_dir / "secrets.json"
        secret_manager = SecretManager(str(secret_file))
        
        # Set secret with expiration
        from datetime import datetime, timedelta
        expiry = datetime.now() + timedelta(seconds=1)
        
        secret_manager.set_secret("expiring_secret", "temporary_value", expires_at=expiry)
        
        # Should be accessible immediately
        assert secret_manager.get_secret("expiring_secret") == "temporary_value"
        
        # Mock time passing
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = expiry + timedelta(seconds=1)
            
            with pytest.raises(KeyError):
                secret_manager.get_secret("expiring_secret")


@pytest.mark.unit
class TestCryptographyFunctions:
    """Test cryptography utility functions."""
    
    def test_encrypt_decrypt_data(self):
        """Test data encryption and decryption."""
        key = b'test_encryption_key_32_bytes_long!'
        original_data = "sensitive information"
        
        encrypted = encrypt_data(original_data, key)
        decrypted = decrypt_data(encrypted, key)
        
        assert encrypted != original_data
        assert decrypted == original_data
    
    def test_encrypt_decrypt_binary_data(self):
        """Test encryption and decryption of binary data."""
        key = b'test_encryption_key_32_bytes_long!'
        original_data = b"binary\x00\x01\x02\xff data"
        
        encrypted = encrypt_data(original_data, key)
        decrypted = decrypt_data(encrypted, key)
        
        assert encrypted != original_data
        assert decrypted == original_data
    
    def test_encrypt_decrypt_large_data(self):
        """Test encryption and decryption of large data."""
        key = b'test_encryption_key_32_bytes_long!'
        original_data = "A" * 10000  # 10KB of data
        
        encrypted = encrypt_data(original_data, key)
        decrypted = decrypt_data(encrypted, key)
        
        assert len(encrypted) > len(original_data)  # Should be larger due to encryption
        assert decrypted == original_data
    
    def test_encrypt_with_wrong_key(self):
        """Test decryption with wrong key fails."""
        key1 = b'test_encryption_key_32_bytes_long!'
        key2 = b'different_key_32_bytes_long_pad!!'
        original_data = "sensitive information"
        
        encrypted = encrypt_data(original_data, key1)
        
        with pytest.raises(Exception):  # Should raise decryption error
            decrypt_data(encrypted, key2)
    
    def test_hash_data_consistency(self):
        """Test data hashing consistency."""
        data = "test data for hashing"
        
        hash1 = hashlib.sha256(data.encode()).hexdigest()
        hash2 = hashlib.sha256(data.encode()).hexdigest()
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64-character hex string
    
    def test_hash_different_data(self):
        """Test that different data produces different hashes."""
        data1 = "test data 1"
        data2 = "test data 2"
        
        hash1 = hashlib.sha256(data1.encode()).hexdigest()
        hash2 = hashlib.sha256(data2.encode()).hexdigest()
        
        assert hash1 != hash2


@pytest.mark.unit
class TestUtilityHelpers:
    """Test miscellaneous utility helper functions."""
    
    def test_file_size_formatting(self):
        """Test file size formatting utility."""
        def format_file_size(size_bytes):
            """Format file size in human-readable format."""
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size_bytes < 1024.0:
                    return f"{size_bytes:.1f} {unit}"
                size_bytes /= 1024.0
            return f"{size_bytes:.1f} PB"
        
        assert format_file_size(512) == "512.0 B"
        assert format_file_size(1536) == "1.5 KB"
        assert format_file_size(1048576) == "1.0 MB"
        assert format_file_size(1073741824) == "1.0 GB"
    
    def test_duration_formatting(self):
        """Test duration formatting utility."""
        def format_duration(seconds):
            """Format duration in human-readable format."""
            if seconds < 60:
                return f"{seconds:.1f} seconds"
            elif seconds < 3600:
                return f"{seconds/60:.1f} minutes"
            elif seconds < 86400:
                return f"{seconds/3600:.1f} hours"
            else:
                return f"{seconds/86400:.1f} days"
        
        assert format_duration(30) == "30.0 seconds"
        assert format_duration(150) == "2.5 minutes"
        assert format_duration(7200) == "2.0 hours"
        assert format_duration(172800) == "2.0 days"
    
    def test_safe_division(self):
        """Test safe division utility."""
        def safe_divide(numerator, denominator, default=0):
            """Safely divide two numbers, returning default for division by zero."""
            try:
                return numerator / denominator
            except ZeroDivisionError:
                return default
        
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0
        assert safe_divide(10, 0, default=-1) == -1
    
    def test_deep_merge_dictionaries(self):
        """Test deep merging of dictionaries."""
        def deep_merge(dict1, dict2):
            """Deep merge two dictionaries."""
            result = dict1.copy()
            for key, value in dict2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        dict1 = {"a": 1, "b": {"x": 1, "y": 2}}
        dict2 = {"b": {"z": 3}, "c": 4}
        
        merged = deep_merge(dict1, dict2)
        
        assert merged["a"] == 1
        assert merged["b"]["x"] == 1
        assert merged["b"]["y"] == 2
        assert merged["b"]["z"] == 3
        assert merged["c"] == 4
    
    def test_retry_decorator(self):
        """Test retry decorator utility."""
        def retry(max_attempts=3, delay=0.1):
            """Decorator for retrying failed operations."""
            def decorator(func):
                def wrapper(*args, **kwargs):
                    last_exception = None
                    for attempt in range(max_attempts):
                        try:
                            return func(*args, **kwargs)
                        except Exception as e:
                            last_exception = e
                            if attempt < max_attempts - 1:
                                import time
                                time.sleep(delay)
                            continue
                    raise last_exception
                return wrapper
            return decorator
        
        call_count = 0
        
        @retry(max_attempts=3, delay=0.01)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = flaky_function()
        
        assert result == "success"
        assert call_count == 3
    
    def test_batch_processor(self):
        """Test batch processing utility."""
        def process_in_batches(items, batch_size, processor):
            """Process items in batches."""
            results = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                batch_result = processor(batch)
                results.extend(batch_result)
            return results
        
        def double_items(batch):
            return [item * 2 for item in batch]
        
        items = list(range(10))
        results = process_in_batches(items, batch_size=3, processor=double_items)
        
        expected = [item * 2 for item in items]
        assert results == expected
    
    def test_progress_tracker(self):
        """Test progress tracking utility."""
        class ProgressTracker:
            def __init__(self, total):
                self.total = total
                self.current = 0
                self.start_time = datetime.now()
            
            def update(self, increment=1):
                self.current += increment
            
            def get_progress(self):
                if self.total == 0:
                    return 1.0
                return min(self.current / self.total, 1.0)
            
            def get_eta(self):
                if self.current == 0:
                    return None
                
                elapsed = (datetime.now() - self.start_time).total_seconds()
                rate = self.current / elapsed
                remaining = self.total - self.current
                
                if rate == 0:
                    return None
                
                return remaining / rate
        
        tracker = ProgressTracker(100)
        
        assert tracker.get_progress() == 0.0
        
        tracker.update(25)
        assert tracker.get_progress() == 0.25
        
        tracker.update(75)
        assert tracker.get_progress() == 1.0


@pytest.mark.unit
class TestUtilityEdgeCases:
    """Test edge cases and error conditions for utilities."""
    
    def test_empty_data_handling(self):
        """Test handling of empty data structures."""
        def safe_max(items, default=None):
            """Safely get maximum value from iterable."""
            try:
                return max(items)
            except ValueError:
                return default
        
        assert safe_max([1, 2, 3]) == 3
        assert safe_max([]) is None
        assert safe_max([], default=0) == 0
    
    def test_none_value_handling(self):
        """Test handling of None values."""
        def safe_string_operation(value, default=""):
            """Safely perform string operations on potentially None values."""
            if value is None:
                return default
            return str(value).upper()
        
        assert safe_string_operation("hello") == "HELLO"
        assert safe_string_operation(None) == ""
        assert safe_string_operation(None, "DEFAULT") == "DEFAULT"
    
    def test_very_large_numbers(self):
        """Test handling of very large numbers."""
        def safe_percentage(part, total):
            """Safely calculate percentage."""
            if total == 0:
                return 0.0
            return (part / total) * 100
        
        large_number = 10**18
        assert safe_percentage(large_number, large_number) == 100.0
        assert safe_percentage(0, large_number) == 0.0
        assert safe_percentage(large_number, 0) == 0.0
    
    def test_unicode_handling(self):
        """Test handling of unicode strings."""
        def safe_unicode_length(text):
            """Safely get length of unicode text."""
            if text is None:
                return 0
            return len(str(text))
        
        unicode_text = "Hello 世界 🌍"
        assert safe_unicode_length(unicode_text) == 9
        assert safe_unicode_length(None) == 0
        assert safe_unicode_length("") == 0
    
    def test_memory_efficient_processing(self):
        """Test memory-efficient data processing."""
        def chunked_reader(data, chunk_size):
            """Generator for reading data in chunks."""
            for i in range(0, len(data), chunk_size):
                yield data[i:i + chunk_size]
        
        large_data = list(range(1000))
        chunks = list(chunked_reader(large_data, 100))
        
        assert len(chunks) == 10
        assert len(chunks[0]) == 100
        assert len(chunks[-1]) == 100
        
        # Verify all data is preserved
        flattened = []
        for chunk in chunks:
            flattened.extend(chunk)
        assert flattened == large_data