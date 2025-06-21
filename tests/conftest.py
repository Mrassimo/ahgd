"""
Pytest configuration and fixtures for AHGD ETL pipeline tests.

This module provides comprehensive fixtures for testing all components
of the AHGD ETL pipeline, including database fixtures, mock data generators,
and test configuration management.
"""

import os
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator
from unittest.mock import Mock, MagicMock
import logging

import pytest
import pandas as pd
from pydantic import ValidationError

from src.utils.interfaces import (
    AuditTrail,
    ColumnMapping,
    DataBatch,
    DataFormat,
    DataPartition,
    DataRecord,
    ProcessingMetadata,
    ProcessingStatus,
    SourceMetadata,
    ValidationResult,
    ValidationSeverity,
)
from src.utils.config import ConfigurationManager
from src.utils.logging import get_logger
from src.extractors.base import BaseExtractor
from src.transformers.base import BaseTransformer
from src.validators.base import BaseValidator
from src.loaders.base import BaseLoader


# Test configuration constants
TEST_DATA_DIR = Path(__file__).parent / "fixtures" / "data"
TEST_CONFIG_DIR = Path(__file__).parent / "fixtures" / "config"
MOCK_SA2_CODES = ["101011001", "101011002", "101021003", "102011004", "102021005"]


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Provide the test data directory path."""
    return TEST_DATA_DIR


@pytest.fixture(scope="session")
def test_config_dir() -> Path:
    """Provide the test configuration directory path."""
    return TEST_CONFIG_DIR


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_logger() -> logging.Logger:
    """Provide a mock logger for testing."""
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.fixture
def sample_sa2_codes() -> List[str]:
    """Provide sample SA2 codes for testing."""
    return MOCK_SA2_CODES.copy()


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide sample configuration for testing."""
    return {
        "extractors": {
            "csv_extractor": {
                "batch_size": 100,
                "max_retries": 3,
                "retry_delay": 1.0,
                "checkpoint_interval": 500
            }
        },
        "transformers": {
            "sa2_transformer": {
                "batch_size": 200,
                "output_format": "parquet"
            }
        },
        "validators": {
            "schema_validator": {
                "required_columns": ["sa2_code", "value", "year"],
                "column_types": {
                    "sa2_code": "string",
                    "value": "float",
                    "year": "integer"
                },
                "business_rules": [
                    {
                        "id": "year_range_check",
                        "type": "range_check",
                        "column": "year",
                        "min": 2000,
                        "max": 2025
                    }
                ]
            }
        },
        "loaders": {
            "sqlite_loader": {
                "database_path": ":memory:",
                "table_name": "test_data",
                "batch_size": 1000
            }
        },
        "geographic": {
            "sa2_column": "sa2_code",
            "sa2_codes_file": None
        }
    }


@pytest.fixture
def sample_source_metadata() -> SourceMetadata:
    """Provide sample source metadata for testing."""
    return SourceMetadata(
        source_id="test_source_001",
        source_type="csv",
        source_file=Path("/tmp/test_data.csv"),
        last_modified=datetime.now() - timedelta(hours=1),
        file_size=1024,
        encoding="utf-8",
        delimiter=",",
        headers=["sa2_code", "value", "year"],
        row_count=1000,
        column_count=3,
        schema_version="1.0.0"
    )


@pytest.fixture
def sample_processing_metadata() -> ProcessingMetadata:
    """Provide sample processing metadata for testing."""
    return ProcessingMetadata(
        operation_id="test_op_001",
        operation_type="extraction",
        status=ProcessingStatus.RUNNING,
        start_time=datetime.now(),
        records_processed=500,
        parameters={"batch_size": 100}
    )


@pytest.fixture
def sample_validation_results() -> List[ValidationResult]:
    """Provide sample validation results for testing."""
    return [
        ValidationResult(
            is_valid=False,
            severity=ValidationSeverity.ERROR,
            rule_id="schema_missing_columns",
            message="Missing required column: sa2_code",
            details={"missing_columns": ["sa2_code"]},
            affected_records=[1, 5, 10]
        ),
        ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.WARNING,
            rule_id="statistical_outlier",
            message="Potential outlier detected in value column",
            details={"method": "iqr", "threshold": 1.5},
            affected_records=[25]
        ),
        ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            rule_id="completeness_check",
            message="Column completeness within acceptable range",
            details={"completeness": 0.95, "threshold": 0.90}
        )
    ]


@pytest.fixture
def sample_data_batch() -> DataBatch:
    """Provide sample data batch for testing."""
    return [
        {"sa2_code": "101011001", "value": 25.5, "year": 2021, "indicator": "health_index"},
        {"sa2_code": "101011002", "value": 30.2, "year": 2021, "indicator": "health_index"},
        {"sa2_code": "101021003", "value": 22.8, "year": 2021, "indicator": "health_index"},
        {"sa2_code": "102011004", "value": 28.1, "year": 2021, "indicator": "health_index"},
        {"sa2_code": "102021005", "value": 31.7, "year": 2021, "indicator": "health_index"}
    ]


@pytest.fixture
def large_data_batch() -> DataBatch:
    """Provide large data batch for performance testing."""
    batch = []
    for i in range(10000):
        sa2_code = f"10{i % 10}0{i % 100:02d}0{i % 1000:03d}"
        batch.append({
            "sa2_code": sa2_code,
            "value": 20.0 + (i % 50),
            "year": 2020 + (i % 5),
            "indicator": f"indicator_{i % 10}"
        })
    return batch


@pytest.fixture
def invalid_data_batch() -> DataBatch:
    """Provide invalid data batch for error testing."""
    return [
        {"sa2_code": None, "value": 25.5, "year": 2021},  # Missing SA2 code
        {"sa2_code": "invalid_code", "value": "not_a_number", "year": 2021},  # Invalid types
        {"sa2_code": "101011001", "value": -999, "year": 1900},  # Out of range values
        {},  # Empty record
        {"sa2_code": "101011001", "value": float('inf'), "year": 2021}  # Invalid float
    ]


@pytest.fixture
def sample_column_mappings() -> List[ColumnMapping]:
    """Provide sample column mappings for testing."""
    return [
        ColumnMapping(
            source_column="sa2_code",
            target_column="statistical_area_2",
            data_type="string",
            validation_rules=["pattern_match"],
            is_required=True
        ),
        ColumnMapping(
            source_column="value",
            target_column="health_indicator_value",
            data_type="float",
            transformation="round_2dp",
            validation_rules=["range_check"],
            is_required=True
        ),
        ColumnMapping(
            source_column="year",
            target_column="reference_year",
            data_type="integer",
            validation_rules=["range_check"],
            is_required=True
        )
    ]


@pytest.fixture
def sqlite_db(temp_dir: Path) -> Generator[sqlite3.Connection, None, None]:
    """Provide an in-memory SQLite database for testing."""
    db_path = temp_dir / "test.db"
    conn = sqlite3.connect(str(db_path))
    
    # Create test tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sa2_data (
            id INTEGER PRIMARY KEY,
            sa2_code TEXT NOT NULL,
            value REAL,
            year INTEGER,
            indicator TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY,
            operation_id TEXT NOT NULL,
            operation_type TEXT,
            status TEXT,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            records_processed INTEGER,
            error_message TEXT
        )
    """)
    
    conn.commit()
    
    yield conn
    
    conn.close()


@pytest.fixture
def mock_postgresql_config() -> Dict[str, Any]:
    """Provide mock PostgreSQL configuration for testing."""
    return {
        "host": "localhost",
        "port": 5432,
        "database": "test_ahgd",
        "username": "test_user",
        "password": "test_password",
        "schema": "public"
    }


@pytest.fixture
def mock_progress_callback() -> Mock:
    """Provide a mock progress callback for testing."""
    return Mock()


@pytest.fixture
def sample_csv_file(temp_dir: Path, sample_data_batch: DataBatch) -> Path:
    """Create a sample CSV file for testing."""
    csv_file = temp_dir / "sample_data.csv"
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(sample_data_batch)
    df.to_csv(csv_file, index=False)
    
    return csv_file


@pytest.fixture
def sample_json_file(temp_dir: Path, sample_data_batch: DataBatch) -> Path:
    """Create a sample JSON file for testing."""
    import json
    
    json_file = temp_dir / "sample_data.json"
    
    with open(json_file, 'w') as f:
        json.dump(sample_data_batch, f, indent=2)
    
    return json_file


@pytest.fixture
def sample_excel_file(temp_dir: Path, sample_data_batch: DataBatch) -> Path:
    """Create a sample Excel file for testing."""
    excel_file = temp_dir / "sample_data.xlsx"
    
    # Convert to DataFrame and save as Excel
    df = pd.DataFrame(sample_data_batch)
    df.to_excel(excel_file, index=False)
    
    return excel_file


@pytest.fixture
def mock_extractor(sample_config: Dict[str, Any], mock_logger: logging.Logger) -> Mock:
    """Provide a mock extractor for testing."""
    extractor = Mock(spec=BaseExtractor)
    extractor.extractor_id = "test_extractor"
    extractor.config = sample_config["extractors"]["csv_extractor"]
    extractor.logger = mock_logger
    extractor.batch_size = 100
    extractor.max_retries = 3
    
    # Mock methods
    extractor.extract.return_value = iter([])
    extractor.get_source_metadata.return_value = Mock(spec=SourceMetadata)
    extractor.validate_source.return_value = True
    
    return extractor


@pytest.fixture
def mock_transformer(sample_config: Dict[str, Any], mock_logger: logging.Logger) -> Mock:
    """Provide a mock transformer for testing."""
    transformer = Mock(spec=BaseTransformer)
    transformer.transformer_id = "test_transformer"
    transformer.config = sample_config["transformers"]["sa2_transformer"]
    transformer.logger = mock_logger
    
    # Mock methods
    transformer.transform.return_value = iter([])
    transformer.get_schema.return_value = {}
    transformer.validate_schema.return_value = True
    
    return transformer


@pytest.fixture
def mock_validator(sample_config: Dict[str, Any], mock_logger: logging.Logger) -> Mock:
    """Provide a mock validator for testing."""
    validator = Mock(spec=BaseValidator)
    validator.validator_id = "test_validator"
    validator.config = sample_config["validators"]["schema_validator"]
    validator.logger = mock_logger
    
    # Mock methods
    validator.validate.return_value = []
    validator.get_validation_rules.return_value = ["schema_check", "business_rules"]
    validator.validate_comprehensive.return_value = []
    
    return validator


@pytest.fixture
def mock_loader(sample_config: Dict[str, Any], mock_logger: logging.Logger) -> Mock:
    """Provide a mock loader for testing."""
    loader = Mock(spec=BaseLoader)
    loader.loader_id = "test_loader"
    loader.config = sample_config["loaders"]["sqlite_loader"]
    loader.logger = mock_logger
    
    # Mock methods
    loader.load.return_value = Mock()
    loader.get_output_metadata.return_value = {}
    loader.validate_output.return_value = True
    
    return loader


@pytest.fixture
def config_manager(temp_dir: Path) -> ConfigurationManager:
    """Provide a ConfigurationManager instance for testing."""
    config_file = temp_dir / "test_config.yaml"
    
    # Create a basic config file
    config_content = """
logging:
  level: DEBUG
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

database:
  type: sqlite
  path: ":memory:"

validation:
  strict_mode: true
  fail_on_error: false
"""
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    # Create config directory structure expected by ConfigurationManager
    config_dir = temp_dir / "configs"
    config_dir.mkdir(exist_ok=True)
    
    # Create default.yaml in the configs directory
    default_config = config_dir / "default.yaml"
    with open(default_config, 'w') as f:
        f.write(config_content)
    
    return ConfigurationManager(config_dir=config_dir, environment="testing")


@pytest.fixture
def audit_trail(
    sample_source_metadata: SourceMetadata,
    sample_processing_metadata: ProcessingMetadata,
    sample_validation_results: List[ValidationResult]
) -> AuditTrail:
    """Provide sample audit trail for testing."""
    return AuditTrail(
        operation_id="test_audit_001",
        operation_type="full_pipeline",
        source_metadata=sample_source_metadata,
        processing_metadata=sample_processing_metadata,
        validation_results=sample_validation_results,
        output_metadata={"records_exported": 1000, "format": "parquet"}
    )


# Performance testing fixtures
@pytest.fixture
def performance_data_small() -> DataBatch:
    """Provide small dataset for performance baseline."""
    return [{"sa2_code": f"10101{i:04d}", "value": i * 0.5} for i in range(1000)]


@pytest.fixture
def performance_data_medium() -> DataBatch:
    """Provide medium dataset for performance testing."""
    return [{"sa2_code": f"10101{i:04d}", "value": i * 0.5} for i in range(10000)]


@pytest.fixture
def performance_data_large() -> DataBatch:
    """Provide large dataset for performance testing."""
    return [{"sa2_code": f"10101{i:04d}", "value": i * 0.5} for i in range(100000)]


# Error simulation fixtures
@pytest.fixture
def flaky_extractor(mock_extractor: Mock) -> Mock:
    """Provide an extractor that fails intermittently."""
    def flaky_extract(*args, **kwargs):
        if not hasattr(flaky_extract, 'call_count'):
            flaky_extract.call_count = 0
        flaky_extract.call_count += 1
        
        if flaky_extract.call_count <= 2:  # Fail first two attempts
            raise Exception(f"Simulated failure #{flaky_extract.call_count}")
        
        return iter([])  # Success on third attempt
    
    mock_extractor.extract.side_effect = flaky_extract
    return mock_extractor


@pytest.fixture
def memory_intensive_data() -> DataBatch:
    """Provide data that simulates memory-intensive operations."""
    return [
        {
            "sa2_code": f"10101{i:04d}",
            "large_text_field": "x" * 10000,  # 10KB per record
            "nested_data": {"key": list(range(1000))},
            "value": i * 0.5
        }
        for i in range(1000)  # ~10MB total
    ]


# Schema testing fixtures
@pytest.fixture
def schema_evolution_data() -> List[Dict[str, Any]]:
    """Provide data representing schema evolution scenarios."""
    return [
        {
            "version": "1.0",
            "schema": {
                "sa2_code": {"type": "string", "required": True},
                "value": {"type": "float", "required": True}
            },
            "data": [{"sa2_code": "101011001", "value": 25.5}]
        },
        {
            "version": "1.1",
            "schema": {
                "sa2_code": {"type": "string", "required": True},
                "value": {"type": "float", "required": True},
                "year": {"type": "integer", "required": True, "default": 2021}
            },
            "data": [{"sa2_code": "101011001", "value": 25.5, "year": 2021}]
        },
        {
            "version": "2.0",
            "schema": {
                "statistical_area_2": {"type": "string", "required": True},  # Renamed column
                "health_indicator_value": {"type": "float", "required": True},  # Renamed column
                "reference_year": {"type": "integer", "required": True},  # Renamed column
                "indicator_type": {"type": "string", "required": True}  # New required column
            },
            "data": [{"statistical_area_2": "101011001", "health_indicator_value": 25.5, "reference_year": 2021, "indicator_type": "mortality"}]
        }
    ]


# Geographic data fixtures
@pytest.fixture
def sample_geographic_data() -> DataBatch:
    """Provide sample geographic data for testing."""
    return [
        {
            "sa2_code": "101011001",
            "sa2_name": "Sydney - Haymarket - The Rocks",
            "latitude": -33.8688,
            "longitude": 151.2093,
            "state": "NSW",
            "area_sqkm": 2.5
        },
        {
            "sa2_code": "201011002",
            "sa2_name": "Melbourne - Southbank",
            "latitude": -37.8136,
            "longitude": 144.9631,
            "state": "VIC",
            "area_sqkm": 1.8
        },
        {
            "sa2_code": "301011003",
            "sa2_name": "Brisbane - CBD",
            "latitude": -27.4698,
            "longitude": 153.0251,
            "state": "QLD",
            "area_sqkm": 3.2
        }
    ]


# Health data fixtures
@pytest.fixture
def sample_health_indicators() -> DataBatch:
    """Provide sample health indicator data for testing."""
    return [
        {
            "sa2_code": "101011001",
            "mortality_rate": 5.2,
            "birth_rate": 12.8,
            "life_expectancy": 82.5,
            "diabetes_prevalence": 6.1,
            "obesity_rate": 24.3,
            "year": 2021
        },
        {
            "sa2_code": "101011002",
            "mortality_rate": 4.8,
            "birth_rate": 14.2,
            "life_expectancy": 83.1,
            "diabetes_prevalence": 5.9,
            "obesity_rate": 22.7,
            "year": 2021
        }
    ]


# Census data fixtures
@pytest.fixture
def sample_census_data() -> DataBatch:
    """Provide sample census data for testing."""
    return [
        {
            "sa2_code": "101011001",
            "total_population": 15420,
            "median_age": 34.5,
            "median_income": 68000,
            "unemployment_rate": 4.2,
            "indigenous_population": 156,
            "overseas_born": 7832,
            "year": 2021
        },
        {
            "sa2_code": "101011002",
            "total_population": 12890,
            "median_age": 36.8,
            "median_income": 72000,
            "unemployment_rate": 3.8,
            "indigenous_population": 98,
            "overseas_born": 6445,
            "year": 2021
        }
    ]


# SEIFA data fixtures
@pytest.fixture
def sample_seifa_data() -> DataBatch:
    """Provide sample SEIFA data for testing."""
    return [
        {
            "sa2_code": "101011001",
            "irsad_score": 1050,
            "irsad_decile": 9,
            "irsad_rank": 156,
            "irsd_score": 980,
            "irsd_decile": 8,
            "ier_score": 1120,
            "ier_decile": 10,
            "ieo_score": 1080,
            "ieo_decile": 9,
            "year": 2021
        }
    ]


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Automatically set up test environment for all tests."""
    # Set test environment variables
    monkeypatch.setenv("AHGD_ENV", "test")
    monkeypatch.setenv("AHGD_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("AHGD_DISABLE_CACHE", "true")
    
    # Disable external API calls during testing
    monkeypatch.setenv("AHGD_OFFLINE_MODE", "true")


@pytest.fixture
def cleanup_test_data():
    """Fixture to clean up test data after test completion."""
    created_files = []
    created_dirs = []
    
    yield created_files, created_dirs
    
    # Cleanup files
    for file_path in created_files:
        if file_path.exists():
            file_path.unlink()
    
    # Cleanup directories
    for dir_path in created_dirs:
        if dir_path.exists():
            import shutil
            shutil.rmtree(dir_path)


# Parametrisation helpers
def pytest_generate_tests(metafunc):
    """Generate parametrised tests based on test function parameters."""
    
    # Parametrise data format tests
    if "data_format" in metafunc.fixturenames:
        metafunc.parametrize("data_format", [
            DataFormat.CSV,
            DataFormat.JSON,
            DataFormat.PARQUET,
            DataFormat.XLSX
        ])
    
    # Parametrise validation severity tests
    if "validation_severity" in metafunc.fixturenames:
        metafunc.parametrize("validation_severity", [
            ValidationSeverity.ERROR,
            ValidationSeverity.WARNING,
            ValidationSeverity.INFO
        ])
    
    # Parametrise processing status tests
    if "processing_status" in metafunc.fixturenames:
        metafunc.parametrize("processing_status", [
            ProcessingStatus.PENDING,
            ProcessingStatus.RUNNING,
            ProcessingStatus.COMPLETED,
            ProcessingStatus.FAILED,
            ProcessingStatus.CANCELLED
        ])


# Custom markers for test categorisation
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "database: mark test as requiring database"
    )
    config.addinivalue_line(
        "markers", "external: mark test as requiring external resources"
    )
    config.addinivalue_line(
        "markers", "memory_intensive: mark test as memory intensive"
    )