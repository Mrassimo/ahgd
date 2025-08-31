"""
API Test Configuration and Fixtures

Shared test configuration and fixtures for API testing.
"""

import asyncio
import os
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Import API application
from src.api.main import create_app
from src.utils.logging import get_logger

logger = get_logger(__name__)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config() -> dict[str, Any]:
    """Test configuration overrides."""
    return {
        "database": {"url": "sqlite:///:memory:", "echo": False},
        "cache": {"type": "memory", "ttl": 300},
        "auth": {"enabled": False},
        "rate_limiting": {"enabled": False},
        "metrics": {"enabled": True, "update_interval": 0.1},
        "websocket": {"enabled": True, "heartbeat_interval": 1},
        "logging": {"level": "INFO", "structured": False},
    }


@pytest.fixture(scope="session")
def temp_data_dir():
    """Create temporary data directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create subdirectories
        (temp_path / "data_raw").mkdir()
        (temp_path / "data_processed").mkdir()
        (temp_path / "outputs").mkdir()
        (temp_path / "metrics").mkdir()
        (temp_path / "logs").mkdir()

        yield temp_path


@pytest.fixture(scope="session")
def app(test_config: dict[str, Any], temp_data_dir: Path) -> FastAPI:
    """Create FastAPI test application."""
    # Set test environment variables
    os.environ["ENVIRONMENT"] = "testing"
    os.environ["DATA_ROOT"] = str(temp_data_dir)

    # Override configuration for testing
    from src.utils.config import get_config_manager

    config_manager = get_config_manager()
    for key, value in test_config.items():
        config_manager.set(key, value)

    app = create_app()
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(app)


@pytest.fixture
async def async_client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_sa1_code() -> str:
    """Sample valid SA1 code."""
    return "10101000001"


@pytest.fixture
def sample_quality_metrics() -> dict[str, Any]:
    """Sample quality metrics data."""
    return {
        "completeness_rate": 98.5,
        "accuracy_score": 94.2,
        "consistency_score": 96.8,
        "timeliness_score": 92.0,
        "overall_score": 95.4,
        "record_count": 15000,
        "error_count": 125,
        "warning_count": 45,
    }


@pytest.fixture
def sample_validation_result() -> dict[str, Any]:
    """Sample validation result data."""
    return {
        "rule_name": "sa1_code_format",
        "rule_type": "schema",
        "status": "passed",
        "severity": "error",
        "records_tested": 1000,
        "records_passed": 995,
        "records_failed": 5,
        "success_rate": 99.5,
        "message": "SA1 codes format validation",
        "details": {
            "expected_format": "11-digit numeric string",
            "common_errors": ["10-digit codes", "non-numeric characters"],
        },
    }


@pytest.fixture
def sample_pipeline_config() -> dict[str, Any]:
    """Sample pipeline configuration."""
    return {
        "name": "test_etl_pipeline",
        "stages": ["extract", "transform", "validate", "load"],
        "parameters": {
            "source": "test_data",
            "geographic_level": "sa1",
            "validation_rules": ["schema", "business", "statistical"],
            "output_formats": ["csv", "parquet", "geojson"],
        },
        "resource_limits": {"max_memory": "1GB", "max_duration": 300, "max_workers": 2},
    }


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Authentication headers for testing."""
    return {"Authorization": "Bearer test_token_123"}


@pytest.fixture
def websocket_url(app: FastAPI) -> str:
    """WebSocket URL for testing."""
    return "/ws/metrics"


@pytest.fixture
def sample_geographic_bounds() -> dict[str, float]:
    """Sample geographic boundaries."""
    return {"min_lat": -43.6345, "max_lat": -10.6681, "min_lon": 113.3389, "max_lon": 153.5697}


@pytest.fixture
def mock_data_files(temp_data_dir: Path):
    """Create mock data files for testing."""
    files = {}

    # Sample SA1 data
    sa1_data = """sa1_code,state,population,area_sqkm
10101000001,NSW,450,2.5
10101000002,NSW,380,1.8
20201000001,VIC,520,3.2"""

    sa1_file = temp_data_dir / "data_processed" / "sa1_data.csv"
    sa1_file.write_text(sa1_data)
    files["sa1_data"] = sa1_file

    # Sample health indicators
    health_data = """sa1_code,indicator,value,year
10101000001,life_expectancy,82.5,2021
10101000001,obesity_rate,28.3,2021
10101000002,life_expectancy,81.8,2021"""

    health_file = temp_data_dir / "data_processed" / "health_indicators.csv"
    health_file.write_text(health_data)
    files["health_data"] = health_file

    return files
