"""
API Test Runner

Convenience script to run different categories of API tests.
"""

import sys

import pytest


def run_unit_tests():
    """Run API unit tests."""
    return pytest.main(
        ["tests/api/unit/", "-v", "--tb=short", "--cov=src.api", "--cov-report=term-missing"]
    )


def run_integration_tests():
    """Run API integration tests."""
    return pytest.main(["tests/api/integration/", "-v", "--tb=short"])


def run_performance_tests():
    """Run API performance tests."""
    return pytest.main(["tests/api/performance/", "-v", "--tb=short", "-m", "not slow"])


def run_all_api_tests():
    """Run all API tests."""
    return pytest.main(
        [
            "tests/api/",
            "-v",
            "--tb=short",
            "--cov=src.api",
            "--cov-report=html:htmlcov/api",
            "--cov-report=term-missing",
            "-m",
            "not slow",
        ]
    )


def run_slow_tests():
    """Run slow/long-running tests."""
    return pytest.main(["tests/api/", "-v", "--tb=short", "-m", "slow"])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()

        if test_type == "unit":
            exit_code = run_unit_tests()
        elif test_type == "integration":
            exit_code = run_integration_tests()
        elif test_type == "performance":
            exit_code = run_performance_tests()
        elif test_type == "slow":
            exit_code = run_slow_tests()
        elif test_type == "all":
            exit_code = run_all_api_tests()
        else:
            print(f"Unknown test type: {test_type}")
            print("Available options: unit, integration, performance, slow, all")
            sys.exit(1)
    else:
        # Default: run all tests
        exit_code = run_all_api_tests()

    sys.exit(exit_code)
