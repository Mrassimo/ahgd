#!/usr/bin/env python3
"""
Test runner script for AHGD testing framework.

This script provides convenient commands for running different test suites
and generating coverage reports.
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle the results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=False, capture_output=False)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n\nTest execution interrupted by user.")
        return False
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="AHGD Test Runner")
    parser.add_argument("--working", action="store_true", 
                       help="Run only the working test suites")
    parser.add_argument("--unit", action="store_true",
                       help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", 
                       help="Run integration tests only")
    parser.add_argument("--config", action="store_true",
                       help="Run configuration tests only")
    parser.add_argument("--coverage", action="store_true",
                       help="Generate coverage report")
    parser.add_argument("--no-cov", action="store_true",
                       help="Run without coverage requirements")
    parser.add_argument("--fast", action="store_true",
                       help="Run fastest subset of tests")
    parser.add_argument("--all", action="store_true",
                       help="Run all tests (may have failures)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Run with verbose output")
    
    args = parser.parse_args()
    
    # Base pytest command
    base_cmd = [sys.executable, "-m", "pytest"]
    
    # Add verbose flag if requested
    if args.verbose:
        base_cmd.append("-v")
    
    # Add no coverage flag if requested
    if args.no_cov:
        base_cmd.extend(["--cov-fail-under=0"])
    
    success = True
    
    if args.working:
        # Run only the working test suites
        cmd = base_cmd + [
            "tests/unit/test_config.py",
            "tests/unit/test_main_application.py",
            "tests/fixtures/"
        ]
        success = run_command(cmd, "Working Test Suites (Config + Main Application)")
        
    elif args.unit:
        # Run unit tests only
        cmd = base_cmd + ["-m", "unit", "tests/unit/"]
        success = run_command(cmd, "Unit Tests")
        
    elif args.integration:
        # Run integration tests only  
        cmd = base_cmd + ["-m", "integration", "tests/integration/"]
        success = run_command(cmd, "Integration Tests")
        
    elif args.config:
        # Run configuration tests only
        cmd = base_cmd + ["tests/unit/test_config.py"]
        success = run_command(cmd, "Configuration System Tests")
        
    elif args.fast:
        # Run fastest subset of tests
        cmd = base_cmd + [
            "tests/unit/test_config.py::TestEnvironmentEnum",
            "tests/unit/test_config.py::TestDatabaseConfig::test_default_values",
            "tests/unit/test_main_application.py::TestMainModule",
            "--tb=line"
        ]
        success = run_command(cmd, "Fast Test Subset")
        
    elif args.all:
        # Run all tests
        cmd = base_cmd + ["tests/"]
        success = run_command(cmd, "All Tests (Including Failing)")
        
    elif args.coverage:
        # Generate coverage report
        cmd = base_cmd + [
            "tests/unit/test_config.py",
            "tests/unit/test_main_application.py", 
            "--cov-report=html",
            "--cov-report=term-missing"
        ]
        success = run_command(cmd, "Coverage Report Generation")
        
        if success:
            print("\n" + "="*60)
            print("Coverage report generated!")
            print("View HTML report: open htmlcov/index.html")
            print("="*60)
            
    else:
        # Default: run working tests
        cmd = base_cmd + [
            "tests/unit/test_config.py",
            "tests/unit/test_main_application.py",
            "--tb=short"
        ]
        success = run_command(cmd, "Default Test Suite (Working Tests)")
    
    # Print summary
    print(f"\n{'='*60}")
    if success:
        print("✅ Tests completed successfully!")
    else:
        print("❌ Some tests failed or were interrupted.")
        
    print("\nTest Options:")
    print("  python run_tests.py --working    # Working test suites only")
    print("  python run_tests.py --config     # Configuration tests only") 
    print("  python run_tests.py --fast       # Fastest subset")
    print("  python run_tests.py --coverage   # Generate coverage report")
    print("  python run_tests.py --all        # All tests (may fail)")
    print("  python run_tests.py --no-cov -v  # Verbose without coverage")
    print(f"{'='*60}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())