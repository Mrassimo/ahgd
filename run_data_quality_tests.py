#!/usr/bin/env python3
"""
Data Quality Testing Suite Runner

Comprehensive test runner for the Australian Health Analytics data quality framework.
Executes all data quality validation tests and generates a summary report.

Usage:
    python run_data_quality_tests.py [--verbose] [--coverage] [--specific-test TEST_NAME]
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import json

from loguru import logger


class DataQualityTestRunner:
    """Test runner for data quality validation framework."""
    
    def __init__(self, verbose: bool = False, coverage: bool = False):
        """Initialize test runner."""
        self.verbose = verbose
        self.coverage = coverage
        self.project_root = Path(__file__).parent
        self.test_results = {}
        
        # Configure logger
        logger.remove()
        log_level = "DEBUG" if verbose else "INFO"
        logger.add(sys.stdout, level=log_level, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}")
    
    def run_test_suite(self, specific_test: str = None) -> bool:
        """
        Run the complete data quality test suite.
        
        Args:
            specific_test: Optional specific test to run
            
        Returns:
            True if all tests pass, False otherwise
        """
        logger.info("🧪 Starting Data Quality Testing Suite")
        logger.info("=" * 60)
        
        # Test categories
        test_categories = {
            "Australian Data Standards": "tests/data_quality/test_australian_data_standards.py",
            "Schema Validation": "tests/data_quality/test_schema_validation.py", 
            "Data Quality Rules": "tests/data_quality/test_data_quality_rules.py",
            "Cross-Dataset Consistency": "tests/data_quality/test_cross_dataset_consistency.py",
            "Data Lineage Tracking": "tests/data_quality/test_data_lineage_tracking.py",
            "Privacy Compliance": "tests/data_quality/test_privacy_compliance.py"
        }
        
        overall_success = True
        
        if specific_test:
            # Run specific test
            if specific_test in test_categories:
                success = self._run_test_category(specific_test, test_categories[specific_test])
                overall_success = success
            else:
                logger.error(f"❌ Unknown test category: {specific_test}")
                logger.info(f"Available tests: {list(test_categories.keys())}")
                return False
        else:
            # Run all test categories
            for category_name, test_file in test_categories.items():
                success = self._run_test_category(category_name, test_file)
                overall_success = overall_success and success
        
        # Generate summary report
        self._generate_summary_report()
        
        if overall_success:
            logger.success("✅ All data quality tests passed!")
        else:
            logger.error("❌ Some data quality tests failed!")
        
        return overall_success
    
    def _run_test_category(self, category_name: str, test_file: str) -> bool:
        """Run tests for a specific category."""
        logger.info(f"\n📋 Running {category_name} Tests")
        logger.info("-" * 40)
        
        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            test_file,
            "-v" if self.verbose else "-q",
            "--tb=short"
        ]
        
        if self.coverage:
            cmd.extend(["--cov=tests/data_quality/validators", "--cov-report=term-missing"])
        
        # Run tests
        start_time = datetime.now()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Parse results
            success = result.returncode == 0
            
            # Store results
            self.test_results[category_name] = {
                "success": success,
                "duration": duration,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            # Log results
            if success:
                logger.success(f"✅ {category_name}: PASSED ({duration:.2f}s)")
            else:
                logger.error(f"❌ {category_name}: FAILED ({duration:.2f}s)")
                
                if self.verbose:
                    logger.error("STDOUT:")
                    logger.error(result.stdout)
                    logger.error("STDERR:")
                    logger.error(result.stderr)
                else:
                    # Show key error information
                    lines = result.stdout.split('\n')
                    error_lines = [line for line in lines if 'FAILED' in line or 'ERROR' in line]
                    for error_line in error_lines[:5]:  # Show first 5 errors
                        logger.error(f"  {error_line}")
            
            return success
            
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {category_name}: TIMEOUT (exceeded 5 minutes)")
            self.test_results[category_name] = {
                "success": False,
                "duration": 300,
                "return_code": -1,
                "stdout": "",
                "stderr": "Test timed out"
            }
            return False
        
        except Exception as e:
            logger.error(f"❌ {category_name}: ERROR - {str(e)}")
            self.test_results[category_name] = {
                "success": False,
                "duration": 0,
                "return_code": -1,
                "stdout": "",
                "stderr": str(e)
            }
            return False
    
    def _generate_summary_report(self):
        """Generate and display summary report."""
        logger.info("\n📊 Test Summary Report")
        logger.info("=" * 60)
        
        total_categories = len(self.test_results)
        passed_categories = sum(1 for result in self.test_results.values() if result["success"])
        failed_categories = total_categories - passed_categories
        total_duration = sum(result["duration"] for result in self.test_results.values())
        
        logger.info(f"Total Test Categories: {total_categories}")
        logger.info(f"Passed: {passed_categories}")
        logger.info(f"Failed: {failed_categories}")
        logger.info(f"Total Duration: {total_duration:.2f} seconds")
        logger.info(f"Success Rate: {(passed_categories / total_categories) * 100:.1f}%")
        
        # Detailed results
        logger.info("\n📝 Detailed Results:")
        for category, result in self.test_results.items():
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            logger.info(f"  {category:<30} {status:<10} ({result['duration']:.2f}s)")
        
        # Save results to file
        report_file = self.project_root / "data_quality_test_report.json"
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_categories": total_categories,
                "passed_categories": passed_categories,
                "failed_categories": failed_categories,
                "total_duration": total_duration,
                "success_rate": (passed_categories / total_categories) * 100
            },
            "detailed_results": self.test_results
        }
        
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"\n📄 Detailed report saved to: {report_file}")
    
    def run_quick_validation(self) -> bool:
        """Run a quick validation of core data quality features."""
        logger.info("🚀 Running Quick Data Quality Validation")
        logger.info("=" * 50)
        
        # Test imports
        try:
            logger.info("Testing imports...")
            from tests.data_quality.validators import (
                AustralianHealthDataValidator,
                SchemaValidator,
                AustralianHealthQualityMetrics
            )
            logger.success("✅ All validators imported successfully")
        except ImportError as e:
            logger.error(f"❌ Import error: {e}")
            return False
        
        # Test basic functionality
        try:
            logger.info("Testing basic validator functionality...")
            
            # Test Australian Health Data Validator
            validator = AustralianHealthDataValidator()
            sa2_result = validator.validate_sa2_code("101021007")
            assert sa2_result["valid"] is True
            
            # Test Schema Validator
            schema_validator = SchemaValidator()
            assert schema_validator is not None
            
            # Test Quality Metrics
            quality_metrics = AustralianHealthQualityMetrics(validator)
            assert quality_metrics is not None
            
            logger.success("✅ Basic functionality tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Functionality test error: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run data quality testing suite")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", help="Run with coverage")
    parser.add_argument("--specific-test", "-t", help="Run specific test category")
    parser.add_argument("--quick", "-q", action="store_true", help="Run quick validation only")
    
    args = parser.parse_args()
    
    runner = DataQualityTestRunner(verbose=args.verbose, coverage=args.coverage)
    
    if args.quick:
        success = runner.run_quick_validation()
    else:
        success = runner.run_test_suite(args.specific_test)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()