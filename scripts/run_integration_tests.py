#!/usr/bin/env python3
"""
Integration Test Runner for Australian Health Analytics Platform

Comprehensive test runner for Phase 5.2 integration testing framework.
Executes complete pipeline validation with real Australian health data patterns.

Usage:
    python run_integration_tests.py [options]

Options:
    --module MODULE     Run specific test module (complete_pipeline, cross_component, etc.)
    --performance       Run performance tests only
    --quick             Run quick validation tests (reduced data volumes)
    --scale SCALE       Set test scale: small, medium, large, production
    --workers N         Number of parallel test workers
    --coverage          Generate coverage report
    --benchmark         Run performance benchmarking
    --verbose           Enable verbose output
    --debug             Enable debug logging
    --report-dir DIR    Directory for test reports
    --help              Show this help message

Examples:
    python run_integration_tests.py --quick --verbose
    python run_integration_tests.py --module complete_pipeline --coverage
    python run_integration_tests.py --performance --benchmark --scale large
    python run_integration_tests.py --workers 4 --report-dir ./test_reports
"""

import os
import sys
import argparse
import subprocess
import time
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

class IntegrationTestRunner:
    """Comprehensive integration test runner for Australian Health Analytics platform."""
    
    def __init__(self):
        self.start_time = time.time()
        self.test_results = {}
        self.performance_metrics = {}
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging for test runner."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'integration_test_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="Integration Test Runner for Australian Health Analytics Platform",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__.split('Usage:')[1] if 'Usage:' in __doc__ else ""
        )
        
        # Test selection options
        parser.add_argument('--module', 
                          choices=['complete_pipeline', 'cross_component', 'data_lake_operations', 
                                 'real_data_processing', 'performance_integration', 'concurrent_operations', 
                                 'error_recovery'],
                          help='Run specific test module')
        
        parser.add_argument('--performance', action='store_true',
                          help='Run performance tests only')
        
        parser.add_argument('--quick', action='store_true',
                          help='Run quick validation tests (reduced data volumes)')
        
        # Test configuration options
        parser.add_argument('--scale', 
                          choices=['small', 'medium', 'large', 'production'],
                          default='medium',
                          help='Set test scale (default: medium)')
        
        parser.add_argument('--workers', type=int, default=1,
                          help='Number of parallel test workers (default: 1)')
        
        # Output options
        parser.add_argument('--coverage', action='store_true',
                          help='Generate coverage report')
        
        parser.add_argument('--benchmark', action='store_true',
                          help='Run performance benchmarking')
        
        parser.add_argument('--verbose', action='store_true',
                          help='Enable verbose output')
        
        parser.add_argument('--debug', action='store_true',
                          help='Enable debug logging')
        
        parser.add_argument('--report-dir', type=str, default='./integration_test_reports',
                          help='Directory for test reports (default: ./integration_test_reports)')
        
        return parser.parse_args()
    
    def validate_environment(self) -> bool:
        """Validate test environment and dependencies."""
        self.logger.info("Validating test environment...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.logger.error("Python 3.8+ required for integration tests")
            return False
        
        # Check required packages
        required_packages = [
            'pytest', 'polars', 'numpy', 'psutil', 'openpyxl'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.logger.error(f"Missing required packages: {missing_packages}")
            self.logger.info("Install with: pip install " + " ".join(missing_packages))
            return False
        
        # Check test directory structure
        test_dir = Path('tests/integration')
        if not test_dir.exists():
            self.logger.error("Integration test directory not found: tests/integration")
            return False
        
        required_test_files = [
            'test_complete_pipeline.py',
            'test_cross_component_integration.py',
            'test_data_lake_operations.py',
            'test_real_data_processing.py',
            'test_performance_integration.py',
            'test_concurrent_operations.py',
            'test_error_recovery.py'
        ]
        
        missing_files = []
        for test_file in required_test_files:
            if not (test_dir / test_file).exists():
                missing_files.append(test_file)
        
        if missing_files:
            self.logger.error(f"Missing test files: {missing_files}")
            return False
        
        # Create report directory
        report_dir = Path(self.args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Environment validation successful")
        return True
    
    def configure_test_environment(self):
        """Configure test environment variables and settings."""
        self.logger.info(f"Configuring test environment for scale: {self.args.scale}")
        
        # Set test scale environment variables
        scale_configs = {
            'small': {
                'TEST_SA2_AREAS': '100',
                'TEST_HEALTH_RECORDS': '1000',
                'TEST_TIMEOUT': '60'
            },
            'medium': {
                'TEST_SA2_AREAS': '500',
                'TEST_HEALTH_RECORDS': '5000',
                'TEST_TIMEOUT': '120'
            },
            'large': {
                'TEST_SA2_AREAS': '1500',
                'TEST_HEALTH_RECORDS': '25000',
                'TEST_TIMEOUT': '300'
            },
            'production': {
                'TEST_SA2_AREAS': '2454',
                'TEST_HEALTH_RECORDS': '50000',
                'TEST_TIMEOUT': '600'
            }
        }
        
        config = scale_configs[self.args.scale]
        for key, value in config.items():
            os.environ[key] = value
        
        # Set other test environment variables
        os.environ['PYTEST_CURRENT_TEST'] = 'integration'
        os.environ['PYTHONPATH'] = f"{os.environ.get('PYTHONPATH', '')}:{os.getcwd()}"
        
        if self.args.debug:
            os.environ['PYTEST_LOG_LEVEL'] = 'DEBUG'
            logging.getLogger().setLevel(logging.DEBUG)
    
    def build_pytest_command(self) -> List[str]:
        """Build pytest command with appropriate options."""
        cmd = ['python', '-m', 'pytest']
        
        # Test selection
        if self.args.module:
            cmd.append(f'tests/integration/test_{self.args.module}.py')
        elif self.args.performance:
            cmd.extend([
                'tests/integration/test_performance_integration.py',
                'tests/integration/test_concurrent_operations.py'
            ])
        else:
            cmd.append('tests/integration/')
        
        # Parallel execution
        if self.args.workers > 1:
            try:
                import pytest_xdist
                cmd.extend(['-n', str(self.args.workers)])
            except ImportError:
                self.logger.warning("pytest-xdist not available, running sequentially")
        
        # Output options
        if self.args.verbose:
            cmd.append('-v')
        
        if self.args.debug:
            cmd.extend(['-s', '--log-cli-level=DEBUG'])
        
        # Coverage reporting
        if self.args.coverage:
            cmd.extend([
                '--cov=src',
                '--cov-report=html',
                '--cov-report=term-missing',
                f'--cov-report=xml:{self.args.report_dir}/coverage.xml'
            ])
        
        # Benchmarking
        if self.args.benchmark:
            try:
                import pytest_benchmark
                cmd.extend([
                    '--benchmark-only',
                    f'--benchmark-json={self.args.report_dir}/benchmark.json'
                ])
            except ImportError:
                self.logger.warning("pytest-benchmark not available, skipping benchmarks")
        
        # Test reporting
        cmd.extend([
            '--tb=short',
            f'--html={self.args.report_dir}/integration_test_report.html',
            '--self-contained-html'
        ])
        
        return cmd
    
    def run_integration_tests(self) -> bool:
        """Execute integration tests and capture results."""
        self.logger.info("Starting integration test execution...")
        
        cmd = self.build_pytest_command()
        self.logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            # Run tests
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=int(os.environ.get('TEST_TIMEOUT', '300'))
            )
            
            # Capture results
            self.test_results = {
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': time.time() - self.start_time,
                'success': result.returncode == 0
            }
            
            # Log results
            if result.returncode == 0:
                self.logger.info("Integration tests completed successfully")
            else:
                self.logger.error(f"Integration tests failed with return code: {result.returncode}")
            
            if self.args.verbose or result.returncode != 0:
                self.logger.info(f"STDOUT:\n{result.stdout}")
                if result.stderr:
                    self.logger.error(f"STDERR:\n{result.stderr}")
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Tests timed out after {os.environ.get('TEST_TIMEOUT', '300')} seconds")
            return False
        
        except Exception as e:
            self.logger.error(f"Error running tests: {e}")
            return False
    
    def generate_performance_metrics(self):
        """Generate performance metrics from test results."""
        self.logger.info("Generating performance metrics...")
        
        # Parse pytest output for performance data
        output = self.test_results.get('stdout', '')
        
        self.performance_metrics = {
            'execution_time': self.test_results.get('execution_time', 0),
            'test_scale': self.args.scale,
            'parallel_workers': self.args.workers,
            'timestamp': datetime.now().isoformat(),
            'success': self.test_results.get('success', False)
        }
        
        # Extract test timing information
        timing_data = {}
        for line in output.split('\n'):
            if 'PASSED' in line and '::' in line:
                parts = line.split('::')
                if len(parts) >= 2:
                    test_name = parts[-1].split()[0]
                    # Extract timing if available
                    if '[' in line and 's]' in line:
                        timing_str = line.split('[')[-1].split('s]')[0]
                        try:
                            timing_data[test_name] = float(timing_str)
                        except ValueError:
                            pass
        
        self.performance_metrics['test_timings'] = timing_data
        
        # Calculate summary statistics
        if timing_data:
            timings = list(timing_data.values())
            self.performance_metrics['summary'] = {
                'total_tests': len(timings),
                'average_test_time': sum(timings) / len(timings),
                'longest_test_time': max(timings),
                'shortest_test_time': min(timings)
            }
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        self.logger.info("Generating test report...")
        
        report_dir = Path(self.args.report_dir)
        
        # Generate summary report
        summary_report = {
            'execution_summary': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_execution_time': time.time() - self.start_time,
                'success': self.test_results.get('success', False),
                'return_code': self.test_results.get('return_code', -1)
            },
            'test_configuration': {
                'test_scale': self.args.scale,
                'parallel_workers': self.args.workers,
                'module_filter': self.args.module,
                'performance_only': self.args.performance,
                'quick_mode': self.args.quick,
                'coverage_enabled': self.args.coverage,
                'benchmark_enabled': self.args.benchmark
            },
            'environment_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'working_directory': os.getcwd(),
                'test_sa2_areas': os.environ.get('TEST_SA2_AREAS', 'N/A'),
                'test_health_records': os.environ.get('TEST_HEALTH_RECORDS', 'N/A')
            },
            'performance_metrics': self.performance_metrics,
            'test_results': self.test_results
        }
        
        # Save JSON report
        json_report_path = report_dir / 'integration_test_summary.json'
        with open(json_report_path, 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)
        
        # Generate markdown report
        self.generate_markdown_report(summary_report, report_dir)
        
        self.logger.info(f"Test reports generated in: {report_dir}")
        self.logger.info(f"Summary report: {json_report_path}")
    
    def generate_markdown_report(self, summary: Dict[str, Any], report_dir: Path):
        """Generate markdown test report."""
        
        markdown_content = f"""# Integration Test Report
        
## Execution Summary

- **Start Time**: {summary['execution_summary']['start_time']}
- **End Time**: {summary['execution_summary']['end_time']}
- **Total Execution Time**: {summary['execution_summary']['total_execution_time']:.2f} seconds
- **Success**: {'✅ PASSED' if summary['execution_summary']['success'] else '❌ FAILED'}
- **Return Code**: {summary['execution_summary']['return_code']}

## Test Configuration

- **Test Scale**: {summary['test_configuration']['test_scale']}
- **Parallel Workers**: {summary['test_configuration']['parallel_workers']}
- **Module Filter**: {summary['test_configuration']['module_filter'] or 'All modules'}
- **Performance Only**: {'Yes' if summary['test_configuration']['performance_only'] else 'No'}
- **Quick Mode**: {'Yes' if summary['test_configuration']['quick_mode'] else 'No'}
- **Coverage Enabled**: {'Yes' if summary['test_configuration']['coverage_enabled'] else 'No'}
- **Benchmark Enabled**: {'Yes' if summary['test_configuration']['benchmark_enabled'] else 'No'}

## Environment Information

- **Python Version**: {summary['environment_info']['python_version']}
- **Platform**: {summary['environment_info']['platform']}
- **Test SA2 Areas**: {summary['environment_info']['test_sa2_areas']}
- **Test Health Records**: {summary['environment_info']['test_health_records']}

## Performance Metrics

- **Execution Time**: {summary['performance_metrics'].get('execution_time', 0):.2f} seconds
- **Test Scale**: {summary['performance_metrics'].get('test_scale', 'N/A')}
"""
        
        if 'summary' in summary['performance_metrics']:
            perf_summary = summary['performance_metrics']['summary']
            markdown_content += f"""
- **Total Tests**: {perf_summary['total_tests']}
- **Average Test Time**: {perf_summary['average_test_time']:.2f} seconds
- **Longest Test**: {perf_summary['longest_test_time']:.2f} seconds
- **Shortest Test**: {perf_summary['shortest_test_time']:.2f} seconds
"""
        
        markdown_content += f"""
## Test Results Summary

```
Return Code: {summary['test_results']['return_code']}
Success: {summary['test_results']['success']}
Execution Time: {summary['test_results']['execution_time']:.2f} seconds
```

## Next Steps

"""
        
        if summary['execution_summary']['success']:
            markdown_content += """
✅ **All integration tests passed successfully!**

- Review performance metrics for optimization opportunities
- Check coverage report if enabled
- Consider running with larger scale for production validation
"""
        else:
            markdown_content += """
❌ **Integration tests failed. Recommended actions:**

1. Review test output and error messages
2. Check environment configuration and dependencies
3. Verify data file availability and permissions
4. Run individual test modules to isolate issues
5. Check system resources (memory, disk space)
"""
        
        markdown_content += f"""
## Generated Reports

- **Summary Report**: `integration_test_summary.json`
- **HTML Report**: `integration_test_report.html`
- **Coverage Report**: `htmlcov/index.html` (if enabled)
- **Benchmark Report**: `benchmark.json` (if enabled)

## Command to Reproduce

```bash
python run_integration_tests.py \\
    --scale {summary['test_configuration']['test_scale']} \\
    --workers {summary['test_configuration']['parallel_workers']}"""
        
        if summary['test_configuration']['module_filter']:
            markdown_content += f" \\\n    --module {summary['test_configuration']['module_filter']}"
        
        if summary['test_configuration']['performance_only']:
            markdown_content += " \\\n    --performance"
        
        if summary['test_configuration']['coverage_enabled']:
            markdown_content += " \\\n    --coverage"
        
        if summary['test_configuration']['benchmark_enabled']:
            markdown_content += " \\\n    --benchmark"
        
        markdown_content += "\n```\n"
        
        # Save markdown report
        markdown_path = report_dir / 'integration_test_report.md'
        with open(markdown_path, 'w') as f:
            f.write(markdown_content)
    
    def print_summary(self):
        """Print execution summary to console."""
        success = self.test_results.get('success', False)
        execution_time = time.time() - self.start_time
        
        print("\n" + "="*80)
        print("INTEGRATION TEST EXECUTION SUMMARY")
        print("="*80)
        print(f"Result: {'✅ PASSED' if success else '❌ FAILED'}")
        print(f"Execution Time: {execution_time:.2f} seconds")
        print(f"Test Scale: {self.args.scale}")
        print(f"Parallel Workers: {self.args.workers}")
        
        if self.args.module:
            print(f"Module Filter: {self.args.module}")
        
        print(f"Report Directory: {self.args.report_dir}")
        print("="*80)
        
        if not success:
            print("\n❌ TESTS FAILED - Check logs and reports for details")
            print("Common troubleshooting steps:")
            print("1. Verify all dependencies are installed")
            print("2. Check system resources (memory, disk space)")
            print("3. Review test output for specific error messages")
            print("4. Try running with --debug for detailed logging")
        else:
            print("\n✅ ALL TESTS PASSED - Integration validation successful!")
            print("Next steps:")
            print("1. Review performance metrics in the report")
            print("2. Check coverage report if enabled")
            print("3. Consider production-scale validation")
    
    def run(self):
        """Main execution method."""
        self.args = self.parse_arguments()
        
        self.logger.info("Starting Australian Health Analytics Integration Test Runner")
        self.logger.info(f"Test scale: {self.args.scale}, Workers: {self.args.workers}")
        
        # Validate environment
        if not self.validate_environment():
            sys.exit(1)
        
        # Configure test environment
        self.configure_test_environment()
        
        # Run integration tests
        success = self.run_integration_tests()
        
        # Generate performance metrics
        self.generate_performance_metrics()
        
        # Generate reports
        self.generate_test_report()
        
        # Print summary
        self.print_summary()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    runner = IntegrationTestRunner()
    runner.run()