#!/usr/bin/env python3
"""
Real Data Test Runner for AHGD Extractors.

This script runs comprehensive tests to verify that extractors can download
and process real Australian government data. Use this to validate production
readiness.

Usage:
    python tests/test_real_data_runner.py --force-real-data
    python tests/test_real_data_runner.py --check-urls
    python tests/test_real_data_runner.py --validate-config
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import pytest
import requests
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractors.abs_extractor import (
    ABSGeographicExtractor,
    ABSCensusExtractor,
    ABSSEIFAExtractor
)
from src.extractors.aihw_extractor import (
    AIHWMortalityExtractor,
    AIHWHospitalisationExtractor
)
from src.extractors.bom_extractor import (
    BOMClimateExtractor,
    BOMWeatherStationExtractor
)
from src.utils.config import get_config
from src.utils.logging import get_logger


logger = get_logger(__name__)


class RealDataTestRunner:
    """Test runner for real data extraction validation."""
    
    def __init__(self):
        self.results = {
            'url_accessibility': {},
            'data_extraction': {},
            'record_counts': {},
            'errors': []
        }
    
    def check_url_accessibility(self) -> Dict[str, bool]:
        """Check if all configured URLs are accessible."""
        
        logger.info("Checking URL accessibility...")
        
        urls_to_test = self._get_all_configured_urls()
        results = {}
        
        for source, url in urls_to_test.items():
            try:
                logger.info(f"Testing {source}: {url}")
                
                response = requests.head(url, timeout=30, allow_redirects=True)
                accessible = response.status_code in [200, 301, 302]
                
                results[source] = {
                    'url': url,
                    'accessible': accessible,
                    'status_code': response.status_code,
                    'content_type': response.headers.get('content-type', 'unknown')
                }
                
                if accessible:
                    logger.info(f"✓ {source} is accessible")
                else:
                    logger.warning(f"✗ {source} returned status {response.status_code}")
                    
            except Exception as e:
                logger.error(f"✗ {source} failed: {e}")
                results[source] = {
                    'url': url,
                    'accessible': False,
                    'error': str(e)
                }
        
        self.results['url_accessibility'] = results
        return results
    
    def test_real_data_extraction(self) -> Dict[str, Dict]:
        """Test actual data extraction from real sources."""
        
        logger.info("Testing real data extraction...")
        
        extractors = self._get_extractors()
        results = {}
        
        for name, extractor in extractors.items():
            logger.info(f"Testing {name}...")
            
            try:
                start_time = time.time()
                
                # Get appropriate source for extractor
                source = self._get_test_source_for_extractor(name, extractor)
                
                # Force real data extraction (disable demo fallback)
                with self._force_real_data_mode(extractor):
                    batches = list(extractor.extract(source))
                
                extraction_time = time.time() - start_time
                total_records = sum(len(batch) for batch in batches)
                
                results[name] = {
                    'success': True,
                    'total_records': total_records,
                    'num_batches': len(batches),
                    'extraction_time_seconds': extraction_time,
                    'avg_records_per_second': total_records / extraction_time if extraction_time > 0 else 0
                }
                
                logger.info(f"✓ {name}: {total_records} records in {extraction_time:.2f}s")
                
                # Validate record counts meet expectations
                self._validate_record_counts(name, total_records)
                
            except Exception as e:
                logger.error(f"✗ {name} failed: {e}")
                results[name] = {
                    'success': False,
                    'error': str(e)
                }
                self.results['errors'].append(f"{name}: {e}")
        
        self.results['data_extraction'] = results
        return results
    
    def validate_configurations(self) -> Dict[str, bool]:
        """Validate that all configurations are production-ready."""
        
        logger.info("Validating configurations...")
        
        validation_results = {}
        
        # Test ABS configuration
        abs_config = get_config("extractors.abs")
        validation_results['abs_config'] = self._validate_abs_config(abs_config)
        
        # Test AIHW configuration
        aihw_config = get_config("extractors.aihw")
        validation_results['aihw_config'] = self._validate_aihw_config(aihw_config)
        
        # Test BOM configuration
        bom_config = get_config("extractors.bom")
        validation_results['bom_config'] = self._validate_bom_config(bom_config)
        
        return validation_results
    
    def generate_production_readiness_report(self) -> str:
        """Generate a comprehensive production readiness report."""
        
        report_lines = [
            "AHGD Extractor Production Readiness Report",
            "=" * 50,
            f"Generated: {datetime.now().isoformat()}",
            ""
        ]
        
        # URL Accessibility Summary
        report_lines.append("URL Accessibility Results:")
        report_lines.append("-" * 30)
        
        url_results = self.results.get('url_accessibility', {})
        accessible_count = sum(1 for r in url_results.values() if r.get('accessible', False))
        total_urls = len(url_results)
        
        report_lines.append(f"Accessible URLs: {accessible_count}/{total_urls}")
        
        for source, result in url_results.items():
            status = "✓" if result.get('accessible', False) else "✗"
            report_lines.append(f"  {status} {source}")
        
        report_lines.append("")
        
        # Data Extraction Summary
        report_lines.append("Data Extraction Results:")
        report_lines.append("-" * 30)
        
        extraction_results = self.results.get('data_extraction', {})
        successful_extractions = sum(1 for r in extraction_results.values() if r.get('success', False))
        total_extractors = len(extraction_results)
        
        report_lines.append(f"Successful Extractions: {successful_extractions}/{total_extractors}")
        
        for extractor, result in extraction_results.items():
            if result.get('success', False):
                records = result['total_records']
                time_taken = result['extraction_time_seconds']
                report_lines.append(f"  ✓ {extractor}: {records} records ({time_taken:.2f}s)")
            else:
                error = result.get('error', 'Unknown error')
                report_lines.append(f"  ✗ {extractor}: {error}")
        
        report_lines.append("")
        
        # Record Count Validation
        report_lines.append("Record Count Validation:")
        report_lines.append("-" * 30)
        
        record_counts = self.results.get('record_counts', {})
        for extractor, validation in record_counts.items():
            status = "✓" if validation.get('meets_expectations', False) else "✗"
            actual = validation.get('actual_count', 0)
            expected = validation.get('expected_range', 'unknown')
            report_lines.append(f"  {status} {extractor}: {actual} records (expected: {expected})")
        
        report_lines.append("")
        
        # Errors Summary
        if self.results['errors']:
            report_lines.append("Errors Encountered:")
            report_lines.append("-" * 20)
            for error in self.results['errors']:
                report_lines.append(f"  • {error}")
            report_lines.append("")
        
        # Overall Assessment
        report_lines.append("Overall Production Readiness Assessment:")
        report_lines.append("-" * 40)
        
        total_issues = len(self.results['errors'])
        accessibility_rate = accessible_count / total_urls if total_urls > 0 else 0
        extraction_rate = successful_extractions / total_extractors if total_extractors > 0 else 0
        
        if total_issues == 0 and accessibility_rate >= 0.8 and extraction_rate >= 0.8:
            assessment = "READY FOR PRODUCTION"
        elif total_issues < 3 and accessibility_rate >= 0.6 and extraction_rate >= 0.6:
            assessment = "MOSTLY READY - Minor issues need attention"
        else:
            assessment = "NOT READY FOR PRODUCTION - Significant issues detected"
        
        report_lines.append(f"Status: {assessment}")
        report_lines.append(f"URL Accessibility: {accessibility_rate:.1%}")
        report_lines.append(f"Extraction Success Rate: {extraction_rate:.1%}")
        report_lines.append(f"Total Issues: {total_issues}")
        
        return "\n".join(report_lines)
    
    def _get_all_configured_urls(self) -> Dict[str, str]:
        """Get all configured URLs for testing."""
        
        urls = {}
        
        # ABS URLs
        abs_config = get_config("extractors.abs")
        abs_geo = ABSGeographicExtractor(abs_config)
        abs_census = ABSCensusExtractor(abs_config)
        
        try:
            urls['ABS_SA2_Boundaries'] = abs_geo._get_default_abs_url('SA2', '2021')
            urls['ABS_SA3_Boundaries'] = abs_geo._get_default_abs_url('SA3', '2021')
            urls['ABS_Census_G01'] = abs_census._get_default_census_url('G01')
        except Exception as e:
            logger.warning(f"Could not get ABS URLs: {e}")
        
        # AIHW URLs
        aihw_config = get_config("extractors.aihw")
        aihw_mortality = AIHWMortalityExtractor(aihw_config)
        
        try:
            urls['AIHW_Mortality'] = aihw_mortality._get_default_aihw_url('grim-deaths')
            urls['AIHW_Hospital'] = aihw_mortality._get_default_aihw_url('hospital-data')
        except Exception as e:
            logger.warning(f"Could not get AIHW URLs: {e}")
        
        # BOM URLs
        bom_config = get_config("extractors.bom")
        bom_climate = BOMClimateExtractor(bom_config)
        
        urls['BOM_Climate_Data'] = bom_climate.base_url
        urls['BOM_Stations'] = "http://www.bom.gov.au/climate/data/stations"
        
        return urls
    
    def _get_extractors(self) -> Dict[str, object]:
        """Get all extractor instances for testing."""
        
        abs_config = get_config("extractors.abs")
        aihw_config = get_config("extractors.aihw")
        bom_config = get_config("extractors.bom")
        
        return {
            'ABS_Geographic': ABSGeographicExtractor(abs_config),
            'ABS_Census': ABSCensusExtractor(abs_config),
            'AIHW_Mortality': AIHWMortalityExtractor(aihw_config),
            'BOM_Climate': BOMClimateExtractor(bom_config)
        }
    
    def _get_test_source_for_extractor(self, name: str, extractor) -> Dict:
        """Get appropriate test source for each extractor."""
        
        if 'ABS_Geographic' in name:
            return {'level': 'SA4', 'year': '2021'}  # SA4 is smaller than SA2
        elif 'ABS_Census' in name:
            return {'table_id': 'G01'}
        elif 'AIHW' in name:
            return {'dataset_id': 'grim-deaths'}
        elif 'BOM' in name:
            return {'station_ids': ['066062'], 'start_date': '2023-01-01', 'end_date': '2023-01-31'}
        else:
            return {}
    
    def _force_real_data_mode(self, extractor):
        """Context manager to force real data extraction mode."""
        
        class RealDataMode:
            def __enter__(self):
                # Patch demo methods to raise errors
                self.patches = []
                
                if hasattr(extractor, '_extract_demo_geographic_data'):
                    from unittest.mock import patch
                    patcher = patch.object(extractor, '_extract_demo_geographic_data')
                    patcher.start()
                    patcher.return_value.side_effect = Exception("Demo data disabled for real data test")
                    self.patches.append(patcher)
                
                if hasattr(extractor, '_extract_demo_census_data'):
                    from unittest.mock import patch
                    patcher = patch.object(extractor, '_extract_demo_census_data')
                    patcher.start()
                    patcher.return_value.side_effect = Exception("Demo data disabled for real data test")
                    self.patches.append(patcher)
                
                if hasattr(extractor, '_extract_demo_data'):
                    from unittest.mock import patch
                    patcher = patch.object(extractor, '_extract_demo_data')
                    patcher.start()
                    patcher.return_value.side_effect = Exception("Demo data disabled for real data test")
                    self.patches.append(patcher)
                
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                for patcher in self.patches:
                    patcher.stop()
        
        return RealDataMode()
    
    def _validate_record_counts(self, extractor_name: str, actual_count: int):
        """Validate that record counts meet production expectations."""
        
        expectations = {
            'ABS_Geographic': (100, 700),      # SA4 boundaries: ~87 areas
            'ABS_Census': (2000, 2600),       # SA2 Census: ~2400 areas
            'AIHW_Mortality': (100, 10000),   # Mortality data varies widely
            'BOM_Climate': (1, 100)           # Single station, limited date range
        }
        
        expected_range = expectations.get(extractor_name, (1, float('inf')))
        min_expected, max_expected = expected_range
        
        meets_expectations = min_expected <= actual_count <= max_expected
        
        self.results['record_counts'][extractor_name] = {
            'actual_count': actual_count,
            'expected_range': f"{min_expected}-{max_expected}",
            'meets_expectations': meets_expectations
        }
        
        if not meets_expectations:
            self.results['errors'].append(
                f"{extractor_name} record count {actual_count} outside expected range {expected_range}"
            )
    
    def _validate_abs_config(self, config: Dict) -> Dict[str, bool]:
        """Validate ABS configuration."""
        
        checks = {}
        
        # Check required fields
        required_fields = ['timeout_seconds', 'retry_attempts', 'batch_size', 'coordinate_system']
        for field in required_fields:
            checks[f'has_{field}'] = field in config
        
        # Check coordinate system
        checks['uses_gda2020'] = config.get('coordinate_system') == 'GDA2020'
        
        # Check geographic configuration
        checks['has_geographic_config'] = 'geographic' in config
        if 'geographic' in config:
            checks['has_sa2_config'] = 'sa2' in config['geographic']
        
        return checks
    
    def _validate_aihw_config(self, config: Dict) -> Dict[str, bool]:
        """Validate AIHW configuration."""
        
        checks = {}
        
        # Check required fields
        required_fields = ['base_url', 'timeout_seconds', 'retry_attempts']
        for field in required_fields:
            checks[f'has_{field}'] = field in config
        
        # Check URL format
        base_url = config.get('base_url', '')
        checks['valid_base_url'] = base_url.startswith('https://') and 'aihw.gov.au' in base_url
        
        return checks
    
    def _validate_bom_config(self, config: Dict) -> Dict[str, bool]:
        """Validate BOM configuration."""
        
        checks = {}
        
        # Check required fields
        required_fields = ['base_url', 'timeout_seconds', 'retry_attempts']
        for field in required_fields:
            checks[f'has_{field}'] = field in config
        
        # Check URL format
        base_url = config.get('base_url', '')
        checks['valid_base_url'] = base_url.startswith('http://') and 'bom.gov.au' in base_url
        
        return checks


def main():
    """Main function to run real data tests."""
    
    parser = argparse.ArgumentParser(description='Test AHGD extractors with real data')
    parser.add_argument('--force-real-data', action='store_true',
                       help='Force extraction of real data (disable demo fallbacks)')
    parser.add_argument('--check-urls', action='store_true',
                       help='Check URL accessibility only')
    parser.add_argument('--validate-config', action='store_true',
                       help='Validate configurations only')
    parser.add_argument('--report-file', type=str,
                       help='Save report to file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    runner = RealDataTestRunner()
    
    try:
        if args.check_urls:
            logger.info("Running URL accessibility tests...")
            url_results = runner.check_url_accessibility()
            
            accessible = sum(1 for r in url_results.values() if r.get('accessible', False))
            total = len(url_results)
            
            print(f"\nURL Accessibility Results: {accessible}/{total} accessible")
            for source, result in url_results.items():
                status = "✓" if result.get('accessible', False) else "✗"
                print(f"  {status} {source}")
        
        elif args.validate_config:
            logger.info("Running configuration validation...")
            config_results = runner.validate_configurations()
            
            print("\nConfiguration Validation Results:")
            for config_name, checks in config_results.items():
                print(f"\n{config_name}:")
                for check, passed in checks.items():
                    status = "✓" if passed else "✗"
                    print(f"  {status} {check}")
        
        elif args.force_real_data:
            logger.info("Running comprehensive real data tests...")
            
            # Run all tests
            runner.check_url_accessibility()
            runner.test_real_data_extraction()
            
            # Generate report
            report = runner.generate_production_readiness_report()
            
            if args.report_file:
                with open(args.report_file, 'w') as f:
                    f.write(report)
                print(f"Report saved to: {args.report_file}")
            else:
                print("\n" + report)
        
        else:
            parser.print_help()
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Test run interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test run failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()