#!/usr/bin/env python3
"""
Test script for real data extraction implementation.

This script tests the enhanced extractors to verify they can attempt
real data extraction from Australian government sources.
"""

import sys
import logging
import traceback
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Change to project directory for proper relative imports
import os
os.chdir(project_root)

from src.extractors.abs_extractor import ABSGeographicExtractor, ABSCensusExtractor, ABSSEIFAExtractor
from src.extractors.aihw_extractor import AIHWMortalityExtractor
from src.utils.logging import get_logger

# Configure logging for testing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = get_logger(__name__)


def test_abs_geographic_extraction():
    """Test ABS geographic boundary extraction."""
    logger.info("=== Testing ABS Geographic Extraction ===")
    
    try:
        config = {
            'batch_size': 100,
            'coordinate_system': 'GDA2020',
            'abs_base_url': 'https://www.abs.gov.au'
        }
        
        extractor = ABSGeographicExtractor(config)
        
        # Test with SA2 boundaries (most important for the pipeline)
        source = {
            'level': 'SA2',
            'year': '2021'
        }
        
        logger.info("Attempting to extract SA2 boundaries...")
        record_count = 0
        batch_count = 0
        
        for batch in extractor.extract(source):
            batch_count += 1
            record_count += len(batch)
            logger.info(f"Batch {batch_count}: {len(batch)} SA2 records")
            
            # Show sample record
            if batch_count == 1 and batch:
                sample_record = batch[0]
                logger.info(f"Sample SA2 record: {sample_record.get('geographic_id', 'N/A')} - {sample_record.get('geographic_name', 'N/A')}")
            
            # Limit for testing
            if batch_count >= 3:
                break
        
        logger.info(f"ABS Geographic test completed: {record_count} records in {batch_count} batches")
        return True
        
    except Exception as e:
        logger.error(f"ABS Geographic extraction test failed: {e}")
        logger.debug(traceback.format_exc())
        return False


def test_abs_census_extraction():
    """Test ABS Census data extraction."""
    logger.info("=== Testing ABS Census Extraction ===")
    
    try:
        config = {
            'batch_size': 100,
            'census_year': 2021
        }
        
        extractor = ABSCensusExtractor(config)
        
        # Test with G01 (General Community Profile)
        source = {
            'table_id': 'G01'
        }
        
        logger.info("Attempting to extract Census data...")
        record_count = 0
        batch_count = 0
        
        for batch in extractor.extract(source):
            batch_count += 1
            record_count += len(batch)
            logger.info(f"Batch {batch_count}: {len(batch)} Census records")
            
            # Show sample record
            if batch_count == 1 and batch:
                sample_record = batch[0]
                logger.info(f"Sample Census record: SA2 {sample_record.get('geographic_id', 'N/A')} - Pop: {sample_record.get('total_population', 'N/A')}")
            
            # Limit for testing
            if batch_count >= 2:
                break
        
        logger.info(f"ABS Census test completed: {record_count} records in {batch_count} batches")
        return True
        
    except Exception as e:
        logger.error(f"ABS Census extraction test failed: {e}")
        logger.debug(traceback.format_exc())
        return False


def test_aihw_mortality_extraction():
    """Test AIHW mortality data extraction."""
    logger.info("=== Testing AIHW Mortality Extraction ===")
    
    try:
        config = {
            'batch_size': 100,
            'aihw_base_url': 'https://www.aihw.gov.au',
            'data_format': 'csv'
        }
        
        extractor = AIHWMortalityExtractor(config)
        
        # Test with GRIM mortality data
        source = {
            'dataset_id': 'grim-deaths'
        }
        
        logger.info("Attempting to extract AIHW mortality data...")
        record_count = 0
        batch_count = 0
        
        for batch in extractor.extract(source):
            batch_count += 1
            record_count += len(batch)
            logger.info(f"Batch {batch_count}: {len(batch)} mortality records")
            
            # Show sample record
            if batch_count == 1 and batch:
                sample_record = batch[0]
                logger.info(f"Sample mortality record: SA2 {sample_record.get('geographic_id', 'N/A')} - {sample_record.get('indicator_name', 'N/A')}")
            
            # Limit for testing
            if batch_count >= 2:
                break
        
        logger.info(f"AIHW Mortality test completed: {record_count} records in {batch_count} batches")
        return True
        
    except Exception as e:
        logger.error(f"AIHW Mortality extraction test failed: {e}")
        logger.debug(traceback.format_exc())
        return False


def test_data_quality():
    """Test data quality of extracted records."""
    logger.info("=== Testing Data Quality ===")
    
    try:
        # Test ABS SA2 data quality
        config = {'batch_size': 10}
        extractor = ABSGeographicExtractor(config)
        
        # Get a small sample
        source = {'level': 'SA2', 'year': '2021'}
        for batch in extractor.extract(source):
            if batch:
                record = batch[0]
                
                # Check required fields
                required_fields = ['geographic_id', 'geographic_level', 'data_source_id']
                missing_fields = [field for field in required_fields if field not in record]
                
                if missing_fields:
                    logger.warning(f"Missing required fields: {missing_fields}")
                    return False
                
                # Check data types
                if not isinstance(record.get('geographic_id'), str):
                    logger.warning("geographic_id should be string")
                    return False
                
                # Check SA2 code format (9 digits)
                import re
                sa2_code = record.get('geographic_id', '')
                if not re.match(r'^\d{9}$', sa2_code):
                    logger.warning(f"Invalid SA2 code format: {sa2_code}")
                    return False
                
                logger.info("Data quality checks passed")
                return True
            break
        
        logger.warning("No data returned for quality testing")
        return False
        
    except Exception as e:
        logger.error(f"Data quality test failed: {e}")
        return False


def main():
    """Run all extraction tests."""
    logger.info("Starting real data extraction tests...")
    
    test_results = {
        "ABS Geographic": test_abs_geographic_extraction(),
        "ABS Census": test_abs_census_extraction(), 
        "AIHW Mortality": test_aihw_mortality_extraction(),
        "Data Quality": test_data_quality()
    }
    
    # Summary
    logger.info("\n=== TEST RESULTS SUMMARY ===")
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All real data extraction tests PASSED!")
        logger.info("Real data extraction implementation is working correctly.")
    else:
        logger.warning("⚠️  Some tests failed - extractors falling back to demo data")
        logger.info("This is expected if network access is limited or data sources are unavailable")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)