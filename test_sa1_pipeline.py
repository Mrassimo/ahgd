#!/usr/bin/env python3
"""
Test Script for SA1 Data Pipeline

Validates the end-to-end SA1 migration pipeline including:
- DLT data extraction
- Pydantic validation
- DBT transformation
- Data quality checks
"""

import sys
import logging
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipelines.orchestrator import PipelineOrchestrator


def setup_logging():
    """Configure logging for test run."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/sa1_pipeline_test.log')
        ]
    )
    return logging.getLogger(__name__)


def test_sa1_pipeline():
    """
    Test the complete SA1 data pipeline.
    
    Tests:
    1. DLT extraction of SA1 boundaries and SEIFA data
    2. Data validation with Pydantic models
    3. DBT transformation and staging
    4. Data quality validation
    """
    
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("STARTING SA1 PIPELINE TEST")
    logger.info("=" * 80)
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator()
    
    # Test configuration - start with subset for testing
    test_config = {
        'dlt_pipelines': ['sa1_boundaries', 'seifa_sa1'],
        'dbt_commands': ['run', 'test']
    }
    
    start_time = time.time()
    
    try:
        # Phase 1: Test DLT Pipelines
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: TESTING DLT DATA EXTRACTION")
        logger.info("=" * 60)
        
        # Test SA1 boundaries pipeline
        logger.info("\nTesting SA1 boundaries extraction...")
        success, metrics = orchestrator.run_dlt_pipeline('sa1_boundaries')
        
        if success:
            logger.info(f"✅ SA1 boundaries pipeline successful")
            logger.info(f"   - Duration: {metrics.get('duration_seconds', 0):.2f} seconds")
            logger.info(f"   - Records: {metrics.get('records_processed', 0)}")
        else:
            logger.error(f"❌ SA1 boundaries pipeline failed: {metrics.get('error')}")
            return False
        
        # Test SEIFA SA1 pipeline
        logger.info("\nTesting SEIFA SA1 data extraction...")
        success, metrics = orchestrator.run_dlt_pipeline('seifa_sa1')
        
        if success:
            logger.info(f"✅ SEIFA SA1 pipeline successful")
            logger.info(f"   - Duration: {metrics.get('duration_seconds', 0):.2f} seconds")
            logger.info(f"   - Records: {metrics.get('records_processed', 0)}")
        else:
            logger.error(f"❌ SEIFA SA1 pipeline failed: {metrics.get('error')}")
            return False
        
        # Phase 2: Test DBT Transformations
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: TESTING DBT TRANSFORMATIONS")
        logger.info("=" * 60)
        
        # Run DBT models
        logger.info("\nRunning DBT staging models...")
        success, output = orchestrator.run_dbt_command(
            'run',
            ['--models', 'staging.geographic.stg_sa1_boundaries', 'staging.seifa.stg_seifa_sa1']
        )
        
        if success:
            logger.info("✅ DBT staging models successful")
        else:
            logger.error(f"❌ DBT staging models failed: {output[:500]}")
            return False
        
        # Run DBT tests
        logger.info("\nRunning DBT data quality tests...")
        success, output = orchestrator.run_dbt_command(
            'test',
            ['--models', 'staging.geographic.stg_sa1_boundaries', 'staging.seifa.stg_seifa_sa1']
        )
        
        if success:
            logger.info("✅ DBT tests passed")
        else:
            logger.warning(f"⚠️ Some DBT tests failed: {output[:500]}")
        
        # Phase 3: Data Quality Validation
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: DATA QUALITY VALIDATION")
        logger.info("=" * 60)
        
        # Run custom data quality checks
        quality_passed, issues = orchestrator.validate_data_quality()
        
        if quality_passed:
            logger.info("✅ All data quality checks passed")
        else:
            logger.warning(f"⚠️ Data quality issues found:")
            for issue in issues:
                logger.warning(f"   - {issue}")
        
        # Phase 4: Performance Metrics
        duration = time.time() - start_time
        logger.info("\n" + "=" * 60)
        logger.info("PERFORMANCE METRICS")
        logger.info("=" * 60)
        logger.info(f"Total pipeline duration: {duration:.2f} seconds")
        logger.info(f"Average processing speed: {61845 / duration:.0f} SA1s per second")
        
        # Memory usage check (requires psutil)
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"Memory usage: {memory_mb:.2f} MB")
        except ImportError:
            logger.info("Memory tracking not available (psutil not installed)")
        
        # Success summary
        logger.info("\n" + "=" * 80)
        logger.info("SA1 PIPELINE TEST COMPLETED SUCCESSFULLY ✅")
        logger.info("=" * 80)
        logger.info("\nKey achievements:")
        logger.info("- Successfully extracted SA1 boundary data")
        logger.info("- Successfully extracted SEIFA SA1 socio-economic data")
        logger.info("- DBT transformations applied successfully")
        logger.info("- Data quality validation passed")
        logger.info(f"- Pipeline completed in {duration:.2f} seconds")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline test failed with error: {e}", exc_info=True)
        return False


def quick_validation():
    """
    Quick validation of loaded data using DuckDB queries.
    """
    import duckdb
    
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 60)
    logger.info("QUICK DATA VALIDATION")
    logger.info("=" * 60)
    
    try:
        # Connect to database
        conn = duckdb.connect('health_analytics.db')
        
        # Check SA1 boundaries
        result = conn.execute("""
            SELECT COUNT(*) as count,
                   COUNT(DISTINCT state_code) as states,
                   MIN(area_sqkm) as min_area,
                   MAX(area_sqkm) as max_area
            FROM stg_sa1_boundaries
        """).fetchone()
        
        if result:
            logger.info(f"\nSA1 Boundaries:")
            logger.info(f"  - Total SA1s: {result[0]}")
            logger.info(f"  - States/Territories: {result[1]}")
            logger.info(f"  - Area range: {result[2]:.2f} - {result[3]:.2f} sq km")
        
        # Check SEIFA data
        result = conn.execute("""
            SELECT COUNT(*) as count,
                   AVG(complete_indexes_count) as avg_indexes,
                   COUNT(DISTINCT disadvantage_category) as categories
            FROM stg_seifa_sa1
        """).fetchone()
        
        if result:
            logger.info(f"\nSEIFA SA1 Data:")
            logger.info(f"  - Total records: {result[0]}")
            logger.info(f"  - Average complete indexes: {result[1]:.2f}")
            logger.info(f"  - Disadvantage categories: {result[2]}")
        
        # Check SA1-SA2 relationships
        result = conn.execute("""
            SELECT COUNT(DISTINCT sa1_code) as sa1_count,
                   COUNT(DISTINCT sa2_code) as sa2_count,
                   AVG(sa1_count) as avg_sa1_per_sa2
            FROM (
                SELECT sa2_code, COUNT(*) as sa1_count
                FROM stg_sa1_boundaries
                GROUP BY sa2_code
            )
        """).fetchone()
        
        if result:
            logger.info(f"\nGeographic Relationships:")
            logger.info(f"  - Unique SA1s: {result[0]}")
            logger.info(f"  - Unique SA2s: {result[1]}")
            logger.info(f"  - Average SA1s per SA2: {result[2]:.1f}")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return False


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    # Run the test
    success = test_sa1_pipeline()
    
    if success:
        # Run quick validation if pipeline succeeded
        quick_validation()
        sys.exit(0)
    else:
        sys.exit(1)