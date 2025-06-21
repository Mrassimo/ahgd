"""
AHGD Logging Framework - Usage Examples

This module demonstrates comprehensive usage of the AHGD logging framework
including structured logging, performance monitoring, health checks, and 
data lineage tracking.
"""

import asyncio
import time
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

# Import the AHGD logging framework
from src.utils.logging import (
    setup_logging, 
    get_logger, 
    log_operation, 
    monitor_performance,
    track_lineage
)
from src.utils.monitoring import (
    get_system_monitor,
    get_health_checker,
    get_error_tracker,
    check_database_connection,
    check_file_system_access
)


def example_basic_logging():
    """Example 1: Basic structured logging setup"""
    print("=== Example 1: Basic Logging ===")
    
    # Setup logging with configuration
    logger = setup_logging('configs/logging_config.yaml')
    
    # Basic logging
    logger.log.info("Application started", component="main", version="1.0.0")
    logger.log.debug("Debug information", user_id="12345", action="login")
    logger.log.warning("Warning message", resource="database", status="slow")
    logger.log.error("Error occurred", error_code="E001", details="Connection failed")
    
    # Structured logging with context
    logger.set_context(user_id="12345", session="abc-123")
    logger.log.info("User action performed", action="data_export", records=1000)
    logger.clear_context()


def example_operation_context():
    """Example 2: Operation context management"""
    print("=== Example 2: Operation Context ===")
    
    logger = get_logger()
    
    # Using context manager for operations
    with logger.operation_context("data_processing", 
                                 component="etl", 
                                 dataset="health_data"):
        
        logger.log.info("Starting data validation", records=5000)
        time.sleep(1)  # Simulate work
        logger.log.info("Validation completed", valid_records=4950, invalid_records=50)
        
        # Nested operations
        with logger.operation_context("data_transformation"):
            logger.log.info("Applying transformations", transformation_count=12)
            time.sleep(0.5)  # Simulate work
            logger.log.info("Transformations completed")


@monitor_performance("data_extraction")
def example_performance_monitoring():
    """Example 3: Performance monitoring with decorators"""
    print("=== Example 3: Performance Monitoring ===")
    
    logger = get_logger()
    
    # This function is automatically monitored
    logger.log.info("Extracting data from source")
    time.sleep(2)  # Simulate data extraction
    logger.log.info("Data extraction completed", records=10000)
    
    return {"status": "success", "records": 10000}


async def example_async_performance_monitoring():
    """Example 4: Async performance monitoring"""
    print("=== Example 4: Async Performance Monitoring ===")
    
    @monitor_performance("async_data_processing")
    async def process_data_async():
        logger = get_logger()
        logger.log.info("Starting async data processing")
        await asyncio.sleep(1)  # Simulate async work
        logger.log.info("Async processing completed")
        return {"processed": True}
    
    result = await process_data_async()
    return result


def example_data_lineage_tracking():
    """Example 5: Data lineage tracking"""
    print("=== Example 5: Data Lineage Tracking ===")
    
    logger = get_logger()
    
    # Track data lineage for ETL operations
    
    # Extract phase
    track_lineage(
        source_id="aihw_mortality_data",
        target_id="raw_mortality_staging",
        operation="data_extraction",
        schema_version="v1.2",
        row_count=50000,
        transformations=["csv_to_parquet", "date_standardization"]
    )
    
    # Transform phase
    track_lineage(
        source_id="raw_mortality_staging",
        target_id="clean_mortality_data",
        operation="data_cleaning",
        schema_version="v1.2",
        row_count=49800,
        transformations=["null_removal", "outlier_detection", "data_validation"],
        validation_status="passed"
    )
    
    # Load phase
    track_lineage(
        source_id="clean_mortality_data",
        target_id="health_analytics_db.mortality_facts",
        operation="data_loading",
        schema_version="v1.2",
        row_count=49800,
        transformations=["dimension_lookup", "fact_table_insert"]
    )
    
    # Get lineage records
    lineage_records = logger.get_lineage_records()
    print(f"Tracked {len(lineage_records)} lineage records")


def example_health_checks():
    """Example 6: Health check implementation"""
    print("=== Example 6: Health Checks ===")
    
    health_checker = get_health_checker()
    
    # Register custom health checks
    def check_data_directory():
        """Check if data directories are accessible"""
        return (Path("data_raw").exists() and 
                Path("data_processed").exists() and
                Path("logs").exists())
    
    def check_configuration():
        """Check if configuration is valid"""
        try:
            config_path = Path("configs/logging_config.yaml")
            return config_path.exists() and config_path.stat().st_size > 0
        except Exception:
            return False
    
    def check_database():
        """Check database connectivity"""
        # This would use real connection string in production
        return check_database_connection("sqlite:///data/health_analytics.db")
    
    # Register health checks
    health_checker.register_health_check(
        "data_directories", 
        check_data_directory,
        "Verify data directory structure exists"
    )
    
    health_checker.register_health_check(
        "configuration",
        check_configuration,
        "Verify logging configuration is valid"
    )
    
    health_checker.register_health_check(
        "database_connectivity",
        check_database,
        "Verify database connection is working"
    )
    
    # Run health checks
    health_summary = health_checker.get_health_summary()
    print(f"Overall health status: {health_summary['overall_status']}")
    print(f"Healthy checks: {health_summary['summary']['healthy']}")
    print(f"Unhealthy checks: {health_summary['summary']['unhealthy']}")


def example_system_monitoring():
    """Example 7: System resource monitoring"""
    print("=== Example 7: System Monitoring ===")
    
    monitor = get_system_monitor()
    
    # Get current system metrics
    current_metrics = monitor.get_current_metrics()
    print(f"CPU Usage: {current_metrics.cpu_percent:.1f}%")
    print(f"Memory Usage: {current_metrics.memory_percent:.1f}%")
    print(f"Disk Usage: {current_metrics.disk_usage_percent:.1f}%")
    
    # Collect metrics (this would normally run continuously)
    for i in range(5):
        metrics = monitor.collect_metrics()
        time.sleep(1)
    
    # Get metrics summary
    summary = monitor.get_metrics_summary(hours=1)
    if summary:
        print(f"Average CPU over last hour: {summary['cpu']['avg']:.1f}%")
        print(f"Peak memory usage: {summary['memory']['max']:.1f}%")


def example_error_tracking():
    """Example 8: Error tracking and analysis"""
    print("=== Example 8: Error Tracking ===")
    
    error_tracker = get_error_tracker()
    
    # Simulate some errors
    try:
        # This will raise an exception
        raise ValueError("Invalid data format in health records")
    except Exception as e:
        error_tracker.track_error(e, {
            "component": "data_validation",
            "dataset": "mortality_data",
            "record_id": "12345"
        })
    
    try:
        # This will raise another exception
        raise ConnectionError("Failed to connect to external API")
    except Exception as e:
        error_tracker.track_error(e, {
            "component": "api_client",
            "endpoint": "https://api.aihw.gov.au/mortality",
            "retry_count": 3
        })
    
    # Get error summary
    error_summary = error_tracker.get_error_summary(hours=24)
    print(f"Total errors in last 24h: {error_summary['total_errors']}")
    print(f"Error rate: {error_summary['error_rate']:.2f} errors/hour")
    print(f"Error types: {error_summary['error_types']}")


def example_etl_pipeline_logging():
    """Example 9: Complete ETL pipeline with logging"""
    print("=== Example 9: ETL Pipeline Logging ===")
    
    logger = get_logger()
    
    # Simulate a complete ETL pipeline with comprehensive logging
    pipeline_data = {
        "source": "AIHW Mortality Data",
        "target": "Health Analytics Database",
        "expected_records": 50000
    }
    
    with logger.operation_context("etl_pipeline", **pipeline_data):
        
        # Extract phase
        with logger.operation_context("extract_phase"):
            logger.log.info("Starting data extraction", 
                          source="aihw_api", 
                          endpoint="/mortality/2023")
            
            # Simulate extraction work
            time.sleep(1)
            extracted_records = 50000
            
            logger.log.info("Data extraction completed",
                          records_extracted=extracted_records,
                          status="success")
            
            track_lineage(
                source_id="aihw_mortality_api",
                target_id="raw_mortality_staging",
                operation="api_extraction",
                row_count=extracted_records
            )
        
        # Transform phase
        with logger.operation_context("transform_phase"):
            logger.log.info("Starting data transformation",
                          input_records=extracted_records)
            
            # Simulate data quality issues
            valid_records = int(extracted_records * 0.98)  # 2% data quality issues
            invalid_records = extracted_records - valid_records
            
            if invalid_records > 0:
                logger.log.warning("Data quality issues detected",
                                 invalid_records=invalid_records,
                                 quality_score=0.98)
            
            # Simulate transformation work
            time.sleep(1.5)
            
            logger.log.info("Data transformation completed",
                          input_records=extracted_records,
                          output_records=valid_records,
                          quality_score=valid_records/extracted_records)
            
            track_lineage(
                source_id="raw_mortality_staging",
                target_id="clean_mortality_data",
                operation="data_transformation",
                row_count=valid_records,
                transformations=["validation", "cleansing", "enrichment"],
                validation_status="passed"
            )
        
        # Load phase
        with logger.operation_context("load_phase"):
            logger.log.info("Starting data loading",
                          target_table="mortality_facts",
                          records_to_load=valid_records)
            
            # Simulate loading work
            time.sleep(1)
            loaded_records = valid_records
            
            logger.log.info("Data loading completed",
                          records_loaded=loaded_records,
                          status="success")
            
            track_lineage(
                source_id="clean_mortality_data",
                target_id="health_db.mortality_facts",
                operation="data_loading",
                row_count=loaded_records
            )
    
    # Export metrics and lineage
    logger.export_metrics("logs/etl_pipeline_metrics.json")
    print("ETL pipeline completed with comprehensive logging")


def example_monitoring_integration():
    """Example 10: Integrated monitoring and alerting"""
    print("=== Example 10: Monitoring Integration ===")
    
    # Start system monitoring (in production, this would run as a service)
    monitor = get_system_monitor({
        'notifications': {
            'webhook': {
                'enabled': False,  # Would be True in production
                'url': 'https://alerts.example.com/ahgd'
            }
        }
    })
    
    # This would normally run continuously
    print("System monitoring configured (would run continuously in production)")
    
    # Demonstrate health check integration
    health_checker = get_health_checker()
    
    # Register a health check that integrates with monitoring
    def check_etl_pipeline_health():
        """Check if ETL pipeline is healthy"""
        # This would check actual pipeline status
        return True  # Simulated healthy status
    
    health_checker.register_health_check(
        "etl_pipeline",
        check_etl_pipeline_health,
        "ETL pipeline operational status"
    )
    
    # Run integrated health check
    health_summary = health_checker.get_health_summary()
    logger = get_logger()
    logger.log.info("Health check completed", 
                   overall_status=health_summary['overall_status'],
                   healthy_checks=health_summary['summary']['healthy'])


async def run_all_examples():
    """Run all logging examples"""
    print("AHGD Logging Framework - Complete Examples")
    print("=" * 50)
    
    # Run synchronous examples
    example_basic_logging()
    print()
    
    example_operation_context()
    print()
    
    example_performance_monitoring()
    print()
    
    # Run async example
    await example_async_performance_monitoring()
    print()
    
    example_data_lineage_tracking()
    print()
    
    example_health_checks()
    print()
    
    example_system_monitoring()
    print()
    
    example_error_tracking()
    print()
    
    example_etl_pipeline_logging()
    print()
    
    example_monitoring_integration()
    print()
    
    print("All examples completed successfully!")
    
    # Show final metrics
    logger = get_logger()
    metrics = logger.get_performance_metrics()
    print(f"\nFinal Performance Metrics:")
    for operation, data in metrics.items():
        if data:
            avg_duration = sum(d['duration'] for d in data) / len(data)
            print(f"  {operation}: {len(data)} runs, avg {avg_duration:.3f}s")
    
    # Show lineage summary
    lineage = logger.get_lineage_records()
    print(f"\nData Lineage Records: {len(lineage)} tracked operations")


if __name__ == "__main__":
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("data_raw").mkdir(exist_ok=True)
    Path("data_processed").mkdir(exist_ok=True)
    
    # Run all examples
    asyncio.run(run_all_examples())


# Additional examples for specific use cases

class ExampleDataProcessor:
    """Example class showing logging integration in data processing components"""
    
    def __init__(self):
        self.logger = get_logger()
        self.logger.set_context(component="data_processor", class="ExampleDataProcessor")
    
    @monitor_performance("process_health_data")
    def process_health_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process health data with comprehensive logging"""
        
        with self.logger.operation_context("health_data_processing", 
                                         input_records=len(data)):
            
            self.logger.log.info("Starting health data processing", 
                               columns=list(data.columns),
                               data_types=data.dtypes.to_dict())
            
            # Data validation
            null_counts = data.isnull().sum()
            if null_counts.sum() > 0:
                self.logger.log.warning("Null values detected",
                                      null_counts=null_counts.to_dict())
            
            # Process data (simplified example)
            processed_data = data.dropna()
            
            # Track the transformation
            track_lineage(
                source_id="raw_health_data",
                target_id="processed_health_data",
                operation="null_removal",
                row_count=len(processed_data),
                transformations=["dropna"]
            )
            
            self.logger.log.info("Health data processing completed",
                               output_records=len(processed_data),
                               removed_records=len(data) - len(processed_data))
            
            return processed_data


class ExampleETLOrchestrator:
    """Example ETL orchestrator with comprehensive logging"""
    
    def __init__(self):
        self.logger = get_logger()
        self.monitor = get_system_monitor()
        self.health_checker = get_health_checker()
        self.error_tracker = get_error_tracker()
    
    async def run_etl_pipeline(self, config: Dict[str, Any]):
        """Run complete ETL pipeline with monitoring"""
        
        pipeline_id = config.get('pipeline_id', 'default')
        
        with self.logger.operation_context("etl_orchestrator", 
                                         pipeline_id=pipeline_id):
            
            try:
                # Pre-flight checks
                health_status = self.health_checker.get_health_summary()
                if health_status['overall_status'] != 'healthy':
                    raise RuntimeError("System health check failed")
                
                # Monitor system resources
                initial_metrics = self.monitor.get_current_metrics()
                self.logger.log.info("ETL pipeline started",
                                   initial_cpu=initial_metrics.cpu_percent,
                                   initial_memory=initial_metrics.memory_percent)
                
                # Run pipeline stages
                await self._extract_stage(config)
                await self._transform_stage(config)
                await self._load_stage(config)
                
                # Final metrics
                final_metrics = self.monitor.get_current_metrics()
                self.logger.log.info("ETL pipeline completed successfully",
                                   final_cpu=final_metrics.cpu_percent,
                                   final_memory=final_metrics.memory_percent)
                
            except Exception as e:
                self.error_tracker.track_error(e, {
                    'pipeline_id': pipeline_id,
                    'stage': 'orchestration'
                })
                raise
    
    async def _extract_stage(self, config: Dict[str, Any]):
        """Extract stage with logging"""
        with self.logger.operation_context("extract_stage"):
            self.logger.log.info("Extract stage started")
            await asyncio.sleep(1)  # Simulate work
            self.logger.log.info("Extract stage completed")
    
    async def _transform_stage(self, config: Dict[str, Any]):
        """Transform stage with logging"""
        with self.logger.operation_context("transform_stage"):
            self.logger.log.info("Transform stage started")
            await asyncio.sleep(1)  # Simulate work
            self.logger.log.info("Transform stage completed")
    
    async def _load_stage(self, config: Dict[str, Any]):
        """Load stage with logging"""
        with self.logger.operation_context("load_stage"):
            self.logger.log.info("Load stage started")
            await asyncio.sleep(1)  # Simulate work
            self.logger.log.info("Load stage completed")


def example_production_deployment():
    """Example of production deployment configuration"""
    print("=== Production Deployment Example ===")
    
    # In production, you would load this from environment or config file
    production_config = {
        'environment': 'production',
        'log_level': 'INFO',
        'monitoring': {
            'enabled': True,
            'interval_seconds': 30
        },
        'notifications': {
            'email': {
                'enabled': True,
                'smtp_server': 'smtp.company.com',
                'from': 'ahgd-alerts@company.com',
                'to': ['devops@company.com', 'data-team@company.com']
            },
            'slack': {
                'enabled': True,
                'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
            }
        }
    }
    
    # Setup production logging
    logger = setup_logging('configs/logging_config.yaml')
    
    # Configure monitoring for production
    monitor = get_system_monitor(production_config)
    
    # Start continuous monitoring (in production)
    # monitor.start_monitoring(interval=30)
    
    logger.log.info("Production deployment configured",
                   environment=production_config['environment'],
                   monitoring_enabled=production_config['monitoring']['enabled'])
    
    print("Production configuration applied")


# Configuration examples for different scenarios
EXAMPLE_CONFIGS = {
    'development': {
        'log_level': 'DEBUG',
        'console_logs': True,
        'json_logs': True,
        'performance_logging': True,
        'lineage_tracking': True
    },
    'testing': {
        'log_level': 'WARNING',
        'console_logs': False,
        'json_logs': True,
        'performance_logging': False,
        'lineage_tracking': False
    },
    'production': {
        'log_level': 'INFO',
        'console_logs': False,
        'json_logs': True,
        'performance_logging': True,
        'lineage_tracking': True,
        'monitoring': {
            'enabled': True,
            'alerts': True,
            'notifications': True
        }
    }
}


if __name__ == "__main__":
    print("AHGD Logging Framework Examples")
    print("Run this script to see comprehensive logging examples")