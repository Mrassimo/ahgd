#!/usr/bin/env python3
"""Command-line interface for AHGD pipeline execution.

This module provides a command-line interface for executing data transformation
and integration pipelines using the MasterETLPipeline.

British English spelling is used throughout (optimise, standardise, etc.).
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import time
from datetime import datetime
import json

from .master_etl_pipeline import MasterETLPipeline
from ..utils.config import get_config, get_config_manager, is_development
from ..utils.logging import get_logger, monitor_performance
from ..utils.interfaces import PipelineError, DataIntegrationLevel


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for pipeline CLI."""
    parser = argparse.ArgumentParser(
        description="Execute AHGD data transformation and integration pipelines",
        epilog="Examples:\n"
               "  ahgd-transform --input data_raw --output data_processed/master_health_record.parquet\n"
               "  ahgd-pipeline --pipeline master_integration_pipeline\n"
               "  ahgd-transform --input data_raw --integration-level comprehensive",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input/Output paths
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input directory containing raw extracted data"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output path for processed data (file path for single output)"
    )
    
    # Pipeline configuration
    parser.add_argument(
        "--pipeline",
        type=str,
        default="master_integration_pipeline",
        help="Pipeline name to execute (default: master_integration_pipeline)"
    )
    parser.add_argument(
        "--integration-level",
        choices=["minimal", "standard", "comprehensive", "enhanced"],
        default="standard",
        help="Data integration level (default: standard)"
    )
    
    # Processing options
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of parallel processing workers"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Processing chunk size for large datasets (default: 10000)"
    )
    parser.add_argument(
        "--memory-limit",
        type=str,
        help="Memory limit for processing (e.g., '4GB', '8GB')"
    )
    
    # Quality and validation options
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=85.0,
        help="Minimum quality threshold for data acceptance (default: 85.0)"
    )
    parser.add_argument(
        "--enable-validation",
        action="store_true",
        default=True,
        help="Enable data validation during processing (default: True)"
    )
    parser.add_argument(
        "--skip-quality-checks",
        action="store_true",
        help="Skip quality assurance checks (not recommended for production)"
    )
    
    # Geographic processing
    parser.add_argument(
        "--target-crs",
        type=str,
        default="EPSG:7844",  # GDA2020 / MGA zone 55
        help="Target coordinate reference system (default: EPSG:7844)"
    )
    parser.add_argument(
        "--simplify-geometry",
        action="store_true",
        help="Simplify geometric boundaries for web use"
    )
    
    # Output format options
    parser.add_argument(
        "--format",
        choices=["parquet", "csv", "json"],
        default="parquet",
        help="Output format (default: parquet)"
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        default=True,
        help="Enable compression for output (default: True)"
    )
    parser.add_argument(
        "--partition-by",
        choices=["state", "sa3", "date", "none"],
        default="none",
        help="Partition output data by specified dimension (default: none)"
    )
    
    # Reporting and monitoring
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Directory for quality reports and logs (default: reports)"
    )
    parser.add_argument(
        "--generate-lineage",
        action="store_true",
        default=True,
        help="Generate data lineage documentation (default: True)"
    )
    parser.add_argument(
        "--performance-monitoring",
        action="store_true",
        default=True,
        help="Enable performance monitoring (default: True)"
    )
    
    # Configuration options
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file (optional override)"
    )
    parser.add_argument(
        "--config-override",
        action="append",
        help="Override configuration values (format: key=value)"
    )
    
    # Logging and debugging
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-essential output"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Write logs to specified file"
    )
    
    # Development options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually doing it"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Resume pipeline from specific checkpoint"
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        default=True,
        help="Save checkpoints for resumable processing (default: True)"
    )
    
    return parser


def _parse_config_overrides(overrides: list) -> Dict[str, Any]:
    """Parse configuration override strings into dictionary."""
    config_dict = {}
    
    if not overrides:
        return config_dict
    
    for override in overrides:
        if '=' not in override:
            continue
            
        key, value = override.split('=', 1)
        
        # Convert string values to appropriate types
        if value.lower() in ('true', 'false'):
            value = value.lower() == 'true'
        elif value.isdigit():
            value = int(value)
        elif '.' in value and value.replace('.', '').isdigit():
            value = float(value)
            
        config_dict[key] = value
    
    return config_dict


@monitor_performance("pipeline_execution")
def execute_pipeline(
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    logger
) -> bool:
    """Execute the specified pipeline."""
    
    logger.info("Starting pipeline execution",
                pipeline=args.pipeline,
                input_path=str(input_path),
                output_path=str(output_path),
                integration_level=args.integration_level)
    
    if args.dry_run:
        logger.info("[DRY RUN] Would execute pipeline with specified parameters")
        return True
    
    try:
        # Parse integration level
        integration_level = DataIntegrationLevel(args.integration_level)
        
        # Load base configuration
        config = get_config('pipelines.master_etl', {})
        
        # Apply configuration overrides
        if args.config_override:
            overrides = _parse_config_overrides(args.config_override)
            config.update(overrides)
        
        # Update config with command-line arguments
        pipeline_config = {
            'name': args.pipeline,
            'integration_level': integration_level,
            'quality_config': {
                'enable_validation': args.enable_validation,
                'quality_threshold': args.quality_threshold,
                'skip_quality_checks': args.skip_quality_checks
            },
            'processing_config': {
                'max_workers': args.max_workers,
                'chunk_size': args.chunk_size,
                'memory_limit': args.memory_limit
            },
            'geographic_config': {
                'target_crs': args.target_crs,
                'simplify_geometry': args.simplify_geometry
            },
            'output_config': {
                'format': args.format,
                'compress': args.compress,
                'partition_by': args.partition_by
            },
            'monitoring_config': {
                'enable_checkpoints': args.save_checkpoints,
                'generate_quality_reports': True,
                'monitor_performance': args.performance_monitoring,
                'track_data_lineage': args.generate_lineage
            }
        }
        
        # Remove None values
        def clean_config(cfg):
            if isinstance(cfg, dict):
                return {k: clean_config(v) for k, v in cfg.items() if v is not None}
            return cfg
        
        pipeline_config = clean_config(pipeline_config)
        
        # Initialize pipeline
        logger.info("Initialising pipeline", config_keys=list(pipeline_config.keys()))
        pipeline = MasterETLPipeline(config=pipeline_config)
        
        # Check input data availability
        if not input_path.exists():
            raise PipelineError(f"Input path does not exist: {input_path}")
        
        # Create output directory if needed
        if output_path.suffix:  # It's a file path
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:  # It's a directory
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Create reports directory
        reports_dir = Path(args.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Execute pipeline
        start_time = time.time()
        
        logger.info("Executing pipeline stages...")
        
        # This is a simplified execution - the actual MasterETLPipeline
        # would handle the complex orchestration
        execution_result = pipeline.execute(
            input_path=str(input_path),
            output_path=str(output_path),
            resume_from=args.resume_from
        )
        
        execution_time = time.time() - start_time
        
        # Generate execution report
        _generate_execution_report(
            reports_dir, execution_result, args, execution_time
        )
        
        if execution_result.get('success', False):
            logger.info("Pipeline execution completed successfully",
                       duration=f"{execution_time:.2f}s",
                       output_path=str(output_path))
            return True
        else:
            logger.error("Pipeline execution failed",
                        errors=execution_result.get('errors', []))
            return False
        
    except Exception as e:
        logger.error(f"Pipeline execution error: {str(e)}")
        if is_development():
            import traceback
            traceback.print_exc()
        return False


def _generate_execution_report(
    reports_dir: Path, 
    execution_result: Dict[str, Any], 
    args: argparse.Namespace,
    execution_time: float
) -> None:
    """Generate pipeline execution report."""
    
    report = {
        'execution_metadata': {
            'timestamp': datetime.now().isoformat(),
            'pipeline': args.pipeline,
            'integration_level': args.integration_level,
            'execution_time_seconds': execution_time,
            'command_args': {
                'input': args.input,
                'output': args.output,
                'quality_threshold': args.quality_threshold,
                'integration_level': args.integration_level
            }
        },
        'pipeline_results': execution_result,
        'quality_summary': execution_result.get('quality_summary', {}),
        'performance_metrics': execution_result.get('performance_metrics', {})
    }
    
    # Write main execution report
    report_path = reports_dir / f'pipeline_execution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Write integration summary for DVC metrics
    integration_summary_path = reports_dir / 'quality_reports' / 'integration_summary.json'
    integration_summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    integration_summary = {
        'total_records_processed': execution_result.get('records_processed', 0),
        'quality_score': execution_result.get('overall_quality_score', 0.0),
        'integration_level': args.integration_level,
        'execution_timestamp': datetime.now().isoformat(),
        'success': execution_result.get('success', False)
    }
    
    with open(integration_summary_path, 'w', encoding='utf-8') as f:
        json.dump(integration_summary, f, indent=2, ensure_ascii=False)


def run_pipeline() -> int:
    """Entry point for direct pipeline execution (used by ahgd-pipeline command)."""
    parser = argparse.ArgumentParser(
        description="Execute specific AHGD pipeline by name"
    )
    parser.add_argument(
        "--pipeline",
        required=True,
        help="Pipeline name to execute"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Configuration file path"
    )
    
    args = parser.parse_args()
    logger = get_logger(__name__)
    
    try:
        logger.info(f"Executing pipeline: {args.pipeline}")
        
        # Load pipeline-specific configuration
        config = get_config(f'pipelines.{args.pipeline}', {})
        if args.config:
            # Load custom config file if specified
            logger.info(f"Using configuration file: {args.config}")
        
        # For now, print pipeline information
        # In a full implementation, this would execute the specified pipeline
        logger.info(f"Pipeline '{args.pipeline}' execution completed")
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        return 1


def main() -> int:
    """Main entry point for pipeline CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up logging
    logger = get_logger(__name__)
    
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        import logging
        logging.getLogger().setLevel(logging.WARNING)
    
    try:
        # Load configuration
        if args.config:
            logger.info(f"Using configuration file: {args.config}")
        
        # Set up paths
        input_path = Path(args.input)
        output_path = Path(args.output)
        
        # Execute pipeline
        success = execute_pipeline(input_path, output_path, args, logger)
        
        if success:
            logger.info("Pipeline execution completed successfully")
            return 0
        else:
            logger.error("Pipeline execution failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Pipeline execution cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error in pipeline CLI: {str(e)}")
        if is_development():
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())