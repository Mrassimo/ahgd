#!/usr/bin/env python3
"""Command-line interface for AHGD data loading and export.

This module provides a command-line interface for exporting processed data
to multiple formats using the ProductionLoader.

British English spelling is used throughout (optimise, standardise, etc.).
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import time
from datetime import datetime
import json

from .production_loader import ProductionLoader
from ..utils.config import get_config, get_config_manager, is_development
from ..utils.logging import get_logger, monitor_performance
from ..utils.interfaces import LoadingError, DataFormat


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for loader CLI."""
    parser = argparse.ArgumentParser(
        description="Export AHGD processed data to multiple formats for production use",
        epilog="Examples:\n"
               "  ahgd-loader --input data_processed/master_health_record.parquet --output data_exports\n"
               "  ahgd-loader --input data.parquet --formats parquet csv json --compress\n"
               "  ahgd-loader --input data.parquet --output exports/ --partition-by state --optimise-web",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input/Output paths
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input data file or directory to export"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for exported data"
    )
    
    # Format selection
    parser.add_argument(
        "--formats", "-f",
        nargs="+",
        choices=["parquet", "csv", "json", "geojson", "xlsx"],
        default=["parquet", "csv", "json"],
        help="Export formats (default: parquet csv json)"
    )
    parser.add_argument(
        "--list-formats",
        action="store_true",
        help="List all supported export formats and exit"
    )
    
    # Compression options
    parser.add_argument(
        "--compress",
        action="store_true",
        default=True,
        help="Enable compression for exports (default: True)"
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        choices=range(1, 13),
        help="Compression level (1-12, format-dependent)"
    )
    parser.add_argument(
        "--compression-algorithm",
        choices=["gzip", "brotli", "lz4", "snappy"],
        help="Force specific compression algorithm"
    )
    
    # Partitioning options
    parser.add_argument(
        "--partition",
        action="store_true",
        default=True,
        help="Enable data partitioning for large datasets (default: True)"
    )
    parser.add_argument(
        "--partition-by",
        choices=["state", "sa3", "temporal", "size", "auto", "none"],
        default="auto",
        help="Partitioning strategy (default: auto)"
    )
    parser.add_argument(
        "--partition-size",
        type=str,
        help="Target partition size (e.g., '50MB', '100000' rows)"
    )
    
    # Web optimisation
    parser.add_argument(
        "--optimise-web",
        action="store_true",
        default=True,
        help="Optimise exports for web delivery (default: True)"
    )
    parser.add_argument(
        "--reduce-precision",
        action="store_true",
        help="Reduce numeric precision for smaller file sizes"
    )
    parser.add_argument(
        "--generate-cache-headers",
        action="store_true",
        default=True,
        help="Generate cache headers metadata (default: True)"
    )
    
    # Quality and validation
    parser.add_argument(
        "--validate-exports",
        action="store_true",
        default=True,
        help="Validate exports after creation (default: True)"
    )
    parser.add_argument(
        "--quality-check",
        action="store_true",
        help="Perform quality checks on exported data"
    )
    parser.add_argument(
        "--sample-validation",
        type=float,
        default=0.1,
        help="Fraction of data to sample for validation (default: 0.1)"
    )
    
    # Performance options
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel export processing"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of parallel export workers"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Processing chunk size for large datasets (default: 50000)"
    )
    parser.add_argument(
        "--memory-limit",
        type=str,
        help="Memory limit for processing (e.g., '4GB', '8GB')"
    )
    
    # Metadata and documentation
    parser.add_argument(
        "--generate-metadata",
        action="store_true",
        default=True,
        help="Generate comprehensive export metadata (default: True)"
    )
    parser.add_argument(
        "--include-lineage",
        action="store_true",
        default=True,
        help="Include data lineage information (default: True)"
    )
    parser.add_argument(
        "--generate-docs",
        action="store_true",
        help="Generate documentation for exported datasets"
    )
    
    # Configuration options
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to export configuration file"
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["development", "testing", "production"],
        default="production",
        help="Export profile to use (default: production)"
    )
    parser.add_argument(
        "--config-override",
        action="append",
        help="Override configuration values (format: key=value)"
    )
    
    # Output customisation
    parser.add_argument(
        "--prefix",
        type=str,
        help="Prefix for exported file names"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        help="Suffix for exported file names"
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Add timestamp to exported file names"
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
        "--debug",
        action="store_true",
        help="Enable debug logging and detailed traces"
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
        help="Show what would be exported without actually doing it"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing export files"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Resume export from specific format/partition"
    )
    
    return parser


def list_supported_formats(loader: ProductionLoader) -> None:
    """List all supported export formats."""
    print("Supported export formats:")
    print("=" * 40)
    
    supported_formats = loader.get_supported_formats()
    
    format_descriptions = {
        DataFormat.PARQUET: "Apache Parquet - Columnar storage with excellent compression",
        DataFormat.CSV: "Comma-separated values - Universal compatibility",
        DataFormat.JSON: "JavaScript Object Notation - Web API ready",
        DataFormat.GEOJSON: "Geographic JSON - Mapping and GIS applications",
        DataFormat.XLSX: "Excel format - Business user friendly"
    }
    
    for format_enum in supported_formats:
        format_name = format_enum.value
        description = format_descriptions.get(format_enum, "No description available")
        print(f"\n📄 {format_name.upper()}")
        print(f"   Description: {description}")
        print(f"   Enum: {format_enum}")


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


@monitor_performance("data_export")
def export_data(
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    logger
) -> Dict[str, Any]:
    """Export data using the ProductionLoader."""
    
    logger.info("Starting data export",
                input_path=str(input_path),
                output_path=str(output_path),
                formats=args.formats)
    
    if args.dry_run:
        logger.info("[DRY RUN] Would export data with specified parameters")
        return {
            'success': True,
            'formats': {fmt: {'files': [], 'total_size_bytes': 0} for fmt in args.formats},
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'total_rows': 1000,  # Mock data
                'total_columns': 20,
                'data_size_mb': 50.0
            }
        }
    
    try:
        # Load export configuration
        config = get_config('exports', {})
        
        # Apply configuration overrides
        if args.config_override:
            overrides = _parse_config_overrides(args.config_override)
            config.update(overrides)
        
        # Initialize production loader
        logger.info("Initialising production loader")
        loader = ProductionLoader(config=config)
        
        # Check input data exists
        if not input_path.exists():
            raise LoadingError(f"Input path does not exist: {input_path}")
        
        # Load input data
        logger.info("Loading input data...")
        
        if input_path.suffix == '.parquet':
            import pandas as pd
            data = pd.read_parquet(input_path)
        elif input_path.suffix == '.csv':
            import pandas as pd
            data = pd.read_csv(input_path)
        elif input_path.suffix == '.json':
            import pandas as pd
            data = pd.read_json(input_path)
        else:
            raise LoadingError(f"Unsupported input format: {input_path.suffix}")
        
        logger.info(f"Loaded {len(data)} rows, {len(data.columns)} columns")
        
        # Apply file naming
        export_kwargs = {}
        if args.prefix:
            export_kwargs['file_prefix'] = args.prefix
        if args.suffix:
            export_kwargs['file_suffix'] = args.suffix
        if args.timestamp:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_kwargs['file_suffix'] = f"{export_kwargs.get('file_suffix', '')}_{timestamp_str}".strip('_')
        
        # Configure compression
        compression_config = {}
        if args.compression_algorithm:
            compression_config['algorithm'] = args.compression_algorithm
        if args.compression_level:
            compression_config['level'] = args.compression_level
        
        # Execute export
        start_time = time.time()
        
        logger.info("Executing export...")
        
        export_result = loader.load(
            data=data,
            output_path=output_path,
            formats=args.formats,
            compress=args.compress,
            partition=args.partition,
            optimise_for_web=args.optimise_web,
            partition_strategy=args.partition_by,
            **export_kwargs
        )
        
        export_time = time.time() - start_time
        
        # Validate exports if requested
        if args.validate_exports:
            logger.info("Validating exports...")
            validation_success = loader.validate_export(export_result)
            export_result['validation'] = {
                'success': validation_success,
                'timestamp': datetime.now().isoformat()
            }
            
            if not validation_success:
                logger.error("Export validation failed")
                export_result['success'] = False
        
        # Add timing information
        export_result['metadata']['export_time_seconds'] = export_time
        export_result['metadata']['export_timestamp'] = datetime.now().isoformat()
        
        logger.info("Export completed successfully",
                   formats=len(export_result['formats']),
                   total_files=sum(len(fmt_info['files']) for fmt_info in export_result['formats'].values()),
                   duration=f"{export_time:.2f}s")
        
        return export_result
        
    except Exception as e:
        logger.error(f"Export execution error: {str(e)}")
        if is_development():
            import traceback
            traceback.print_exc()
        raise


def _generate_export_summary(
    results: Dict[str, Any],
    output_path: Path,
    logger
) -> None:
    """Generate export summary report."""
    
    summary_path = output_path / 'export_summary.json'
    
    summary = {
        'export_metadata': results['metadata'],
        'format_summary': {},
        'file_listing': [],
        'quality_metrics': {
            'total_formats': len(results['formats']),
            'total_files': 0,
            'total_size_bytes': 0,
            'average_compression_ratio': 0.0
        }
    }
    
    total_size = 0
    total_files = 0
    
    for format_name, format_info in results['formats'].items():
        format_files = len(format_info['files'])
        format_size = format_info['total_size_bytes']
        
        summary['format_summary'][format_name] = {
            'file_count': format_files,
            'total_size_bytes': format_size,
            'size_mb': format_size / (1024 * 1024)
        }
        
        # Add individual files to listing
        for file_info in format_info['files']:
            summary['file_listing'].append({
                'format': format_name,
                'filename': file_info['filename'],
                'size_bytes': file_info['size_bytes'],
                'rows': file_info.get('rows', 0),
                'compression': file_info.get('compression')
            })
        
        total_files += format_files
        total_size += format_size
    
    summary['quality_metrics']['total_files'] = total_files
    summary['quality_metrics']['total_size_bytes'] = total_size
    
    # Write summary
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Export summary generated: {summary_path}")


def main() -> int:
    """Main entry point for loader CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up logging
    logger = get_logger(__name__)
    
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.INFO)
    elif args.debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        import logging
        logging.getLogger().setLevel(logging.WARNING)
    
    try:
        # Handle list formats command
        if args.list_formats:
            try:
                # Create a temporary loader to get supported formats
                loader = ProductionLoader()
                list_supported_formats(loader)
                return 0
            except Exception as e:
                logger.error(f"Failed to list formats: {str(e)}")
                return 1
        
        # Load configuration
        if args.config:
            logger.info(f"Using configuration file: {args.config}")
        
        # Set up paths
        input_path = Path(args.input)
        output_path = Path(args.output)
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Check for existing files if not forcing
        if not args.force:
            existing_files = list(output_path.rglob('*'))
            if existing_files:
                logger.warning(f"Output directory contains {len(existing_files)} files")
                logger.info("Use --force to overwrite existing files")
        
        # Execute export
        results = export_data(input_path, output_path, args, logger)
        
        # Generate export summary
        if args.generate_metadata:
            _generate_export_summary(results, output_path, logger)
        
        # Report success
        if results.get('success', True):
            total_files = sum(len(fmt_info['files']) for fmt_info in results['formats'].values())
            total_size_mb = sum(fmt_info['total_size_bytes'] for fmt_info in results['formats'].values()) / (1024 * 1024)
            
            logger.info("Export completed successfully",
                       formats=len(results['formats']),
                       total_files=total_files,
                       total_size=f"{total_size_mb:.1f} MB")
            return 0
        else:
            logger.error("Export failed")
            return 1
        
    except KeyboardInterrupt:
        logger.info("Export cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error in export CLI: {str(e)}")
        if is_development():
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())