#!/usr/bin/env python3
"""Command-line interface for AHGD data extraction.

This module provides a command-line interface for extracting data from various
Australian health and geographic data sources using the ExtractorRegistry.

British English spelling is used throughout (optimise, standardise, etc.).
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import time
from datetime import datetime

from .extractor_registry import ExtractorRegistry
from ..utils.config import get_config, get_config_manager, is_development
from ..utils.logging import get_logger, monitor_performance
from ..utils.interfaces import ExtractionError


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for extraction CLI."""
    parser = argparse.ArgumentParser(
        description="Extract data from Australian health and geographic sources",
        epilog="Examples:\n"
               "  ahgd-extract --all --output data_raw\n"
               "  ahgd-extract --sources abs_census aihw_mortality --output ./data\n"
               "  ahgd-extract --list-sources",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Output directory
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data_raw",
        help="Output directory for extracted data (default: data_raw)"
    )
    
    # Source selection
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--all",
        action="store_true",
        help="Extract from all available sources"
    )
    source_group.add_argument(
        "--sources", "-s",
        nargs="+",
        help="Specific sources to extract (space-separated list)"
    )
    source_group.add_argument(
        "--list-sources",
        action="store_true",
        help="List all available extraction sources and exit"
    )
    
    # Configuration options
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file (optional override)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of parallel extraction workers"
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=3,
        help="Number of retry attempts for failed extractions (default: 3)"
    )
    
    # Output options
    parser.add_argument(
        "--format",
        choices=["parquet", "csv", "json"],
        default="parquet",
        help="Output format for extracted data (default: parquet)"
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        default=True,
        help="Enable compression for output files (default: True)"
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
        help="Show what would be extracted without actually doing it"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if data already exists"
    )
    
    return parser


def list_available_sources(registry: ExtractorRegistry) -> None:
    """List all available extraction sources."""
    print("Available extraction sources:")
    print("=" * 50)
    
    for extractor_metadata in registry.list_extractors():
        status = "✅ Available" if extractor_metadata.enabled else "❌ Unavailable"
        source_id = extractor_metadata.extractor_type.value
        description = extractor_metadata.description
        data_category = extractor_metadata.data_category.value
        
        print(f"\n📊 {source_id}")
        print(f"   Status: {status}")
        print(f"   Description: {description}")
        print(f"   Data category: {data_category}")
        print(f"   Source: {extractor_metadata.source_organization}")
        print(f"   Update frequency: {extractor_metadata.update_frequency}")
        print(f"   Geographic coverage: {extractor_metadata.geographic_coverage}")
        print(f"   Priority: {extractor_metadata.priority}")
        print(f"   Class: {extractor_metadata.extractor_class.__name__}")


@monitor_performance("extraction_cli")
def extract_sources(
    sources: List[str],
    output_dir: Path,
    registry: ExtractorRegistry,
    args: argparse.Namespace,
    logger
) -> bool:
    """Extract data from specified sources."""
    success_count = 0
    total_sources = len(sources)
    
    logger.info("Starting data extraction", 
                sources=sources, output_dir=str(output_dir), 
                total_sources=total_sources)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extraction_results = {}
    
    for i, source_id in enumerate(sources, 1):
        logger.info(f"Extracting from source {i}/{total_sources}: {source_id}")
        
        if args.dry_run:
            logger.info(f"[DRY RUN] Would extract from {source_id}")
            success_count += 1
            continue
            
        try:
            # Get extractor instance
            extractor = registry.get_extractor(source_id)
            if not extractor:
                logger.error(f"Extractor not found: {source_id}")
                continue
            
            # Check if data already exists and force flag
            source_output_dir = output_dir / source_id
            if source_output_dir.exists() and not args.force:
                logger.info(f"Data already exists for {source_id}, skipping (use --force to re-extract)")
                success_count += 1
                continue
            
            # Perform extraction - pass empty dict as source to use default public URLs
            start_time = time.time()
            
            extraction_result = extractor.extract(
                source={},  # Use default public data sources
                output_dir=str(source_output_dir),
                format=args.format,
                compress=args.compress
            )
            
            extraction_time = time.time() - start_time
            
            if extraction_result:
                record_count = len(extraction_result) if hasattr(extraction_result, '__len__') else 'Unknown'
                logger.info(f"Successfully extracted from {source_id}",
                           records=record_count, duration=f"{extraction_time:.2f}s")
                extraction_results[source_id] = {
                    'status': 'success',
                    'records': record_count,
                    'duration': extraction_time,
                    'output_dir': str(source_output_dir)
                }
                success_count += 1
            else:
                logger.error(f"Extraction returned no data for {source_id}")
                extraction_results[source_id] = {
                    'status': 'no_data',
                    'error': 'No data returned'
                }
            
        except ExtractionError as e:
            logger.error(f"Extraction error for {source_id}: {str(e)}")
            extraction_results[source_id] = {
                'status': 'error',
                'error': str(e)
            }
        except Exception as e:
            logger.error(f"Unexpected error extracting {source_id}: {str(e)}")
            extraction_results[source_id] = {
                'status': 'error',
                'error': f"Unexpected error: {str(e)}"
            }
    
    # Generate extraction summary
    success_rate = (success_count / total_sources) * 100 if total_sources > 0 else 0
    
    logger.info("Extraction completed",
                successful=success_count, total=total_sources, 
                success_rate=f"{success_rate:.1f}%")
    
    # Write extraction report
    _write_extraction_report(output_dir, extraction_results, args)
    
    return success_count == total_sources


def _write_extraction_report(output_dir: Path, results: dict, args: argparse.Namespace) -> None:
    """Write extraction report to output directory."""
    import json
    
    report = {
        'extraction_metadata': {
            'timestamp': datetime.now().isoformat(),
            'command_args': vars(args),
            'total_sources': len(results),
            'successful_extractions': sum(1 for r in results.values() if r['status'] == 'success'),
            'failed_extractions': sum(1 for r in results.values() if r['status'] != 'success')
        },
        'source_results': results
    }
    
    report_path = output_dir / 'extraction_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def main() -> int:
    """Main entry point for extraction CLI."""
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
            # Override config file if specified
            # This would need implementation in config manager
            logger.info(f"Using configuration file: {args.config}")
        
        # Initialize extractor registry
        registry = ExtractorRegistry()
        
        # Handle list sources command
        if args.list_sources:
            list_available_sources(registry)
            return 0
        
        # Determine sources to extract
        if args.all:
            available_extractors = registry.list_extractors()
            sources = [extractor.extractor_type.value for extractor in available_extractors]
            logger.info(f"Extracting from all available sources: {sources}")
        else:
            sources = args.sources
            
        if not sources:
            logger.error("No sources specified for extraction")
            return 1
        
        # Validate sources exist
        available_extractors = registry.list_extractors()
        available_source_ids = [extractor.extractor_type.value for extractor in available_extractors]
        invalid_sources = [s for s in sources if s not in available_source_ids]
        if invalid_sources:
            logger.error(f"Invalid sources specified: {invalid_sources}")
            logger.info("Use --list-sources to see available options")
            return 1
        
        # Set up output directory
        output_dir = Path(args.output)
        
        # Perform extraction
        success = extract_sources(sources, output_dir, registry, args, logger)
        
        if success:
            logger.info("All extractions completed successfully")
            return 0
        else:
            logger.error("Some extractions failed - check logs for details")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Extraction cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error in extraction CLI: {str(e)}")
        if is_development():
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())