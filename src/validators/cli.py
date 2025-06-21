#!/usr/bin/env python3
"""Command-line interface for AHGD data validation.

This module provides a command-line interface for validating processed data
using the comprehensive validation framework.

British English spelling is used throughout (optimise, standardise, etc.).
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import time
from datetime import datetime
import json

from .validation_orchestrator import ValidationOrchestrator
from ..utils.config import get_config, get_config_manager, is_development
from ..utils.logging import get_logger, monitor_performance
from ..utils.interfaces import ValidationError


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for validation CLI."""
    parser = argparse.ArgumentParser(
        description="Validate AHGD processed data for quality and compliance",
        epilog="Examples:\n"
               "  ahgd-validate --input data_processed/master_health_record.parquet --rules schemas/\n"
               "  ahgd-validate --input data_processed/ --report reports/validation_report.html\n"
               "  ahgd-validate --input data.parquet --validation-types schema business geographic",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input specification
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input data file or directory to validate"
    )
    parser.add_argument(
        "--rules",
        type=str,
        default="schemas/",
        help="Path to validation rules directory (default: schemas/)"
    )
    
    # Output options
    parser.add_argument(
        "--report", "-r",
        type=str,
        default="reports/validation_report.html",
        help="Output path for validation report (default: reports/validation_report.html)"
    )
    parser.add_argument(
        "--format",
        choices=["html", "json", "csv", "txt"],
        default="html",
        help="Report format (default: html)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for validation reports (default: reports)"
    )
    
    # Validation configuration
    parser.add_argument(
        "--validation-types",
        nargs="+",
        choices=["schema", "business", "geographic", "statistical", "quality"],
        default=["schema", "business", "geographic", "statistical", "quality"],
        help="Types of validation to perform (default: all)"
    )
    parser.add_argument(
        "--severity-threshold",
        choices=["low", "medium", "high", "critical"],
        default="medium",
        help="Minimum severity level to report (default: medium)"
    )
    parser.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="Exit with error code if validation errors are found"
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=100,
        help="Maximum number of errors to collect per validation type (default: 100)"
    )
    
    # Quality thresholds
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=85.0,
        help="Minimum overall quality threshold (default: 85.0)"
    )
    parser.add_argument(
        "--completeness-threshold",
        type=float,
        default=90.0,
        help="Minimum data completeness threshold (default: 90.0)"
    )
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=95.0,
        help="Minimum data accuracy threshold (default: 95.0)"
    )
    
    # Geographic validation options
    parser.add_argument(
        "--geographic-boundaries",
        type=str,
        help="Path to geographic boundary files for validation"
    )
    parser.add_argument(
        "--coordinate-system",
        type=str,
        default="EPSG:7844",
        help="Expected coordinate reference system (default: EPSG:7844)"
    )
    
    # Statistical validation options
    parser.add_argument(
        "--enable-outlier-detection",
        action="store_true",
        default=True,
        help="Enable statistical outlier detection (default: True)"
    )
    parser.add_argument(
        "--outlier-method",
        choices=["iqr", "zscore", "isolation"],
        default="iqr",
        help="Outlier detection method (default: iqr)"
    )
    
    # Performance options
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Sample size for large dataset validation (default: full dataset)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel validation processing"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of parallel validation workers"
    )
    
    # Configuration options
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to validation configuration file"
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["development", "testing", "production"],
        help="Validation profile to use"
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
        help="Enable debug logging and detailed error traces"
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
        help="Show what would be validated without actually doing it"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue validation even if errors are encountered (default: True)"
    )
    
    return parser


@monitor_performance("data_validation")
def validate_data(
    input_path: Path,
    rules_path: Path,
    args: argparse.Namespace,
    logger
) -> Dict[str, Any]:
    """Validate data using the ValidationOrchestrator."""
    
    logger.info("Starting data validation",
                input_path=str(input_path),
                rules_path=str(rules_path),
                validation_types=args.validation_types)
    
    if args.dry_run:
        logger.info("[DRY RUN] Would validate data with specified parameters")
        return {
            'success': True,
            'total_validations': len(args.validation_types),
            'validation_results': {},
            'overall_quality_score': 95.0
        }
    
    try:
        # Load validation configuration
        config = get_config('validation', {})
        
        # Override config with command-line arguments
        validation_config = {
            'validation_types': args.validation_types,
            'severity_threshold': args.severity_threshold,
            'max_errors_per_type': args.max_errors,
            'quality_thresholds': {
                'overall_quality': args.quality_threshold,
                'completeness': args.completeness_threshold,
                'accuracy': args.accuracy_threshold
            },
            'geographic_config': {
                'boundary_files': args.geographic_boundaries,
                'coordinate_system': args.coordinate_system
            },
            'statistical_config': {
                'enable_outlier_detection': args.enable_outlier_detection,
                'outlier_method': args.outlier_method
            },
            'performance_config': {
                'sample_size': args.sample_size,
                'parallel_processing': args.parallel,
                'max_workers': args.max_workers
            },
            'error_handling': {
                'continue_on_error': args.continue_on_error,
                'fail_on_errors': args.fail_on_errors
            }
        }
        
        # Remove None values
        def clean_config(cfg):
            if isinstance(cfg, dict):
                return {k: clean_config(v) for k, v in cfg.items() if v is not None}
            return cfg
        
        validation_config = clean_config(validation_config)
        
        # Initialize validation orchestrator
        logger.info("Initialising validation orchestrator",
                   config_keys=list(validation_config.keys()))
        
        orchestrator = ValidationOrchestrator(config=validation_config)
        
        # Check input data exists
        if not input_path.exists():
            raise ValidationError(f"Input path does not exist: {input_path}")
        
        # Execute validation
        start_time = time.time()
        
        logger.info("Executing validation suite...")
        
        validation_results = orchestrator.validate_dataset(
            input_path=str(input_path),
            rules_path=str(rules_path) if rules_path.exists() else None
        )
        
        validation_time = time.time() - start_time
        
        # Process results
        total_errors = sum(
            len(result.get('errors', [])) 
            for result in validation_results.values()
        )
        total_warnings = sum(
            len(result.get('warnings', [])) 
            for result in validation_results.values()
        )
        
        overall_quality_score = _calculate_overall_quality_score(validation_results)
        
        # Create comprehensive results summary
        results_summary = {
            'success': total_errors == 0 or not args.fail_on_errors,
            'validation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'input_path': str(input_path),
                'rules_path': str(rules_path),
                'validation_time_seconds': validation_time,
                'validation_types': args.validation_types
            },
            'validation_results': validation_results,
            'summary_statistics': {
                'total_validations': len(validation_results),
                'total_errors': total_errors,
                'total_warnings': total_warnings,
                'overall_quality_score': overall_quality_score,
                'quality_grade': _get_quality_grade(overall_quality_score)
            },
            'quality_assessment': {
                'meets_quality_threshold': overall_quality_score >= args.quality_threshold,
                'quality_threshold': args.quality_threshold,
                'detailed_scores': _extract_detailed_scores(validation_results)
            }
        }
        
        logger.info("Validation completed",
                   total_errors=total_errors,
                   total_warnings=total_warnings,
                   overall_quality=f"{overall_quality_score:.1f}%",
                   duration=f"{validation_time:.2f}s")
        
        return results_summary
        
    except Exception as e:
        logger.error(f"Validation execution error: {str(e)}")
        if is_development():
            import traceback
            traceback.print_exc()
        raise


def _calculate_overall_quality_score(validation_results: Dict[str, Any]) -> float:
    """Calculate overall quality score from validation results."""
    scores = []
    
    for validation_type, result in validation_results.items():
        quality_score = result.get('quality_score', 0.0)
        if quality_score > 0:
            scores.append(quality_score)
    
    if not scores:
        return 0.0
    
    # Weighted average - could be enhanced with type-specific weights
    return sum(scores) / len(scores)


def _get_quality_grade(score: float) -> str:
    """Convert quality score to letter grade."""
    if score >= 95:
        return "A+"
    elif score >= 90:
        return "A"
    elif score >= 85:
        return "B+"
    elif score >= 80:
        return "B"
    elif score >= 75:
        return "C+"
    elif score >= 70:
        return "C"
    elif score >= 65:
        return "D"
    else:
        return "F"


def _extract_detailed_scores(validation_results: Dict[str, Any]) -> Dict[str, float]:
    """Extract detailed quality scores by validation type."""
    detailed_scores = {}
    
    for validation_type, result in validation_results.items():
        detailed_scores[validation_type] = result.get('quality_score', 0.0)
    
    return detailed_scores


def _generate_validation_report(
    results: Dict[str, Any],
    output_path: Path,
    format_type: str,
    logger
) -> None:
    """Generate validation report in specified format."""
    
    logger.info(f"Generating {format_type} validation report",
               output_path=str(output_path))
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format_type == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    elif format_type == "html":
        _generate_html_report(results, output_path)
    
    elif format_type == "csv":
        _generate_csv_report(results, output_path)
    
    elif format_type == "txt":
        _generate_text_report(results, output_path)
    
    logger.info(f"Validation report generated: {output_path}")


def _generate_html_report(results: Dict[str, Any], output_path: Path) -> None:
    """Generate HTML validation report."""
    
    summary = results['summary_statistics']
    quality = results['quality_assessment']
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AHGD Validation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
            .summary {{ margin: 20px 0; }}
            .score {{ font-size: 24px; font-weight: bold; }}
            .pass {{ color: green; }}
            .fail {{ color: red; }}
            .warning {{ color: orange; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>AHGD Data Validation Report</h1>
            <p>Generated: {results['validation_metadata']['timestamp']}</p>
            <p>Input: {results['validation_metadata']['input_path']}</p>
        </div>
        
        <div class="summary">
            <h2>Validation Summary</h2>
            <p class="score">Overall Quality Score: {summary['overall_quality_score']:.1f}% 
               (Grade: {summary['quality_grade']})</p>
            <p>Total Errors: <span class="{'pass' if summary['total_errors'] == 0 else 'fail'}">{summary['total_errors']}</span></p>
            <p>Total Warnings: <span class="warning">{summary['total_warnings']}</span></p>
            <p>Quality Threshold Met: <span class="{'pass' if quality['meets_quality_threshold'] else 'fail'}">
               {'Yes' if quality['meets_quality_threshold'] else 'No'}</span></p>
        </div>
        
        <div class="details">
            <h2>Validation Details</h2>
            <table>
                <tr>
                    <th>Validation Type</th>
                    <th>Quality Score</th>
                    <th>Status</th>
                </tr>
    """
    
    for validation_type, score in quality['detailed_scores'].items():
        status_class = "pass" if score >= quality['quality_threshold'] else "fail"
        html_content += f"""
                <tr>
                    <td>{validation_type.title()}</td>
                    <td>{score:.1f}%</td>
                    <td class="{status_class}">{'Pass' if score >= quality['quality_threshold'] else 'Fail'}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def _generate_csv_report(results: Dict[str, Any], output_path: Path) -> None:
    """Generate CSV validation report."""
    import csv
    
    summary = results['summary_statistics']
    quality = results['quality_assessment']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write summary
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Overall Quality Score', f"{summary['overall_quality_score']:.1f}%"])
        writer.writerow(['Quality Grade', summary['quality_grade']])
        writer.writerow(['Total Errors', summary['total_errors']])
        writer.writerow(['Total Warnings', summary['total_warnings']])
        writer.writerow(['Quality Threshold Met', 'Yes' if quality['meets_quality_threshold'] else 'No'])
        writer.writerow([])
        
        # Write detailed scores
        writer.writerow(['Validation Type', 'Quality Score', 'Status'])
        for validation_type, score in quality['detailed_scores'].items():
            status = 'Pass' if score >= quality['quality_threshold'] else 'Fail'
            writer.writerow([validation_type.title(), f"{score:.1f}%", status])


def _generate_text_report(results: Dict[str, Any], output_path: Path) -> None:
    """Generate text validation report."""
    
    summary = results['summary_statistics']
    quality = results['quality_assessment']
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("AHGD Data Validation Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Generated: {results['validation_metadata']['timestamp']}\n")
        f.write(f"Input: {results['validation_metadata']['input_path']}\n\n")
        
        f.write("VALIDATION SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Overall Quality Score: {summary['overall_quality_score']:.1f}% (Grade: {summary['quality_grade']})\n")
        f.write(f"Total Errors: {summary['total_errors']}\n")
        f.write(f"Total Warnings: {summary['total_warnings']}\n")
        f.write(f"Quality Threshold Met: {'Yes' if quality['meets_quality_threshold'] else 'No'}\n\n")
        
        f.write("DETAILED SCORES\n")
        f.write("-" * 15 + "\n")
        for validation_type, score in quality['detailed_scores'].items():
            status = 'Pass' if score >= quality['quality_threshold'] else 'Fail'
            f.write(f"{validation_type.title()}: {score:.1f}% ({status})\n")


def main() -> int:
    """Main entry point for validation CLI."""
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
        # Load configuration
        if args.config:
            logger.info(f"Using configuration file: {args.config}")
        
        # Set up paths
        input_path = Path(args.input)
        rules_path = Path(args.rules)
        output_path = Path(args.report)
        
        # Determine output format from file extension if not specified
        if args.format == "html" and not output_path.suffix:
            output_path = output_path.with_suffix('.html')
        elif output_path.suffix:
            format_map = {'.json': 'json', '.csv': 'csv', '.txt': 'txt', '.html': 'html'}
            detected_format = format_map.get(output_path.suffix.lower())
            if detected_format and args.format == "html":  # Default format
                args.format = detected_format
        
        # Execute validation
        results = validate_data(input_path, rules_path, args, logger)
        
        # Generate report
        _generate_validation_report(results, output_path, args.format, logger)
        
        # Determine exit code
        if args.fail_on_errors and results['summary_statistics']['total_errors'] > 0:
            logger.error("Validation failed with errors")
            return 1
        elif not results['quality_assessment']['meets_quality_threshold']:
            logger.warning("Validation completed but quality threshold not met")
            if args.fail_on_errors:
                return 1
        
        logger.info("Validation completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Validation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error in validation CLI: {str(e)}")
        if is_development():
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())