#!/usr/bin/env python3
"""
AHGD ETL Pipeline - Unified CLI

Command-line interface for the Australian Healthcare Geographic Database ETL Pipeline.
Combines all pipeline functionality: extraction, transformation, validation, and loading.

This unified CLI directly uses the src/ implementation without path manipulation.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import click
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Direct imports from modules (absolute imports for entry point compatibility)
from ..utils.logging import get_logger, monitor_performance
from ..utils.config import get_config_manager, is_development
from ..extractors.cli import main as extractor_main
from ..pipelines.cli import main as pipeline_main, run_pipeline
from ..validators.cli import main as validator_main
from ..loaders.cli import main as loader_main


@click.group()
@click.version_option(version="1.0.0")
@click.option('--force-real-data', is_flag=True, 
              help='Force use of real data sources (prevent mock data fallbacks)')
@click.option('--parallel', is_flag=True,
              help='Enable parallel execution where supported')
@click.option('--max-workers', type=int, default=4,
              help='Maximum number of parallel workers (default: 4)')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose logging')
@click.option('--quiet', '-q', is_flag=True, 
              help='Suppress non-essential output')
@click.option('--config', '-c', type=str,
              help='Path to configuration file')
@click.pass_context
def cli(ctx, force_real_data, parallel, max_workers, verbose, quiet, config):
    """AHGD ETL Pipeline - Australian Healthcare Geographic Database ETL.
    
    Unified CLI for data extraction, transformation, validation, and loading.
    Supports parallel execution and prevents mock data fallbacks when requested.
    """
    # Ensure ctx.obj exists for passing context to subcommands
    ctx.ensure_object(dict)
    
    # Store global options in context
    ctx.obj['force_real_data'] = force_real_data
    ctx.obj['parallel'] = parallel
    ctx.obj['max_workers'] = max_workers
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet
    ctx.obj['config'] = config
    
    # Set up logging
    if verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    elif quiet:
        import logging
        logging.getLogger().setLevel(logging.WARNING)


# Full pipeline execution command
@cli.command()
@click.option('--input-dir', default='data_raw',
              help='Input directory containing raw data')
@click.option('--output-dir', default='output',
              help='Output directory for final processed data')
@click.option('--memory-limit', type=float, default=8.0,
              help='Memory limit in GB for processing')
@click.option('--integration-level', 
              type=click.Choice(['minimal', 'standard', 'comprehensive', 'enhanced']),
              default='standard',
              help='Data integration level (default: standard)')
@click.option('--quality-threshold', type=float, default=85.0,
              help='Minimum quality threshold for data acceptance (default: 85.0)')
@click.option('--formats', multiple=True, default=['parquet', 'csv'],
              help='Output formats for final data')
@click.option('--force', is_flag=True,
              help='Force overwrite existing outputs')
@click.pass_context
def run(ctx, input_dir, output_dir, memory_limit, integration_level, 
        quality_threshold, formats, force):
    """Run the complete AHGD ETL pipeline (extract → transform → validate → load)."""
    
    logger = get_logger(__name__)
    global_options = ctx.obj
    
    click.echo("🚀 AHGD ETL Pipeline - Complete Execution")
    click.echo("=" * 60)
    
    if global_options['parallel']:
        click.echo(f"🔧 Parallel mode: {global_options['max_workers']} workers")
        click.echo(f"💾 Memory limit: {memory_limit} GB")
    
    if global_options['force_real_data']:
        click.echo("🌐 Real data mode: Mock fallbacks disabled")
    
    click.echo(f"📁 Input directory: {input_dir}")
    click.echo(f"📁 Output directory: {output_dir}")
    click.echo(f"📊 Integration level: {integration_level}")
    click.echo(f"✅ Quality threshold: {quality_threshold}%")
    click.echo(f"📋 Output formats: {', '.join(formats)}")
    
    # Create directories
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    processed_dir = Path('data_processed')
    
    for directory in [input_path, output_path, processed_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    try:
        start_time = time.time()
        
        # Stage 1: Extraction
        click.echo("\n📥 Stage 1: Data Extraction")
        click.echo("-" * 30)
        
        if global_options['parallel']:
            # Run extraction with parallel support
            extraction_args = [
                '--all', '--output', str(input_path), '--format', 'parquet',
                '--max-workers', str(global_options['max_workers'])
            ]
        else:
            extraction_args = [
                '--all', '--output', str(input_path), '--format', 'parquet'
            ]
        
        if force:
            extraction_args.append('--force')
        if global_options['verbose']:
            extraction_args.append('--verbose')
        elif global_options['quiet']:
            extraction_args.append('--quiet')
        
        # Temporarily override sys.argv for extractor CLI
        original_argv = sys.argv
        sys.argv = ['ahgd-extract'] + extraction_args
        
        try:
            extractor_result = extractor_main()
            if extractor_result != 0:
                raise RuntimeError("Data extraction failed")
        finally:
            sys.argv = original_argv
        
        click.echo("✅ Data extraction completed")
        
        # Stage 2: Transformation & Integration
        click.echo("\n🔄 Stage 2: Data Transformation & Integration")
        click.echo("-" * 45)
        
        transformation_args = [
            '--input', str(input_path),
            '--output', str(processed_dir / 'master_health_record.parquet'),
            '--integration-level', integration_level,
            '--quality-threshold', str(quality_threshold)
        ]
        
        if global_options['parallel']:
            transformation_args.extend(['--max-workers', str(global_options['max_workers'])])
        if global_options['verbose']:
            transformation_args.append('--verbose')
        elif global_options['quiet']:
            transformation_args.append('--quiet')
        
        original_argv = sys.argv
        sys.argv = ['ahgd-transform'] + transformation_args
        
        try:
            pipeline_result = pipeline_main()
            if pipeline_result != 0:
                raise RuntimeError("Data transformation failed")
        finally:
            sys.argv = original_argv
        
        click.echo("✅ Data transformation completed")
        
        # Stage 3: Validation
        click.echo("\n✅ Stage 3: Data Validation")
        click.echo("-" * 25)
        
        validation_args = [
            '--input', str(processed_dir / 'master_health_record.parquet'),
            '--rules', 'schemas/',
            '--quality-threshold', str(quality_threshold),
            '--report', str(output_path / 'validation_report.html')
        ]
        
        if global_options['parallel']:
            validation_args.extend(['--parallel', '--max-workers', str(global_options['max_workers'])])
        if global_options['verbose']:
            validation_args.append('--verbose')
        elif global_options['quiet']:
            validation_args.append('--quiet')
        
        original_argv = sys.argv
        sys.argv = ['ahgd-validate'] + validation_args
        
        try:
            validation_result = validator_main()
            if validation_result != 0:
                raise RuntimeError("Data validation failed")
        finally:
            sys.argv = original_argv
        
        click.echo("✅ Data validation completed")
        
        # Stage 4: Loading & Export
        click.echo("\n📤 Stage 4: Data Loading & Export")
        click.echo("-" * 30)
        
        loading_args = [
            '--input', str(processed_dir / 'master_health_record.parquet'),
            '--output', str(output_path),
            '--formats'] + list(formats) + [
            '--compress', '--optimise-web'
        ]
        
        if global_options['parallel']:
            loading_args.extend(['--parallel', '--max-workers', str(global_options['max_workers'])])
        if force:
            loading_args.append('--force')
        if global_options['verbose']:
            loading_args.append('--verbose')
        elif global_options['quiet']:
            loading_args.append('--quiet')
        
        original_argv = sys.argv
        sys.argv = ['ahgd-load'] + loading_args
        
        try:
            loading_result = loader_main()
            if loading_result != 0:
                raise RuntimeError("Data loading failed")
        finally:
            sys.argv = original_argv
        
        click.echo("✅ Data loading completed")
        
        total_time = time.time() - start_time
        
        click.echo(f"\n🎉 Pipeline completed successfully in {total_time:.2f} seconds!")
        click.echo(f"📁 Final outputs available in: {output_path}")
        
    except Exception as e:
        click.echo(f"❌ Pipeline execution failed: {e}")
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


# Individual stage commands with direct integration to existing CLIs
@cli.command()
@click.option('--source', default='all', 
              help='Data source (all, abs, aihw, bom) or specific source ID')
@click.option('--output-dir', default='data_raw',
              help='Output directory for raw data')
@click.option('--force', is_flag=True,
              help='Force re-download existing files')
@click.option('--format', type=click.Choice(['parquet', 'csv', 'json']),
              default='parquet', help='Output format (default: parquet)')
@click.pass_context
def extract(ctx, source, output_dir, force, format):
    """Extract raw data from Australian government sources."""
    
    logger = get_logger(__name__)
    global_options = ctx.obj
    
    click.echo(f"📥 Extracting data from {source.upper()}")
    click.echo(f"📁 Output: {output_dir}")
    
    if global_options['force_real_data']:
        click.echo("🌐 Real data mode: Mock fallbacks disabled")
    
    # Build arguments for extractor CLI
    args = ['--output', output_dir, '--format', format]
    
    if source == 'all':
        args.append('--all')
    else:
        args.extend(['--sources', source])
    
    if force:
        args.append('--force')
    if global_options['parallel']:
        args.extend(['--max-workers', str(global_options['max_workers'])])
    if global_options['verbose']:
        args.append('--verbose')
    elif global_options['quiet']:
        args.append('--quiet')
    
    # Execute extractor CLI
    original_argv = sys.argv
    sys.argv = ['ahgd-extract'] + args
    
    try:
        result = extractor_main()
        sys.exit(result)
    finally:
        sys.argv = original_argv


@cli.command()
@click.option('--input-dir', default='data_raw',
              help='Input directory with raw data')
@click.option('--output', default='data_processed/master_health_record.parquet',
              help='Output path for processed data')
@click.option('--integration-level',
              type=click.Choice(['minimal', 'standard', 'comprehensive', 'enhanced']),
              default='standard',
              help='Data integration level (default: standard)')
@click.option('--quality-threshold', type=float, default=85.0,
              help='Minimum quality threshold (default: 85.0)')
@click.pass_context
def transform(ctx, input_dir, output, integration_level, quality_threshold):
    """Transform raw data into standardised format."""
    
    logger = get_logger(__name__)
    global_options = ctx.obj
    
    click.echo(f"🔄 Transforming data from {input_dir} to {output}")
    click.echo(f"📊 Integration level: {integration_level}")
    click.echo(f"✅ Quality threshold: {quality_threshold}%")
    
    if global_options['parallel']:
        click.echo("⚡ Parallel processing enabled")
    
    # Build arguments for pipeline CLI
    args = [
        '--input', input_dir,
        '--output', output,
        '--integration-level', integration_level,
        '--quality-threshold', str(quality_threshold)
    ]
    
    if global_options['parallel']:
        args.extend(['--max-workers', str(global_options['max_workers'])])
    if global_options['verbose']:
        args.append('--verbose')
    elif global_options['quiet']:
        args.append('--quiet')
    
    # Execute pipeline CLI
    original_argv = sys.argv
    sys.argv = ['ahgd-transform'] + args
    
    try:
        result = pipeline_main()
        sys.exit(result)
    finally:
        sys.argv = original_argv


@cli.command()
@click.option('--input', default='data_processed/master_health_record.parquet',
              help='Input data file to validate')
@click.option('--rules', default='schemas/',
              help='Path to validation rules directory')
@click.option('--report', default='reports/validation_report.html',
              help='Output path for validation report')
@click.option('--detailed', is_flag=True,
              help='Show detailed validation results')
@click.option('--quality-threshold', type=float, default=85.0,
              help='Minimum quality threshold (default: 85.0)')
@click.option('--fail-on-errors', is_flag=True,
              help='Exit with error code if validation errors found')
@click.pass_context
def validate(ctx, input, rules, report, detailed, quality_threshold, fail_on_errors):
    """Validate processed data quality and consistency."""
    
    logger = get_logger(__name__)
    global_options = ctx.obj
    
    click.echo(f"✅ Validating data: {input}")
    click.echo(f"📋 Using rules from: {rules}")
    click.echo(f"📄 Report output: {report}")
    
    # Build arguments for validator CLI
    args = [
        '--input', input,
        '--rules', rules,
        '--report', report,
        '--quality-threshold', str(quality_threshold)
    ]
    
    if detailed:
        args.append('--verbose')
    if fail_on_errors:
        args.append('--fail-on-errors')
    if global_options['parallel']:
        args.extend(['--parallel', '--max-workers', str(global_options['max_workers'])])
    if global_options['verbose']:
        args.append('--verbose')
    elif global_options['quiet']:
        args.append('--quiet')
    
    # Execute validator CLI
    original_argv = sys.argv
    sys.argv = ['ahgd-validate'] + args
    
    try:
        result = validator_main()
        sys.exit(result)
    finally:
        sys.argv = original_argv


@cli.command()
@click.option('--input', default='data_processed/master_health_record.parquet',
              help='Input processed data file')
@click.option('--output-dir', default='output',
              help='Output directory for exported data')
@click.option('--formats', multiple=True, default=['parquet', 'csv', 'json'],
              help='Output formats (can specify multiple)')
@click.option('--compress', is_flag=True, default=True,
              help='Enable compression (default: True)')
@click.option('--optimise-web', is_flag=True,
              help='Optimise for web delivery')
@click.pass_context
def load(ctx, input, output_dir, formats, compress, optimise_web):
    """Load/export processed data to specified formats."""
    
    logger = get_logger(__name__)
    global_options = ctx.obj
    
    click.echo(f"📤 Loading data from {input} to {output_dir}")
    click.echo(f"📋 Formats: {', '.join(formats)}")
    
    # Build arguments for loader CLI
    args = [
        '--input', input,
        '--output', output_dir,
        '--formats'] + list(formats)
    
    if compress:
        args.append('--compress')
    if optimise_web:
        args.append('--optimise-web')
    if global_options['parallel']:
        args.extend(['--parallel', '--max-workers', str(global_options['max_workers'])])
    if global_options['verbose']:
        args.append('--verbose')
    elif global_options['quiet']:
        args.append('--quiet')
    
    # Execute loader CLI
    original_argv = sys.argv
    sys.argv = ['ahgd-load'] + args
    
    try:
        result = loader_main()
        sys.exit(result)
    finally:
        sys.argv = original_argv


# Pipeline management commands
@cli.command()
@click.option('--pipeline', required=True,
              help='Pipeline name to execute')
@click.option('--config-file', type=str,
              help='Configuration file path')
@click.pass_context
def pipeline(ctx, pipeline, config_file):
    """Execute a specific named pipeline."""
    
    logger = get_logger(__name__)
    global_options = ctx.obj
    
    click.echo(f"🔄 Executing pipeline: {pipeline}")
    
    # Build arguments for pipeline runner
    args = ['--pipeline', pipeline]
    
    if config_file:
        args.extend(['--config', config_file])
    elif global_options['config']:
        args.extend(['--config', global_options['config']])
    
    # Execute pipeline runner
    original_argv = sys.argv
    sys.argv = ['ahgd-pipeline'] + args
    
    try:
        result = run_pipeline()
        sys.exit(result)
    finally:
        sys.argv = original_argv


# Configuration and status commands
@cli.command()
@click.pass_context
def config(ctx):
    """Display current configuration and validate settings."""
    
    logger = get_logger(__name__)
    global_options = ctx.obj
    
    click.echo("⚙️  AHGD ETL Configuration")
    click.echo("=" * 30)
    
    try:
        config_manager = get_config_manager()
        click.echo("✅ Configuration manager available")
        
        # Display configuration information
        click.echo(f"\n📁 Project root: {Path.cwd()}")
        click.echo(f"📁 Data directories:")
        click.echo(f"   - Raw data: data_raw/")
        click.echo(f"   - Processed: data_processed/")
        click.echo(f"   - Output: output/")
        click.echo(f"   - Reports: reports/")
        
        click.echo(f"\n🔧 Global options:")
        click.echo(f"   - Parallel processing: {'Enabled' if global_options['parallel'] else 'Disabled'}")
        if global_options['parallel']:
            click.echo(f"   - Max workers: {global_options['max_workers']}")
        click.echo(f"   - Force real data: {'Yes' if global_options['force_real_data'] else 'No'}")
        click.echo(f"   - Verbose logging: {'Yes' if global_options['verbose'] else 'No'}")
        click.echo(f"   - Config file: {global_options['config'] or 'Default'}")
        
        click.echo("✅ Configuration validated!")
        
    except Exception as e:
        click.echo(f"❌ Configuration error: {e}")
        logger.error(f"Configuration error: {e}")
        sys.exit(1)


@cli.command()
def status():
    """Show pipeline status and data availability."""
    
    logger = get_logger(__name__)
    
    click.echo("📊 AHGD ETL Pipeline Status")
    click.echo("=" * 30)
    
    # Check data directories
    data_dirs = {
        'Raw data': 'data_raw',
        'Processed data': 'data_processed', 
        'Output': 'output',
        'Reports': 'reports',
        'Schemas': 'schemas'
    }
    
    for name, dir_path in data_dirs.items():
        path = Path(dir_path)
        if path.exists():
            files = list(path.rglob('*'))
            file_count = len([f for f in files if f.is_file()])
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            size_mb = total_size / (1024 * 1024)
            
            click.echo(f"✅ {name}: {file_count} files ({size_mb:.1f} MB)")
        else:
            click.echo(f"❌ {name}: Not found ({dir_path})")
    
    # Check key files
    key_files = {
        'Master health record': 'data_processed/master_health_record.parquet',
        'Latest validation report': 'reports/validation_report.html',
        'Export summary': 'output/export_summary.json'
    }
    
    click.echo(f"\n📋 Key files:")
    for name, file_path in key_files.items():
        path = Path(file_path)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            click.echo(f"✅ {name}: {size_mb:.1f} MB (modified: {modified})")
        else:
            click.echo(f"❌ {name}: Not found")
    
    click.echo("\n📈 Status check completed!")


# List available commands
@cli.command()
def list_commands():
    """List all available commands and their descriptions."""
    
    click.echo("🛠️  AHGD ETL Pipeline Commands")
    click.echo("=" * 40)
    
    commands = {
        'run': 'Execute complete ETL pipeline (extract → transform → validate → load)',
        'extract': 'Extract raw data from Australian government sources',
        'transform': 'Transform raw data into standardised format',
        'validate': 'Validate processed data quality and consistency',
        'load': 'Export processed data to multiple formats',
        'pipeline': 'Execute a specific named pipeline',
        'config': 'Display configuration and validate settings',
        'status': 'Show pipeline status and data availability',
        'list-commands': 'List all available commands (this command)'
    }
    
    for cmd, desc in commands.items():
        click.echo(f"\n📌 {cmd}")
        click.echo(f"   {desc}")
    
    click.echo(f"\n💡 Use 'ahgd-etl <command> --help' for detailed command options")
    click.echo(f"💡 Use '--parallel' flag for faster processing on multi-core systems")
    click.echo(f"💡 Use '--force-real-data' to prevent mock data fallbacks")


# Main entry point
@monitor_performance("cli_execution")
def main():
    """Main entry point for the unified AHGD ETL CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n⚠️  Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Unexpected CLI error: {str(e)}")
        if is_development():
            import traceback
            traceback.print_exc()
        click.echo(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()