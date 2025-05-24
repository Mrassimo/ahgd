#!/usr/bin/env python3
"""
AHGD ETL Unified Command Line Interface

This is the single entry point for all ETL operations, consolidating functionality
from multiple legacy runners into one cohesive interface.

Usage:
    python -m ahgd_etl.cli.main [options]
    
Examples:
    # Run full pipeline with fixes
    python -m ahgd_etl.cli.main --mode full
    
    # Run specific steps
    python -m ahgd_etl.cli.main --steps download geo time
    
    # Run with validation only
    python -m ahgd_etl.cli.main --mode validate
    
    # Export to Snowflake
    python -m ahgd_etl.cli.main --mode export --target snowflake
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import os

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ahgd_etl.config import settings
from ahgd_etl.core.pipeline import Pipeline
from ahgd_etl.core.orchestrator import Orchestrator
from ahgd_etl.utils import setup_logging


class UnifiedCLI:
    """Unified command-line interface for AHGD ETL operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
        
    def parse_arguments(self) -> argparse.Namespace:
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(
            description="AHGD ETL Unified Pipeline",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self.__doc__
        )
        
        # Execution mode
        parser.add_argument(
            "--mode",
            choices=["full", "steps", "validate", "export", "fix-only"],
            default="full",
            help="Execution mode (default: full)"
        )
        
        # Specific steps
        parser.add_argument(
            "--steps",
            nargs="+",
            choices=[
                "download", "geo", "time", "dimensions",
                "g01", "g17", "g18", "g19", "g20", "g21", "g25",
                "validate", "fix"
            ],
            help="Specific steps to run (use with --mode steps)"
        )
        
        # Options
        parser.add_argument(
            "--force-download",
            action="store_true",
            help="Force re-download of data files"
        )
        
        parser.add_argument(
            "--skip-validation",
            action="store_true",
            help="Skip validation steps"
        )
        
        parser.add_argument(
            "--stop-on-error",
            action="store_true",
            help="Stop pipeline on first error"
        )
        
        parser.add_argument(
            "--fix-inline",
            action="store_true",
            default=True,
            help="Apply fixes inline during processing (default: True)"
        )
        
        # Export options
        parser.add_argument(
            "--target",
            choices=["snowflake", "parquet", "csv"],
            default="parquet",
            help="Export target format (default: parquet)"
        )
        
        parser.add_argument(
            "--snowflake-config",
            type=str,
            help="Path to Snowflake configuration file"
        )
        
        # Paths
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path("output"),
            help="Output directory for processed files"
        )
        
        parser.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            default="INFO",
            help="Logging level (default: INFO)"
        )
        
        return parser.parse_args()
    
    def get_pipeline_steps(self, args: argparse.Namespace) -> List[str]:
        """Determine which pipeline steps to run based on arguments."""
        if args.mode == "full":
            steps = [
                "download", "geo", "time", "dimensions",
                "g01", "g17", "g18", "g19", "g20", "g21", "g25"
            ]
            if not args.skip_validation:
                steps.append("validate")
            return steps
            
        elif args.mode == "steps":
            if not args.steps:
                self.logger.error("--steps required when using --mode steps")
                sys.exit(1)
            return args.steps
            
        elif args.mode == "validate":
            return ["validate"]
            
        elif args.mode == "fix-only":
            return ["fix"]
            
        elif args.mode == "export":
            # Run validation first to ensure data quality
            return ["validate", "export"]
            
        return []
    
    def run(self) -> bool:
        """Run the unified ETL pipeline."""
        args = self.parse_arguments()
        
        # Setup logging
        log_file = args.output_dir / "logs" / f"etl_{datetime.now():%Y%m%d_%H%M%S}.log"
        setup_logging(log_level=args.log_level, log_file=log_file)
        
        self.logger.info("=" * 80)
        self.logger.info("AHGD ETL Unified Pipeline Starting")
        self.logger.info("=" * 80)
        self.logger.info(f"Mode: {args.mode}")
        self.logger.info(f"Output Directory: {args.output_dir}")
        self.logger.info(f"Fix Inline: {args.fix_inline}")
        
        # Initialize pipeline components
        pipeline_config = {
            "output_dir": args.output_dir,
            "force_download": args.force_download,
            "stop_on_error": args.stop_on_error,
            "fix_inline": args.fix_inline,
            "target_format": args.target,
            "snowflake_config": args.snowflake_config
        }
        
        # Get steps to run
        steps = self.get_pipeline_steps(args)
        self.logger.info(f"Steps to run: {steps}")
        
        # Create and run orchestrator
        orchestrator = Orchestrator(pipeline_config)
        
        try:
            # Run the pipeline
            success = orchestrator.run(steps)
            
            # Export if requested
            if args.mode == "export" and success:
                self.logger.info("Starting export process...")
                export_success = orchestrator.export(args.target, args.snowflake_config)
                success = success and export_success
            
            # Print summary
            self._print_summary(orchestrator, success)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
            return False
    
    def _print_summary(self, orchestrator: Orchestrator, success: bool) -> None:
        """Print execution summary."""
        elapsed_time = time.time() - self.start_time
        
        self.logger.info("=" * 80)
        self.logger.info("EXECUTION SUMMARY")
        self.logger.info("=" * 80)
        
        # Step results
        for step_name, result in orchestrator.get_results().items():
            status = "SUCCESS" if result["success"] else "FAILED"
            duration = result.get("duration", 0)
            self.logger.info(f"{step_name:<20} {status:<10} {duration:>8.2f}s")
        
        self.logger.info("-" * 80)
        self.logger.info(f"Total Execution Time: {elapsed_time:.2f}s")
        self.logger.info(f"Overall Status: {'SUCCESS' if success else 'FAILED'}")
        self.logger.info("=" * 80)


def main():
    """Main entry point."""
    cli = UnifiedCLI()
    success = cli.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()