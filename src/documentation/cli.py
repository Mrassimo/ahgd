"""
CLI interface for AHGD documentation generation.

This module provides command-line access to the data dictionary generator
and other documentation tools.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .data_dictionary_generator import DataDictionaryGenerator
from src.utils.logging import get_logger

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for documentation CLI."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive documentation for AHGD schemas",
        prog="ahgd-generate-docs"
    )
    
    parser.add_argument(
        "--type",
        choices=["data-dictionary"],
        default="data-dictionary",
        help="Type of documentation to generate (default: data-dictionary)"
    )
    
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["markdown", "html", "csv", "all"],
        default=["all"],
        help="Output formats to generate (default: all)"
    )
    
    parser.add_argument(
        "--schemas-path",
        type=Path,
        help="Path to schemas directory (default: project schemas/)"
    )
    
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Output directory for documentation (default: docs/data_dictionary/)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except errors"
    )
    
    return parser


def generate_data_dictionary_cli(
    formats: List[str],
    schemas_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    verbose: bool = False,
    quiet: bool = False
) -> int:
    """
    CLI wrapper for data dictionary generation.
    
    Args:
        formats: List of format strings to generate
        schemas_path: Path to schemas directory
        output_path: Output directory for documentation
        verbose: Enable verbose logging
        quiet: Suppress non-error output
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Set default paths if not provided
        if schemas_path is None:
            schemas_path = Path("schemas")
        if output_path is None:
            output_path = Path("docs/data_dictionary")
        
        if not quiet:
            print(f"🏥 AHGD Data Dictionary Generator")
            print(f"Generating documentation in formats: {', '.join(formats)}")
            print(f"Schemas path: {schemas_path}")
            print(f"Output path: {output_path}")
            print()
        
        # Create generator
        generator = DataDictionaryGenerator(str(schemas_path))
        
        # Generate documentation
        if "all" in formats:
            output_files = generator.generate_all_formats(output_path)
        else:
            # Create output directory
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Analyse schemas first
            generator.analyse_schemas()
            
            output_files = {}
            for fmt in formats:
                if fmt == "markdown":
                    md_path = output_path / "data_dictionary.md"
                    generator.generate_markdown(md_path)
                    output_files["markdown"] = md_path
                elif fmt == "html":
                    html_path = output_path / "data_dictionary.html"
                    generator.generate_html(html_path)
                    output_files["html"] = html_path
                elif fmt == "csv":
                    csv_path = output_path / "data_dictionary.csv"
                    generator.generate_csv(csv_path)
                    output_files["csv"] = csv_path
        
        if not quiet:
            print("✅ Documentation generation completed successfully!")
            print()
            print("Generated files:")
            for format_name, file_path in output_files.items():
                print(f"  📄 {format_name}: {file_path}")
            
            # Provide usage suggestions
            print()
            print("💡 Usage suggestions:")
            if "markdown" in output_files:
                print(f"  • View markdown: open {output_files['markdown']}")
            if "html" in output_files:
                print(f"  • Open HTML: open {output_files['html']}")
            if "csv" in output_files:
                print(f"  • Import CSV to spreadsheet: {output_files['csv']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Documentation generation failed: {e}")
        if verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    import logging
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    # Route to appropriate generator
    if args.type == "data-dictionary":
        return generate_data_dictionary_cli(
            formats=args.formats,
            schemas_path=args.schemas_path,
            output_path=args.output_path,
            verbose=args.verbose,
            quiet=args.quiet
        )
    else:
        logger.error(f"Unknown documentation type: {args.type}")
        return 1


if __name__ == "__main__":
    sys.exit(main())