#!/usr/bin/env python3
"""
Entry point for AHGD ETL CLI.

This module serves as the main entry point for the ahgd-etl console script.
It properly imports the CLI main function while handling the package structure.
"""

import sys
import os
from pathlib import Path

def main():
    """Main entry point that sets up the environment and calls the CLI."""
    # Get the directory containing this file (src/)
    src_dir = Path(__file__).parent.resolve()
    
    # Get the project root (parent of src/)
    project_root = src_dir.parent
    
    # Add both to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    # Set PYTHONPATH environment variable
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if current_pythonpath:
        os.environ['PYTHONPATH'] = f"{src_dir}{os.pathsep}{current_pythonpath}"
    else:
        os.environ['PYTHONPATH'] = str(src_dir)
    
    # Import and run the CLI using the package structure
    try:
        from src.cli.main import main as cli_main
        cli_main()
    except ImportError:
        # Fallback to direct import if package import fails
        import cli.main
        cli.main.main()

if __name__ == "__main__":
    main()