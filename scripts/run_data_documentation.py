 """
 Script for automating data documentation tasks in the AHGD ETL pipeline.

 This script orchestrates the installation of required packages, extraction of schemas, generation of Mermaid ERD diagrams, and creation of data profiling reports for the output data.

 Usage:
     Run this script to generate documentation artifacts. It uses the output directory defined in the project configuration.
 """

import logging
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path to allow importing etl_logic
project_root = Path(__file__).resolve().parents[1] # Assuming scripts/ is one level down
sys.path.insert(0, str(project_root))

# Import config and utils after setting path
from etl_logic import config
from etl_logic import utils # Assuming setup_logging is in utils

# Setup logging
logger = utils.setup_logging(config.PATHS.get('LOG_DIR', project_root / 'logs'))

def run_command(command, description=None):
    """Run a shell command and print its output."""
    if description:
        print(f"\n{description}")
        print("=" * len(description))
    
    print(f"Executing: {command}")
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
        return False
    
    return True

def generate_data_dictionary(output_dir: Path, output_file: Path) -> bool:
    """Generates a Markdown data dictionary using output_schema_extractor.py script."""
    logger.info("Generating Data Dictionary (Markdown)...")
    script_path = project_root / "scripts/output_schema_extractor.py"
    if not script_path.exists():
        logger.error(f"Schema extractor script not found: {script_path}")
        return False
    try:
        # Run the schema extractor script which should now generate the markdown file
        # Ensure the extractor script uses config paths correctly
        result = subprocess.run([sys.executable, str(script_path)], check=True, capture_output=True, text=True)
        logger.info("Schema extractor script executed successfully.")
        logger.debug(f"Script output:\n{result.stdout}")
        # The extractor script should save the MD file to documentation_dir / data_schema_extracted.md
        expected_md_path = config.PATHS.get('DOCUMENTATION_DIR', project_root / 'documentation') / "data_schema_extracted.md"
        if expected_md_path.exists():
             logger.info(f"Markdown data dictionary generated: {expected_md_path}")
             return True
        else:
             logger.error(f"Markdown data dictionary file not found at expected location: {expected_md_path}")
             return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running schema extractor script: {e}")
        logger.error(f"Stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return False

def generate_erd_diagram(output_dir: Path, output_file: Path) -> bool:
    """Generates a Mermaid ERD diagram using output_schema_extractor.py script."""
    logger.info("Generating ERD Diagram (Mermaid .mmd)...")
    # The output_schema_extractor script should already generate this
    # We just need to confirm it exists
    expected_mermaid_path = output_dir / "data_schema_extracted.mmd"
    if expected_mermaid_path.exists():
        logger.info(f"Mermaid ERD diagram found: {expected_mermaid_path}")
        return True
    else:
        # If not found, maybe the extractor script needs to be run first?
        logger.warning(f"Mermaid file not found at {expected_mermaid_path}. Ensure schema extractor has run.")
        # Optionally, call generate_data_dictionary first if needed.
        # success = generate_data_dictionary(output_dir, Path("dummy.md")) # Rerun schema extractor
        # if success and expected_mermaid_path.exists():
        #     logger.info(f"Mermaid ERD diagram generated: {expected_mermaid_path}")
        #     return True
        logger.error(f"Mermaid ERD file not found at expected location: {expected_mermaid_path}")
        return False

def generate_profiling(output_dir: Path, reports_subdir="profiling_reports") -> bool:
    """Generates data profiling reports using generate_profiling_reports.py script."""
    logger.info("Generating Data Profiling Reports (HTML)...")
    script_path = project_root / "scripts/generate_profiling_reports.py"
    if not script_path.exists():
        logger.error(f"Profiling reports script not found: {script_path}")
        return False
    try:
        result = subprocess.run([sys.executable, str(script_path)], check=True, capture_output=True, text=True)
        logger.info("Profiling reports script executed successfully.")
        logger.debug(f"Script output:\n{result.stdout}")
        # Check if the directory was created
        profiling_dir = output_dir / reports_subdir
        if profiling_dir.exists() and any(profiling_dir.glob("*.html")):
            logger.info(f"Profiling reports generated in: {profiling_dir}")
            return True
        else:
            logger.error(f"Profiling reports directory or HTML files not found in {profiling_dir}")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running profiling reports script: {e}")
        logger.error(f"Stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during profiling: {e}")
        return False

def main(run_profiling: bool = False):
    """Main function to orchestrate documentation generation."""
    logger.info("=== Starting Data Documentation Generation ===")

    # Use paths from config
    output_dir = config.PATHS['OUTPUT_DIR']
    documentation_dir = config.PATHS.get('DOCUMENTATION_DIR', project_root / 'documentation')
    output_dir.mkdir(parents=True, exist_ok=True)
    documentation_dir.mkdir(parents=True, exist_ok=True)

    # Define output file paths (though scripts might define their own)
    # These paths are mainly for checking existence
    data_dict_file = documentation_dir / "data_schema_extracted.md"
    erd_file = output_dir / "data_schema_extracted.mmd"
    profiling_dir = output_dir / "profiling_reports"

    # Generate Data Dictionary and ERD (via schema extractor script)
    dict_success = generate_data_dictionary(output_dir, data_dict_file)
    # ERD generation relies on the previous step succeeding and creating the file
    erd_success = generate_erd_diagram(output_dir, erd_file)

    # Generate Profiling Reports if requested
    profile_success = True # Assume success if not run
    if run_profiling:
        profile_success = generate_profiling(output_dir, profiling_dir.name)

    logger.info("\n=== Documentation Generation Summary ===")
    logger.info(f"Data Dictionary: {'Success' if dict_success else 'Failed'} ({data_dict_file})")
    logger.info(f"ERD Diagram: {'Success' if erd_success else 'Failed'} ({erd_file})")
    if run_profiling:
        logger.info(f"Profiling Reports: {'Success' if profile_success else 'Failed'} ({profiling_dir})")
    else:
        logger.info("Profiling Reports: Skipped")

    if dict_success and erd_success and profile_success:
        logger.info("All documentation generated successfully.")
    else:
        logger.warning("Some documentation generation steps failed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate data documentation for the AHGD project.")
    parser.add_argument("--profile", action="store_true", help="Generate data profiling reports (can be slow).")
    args = parser.parse_args()

    main(run_profiling=args.profile) 