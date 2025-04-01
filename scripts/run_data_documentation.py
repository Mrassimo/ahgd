import os
import subprocess
import sys

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

def main():
    """Main function to run all scripts."""
    print("===== DATA DOCUMENTATION TOOLKIT =====")
    print("This script will:")
    print("1. Install required packages")
    print("2. Extract and print Parquet file schemas")
    print("3. Generate Mermaid ERD diagrams")
    print("4. Create data profiling reports")
    print("=" * 39)
    
    # Install required packages
    if not run_command(
        "pip install polars pyarrow ydata-profiling", 
        "Installing required packages"
    ):
        print("Failed to install required packages. Exiting.")
        sys.exit(1)
    
    # Run schema extraction and Mermaid ERD generation
    if not run_command(
        "python output_schema_extractor.py", 
        "Extracting schemas and generating Mermaid ERD"
    ):
        print("Schema extraction failed. Continuing to next step.")
    
    # Run data profiling
    if not run_command(
        "python generate_profiling_reports.py", 
        "Generating data profiling reports"
    ):
        print("Data profiling failed.")
        sys.exit(1)
    
    print("\n===== PROCESS COMPLETED =====")
    print("The following artifacts have been generated:")
    print("1. Schema information (printed to console)")
    print(f"2. Mermaid ERD diagram: /Users/massimoraso/AHGD3/output/data_schema.mmd")
    print(f"3. Data profiling reports: /Users/massimoraso/AHGD3/output/profiling_reports/*.html")
    print("\nYou can open the .mmd file in a Mermaid editor or viewer.")
    print("The HTML profiling reports can be opened in any web browser.")

if __name__ == "__main__":
    main() 