#!/usr/bin/env python3
"""
Setup and Run Script for Australian Health Data Analytics

This script:
1. Checks and installs required dependencies
2. Runs the data processing pipeline
3. Provides feedback on setup and execution

Author: Generated for AHGD Project  
Date: 2025-06-17
"""

import subprocess
import sys
import importlib
from pathlib import Path
import os

def check_and_install_dependencies():
    """Check for required packages and install if missing."""
    required_packages = [
        'polars', 'duckdb', 'httpx', 'geopandas', 
        'folium', 'rich', 'openpyxl'
    ]
    
    missing_packages = []
    
    print("🔍 Checking dependencies...")
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - already installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - missing")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        
        # Create a minimal requirements for missing packages
        install_list = []
        package_mapping = {
            'polars': 'polars[all]',
            'duckdb': 'duckdb',
            'httpx': 'httpx',
            'geopandas': 'geopandas',
            'folium': 'folium',
            'rich': 'rich',
            'openpyxl': 'openpyxl'
        }
        
        for pkg in missing_packages:
            install_list.append(package_mapping.get(pkg, pkg))
        
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '--upgrade'
            ] + install_list)
            print("✅ Dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    else:
        print("✅ All dependencies already installed!")
        return True

def run_processing_pipeline():
    """Run the main data processing pipeline."""
    script_path = Path(__file__).parent / "scripts" / "data_processing" / "process_data.py"
    
    if not script_path.exists():
        print(f"❌ Processing script not found: {script_path}")
        return False
    
    print("\n🚀 Running data processing pipeline...")
    print("This may take several minutes to download and process Australian government data...")
    
    try:
        # Change to the project directory to ensure relative paths work
        os.chdir(Path(__file__).parent)
        
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
        
        if result.returncode == 0:
            print("✅ Data processing completed successfully!")
            print("\n📊 Output:")
            print(result.stdout)
            return True
        else:
            print("❌ Data processing failed!")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Processing timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"❌ Failed to run processing pipeline: {e}")
        return False

def main():
    """Main setup and execution function."""
    print("🇦🇺 Australian Health Data Analytics - Setup & Run")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Install dependencies
    if not check_and_install_dependencies():
        print("❌ Failed to install dependencies. Please install manually using:")
        print("uv sync  # or pip install -e .")
        sys.exit(1)
    
    # Run processing pipeline
    if not run_processing_pipeline():
        print("\n❌ Setup completed but processing failed.")
        print("You can try running the processing script directly:")
        print("python scripts/process_data.py")
        sys.exit(1)
    
    print("\n🎉 Setup and processing completed successfully!")
    print("\nNext steps:")
    print("1. Check the generated map: docs/assets/initial_map.html")
    print("2. Explore the database: data/health_analytics.db")
    print("3. Review the processed data in: data/processed/")

if __name__ == "__main__":
    main()