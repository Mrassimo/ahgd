#!/usr/bin/env python3
"""
Dashboard Launcher for Australian Health Analytics

This script provides a convenient way to launch the Streamlit dashboard
with proper error handling and environment setup.

Usage:
    python run_dashboard.py
"""

import subprocess
import sys
from pathlib import Path


def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        "streamlit",
        "folium",
        "streamlit_folium",
        "altair",
        "plotly",
        "pandas",
        "numpy",
        "geopandas",
        "pyarrow",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("📦 Install missing packages with:")
        print("   uv sync  # or pip install -e .")
        return False

    print("✅ All required packages are installed")
    return True


def check_data_files():
    """Check if required data files exist"""
    required_files = [
        "data/processed/seifa_2021_sa2.parquet",
        "data/processed/sa2_boundaries_2021.parquet",
    ]

    missing_files = []

    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print("❌ Missing required data files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n📊 Generate missing data files with:")
        print("   python scripts/process_data.py")
        return False

    print("✅ All required data files are available")
    return True


def launch_dashboard():
    """Launch the Streamlit dashboard"""
    dashboard_script = "src/dashboard/app.py"

    if not Path(dashboard_script).exists():
        print(f"❌ Dashboard script not found: {dashboard_script}")
        return False

    print("🚀 Launching Australian Health Analytics Dashboard...")
    print("📱 Dashboard will open in your web browser at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print()

    try:
        # Launch Streamlit dashboard
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                dashboard_script,
                "--server.headless",
                "false",
                "--server.port",
                "8501",
                "--server.address",
                "localhost",
            ],
            check=True,
        )

    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching dashboard: {e}")
        return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def main():
    """Main launcher function"""
    print("🏥 Australian Health Analytics Dashboard Launcher")
    print("=" * 50)

    # Check system requirements
    if not check_requirements():
        sys.exit(1)

    # Check data files
    if not check_data_files():
        print("\n💡 Tip: Run the data processing pipeline first:")
        print("   python setup_and_run.py")
        sys.exit(1)

    print("\n🎯 All checks passed! Starting dashboard...")
    print()

    # Launch dashboard
    if not launch_dashboard():
        sys.exit(1)


if __name__ == "__main__":
    main()
