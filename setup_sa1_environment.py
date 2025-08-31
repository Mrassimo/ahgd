#!/usr/bin/env python3
"""
SA1 Environment Setup Script

Prepares the development environment for SA1-level data processing
by installing dependencies and setting up necessary directories.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a shell command with error handling."""
    print(f"\n🔧 {description}")
    print(f"Running: {command}")

    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def setup_directories():
    """Create necessary directories for SA1 processing."""
    print("\n📁 Setting up directories...")

    directories = [
        "logs",
        "data/raw/sa1",
        "data/processed/sa1",
        "data/temp",
        "pipelines/dbt/target",
        "reports/sa1_migration",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {directory}")


def install_dependencies():
    """Install required Python dependencies."""
    print("\n📦 Installing dependencies...")

    # Install the updated requirements
    success = run_command(
        f"{sys.executable} -m pip install -e .", "Installing AHGD package with new dependencies"
    )

    if not success:
        print("❌ Failed to install dependencies")
        return False

    # Verify key dependencies are installed
    key_deps = ["dlt", "dbt-duckdb", "pydantic", "geopandas", "shapely"]

    for dep in key_deps:
        try:
            __import__(dep.replace("-", "_"))
            print(f"✅ {dep} is available")
        except ImportError:
            print(f"❌ {dep} is not available")
            return False

    return True


def setup_dbt():
    """Initialize DBT project."""
    print("\n🛠️ Setting up DBT...")

    # Navigate to DBT directory
    dbt_dir = Path("pipelines/dbt")

    if not dbt_dir.exists():
        print("❌ DBT directory not found")
        return False

    # Initialize DBT (if not already done)
    os.chdir(dbt_dir)

    # Create DBT profiles directory if it doesn't exist
    profiles_dir = Path.home() / ".dbt"
    profiles_dir.mkdir(exist_ok=True)

    # Copy profiles.yml to user directory if it doesn't exist
    user_profiles = profiles_dir / "profiles.yml"
    local_profiles = Path("profiles.yml")

    if local_profiles.exists() and not user_profiles.exists():
        import shutil

        shutil.copy(local_profiles, user_profiles)
        print("✅ DBT profiles.yml copied to ~/.dbt/")

    # Return to project root
    os.chdir(Path(__file__).parent)

    return True


def test_environment():
    """Test that the environment is set up correctly."""
    print("\n🧪 Testing environment...")

    # Test DLT
    try:
        import dlt

        print("✅ DLT import successful")
    except ImportError as e:
        print(f"❌ DLT import failed: {e}")
        return False

    # Test DBT
    result = run_command("dbt --version", "Testing DBT installation")
    if not result:
        return False

    # Test Pydantic models
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from src.models.geographic import SA1Boundary
        from src.models.seifa import SEIFARecord

        print("✅ Pydantic models import successful")
    except ImportError as e:
        print(f"❌ Pydantic models import failed: {e}")
        return False

    # Test DuckDB with spatial extensions
    try:
        import duckdb

        conn = duckdb.connect(":memory:")
        conn.execute("INSTALL spatial")
        conn.execute("LOAD spatial")
        conn.close()
        print("✅ DuckDB with spatial extensions working")
    except Exception as e:
        print(f"❌ DuckDB spatial extensions failed: {e}")
        return False

    return True


def main():
    """Main setup function."""
    print("🇦🇺 AHGD SA1 Environment Setup")
    print("=" * 50)

    success_steps = []

    # Step 1: Setup directories
    setup_directories()
    success_steps.append("directories")

    # Step 2: Install dependencies
    if install_dependencies():
        success_steps.append("dependencies")
    else:
        print("\n❌ Environment setup failed at dependency installation")
        return False

    # Step 3: Setup DBT
    if setup_dbt():
        success_steps.append("dbt")
    else:
        print("\n❌ Environment setup failed at DBT setup")
        return False

    # Step 4: Test environment
    if test_environment():
        success_steps.append("testing")
    else:
        print("\n❌ Environment setup failed at testing")
        return False

    # Success message
    print("\n" + "=" * 60)
    print("🎉 SA1 ENVIRONMENT SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\n✅ Completed steps: {', '.join(success_steps)}")
    print("\nYour environment is now ready for SA1-level data processing.")
    print("\nNext steps:")
    print("1. Run the SA1 pipeline test: python test_sa1_pipeline.py")
    print("2. Execute full pipeline: python pipelines/orchestrator.py --pipeline sa1_migration")
    print("3. Launch dashboard with SA1 data: python run_dashboard.py")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
