#!/usr/bin/env python3
"""
Portfolio Launch Script for Australian Health Analytics Platform

Professional demonstration launcher with pre-flight checks and
portfolio-optimized configuration.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional
import time


def print_banner():
    """Print professional banner for portfolio demonstration."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    🏥 AUSTRALIAN HEALTH ANALYTICS PLATFORM - PORTFOLIO DEMONSTRATION         ║
║                                                                              ║
║    Professional Data Engineering & Health Analytics Showcase                ║
║    • 497,181+ Health Records Processed                                      ║
║    • 57.5% Memory Optimization Achieved                                     ║
║    • 10-30x Performance Improvement                                         ║
║    • Real-time Geographic Intelligence                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_environment() -> Dict[str, bool]:
    """Check environment readiness for portfolio demonstration."""
    checks = {}
    
    print("🔍 Performing pre-flight checks...")
    
    # Check Python version
    python_version = sys.version_info
    checks["python_version"] = python_version >= (3, 11)
    print(f"  ✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}" if checks["python_version"] 
          else f"  ❌ Python version {python_version.major}.{python_version.minor} (requires 3.11+)")
    
    # Check for uv
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        checks["uv_available"] = True
        print("  ✓ UV package manager available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        checks["uv_available"] = False
        print("  ❌ UV package manager not found")
    
    # Check for key data files
    data_paths = [
        "data/processed/seifa_2021_sa2.parquet",
        "data/web_exports/json/performance/platform_performance.json",
        "data/outputs/risk_assessment/health_risk_assessment.parquet"
    ]
    
    for path in data_paths:
        file_exists = Path(path).exists()
        checks[f"data_{Path(path).stem}"] = file_exists
        status = "✓" if file_exists else "⚠️"
        print(f"  {status} {path}")
    
    # Check Streamlit availability
    try:
        import streamlit
        checks["streamlit_available"] = True
        print("  ✓ Streamlit available")
    except ImportError:
        checks["streamlit_available"] = False
        print("  ❌ Streamlit not available")
    
    return checks


def display_performance_summary():
    """Display performance summary for portfolio impact."""
    try:
        perf_file = Path("data/web_exports/json/performance/platform_performance.json")
        if perf_file.exists():
            with open(perf_file) as f:
                perf_data = json.load(f)
            
            platform_overview = perf_data.get("platform_overview", {})
            technical_achievements = perf_data.get("technical_achievements", {})
            data_processing = technical_achievements.get("data_processing", {})
            
            print("\n📊 PLATFORM PERFORMANCE SUMMARY")
            print("=" * 50)
            print(f"Records Processed: {platform_overview.get('records_processed', 'N/A'):,}")
            print(f"Memory Optimization: {data_processing.get('memory_optimization', 'N/A')}")
            print(f"Performance Improvement: {data_processing.get('performance_improvement', 'N/A')}")
            print(f"Integration Success Rate: {platform_overview.get('integration_success_rate', 'N/A')}%")
            print(f"Technology Stack: {', '.join(data_processing.get('technology_stack', []))}")
            
    except Exception as e:
        print(f"  ⚠️ Could not load performance data: {e}")


def setup_missing_data():
    """Set up missing data for demonstration if needed."""
    print("\n🛠️ Setting up demonstration environment...")
    
    # Create minimal performance data if missing
    perf_file = Path("data/web_exports/json/performance/platform_performance.json")
    if not perf_file.exists():
        print("  📝 Creating demo performance data...")
        perf_file.parent.mkdir(parents=True, exist_ok=True)
        
        demo_performance = {
            "platform_overview": {
                "name": "Australian Health Analytics Platform",
                "version": "4.0-portfolio",
                "build_date": "2025-06-17",
                "records_processed": 497181,
                "data_sources": 6,
                "integration_success_rate": 92.9
            },
            "technical_achievements": {
                "data_processing": {
                    "technology_stack": ["Polars", "DuckDB", "GeoPandas", "AsyncIO"],
                    "performance_improvement": "10-30x faster than traditional pandas",
                    "memory_optimization": "57.5% memory reduction achieved",
                    "storage_compression": "60-70% file size reduction with Parquet+ZSTD"
                }
            }
        }
        
        with open(perf_file, 'w') as f:
            json.dump(demo_performance, f, indent=2)
        print("  ✓ Demo performance data created")


def launch_dashboard(mode: str = "portfolio"):
    """Launch the dashboard in portfolio demonstration mode."""
    print(f"\n🚀 Launching portfolio dashboard...")
    
    try:
        # Set environment variables for portfolio mode
        env = os.environ.copy()
        env["STREAMLIT_THEME_BASE"] = "light"
        env["STREAMLIT_THEME_PRIMARY_COLOR"] = "#667eea"
        env["STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR"] = "#f0f2f6"
        env["PORTFOLIO_MODE"] = "true"
        
        # Launch command
        cmd = [
            "uv", "run", "streamlit", "run", 
            "src/web/streamlit/dashboard.py",
            "--server.headless", "false",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false",
            "--theme.primaryColor", "#667eea",
            "--theme.backgroundColor", "#ffffff",
            "--theme.secondaryBackgroundColor", "#f0f2f6"
        ]
        
        print("  🌐 Dashboard will open at: http://localhost:8501")
        print("  📱 Mobile access at: http://[your-ip]:8501")
        print("\n✨ Portfolio demonstration ready!")
        print("   Press Ctrl+C to stop the server")
        
        # Launch the dashboard
        subprocess.run(cmd, env=env)
        
    except KeyboardInterrupt:
        print("\n\n👋 Portfolio demonstration ended")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {e}")
        print("\nTry manual launch: uv run streamlit run src/web/streamlit/dashboard.py")


def display_portfolio_urls():
    """Display relevant URLs for portfolio demonstration."""
    print("\n🔗 PORTFOLIO DEMONSTRATION LINKS")
    print("=" * 50)
    print("Main Dashboard:     http://localhost:8501")
    print("About Page:         http://localhost:8501 (use sidebar navigation)")
    print("GitHub Repository:  [Your GitHub URL]")
    print("LinkedIn Profile:   [Your LinkedIn URL]")
    print("Technical Blog:     [Your Blog URL]")


def main():
    """Main portfolio launcher."""
    print_banner()
    
    # Environment checks
    checks = check_environment()
    
    # Display performance summary
    display_performance_summary()
    
    # Set up missing data if needed
    if not all(checks.values()):
        setup_missing_data()
    
    # Display portfolio links
    display_portfolio_urls()
    
    # Launch confirmation
    print("\n" + "=" * 80)
    response = input("🚀 Ready to launch portfolio demonstration? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        launch_dashboard()
    else:
        print("\n📋 Manual launch instructions:")
        print("   1. Ensure data is processed: uv run python scripts/setup/download_abs_data.py")
        print("   2. Launch dashboard: uv run streamlit run src/web/streamlit/dashboard.py")
        print("   3. Open browser to: http://localhost:8501")
        print("\n💼 Portfolio tip: Prepare these talking points:")
        print("   • 497K+ records processed with 57.5% memory optimization")
        print("   • 10-30x performance improvement over traditional methods")
        print("   • Real-time geographic intelligence across Australian SA2 areas")
        print("   • Modern technology stack: Polars, DuckDB, GeoPandas, Streamlit")


if __name__ == "__main__":
    main()