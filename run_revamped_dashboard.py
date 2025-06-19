#!/usr/bin/env python3
"""
🌑 Launch Script for Revamped Australian Health Analytics Dashboard

Ultra-modern dark mode dashboard with:
- Simplified 3-section navigation
- Data download capabilities
- Interactive charts and maps
- Data integrity monitoring
- Clean, professional interface
"""

import sys
import subprocess
from pathlib import Path

def launch_revamped_dashboard():
    """Launch the revamped dark mode dashboard"""
    
    print("🌑 Launching Australian Health Analytics - Dark Mode Dashboard")
    print("=" * 60)
    print("✨ Features:")
    print("  📊 Data Overview with charts and downloads")
    print("  🔍 Data Quality and integrity monitoring")
    print("  🗺️ Interactive health mapping")
    print("  🌑 Full dark mode interface")
    print("=" * 60)
    
    # Get the path to the revamped dashboard
    dashboard_path = Path(__file__).parent / "src" / "dashboard" / "revamped_app.py"
    
    if not dashboard_path.exists():
        print("❌ Error: Revamped dashboard not found!")
        print(f"Expected at: {dashboard_path}")
        return False
    
    try:
        # Launch with streamlit
        print(f"🚀 Starting dashboard...")
        print(f"📁 Dashboard location: {dashboard_path}")
        print(f"🌐 Opening in browser at: http://localhost:8501")
        print("\n💡 Tip: Use Ctrl+C to stop the dashboard")
        print("-" * 60)
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port=8501",
            "--server.address=localhost",
            "--server.headless=false",
            "--theme.base=dark",  # Enable dark theme
            "--theme.primaryColor=#2E8B57",
            "--theme.backgroundColor=#0E1117",
            "--theme.secondaryBackgroundColor=#1E1E1E"
        ])
        
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return False

if __name__ == "__main__":
    launch_revamped_dashboard()