"""
Australian Health Analytics Dashboard - DEPRECATED WRAPPER

⚠️  DEPRECATION NOTICE: This file is now a thin wrapper around the new modular dashboard architecture.
    The main application logic has been moved to src/dashboard/app.py for better maintainability.

New Architecture:
- Core application: src/dashboard/app.py
- UI components: src/dashboard/ui/
- Data processing: src/dashboard/data/
- Visualisations: src/dashboard/visualisation/

This wrapper maintains backward compatibility but will be removed in a future version.
Please update your deployment scripts to use: python -m streamlit run src/dashboard/app.py

Author: Portfolio Demonstration
Date: June 2025
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import and run the new modular application
from src.dashboard.app import main

# Deprecation warning for developers
import streamlit as st

st.warning("""
⚠️ **Deprecation Notice**: You are using the legacy dashboard entry point.

**Current:** `scripts/streamlit_dashboard.py` (deprecated)  
**New:** `src/dashboard/app.py` (recommended)

Please update your deployment to use the new modular architecture for better performance and maintainability.
""")

def legacy_main():
    """Legacy main function - now delegates to new modular app"""
    main()


if __name__ == "__main__":
    legacy_main()