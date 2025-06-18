#!/usr/bin/env python3
"""
Standalone Performance Monitoring Dashboard

Run this script to access the performance monitoring dashboard:
    streamlit run performance_dashboard.py --server.port 8503

This provides comprehensive monitoring of the Australian Health Analytics Dashboard:
- Real-time system metrics
- Application performance tracking
- Health status monitoring
- Cache performance analytics
- Alert management
- Database performance metrics
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import performance dashboard
from src.performance.dashboard import create_performance_dashboard_page

if __name__ == "__main__":
    # Create and run the performance monitoring dashboard
    create_performance_dashboard_page()