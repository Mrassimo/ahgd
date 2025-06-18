#!/usr/bin/env python3
"""
Health check script for AHGD Health Analytics application.

This script performs comprehensive health checks for the application including:
- Database connectivity
- Application startup
- API endpoints
- Data integrity
- Performance benchmarks
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import sqlite3
import json
import subprocess
import urllib.request
import urllib.error

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class HealthChecker:
    """Comprehensive health checking for AHGD application."""
    
    def __init__(self, db_path: str = "health_analytics.db", base_url: str = "http://localhost:8501"):
        self.db_path = Path(db_path)
        self.base_url = base_url
        self.results: Dict[str, Any] = {}
        
    def run_all_checks(self) -> bool:
        """Run all health checks and return overall success status."""
        checks = [
            ("database", self.check_database),
            ("files", self.check_files),
            ("dependencies", self.check_dependencies),
            ("application", self.check_application),
            ("performance", self.check_performance),
        ]
        
        overall_success = True
        
        for check_name, check_func in checks:
            print(f"Running {check_name} check...")
            try:
                success = check_func()
                self.results[check_name] = {
                    "status": "PASS" if success else "FAIL",
                    "timestamp": time.time()
                }
                if not success:
                    overall_success = False
                    print(f"❌ {check_name} check failed")
                else:
                    print(f"✅ {check_name} check passed")
            except Exception as e:
                self.results[check_name] = {
                    "status": "ERROR",
                    "error": str(e),
                    "timestamp": time.time()
                }
                overall_success = False
                print(f"💥 {check_name} check error: {e}")
            
        return overall_success
    
    def check_database(self) -> bool:
        """Check database connectivity and basic structure."""
        try:
            if not self.db_path.exists():
                print(f"⚠️  Database file does not exist: {self.db_path}")
                return False
            
            # Test connection
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if we can perform basic operations
            cursor.execute("SELECT sqlite_version()")
            version = cursor.fetchone()[0]
            print(f"📊 Database version: {version}")
            
            # Check for expected tables (adapt based on your schema)
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"📋 Tables found: {len(tables)}")
            
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Database check failed: {e}")
            return False
    
    def check_files(self) -> bool:
        """Check for required files and directories."""
        required_paths = [
            "src/",
            "src/dashboard/",
            "src/dashboard/app.py",
            "pyproject.toml",
            "README.md",
        ]
        
        base_path = Path(__file__).parent.parent
        missing_paths = []
        
        for path_str in required_paths:
            path = base_path / path_str
            if not path.exists():
                missing_paths.append(str(path))
        
        if missing_paths:
            print(f"❌ Missing required files/directories: {missing_paths}")
            return False
        
        print("✅ All required files present")
        return True
    
    def check_dependencies(self) -> bool:
        """Check that critical dependencies are available."""
        critical_packages = [
            "streamlit",
            "pandas",
            "plotly",
            "folium",
            "geopandas",
        ]
        
        missing_packages = []
        
        for package in critical_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing critical packages: {missing_packages}")
            return False
        
        print("✅ All critical dependencies available")
        return True
    
    def check_application(self) -> bool:
        """Check if the application can start and respond."""
        try:
            # Try to access the health endpoint
            health_url = f"{self.base_url}/_stcore/health"
            
            try:
                with urllib.request.urlopen(health_url, timeout=10) as response:
                    if response.status == 200:
                        print("✅ Application health endpoint responding")
                        return True
            except urllib.error.URLError:
                print("⚠️  Application not running or health endpoint not accessible")
                print(f"   Tried: {health_url}")
                return False
        
        except Exception as e:
            print(f"❌ Application check failed: {e}")
            return False
    
    def check_performance(self) -> bool:
        """Basic performance checks."""
        try:
            # Test database query performance
            if self.db_path.exists():
                start_time = time.time()
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM sqlite_master")
                cursor.fetchone()
                conn.close()
                query_time = time.time() - start_time
                
                print(f"📈 Database query time: {query_time:.3f}s")
                
                if query_time > 1.0:  # Threshold: 1 second
                    print("⚠️  Database query slower than expected")
                    return False
            
            print("✅ Performance checks passed")
            return True
            
        except Exception as e:
            print(f"❌ Performance check failed: {e}")
            return False
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive health report."""
        return {
            "timestamp": time.time(),
            "overall_status": "HEALTHY" if all(
                result.get("status") == "PASS" 
                for result in self.results.values()
            ) else "UNHEALTHY",
            "checks": self.results,
            "summary": {
                "total_checks": len(self.results),
                "passed": sum(1 for r in self.results.values() if r.get("status") == "PASS"),
                "failed": sum(1 for r in self.results.values() if r.get("status") == "FAIL"),
                "errors": sum(1 for r in self.results.values() if r.get("status") == "ERROR"),
            }
        }


def main():
    """Main entry point for health checks."""
    parser = argparse.ArgumentParser(description="AHGD Health Analytics - Health Check")
    parser.add_argument(
        "--database", 
        default="health_analytics.db", 
        help="Path to database file"
    )
    parser.add_argument(
        "--url", 
        default="http://localhost:8501", 
        help="Base URL for application"
    )
    parser.add_argument(
        "--output", 
        help="Output file for health report (JSON format)"
    )
    parser.add_argument(
        "--baseline", 
        action="store_true", 
        help="Run baseline performance checks"
    )
    
    args = parser.parse_args()
    
    print("🏥 AHGD Health Analytics - Health Check")
    print("=" * 50)
    
    checker = HealthChecker(args.database, args.url)
    success = checker.run_all_checks()
    
    # Generate report
    report = checker.generate_report()
    
    print("\n📊 Health Check Summary")
    print("=" * 30)
    print(f"Overall Status: {report['overall_status']}")
    print(f"Checks Passed: {report['summary']['passed']}/{report['summary']['total_checks']}")
    
    if report['summary']['failed'] > 0:
        print(f"⚠️  Failed Checks: {report['summary']['failed']}")
    
    if report['summary']['errors'] > 0:
        print(f"💥 Error Checks: {report['summary']['errors']}")
    
    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(report, indent=2))
        print(f"📄 Report saved to: {output_path}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()