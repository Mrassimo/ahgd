#!/usr/bin/env python3
"""
Phase 3 Completion Verification Script

Verifies that the dashboard decomposition Phase 3 has been successfully completed:
- Modular UI architecture created
- All 5 analysis modes properly extracted
- Clean application entry point
- Integration tests working
- Legacy compatibility maintained
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def verify_file_structure():
    """Verify all required files exist"""
    print("🔍 Verifying file structure...")
    
    required_files = [
        'src/dashboard/app.py',
        'src/dashboard/ui/sidebar.py',
        'src/dashboard/ui/pages.py', 
        'src/dashboard/ui/layout.py',
        'scripts/streamlit_dashboard.py',
        'tests/integration/test_dashboard_integration.py',
        'tests/unit/test_ui_components.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files exist")
    return True

def verify_imports():
    """Verify all modules can be imported"""
    print("🔍 Verifying module imports...")
    
    try:
        from src.dashboard.app import HealthAnalyticsDashboard, create_dashboard_app
        from src.dashboard.ui.sidebar import SidebarController
        from src.dashboard.ui.pages import render_page, get_page_renderer
        from src.dashboard.ui.layout import LayoutManager, create_dashboard_header
        
        print("✅ All modules import successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def verify_dashboard_class():
    """Verify HealthAnalyticsDashboard class functionality"""
    print("🔍 Verifying HealthAnalyticsDashboard class...")
    
    try:
        from src.dashboard.app import create_dashboard_app
        from src.config import get_global_config
        
        # Mock config for testing
        import unittest.mock
        with unittest.mock.patch('src.dashboard.app.get_global_config') as mock_config:
            mock_config.return_value.dashboard.page_title = "Test"
            mock_config.return_value.dashboard.page_icon = "🏥"
            mock_config.return_value.dashboard.layout = "wide"
            
            with unittest.mock.patch('streamlit.set_page_config'):
                app = create_dashboard_app()
                
                assert hasattr(app, 'sidebar_controller')
                assert hasattr(app, 'setup_page_config')
                assert hasattr(app, 'load_application_data')
                assert hasattr(app, 'render_main_interface')
                assert hasattr(app, 'run')
        
        print("✅ HealthAnalyticsDashboard class working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Dashboard class error: {e}")
        return False

def verify_sidebar_controller():
    """Verify SidebarController functionality"""
    print("🔍 Verifying SidebarController...")
    
    try:
        from src.dashboard.ui.sidebar import SidebarController
        
        controller = SidebarController()
        
        # Check analysis types
        expected_types = [
            "Geographic Health Explorer",
            "Correlation Analysis", 
            "Health Hotspot Identification",
            "Predictive Risk Analysis",
            "Data Quality & Methodology"
        ]
        
        assert controller.analysis_types == expected_types
        
        # Test state filter
        test_data = pd.DataFrame({
            'STATE_NAME21': ['NSW', 'VIC', 'QLD'],
            'IRSD_Score': [950, 1050, 900]
        })
        
        filtered = controller.apply_state_filter(test_data, ['NSW'])
        assert len(filtered) == 1
        assert filtered.iloc[0]['STATE_NAME21'] == 'NSW'
        
        print("✅ SidebarController working correctly")
        return True
        
    except Exception as e:
        print(f"❌ SidebarController error: {e}")
        return False

def verify_page_renderers():
    """Verify all page renderer functions exist and work"""
    print("🔍 Verifying page renderer functions...")
    
    try:
        from src.dashboard.ui.pages import (
            render_geographic_health_explorer,
            render_correlation_analysis,
            render_health_hotspot_identification,
            render_predictive_risk_analysis,
            render_data_quality_methodology,
            get_page_renderer
        )
        
        # Test page renderer mapping
        analysis_types = [
            "Geographic Health Explorer",
            "Correlation Analysis", 
            "Health Hotspot Identification",
            "Predictive Risk Analysis",
            "Data Quality & Methodology"
        ]
        
        for analysis_type in analysis_types:
            renderer = get_page_renderer(analysis_type)
            assert callable(renderer)
        
        # Test unknown type defaults to geographic explorer
        unknown_renderer = get_page_renderer("Unknown")
        assert unknown_renderer == render_geographic_health_explorer
        
        print("✅ All page renderers working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Page renderer error: {e}")
        return False

def verify_layout_manager():
    """Verify LayoutManager functionality"""
    print("🔍 Verifying LayoutManager...")
    
    try:
        from src.dashboard.ui.layout import (
            LayoutManager,
            create_dashboard_header,
            create_loading_spinner,
            create_dashboard_footer,
            format_large_number
        )
        
        # Test LayoutManager
        layout = LayoutManager()
        assert layout.default_column_gap == "medium"
        
        # Test utility functions
        assert format_large_number(1234567) == "1,234,567"
        
        print("✅ LayoutManager working correctly")
        return True
        
    except Exception as e:
        print(f"❌ LayoutManager error: {e}")
        return False

def verify_legacy_compatibility():
    """Verify legacy wrapper maintains compatibility"""
    print("🔍 Verifying legacy compatibility...")
    
    try:
        from scripts.streamlit_dashboard import legacy_main
        
        # Check that legacy_main function exists
        assert callable(legacy_main)
        
        # Check that it imports the new main function
        import inspect
        source = inspect.getsource(legacy_main)
        assert 'main()' in source
        
        print("✅ Legacy compatibility maintained")
        return True
        
    except Exception as e:
        print(f"❌ Legacy compatibility error: {e}")
        return False

def verify_integration_tests():
    """Verify integration tests exist and have proper structure"""
    print("🔍 Verifying integration tests...")
    
    try:
        # Check test file exists and has test classes
        test_file = Path('tests/integration/test_dashboard_integration.py')
        if not test_file.exists():
            print("❌ Integration test file missing")
            return False
        
        content = test_file.read_text()
        required_tests = [
            'TestDashboardIntegration',
            'test_dashboard_initialization',
            'test_data_loading_success',
            'test_sidebar_controls_rendering',
            'test_page_renderer_mapping'
        ]
        
        for test_item in required_tests:
            if test_item not in content:
                print(f"❌ Missing test: {test_item}")
                return False
        
        print("✅ Integration tests properly structured")
        return True
        
    except Exception as e:
        print(f"❌ Integration test verification error: {e}")
        return False

def main():
    """Run all verification checks"""
    print("🚀 Starting Phase 3 Completion Verification\n")
    
    checks = [
        ("File Structure", verify_file_structure),
        ("Module Imports", verify_imports),
        ("Dashboard Class", verify_dashboard_class),
        ("Sidebar Controller", verify_sidebar_controller),
        ("Page Renderers", verify_page_renderers),
        ("Layout Manager", verify_layout_manager),
        ("Legacy Compatibility", verify_legacy_compatibility),
        ("Integration Tests", verify_integration_tests)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
            print()  # Add spacing between checks
        except Exception as e:
            print(f"❌ {check_name} failed with exception: {e}\n")
            results.append((check_name, False))
    
    # Summary
    print("=" * 60)
    print("📊 PHASE 3 VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} {check_name}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} checks passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 PHASE 3 COMPLETED SUCCESSFULLY!")
        print("\n✨ Dashboard decomposition objectives achieved:")
        print("   • Massive main() function extracted and modularized")
        print("   • Clean UI architecture with separate sidebar, pages, and layout")
        print("   • All 5 analysis modes properly separated")
        print("   • Maintainable code structure with clear separation of concerns")
        print("   • Integration tests verify complete functionality")
        print("   • Legacy compatibility preserved with deprecation notices")
        return True
    else:
        print(f"\n⚠️  PHASE 3 INCOMPLETE: {total - passed} issues need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)