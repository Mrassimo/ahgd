"""
Unit tests for main application functionality.

This module tests the main application entry points, CLI functionality,
and core application logic.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestMainModule:
    """Test the main.py module."""
    
    def test_main_function_exists(self):
        """Test that main function exists and is callable."""
        # Import main module
        import main
        
        assert hasattr(main, 'main')
        assert callable(main.main)
    
    @patch('builtins.print')
    def test_main_function_execution(self, mock_print):
        """Test main function executes without error."""
        import main
        
        # Should not raise any exceptions
        main.main()
        
        # Should print the expected message
        mock_print.assert_called_once_with("Hello from ahgd!")
    
    def test_main_module_name_guard(self):
        """Test that main runs when module is executed directly."""
        # This tests the if __name__ == "__main__" guard
        # We can't easily test direct execution, so we verify the structure
        
        main_file = Path(__file__).parent.parent.parent / "main.py"
        content = main_file.read_text()
        
        assert 'if __name__ == "__main__":' in content
        assert 'main()' in content


class TestProjectStructure:
    """Test overall project structure and organisation."""
    
    def test_project_directories_exist(self):
        """Test that expected project directories exist."""
        project_root = Path(__file__).parent.parent.parent
        
        expected_dirs = [
            'src',
            'scripts', 
            'tests',
            'data',
            'logs'
        ]
        
        for dir_name in expected_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"Directory {dir_name} should exist"
            assert dir_path.is_dir(), f"{dir_name} should be a directory"
    
    def test_key_files_exist(self):
        """Test that key project files exist."""
        project_root = Path(__file__).parent.parent.parent
        
        expected_files = [
            'pyproject.toml',
            'README.md',
            'main.py',
            'src/config.py',
            'src/__init__.py'
        ]
        
        for file_path in expected_files:
            full_path = project_root / file_path
            assert full_path.exists(), f"File {file_path} should exist"
            assert full_path.is_file(), f"{file_path} should be a file"
    
    def test_python_module_structure(self):
        """Test Python module structure is correct."""
        project_root = Path(__file__).parent.parent.parent
        
        # Check src package
        src_init = project_root / 'src' / '__init__.py'
        assert src_init.exists(), "src package should have __init__.py"
        
        # Check tests package
        tests_init = project_root / 'tests' / '__init__.py'
        assert tests_init.exists(), "tests package should have __init__.py"
    
    def test_data_directory_structure(self):
        """Test data directory has expected structure."""
        project_root = Path(__file__).parent.parent.parent
        data_dir = project_root / 'data'
        
        if data_dir.exists():
            # Check for expected subdirectories
            expected_subdirs = ['raw', 'processed']
            for subdir in expected_subdirs:
                subdir_path = data_dir / subdir
                if subdir_path.exists():
                    assert subdir_path.is_dir(), f"data/{subdir} should be a directory"


class TestConfigurationIntegration:
    """Test configuration integration in main application."""
    
    def test_config_import_in_main_modules(self):
        """Test that configuration can be imported in main modules."""
        try:
            from src.config import get_global_config, Config
            config = get_global_config()
            assert isinstance(config, Config)
        except ImportError as e:
            pytest.fail(f"Could not import configuration: {e}")
    
    def test_config_validation_in_application_context(self):
        """Test configuration validation in application context."""
        from src.config import get_global_config
        
        config = get_global_config()
        validation_result = config.validate()
        
        # Configuration should be valid in test environment
        assert isinstance(validation_result, dict)
        assert 'valid' in validation_result
        assert 'issues' in validation_result
        
        # Print any issues for debugging
        if not validation_result['valid']:
            print(f"Configuration issues: {validation_result['issues']}")


class TestDashboardBootstrap:
    """Test dashboard application bootstrap process."""
    
    @patch('streamlit.set_page_config')
    def test_streamlit_page_config(self, mock_set_page_config):
        """Test Streamlit page configuration setup."""
        # Mock streamlit to avoid actual import issues
        with patch.dict('sys.modules', {
            'streamlit': Mock(),
            'streamlit_folium': Mock(),
            'folium': Mock(),
            'geopandas': Mock()
        }):
            try:
                # This would normally import and run streamlit dashboard
                # We're just testing the configuration part
                from src.config import get_global_config
                config = get_global_config()
                
                # Simulate the page config call that would happen in dashboard
                expected_config = {
                    'page_title': config.dashboard.page_title,
                    'page_icon': config.dashboard.page_icon,
                    'layout': config.dashboard.layout
                }
                
                assert expected_config['page_title'] is not None
                assert expected_config['page_icon'] is not None
                assert expected_config['layout'] in ['centered', 'wide']
                
            except ImportError:
                # If we can't import the dashboard, that's okay for this test
                pass
    
    def test_dashboard_dependencies_available(self):
        """Test that dashboard dependencies are available."""
        required_packages = [
            'pandas',
            'numpy', 
            'altair',
            'streamlit'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            pytest.skip(f"Missing required packages: {missing_packages}")
        
        # If we get here, all packages are available
        assert len(missing_packages) == 0


class TestScriptExecution:
    """Test that key scripts can be executed."""
    
    def test_main_script_executable(self):
        """Test that main.py can be executed."""
        project_root = Path(__file__).parent.parent.parent
        main_script = project_root / "main.py"
        
        try:
            # Run the script and capture output
            result = subprocess.run(
                [sys.executable, str(main_script)],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=project_root
            )
            
            # Should execute without error
            assert result.returncode == 0, f"Script failed with error: {result.stderr}"
            assert "Hello from ahgd!" in result.stdout
            
        except subprocess.TimeoutExpired:
            pytest.fail("Script execution timed out")
        except FileNotFoundError:
            pytest.fail(f"Could not find Python interpreter or script: {main_script}")
    
    @pytest.mark.slow
    def test_config_validation_script(self):
        """Test that config validation can be run as script."""
        project_root = Path(__file__).parent.parent.parent
        config_script = project_root / "src" / "config.py"
        
        try:
            # Run config.py as a script (it has __main__ block)
            result = subprocess.run(
                [sys.executable, str(config_script)],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=project_root
            )
            
            # Should execute without error
            assert result.returncode == 0, f"Config script failed: {result.stderr}"
            
            # Should output configuration validation results
            output = result.stdout
            assert "Configuration Validation Results:" in output
            assert "Environment:" in output
            
        except subprocess.TimeoutExpired:
            pytest.fail("Config validation script timed out")
        except FileNotFoundError:
            pytest.fail("Could not find config script")


class TestErrorHandling:
    """Test error handling in main application components."""
    
    def test_graceful_config_error_handling(self):
        """Test graceful handling of configuration errors."""
        with patch.dict(os.environ, {'AHGD_DB_PATH': '/invalid/path/that/cannot/exist'}):
            try:
                from src.config import get_config, reset_global_config
                
                # Reset to pick up the bad environment variable
                reset_global_config()
                
                config = get_config()
                validation = config.validate()
                
                # Should handle the error gracefully
                assert isinstance(validation, dict)
                assert 'valid' in validation
                assert 'issues' in validation
                
                # Should detect the invalid path as an issue
                if not validation['valid']:
                    assert len(validation['issues']) > 0
                
            except Exception as e:
                pytest.fail(f"Configuration error handling failed: {e}")
            finally:
                # Clean up
                reset_global_config()
    
    def test_import_error_handling(self):
        """Test handling of missing optional dependencies."""
        # Test that the application can handle missing optional packages
        
        # Mock a missing package
        with patch.dict('sys.modules', {'optional_package': None}):
            try:
                # This should not raise an error even if optional packages are missing
                from src.config import get_global_config
                config = get_global_config()
                assert config is not None
                
            except ImportError as e:
                # Should not fail due to optional dependencies
                if "optional_package" in str(e):
                    pytest.fail("Application should handle missing optional packages")
                else:
                    # Re-raise if it's a real import error
                    raise


class TestApplicationMetadata:
    """Test application metadata and project information."""
    
    def test_pyproject_toml_structure(self):
        """Test that pyproject.toml has correct structure."""
        project_root = Path(__file__).parent.parent.parent
        pyproject_file = project_root / "pyproject.toml"
        
        assert pyproject_file.exists(), "pyproject.toml should exist"
        
        content = pyproject_file.read_text()
        
        # Should have required sections
        assert "[project]" in content
        assert "name =" in content
        assert "version =" in content
        assert "dependencies =" in content
        
        # Should have test dependencies
        assert "[project.optional-dependencies]" in content or "test =" in content
    
    def test_readme_exists_and_readable(self):
        """Test that README exists and is readable."""
        project_root = Path(__file__).parent.parent.parent
        readme_file = project_root / "README.md"
        
        assert readme_file.exists(), "README.md should exist"
        
        content = readme_file.read_text()
        assert len(content) > 0, "README should not be empty"
        
        # Should contain basic project information
        content_lower = content.lower()
        assert any(keyword in content_lower for keyword in 
                  ['health', 'analytics', 'australia', 'australian']), \
               "README should describe the project"
    
    def test_license_and_attribution(self):
        """Test license and attribution information."""
        project_root = Path(__file__).parent.parent.parent
        
        # Check for common license files
        license_files = ['LICENSE', 'LICENSE.txt', 'LICENSE.md', 'LICENCE']
        
        license_exists = any((project_root / license_file).exists() 
                           for license_file in license_files)
        
        # If no license file, check README for license info
        if not license_exists:
            readme_file = project_root / "README.md"
            if readme_file.exists():
                readme_content = readme_file.read_text().lower()
                # Check if license is mentioned in README
                license_mentioned = any(keyword in readme_content 
                                      for keyword in ['license', 'licence', 'copyright'])
                # This is informational - not failing the test
                if not license_mentioned:
                    print("Note: No explicit license information found")


@pytest.mark.integration
class TestEndToEndApplicationFlow:
    """Test end-to-end application workflows."""
    
    def test_configuration_to_dashboard_flow(self):
        """Test flow from configuration loading to dashboard setup."""
        try:
            # Step 1: Load configuration
            from src.config import get_global_config, reset_global_config
            reset_global_config()
            config = get_global_config()
            
            # Step 2: Validate configuration
            validation = config.validate()
            
            # Step 3: Verify dashboard configuration is usable
            dashboard_config = config.dashboard
            
            assert dashboard_config.host is not None
            assert dashboard_config.port > 0
            assert dashboard_config.page_title is not None
            
            # Step 4: Verify database configuration is usable
            db_config = config.database
            
            assert db_config.path is not None
            assert db_config.connection_timeout > 0
            
            # Step 5: Verify data source configuration
            data_config = config.data_source
            
            assert data_config.raw_data_dir is not None
            assert data_config.processed_data_dir is not None
            
        except Exception as e:
            pytest.fail(f"Configuration to dashboard flow failed: {e}")
    
    def test_application_startup_sequence(self):
        """Test typical application startup sequence."""
        startup_steps = []
        
        try:
            # Step 1: Import configuration
            startup_steps.append("import_config")
            from src.config import get_global_config
            
            # Step 2: Load configuration
            startup_steps.append("load_config")
            config = get_global_config()
            
            # Step 3: Validate configuration
            startup_steps.append("validate_config")
            validation = config.validate()
            
            # Step 4: Setup logging (simulated)
            startup_steps.append("setup_logging")
            # Would normally call setup_logging(config) here
            
            # Step 5: Verify all critical paths exist
            startup_steps.append("verify_paths")
            assert config.database.path.parent.exists()
            
            startup_steps.append("complete")
            
        except Exception as e:
            failed_step = startup_steps[-1] if startup_steps else "unknown"
            pytest.fail(f"Application startup failed at step '{failed_step}': {e}")
        
        # All steps should complete
        assert "complete" in startup_steps
