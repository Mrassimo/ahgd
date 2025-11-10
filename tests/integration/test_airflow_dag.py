"""
Tests for Airflow DAG structure and execution.

This module tests:
- DAG definition and configuration
- Task dependencies and ordering
- Task execution logic
- Error handling in DAG tasks
"""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Import DAG components
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dags.ahgd_pipeline import (
    load_raw_data_to_duckdb,
    export_data_from_duckdb,
)


class TestAirflowDAG:
    """Test Airflow DAG structure and configuration."""

    def test_dag_import(self):
        """Test that DAG can be imported without errors."""
        from dags.ahgd_pipeline import dag

        assert dag is not None
        assert dag.dag_id == "ahgd_etl"

    def test_dag_configuration(self):
        """Test DAG configuration parameters."""
        from dags.ahgd_pipeline import dag

        assert dag.schedule_interval is None
        assert dag.catchup is False
        assert "ahgd" in dag.tags
        assert "etl" in dag.tags

    def test_dag_task_count(self):
        """Test that DAG has correct number of tasks."""
        from dags.ahgd_pipeline import dag

        tasks = dag.tasks
        assert len(tasks) == 5, f"Expected 5 tasks, found {len(tasks)}"

    def test_dag_task_dependencies(self):
        """Test that task dependencies are correctly defined."""
        from dags.ahgd_pipeline import dag

        task_dict = {task.task_id: task for task in dag.tasks}

        # Check tasks exist
        assert "extract_data" in task_dict
        assert "load_raw_to_duckdb" in task_dict
        assert "dbt_build" in task_dict
        assert "dbt_test" in task_dict
        assert "export_final_data" in task_dict

        # Check dependencies: extract_data >> load_raw_to_duckdb
        extract_task = task_dict["extract_data"]
        load_task = task_dict["load_raw_to_duckdb"]
        assert load_task in extract_task.downstream_list

        # Check dependencies: load_raw_to_duckdb >> dbt_build
        dbt_build_task = task_dict["dbt_build"]
        assert dbt_build_task in load_task.downstream_list

        # Check dependencies: dbt_build >> dbt_test
        dbt_test_task = task_dict["dbt_test"]
        assert dbt_test_task in dbt_build_task.downstream_list

        # Check dependencies: dbt_test >> export_final_data
        export_task = task_dict["export_final_data"]
        assert export_task in dbt_test_task.downstream_list


class TestLoadRawDataToDuckDB:
    """Test DuckDB data loading functionality."""

    @pytest.fixture
    def mock_context(self):
        """Provide mock Airflow context."""
        return {
            "task_instance": Mock(),
            "execution_date": datetime(2023, 1, 1),
        }

    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create temporary data directory with test files."""
        data_dir = tmp_path / "data_raw"
        data_dir.mkdir()

        # Create test parquet file
        import polars as pl

        df = pl.DataFrame(
            {
                "sa2_code": ["101011001", "101011002"],
                "sa2_name": ["Sydney - CBD", "Sydney - Haymarket"],
                "value": [100, 200],
            }
        )

        parquet_file = data_dir / "test_data.parquet"
        df.write_parquet(parquet_file)

        return data_dir

    @patch("dags.ahgd_pipeline.Path")
    @patch("dags.ahgd_pipeline.duckdb")
    @patch("dags.ahgd_pipeline.glob")
    def test_load_raw_data_success(
        self, mock_glob, mock_duckdb, mock_path, mock_context, temp_data_dir
    ):
        """Test successful data loading to DuckDB."""
        # Setup mocks
        mock_db_path = "/tmp/test.db"
        mock_path.return_value = mock_db_path
        mock_glob.return_value = [str(temp_data_dir / "test_data.parquet")]

        mock_con = MagicMock()
        mock_duckdb.connect.return_value = mock_con
        mock_con.execute.return_value.fetchall.return_value = [("raw_test_data",)]

        # Execute function
        result = load_raw_data_to_duckdb(**mock_context)

        # Assertions
        assert result["status"] == "success"
        assert result["tables_created"] >= 0
        assert "total_records" in result
        assert "table_list" in result

    @patch("dags.ahgd_pipeline.Path")
    @patch("dags.ahgd_pipeline.glob")
    def test_load_raw_data_no_files(self, mock_glob, mock_path, mock_context):
        """Test loading when no Parquet files are found."""
        # Setup mocks
        mock_glob.return_value = []

        # Execute function
        result = load_raw_data_to_duckdb(**mock_context)

        # Assertions
        assert result["status"] == "warning"
        assert "No data files found" in result["message"]
        assert result["tables_created"] == 0

    @patch("dags.ahgd_pipeline.Path")
    @patch("dags.ahgd_pipeline.duckdb")
    def test_load_raw_data_connection_error(
        self, mock_duckdb, mock_path, mock_context
    ):
        """Test error handling when DuckDB connection fails."""
        # Setup mocks
        mock_duckdb.connect.side_effect = Exception("Connection failed")

        # Execute function - should raise exception
        with pytest.raises(Exception, match="Connection failed"):
            load_raw_data_to_duckdb(**mock_context)


class TestExportDataFromDuckDB:
    """Test data export functionality."""

    @pytest.fixture
    def mock_context(self):
        """Provide mock Airflow context."""
        return {
            "task_instance": Mock(),
            "execution_date": datetime(2023, 1, 1),
        }

    @pytest.fixture
    def temp_export_dir(self, tmp_path):
        """Create temporary export directory."""
        export_dir = tmp_path / "data_exports"
        export_dir.mkdir()
        return export_dir

    @patch("dags.ahgd_pipeline.Path")
    @patch("dags.ahgd_pipeline.duckdb")
    def test_export_data_success(
        self, mock_duckdb, mock_path, mock_context, temp_export_dir
    ):
        """Test successful data export from DuckDB."""
        import polars as pl

        # Setup mocks
        mock_path.return_value = temp_export_dir
        mock_con = MagicMock()
        mock_duckdb.connect.return_value = mock_con

        # Mock table check
        mock_con.execute.return_value.fetchone.return_value = (1,)

        # Mock data retrieval
        test_df = pl.DataFrame(
            {
                "sa2_code": ["101011001", "101011002"],
                "indicator": ["health_status", "health_status"],
                "value": [85.5, 90.2],
            }
        )
        mock_con.execute.return_value.pl.return_value = test_df

        # Execute function
        result = export_data_from_duckdb(**mock_context)

        # Assertions
        assert "tables_exported" in result
        assert "total_files_created" in result
        assert "table_statistics" in result

    @patch("dags.ahgd_pipeline.Path")
    @patch("dags.ahgd_pipeline.duckdb")
    def test_export_data_no_tables(self, mock_duckdb, mock_path, mock_context):
        """Test export when no tables exist."""
        # Setup mocks
        mock_con = MagicMock()
        mock_duckdb.connect.return_value = mock_con

        # Mock table check - no tables found
        mock_con.execute.return_value.fetchone.return_value = (0,)

        # Execute function
        result = export_data_from_duckdb(**mock_context)

        # Assertions
        assert result["tables_exported"] == 0

    @patch("dags.ahgd_pipeline.Path")
    @patch("dags.ahgd_pipeline.duckdb")
    def test_export_data_connection_error(
        self, mock_duckdb, mock_path, mock_context
    ):
        """Test error handling when DuckDB connection fails."""
        # Setup mocks
        mock_duckdb.connect.side_effect = Exception("Connection failed")

        # Execute function - should raise exception
        with pytest.raises(Exception, match="Connection failed"):
            export_data_from_duckdb(**mock_context)


class TestDAGIntegration:
    """Integration tests for complete DAG execution."""

    def test_dag_import_no_syntax_errors(self):
        """Test that DAG file has no syntax errors."""
        from dags import ahgd_pipeline

        assert hasattr(ahgd_pipeline, "dag")
        assert hasattr(ahgd_pipeline, "load_raw_data_to_duckdb")
        assert hasattr(ahgd_pipeline, "export_data_from_duckdb")

    def test_dag_task_operators(self):
        """Test that tasks use correct operators."""
        from dags.ahgd_pipeline import dag
        from airflow.operators.bash import BashOperator
        from airflow.operators.python import PythonOperator

        task_dict = {task.task_id: task for task in dag.tasks}

        # BashOperators
        assert isinstance(task_dict["extract_data"], BashOperator)
        assert isinstance(task_dict["dbt_build"], BashOperator)
        assert isinstance(task_dict["dbt_test"], BashOperator)

        # PythonOperators
        assert isinstance(task_dict["load_raw_to_duckdb"], PythonOperator)
        assert isinstance(task_dict["export_final_data"], PythonOperator)

    def test_dag_task_commands(self):
        """Test that Bash tasks have correct commands."""
        from dags.ahgd_pipeline import dag

        task_dict = {task.task_id: task for task in dag.tasks}

        # Check extract command
        extract_cmd = task_dict["extract_data"].bash_command
        assert "python" in extract_cmd
        assert "extract" in extract_cmd
        assert "--all" in extract_cmd

        # Check dbt commands
        dbt_build_cmd = task_dict["dbt_build"].bash_command
        assert "dbt build" in dbt_build_cmd

        dbt_test_cmd = task_dict["dbt_test"].bash_command
        assert "dbt test" in dbt_test_cmd
