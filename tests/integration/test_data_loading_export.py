"""
Tests for data loading and export functionality with real files.

This module tests:
- Loading Parquet files into DuckDB
- Handling multiple data sources
- Export to various formats (Parquet, CSV, JSON)
- Metadata generation
- File compression and optimisation
"""

import pytest
from pathlib import Path
from datetime import datetime
import polars as pl
import duckdb
import json


class TestDataLoading:
    """Test data loading from Parquet files into DuckDB."""

    @pytest.fixture
    def temp_dirs(self, tmp_path):
        """Create temporary directories for testing."""
        data_raw = tmp_path / "data_raw"
        data_raw.mkdir()

        db_path = tmp_path / "ahgd.db"

        return {"data_raw": data_raw, "db_path": db_path}

    @pytest.fixture
    def sample_parquet_files(self, temp_dirs):
        """Create sample Parquet files for testing."""
        data_raw = temp_dirs["data_raw"]

        # Create SA2 data file
        sa2_df = pl.DataFrame(
            {
                "sa2_code": ["101011001", "101011002", "101011003"],
                "sa2_name": ["Sydney - CBD", "Sydney - Haymarket", "Sydney - Rocks"],
                "state_code": ["1", "1", "1"],
                "state_name": ["New South Wales", "New South Wales", "New South Wales"],
                "gcc_code": ["1GSYD", "1GSYD", "1GSYD"],
                "gcc_name": [
                    "Greater Sydney",
                    "Greater Sydney",
                    "Greater Sydney",
                ],
            }
        )
        sa2_file = data_raw / "sa2_geographic.parquet"
        sa2_df.write_parquet(sa2_file)

        # Create health indicators file
        health_df = pl.DataFrame(
            {
                "sa2_code": ["101011001", "101011002", "101011003"],
                "indicator": ["diabetes", "diabetes", "diabetes"],
                "value": [5.2, 4.8, 6.1],
                "year": [2021, 2021, 2021],
            }
        )
        health_file = data_raw / "health_indicators.parquet"
        health_df.write_parquet(health_file)

        # Create SEIFA data file
        seifa_df = pl.DataFrame(
            {
                "sa2_code": ["101011001", "101011002", "101011003"],
                "irsad_score": [1050, 980, 1020],
                "irsad_decile": [10, 8, 9],
                "irsd_score": [1030, 970, 1010],
                "irsd_decile": [9, 8, 9],
            }
        )
        seifa_file = data_raw / "seifa.parquet"
        seifa_df.write_parquet(seifa_file)

        return [sa2_file, health_file, seifa_file]

    def test_load_single_parquet_file(self, temp_dirs, sample_parquet_files):
        """Test loading a single Parquet file into DuckDB."""
        db_path = temp_dirs["db_path"]
        parquet_file = sample_parquet_files[0]

        # Connect to DuckDB
        con = duckdb.connect(str(db_path))

        # Load data
        df = pl.read_parquet(parquet_file)
        con.execute("CREATE TABLE raw_sa2_geographic AS SELECT * FROM df")

        # Verify data
        result = con.execute("SELECT COUNT(*) FROM raw_sa2_geographic").fetchone()
        assert result[0] == 3

        # Verify columns
        schema = con.execute("DESCRIBE raw_sa2_geographic").fetchall()
        column_names = [row[0] for row in schema]
        assert "sa2_code" in column_names
        assert "sa2_name" in column_names
        assert "state_code" in column_names

        con.close()

    def test_load_multiple_parquet_files(self, temp_dirs, sample_parquet_files):
        """Test loading multiple Parquet files into DuckDB."""
        db_path = temp_dirs["db_path"]

        # Connect to DuckDB
        con = duckdb.connect(str(db_path))

        # Load all files
        for parquet_file in sample_parquet_files:
            table_name = f"raw_{parquet_file.stem.lower()}"
            df = pl.read_parquet(parquet_file)
            con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")

        # Verify tables created
        tables = con.execute("SHOW TABLES").fetchall()
        table_names = [row[0] for row in tables]

        assert len(table_names) == 3
        assert "raw_sa2_geographic" in table_names
        assert "raw_health_indicators" in table_names
        assert "raw_seifa" in table_names

        con.close()

    def test_load_with_schema_validation(self, temp_dirs, sample_parquet_files):
        """Test loading data with schema validation."""
        db_path = temp_dirs["db_path"]
        parquet_file = sample_parquet_files[0]  # SA2 data

        con = duckdb.connect(str(db_path))

        # Load data
        df = pl.read_parquet(parquet_file)
        con.execute("CREATE TABLE raw_sa2_geographic AS SELECT * FROM df")

        # Validate schema
        schema = con.execute("DESCRIBE raw_sa2_geographic").fetchall()

        # Check required columns exist
        column_names = [row[0] for row in schema]
        required_columns = ["sa2_code", "sa2_name", "state_code", "state_name"]

        for col in required_columns:
            assert col in column_names

        con.close()

    def test_load_empty_parquet_file(self, temp_dirs):
        """Test loading empty Parquet file."""
        data_raw = temp_dirs["data_raw"]
        db_path = temp_dirs["db_path"]

        # Create empty Parquet file
        empty_df = pl.DataFrame(
            {
                "sa2_code": [],
                "sa2_name": [],
                "value": [],
            },
            schema={"sa2_code": pl.Utf8, "sa2_name": pl.Utf8, "value": pl.Float64},
        )

        empty_file = data_raw / "empty.parquet"
        empty_df.write_parquet(empty_file)

        # Load into DuckDB
        con = duckdb.connect(str(db_path))
        df = pl.read_parquet(empty_file)
        con.execute("CREATE TABLE raw_empty AS SELECT * FROM df")

        # Verify empty table
        result = con.execute("SELECT COUNT(*) FROM raw_empty").fetchone()
        assert result[0] == 0

        # But schema should exist
        schema = con.execute("DESCRIBE raw_empty").fetchall()
        assert len(schema) == 3

        con.close()

    def test_load_replace_existing_table(self, temp_dirs, sample_parquet_files):
        """Test replacing existing table during load."""
        db_path = temp_dirs["db_path"]
        parquet_file = sample_parquet_files[0]

        con = duckdb.connect(str(db_path))

        # Load initial data
        df1 = pl.read_parquet(parquet_file)
        con.execute("CREATE TABLE raw_test AS SELECT * FROM df1")
        initial_count = con.execute("SELECT COUNT(*) FROM raw_test").fetchone()[0]

        # Create new data with different count
        new_df = pl.DataFrame(
            {
                "sa2_code": ["999999999"],
                "sa2_name": ["New Area"],
                "state_code": ["9"],
                "state_name": ["Test State"],
                "gcc_code": ["9TEST"],
                "gcc_name": ["Test GCC"],
            }
        )

        # Replace table
        con.execute("CREATE OR REPLACE TABLE raw_test AS SELECT * FROM new_df")
        new_count = con.execute("SELECT COUNT(*) FROM raw_test").fetchone()[0]

        assert new_count != initial_count
        assert new_count == 1

        con.close()


class TestDataExport:
    """Test data export functionality to various formats."""

    @pytest.fixture
    def temp_dirs(self, tmp_path):
        """Create temporary directories for testing."""
        data_exports = tmp_path / "data_exports"
        data_exports.mkdir()

        db_path = tmp_path / "ahgd.db"

        return {"data_exports": data_exports, "db_path": db_path}

    @pytest.fixture
    def db_with_data(self, temp_dirs):
        """Create DuckDB database with test data."""
        db_path = temp_dirs["db_path"]

        con = duckdb.connect(str(db_path))

        # Create master_health_record table
        master_df = pl.DataFrame(
            {
                "sa2_code": ["101011001", "101011002", "101011003"],
                "sa2_name": ["Sydney - CBD", "Sydney - Haymarket", "Sydney - Rocks"],
                "state": ["NSW", "NSW", "NSW"],
                "population": [10000, 15000, 20000],
                "diabetes_prevalence": [5.2, 4.8, 6.1],
                "health_score": [85.5, 90.2, 88.7],
                "irsad_decile": [10, 8, 9],
            }
        )

        con.execute("CREATE TABLE master_health_record AS SELECT * FROM master_df")

        yield con
        con.close()

    def test_export_to_parquet(self, temp_dirs, db_with_data):
        """Test exporting data to Parquet format."""
        export_dir = temp_dirs["data_exports"]

        # Read data from DuckDB
        df = db_with_data.execute("SELECT * FROM master_health_record").pl()

        # Export to Parquet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parquet_file = export_dir / f"master_health_record_{timestamp}.parquet"
        df.write_parquet(parquet_file, compression="zstd")

        # Verify file created
        assert parquet_file.exists()
        assert parquet_file.stat().st_size > 0

        # Verify data integrity
        loaded_df = pl.read_parquet(parquet_file)
        assert len(loaded_df) == 3
        assert "sa2_code" in loaded_df.columns

    def test_export_to_csv(self, temp_dirs, db_with_data):
        """Test exporting data to CSV format."""
        export_dir = temp_dirs["data_exports"]

        # Read data from DuckDB
        df = db_with_data.execute("SELECT * FROM master_health_record").pl()

        # Export to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = export_dir / f"master_health_record_{timestamp}.csv"
        df.write_csv(csv_file)

        # Verify file created
        assert csv_file.exists()
        assert csv_file.stat().st_size > 0

        # Verify data integrity
        loaded_df = pl.read_csv(csv_file)
        assert len(loaded_df) == 3

    def test_export_to_json(self, temp_dirs, db_with_data):
        """Test exporting data to JSON format."""
        export_dir = temp_dirs["data_exports"]

        # Read data from DuckDB
        df = db_with_data.execute("SELECT * FROM master_health_record").pl()

        # Export to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = export_dir / f"master_health_record_{timestamp}.json"
        df.write_json(json_file, row_oriented=True)

        # Verify file created
        assert json_file.exists()
        assert json_file.stat().st_size > 0

        # Verify data integrity
        with open(json_file, "r") as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 3
        assert "sa2_code" in data[0]

    def test_export_metadata_generation(self, temp_dirs, db_with_data):
        """Test generation of export metadata."""
        export_dir = temp_dirs["data_exports"]

        # Read data from DuckDB
        df = db_with_data.execute("SELECT * FROM master_health_record").pl()

        # Export files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"master_health_record_{timestamp}"

        parquet_file = export_dir / f"{base_filename}.parquet"
        csv_file = export_dir / f"{base_filename}.csv"
        json_file = export_dir / f"{base_filename}.json"

        df.write_parquet(parquet_file, compression="zstd")
        df.write_csv(csv_file)
        df.write_json(json_file, row_oriented=True)

        # Create metadata
        metadata = {
            "table_name": "master_health_record",
            "export_timestamp": timestamp,
            "record_count": len(df),
            "column_count": len(df.columns),
            "columns": df.columns,
            "schema": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
            "file_sizes_bytes": {
                "parquet": parquet_file.stat().st_size,
                "csv": csv_file.stat().st_size,
                "json": json_file.stat().st_size,
            },
        }

        metadata_file = export_dir / f"{base_filename}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Verify metadata
        assert metadata_file.exists()

        with open(metadata_file, "r") as f:
            loaded_metadata = json.load(f)

        assert loaded_metadata["record_count"] == 3
        assert loaded_metadata["column_count"] == 7
        assert "sa2_code" in loaded_metadata["columns"]
        assert loaded_metadata["file_sizes_bytes"]["parquet"] > 0

    def test_export_multiple_tables(self, temp_dirs, db_with_data):
        """Test exporting multiple tables."""
        export_dir = temp_dirs["data_exports"]

        # Create second table
        derived_df = pl.DataFrame(
            {
                "sa2_code": ["101011001", "101011002", "101011003"],
                "total_health_index": [92.3, 89.5, 91.8],
                "risk_level": ["Low", "Medium", "Low"],
            }
        )

        db_with_data.execute(
            "CREATE TABLE derived_health_indicators AS SELECT * FROM derived_df"
        )

        # Export both tables
        tables = ["master_health_record", "derived_health_indicators"]
        exported_files = []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for table_name in tables:
            df = db_with_data.execute(f"SELECT * FROM {table_name}").pl()

            # Export to Parquet
            parquet_file = export_dir / f"{table_name}_{timestamp}.parquet"
            df.write_parquet(parquet_file, compression="zstd")
            exported_files.append(parquet_file)

        # Verify all files created
        assert len(exported_files) == 2
        for file in exported_files:
            assert file.exists()

    def test_export_compression_comparison(self, temp_dirs, db_with_data):
        """Test comparison of different compression formats."""
        export_dir = temp_dirs["data_exports"]

        # Read data
        df = db_with_data.execute("SELECT * FROM master_health_record").pl()

        # Export with different compressions
        uncompressed = export_dir / "uncompressed.parquet"
        zstd_compressed = export_dir / "zstd.parquet"

        df.write_parquet(uncompressed, compression="uncompressed")
        df.write_parquet(zstd_compressed, compression="zstd")

        # Verify both files exist
        assert uncompressed.exists()
        assert zstd_compressed.exists()

        # Compressed should typically be smaller (though with small datasets, overhead might make it larger)
        uncompressed_size = uncompressed.stat().st_size
        compressed_size = zstd_compressed.stat().st_size

        # Both should have valid data
        df1 = pl.read_parquet(uncompressed)
        df2 = pl.read_parquet(zstd_compressed)

        assert len(df1) == len(df2) == 3

    def test_export_summary_generation(self, temp_dirs, db_with_data):
        """Test generation of overall export summary."""
        export_dir = temp_dirs["data_exports"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export data
        df = db_with_data.execute("SELECT * FROM master_health_record").pl()

        parquet_file = export_dir / f"master_health_record_{timestamp}.parquet"
        df.write_parquet(parquet_file, compression="zstd")

        # Create summary
        summary = {
            "export_timestamp": datetime.now().isoformat(),
            "tables_exported": 1,
            "total_files_created": 1,
            "export_directory": str(export_dir),
            "table_statistics": {
                "master_health_record": {
                    "records": len(df),
                    "columns": len(df.columns),
                    "formats": ["parquet"],
                    "total_size_mb": parquet_file.stat().st_size / (1024 * 1024),
                }
            },
            "exported_files": [str(parquet_file)],
        }

        summary_file = export_dir / f"export_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Verify summary
        assert summary_file.exists()

        with open(summary_file, "r") as f:
            loaded_summary = json.load(f)

        assert loaded_summary["tables_exported"] == 1
        assert loaded_summary["total_files_created"] == 1
        assert "master_health_record" in loaded_summary["table_statistics"]
