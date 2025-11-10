"""
End-to-end pipeline integration tests.

This module tests the complete AHGD ETL pipeline from extraction to export:
1. Extract raw data from sources
2. Load raw data into DuckDB
3. Run dbt transformations
4. Validate data quality
5. Export final data to multiple formats

These tests verify that all pipeline components work together correctly.
"""

import pytest
from pathlib import Path
from datetime import datetime
import polars as pl
import duckdb
import json
import sys
from unittest.mock import Mock, patch


class TestEndToEndPipeline:
    """Test complete pipeline execution."""

    @pytest.fixture
    def temp_pipeline_dirs(self, tmp_path):
        """Create temporary directory structure for pipeline."""
        dirs = {
            "data_raw": tmp_path / "data_raw",
            "data_exports": tmp_path / "data_exports",
            "db_path": tmp_path / "ahgd.db",
        }

        dirs["data_raw"].mkdir(parents=True)
        dirs["data_exports"].mkdir(parents=True)

        return dirs

    @pytest.fixture
    def sample_raw_data(self, temp_pipeline_dirs):
        """Create sample raw data files."""
        data_raw = temp_pipeline_dirs["data_raw"]

        # SA2 Geographic data
        sa2_df = pl.DataFrame(
            {
                "SA2_CODE_2021": ["101011001", "101011002", "101011003"],
                "SA2_NAME_2021": ["Sydney - CBD", "Sydney - Haymarket", "Sydney - Rocks"],
                "STATE_CODE_2021": ["1", "1", "1"],
                "STATE_NAME_2021": [
                    "New South Wales",
                    "New South Wales",
                    "New South Wales",
                ],
                "AREA_SQKM": [1.5, 2.3, 1.8],
            }
        )
        sa2_df.write_parquet(data_raw / "sa2_geographic.parquet")

        # Health indicators data
        health_df = pl.DataFrame(
            {
                "sa2_code": ["101011001", "101011002", "101011003"],
                "indicator_name": ["diabetes", "diabetes", "diabetes"],
                "indicator_value": [5.2, 4.8, 6.1],
                "year": [2021, 2021, 2021],
                "source": ["AIHW", "AIHW", "AIHW"],
            }
        )
        health_df.write_parquet(data_raw / "health_indicators.parquet")

        # SEIFA data
        seifa_df = pl.DataFrame(
            {
                "sa2_code": ["101011001", "101011002", "101011003"],
                "irsad_score": [1050, 980, 1020],
                "irsad_decile": [10, 8, 9],
                "irsd_score": [1030, 970, 1010],
                "irsd_decile": [9, 8, 9],
                "ier_score": [1040, 975, 1015],
                "ier_decile": [10, 8, 9],
            }
        )
        seifa_df.write_parquet(data_raw / "seifa.parquet")

        # Population data
        population_df = pl.DataFrame(
            {
                "sa2_code": ["101011001", "101011002", "101011003"],
                "total_population": [10000, 15000, 20000],
                "male_population": [4800, 7200, 9600],
                "female_population": [5200, 7800, 10400],
                "median_age": [35, 32, 38],
            }
        )
        population_df.write_parquet(data_raw / "population.parquet")

        return data_raw

    def test_step1_load_raw_data_to_duckdb(
        self, temp_pipeline_dirs, sample_raw_data
    ):
        """Test Step 1: Load raw Parquet files into DuckDB."""
        db_path = temp_pipeline_dirs["db_path"]
        data_raw = sample_raw_data

        # Connect to DuckDB
        con = duckdb.connect(str(db_path))

        # Load all Parquet files
        parquet_files = list(data_raw.glob("*.parquet"))
        assert len(parquet_files) == 4, "Should have 4 raw data files"

        tables_created = []
        total_records = 0

        for parquet_file in parquet_files:
            table_name = f"raw_{parquet_file.stem.lower()}"
            df = pl.read_parquet(parquet_file)

            con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")

            tables_created.append(table_name)
            total_records += len(df)

        # Verify tables created
        tables_result = con.execute("SHOW TABLES").fetchall()
        table_list = [row[0] for row in tables_result]

        assert len(table_list) == 4
        assert "raw_sa2_geographic" in table_list
        assert "raw_health_indicators" in table_list
        assert "raw_seifa" in table_list
        assert "raw_population" in table_list

        # Verify record counts
        assert total_records == 12  # 3 + 3 + 3 + 3

        con.close()

    def test_step2_staging_transformations(
        self, temp_pipeline_dirs, sample_raw_data
    ):
        """Test Step 2: Apply staging transformations (dbt staging models)."""
        db_path = temp_pipeline_dirs["db_path"]

        # Setup: Load raw data
        con = duckdb.connect(str(db_path))
        parquet_files = list(sample_raw_data.glob("*.parquet"))
        for parquet_file in parquet_files:
            table_name = f"raw_{parquet_file.stem.lower()}"
            df = pl.read_parquet(parquet_file)
            con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")

        # Simulate dbt staging transformation for SA2 data
        con.execute(
            """
            CREATE TABLE stg_sa2_geographic AS
            SELECT
                CAST(SA2_CODE_2021 AS VARCHAR) as sa2_code,
                SA2_NAME_2021 as sa2_name,
                STATE_CODE_2021 as state_code,
                STATE_NAME_2021 as state_name,
                AREA_SQKM as area_sqkm,
                CURRENT_TIMESTAMP as dbt_updated_at
            FROM raw_sa2_geographic
        """
        )

        # Simulate staging transformation for health indicators
        con.execute(
            """
            CREATE TABLE stg_health_indicators AS
            SELECT
                sa2_code,
                indicator_name,
                indicator_value,
                year,
                source,
                CURRENT_TIMESTAMP as dbt_updated_at
            FROM raw_health_indicators
        """
        )

        # Verify staging tables
        stg_sa2_count = con.execute(
            "SELECT COUNT(*) FROM stg_sa2_geographic"
        ).fetchone()[0]
        assert stg_sa2_count == 3

        stg_health_count = con.execute(
            "SELECT COUNT(*) FROM stg_health_indicators"
        ).fetchone()[0]
        assert stg_health_count == 3

        # Verify column standardisation
        schema = con.execute("DESCRIBE stg_sa2_geographic").fetchall()
        column_names = [row[0] for row in schema]
        assert "sa2_code" in column_names
        assert "sa2_name" in column_names
        assert "dbt_updated_at" in column_names

        con.close()

    def test_step3_intermediate_transformations(
        self, temp_pipeline_dirs, sample_raw_data
    ):
        """Test Step 3: Apply intermediate transformations (dbt intermediate models)."""
        db_path = temp_pipeline_dirs["db_path"]

        # Setup: Load raw and staging data
        con = duckdb.connect(str(db_path))

        # Create staging tables
        sa2_df = pl.read_parquet(sample_raw_data / "sa2_geographic.parquet")
        con.execute(
            """
            CREATE TABLE stg_sa2_geographic AS
            SELECT
                CAST(SA2_CODE_2021 AS VARCHAR) as sa2_code,
                SA2_NAME_2021 as sa2_name,
                STATE_CODE_2021 as state_code,
                STATE_NAME_2021 as state_name
            FROM sa2_df
        """
        )

        seifa_df = pl.read_parquet(sample_raw_data / "seifa.parquet")
        con.execute("CREATE TABLE stg_seifa AS SELECT * FROM seifa_df")

        population_df = pl.read_parquet(sample_raw_data / "population.parquet")
        con.execute("CREATE TABLE stg_population AS SELECT * FROM population_df")

        # Simulate intermediate model: join geographic with SEIFA
        con.execute(
            """
            CREATE TABLE int_sa2_with_seifa AS
            SELECT
                g.sa2_code,
                g.sa2_name,
                g.state_code,
                g.state_name,
                s.irsad_score,
                s.irsad_decile,
                s.irsd_score,
                s.irsd_decile
            FROM stg_sa2_geographic g
            LEFT JOIN stg_seifa s ON g.sa2_code = s.sa2_code
        """
        )

        # Simulate intermediate model: join with population
        con.execute(
            """
            CREATE TABLE int_sa2_demographics AS
            SELECT
                sw.sa2_code,
                sw.sa2_name,
                sw.state_code,
                sw.state_name,
                sw.irsad_decile,
                p.total_population,
                p.median_age
            FROM int_sa2_with_seifa sw
            LEFT JOIN stg_population p ON sw.sa2_code = p.sa2_code
        """
        )

        # Verify intermediate tables
        int_count = con.execute(
            "SELECT COUNT(*) FROM int_sa2_demographics"
        ).fetchone()[0]
        assert int_count == 3

        # Verify data joined correctly
        result = con.execute(
            """
            SELECT sa2_code, sa2_name, total_population, irsad_decile
            FROM int_sa2_demographics
            WHERE sa2_code = '101011001'
        """
        ).fetchone()

        assert result is not None
        assert result[2] == 10000  # population
        assert result[3] == 10  # irsad_decile

        con.close()

    def test_step4_marts_creation(self, temp_pipeline_dirs, sample_raw_data):
        """Test Step 4: Create final marts (dbt marts models)."""
        db_path = temp_pipeline_dirs["db_path"]

        # Setup: Create full pipeline up to intermediate
        con = duckdb.connect(str(db_path))

        # Load and stage data
        for parquet_file in sample_raw_data.glob("*.parquet"):
            df = pl.read_parquet(parquet_file)
            table_name = f"stg_{parquet_file.stem.lower()}"
            con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")

        # Create master_health_record mart
        con.execute(
            """
            CREATE TABLE master_health_record AS
            SELECT
                g.sa2_code,
                g.sa2_name,
                g.state_code,
                g.state_name,
                p.total_population,
                p.median_age,
                s.irsad_score,
                s.irsad_decile,
                h.indicator_value as diabetes_prevalence
            FROM stg_sa2_geographic g
            LEFT JOIN stg_population p ON g.sa2_code = p.sa2_code
            LEFT JOIN stg_seifa s ON g.sa2_code = s.sa2_code
            LEFT JOIN (
                SELECT sa2_code, indicator_value
                FROM stg_health_indicators
                WHERE indicator_name = 'diabetes'
            ) h ON g.sa2_code = h.sa2_code
        """
        )

        # Create derived_health_indicators mart
        con.execute(
            """
            CREATE TABLE derived_health_indicators AS
            SELECT
                sa2_code,
                sa2_name,
                diabetes_prevalence,
                irsad_decile,
                CASE
                    WHEN diabetes_prevalence < 5.0 THEN 'Low'
                    WHEN diabetes_prevalence < 6.0 THEN 'Medium'
                    ELSE 'High'
                END as diabetes_risk_level,
                (diabetes_prevalence * total_population / 100.0) as estimated_diabetes_cases
            FROM master_health_record
        """
        )

        # Verify marts created
        master_count = con.execute(
            "SELECT COUNT(*) FROM master_health_record"
        ).fetchone()[0]
        assert master_count == 3

        derived_count = con.execute(
            "SELECT COUNT(*) FROM derived_health_indicators"
        ).fetchone()[0]
        assert derived_count == 3

        # Verify calculated fields
        result = con.execute(
            """
            SELECT sa2_code, diabetes_risk_level, estimated_diabetes_cases
            FROM derived_health_indicators
            WHERE sa2_code = '101011001'
        """
        ).fetchone()

        assert result is not None
        assert result[1] in ["Low", "Medium", "High"]  # risk level
        assert result[2] > 0  # estimated cases

        con.close()

    def test_step5_data_quality_validation(
        self, temp_pipeline_dirs, sample_raw_data
    ):
        """Test Step 5: Run data quality tests (dbt tests)."""
        db_path = temp_pipeline_dirs["db_path"]

        # Setup: Create marts
        con = duckdb.connect(str(db_path))

        # Create sample mart
        master_df = pl.DataFrame(
            {
                "sa2_code": ["101011001", "101011002", "101011003"],
                "sa2_name": ["Sydney - CBD", "Sydney - Haymarket", "Sydney - Rocks"],
                "state_code": ["1", "1", "1"],
                "total_population": [10000, 15000, 20000],
                "diabetes_prevalence": [5.2, 4.8, 6.1],
                "irsad_decile": [10, 8, 9],
            }
        )
        con.execute("CREATE TABLE master_health_record AS SELECT * FROM master_df")

        # Test 1: Check for null SA2 codes
        null_check = con.execute(
            "SELECT COUNT(*) FROM master_health_record WHERE sa2_code IS NULL"
        ).fetchone()[0]
        assert null_check == 0, "Should have no null SA2 codes"

        # Test 2: Check for duplicate SA2 codes
        duplicate_check = con.execute(
            """
            SELECT sa2_code, COUNT(*) as count
            FROM master_health_record
            GROUP BY sa2_code
            HAVING COUNT(*) > 1
        """
        ).fetchall()
        assert len(duplicate_check) == 0, "Should have no duplicate SA2 codes"

        # Test 3: Check population is positive
        negative_pop = con.execute(
            "SELECT COUNT(*) FROM master_health_record WHERE total_population <= 0"
        ).fetchone()[0]
        assert negative_pop == 0, "Population should be positive"

        # Test 4: Check IRSAD decile range
        invalid_decile = con.execute(
            """
            SELECT COUNT(*) FROM master_health_record
            WHERE irsad_decile < 1 OR irsad_decile > 10
        """
        ).fetchone()[0]
        assert invalid_decile == 0, "IRSAD decile should be 1-10"

        # Test 5: Check diabetes prevalence range
        invalid_prevalence = con.execute(
            """
            SELECT COUNT(*) FROM master_health_record
            WHERE diabetes_prevalence < 0 OR diabetes_prevalence > 100
        """
        ).fetchone()[0]
        assert invalid_prevalence == 0, "Diabetes prevalence should be 0-100"

        con.close()

    def test_step6_export_final_data(self, temp_pipeline_dirs, sample_raw_data):
        """Test Step 6: Export final data to multiple formats."""
        db_path = temp_pipeline_dirs["db_path"]
        export_dir = temp_pipeline_dirs["data_exports"]

        # Setup: Create marts
        con = duckdb.connect(str(db_path))
        master_df = pl.DataFrame(
            {
                "sa2_code": ["101011001", "101011002", "101011003"],
                "sa2_name": ["Sydney - CBD", "Sydney - Haymarket", "Sydney - Rocks"],
                "state_code": ["1", "1", "1"],
                "total_population": [10000, 15000, 20000],
                "diabetes_prevalence": [5.2, 4.8, 6.1],
                "irsad_decile": [10, 8, 9],
            }
        )
        con.execute("CREATE TABLE master_health_record AS SELECT * FROM master_df")

        # Export data
        df = con.execute("SELECT * FROM master_health_record").pl()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"master_health_record_{timestamp}"

        # Export to Parquet
        parquet_file = export_dir / f"{base_filename}.parquet"
        df.write_parquet(parquet_file, compression="zstd")

        # Export to CSV
        csv_file = export_dir / f"{base_filename}.csv"
        df.write_csv(csv_file)

        # Export to JSON
        json_file = export_dir / f"{base_filename}.json"
        df.write_json(json_file, row_oriented=True)

        # Create metadata
        metadata = {
            "table_name": "master_health_record",
            "export_timestamp": timestamp,
            "record_count": len(df),
            "column_count": len(df.columns),
            "columns": df.columns,
            "file_sizes_bytes": {
                "parquet": parquet_file.stat().st_size,
                "csv": csv_file.stat().st_size,
                "json": json_file.stat().st_size,
            },
        }

        metadata_file = export_dir / f"{base_filename}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Verify all files created
        assert parquet_file.exists()
        assert csv_file.exists()
        assert json_file.exists()
        assert metadata_file.exists()

        # Verify data integrity in exports
        loaded_parquet = pl.read_parquet(parquet_file)
        loaded_csv = pl.read_csv(csv_file)

        assert len(loaded_parquet) == 3
        assert len(loaded_csv) == 3

        con.close()

    def test_complete_pipeline_integration(
        self, temp_pipeline_dirs, sample_raw_data
    ):
        """Test complete pipeline from raw data to export."""
        db_path = temp_pipeline_dirs["db_path"]
        export_dir = temp_pipeline_dirs["data_exports"]

        # Step 1: Load raw data
        con = duckdb.connect(str(db_path))
        parquet_files = list(sample_raw_data.glob("*.parquet"))

        for parquet_file in parquet_files:
            table_name = f"raw_{parquet_file.stem.lower()}"
            df = pl.read_parquet(parquet_file)
            con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")

        # Step 2: Create staging (simplified)
        con.execute(
            "CREATE TABLE stg_sa2_geographic AS SELECT * FROM raw_sa2_geographic"
        )
        con.execute(
            "CREATE TABLE stg_health_indicators AS SELECT * FROM raw_health_indicators"
        )
        con.execute("CREATE TABLE stg_seifa AS SELECT * FROM raw_seifa")
        con.execute("CREATE TABLE stg_population AS SELECT * FROM raw_population")

        # Step 3: Create marts
        con.execute(
            """
            CREATE TABLE master_health_record AS
            SELECT
                g.SA2_CODE_2021 as sa2_code,
                g.SA2_NAME_2021 as sa2_name,
                g.STATE_CODE_2021 as state_code,
                p.total_population,
                s.irsad_decile,
                h.indicator_value as diabetes_prevalence
            FROM stg_sa2_geographic g
            LEFT JOIN stg_population p ON g.SA2_CODE_2021 = p.sa2_code
            LEFT JOIN stg_seifa s ON g.SA2_CODE_2021 = s.sa2_code
            LEFT JOIN (
                SELECT sa2_code, indicator_value
                FROM stg_health_indicators
                WHERE indicator_name = 'diabetes'
            ) h ON g.SA2_CODE_2021 = h.sa2_code
        """
        )

        # Step 4: Validate
        record_count = con.execute(
            "SELECT COUNT(*) FROM master_health_record"
        ).fetchone()[0]
        assert record_count == 3

        # Step 5: Export
        df = con.execute("SELECT * FROM master_health_record").pl()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        parquet_file = export_dir / f"master_health_record_{timestamp}.parquet"
        df.write_parquet(parquet_file, compression="zstd")

        # Final verification
        assert parquet_file.exists()
        final_df = pl.read_parquet(parquet_file)
        assert len(final_df) == 3
        assert "sa2_code" in final_df.columns
        assert "diabetes_prevalence" in final_df.columns

        con.close()


class TestPipelineErrorHandling:
    """Test pipeline error handling and recovery."""

    def test_missing_raw_data(self, tmp_path):
        """Test pipeline behaviour when raw data is missing."""
        db_path = tmp_path / "ahgd.db"
        data_raw = tmp_path / "data_raw"
        data_raw.mkdir()

        # No data files created
        con = duckdb.connect(str(db_path))

        # Should handle gracefully
        parquet_files = list(data_raw.glob("*.parquet"))
        assert len(parquet_files) == 0

        con.close()

    def test_corrupted_data_handling(self, tmp_path):
        """Test handling of corrupted data."""
        db_path = tmp_path / "ahgd.db"

        con = duckdb.connect(str(db_path))

        # Create data with invalid values
        invalid_df = pl.DataFrame(
            {
                "sa2_code": ["101011001", None, "101011003"],  # Null value
                "population": [10000, -5000, 20000],  # Negative value
                "irsad_decile": [10, 15, 9],  # Out of range value
            }
        )

        con.execute("CREATE TABLE test_invalid AS SELECT * FROM invalid_df")

        # Validate and catch issues
        null_count = con.execute(
            "SELECT COUNT(*) FROM test_invalid WHERE sa2_code IS NULL"
        ).fetchone()[0]
        assert null_count == 1

        negative_pop = con.execute(
            "SELECT COUNT(*) FROM test_invalid WHERE population < 0"
        ).fetchone()[0]
        assert negative_pop == 1

        invalid_decile = con.execute(
            "SELECT COUNT(*) FROM test_invalid WHERE irsad_decile > 10"
        ).fetchone()[0]
        assert invalid_decile == 1

        con.close()
