"""
Tests for DuckDB database operations.

This module tests:
- Database connection and initialisation
- Table creation and data loading
- Query execution and data retrieval
- Schema validation
- Performance and optimisation
"""

import pytest
from pathlib import Path
from datetime import datetime
import polars as pl
import duckdb
from typing import List


class TestDuckDBConnection:
    """Test DuckDB connection and basic operations."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create temporary DuckDB database."""
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        yield con
        con.close()

    def test_create_connection(self, tmp_path):
        """Test creating DuckDB connection."""
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))

        assert con is not None
        assert db_path.exists()

        con.close()

    def test_connection_read_only(self, tmp_path):
        """Test read-only connection mode."""
        db_path = tmp_path / "test.db"

        # Create database first
        con = duckdb.connect(str(db_path))
        con.execute("CREATE TABLE test (id INTEGER)")
        con.close()

        # Open in read-only mode
        con_readonly = duckdb.connect(str(db_path), read_only=True)

        # Should not be able to write
        with pytest.raises(Exception):
            con_readonly.execute("INSERT INTO test VALUES (1)")

        con_readonly.close()

    def test_in_memory_database(self):
        """Test in-memory DuckDB database."""
        con = duckdb.connect(":memory:")

        assert con is not None

        # Should be able to create tables
        con.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
        result = con.execute("SELECT COUNT(*) FROM test").fetchone()
        assert result[0] == 0

        con.close()


class TestDuckDBTableOperations:
    """Test table creation and manipulation."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create temporary DuckDB database."""
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        yield con
        con.close()

    def test_create_table_from_dataframe(self, temp_db):
        """Test creating table from Polars DataFrame."""
        # Create test DataFrame
        df = pl.DataFrame(
            {
                "sa2_code": ["101011001", "101011002", "101011003"],
                "sa2_name": ["Area 1", "Area 2", "Area 3"],
                "value": [100, 200, 300],
            }
        )

        # Create table
        temp_db.execute("CREATE TABLE test_data AS SELECT * FROM df")

        # Verify table exists
        result = temp_db.execute("SELECT COUNT(*) FROM test_data").fetchone()
        assert result[0] == 3

    def test_create_table_from_parquet(self, temp_db, tmp_path):
        """Test creating table from Parquet file."""
        # Create test Parquet file
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
            }
        )

        parquet_file = tmp_path / "test.parquet"
        df.write_parquet(parquet_file)

        # Load into DuckDB
        temp_db.execute(
            f"CREATE TABLE users AS SELECT * FROM read_parquet('{parquet_file}')"
        )

        # Verify data
        result = temp_db.execute("SELECT COUNT(*) FROM users").fetchone()
        assert result[0] == 3

        # Verify schema
        schema = temp_db.execute("DESCRIBE users").fetchall()
        column_names = [row[0] for row in schema]
        assert "id" in column_names
        assert "name" in column_names
        assert "age" in column_names

    def test_replace_table(self, temp_db):
        """Test replacing existing table."""
        # Create initial table
        df1 = pl.DataFrame({"id": [1, 2], "value": [10, 20]})
        temp_db.execute("CREATE TABLE test AS SELECT * FROM df1")

        # Verify initial data
        result = temp_db.execute("SELECT COUNT(*) FROM test").fetchone()
        assert result[0] == 2

        # Replace table
        df2 = pl.DataFrame({"id": [3, 4, 5], "value": [30, 40, 50]})
        temp_db.execute("CREATE OR REPLACE TABLE test AS SELECT * FROM df2")

        # Verify replaced data
        result = temp_db.execute("SELECT COUNT(*) FROM test").fetchone()
        assert result[0] == 3

    def test_list_tables(self, temp_db):
        """Test listing tables in database."""
        # Create multiple tables
        temp_db.execute("CREATE TABLE table1 (id INTEGER)")
        temp_db.execute("CREATE TABLE table2 (id INTEGER)")
        temp_db.execute("CREATE TABLE table3 (id INTEGER)")

        # List tables
        result = temp_db.execute("SHOW TABLES").fetchall()
        table_names = [row[0] for row in result]

        assert len(table_names) == 3
        assert "table1" in table_names
        assert "table2" in table_names
        assert "table3" in table_names

    def test_drop_table(self, temp_db):
        """Test dropping table."""
        # Create table
        temp_db.execute("CREATE TABLE test_drop (id INTEGER)")

        # Verify exists
        result = temp_db.execute("SHOW TABLES").fetchall()
        assert any(row[0] == "test_drop" for row in result)

        # Drop table
        temp_db.execute("DROP TABLE test_drop")

        # Verify dropped
        result = temp_db.execute("SHOW TABLES").fetchall()
        assert not any(row[0] == "test_drop" for row in result)


class TestDuckDBQueryOperations:
    """Test query execution and data retrieval."""

    @pytest.fixture
    def temp_db_with_data(self, tmp_path):
        """Create database with test data."""
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))

        # Create test data
        df = pl.DataFrame(
            {
                "sa2_code": [
                    "101011001",
                    "101011002",
                    "101011003",
                    "102011001",
                    "102011002",
                ],
                "sa2_name": ["Area 1", "Area 2", "Area 3", "Area 4", "Area 5"],
                "state": ["NSW", "NSW", "NSW", "VIC", "VIC"],
                "population": [10000, 15000, 20000, 12000, 18000],
                "health_score": [85.5, 90.2, 88.7, 92.1, 86.3],
            }
        )

        con.execute("CREATE TABLE health_data AS SELECT * FROM df")

        yield con
        con.close()

    def test_select_all(self, temp_db_with_data):
        """Test SELECT * query."""
        result = temp_db_with_data.execute("SELECT * FROM health_data").fetchall()
        assert len(result) == 5

    def test_select_with_filter(self, temp_db_with_data):
        """Test SELECT with WHERE clause."""
        result = temp_db_with_data.execute(
            "SELECT * FROM health_data WHERE state = 'NSW'"
        ).fetchall()
        assert len(result) == 3

    def test_aggregate_query(self, temp_db_with_data):
        """Test aggregate queries."""
        # COUNT
        result = temp_db_with_data.execute(
            "SELECT COUNT(*) FROM health_data"
        ).fetchone()
        assert result[0] == 5

        # SUM
        result = temp_db_with_data.execute(
            "SELECT SUM(population) FROM health_data"
        ).fetchone()
        assert result[0] == 75000

        # AVG
        result = temp_db_with_data.execute(
            "SELECT AVG(health_score) FROM health_data"
        ).fetchone()
        assert abs(result[0] - 88.56) < 0.01

    def test_group_by_query(self, temp_db_with_data):
        """Test GROUP BY queries."""
        result = temp_db_with_data.execute(
            """
            SELECT state, COUNT(*) as count, AVG(population) as avg_pop
            FROM health_data
            GROUP BY state
            ORDER BY state
        """
        ).fetchall()

        assert len(result) == 2
        assert result[0][0] == "NSW"  # state
        assert result[0][1] == 3  # count
        assert result[1][0] == "VIC"
        assert result[1][1] == 2

    def test_join_query(self, temp_db_with_data):
        """Test JOIN operations."""
        # Create second table
        df2 = pl.DataFrame(
            {
                "sa2_code": ["101011001", "101011002", "102011001"],
                "indicator": ["diabetes", "diabetes", "diabetes"],
                "value": [5.2, 4.8, 6.1],
            }
        )

        temp_db_with_data.execute("CREATE TABLE indicators AS SELECT * FROM df2")

        # Test JOIN
        result = temp_db_with_data.execute(
            """
            SELECT h.sa2_code, h.sa2_name, i.indicator, i.value
            FROM health_data h
            JOIN indicators i ON h.sa2_code = i.sa2_code
            ORDER BY h.sa2_code
        """
        ).fetchall()

        assert len(result) == 3
        assert result[0][1] == "Area 1"
        assert result[0][2] == "diabetes"

    def test_return_polars_dataframe(self, temp_db_with_data):
        """Test returning results as Polars DataFrame."""
        df = temp_db_with_data.execute("SELECT * FROM health_data").pl()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 5
        assert "sa2_code" in df.columns
        assert "population" in df.columns


class TestDuckDBPerformance:
    """Test DuckDB performance and optimisation."""

    @pytest.fixture
    def temp_db_large(self, tmp_path):
        """Create database with larger dataset."""
        db_path = tmp_path / "test_large.db"
        con = duckdb.connect(str(db_path))

        # Create larger test dataset
        import random

        sa2_codes = [f"10{i:07d}" for i in range(10000)]
        df = pl.DataFrame(
            {
                "sa2_code": sa2_codes,
                "sa2_name": [f"Area {i}" for i in range(10000)],
                "state": random.choices(["NSW", "VIC", "QLD", "SA", "WA"], k=10000),
                "population": [random.randint(1000, 50000) for _ in range(10000)],
                "health_score": [random.uniform(70.0, 95.0) for _ in range(10000)],
            }
        )

        con.execute("CREATE TABLE health_data AS SELECT * FROM df")

        yield con
        con.close()

    def test_query_performance(self, temp_db_large):
        """Test query performance on larger dataset."""
        import time

        start = time.time()
        result = temp_db_large.execute(
            """
            SELECT state, COUNT(*) as count, AVG(health_score) as avg_score
            FROM health_data
            GROUP BY state
        """
        ).fetchall()
        end = time.time()

        # Query should complete in reasonable time (< 1 second)
        assert (end - start) < 1.0
        assert len(result) == 5

    def test_index_performance(self, temp_db_large):
        """Test performance with and without indexes."""
        import time

        # Query without index
        start = time.time()
        temp_db_large.execute(
            "SELECT * FROM health_data WHERE sa2_code = '1000005000'"
        ).fetchall()
        time_without_index = time.time() - start

        # Create index
        temp_db_large.execute("CREATE INDEX idx_sa2 ON health_data(sa2_code)")

        # Query with index
        start = time.time()
        temp_db_large.execute(
            "SELECT * FROM health_data WHERE sa2_code = '1000005000'"
        ).fetchall()
        time_with_index = time.time() - start

        # Both should be fast on this dataset size
        assert time_without_index < 0.5
        assert time_with_index < 0.5

    def test_parquet_export_performance(self, temp_db_large, tmp_path):
        """Test Parquet export performance."""
        import time

        export_file = tmp_path / "export.parquet"

        start = time.time()
        temp_db_large.execute(
            f"COPY health_data TO '{export_file}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        end = time.time()

        # Export should be fast (< 2 seconds)
        assert (end - start) < 2.0
        assert export_file.exists()


class TestDuckDBErrorHandling:
    """Test error handling in DuckDB operations."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create temporary DuckDB database."""
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        yield con
        con.close()

    def test_query_nonexistent_table(self, temp_db):
        """Test querying non-existent table."""
        with pytest.raises(Exception):
            temp_db.execute("SELECT * FROM nonexistent_table").fetchall()

    def test_invalid_sql_syntax(self, temp_db):
        """Test invalid SQL syntax."""
        with pytest.raises(Exception):
            temp_db.execute("INVALID SQL SYNTAX").fetchall()

    def test_type_mismatch(self, temp_db):
        """Test type mismatch in operations."""
        temp_db.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")

        # Try to insert wrong type - DuckDB handles this gracefully
        # by attempting type conversion
        temp_db.execute("INSERT INTO test VALUES ('not_a_number', 'Name')")

    def test_constraint_violation(self, temp_db):
        """Test constraint violations."""
        # Create table with unique constraint
        temp_db.execute("CREATE TABLE test (id INTEGER UNIQUE, name VARCHAR)")
        temp_db.execute("INSERT INTO test VALUES (1, 'Alice')")

        # Try to insert duplicate
        with pytest.raises(Exception):
            temp_db.execute("INSERT INTO test VALUES (1, 'Bob')")
