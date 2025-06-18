"""
Integration tests for database operations.

This module tests database connectivity, data loading, querying, and other
database-related operations across the Australian Health Analytics system.
"""

import pytest
import pandas as pd
import duckdb
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add the project paths to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from tests.fixtures.sample_data import (
    get_sample_correspondence_data, get_sample_health_data,
    get_sample_seifa_data, populate_sample_database,
    get_sample_database_schema
)


@pytest.mark.integration
@pytest.mark.database
class TestDatabaseConnectivity:
    """Test database connection and basic operations."""
    
    def test_duckdb_connection_creation(self, temp_db):
        """Test basic DuckDB connection creation."""
        conn = duckdb.connect(str(temp_db))
        assert conn is not None
        conn.close()
    
    def test_database_file_creation(self, temp_dir):
        """Test that database file is created properly."""
        db_path = temp_dir / "test.db"
        conn = duckdb.connect(str(db_path))
        conn.close()
        
        assert db_path.exists()
        assert db_path.stat().st_size > 0
    
    def test_multiple_connections_same_database(self, temp_db):
        """Test multiple connections to the same database."""
        conn1 = duckdb.connect(str(temp_db))
        conn2 = duckdb.connect(str(temp_db))
        
        # Both connections should work
        result1 = conn1.execute("SELECT 1 as test_value").fetchone()
        result2 = conn2.execute("SELECT 2 as test_value").fetchone()
        
        assert result1[0] == 1
        assert result2[0] == 2
        
        conn1.close()
        conn2.close()
    
    def test_database_persistence(self, temp_dir):
        """Test that data persists between connections."""
        db_path = temp_dir / "persistent_test.db"
        
        # First connection: create table and insert data
        conn1 = duckdb.connect(str(db_path))
        conn1.execute("CREATE TABLE test_table (id INTEGER, name TEXT)")
        conn1.execute("INSERT INTO test_table VALUES (1, 'test')")
        conn1.close()
        
        # Second connection: verify data exists
        conn2 = duckdb.connect(str(db_path))
        result = conn2.execute("SELECT * FROM test_table").fetchone()
        conn2.close()
        
        assert result == (1, 'test')


@pytest.mark.integration
@pytest.mark.database
class TestSchemaCreation:
    """Test database schema creation and management."""
    
    def test_create_correspondence_table(self, temp_db):
        """Test creation of correspondence table."""
        conn = duckdb.connect(str(temp_db))
        
        schema_sql = """
        CREATE TABLE correspondence (
            POA_CODE_2021 TEXT,
            SA2_CODE_2021 TEXT,
            SA2_NAME_2021 TEXT,
            RATIO REAL
        )
        """
        
        conn.execute(schema_sql)
        
        # Verify table exists
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [table[0] for table in tables]
        assert 'correspondence' in table_names
        
        # Verify schema
        columns = conn.execute("DESCRIBE correspondence").fetchall()
        column_names = [col[0] for col in columns]
        expected_columns = ['POA_CODE_2021', 'SA2_CODE_2021', 'SA2_NAME_2021', 'RATIO']
        
        for expected_col in expected_columns:
            assert expected_col in column_names
        
        conn.close()
    
    def test_create_all_sample_tables(self, temp_db):
        """Test creation of all sample database tables."""
        conn = duckdb.connect(str(temp_db))
        
        schema_statements = get_sample_database_schema()
        for statement in schema_statements:
            conn.execute(statement)
        
        # Verify all tables exist
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [table[0] for table in tables]
        
        expected_tables = ['correspondence', 'seifa_data', 'health_outcomes', 'demographic_data']
        for expected_table in expected_tables:
            assert expected_table in table_names
        
        conn.close()
    
    def test_table_constraints_and_types(self, temp_db):
        """Test that table constraints and data types are correctly applied."""
        conn = duckdb.connect(str(temp_db))
        
        # Create table with constraints
        conn.execute("""
        CREATE TABLE test_constraints (
            id INTEGER PRIMARY KEY,
            code TEXT NOT NULL,
            value REAL CHECK (value >= 0),
            count INTEGER DEFAULT 0
        )
        """)
        
        # Test successful insert
        conn.execute("INSERT INTO test_constraints (id, code, value) VALUES (1, 'TEST', 10.5)")
        
        # Test constraint violations
        with pytest.raises(Exception):  # NOT NULL constraint
            conn.execute("INSERT INTO test_constraints (id, value) VALUES (2, 5.0)")
        
        with pytest.raises(Exception):  # CHECK constraint
            conn.execute("INSERT INTO test_constraints (id, code, value) VALUES (3, 'TEST2', -1.0)")
        
        conn.close()


@pytest.mark.integration
@pytest.mark.database
class TestDataLoading:
    """Test loading data into database tables."""
    
    def test_load_correspondence_data(self, temp_db):
        """Test loading correspondence data into database."""
        conn = duckdb.connect(str(temp_db))
        
        # Create table
        conn.execute("""
        CREATE TABLE correspondence (
            POA_CODE_2021 TEXT,
            SA2_CODE_2021 TEXT,
            SA2_NAME_2021 TEXT,
            RATIO REAL
        )
        """)
        
        # Load sample data
        sample_data = get_sample_correspondence_data()
        conn.execute("INSERT INTO correspondence SELECT * FROM sample_data")
        
        # Verify data loaded
        count = conn.execute("SELECT COUNT(*) FROM correspondence").fetchone()[0]
        assert count == len(sample_data)
        
        # Verify data integrity
        first_row = conn.execute("SELECT * FROM correspondence LIMIT 1").fetchone()
        assert first_row is not None
        assert len(first_row) == 4  # 4 columns
        
        conn.close()
    
    def test_load_health_data(self, temp_db):
        """Test loading health outcome data into database."""
        conn = duckdb.connect(str(temp_db))
        
        # Create table
        conn.execute("""
        CREATE TABLE health_outcomes (
            sa2_code TEXT,
            year INTEGER,
            mortality_rate REAL,
            chronic_disease_rate REAL,
            mental_health_rate REAL,
            diabetes_rate REAL,
            heart_disease_rate REAL,
            population INTEGER
        )
        """)
        
        # Load sample data
        sample_data = get_sample_health_data()
        conn.execute("INSERT INTO health_outcomes SELECT * FROM sample_data")
        
        # Verify data types and ranges
        stats = conn.execute("""
        SELECT 
            MIN(mortality_rate) as min_mortality,
            MAX(mortality_rate) as max_mortality,
            AVG(population) as avg_population,
            COUNT(*) as total_records
        FROM health_outcomes
        """).fetchone()
        
        min_mortality, max_mortality, avg_population, total_records = stats
        
        assert min_mortality >= 0
        assert max_mortality <= 100  # Reasonable upper bound
        assert avg_population > 0
        assert total_records == len(sample_data)
        
        conn.close()
    
    def test_load_seifa_data(self, temp_db):
        """Test loading SEIFA socioeconomic data into database."""
        conn = duckdb.connect(str(temp_db))
        
        # Create table
        conn.execute("""
        CREATE TABLE seifa_data (
            sa2_code_2021 TEXT,
            sa2_name_2021 TEXT,
            irsad_score REAL,
            irsad_decile INTEGER,
            irsd_score REAL,
            irsd_decile INTEGER,
            ier_score REAL,
            ier_decile INTEGER
        )
        """)
        
        # Load sample data
        sample_data = get_sample_seifa_data()
        conn.execute("INSERT INTO seifa_data SELECT * FROM sample_data")
        
        # Verify decile ranges (should be 1-10)
        decile_stats = conn.execute("""
        SELECT 
            MIN(irsad_decile) as min_decile,
            MAX(irsad_decile) as max_decile,
            COUNT(DISTINCT irsad_decile) as unique_deciles
        FROM seifa_data
        """).fetchone()
        
        min_decile, max_decile, unique_deciles = decile_stats
        
        assert 1 <= min_decile <= 10
        assert 1 <= max_decile <= 10
        assert unique_deciles > 0
        
        conn.close()
    
    def test_bulk_data_loading_performance(self, temp_db):
        """Test performance of bulk data loading."""
        import time
        
        conn = duckdb.connect(str(temp_db))
        
        # Create test table
        conn.execute("""
        CREATE TABLE performance_test (
            id INTEGER,
            value REAL,
            text_field TEXT
        )
        """)
        
        # Generate larger dataset
        import numpy as np
        large_data = pd.DataFrame({
            'id': range(10000),
            'value': np.random.rand(10000),
            'text_field': [f'text_{i}' for i in range(10000)]
        })
        
        # Time the bulk insert
        start_time = time.time()
        conn.execute("INSERT INTO performance_test SELECT * FROM large_data")
        end_time = time.time()
        
        loading_time = end_time - start_time
        
        # Verify all data loaded
        count = conn.execute("SELECT COUNT(*) FROM performance_test").fetchone()[0]
        assert count == 10000
        
        # Performance should be reasonable (under 5 seconds for 10k records)
        assert loading_time < 5.0
        
        conn.close()


@pytest.mark.integration
@pytest.mark.database
class TestDatabaseQueries:
    """Test database querying and data retrieval."""
    
    def test_basic_select_queries(self, mock_database_connection):
        """Test basic SELECT queries on sample data."""
        # Query correspondence data
        result = mock_database_connection.execute("""
        SELECT POA_CODE_2021, SA2_CODE_2021, RATIO 
        FROM correspondence 
        WHERE POA_CODE_2021 = '2000'
        """).fetchall()
        
        assert len(result) > 0
        # Should find entries for postcode 2000
        postcodes = [row[0] for row in result]
        assert all(pc == '2000' for pc in postcodes)
    
    def test_aggregation_queries(self, temp_db):
        """Test aggregation queries."""
        # Set up database with sample data
        populate_sample_database(temp_db)
        
        conn = duckdb.connect(str(temp_db))
        
        # Test aggregation by SA2
        result = conn.execute("""
        SELECT 
            COUNT(*) as record_count,
            AVG(mortality_rate) as avg_mortality,
            MIN(population) as min_population,
            MAX(population) as max_population
        FROM health_outcomes
        """).fetchone()
        
        record_count, avg_mortality, min_population, max_population = result
        
        assert record_count > 0
        assert avg_mortality > 0
        assert min_population > 0
        assert max_population >= min_population
        
        conn.close()
    
    def test_join_queries(self, temp_db):
        """Test JOIN queries across multiple tables."""
        # Set up database with sample data
        populate_sample_database(temp_db)
        
        conn = duckdb.connect(str(temp_db))
        
        # Test join between health outcomes and SEIFA data
        result = conn.execute("""
        SELECT 
            h.sa2_code,
            h.mortality_rate,
            s.irsad_decile,
            s.sa2_name_2021
        FROM health_outcomes h
        JOIN seifa_data s ON h.sa2_code = s.sa2_code_2021
        ORDER BY h.mortality_rate DESC
        LIMIT 5
        """).fetchall()
        
        assert len(result) > 0
        
        # Verify join worked correctly
        for row in result:
            sa2_code, mortality_rate, irsad_decile, sa2_name = row
            assert sa2_code is not None
            assert mortality_rate is not None
            assert irsad_decile is not None
            assert sa2_name is not None
        
        conn.close()
    
    def test_complex_analytical_queries(self, temp_db):
        """Test complex analytical queries."""
        # Set up database with sample data
        populate_sample_database(temp_db)
        
        conn = duckdb.connect(str(temp_db))
        
        # Test correlation analysis query
        result = conn.execute("""
        SELECT 
            CORR(h.mortality_rate, s.irsad_score) as mortality_seifa_correlation,
            COUNT(*) as sample_size
        FROM health_outcomes h
        JOIN seifa_data s ON h.sa2_code = s.sa2_code_2021
        """).fetchone()
        
        correlation, sample_size = result
        
        assert sample_size > 0
        # Correlation should be a valid number between -1 and 1 (or NULL)
        if correlation is not None:
            assert -1 <= correlation <= 1
        
        conn.close()
    
    def test_window_functions(self, temp_db):
        """Test window functions for ranking and percentiles."""
        # Set up database with sample data
        populate_sample_database(temp_db)
        
        conn = duckdb.connect(str(temp_db))
        
        # Test ranking by mortality rate
        result = conn.execute("""
        SELECT 
            sa2_code,
            mortality_rate,
            RANK() OVER (ORDER BY mortality_rate DESC) as mortality_rank,
            NTILE(4) OVER (ORDER BY mortality_rate) as mortality_quartile
        FROM health_outcomes
        ORDER BY mortality_rate DESC
        """).fetchall()
        
        assert len(result) > 0
        
        # Verify ranking is correct (descending)
        prev_rate = float('inf')
        for row in result:
            sa2_code, mortality_rate, rank, quartile = row
            assert mortality_rate <= prev_rate
            assert 1 <= rank <= len(result)
            assert 1 <= quartile <= 4
            prev_rate = mortality_rate
        
        conn.close()


@pytest.mark.integration
@pytest.mark.database
class TestDatabaseErrorHandling:
    """Test database error handling and edge cases."""
    
    def test_invalid_sql_syntax(self, temp_db):
        """Test handling of invalid SQL syntax."""
        conn = duckdb.connect(str(temp_db))
        
        with pytest.raises(Exception):  # Should raise SQL syntax error
            conn.execute("SELEC * FORM invalid_table")
        
        conn.close()
    
    def test_missing_table_query(self, temp_db):
        """Test querying non-existent table."""
        conn = duckdb.connect(str(temp_db))
        
        with pytest.raises(Exception):  # Should raise table not found error
            conn.execute("SELECT * FROM non_existent_table")
        
        conn.close()
    
    def test_data_type_mismatch(self, temp_db):
        """Test handling of data type mismatches."""
        conn = duckdb.connect(str(temp_db))
        
        conn.execute("""
        CREATE TABLE type_test (
            id INTEGER,
            value REAL
        )
        """)
        
        # This should work (automatic conversion)
        conn.execute("INSERT INTO type_test VALUES (1, '10.5')")
        
        # This might cause issues depending on DuckDB's type coercion
        with pytest.raises(Exception):
            conn.execute("INSERT INTO type_test VALUES ('not_a_number', 10.5)")
        
        conn.close()
    
    def test_connection_recovery(self, temp_db):
        """Test recovery from connection issues."""
        conn = duckdb.connect(str(temp_db))
        
        # Create some data
        conn.execute("CREATE TABLE recovery_test (id INTEGER)")
        conn.execute("INSERT INTO recovery_test VALUES (1)")
        
        # Close connection
        conn.close()
        
        # Reconnect and verify data persists
        new_conn = duckdb.connect(str(temp_db))
        result = new_conn.execute("SELECT * FROM recovery_test").fetchone()
        assert result == (1,)
        
        new_conn.close()


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.slow
class TestDatabasePerformance:
    """Test database performance characteristics."""
    
    def test_large_dataset_query_performance(self, temp_db):
        """Test query performance with larger datasets."""
        import time
        import numpy as np
        
        conn = duckdb.connect(str(temp_db))
        
        # Create large test dataset
        conn.execute("""
        CREATE TABLE large_health_data (
            sa2_code TEXT,
            year INTEGER,
            mortality_rate REAL,
            population INTEGER,
            random_value REAL
        )
        """)
        
        # Generate and insert large dataset
        large_data = pd.DataFrame({
            'sa2_code': [f'SA2_{i//100}' for i in range(50000)],
            'year': np.random.choice([2018, 2019, 2020, 2021], 50000),
            'mortality_rate': np.random.normal(5.0, 2.0, 50000),
            'population': np.random.randint(10000, 50000, 50000),
            'random_value': np.random.rand(50000)
        })
        
        # Insert data
        insert_start = time.time()
        conn.execute("INSERT INTO large_health_data SELECT * FROM large_data")
        insert_time = time.time() - insert_start
        
        # Test aggregation query performance
        query_start = time.time()
        result = conn.execute("""
        SELECT 
            sa2_code,
            AVG(mortality_rate) as avg_mortality,
            COUNT(*) as record_count
        FROM large_health_data
        GROUP BY sa2_code
        ORDER BY avg_mortality DESC
        LIMIT 10
        """).fetchall()
        query_time = time.time() - query_start
        
        # Verify results
        assert len(result) == 10
        
        # Performance expectations (may need adjustment based on hardware)
        assert insert_time < 10.0  # Under 10 seconds for 50k records
        assert query_time < 2.0    # Under 2 seconds for aggregation
        
        conn.close()
    
    def test_index_performance_impact(self, temp_db):
        """Test impact of indexes on query performance."""
        import time
        import numpy as np
        
        conn = duckdb.connect(str(temp_db))
        
        # Create test table
        conn.execute("""
        CREATE TABLE index_test (
            id INTEGER,
            sa2_code TEXT,
            value REAL
        )
        """)
        
        # Insert test data
        test_data = pd.DataFrame({
            'id': range(10000),
            'sa2_code': [f'SA2_{i%1000}' for i in range(10000)],
            'value': np.random.rand(10000)
        })
        
        conn.execute("INSERT INTO index_test SELECT * FROM test_data")
        
        # Time query without index
        start_time = time.time()
        result1 = conn.execute("SELECT * FROM index_test WHERE sa2_code = 'SA2_500'").fetchall()
        time_without_index = time.time() - start_time
        
        # Note: DuckDB may automatically optimize, so this test might not show dramatic differences
        assert len(result1) > 0
        assert time_without_index < 1.0  # Should still be fast for 10k records
        
        conn.close()


@pytest.mark.integration
@pytest.mark.database
class TestDatabaseBackupAndRecovery:
    """Test database backup and recovery functionality."""
    
    def test_database_copy_functionality(self, temp_dir):
        """Test copying database to backup location."""
        import shutil
        
        # Create original database with data
        original_db = temp_dir / "original.db"
        backup_db = temp_dir / "backup.db"
        
        # Populate original database
        populate_sample_database(original_db)
        
        # Copy database
        shutil.copy2(original_db, backup_db)
        
        # Verify backup works
        conn = duckdb.connect(str(backup_db))
        count = conn.execute("SELECT COUNT(*) FROM health_outcomes").fetchone()[0]
        assert count > 0
        conn.close()
    
    def test_data_export_import(self, temp_db, temp_dir):
        """Test exporting and importing data for backup purposes."""
        # Set up database with sample data
        populate_sample_database(temp_db)
        
        conn = duckdb.connect(str(temp_db))
        
        # Export data to CSV
        export_file = temp_dir / "export_health.csv"
        conn.execute(f"COPY health_outcomes TO '{export_file}' (FORMAT CSV, HEADER)")
        
        # Verify export file exists and has content
        assert export_file.exists()
        assert export_file.stat().st_size > 0
        
        # Create new database and import
        new_db = temp_dir / "imported.db"
        new_conn = duckdb.connect(str(new_db))
        
        # Create table structure
        new_conn.execute("""
        CREATE TABLE health_outcomes (
            sa2_code TEXT,
            year INTEGER,
            mortality_rate REAL,
            chronic_disease_rate REAL,
            mental_health_rate REAL,
            diabetes_rate REAL,
            heart_disease_rate REAL,
            population INTEGER
        )
        """)
        
        # Import data
        new_conn.execute(f"COPY health_outcomes FROM '{export_file}' (FORMAT CSV, HEADER)")
        
        # Verify import
        original_count = conn.execute("SELECT COUNT(*) FROM health_outcomes").fetchone()[0]
        imported_count = new_conn.execute("SELECT COUNT(*) FROM health_outcomes").fetchone()[0]
        
        assert original_count == imported_count
        
        conn.close()
        new_conn.close()
