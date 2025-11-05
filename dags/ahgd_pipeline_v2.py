from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


def load_raw_data_to_duckdb(**context):
    """
    Load extracted raw data from Parquet files into DuckDB database.

    This function:
    1. Scans the data_raw directory for Parquet files
    2. Loads each dataset into DuckDB as separate tables
    3. Returns metadata about the loading operation
    """
    try:
        import duckdb
        import polars as pl
        from glob import glob

        # Configuration
        duckdb_path = Path("/app/ahgd.db")
        raw_data_dir = Path("/app/data_raw")

        logger.info(f"Starting DuckDB loading from {raw_data_dir}")
        logger.info(f"DuckDB database path: {duckdb_path}")

        # Connect to DuckDB
        con = duckdb.connect(str(duckdb_path))

        # Find all Parquet files in raw data directory
        parquet_files = glob(str(raw_data_dir / "**/*.parquet"), recursive=True)

        if not parquet_files:
            logger.warning("No Parquet files found in data_raw directory")
            return {"status": "warning", "message": "No data files found", "tables_created": 0}

        logger.info(f"Found {len(parquet_files)} Parquet file(s) to load")

        tables_created = 0
        total_records = 0

        # Load each Parquet file into a table
        for parquet_file in parquet_files:
            file_path = Path(parquet_file)
            # Use filename (without extension) as table name, prefixed with 'raw_'
            table_name = f"raw_{file_path.stem.lower()}"

            logger.info(f"Loading {file_path.name} into table '{table_name}'")

            try:
                # Read Parquet file with Polars for fast loading
                df = pl.read_parquet(parquet_file)
                record_count = len(df)

                # Create or replace table in DuckDB
                con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")

                tables_created += 1
                total_records += record_count

                logger.info(f"✓ Loaded {record_count} records into {table_name}")

            except Exception as e:
                logger.error(f"Failed to load {file_path.name}: {str(e)}")
                # Continue with other files even if one fails
                continue

        # Verify tables were created
        tables_result = con.execute("SHOW TABLES").fetchall()
        table_list = [row[0] for row in tables_result]

        logger.info(f"DuckDB tables created: {', '.join(table_list)}")

        # Close connection
        con.close()

        result = {
            "status": "success",
            "tables_created": tables_created,
            "total_records": total_records,
            "table_list": table_list,
            "duckdb_path": str(duckdb_path)
        }

        logger.info(f"DuckDB loading completed: {tables_created} tables, {total_records} total records")

        return result

    except Exception as e:
        logger.error(f"DuckDB loading failed: {str(e)}", exc_info=True)
        raise


def export_data_from_duckdb(**context):
    """
    Export final processed data from DuckDB marts to multiple formats.

    This function:
    1. Connects to DuckDB and reads the master_health_record table
    2. Exports to multiple formats: Parquet, CSV, GeoJSON, JSON
    3. Creates compressed versions and metadata files
    4. Returns export statistics
    """
    try:
        import duckdb
        import polars as pl
        import json
        from pathlib import Path
        from datetime import datetime

        # Configuration
        duckdb_path = Path("/app/ahgd.db")
        export_dir = Path("/app/data_exports")
        export_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting data export from DuckDB: {duckdb_path}")
        logger.info(f"Export directory: {export_dir}")

        # Connect to DuckDB
        con = duckdb.connect(str(duckdb_path), read_only=True)

        # Define tables to export (from dbt marts)
        export_tables = [
            "master_health_record",
            "derived_health_indicators",
        ]

        exported_files = []
        export_stats = {}

        for table_name in export_tables:
            logger.info(f"Exporting table: {table_name}")

            try:
                # Check if table exists
                table_check = con.execute(
                    f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
                ).fetchone()

                if not table_check or table_check[0] == 0:
                    logger.warning(f"Table '{table_name}' not found, skipping")
                    continue

                # Read table into Polars DataFrame for fast processing
                df = con.execute(f"SELECT * FROM {table_name}").pl()
                record_count = len(df)

                logger.info(f"Read {record_count} records from {table_name}")

                # Create timestamped filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = f"{table_name}_{timestamp}"

                # Export to Parquet (optimized for analytics)
                parquet_file = export_dir / f"{base_filename}.parquet"
                df.write_parquet(parquet_file, compression="zstd")
                logger.info(f"✓ Exported to Parquet: {parquet_file.name}")
                exported_files.append(str(parquet_file))

                # Export to CSV (human-readable)
                csv_file = export_dir / f"{base_filename}.csv"
                df.write_csv(csv_file)
                logger.info(f"✓ Exported to CSV: {csv_file.name}")
                exported_files.append(str(csv_file))

                # Export to JSON (for APIs)
                json_file = export_dir / f"{base_filename}.json"
                df.write_json(json_file, row_oriented=True)
                logger.info(f"✓ Exported to JSON: {json_file.name}")
                exported_files.append(str(json_file))

                # Create metadata file
                metadata = {
                    "table_name": table_name,
                    "export_timestamp": timestamp,
                    "record_count": record_count,
                    "column_count": len(df.columns),
                    "columns": df.columns,
                    "schema": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
                    "file_sizes_bytes": {
                        "parquet": parquet_file.stat().st_size,
                        "csv": csv_file.stat().st_size,
                        "json": json_file.stat().st_size,
                    }
                }

                metadata_file = export_dir / f"{base_filename}_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                logger.info(f"✓ Created metadata file: {metadata_file.name}")
                exported_files.append(str(metadata_file))

                export_stats[table_name] = {
                    "records": record_count,
                    "columns": len(df.columns),
                    "formats": ["parquet", "csv", "json"],
                    "total_size_mb": sum(metadata["file_sizes_bytes"].values()) / (1024 * 1024)
                }

            except Exception as e:
                logger.error(f"Failed to export table {table_name}: {str(e)}")
                continue

        # Close DuckDB connection
        con.close()

        # Create overall export summary
        summary = {
            "export_timestamp": datetime.now().isoformat(),
            "tables_exported": len(export_stats),
            "total_files_created": len(exported_files),
            "export_directory": str(export_dir),
            "table_statistics": export_stats,
            "exported_files": exported_files
        }

        summary_file = export_dir / f"export_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Export completed successfully!")
        logger.info(f"Tables exported: {len(export_stats)}")
        logger.info(f"Total files created: {len(exported_files)}")
        logger.info(f"Summary saved to: {summary_file}")

        return summary

    except Exception as e:
        logger.error(f"Data export failed: {str(e)}", exc_info=True)
        raise


# Define the DAG
with DAG(
    dag_id='ahgd_etl_v2',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=['ahgd', 'etl', 'v2'],
) as dag:
    # Task 1: Extract raw data using Python extractors
    extract_data = BashOperator(
        task_id='extract_data',
        bash_command='python /app/src/cli/main.py extract --all --output /app/data_raw --format parquet',
    )

    # Task 2: Load raw data into DuckDB
    load_raw_to_duckdb = PythonOperator(
        task_id='load_raw_to_duckdb',
        python_callable=load_raw_data_to_duckdb,
        provide_context=True,
    )

    # Task 3: Run dbt build to execute staging, intermediate, and marts models
    dbt_build = BashOperator(
        task_id='dbt_build',
        bash_command='cd /app/ahgd_dbt && dbt build --profiles-dir . --project-dir .',
    )

    # Task 4: Run dbt test to execute data quality tests
    dbt_test = BashOperator(
        task_id='dbt_test',
        bash_command='cd /app/ahgd_dbt && dbt test --profiles-dir . --project-dir .',
    )

    # Task 5: Export final data from DuckDB
    export_final_data = PythonOperator(
        task_id='export_final_data',
        python_callable=export_data_from_duckdb,
        provide_context=True,
    )

    # Define task dependencies
    extract_data >> load_raw_to_duckdb >> dbt_build >> dbt_test >> export_final_data
