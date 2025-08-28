from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime

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
    # This task will need a Python script to read the raw data and load it into DuckDB
    load_raw_to_duckdb = PythonOperator(
        task_id='load_raw_to_duckdb',
        python_callable=lambda: print("Loading raw data to DuckDB - TO BE IMPLEMENTED"),
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
    # This task will need a Python script to read from DuckDB and export to desired format
    export_final_data = PythonOperator(
        task_id='export_final_data',
        python_callable=lambda: print("Exporting final data - TO BE IMPLEMENTED"),
    )

    # Define task dependencies
    extract_data >> load_raw_to_duckdb >> dbt_build >> dbt_test >> export_final_data
