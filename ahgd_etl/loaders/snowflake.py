"""
Snowflake Loader Module

Handles loading data from Parquet files to Snowflake data warehouse.
Supports both dimension and fact table loading with proper type mapping.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import polars as pl
try:
    import snowflake.connector
    from snowflake.connector.pandas_tools import write_pandas
    import pandas as pd
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False

from ahgd_etl.config import settings


class SnowflakeLoader:
    """Loads AHGD data into Snowflake data warehouse."""
    
    # Polars to Snowflake type mapping
    TYPE_MAPPING = {
        "Int64": "NUMBER(19,0)",
        "Int32": "NUMBER(10,0)",
        "Int16": "NUMBER(5,0)",
        "Int8": "NUMBER(3,0)",
        "UInt64": "NUMBER(20,0)",
        "UInt32": "NUMBER(10,0)",
        "Float64": "FLOAT",
        "Float32": "FLOAT",
        "Utf8": "VARCHAR",
        "Boolean": "BOOLEAN",
        "Date": "DATE",
        "Datetime": "TIMESTAMP_NTZ",
        "Categorical": "VARCHAR"
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize Snowflake loader.
        
        Args:
            config_file: Path to Snowflake configuration JSON file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_file)
        self.connection = None
        
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load Snowflake configuration from file or environment."""
        if config_file:
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Load from environment or settings
            return {
                "account": settings.get("snowflake.account"),
                "user": settings.get("snowflake.user"),
                "password": settings.get("snowflake.password"),
                "warehouse": settings.get("snowflake.warehouse", "COMPUTE_WH"),
                "database": settings.get("snowflake.database", "AHGD"),
                "schema": settings.get("snowflake.schema", "PUBLIC"),
                "role": settings.get("snowflake.role")
            }
    
    def connect(self):
        """Establish connection to Snowflake."""
        if not SNOWFLAKE_AVAILABLE:
            raise ImportError("Snowflake connector not available. Install with: pip install snowflake-connector-python")
            
        if not self.connection:
            self.logger.info("Connecting to Snowflake...")
            self.connection = snowflake.connector.connect(
                account=self.config["account"],
                user=self.config["user"],
                password=self.config["password"],
                warehouse=self.config["warehouse"],
                database=self.config["database"],
                schema=self.config["schema"],
                role=self.config.get("role")
            )
            self.logger.info("Successfully connected to Snowflake")
        return self.connection
    
    def close(self):
        """Close Snowflake connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            self.logger.info("Closed Snowflake connection")
    
    def create_database_if_not_exists(self):
        """Create database and schema if they don't exist."""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # Create database
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.config['database']}")
            
            # Use database
            cursor.execute(f"USE DATABASE {self.config['database']}")
            
            # Create schema
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.config['schema']}")
            
            # Use schema
            cursor.execute(f"USE SCHEMA {self.config['schema']}")
            
            self.logger.info(f"Database {self.config['database']}.{self.config['schema']} ready")
            
        finally:
            cursor.close()
    
    def create_table_from_parquet(self, parquet_path: Path, table_name: str, 
                                  is_dimension: bool = False) -> str:
        """
        Generate CREATE TABLE DDL from Parquet schema.
        
        Args:
            parquet_path: Path to Parquet file
            table_name: Target table name
            is_dimension: Whether this is a dimension table
            
        Returns:
            CREATE TABLE DDL statement
        """
        # Read Parquet schema
        df = pl.read_parquet(parquet_path, n_rows=1)
        
        # Build column definitions
        columns = []
        for col_name, col_type in zip(df.columns, df.dtypes):
            sf_type = self.TYPE_MAPPING.get(str(col_type), "VARCHAR")
            
            # Add constraints for specific columns
            constraints = ""
            if col_name.endswith("_sk") and is_dimension:
                constraints = " PRIMARY KEY"
            elif col_name == "etl_processed_at":
                constraints = " DEFAULT CURRENT_TIMESTAMP()"
                
            columns.append(f"    {col_name} {sf_type}{constraints}")
        
        # Add table properties
        table_props = []
        if table_name.startswith("fact_"):
            # Cluster fact tables by common query patterns
            cluster_keys = []
            if "time_sk" in df.columns:
                cluster_keys.append("time_sk")
            if "geo_sk" in df.columns:
                cluster_keys.append("geo_sk")
            if cluster_keys:
                table_props.append(f"CLUSTER BY ({', '.join(cluster_keys)})")
        
        # Build DDL
        ddl = f"CREATE OR REPLACE TABLE {table_name} (\n"
        ddl += ",\n".join(columns)
        ddl += "\n)"
        
        if table_props:
            ddl += "\n" + "\n".join(table_props)
            
        return ddl
    
    def load_parquet_to_table(self, parquet_path: Path, table_name: str,
                              truncate: bool = True) -> bool:
        """
        Load Parquet file to Snowflake table.
        
        Args:
            parquet_path: Path to Parquet file
            table_name: Target table name
            truncate: Whether to truncate table before loading
            
        Returns:
            bool: True if successful
        """
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            # Read Parquet file
            self.logger.info(f"Reading {parquet_path}...")
            df_polars = pl.read_parquet(parquet_path)
            
            # Convert to pandas for Snowflake loading
            if not SNOWFLAKE_AVAILABLE:
                raise ImportError("Snowflake connector not available")
                
            df_pandas = df_polars.to_pandas()
            
            # Truncate if requested
            if truncate:
                cursor.execute(f"TRUNCATE TABLE IF EXISTS {table_name}")
            
            # Load data
            self.logger.info(f"Loading {len(df_pandas)} rows to {table_name}...")
            success, nchunks, nrows, _ = write_pandas(
                conn,
                df_pandas,
                table_name,
                database=self.config["database"],
                schema=self.config["schema"],
                auto_create_table=True
            )
            
            if success:
                self.logger.info(f"Successfully loaded {nrows} rows to {table_name}")
                return True
            else:
                self.logger.error(f"Failed to load data to {table_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading {parquet_path} to {table_name}: {str(e)}", exc_info=True)
            return False
        finally:
            cursor.close()
    
    def load_all(self, output_dir: Path) -> bool:
        """
        Load all Parquet files from output directory to Snowflake.
        
        Args:
            output_dir: Directory containing Parquet files
            
        Returns:
            bool: True if all loads successful
        """
        self.create_database_if_not_exists()
        
        success = True
        
        # Define load order (dimensions first, then facts)
        load_order = [
            # Dimensions
            ("geo_dimension.parquet", "dim_geography", True),
            ("dim_time.parquet", "dim_time", True),
            ("dim_health_condition.parquet", "dim_health_condition", True),
            ("dim_demographic.parquet", "dim_demographic", True),
            ("dim_person_characteristic.parquet", "dim_person_characteristic", True),
            # Facts
            ("fact_population.parquet", "fact_population", False),
            ("fact_income.parquet", "fact_income", False),
            ("fact_assistance_needed.parquet", "fact_assistance_need", False),
            ("fact_health_conditions_refined.parquet", "fact_health_condition", False),
            ("fact_health_conditions_by_characteristic_refined.parquet", 
             "fact_health_condition_by_characteristic", False),
            ("fact_unpaid_assistance.parquet", "fact_unpaid_assistance", False)
        ]
        
        for file_name, table_name, is_dimension in load_order:
            file_path = output_dir / file_name
            
            if not file_path.exists():
                self.logger.warning(f"File {file_name} not found, skipping")
                continue
                
            self.logger.info(f"Loading {file_name} to {table_name}...")
            
            if not self.load_parquet_to_table(file_path, table_name):
                success = False
                self.logger.error(f"Failed to load {file_name}")
                
        return success
    
    def generate_ddl_scripts(self, output_dir: Path, script_dir: Path) -> bool:
        """
        Generate DDL scripts for all tables.
        
        Args:
            output_dir: Directory containing Parquet files
            script_dir: Directory to save DDL scripts
            
        Returns:
            bool: True if successful
        """
        script_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate master script
        master_script = [
            "-- AHGD Snowflake DDL Scripts",
            "-- Generated from Parquet schemas",
            "",
            f"CREATE DATABASE IF NOT EXISTS {self.config['database']};",
            f"USE DATABASE {self.config['database']};",
            f"CREATE SCHEMA IF NOT EXISTS {self.config['schema']};",
            f"USE SCHEMA {self.config['schema']};",
            "",
            "-- Dimension Tables",
        ]
        
        # Process each table
        tables = [
            ("geo_dimension.parquet", "dim_geography", True),
            ("dim_time.parquet", "dim_time", True),
            ("dim_health_condition.parquet", "dim_health_condition", True),
            ("dim_demographic.parquet", "dim_demographic", True),
            ("dim_person_characteristic.parquet", "dim_person_characteristic", True),
            ("fact_population.parquet", "fact_population", False),
            ("fact_income.parquet", "fact_income", False),
            ("fact_assistance_needed.parquet", "fact_assistance_need", False),
            ("fact_health_conditions_refined.parquet", "fact_health_condition", False),
            ("fact_unpaid_assistance.parquet", "fact_unpaid_assistance", False)
        ]
        
        for file_name, table_name, is_dimension in tables:
            file_path = output_dir / file_name
            
            if not file_path.exists():
                continue
                
            # Generate DDL
            ddl = self.create_table_from_parquet(file_path, table_name, is_dimension)
            
            # Save individual script
            script_path = script_dir / f"{table_name}.sql"
            with open(script_path, 'w') as f:
                f.write(ddl + ";\n")
                
            # Add to master script
            if not is_dimension and "-- Fact Tables" not in master_script:
                master_script.append("")
                master_script.append("-- Fact Tables")
                
            master_script.append(f"")
            master_script.append(f"-- {table_name}")
            master_script.append(ddl + ";")
            
        # Save master script
        master_path = script_dir / "create_all_tables.sql"
        with open(master_path, 'w') as f:
            f.write("\n".join(master_script))
            
        self.logger.info(f"Generated DDL scripts in {script_dir}")
        return True