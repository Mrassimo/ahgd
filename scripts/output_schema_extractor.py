"""Script to extract schemas from Parquet files in the output directory.

This script reads all Parquet files in the configured output directory,
extracts their schemas using pyarrow, and generates:
1. A Python dictionary representation suitable for etl_logic.config.SCHEMAS.
2. A Markdown file detailing the schemas.
3. A Mermaid ERD diagram file (.mmd).
"""

import logging
import sys
import json
from pathlib import Path
from collections import defaultdict
import pyarrow.parquet as pq

# Add project root to sys.path to allow importing etl_logic
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Import config and utils after setting path
from etl_logic import config
from etl_logic import utils # Assuming setup_logging is in utils

# Setup logging
logger = utils.setup_logging(config.PATHS.get('LOG_DIR', project_root / 'logs'))


def map_pyarrow_to_polars(pa_type):
    """Maps PyArrow data types to Polars data type strings."""
    type_str = str(pa_type).lower()
    if type_str.startswith('int'):
        bits = type_str.replace('int', '')
        # Choose appropriate Polars int type
        if bits == '8': return "pl.Int8"
        if bits == '16': return "pl.Int16"
        if bits == '32': return "pl.Int32"
        return "pl.Int64" # Default to Int64 for safety
    if type_str.startswith('uint'):
        bits = type_str.replace('uint', '')
        if bits == '8': return "pl.UInt8"
        if bits == '16': return "pl.UInt16"
        if bits == '32': return "pl.UInt32"
        return "pl.UInt64"
    if type_str.startswith('float') or type_str.startswith('double'):
        return "pl.Float64"
    if type_str == 'bool':
        return "pl.Boolean"
    if type_str == 'string' or type_str == 'large_string':
        return "pl.Utf8"
    if type_str == 'date32[day]' or type_str == 'date64':
        return "pl.Date"
    if type_str.startswith('timestamp'):
        # Extract time unit if present
        unit = 'us' # Default to microseconds
        if '[' in type_str and ']' in type_str:
            unit = type_str.split('[')[1].split(']')[0]
        return f'pl.Datetime(time_unit="{unit}")'
    if type_str == 'binary' or type_str == 'large_binary':
        return "pl.Binary"
    if type_str.startswith('dictionary'): # Handle categorical data
        # Extract value type from dictionary(indices=*, values=*)
        try:
            value_type = pa_type.value_type
            polars_value_type = map_pyarrow_to_polars(value_type)
            # Return as Categorical or the underlying type if preferred
            return "pl.Categorical" # Often desired representation
            # return polars_value_type # Alternatively, use the underlying type
        except Exception:
            return "pl.Categorical" # Fallback

    # Default fallback
    logger.warning(f"Unmapped PyArrow type: {pa_type}. Defaulting to pl.Utf8.")
    return "pl.Utf8"

def extract_schemas(output_dir: Path) -> Dict[str, Dict[str, str]]:
    """Extracts schemas from all Parquet files in the output directory.

    Args:
        output_dir (Path): The directory containing Parquet files.

    Returns:
        Dict[str, Dict[str, str]]: A dictionary where keys are table names
                                   and values are dictionaries mapping column
                                   names to Polars type strings.
    """
    schemas = {}
    logger.info(f"Scanning for Parquet files in: {output_dir}")
    try:
        # Use output_dir directly
        parquet_files = list(output_dir.glob('*.parquet'))

        if not parquet_files:
            logger.warning(f"No Parquet files found in {output_dir}.")
            return schemas

        logger.info(f"Found {len(parquet_files)} Parquet files.")

        for parquet_file in parquet_files:
            try:
                # Read schema using PyArrow
                schema = pq.read_schema(parquet_file)
                table_name = parquet_file.stem # Use stem to remove .parquet
                logger.info(f"Extracting schema for: {table_name}")

                column_schemas = {}
                for field in schema:
                    polars_type = map_pyarrow_to_polars(field.type)
                    column_schemas[field.name] = polars_type

                schemas[table_name] = column_schemas
                logger.debug(f"Schema for {table_name}: {column_schemas}")

            except Exception as e:
                logger.error(f"Error reading schema from {parquet_file.name}: {e}")

    except Exception as e:
        logger.critical(f"Error scanning directory {output_dir}: {e}")

    return schemas

def generate_python_dict_output(schemas: Dict[str, Dict[str, str]], output_file: Path):
    """Generates a Python file containing the schemas as a dictionary.

    Args:
        schemas (Dict): The extracted schemas.
        output_file (Path): Path to save the Python dictionary output.
    """
    logger.info(f"Generating Python dictionary output to: {output_file}")
    try:
        with open(output_file, 'w') as f:
            f.write("import polars as pl\n\n")
            f.write("# Extracted schemas from Parquet files\n")
            f.write("SCHEMAS = {\n")
            for i, (table_name, columns) in enumerate(schemas.items()):
                f.write(f'    "{table_name}": {{\n')
                for col_name, col_type in columns.items():
                    f.write(f'        "{col_name}": {col_type},\n')
                f.write("    }")
                if i < len(schemas) - 1:
                    f.write(",")
                f.write("\n")
            f.write("}\n")
        logger.info(f"Successfully wrote Python dictionary to {output_file}")
    except Exception as e:
        logger.error(f"Error writing Python dictionary file: {e}")

def generate_markdown_output(schemas: Dict[str, Dict[str, str]], output_file: Path):
    """Generates a Markdown file documenting the schemas.

    Args:
        schemas (Dict): The extracted schemas.
        output_file (Path): Path to save the Markdown output.
    """
    logger.info(f"Generating Markdown schema documentation to: {output_file}")
    try:
        with open(output_file, 'w') as f:
            f.write("# Data Warehouse Schema Documentation\n\n")
            f.write("This document details the schemas of the Parquet tables generated by the ETL process.\n\n")
            for table_name, columns in schemas.items():
                f.write(f"## Table: `{table_name}`\n\n")
                f.write("| Column Name | Polars Data Type |\n")
                f.write("|-------------|------------------|\n")
                for col_name, col_type in columns.items():
                    f.write(f"| `{col_name}` | `{col_type}` |\n")
                f.write("\n")
        logger.info(f"Successfully wrote Markdown documentation to {output_file}")
    except Exception as e:
        logger.error(f"Error writing Markdown file: {e}")

def generate_mermaid_erd(schemas: Dict[str, Dict[str, str]], output_file: Path):
    """Generates a Mermaid ERD file based on extracted schemas.

    Args:
        schemas (Dict): The extracted schemas.
        output_file (Path): Path to save the Mermaid ERD (.mmd) file.
    """
    logger.info(f"Generating Mermaid ERD file to: {output_file}")
    # Basic relationship inference (can be enhanced)
    relationships = []
    dim_tables = {name for name in schemas if name.startswith('dim_')}
    fact_tables = {name for name in schemas if name.startswith('fact_')}

    for fact_table, fact_cols in schemas.items():
        if fact_table not in fact_tables:
            continue
        for dim_table, dim_cols in schemas.items():
            if dim_table not in dim_tables:
                continue

            # Infer relationship if fact table has a column named dim_table_sk or similar
            # And the dimension table has that SK as its primary key
            dim_prefix = dim_table.split('dim_')[1]
            possible_fk_names = [f'{dim_prefix}_sk', f'{dim_table}_sk']
            # Usually the first column is the primary key
            dim_pk = next(iter(dim_cols)) if dim_cols else None

            for fk_name in possible_fk_names:
                if fk_name in fact_cols and fk_name == dim_pk:
                    # Mermaid relationship: Fact ||--o{ Dim : has
                    relationships.append(f'    "{fact_table}" ||--o{{ "{dim_table}" : "{fk_name}"')
                    break # Assume one relationship per dim

    try:
        with open(output_file, 'w') as f:
            f.write("erDiagram\n")
            # Define tables and columns
            for table_name, columns in schemas.items():
                f.write(f'    "{table_name}" {{\n')
                for col_name, col_type in columns.items():
                    # Clean type for mermaid (remove pl.)
                    mermaid_type = col_type.replace('pl.', '')
                    # Indicate PK for dimension tables (assuming first col is PK)
                    pk_indicator = " PK" if table_name in dim_tables and col_name == next(iter(columns)) else ""
                    # Indicate FK for fact tables
                    fk_indicator = " FK" if table_name in fact_tables and col_name.endswith('_sk') else ""
                    f.write(f'        {mermaid_type} {col_name}{pk_indicator}{fk_indicator}\n')
                f.write("    }\n")

            # Add relationships
            f.write("\n    %% Relationships\n")
            for rel in relationships:
                f.write(f"{rel}\n")

        logger.info(f"Successfully wrote Mermaid ERD to {output_file}")
    except Exception as e:
        logger.error(f"Error writing Mermaid file: {e}")

def main():
    """Main function to extract schemas and generate outputs."""
    logger.info("=== Starting Schema Extraction and Documentation Generation ===")

    # Use output directory from config
    output_dir = config.PATHS['OUTPUT_DIR']
    documentation_dir = config.PATHS.get('DOCUMENTATION_DIR', project_root / 'documentation')

    # Ensure output and documentation directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    documentation_dir.mkdir(parents=True, exist_ok=True)

    # Define output file paths using config paths
    schema_dict_file = output_dir / "extracted_schemas.py"
    markdown_file = documentation_dir / "data_schema_extracted.md"
    mermaid_file = output_dir / "data_schema_extracted.mmd" # Keep in output or move to docs?

    # Extract schemas
    schemas = extract_schemas(output_dir)

    if not schemas:
        logger.warning("No schemas were extracted. Skipping output generation.")
        return

    # Generate outputs
    generate_python_dict_output(schemas, schema_dict_file)
    generate_markdown_output(schemas, markdown_file)
    generate_mermaid_erd(schemas, mermaid_file)

    logger.info("=== Schema Extraction and Documentation Generation Complete ===")
    logger.info(f"Outputs generated:")
    logger.info(f"  - Python Dictionary: {schema_dict_file}")
    logger.info(f"  - Markdown Doc: {markdown_file}")
    logger.info(f"  - Mermaid ERD: {mermaid_file}")

if __name__ == "__main__":
    # Optional: Ensure directories are initialised if needed
    # config.initialise_directories()
    main() 