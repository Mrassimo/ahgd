import os
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# Define the output directory
OUTPUT_DIR = "/Users/massimoraso/AHGD3/output"
MERMAID_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "data_schema.mmd")

def get_simplified_type(dtype):
    """Convert PyArrow or Polars data types to simplified types for Mermaid diagram."""
    dtype_str = str(dtype).lower()
    
    if any(num_type in dtype_str for num_type in ['int', 'uint']):
        return 'int'
    elif any(float_type in dtype_str for float_type in ['float', 'double']):
        return 'float'
    elif any(date_type in dtype_str for date_type in ['date', 'time', 'timestamp']):
        return 'date'
    elif any(geo_type in dtype_str for geo_type in ['geo', 'geometry', 'wkt', 'point']):
        return 'geometry'
    elif 'bool' in dtype_str:
        return 'bool'
    else:
        return 'string'  # Default for string, binary, etc.

def is_likely_pk(column_name):
    """Determine if a column is likely a primary key based on naming conventions."""
    pk_indicators = ['_id', '_key', '_pk', 'id', 'key', 'pk', '_sk']
    return any(column_name.lower().endswith(indicator) for indicator in pk_indicators)

def is_likely_fk(column_name):
    """Determine if a column is likely a foreign key based on naming conventions."""
    fk_indicators = ['_fk', '_sk', '_code', '_id']
    
    # Check for common foreign key patterns like {table_name}_id or {table_name}_sk
    fk_patterns = [
        'time_', 'geo_', 'location_', 'person_', 'household_', 
        'dim_', 'fact_', 'ref_'
    ]
    
    # Either ends with a FK indicator or starts with a FK pattern
    return (any(column_name.lower().endswith(indicator) for indicator in fk_indicators) or
            any(column_name.lower().startswith(pattern) for pattern in fk_patterns))

def extract_schemas():
    """Extract and print schemas for all Parquet files in the output directory."""
    parquet_files = list(Path(OUTPUT_DIR).glob('*.parquet'))
    
    if not parquet_files:
        print(f"No Parquet files found in {OUTPUT_DIR}")
        return {}
    
    schemas = {}
    
    print("\n===== PARQUET FILE SCHEMAS =====\n")
    
    for parquet_file in parquet_files:
        file_name = parquet_file.name
        file_path = str(parquet_file)
        
        try:
            # Read schema using PyArrow
            schema = pq.read_schema(file_path)
            
            print(f"\nSchema for {file_name}:")
            print("=" * (len(file_name) + 11))
            
            # Print each field with its data type
            field_info = []
            for field in schema:
                field_name = field.name
                field_type = field.type
                print(f"  {field_name}: {field_type}")
                
                # Store for ERD generation
                field_info.append({
                    'name': field_name,
                    'type': field_type,
                    'simplified_type': get_simplified_type(field_type),
                    'is_pk': is_likely_pk(field_name),
                    'is_fk': is_likely_fk(field_name)
                })
            
            # Store schema information 
            table_name = file_name.replace('.parquet', '')
            schemas[table_name] = field_info
            
        except Exception as e:
            print(f"Error reading schema for {file_name}: {e}")
    
    return schemas

def generate_mermaid_erd(schemas):
    """Generate Mermaid ERD syntax based on the extracted schemas."""
    if not schemas:
        print("No schemas available to generate ERD")
        return
    
    print(f"\nGenerating Mermaid ERD syntax to {MERMAID_OUTPUT_FILE}")
    
    # Start building the Mermaid ERD syntax
    mermaid_syntax = """---
title: Data Warehouse Schema
---
erDiagram
    %% This is an auto-generated ERD diagram based on Parquet file schemas
    %% NOTE: Relationship lines need to be added manually after reviewing this generated syntax
    %% Use the PK and FK comments as guides for adding relationship lines
    
"""
    
    # Add entities
    for table_name, fields in schemas.items():
        mermaid_syntax += f"    {table_name} {{\n"
        
        for field in fields:
            field_name = field['name']
            simplified_type = field['simplified_type']
            
            # Add annotations for likely primary and foreign keys
            annotations = []
            if field['is_pk']:
                annotations.append("PK")
            if field['is_fk']:
                annotations.append("FK")
            
            annotation_str = f" %% {', '.join(annotations)}" if annotations else ""
            mermaid_syntax += f"        {simplified_type} {field_name}{annotation_str}\n"
        
        mermaid_syntax += "    }\n\n"
    
    # Add a note about manually adding relationships
    mermaid_syntax += """    %% Example relationship (add these manually based on FK relationships):
    %% dim_time ||--o{ fact_table : "time_sk"
    %%
    %% Relationship types:
    %% ||--|| : one-to-one
    %% ||--o{ : one-to-many
    %% }o--|| : many-to-one
    %% }o--o{ : many-to-many
"""
    
    # Write to file
    with open(MERMAID_OUTPUT_FILE, 'w') as f:
        f.write(mermaid_syntax)
    
    print(f"ERD Mermaid syntax saved to {MERMAID_OUTPUT_FILE}")

if __name__ == "__main__":
    # Extract schemas and print them
    schemas = extract_schemas()
    
    # Generate Mermaid ERD
    generate_mermaid_erd(schemas)
    
    print("\nProcess completed successfully!") 