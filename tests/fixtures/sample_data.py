"""
Sample data fixtures for testing.

This module provides consistent sample data for testing various components
of the Australian Health Geography Data Analytics system.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import json


def get_sample_postcode_data() -> pd.DataFrame:
    """Generate sample postcode-level data for testing."""
    return pd.DataFrame({
        'postcode': ['2000', '2001', '2002', '3000', '3001', '3002', '4000', '4001', '5000', '6000'],
        'population': [15000, 12000, 18000, 25000, 18000, 22000, 20000, 16000, 16000, 19000],
        'median_income': [65000, 58000, 72000, 72000, 55000, 68000, 62000, 59000, 59000, 63000],
        'hospitals': [2, 1, 3, 3, 2, 2, 2, 1, 1, 2],
        'area_sqkm': [5.2, 8.1, 12.3, 12.3, 6.7, 9.4, 15.4, 11.2, 9.8, 13.7],
        'unemployment_rate': [4.2, 5.1, 3.8, 3.5, 6.2, 4.1, 4.8, 5.5, 5.0, 4.3]
    })


def get_sample_sa2_data() -> pd.DataFrame:
    """Generate sample SA2-level data for testing."""
    return pd.DataFrame({
        'sa2_main16': ['101021007', '101021008', '101021009', '201011001', '201011002', 
                      '301011003', '401011004', '501011005', '601011006', '701011007'],
        'sa2_name16': ['Sydney - CBD', 'Sydney - Haymarket', 'Sydney - The Rocks',
                      'Melbourne - CBD', 'Melbourne - Docklands', 'Brisbane - CBD',
                      'Adelaide - CBD', 'Perth - CBD', 'Hobart - CBD', 'Darwin - CBD'],
        'population': [18500, 22000, 15000, 28000, 15500, 23000, 19000, 17000, 12000, 14000],
        'disadvantage_score': [1050, 980, 1150, 1120, 1200, 1030, 1080, 1090, 970, 1010],
        'health_outcome': ['Good', 'Fair', 'Excellent', 'Excellent', 'Good', 
                          'Good', 'Good', 'Good', 'Fair', 'Good']
    })


def get_sample_seifa_data() -> pd.DataFrame:
    """Generate sample SEIFA disadvantage data for testing."""
    return pd.DataFrame({
        'sa2_code_2021': ['101021007', '101021008', '101021009', '201011001', '201011002',
                          '301011003', '401011004', '501011005', '601011006', '701011007'],
        'sa2_name_2021': ['Sydney - CBD', 'Sydney - Haymarket', 'Sydney - The Rocks',
                          'Melbourne - CBD', 'Melbourne - Docklands', 'Brisbane - CBD',
                          'Adelaide - CBD', 'Perth - CBD', 'Hobart - CBD', 'Darwin - CBD'],
        'irsad_score': [1050.5, 980.2, 1150.8, 1120.8, 1200.1, 1030.3, 1080.7, 1090.2, 970.5, 1010.9],
        'irsad_decile': [7, 5, 8, 8, 9, 6, 7, 7, 4, 6],
        'irsd_score': [1040.2, 975.8, 1145.3, 1115.3, 1195.7, 1025.1, 1075.4, 1085.8, 965.2, 1005.6],
        'irsd_decile': [7, 5, 8, 8, 9, 6, 7, 7, 4, 6],
        'ier_score': [1055.8, 985.4, 1155.2, 1125.1, 1205.3, 1035.7, 1085.9, 1095.4, 975.8, 1015.2],
        'ier_decile': [7, 5, 8, 8, 9, 6, 7, 7, 4, 6]
    })


def get_sample_health_data() -> pd.DataFrame:
    """Generate sample health outcome data for testing."""
    return pd.DataFrame({
        'sa2_code': ['101021007', '101021008', '101021009', '201011001', '201011002',
                    '301011003', '401011004', '501011005', '601011006', '701011007'],
        'year': [2021] * 10,
        'mortality_rate': [5.2, 6.8, 4.1, 3.9, 4.5, 5.8, 5.1, 4.7, 7.2, 6.1],
        'chronic_disease_rate': [15.2, 18.4, 12.8, 11.5, 14.1, 16.7, 15.9, 14.3, 20.1, 17.5],
        'mental_health_rate': [8.9, 12.1, 7.6, 6.8, 9.3, 10.2, 9.7, 8.4, 13.5, 11.8],
        'diabetes_rate': [6.1, 7.8, 5.2, 4.9, 6.5, 6.9, 6.3, 5.8, 8.4, 7.2],
        'heart_disease_rate': [4.3, 5.9, 3.7, 3.2, 4.1, 4.8, 4.5, 4.0, 6.2, 5.4],
        'population': [18500, 22000, 15000, 28000, 15500, 23000, 19000, 17000, 12000, 14000]
    })


def get_sample_correspondence_data() -> pd.DataFrame:
    """Generate sample postcode-SA2 correspondence data for testing."""
    return pd.DataFrame({
        'POA_CODE_2021': ['2000', '2000', '2001', '2002', '3000', '3001', '3002', 
                          '4000', '4001', '5000', '6000'],
        'SA2_CODE_2021': ['101021007', '101021008', '101021009', '101021009', '201011001', 
                          '201011002', '201011002', '301011003', '301011003', '401011004', '501011005'],
        'SA2_NAME_2021': ['Sydney - CBD', 'Sydney - Haymarket', 'Sydney - The Rocks', 'Sydney - The Rocks',
                          'Melbourne - CBD', 'Melbourne - Docklands', 'Melbourne - Docklands',
                          'Brisbane - CBD', 'Brisbane - CBD', 'Adelaide - CBD', 'Perth - CBD'],
        'RATIO': [0.6, 0.4, 1.0, 1.0, 1.0, 0.7, 0.3, 0.8, 0.2, 1.0, 1.0]
    })


def get_sample_demographic_data() -> pd.DataFrame:
    """Generate sample demographic data for testing."""
    return pd.DataFrame({
        'sa2_code': ['101021007', '101021008', '101021009', '201011001', '201011002'],
        'total_population': [18500, 22000, 15000, 28000, 15500],
        'male_population': [9200, 11100, 7400, 13800, 7600],
        'female_population': [9300, 10900, 7600, 14200, 7900],
        'median_age': [32.5, 28.1, 35.2, 31.8, 29.7],
        'age_0_14': [2900, 3200, 2100, 4200, 2400],
        'age_15_64': [13800, 16500, 10500, 20300, 11200],
        'age_65_plus': [1800, 2300, 2400, 3500, 1900],
        'indigenous_population': [185, 440, 225, 280, 310],
        'overseas_born': [7400, 8900, 5100, 11200, 6200]
    })


def get_sample_time_series_data() -> pd.DataFrame:
    """Generate sample time series health data for testing."""
    years = [2018, 2019, 2020, 2021]
    sa2_codes = ['101021007', '201011001', '301011003']
    
    data = []
    for year in years:
        for sa2_code in sa2_codes:
            base_mortality = {'101021007': 5.2, '201011001': 3.9, '301011003': 5.8}[sa2_code]
            # Add some year-over-year variation
            mortality_rate = base_mortality + np.random.normal(0, 0.3)
            
            data.append({
                'sa2_code': sa2_code,
                'year': year,
                'mortality_rate': max(0, mortality_rate),
                'chronic_disease_rate': max(0, base_mortality * 3 + np.random.normal(0, 1)),
                'population': 15000 + np.random.randint(-2000, 3000)
            })
    
    return pd.DataFrame(data)


def get_sample_geographic_boundaries() -> Dict[str, Any]:
    """Generate sample geographic boundary data (simplified GeoJSON format)."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "SA2_MAIN16": "101021007",
                    "SA2_NAME16": "Sydney - CBD",
                    "AREASQKM16": 5.2
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [151.205, -33.865],
                        [151.215, -33.865],
                        [151.215, -33.875],
                        [151.205, -33.875],
                        [151.205, -33.865]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "SA2_MAIN16": "201011001",
                    "SA2_NAME16": "Melbourne - CBD",
                    "AREASQKM16": 12.3
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [144.955, -37.815],
                        [144.975, -37.815],
                        [144.975, -37.835],
                        [144.955, -37.835],
                        [144.955, -37.815]
                    ]]
                }
            }
        ]
    }


def get_sample_config_data() -> Dict[str, Any]:
    """Generate sample configuration data for testing."""
    return {
        'environment': 'development',
        'database': {
            'name': 'test_health_analytics.db',
            'connection_timeout': 30,
            'backup_enabled': True
        },
        'data_source': {
            'chunk_size': 1000,
            'max_file_size_mb': 100
        },
        'dashboard': {
            'host': 'localhost',
            'port': 8501,
            'debug': True,
            'page_title': 'Test Dashboard'
        },
        'processing': {
            'max_workers': 2,
            'correlation_threshold': 0.5
        },
        'logging': {
            'level': 'DEBUG',
            'file_enabled': True,
            'console_enabled': True
        }
    }


def populate_sample_database(db_path: str) -> None:
    """Populate a database with sample data for testing."""
    import duckdb
    
    conn = duckdb.connect(db_path)
    
    # Create and populate correspondence table
    correspondence_df = get_sample_correspondence_data()
    conn.execute("CREATE TABLE IF NOT EXISTS correspondence AS SELECT * FROM correspondence_df")
    
    # Create and populate health data table
    health_df = get_sample_health_data()
    conn.execute("CREATE TABLE IF NOT EXISTS health_data AS SELECT * FROM health_df")
    
    # Create and populate SEIFA data table
    seifa_df = get_sample_seifa_data()
    conn.execute("CREATE TABLE IF NOT EXISTS seifa_data AS SELECT * FROM seifa_df")
    
    # Create and populate demographic data table
    demographic_df = get_sample_demographic_data()
    conn.execute("CREATE TABLE IF NOT EXISTS demographic_data AS SELECT * FROM demographic_df")
    
    conn.close()


def get_sample_database_schema() -> Dict[str, List[str]]:
    """Get the schema definition for sample database tables."""
    return {
        'correspondence': [
            'POA_CODE_2021 VARCHAR',
            'SA2_CODE_2021 VARCHAR', 
            'SA2_NAME_2021 VARCHAR',
            'RATIO DOUBLE'
        ],
        'health_data': [
            'sa2_code VARCHAR',
            'year INTEGER',
            'mortality_rate DOUBLE',
            'chronic_disease_rate DOUBLE',
            'mental_health_rate DOUBLE',
            'diabetes_rate DOUBLE',
            'heart_disease_rate DOUBLE',
            'population INTEGER'
        ],
        'seifa_data': [
            'sa2_code_2021 VARCHAR',
            'sa2_name_2021 VARCHAR',
            'irsad_score DOUBLE',
            'irsad_decile INTEGER',
            'irsd_score DOUBLE',
            'irsd_decile INTEGER',
            'ier_score DOUBLE',
            'ier_decile INTEGER'
        ],
        'demographic_data': [
            'sa2_code VARCHAR',
            'total_population INTEGER',
            'male_population INTEGER',
            'female_population INTEGER',
            'median_age DOUBLE',
            'age_0_14 INTEGER',
            'age_15_64 INTEGER',
            'age_65_plus INTEGER',
            'indigenous_population INTEGER',
            'overseas_born INTEGER'
        ]
    }


def get_sample_aihw_data() -> pd.DataFrame:
    """Generate sample AIHW health data for testing."""
    return pd.DataFrame({
        'area_code': ['101021007', '101021008', '201011001', '201011002', '301011003'],
        'area_name': ['Sydney - CBD', 'Sydney - Haymarket', 'Melbourne - CBD', 'Melbourne - Docklands', 'Brisbane - CBD'],
        'indicator': ['Mortality rate', 'Mortality rate', 'Mortality rate', 'Mortality rate', 'Mortality rate'],
        'year': [2021, 2021, 2021, 2021, 2021],
        'value': [5.2, 6.8, 3.9, 4.5, 5.8],
        'numerator': [96, 149, 109, 70, 133],
        'denominator': [18500, 22000, 28000, 15500, 23000],
        'confidence_interval_lower': [4.8, 6.2, 3.6, 4.1, 5.3],
        'confidence_interval_upper': [5.6, 7.4, 4.2, 4.9, 6.3]
    })


def get_sample_performance_data() -> Dict[str, Any]:
    """Generate sample performance testing data."""
    return {
        'large_dataset_size': 10000,
        'memory_usage_mb': 45.2,
        'processing_time_seconds': 2.3,
        'database_query_time_ms': 150,
        'visualization_render_time_ms': 800,
        'concurrent_users': 5,
        'response_time_p95_ms': 1200
    }


def create_large_sample_dataset(num_rows: int = 10000) -> pd.DataFrame:
    """Create a large sample dataset for performance testing."""
    np.random.seed(42)  # For reproducible tests
    
    sa2_codes = [f"10{i:07d}" for i in range(1000, 1000 + num_rows//10)]
    
    data = []
    for i in range(num_rows):
        sa2_code = np.random.choice(sa2_codes)
        data.append({
            'sa2_code': sa2_code,
            'year': np.random.choice([2018, 2019, 2020, 2021]),
            'population': np.random.randint(5000, 50000),
            'mortality_rate': np.random.normal(5.5, 1.2),
            'chronic_disease_rate': np.random.normal(15.0, 3.0),
            'mental_health_rate': np.random.normal(9.0, 2.5),
            'disadvantage_score': np.random.normal(1000, 100),
            'median_income': np.random.normal(60000, 15000)
        })
    
    return pd.DataFrame(data)
    return pd.DataFrame({
        'measure': ['Mortality rate', 'Hospitalisation rate', 'Disease prevalence', 'Risk factor prevalence'],
        'age_group': ['All ages', '0-64 years', '65+ years', 'All ages'],
        'sex': ['Persons', 'Males', 'Females', 'Persons'],
        'year': [2021, 2021, 2021, 2021],
        'value': [5.2, 120.5, 15.3, 25.8],
        'unit': ['per 1,000', 'per 100,000', 'percentage', 'percentage'],
        'geography_level': ['SA2', 'SA2', 'SA2', 'SA2'],
        'geography_code': ['101021007', '101021007', '101021007', '101021007'],
        'data_source': ['ABS', 'AIHW', 'ABS', 'AIHW']
    })


def create_sample_files(temp_dir: Path) -> Dict[str, Path]:
    """Create sample data files in temporary directory for testing."""
    files_created = {}
    
    # Create CSV files
    csv_files = {
        'postcode_data.csv': get_sample_postcode_data(),
        'sa2_data.csv': get_sample_sa2_data(),
        'seifa_data.csv': get_sample_seifa_data(),
        'health_data.csv': get_sample_health_data(),
        'demographic_data.csv': get_sample_demographic_data(),
        'time_series_data.csv': get_sample_time_series_data(),
        'aihw_data.csv': get_sample_aihw_data()
    }
    
    for filename, data in csv_files.items():
        file_path = temp_dir / filename
        data.to_csv(file_path, index=False)
        files_created[filename] = file_path
    
    # Create Excel files
    excel_files = {
        'correspondence_data.xlsx': get_sample_correspondence_data(),
        'seifa_excel.xlsx': get_sample_seifa_data()
    }
    
    for filename, data in excel_files.items():
        file_path = temp_dir / filename
        data.to_excel(file_path, index=False)
        files_created[filename] = file_path
    
    # Create Parquet files
    parquet_files = {
        'health_parquet.parquet': get_sample_health_data(),
        'demographic_parquet.parquet': get_sample_demographic_data()
    }
    
    for filename, data in parquet_files.items():
        file_path = temp_dir / filename
        data.to_parquet(file_path)
        files_created[filename] = file_path
    
    # Create JSON files
    json_files = {
        'geographic_boundaries.json': get_sample_geographic_boundaries(),
        'config.json': get_sample_config_data()
    }
    
    for filename, data in json_files.items():
        file_path = temp_dir / filename
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        files_created[filename] = file_path
    
    return files_created


def get_sample_database_schema() -> List[str]:
    """Get sample database schema DDL statements for testing."""
    return [
        """
        CREATE TABLE correspondence (
            POA_CODE_2021 TEXT,
            SA2_CODE_2021 TEXT,
            SA2_NAME_2021 TEXT,
            RATIO REAL
        )
        """,
        """
        CREATE TABLE seifa_data (
            sa2_code_2021 TEXT PRIMARY KEY,
            sa2_name_2021 TEXT,
            irsad_score REAL,
            irsad_decile INTEGER,
            irsd_score REAL,
            irsd_decile INTEGER
        )
        """,
        """
        CREATE TABLE health_outcomes (
            sa2_code TEXT,
            year INTEGER,
            mortality_rate REAL,
            chronic_disease_rate REAL,
            mental_health_rate REAL,
            population INTEGER,
            PRIMARY KEY (sa2_code, year)
        )
        """,
        """
        CREATE TABLE demographic_data (
            sa2_code TEXT PRIMARY KEY,
            total_population INTEGER,
            male_population INTEGER,
            female_population INTEGER,
            median_age REAL,
            indigenous_population INTEGER,
            overseas_born INTEGER
        )
        """
    ]


def populate_sample_database(db_path: Path) -> None:
    """Populate a database with sample data for testing."""
    import duckdb
    
    conn = duckdb.connect(str(db_path))
    
    try:
        # Create tables
        schema_statements = get_sample_database_schema()
        for statement in schema_statements:
            conn.execute(statement)
        
        # Insert data
        datasets = {
            'correspondence': get_sample_correspondence_data(),
            'seifa_data': get_sample_seifa_data(),
            'health_outcomes': get_sample_health_data(),
            'demographic_data': get_sample_demographic_data()
        }
        
        for table_name, df in datasets.items():
            # Adjust column names to match schema
            if table_name == 'health_outcomes':
                df = df.rename(columns={
                    'mortality_rate': 'mortality_rate',
                    'chronic_disease_rate': 'chronic_disease_rate',
                    'mental_health_rate': 'mental_health_rate'
                })
            
            conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")
        
        conn.commit()
        
    finally:
        conn.close()


# Utility functions for test data validation
def validate_sample_data_consistency() -> Dict[str, bool]:
    """Validate consistency across sample datasets."""
    results = {}
    
    try:
        # Get sample data
        correspondence = get_sample_correspondence_data()
        seifa = get_sample_seifa_data()
        health = get_sample_health_data()
        
        # Check SA2 code consistency
        correspondence_sa2s = set(correspondence['SA2_CODE_2021'])
        seifa_sa2s = set(seifa['sa2_code_2021'])
        health_sa2s = set(health['sa2_code'])
        
        results['sa2_codes_consistent'] = len(correspondence_sa2s & seifa_sa2s & health_sa2s) > 0
        
        # Check population data consistency
        health_populations = health['population'].tolist()
        results['populations_positive'] = all(p > 0 for p in health_populations)
        
        # Check rate data validity
        mortality_rates = health['mortality_rate'].tolist()
        results['mortality_rates_valid'] = all(0 <= r <= 100 for r in mortality_rates)
        
        # Check correspondence ratios sum to reasonable values
        postcode_groups = correspondence.groupby('POA_CODE_2021')['RATIO'].sum()
        results['correspondence_ratios_valid'] = all(0.8 <= r <= 1.2 for r in postcode_groups)
        
    except Exception as e:
        results['validation_error'] = str(e)
    
    return results


if __name__ == "__main__":
    # Test sample data generation and validation
    print("Generating sample data...")
    
    # Test data generation
    postcode_data = get_sample_postcode_data()
    print(f"Generated {len(postcode_data)} postcode records")
    
    health_data = get_sample_health_data()
    print(f"Generated {len(health_data)} health records")
    
    # Test data validation
    validation_results = validate_sample_data_consistency()
    print("\nData validation results:")
    for check, result in validation_results.items():
        print(f"  {check}: {result}")
    
    print("\nSample data generation complete!")
