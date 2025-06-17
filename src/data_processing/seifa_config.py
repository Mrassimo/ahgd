"""
SEIFA 2021 Data Processing Configuration

Configuration file for processing the real Australian SEIFA 2021 SA2 data
based on the comprehensive data structure analysis.

Generated from real data analysis: 2025-06-17
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

@dataclass
class SEIFAIndexConfig:
    """Configuration for a single SEIFA index."""
    name: str
    full_name: str
    score_column: int
    decile_column: int
    description: str
    interpretation: str

@dataclass
class SEIFAProcessingConfig:
    """Complete configuration for SEIFA data processing."""
    
    # File and sheet configuration
    filename: str = "SEIFA_2021_SA2_Indexes.xlsx"
    primary_sheet: str = "Table 1"
    header_row: int = 6
    data_start_row: int = 7
    
    # Expected data dimensions
    expected_records: int = 2368
    total_columns: int = 11
    
    # Column mappings
    sa2_code_column: int = 1
    sa2_name_column: int = 2
    population_column: int = 11
    
    # Data validation ranges
    score_range: Tuple[int, int] = (800, 1200)
    decile_range: Tuple[int, int] = (1, 10)
    min_population: int = 0
    max_population: int = 200000  # Reasonable upper bound for SA2
    
    # SEIFA indices configuration
    indices: Dict[str, SEIFAIndexConfig] = None
    
    def __post_init__(self):
        """Initialize SEIFA indices configuration."""
        if self.indices is None:
            self.indices = {
                'irsd': SEIFAIndexConfig(
                    name='irsd',
                    full_name='Index of Relative Socio-economic Disadvantage',
                    score_column=3,
                    decile_column=4,
                    description='Focuses on low income, low skill, high unemployment, lack of qualifications',
                    interpretation='Lower scores = more disadvantaged. Decile 1 = most disadvantaged 10%'
                ),
                'irsad': SEIFAIndexConfig(
                    name='irsad',
                    full_name='Index of Relative Socio-economic Advantage and Disadvantage',
                    score_column=5,
                    decile_column=6,
                    description='Includes both advantage and disadvantage measures',
                    interpretation='More comprehensive than IRSD, includes high income alongside disadvantage'
                ),
                'ier': SEIFAIndexConfig(
                    name='ier',
                    full_name='Index of Economic Resources',
                    score_column=7,
                    decile_column=8,
                    description='Focuses on household income, rent/mortgage costs, dwelling size',
                    interpretation='Decile 1 = fewest economic resources, 10 = most resources'
                ),
                'ieo': SEIFAIndexConfig(
                    name='ieo',
                    full_name='Index of Education and Occupation',
                    score_column=9,
                    decile_column=10,
                    description='Focuses on education qualifications and skilled occupations',
                    interpretation='Decile 1 = lowest education/occupation, 10 = highest'
                )
            }

# Default configuration instance
SEIFA_CONFIG = SEIFAProcessingConfig()

# Database schema for SEIFA data
SEIFA_DATABASE_SCHEMA = {
    'table_name': 'seifa_2021_sa2',
    'columns': [
        ('sa2_code_2021', 'VARCHAR(9)', 'PRIMARY KEY'),
        ('sa2_name_2021', 'VARCHAR(100)', 'NOT NULL'),
        ('irsd_score', 'INTEGER', 'NOT NULL'),
        ('irsd_decile', 'INTEGER', 'NOT NULL CHECK (irsd_decile BETWEEN 1 AND 10)'),
        ('irsad_score', 'INTEGER', 'NOT NULL'),
        ('irsad_decile', 'INTEGER', 'NOT NULL CHECK (irsad_decile BETWEEN 1 AND 10)'),
        ('ier_score', 'INTEGER', 'NOT NULL'),
        ('ier_decile', 'INTEGER', 'NOT NULL CHECK (ier_decile BETWEEN 1 AND 10)'),
        ('ieo_score', 'INTEGER', 'NOT NULL'),
        ('ieo_decile', 'INTEGER', 'NOT NULL CHECK (ieo_decile BETWEEN 1 AND 10)'),
        ('usual_resident_population', 'INTEGER', 'NOT NULL'),
        ('data_source', 'VARCHAR(50)', 'DEFAULT \'ABS_SEIFA_2021\''),
        ('created_at', 'TIMESTAMP', 'DEFAULT CURRENT_TIMESTAMP'),
        ('updated_at', 'TIMESTAMP', 'DEFAULT CURRENT_TIMESTAMP')
    ],
    'indices': [
        'CREATE INDEX idx_seifa_sa2_code ON seifa_2021_sa2(sa2_code_2021);',
        'CREATE INDEX idx_seifa_irsd_decile ON seifa_2021_sa2(irsd_decile);',
        'CREATE INDEX idx_seifa_irsad_decile ON seifa_2021_sa2(irsad_decile);',
        'CREATE INDEX idx_seifa_ier_decile ON seifa_2021_sa2(ier_decile);',
        'CREATE INDEX idx_seifa_ieo_decile ON seifa_2021_sa2(ieo_decile);'
    ]
}

# Data validation rules
SEIFA_VALIDATION_RULES = {
    'required_fields': [
        'sa2_code_2021',
        'sa2_name_2021',
        'irsd_score',
        'irsd_decile',
        'irsad_score', 
        'irsad_decile',
        'ier_score',
        'ier_decile',
        'ieo_score',
        'ieo_decile',
        'usual_resident_population'
    ],
    'field_types': {
        'sa2_code_2021': str,
        'sa2_name_2021': str,
        'irsd_score': int,
        'irsd_decile': int,
        'irsad_score': int,
        'irsad_decile': int,
        'ier_score': int,
        'ier_decile': int,
        'ieo_score': int,
        'ieo_decile': int,
        'usual_resident_population': int
    },
    'field_constraints': {
        'sa2_code_2021': lambda x: len(str(x)) == 9 and str(x).isdigit(),
        'sa2_name_2021': lambda x: len(str(x).strip()) > 0,
        'irsd_score': lambda x: SEIFA_CONFIG.score_range[0] <= x <= SEIFA_CONFIG.score_range[1],
        'irsd_decile': lambda x: SEIFA_CONFIG.decile_range[0] <= x <= SEIFA_CONFIG.decile_range[1],
        'irsad_score': lambda x: SEIFA_CONFIG.score_range[0] <= x <= SEIFA_CONFIG.score_range[1],
        'irsad_decile': lambda x: SEIFA_CONFIG.decile_range[0] <= x <= SEIFA_CONFIG.decile_range[1],
        'ier_score': lambda x: SEIFA_CONFIG.score_range[0] <= x <= SEIFA_CONFIG.score_range[1],
        'ier_decile': lambda x: SEIFA_CONFIG.decile_range[0] <= x <= SEIFA_CONFIG.decile_range[1],
        'ieo_score': lambda x: SEIFA_CONFIG.score_range[0] <= x <= SEIFA_CONFIG.score_range[1],
        'ieo_decile': lambda x: SEIFA_CONFIG.decile_range[0] <= x <= SEIFA_CONFIG.decile_range[1],
        'usual_resident_population': lambda x: x >= 0
    }
}

# Output format configurations
SEIFA_OUTPUT_FORMATS = {
    'csv': {
        'filename': 'seifa_2021_sa2_processed.csv',
        'encoding': 'utf-8',
        'index': False
    },
    'json': {
        'filename': 'seifa_2021_sa2_processed.json',
        'orient': 'records',
        'indent': 2
    },
    'parquet': {
        'filename': 'seifa_2021_sa2_processed.parquet',
        'compression': 'snappy'
    },
    'geojson': {
        'filename': 'seifa_2021_sa2_with_boundaries.geojson',
        'properties_prefix': 'seifa_'
    }
}

# Processing pipeline configuration
SEIFA_PIPELINE_CONFIG = {
    'processing_steps': [
        'validate_file_exists',
        'load_excel_data',
        'validate_data_structure',
        'clean_and_transform',
        'validate_data_quality',
        'save_to_database',
        'export_formats',
        'generate_summary_stats'
    ],
    'error_handling': {
        'continue_on_validation_errors': False,
        'max_validation_errors': 10,
        'log_validation_errors': True
    },
    'performance': {
        'chunk_size': 1000,
        'parallel_validation': True,
        'memory_limit_mb': 512
    }
}

def get_seifa_file_path(data_dir: Path) -> Path:
    """Get the full path to the SEIFA Excel file."""
    return data_dir / "raw" / SEIFA_CONFIG.filename

def get_column_name_mapping() -> Dict[int, str]:
    """Get mapping from column numbers to field names."""
    mapping = {
        SEIFA_CONFIG.sa2_code_column: 'sa2_code_2021',
        SEIFA_CONFIG.sa2_name_column: 'sa2_name_2021',
        SEIFA_CONFIG.population_column: 'usual_resident_population'
    }
    
    # Add SEIFA index columns
    for index_key, index_config in SEIFA_CONFIG.indices.items():
        mapping[index_config.score_column] = f'{index_key}_score'
        mapping[index_config.decile_column] = f'{index_key}_decile'
    
    return mapping

def get_all_seifa_fields() -> List[str]:
    """Get list of all SEIFA field names in order."""
    mapping = get_column_name_mapping()
    return [mapping[i] for i in sorted(mapping.keys())]

def validate_seifa_record(record: Dict[str, Any]) -> List[str]:
    """
    Validate a single SEIFA record against the validation rules.
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check required fields
    for field in SEIFA_VALIDATION_RULES['required_fields']:
        if field not in record or record[field] is None:
            errors.append(f"Missing required field: {field}")
            continue
            
        # Check field type
        expected_type = SEIFA_VALIDATION_RULES['field_types'][field]
        try:
            if expected_type == int:
                record[field] = int(record[field])
            elif expected_type == str:
                record[field] = str(record[field]).strip()
        except (ValueError, TypeError):
            errors.append(f"Invalid type for {field}: expected {expected_type.__name__}")
            continue
            
        # Check field constraints
        if field in SEIFA_VALIDATION_RULES['field_constraints']:
            constraint_func = SEIFA_VALIDATION_RULES['field_constraints'][field]
            try:
                if not constraint_func(record[field]):
                    errors.append(f"Constraint violation for {field}: {record[field]}")
            except Exception as e:
                errors.append(f"Constraint check error for {field}: {str(e)}")
    
    return errors

# Export main configuration objects
__all__ = [
    'SEIFAIndexConfig',
    'SEIFAProcessingConfig', 
    'SEIFA_CONFIG',
    'SEIFA_DATABASE_SCHEMA',
    'SEIFA_VALIDATION_RULES',
    'SEIFA_OUTPUT_FORMATS',
    'SEIFA_PIPELINE_CONFIG',
    'get_seifa_file_path',
    'get_column_name_mapping',
    'get_all_seifa_fields',
    'validate_seifa_record'
]