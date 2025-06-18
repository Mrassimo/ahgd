#!/usr/bin/env python3
"""
AIHW Health Indicator Data Extraction Script

This script extracts health indicator data from multiple AIHW and related sources:
1. AIHW MORT books (mortality data by geography)
2. AIHW GRIM books (historical mortality data)
3. PHIDU Social Health Atlas data (chronic disease prevalence by SA2/PHA)
4. Australian Atlas of Healthcare Variation data sheets

Target health indicators:
- Chronic disease prevalence (diabetes, heart disease, mental health)
- Health service utilisation rates
- Health outcomes and mortality data
- Preventive health service access

Geographic compatibility: SA2 codes, Population Health Areas, LGA codes
"""

import requests
import pandas as pd
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import zipfile
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIHWDataExtractor:
    """Extract and process AIHW health indicator data for geographic analysis"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.data_dir = base_dir / "data" / "raw" / "health"
        self.processed_dir = base_dir / "data" / "processed"
        self.db_path = base_dir / "health_analytics.db"
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Data source URLs (verified as working)
        self.data_sources = {
            'mort_table1': 'https://data.gov.au/data/dataset/a84a6e8e-dd8f-4bae-a79d-77a5e32877ad/resource/a5de4e7e-d062-4356-9d1b-39f44b1961dc/download/aihw-phe-229-mort-table1-data-gov-au-2025.csv',
            'mort_table2': 'https://data.gov.au/data/dataset/a84a6e8e-dd8f-4bae-a79d-77a5e32877ad/resource/3b7d81af-943f-447d-9d64-9ce220be35e7/download/aihw-phe-229-mort-table2-data-gov-au-2025.csv',
            'grim_data': 'https://data.gov.au/data/dataset/488ef6d4-c763-4b24-b8fb-9c15b67ece19/resource/edcbc14c-ba7c-44ae-9d4f-2622ad3fafe0/download/aihw-phe-229-grim-data-gov-au-2025.csv',
            'phidu_pha_aust': 'https://phidu.torrens.edu.au/current/data/sha-aust/pha/phidu_data_pha_aust.xlsx',
            'phidu_lga_aust': 'https://phidu.torrens.edu.au/current/data/sha-aust/lga/phidu_data_lga_aust.xls'
        }
    
    def download_file(self, url: str, filename: str, max_size_mb: int = 100) -> Optional[Path]:
        """Download a file with size and error checking"""
        file_path = self.data_dir / filename
        
        # Skip if already exists
        if file_path.exists():
            logger.info(f"File already exists: {filename}")
            return file_path
        
        try:
            logger.info(f"Downloading {filename} from {url}")
            
            # Check file size before downloading
            response = requests.head(url, timeout=30)
            if response.status_code == 200:
                size_mb = int(response.headers.get('content-length', 0)) / (1024 * 1024)
                if size_mb > max_size_mb:
                    logger.warning(f"File too large: {size_mb:.1f}MB > {max_size_mb}MB")
                    return None
            
            # Download file
            response = requests.get(url, timeout=300, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Successfully downloaded: {filename} ({file_path.stat().st_size / (1024*1024):.1f}MB)")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to download {filename}: {str(e)}")
            return None
    
    def extract_mort_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Extract AIHW MORT books data (mortality by geography)"""
        logger.info("Extracting AIHW MORT books data...")
        
        # Download MORT Table 1 (main mortality indicators)
        mort1_path = self.download_file(
            self.data_sources['mort_table1'], 
            'aihw_mort_table1_2025.csv'
        )
        
        # Download MORT Table 2 (detailed breakdowns)
        mort2_path = self.download_file(
            self.data_sources['mort_table2'], 
            'aihw_mort_table2_2025.csv'
        )
        
        mort1_df = None
        mort2_df = None
        
        # Process MORT Table 1
        if mort1_path and mort1_path.exists():
            try:
                mort1_df = pd.read_csv(mort1_path, encoding='utf-8')
                logger.info(f"MORT Table 1 loaded: {mort1_df.shape} rows x columns")
                logger.info(f"Columns: {list(mort1_df.columns)}")
            except Exception as e:
                logger.error(f"Error reading MORT Table 1: {str(e)}")
        
        # Process MORT Table 2
        if mort2_path and mort2_path.exists():
            try:
                mort2_df = pd.read_csv(mort2_path, encoding='utf-8')
                logger.info(f"MORT Table 2 loaded: {mort2_df.shape} rows x columns")
                logger.info(f"Columns: {list(mort2_df.columns)}")
            except Exception as e:
                logger.error(f"Error reading MORT Table 2: {str(e)}")
        
        return mort1_df, mort2_df
    
    def extract_grim_data(self) -> Optional[pd.DataFrame]:
        """Extract AIHW GRIM books data (historical mortality)"""
        logger.info("Extracting AIHW GRIM books data...")
        
        grim_path = self.download_file(
            self.data_sources['grim_data'], 
            'aihw_grim_data_2025.csv'
        )
        
        if not grim_path or not grim_path.exists():
            return None
        
        try:
            grim_df = pd.read_csv(grim_path, encoding='utf-8')
            logger.info(f"GRIM data loaded: {grim_df.shape} rows x columns")
            logger.info(f"Columns: {list(grim_df.columns)}")
            return grim_df
        except Exception as e:
            logger.error(f"Error reading GRIM data: {str(e)}")
            return None
    
    def extract_phidu_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Extract PHIDU Social Health Atlas data (chronic disease prevalence)"""
        logger.info("Extracting PHIDU Social Health Atlas data...")
        
        # Download PHA (Population Health Area) data - contains SA2 level data
        pha_path = self.download_file(
            self.data_sources['phidu_pha_aust'], 
            'phidu_pha_australia.xlsx'
        )
        
        # Download LGA (Local Government Area) data
        lga_path = self.download_file(
            self.data_sources['phidu_lga_aust'], 
            'phidu_lga_australia.xls'
        )
        
        pha_df = None
        lga_df = None
        
        # Process PHA data (includes SA2 mappings)
        if pha_path and pha_path.exists():
            try:
                # PHIDU Excel files often have multiple sheets
                # Try to load the main data sheet
                excel_file = pd.ExcelFile(pha_path)
                logger.info(f"PHA Excel sheets: {excel_file.sheet_names}")
                
                # Usually the main data is in the first sheet or 'Data' sheet
                main_sheet = excel_file.sheet_names[0]
                if 'Data' in excel_file.sheet_names:
                    main_sheet = 'Data'
                
                pha_df = pd.read_excel(pha_path, sheet_name=main_sheet)
                logger.info(f"PHA data loaded: {pha_df.shape} rows x columns")
                logger.info(f"Columns: {list(pha_df.columns)[:10]}...")  # Show first 10 columns
                
            except Exception as e:
                logger.error(f"Error reading PHA data: {str(e)}")
        
        # Process LGA data
        if lga_path and lga_path.exists():
            try:
                excel_file = pd.ExcelFile(lga_path)
                logger.info(f"LGA Excel sheets: {excel_file.sheet_names}")
                
                # Try to find a data sheet with actual health indicators
                # Look for sheets with health-related names
                data_sheets = [sheet for sheet in excel_file.sheet_names 
                              if any(keyword in sheet.lower() for keyword in 
                                   ['health', 'chronic', 'mortality', 'census_health_condition'])]
                
                if data_sheets:
                    main_sheet = data_sheets[0]  # Use first health-related sheet
                    logger.info(f"Using health data sheet: {main_sheet}")
                else:
                    # Fall back to a meaningful sheet (avoid front page/contents)
                    avoid_sheets = ['Front_page', 'Topics', 'Contents', 'Key', 'Notes_on_the_data']
                    data_sheets = [sheet for sheet in excel_file.sheet_names if sheet not in avoid_sheets]
                    main_sheet = data_sheets[0] if data_sheets else excel_file.sheet_names[0]
                    logger.info(f"Using fallback sheet: {main_sheet}")
                
                lga_df = pd.read_excel(lga_path, sheet_name=main_sheet)
                logger.info(f"LGA data loaded: {lga_df.shape} rows x columns")
                logger.info(f"Columns: {list(lga_df.columns)[:10]}...")  # Show first 10 columns
                
            except Exception as e:
                logger.error(f"Error reading LGA data: {str(e)}")
        
        return pha_df, lga_df
    
    def identify_chronic_disease_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify columns related to chronic diseases in PHIDU data"""
        if df is None:
            return []
        
        chronic_keywords = [
            'diabetes', 'heart', 'cardiovascular', 'cancer', 'mental', 'depression',
            'anxiety', 'asthma', 'copd', 'arthritis', 'kidney', 'stroke', 'dementia',
            'alzheimer', 'lung', 'obesity', 'hypertension', 'blood pressure'
        ]
        
        chronic_columns = []
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in chronic_keywords):
                chronic_columns.append(col)
        
        logger.info(f"Identified {len(chronic_columns)} chronic disease columns")
        return chronic_columns
    
    def create_database_tables(self):
        """Create database tables for AIHW health indicator data"""
        logger.info("Creating database tables for AIHW data...")
        
        with sqlite3.connect(self.db_path) as conn:
            # MORT data table (mortality indicators by geography)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS aihw_mort_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    year INTEGER,
                    geography_type TEXT,
                    geography_code TEXT,
                    geography_name TEXT,
                    indicator_type TEXT,
                    sex TEXT,
                    age_group TEXT,
                    cause_of_death TEXT,
                    value REAL,
                    unit TEXT,
                    source_table TEXT,
                    extraction_date TEXT
                )
            ''')
            
            # GRIM data table (historical mortality)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS aihw_grim_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    year INTEGER,
                    cause_of_death TEXT,
                    sex TEXT,
                    age_group TEXT,
                    deaths INTEGER,
                    rate REAL,
                    extraction_date TEXT
                )
            ''')
            
            # PHIDU chronic disease data (by Population Health Area)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS phidu_chronic_disease (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    geography_type TEXT,
                    geography_code TEXT,
                    geography_name TEXT,
                    sa2_codes TEXT,  -- For PHA data, store related SA2 codes
                    indicator_name TEXT,
                    indicator_value REAL,
                    indicator_unit TEXT,
                    year TEXT,
                    sex TEXT,
                    age_group TEXT,
                    data_source TEXT,
                    extraction_date TEXT
                )
            ''')
            
            # Index for better query performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_mort_geography ON aihw_mort_data(geography_type, geography_code)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_phidu_geography ON phidu_chronic_disease(geography_type, geography_code)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_mort_indicator ON aihw_mort_data(indicator_type, cause_of_death)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_phidu_indicator ON phidu_chronic_disease(indicator_name)')
            
            conn.commit()
            logger.info("Database tables created successfully")
    
    def process_and_store_data(self):
        """Main processing pipeline: extract, clean, and store all AIHW data"""
        logger.info("Starting AIHW data extraction and processing...")
        
        extraction_date = datetime.now().isoformat()
        
        # Create database tables
        self.create_database_tables()
        
        # Extract all data sources
        mort1_df, mort2_df = self.extract_mort_data()
        grim_df = self.extract_grim_data()
        pha_df, lga_df = self.extract_phidu_data()
        
        # Store extraction results summary
        results = {
            'extraction_date': extraction_date,
            'mort_table1_records': len(mort1_df) if mort1_df is not None else 0,
            'mort_table2_records': len(mort2_df) if mort2_df is not None else 0,
            'grim_records': len(grim_df) if grim_df is not None else 0,
            'phidu_pha_records': len(pha_df) if pha_df is not None else 0,
            'phidu_lga_records': len(lga_df) if lga_df is not None else 0,
        }
        
        # Save processed data as parquet files for efficient loading
        if mort1_df is not None:
            mort1_df.to_parquet(self.processed_dir / 'aihw_mort_table1.parquet')
        if mort2_df is not None:
            mort2_df.to_parquet(self.processed_dir / 'aihw_mort_table2.parquet')
        if grim_df is not None:
            grim_df.to_parquet(self.processed_dir / 'aihw_grim_data.parquet')
        if pha_df is not None:
            pha_df.to_parquet(self.processed_dir / 'phidu_pha_data.parquet')
        if lga_df is not None:
            lga_df.to_parquet(self.processed_dir / 'phidu_lga_data.parquet')
        
        logger.info("AIHW data extraction completed successfully")
        logger.info(f"Results: {results}")
        
        return results
    
    def validate_data_quality(self) -> Dict[str, int]:
        """Validate extracted data quality and coverage"""
        logger.info("Validating data quality...")
        
        validation_results = {}
        
        # Check processed files exist
        processed_files = [
            'aihw_mort_table1.parquet',
            'aihw_mort_table2.parquet', 
            'aihw_grim_data.parquet',
            'phidu_pha_data.parquet',
            'phidu_lga_data.parquet'
        ]
        
        for file_name in processed_files:
            file_path = self.processed_dir / file_name
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    validation_results[file_name] = {
                        'records': len(df),
                        'columns': len(df.columns),
                        'missing_values': df.isnull().sum().sum(),
                        'file_size_mb': file_path.stat().st_size / (1024 * 1024)
                    }
                except Exception as e:
                    validation_results[file_name] = {'error': str(e)}
            else:
                validation_results[file_name] = {'status': 'not_found'}
        
        logger.info(f"Data validation completed: {validation_results}")
        return validation_results

def main():
    """Run the AIHW data extraction process"""
    base_dir = Path(__file__).parent.parent
    extractor = AIHWDataExtractor(base_dir)
    
    try:
        # Run the extraction and processing
        results = extractor.process_and_store_data()
        
        # Validate the results
        validation = extractor.validate_data_quality()
        
        # Create summary report
        summary = {
            'extraction_results': results,
            'validation_results': validation,
            'status': 'success'
        }
        
        logger.info("AIHW data extraction pipeline completed successfully")
        return summary
        
    except Exception as e:
        logger.error(f"Error in AIHW data extraction: {str(e)}")
        return {'status': 'error', 'message': str(e)}

if __name__ == "__main__":
    results = main()
    print(f"Extraction results: {results}")