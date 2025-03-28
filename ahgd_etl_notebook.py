# -*- coding: utf-8 -*-
"""AHGD ETL Pipeline - Colab Version

This notebook performs the Extract, Transform, Load (ETL) process to create:
1. `geo_dimension.parquet`: Geographic boundaries and attributes from ABS ASGS.
2. `population_dimension.parquet`: Population counts (Total, M/F, Indigenous) from ABS Census G01.
"""

# ==============================================================================
# Cell 1: Mount Google Drive
# ==============================================================================
# @title Mount Google Drive
# @markdown Mounts your Google Drive to `/content/drive`. You will need to authenticate.

from google.colab import drive
import os
from pathlib import Path

drive_mount_point = '/content/drive'
my_drive_path = Path(drive_mount_point) / 'MyDrive/'  # Standard path to your Drive root

print(f"Attempting to mount Google Drive at {drive_mount_point}...")
try:
    # force_remount=True can help avoid issues with stale mounts
    drive.mount(drive_mount_point, force_remount=True)
    print("Google Drive mounted successfully.")
    if not my_drive_path.exists():
        print(f"Warning: Standard path {my_drive_path} not found immediately after mount.")
        print("This might be okay, but check the next cell carefully.")
    else:
        print(f"Confirmed access to: {my_drive_path}")
except Exception as e:
    print(f"ERROR mounting Google Drive: {e}")
    print("Please ensure you followed the authentication prompts correctly.")
    raise SystemExit("Google Drive mount failed. Cannot continue.")

print("\n--> Proceed to Cell 2 to configure your project path.")

# ==============================================================================
# Cell 2: Define and Verify Project Path
# ==============================================================================
# @title Define Project Path
# @markdown Set your project path within Google Drive.

# --- IMPORTANT: Define Your Project Path on Google Drive ---
drive_base_path_str = '/content/drive/MyDrive/Colab_Notebooks/AHGD_Project'  # <<<-------- CHANGE THIS AS NEEDED
# -----------------------------------------------------------

DRIVE_PROJECT_PATH = Path(drive_base_path_str)
print(f"Target Project Path set to: {DRIVE_PROJECT_PATH}")

print("\nVerifying project path and creating if necessary...")
try:
    # Create the directory and intermediate parents if they don't exist
    DRIVE_PROJECT_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Project base directory ensured at: {DRIVE_PROJECT_PATH}")

    # --- Crucial Check: Writability ---
    test_file_path = DRIVE_PROJECT_PATH / ".writable_test"
    try:
        with open(test_file_path, 'w') as f:
            f.write('test')
        test_file_path.unlink()  # Clean up test file
        print(f"Successfully wrote and deleted test file in {DRIVE_PROJECT_PATH}. Path is writable.")
        # Define BASE_DIR globally for other cells
        global BASE_DIR
        BASE_DIR = DRIVE_PROJECT_PATH
        print("--> Project path verified and seems ready. Proceed to Cell 3.")
    except Exception as write_error:
        print(f"ERROR: Failed write test in {DRIVE_PROJECT_PATH}: {write_error}")
        print("The project directory might exist but is not writable by Colab.")
        raise SystemExit(f"Cannot write to project directory: {DRIVE_PROJECT_PATH}. Stopping.")

except Exception as e:
    print(f"ERROR: Could not create or access base directory {DRIVE_PROJECT_PATH}: {e}")
    raise SystemExit(f"Failed to ensure project directory: {DRIVE_PROJECT_PATH}. Stopping.")

# ==============================================================================
# Cell 3: Install Required Packages
# ==============================================================================
# @title Install Dependencies
# @markdown Install required Python packages.

print("Installing required packages...")
!pip install pandas geopandas polars requests tqdm pyarrow shapely openpyxl --quiet

print("Verification: Attempting imports...")
try:
    import pandas as pd
    import geopandas as gpd
    import polars as pl
    import requests
    import tqdm
    import pyarrow
    import shapely
    import openpyxl
    print("Core libraries imported successfully.")
except ImportError as e:
    print(f"\nERROR: Failed to import a library after installation: {e}")
    raise SystemExit("Package installation/import failed.")

# ==============================================================================
# Cell 4: Configuration and Utility Functions
# ==============================================================================
# @title Configuration and Utilities
# @markdown Define configuration settings and utility functions.

import logging
import zipfile
import requests
import tempfile
import shutil
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from shapely.geometry import mapping
from shapely.validation import make_valid
from tqdm.notebook import tqdm

# --- Configuration Settings ---
RELATIVE_PATHS = {
    'DATA_DIR': 'data',
    'RAW_DATA_DIR': 'data/raw',
    'OUTPUT_DIR': 'output',
    'TEMP_DIR': 'data/raw/temp',
    'LOG_DIR': 'logs',
    'GEOGRAPHIC_DIR': 'data/raw/geographic',
    'CENSUS_DIR': 'data/raw/census',
    'TEMP_ZIP_DIR': 'data/raw/temp/zips',
    'TEMP_EXTRACT_DIR': 'data/raw/temp/extract'
}

# Data source URLs
DATA_URLS = {
    'SA1_2021_AUST_GDA2020': 'https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA1_2021_AUST_GDA2020_SHP.zip',
    'SA2_2021_AUST_GDA2020': 'https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_GDA2020_SHP.zip',
    'SA3_2021_AUST_GDA2020': 'https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA3_2021_AUST_GDA2020_SHP.zip',
    'SA4_2021_AUST_GDA2020': 'https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA4_2021_AUST_GDA2020_SHP.zip',
    'STE_2021_AUST_GDA2020': 'https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/STE_2021_AUST_GDA2020_SHP.zip',
    'POA_2021_AUST_GDA2020': 'https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/POA_2021_AUST_GDA2020_SHP.zip',
    'CENSUS_GCP_AUS_2021': 'https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_ALL_for_AUS.zip'
}

# Geographic levels to process
GEO_LEVELS_SHP_PROCESS = {
    'SA1': 'SA1_2021_AUST_GDA2020',
    'SA2': 'SA2_2021_AUST_GDA2020',
    'SA3': 'SA3_2021_AUST_GDA2020',
    'SA4': 'SA4_2021_AUST_GDA2020',
    'STATE': 'STE_2021_AUST_GDA2020'
}

# Geographic levels for Census data processing
GEO_LEVELS_CENSUS_PROCESS = ['SA1', 'SA2']

# Census table patterns
CENSUS_TABLE_PATTERNS = {
    "G01": r"2021\s*Census_G01[_\s].*?(" + "|".join(GEO_LEVELS_CENSUS_PROCESS) + r")\.csv$"
}

# --- Utility Functions ---
def setup_logging(log_directory: Path = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger('ahgd_etl')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_directory:
        log_directory.mkdir(parents=True, exist_ok=True)
        log_file = log_directory / 'ahgd_colab_run.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def download_file(url: str, dest_file: Path, desc: str = None, max_retries: int = 3) -> bool:
    """Download a file from a URL with progress bar and retries."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                with open(dest_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            logging.error(f"Failed to download {url}: {str(e)}")
            return False
    return False

def extract_zipfile(zip_file: Path, extract_dir: Path, desc: str = None) -> bool:
    """Extract a ZIP file with progress indication."""
    try:
        with zipfile.ZipFile(zip_file) as zf:
            for member in tqdm(zf.infolist(), desc=desc):
                try:
                    zf.extract(member, extract_dir)
                except Exception as e:
                    logging.error(f"Failed to extract {member.filename}: {str(e)}")
                    return False
        return True
    except Exception as e:
        logging.error(f"Failed to open/process zip file {zip_file}: {str(e)}")
        return False

def download_data(urls_dict: Dict[str, str], download_dir: Path, force_download: bool = False) -> bool:
    """Download multiple files from URLs."""
    download_dir.mkdir(parents=True, exist_ok=True)
    success = True
    
    for filename, url in urls_dict.items():
        dest_file = download_dir / filename
        if dest_file.exists() and not force_download:
            logging.info(f"File {filename} already exists, skipping download.")
            continue
            
        logging.info(f"Downloading {filename} from {url}")
        if not download_file(url, dest_file, desc=f"Downloading {filename}"):
            success = False
            
    return success

def find_geo_column(df: Union[gpd.GeoDataFrame, pd.DataFrame, pl.DataFrame, pl.LazyFrame, Dict[str, Any]],
                    possible_names: List[str]) -> Optional[str]:
    """Find geographic code column in a dataframe."""
    if isinstance(df, (pd.DataFrame, gpd.GeoDataFrame)):
        cols = df.columns
    elif isinstance(df, pl.DataFrame):
        cols = df.columns
    elif isinstance(df, pl.LazyFrame):
        cols = df.columns
    elif isinstance(df, dict):
        cols = df.keys()
    else:
        return None
        
    for name in possible_names:
        if name in cols:
            return name
    return None

def clean_geo_code(code_val: Any) -> Optional[str]:
    """Clean and validate geographic code values."""
    if pd.isna(code_val):
        return None
    try:
        return str(int(float(str(code_val).strip())))
    except (ValueError, TypeError):
        return None

def geometry_to_wkt(geometry: Any) -> Optional[str]:
    """Convert geometry to WKT string."""
    if pd.isna(geometry):
        return None
    try:
        if hasattr(geometry, 'wkt'):
            return geometry.wkt
        elif hasattr(geometry, '__geo_interface__'):
            return mapping(geometry)['wkt']
        return str(geometry)
    except Exception:
        return None

def clean_polars_geo_code(series_expr: pl.Expr) -> pl.Expr:
    """Clean geographic codes in a Polars expression."""
    return (series_expr.cast(pl.Float64)
            .cast(pl.Int64)
            .cast(pl.Utf8))

def safe_polars_int(series_expr: pl.Expr) -> pl.Expr:
    """Safely convert Polars expression to integer."""
    return (series_expr.cast(pl.Float64)
            .cast(pl.Int64))

# Initialize logging
logger = setup_logging(BASE_DIR / RELATIVE_PATHS['LOG_DIR'])

# ==============================================================================
# Cell 5: Geographic and Census Processing Functions
# ==============================================================================
# @title Processing Functions
# @markdown Define core geographic and census data processing functions.

def process_geography(zip_dir: Path, temp_extract_base: Path, output_dir: Path) -> bool:
    """Process geographic boundary files from ABS ASGS."""
    logger.info("=== Starting Geographic Data Processing ===")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    success = True
    all_geo_data = []
    
    for level_name, prefix in GEO_LEVELS_SHP_PROCESS.items():
        logger.info(f"Processing {level_name} boundaries...")
        
        zip_filename = f"{prefix}_SHP.zip"
        zip_path = zip_dir / zip_filename
        
        if not zip_path.exists():
            logger.error(f"ZIP file not found: {zip_path}")
            success = False
            continue
            
        extract_dir = temp_extract_base / level_name
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        if not extract_zipfile(zip_path, extract_dir, desc=f"Extracting {level_name} ZIP"):
            logger.error(f"Failed to extract {zip_filename}")
            success = False
            continue
            
        try:
            shp_files = list(extract_dir.glob("*.shp"))
            if not shp_files:
                logger.error(f"No shapefile found in {extract_dir}")
                success = False
                continue
                
            gdf = gpd.read_file(shp_files[0])
            
            possible_names = [f"{level_name}_CODE_2021", f"{level_name}_CODE21"]
            geo_col = find_geo_column(gdf, possible_names)
            
            if not geo_col:
                logger.error(f"Could not find geographic code column for {level_name}")
                success = False
                continue
                
            gdf['geometry'] = gdf['geometry'].apply(lambda g: make_valid(g) if g else None)
            gdf = gdf.dropna(subset=['geometry'])
            
            gdf['geometry_wkt'] = gdf['geometry'].apply(geometry_to_wkt)
            
            df = pd.DataFrame({
                'geo_code': gdf[geo_col].apply(clean_geo_code),
                'geo_level': level_name,
                'geometry': gdf['geometry_wkt']
            })
            
            df = df.dropna()
            pl_df = pl.from_pandas(df)
            all_geo_data.append(pl_df)
            
            logger.info(f"Successfully processed {len(pl_df)} {level_name} boundaries")
            
        except Exception as e:
            logger.error(f"Error processing {level_name}: {str(e)}")
            success = False
            continue
            
    if not all_geo_data:
        logger.error("No geographic data was successfully processed")
        return False
        
    try:
        logger.info("Combining all geographic levels...")
        combined_df = pl.concat(all_geo_data)
        
        output_file = output_dir / "geo_dimension.parquet"
        combined_df.write_parquet(output_file)
        logger.info(f"Successfully wrote combined geographic data to {output_file}")
        
    except Exception as e:
        logger.error(f"Error combining/writing geographic data: {str(e)}")
        return False
        
    return success

def find_census_files(zip_dir: Path, pattern: str) -> List[Path]:
    """Find Census CSV files in ZIP files matching a pattern."""
    matching_files = []
    for zip_path in zip_dir.glob("*.zip"):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for name in zf.namelist():
                    if re.search(pattern, name, re.IGNORECASE):
                        matching_files.append(zip_path)
                        break
        except Exception as e:
            logger.error(f"Error scanning {zip_path}: {str(e)}")
    return matching_files

def extract_census_files(zip_file: Path, pattern: str, extract_dir: Path) -> List[Path]:
    """Extract Census CSV files from a ZIP file."""
    extracted_files = []
    try:
        with zipfile.ZipFile(zip_file) as zf:
            for name in zf.namelist():
                if re.search(pattern, name, re.IGNORECASE):
                    try:
                        zf.extract(name, extract_dir)
                        extracted_files.append(extract_dir / name)
                    except Exception as e:
                        logger.error(f"Error extracting {name}: {str(e)}")
    except Exception as e:
        logger.error(f"Error opening {zip_file}: {str(e)}")
    return extracted_files

def process_g01_file(csv_file: Path) -> Optional[pl.DataFrame]:
    """Process a G01 Census CSV file."""
    try:
        df = pl.read_csv(csv_file)
        
        geo_col = find_geo_column(df, ['region_id', 'SA1_CODE21', 'SA2_CODE21'])
        if not geo_col:
            logger.error(f"No geographic code column found in {csv_file}")
            return None
            
        selected_cols = {
            geo_col: 'geo_code',
            'Tot_P_P': 'total_persons',
            'Tot_M_P': 'total_male',
            'Tot_F_P': 'total_female',
            'Indigenous_P': 'total_indigenous'
        }
        
        missing_cols = [col for col in selected_cols.keys() if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns in {csv_file}: {missing_cols}")
            return None
            
        df = df.select([
            pl.col(old).alias(new)
            for old, new in selected_cols.items()
        ])
        
        df = df.with_columns([
            clean_polars_geo_code(pl.col('geo_code')).alias('geo_code')
        ])
        
        for col in ['total_persons', 'total_male', 'total_female', 'total_indigenous']:
            df = df.with_columns([
                safe_polars_int(pl.col(col)).alias(col)
            ])
            
        df = df.drop_nulls()
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing {csv_file}: {str(e)}")
        return None

def process_census_data(zip_dir: Path, temp_extract_base: Path, output_dir: Path,
                       geo_output_path: Path) -> bool:
    """Process Census G01 data and link with geographic boundaries."""
    logger.info("=== Starting Census Data Processing ===")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    g01_pattern = CENSUS_TABLE_PATTERNS['G01']
    g01_zips = find_census_files(zip_dir, g01_pattern)
    
    if not g01_zips:
        logger.error("No ZIP files containing G01 data found")
        return False
        
    extract_dir = temp_extract_base / "census"
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    all_census_data = []
    success = True
    
    for zip_file in g01_zips:
        logger.info(f"Processing {zip_file.name}...")
        
        csv_files = extract_census_files(zip_file, g01_pattern, extract_dir)
        
        if not csv_files:
            logger.error(f"No G01 CSV files found in {zip_file}")
            success = False
            continue
            
        for csv_file in csv_files:
            logger.info(f"Processing {csv_file.name}...")
            
            df = process_g01_file(csv_file)
            if df is not None:
                all_census_data.append(df)
                logger.info(f"Successfully processed {len(df)} rows from {csv_file.name}")
            else:
                success = False
                
    if not all_census_data:
        logger.error("No census data was successfully processed")
        return False
        
    try:
        logger.info("Combining all census data...")
        combined_df = pl.concat(all_census_data)
        
        logger.info("Loading geographic boundaries for validation...")
        geo_df = pl.read_parquet(geo_output_path)
        
        logger.info("Validating against geographic boundaries...")
        validated_df = combined_df.join(
            geo_df.select(['geo_code']).unique(),
            on='geo_code',
            how='inner'
        )
        
        output_file = output_dir / "population_dimension.parquet"
        validated_df.write_parquet(output_file)
        logger.info(f"Successfully wrote population data to {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error combining/writing census data: {str(e)}")
        return False

# ==============================================================================
# Cell 6: Main Pipeline Execution
# ==============================================================================
# @title Run ETL Pipeline
# @markdown Execute the complete ETL pipeline.

def run_pipeline(base_dir: Path, force_download: bool = False,
                force_continue: bool = False, cleanup_temp: bool = False) -> bool:
    """Run the complete ETL pipeline."""
    logger.info("================= Pipeline Start =================")
    start_time = time.time()
    overall_status = True
    
    # Get paths
    paths = {
        name: base_dir / path
        for name, path in RELATIVE_PATHS.items()
    }
    
    # Create directories
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    # --- Step 1: Download ---
    logger.info("=== Download Phase Started ===")
    
    # Generate required ZIP URLs
    required_geo_zips = {
        f"{prefix}_SHP.zip": DATA_URLS[prefix]
        for level, prefix in GEO_LEVELS_SHP_PROCESS.items()
        if prefix in DATA_URLS
    }
    required_census_zips = {'2021_GCP_ALL_for_AUS.zip': DATA_URLS['CENSUS_GCP_AUS_2021']}
    all_zips_to_download = {**required_geo_zips, **required_census_zips}
    
    download_success = download_data(
        urls_dict=all_zips_to_download,
        download_dir=paths['TEMP_ZIP_DIR'],
        force_download=force_download
    )
    
    if not download_success:
        logger.error("=== Download Phase Failed ===")
        overall_status = False
        if not force_continue:
            logger.critical("Exiting due to download error.")
            return False
    else:
        logger.info("=== Download Phase Finished Successfully ===")
    
    # --- Step 2: ETL Processing ---
    if overall_status or force_continue:
        logger.info("=== ETL Phase Started ===")
        etl_success_current = True
        
        # Process Geography
        logger.info("--- Starting Geography ETL Step ---")
        geo_success = process_geography(
            zip_dir=paths['TEMP_ZIP_DIR'],
            temp_extract_base=paths['TEMP_EXTRACT_DIR'],
            output_dir=paths['OUTPUT_DIR']
        )
        if not geo_success:
            logger.error("--- Geography ETL Step Failed ---")
            etl_success_current = False
            if not force_continue:
                overall_status = False
        else:
            logger.info("--- Geography ETL Step Finished Successfully ---")
        
        # Process Population (only if Geo succeeded or force_continue)
        geo_output_path = paths['OUTPUT_DIR'] / "geo_dimension.parquet"
        if (geo_success or force_continue) and overall_status:
            if not geo_success:
                logger.warning("Forcing Population ETL despite Geography failure.")
            logger.info("--- Starting Population (G01) ETL Step ---")
            pop_success = process_census_data(
                zip_dir=paths['TEMP_ZIP_DIR'],
                temp_extract_base=paths['TEMP_EXTRACT_DIR'],
                output_dir=paths['OUTPUT_DIR'],
                geo_output_path=geo_output_path
            )
            if not pop_success:
                logger.error("--- Population (G01) ETL Step Failed ---")
                etl_success_current = False
                if not force_continue:
                    overall_status = False
            else:
                logger.info("--- Population (G01) ETL Step Finished Successfully ---")
        elif not overall_status:
            logger.warning("Skipping Population ETL due to previous critical failure.")
        
        if etl_success_current:
            logger.info("=== ETL Phase Finished Successfully ===")
        else:
            logger.error("=== ETL Phase Finished with Errors ===")
        overall_status = overall_status and etl_success_current
        
    else:
        logger.warning("Skipping ETL Phase due to Download Phase failure.")
    
    # --- Step 3: Cleanup (Optional) ---
    if cleanup_temp:
        logger.info("=== Cleanup Phase Started ===")
        try:
            if paths['TEMP_EXTRACT_DIR'].exists():
                shutil.rmtree(paths['TEMP_EXTRACT_DIR'])
                logger.info(f"Removed temp extract directory: {paths['TEMP_EXTRACT_DIR']}")
            if paths['TEMP_ZIP_DIR'].exists():
                shutil.rmtree(paths['TEMP_ZIP_DIR'])
                logger.info(f"Removed temp zip directory: {paths['TEMP_ZIP_DIR']}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    else:
        logger.info("Skipping cleanup phase.")
    
    # --- Final Report ---
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"================= Pipeline End (Duration: {duration:.2f}s) =================")
    print("\n" + "="*70)
    if overall_status:
        logger.info(f"Pipeline completed successfully in {duration:.2f} seconds.")
        print(f"✅ Pipeline finished successfully in {duration:.2f} seconds.")
        print(f"   Output Directory: {paths['OUTPUT_DIR']}")
        print(f"   Log File: {paths['LOG_DIR'] / 'ahgd_colab_run.log'}")
    else:
        logger.error(f"Pipeline completed with errors in {duration:.2f} seconds.")
        print(f"❌ Pipeline finished with errors in {duration:.2f} seconds.")
        print(f"   Please check the log file for details: {paths['LOG_DIR'] / 'ahgd_colab_run.log'}")
        print(f"   Also review the console output above for critical errors.")
    print("="*70 + "\n")
    
    return overall_status

# ==============================================================================
# Cell 7: Execute Pipeline
# ==============================================================================
# @title Run Pipeline
# @markdown Execute the ETL pipeline with the specified options.

# @markdown ---
# @markdown ### Pipeline Options
# @markdown Configure how the pipeline should run:
force_download = False  # @param {type:"boolean"}
force_continue = False  # @param {type:"boolean"}
cleanup_temp = False  # @param {type:"boolean"}

print("Starting AHGD ETL Pipeline...")
success = run_pipeline(
    base_dir=BASE_DIR,
    force_download=force_download,
    force_continue=force_continue,
    cleanup_temp=cleanup_temp
)

if success:
    print("\nPipeline completed successfully! 🎉")
    print(f"Check {BASE_DIR}/output for the generated Parquet files:")
    print("  - geo_dimension.parquet")
    print("  - population_dimension.parquet")
else:
    print("\nPipeline completed with errors. 😕")
    print(f"Please check the log file at {BASE_DIR}/logs/ahgd_colab_run.log for details.") 