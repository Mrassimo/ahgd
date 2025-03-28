# -*- coding: utf-8 -*-
"""AHGD ETL Pipeline - Colab Runner

This notebook runs the ETL pipeline by pulling code from GitHub and executing it in Colab.
"""

# ==============================================================================
# Cell 1: Mount Google Drive
# ==============================================================================
# @title Mount Google Drive
# @markdown Mount your Google Drive to access data and save outputs.

from google.colab import drive
import os
from pathlib import Path

drive_mount_point = '/content/drive'
print(f"Attempting to mount Google Drive at {drive_mount_point}...")
try:
    drive.mount(drive_mount_point, force_remount=True)
    print("Google Drive mounted successfully.")
    # Basic check
    if not Path(drive_mount_point + '/MyDrive').exists():
         print("Warning: /content/drive/MyDrive not immediately found.")
except Exception as e:
    print(f"ERROR mounting Google Drive: {e}")
    raise SystemExit("Google Drive mount failed.")

# ==============================================================================
# Cell 2: Define and Verify Project Base Path
# ==============================================================================
# @title Define Project Base Path
from pathlib import Path
import os

# --- Set the path where your project's code AND data will live on Drive ---
drive_base_path_str = '/content/drive/MyDrive/Colab_Notebooks/AHGD_Project' # <<<--- VERIFY THIS IS CORRECT
# -------------------------------------------------------------------------

BASE_DIR = Path(drive_base_path_str)
print(f"Target Project Base Directory set to: {BASE_DIR}")
try:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    test_file_path = BASE_DIR / ".writable_test"
    test_file_path.write_text('test'); test_file_path.unlink()
    print(f"Path '{BASE_DIR}' exists and is writable.")
except Exception as e:
    print(f"ERROR: Project path '{BASE_DIR}' is not accessible/writable: {e}")
    raise SystemExit("Project path verification failed.")

# ==============================================================================
# Cell 3: Clone/Pull Code Repo & Install Dependencies
# ==============================================================================
# @title Setup Code and Environment

import os
import sys
from pathlib import Path

# --- Git Configuration ---
GIT_REPO_URL = "https://github.com/Mrassimo/ahgd.git"  # Your GitHub repository URL
PROJECT_DIR_NAME = "ahgd" # Changed from ahgd-etl-pipeline to match actual repo name
# ---

if 'BASE_DIR' not in globals(): raise NameError("BASE_DIR not defined. Run Cell 2 first.")
PROJECT_PATH = BASE_DIR / PROJECT_DIR_NAME

# --- Clone or Pull Repository ---
print("Accessing Git repository...")
if not PROJECT_PATH.exists():
    print(f"Cloning repository into {PROJECT_PATH}...")
    "%cd {BASE_DIR}"
    "!git clone {GIT_REPO_URL} {PROJECT_DIR_NAME}"
    "%cd {PROJECT_PATH}"
else:
    print(f"Pulling latest changes into {PROJECT_PATH}...")
    "%cd {PROJECT_PATH}"
    "!git pull origin main"

# --- Install Dependencies ---
print("\nInstalling dependencies...")
req_file = PROJECT_PATH / 'requirements.txt'
if req_file.exists():
     "!pip install --quiet -r {req_file}"
     print("Installed base requirements.")
else: print("Warning: requirements.txt not found.")
"!apt-get update --quiet && apt-get install --quiet -y libspatialindex-dev python3-rtree > /dev/null"
"!pip install --quiet rtree"
print("Geospatial dependencies installed/checked.")

# --- Add project path to sys.path ---
if str(PROJECT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_PATH))
    print(f"Added {PROJECT_PATH} to sys.path")

print("\nSetup complete.")

# ==============================================================================
# Cell 4: Import Logic and Configure Execution
# ==============================================================================
# @title Import Modules and Setup
from pathlib import Path
import os

if 'BASE_DIR' not in globals(): raise NameError("BASE_DIR not defined. Run Cell 2 first.")
if 'PROJECT_PATH' not in globals() or str(PROJECT_PATH) not in sys.path:
     raise NameError("Project path not set up. Run Cell 3 first.")

# Import your custom modules
try:
    from etl_logic import utils, config, geography, census
    print("Successfully imported ETL modules.")
except ImportError as e:
     print(f"ERROR: Could not import modules from {PROJECT_PATH}/etl_logic.")
     raise e

# --- Setup Logging ---
PATHS = config.get_paths(BASE_DIR) # Get absolute paths
logger = utils.setup_logging(log_directory=PATHS['LOG_DIR'])
if not logger: raise RuntimeError("Failed to initialise logger.")
logger.info("Logger initialised for Colab execution.")

# --- Get Configuration for Pipeline ---
required_geo_zips = config.get_required_geo_zips()
required_census_zips = config.get_required_census_zips()
all_zips_to_download = {**required_geo_zips, **required_census_zips}

# Display paths for verification
logger.info(f"Output Dir: {PATHS['OUTPUT_DIR']}")
logger.info(f"Temp Zip Dir: {PATHS['TEMP_ZIP_DIR']}")
logger.info(f"Temp Extract Dir: {PATHS['TEMP_EXTRACT_DIR']}")
logger.info(f"Log File: {PATHS['LOG_DIR'] / 'ahgd_colab_run.log'}")

print("Imports and configuration loaded.")

# ==============================================================================
# Cell 5: Execute the Pipeline
# ==============================================================================
# @title Run ETL Pipeline
# @markdown Execute the complete ETL pipeline using the imported logic.

import time
import shutil

if 'logger' not in globals() or 'PATHS' not in globals():
    raise NameError("Setup incomplete. Run Cell 4 first.")

# --- Pipeline Options ---
force_download = False  # @param {type:"boolean"}
force_continue = False  # @param {type:"boolean"}
cleanup_temp = False    # @param {type:"boolean"}
# ----------------------

logger.info("================= Pipeline Start =================")
start_time = time.time()
overall_status = True

# --- Step 1: Download ---
logger.info("=== Download Phase Started ===")
download_success = utils.download_data(
    urls_dict=all_zips_to_download,
    download_dir=PATHS['TEMP_ZIP_DIR'],
    force_download=force_download
)
if not download_success:
    logger.error("=== Download Phase Failed ===")
    overall_status = False
    if not force_continue: logger.critical("Exiting: Download failed & force_continue=False.")
else: logger.info("=== Download Phase Finished Successfully ===")

# --- Step 2: ETL Processing ---
if overall_status or force_continue:
    if not overall_status: logger.warning("Proceeding with ETL despite Download failure.")
    logger.info("=== ETL Phase Started ===")
    etl_success_current = True

    # Process Geography
    if overall_status or force_continue:
        logger.info("--- Starting Geography ETL Step ---")
        geo_success = geography.process_geography(
            zip_dir=PATHS['TEMP_ZIP_DIR'],
            temp_extract_base=PATHS['TEMP_EXTRACT_DIR'],
            output_dir=PATHS['OUTPUT_DIR']
        )
        if not geo_success:
            logger.error("--- Geography ETL Step Failed ---")
            etl_success_current = False
            if not force_continue: overall_status = False
        else: logger.info("--- Geography ETL Step Finished Successfully ---")

    # Process Population
    geo_output_path = PATHS['OUTPUT_DIR'] / "geo_dimension.parquet"
    if (geo_success or force_continue) and overall_status:
         if not geo_success: logger.warning("Forcing Population ETL despite Geo failure.")
         logger.info("--- Starting Population (G01) ETL Step ---")
         pop_success = census.process_census_data(
              zip_dir=PATHS['TEMP_ZIP_DIR'],
              temp_extract_base=PATHS['TEMP_EXTRACT_DIR'],
              output_dir=PATHS['OUTPUT_DIR'],
              geo_output_path=geo_output_path
         )
         if not pop_success:
              logger.error("--- Population (G01) ETL Step Failed ---")
              etl_success_current = False
              if not force_continue: overall_status = False
         else: logger.info("--- Population (G01) ETL Step Finished Successfully ---")
    elif not overall_status: logger.warning("Skipping Population ETL: Critical failure earlier.")

    if etl_success_current: logger.info("=== ETL Phase Finished Successfully ===")
    else: logger.error("=== ETL Phase Finished with Errors ===")
    overall_status = overall_status and etl_success_current
else: logger.warning("Skipping ETL Phase due to Download failure.")

# --- Step 3: Cleanup ---
if cleanup_temp:
     logger.info("=== Cleanup Phase Started ===")
     try:
         shutil.rmtree(PATHS['TEMP_ZIP_DIR'], ignore_errors=True)
         shutil.rmtree(PATHS['TEMP_EXTRACT_DIR'], ignore_errors=True)
         logger.info("Temporary directories cleaned up.")
     except Exception as e:
         logger.error(f"Cleanup failed: {e}")
else: logger.info("Skipping cleanup phase.")

# --- Final Report ---
end_time = time.time(); duration = end_time - start_time
logger.info(f"================= Pipeline End (Duration: {duration:.2f}s) =================")
log_file_path_final = PATHS['LOG_DIR'] / 'ahgd_colab_run.log'
print("\n" + "="*70)
if overall_status:
    logger.info(f"Pipeline success ({duration:.2f}s).")
    print(f"✅ Success ({duration:.2f}s).")
    print(f"   Output: {PATHS['OUTPUT_DIR']}")
    print(f"   Log: {log_file_path_final}")
else:
    logger.error(f"Pipeline failed ({duration:.2f}s).")
    print(f"❌ Failed ({duration:.2f}s).")
    print(f"   Check Log: {log_file_path_final}")
    print("   Review Colab output above.")
print("="*70 + "\n")
print("Colab execution run complete.") 