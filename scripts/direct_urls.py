"""
This module contains direct URLs to specific ABS files.
It's maintained separately for easier updating when URLs change.
"""

import hashlib
import logging
import requests
from pathlib import Path
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Main ASGS geographic files
ASGS2021_URLS = {
    "SA1": "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA1_2021_AUST_SHP_GDA2020.zip",
    "SA2": "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_SHP_GDA2020.zip",
    "SA3": "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA3_2021_AUST_SHP_GDA2020.zip",
    "SA4": "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA4_2021_AUST_SHP_GDA2020.zip",
    "STE": "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/STE_2021_AUST_SHP_GDA2020.zip",
    "POA": "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/POA_2021_AUST_GDA2020_SHP.zip"
}

# Correspondence package
CORRESPONDENCE_PACKAGE_URL = "https://data.gov.au/data/dataset/2c79581f-600e-4560-80a8-98adb1922dfc/resource/33d822ba-138e-47ae-a15f-460279c3acc3/download/asgs2021_correspondences.zip"

# Census data packages - Updated for 2021 Census
CENSUS_URLS = {
    # General Community Profile (GCP) for all of Australia
    "GCP_ALL": "https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_all_for_AUS_short-header.zip",
    # Specific area packages
    "GCP_SA1": "https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_SA1_for_AUS_short-header.zip",
    "GCP_POA": "https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_POA_for_AUS_short-header.zip"
}

# Known file checksums (SHA-256) for validation
FILE_CHECKSUMS = {
    # These will be populated as files are downloaded
}

def validate_file(file_path: Path, expected_checksum: Optional[str] = None) -> bool:
    """
    Validate a downloaded file using SHA-256 checksum.
    
    Args:
        file_path: Path to the file to validate
        expected_checksum: Expected SHA-256 checksum (if known)
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False
    
    sha256_hash = hashlib.sha256()
    
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        actual_checksum = sha256_hash.hexdigest()
        
        if expected_checksum is None:
            # If no checksum provided, store it for future validation
            FILE_CHECKSUMS[str(file_path)] = actual_checksum
            logger.info(f"Stored checksum for {file_path}: {actual_checksum}")
            return True
        
        if actual_checksum == expected_checksum:
            logger.info(f"Checksum validation passed for {file_path}")
            return True
        else:
            logger.error(f"Checksum validation failed for {file_path}")
            logger.error(f"Expected: {expected_checksum}")
            logger.error(f"Got: {actual_checksum}")
            return False
            
    except Exception as e:
        logger.error(f"Error validating file {file_path}: {str(e)}")
        return False

def verify_url(url: str) -> bool:
    """
    Verify that a URL is accessible.
    
    Args:
        url: URL to verify
    
    Returns:
        bool: True if URL is accessible, False otherwise
    """
    try:
        response = requests.head(url, allow_redirects=True)
        if response.status_code == 200:
            logger.info(f"URL is accessible: {url}")
            return True
        else:
            logger.error(f"URL returned status code {response.status_code}: {url}")
            return False
    except Exception as e:
        logger.error(f"Error verifying URL {url}: {str(e)}")
        return False

def verify_all_urls() -> Dict[str, bool]:
    """
    Verify all URLs in this module.
    
    Returns:
        Dict[str, bool]: Dictionary mapping URL descriptions to their accessibility status
    """
    results = {}
    
    # Check ASGS URLs
    for area_type, url in ASGS2021_URLS.items():
        results[f"ASGS2021_{area_type}"] = verify_url(url)
    
    # Check correspondence package
    results["CORRESPONDENCE"] = verify_url(CORRESPONDENCE_PACKAGE_URL)
    
    # Check Census URLs
    for package_type, url in CENSUS_URLS.items():
        results[f"CENSUS_{package_type}"] = verify_url(url)
    
    return results