"""
Data Downloader Module

Handles downloading of Census and geographic data from ABS.
"""

import logging
import requests
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import zipfile
import time

from ahgd_etl.config import settings


class DataDownloader:
    """Downloads and manages external data sources."""
    
    def __init__(self, force_download: bool = False):
        self.force_download = force_download
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AHGD ETL Pipeline/1.0'
        })
        
    def download_all(self) -> bool:
        """Download all required data files."""
        success = True
        
        # Get data sources from configuration
        data_sources = settings.data_sources
        
        # Download geographic data
        self.logger.info("Downloading geographic data...")
        geo_sources = data_sources.get("geographic", {})
        for name, source in geo_sources.items():
            if not self._download_file(source["url"], source["filename"], settings.paths.geographic_dir):
                success = False
                
        # Download census data
        self.logger.info("Downloading census data...")
        census_sources = data_sources.get("census", {})
        for name, source in census_sources.items():
            if not self._download_file(source["url"], source["filename"], settings.paths.census_dir):
                success = False
                
        return success
    
    def _download_file(self, url: str, filename: str, target_dir: Path) -> bool:
        """
        Download a single file with progress bar.
        
        Args:
            url: URL to download from
            filename: Target filename
            target_dir: Target directory
            
        Returns:
            bool: True if successful
        """
        target_path = target_dir / filename
        
        # Check if file exists and force_download is False
        if target_path.exists() and not self.force_download:
            self.logger.info(f"File {filename} already exists, skipping download")
            return True
            
        # Create directory if needed
        target_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.logger.info(f"Downloading {filename} from {url}")
            
            # Get file with streaming
            response = self.session.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            # Get total file size
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            with open(target_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                            
            self.logger.info(f"Successfully downloaded {filename}")
            
            # Extract if it's a zip file
            if filename.endswith('.zip'):
                self._extract_zip(target_path, target_dir)
                
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to download {filename}: {str(e)}")
            # Clean up partial download
            if target_path.exists():
                target_path.unlink()
            return False
            
        except Exception as e:
            self.logger.error(f"Unexpected error downloading {filename}: {str(e)}", exc_info=True)
            return False
    
    def _extract_zip(self, zip_path: Path, extract_dir: Path) -> bool:
        """
        Extract a zip file.
        
        Args:
            zip_path: Path to zip file
            extract_dir: Directory to extract to
            
        Returns:
            bool: True if successful
        """
        try:
            self.logger.info(f"Extracting {zip_path.name}...")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract all files
                zip_ref.extractall(extract_dir)
                
            self.logger.info(f"Successfully extracted {zip_path.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to extract {zip_path.name}: {str(e)}", exc_info=True)
            return False
    
    def verify_downloads(self) -> Dict[str, bool]:
        """
        Verify that all required files have been downloaded.
        
        Returns:
            Dict mapping filename to existence status
        """
        verification = {}
        
        # Check geographic files
        geo_sources = settings.data_sources.get("geographic", {})
        for name, source in geo_sources.items():
            file_path = settings.paths.geographic_dir / source["filename"]
            verification[source["filename"]] = file_path.exists()
            
        # Check census files
        census_sources = settings.data_sources.get("census", {})
        for name, source in census_sources.items():
            file_path = settings.paths.census_dir / source["filename"]
            verification[source["filename"]] = file_path.exists()
            
        # Log summary
        missing = [f for f, exists in verification.items() if not exists]
        if missing:
            self.logger.warning(f"Missing files: {missing}")
        else:
            self.logger.info("All required files are present")
            
        return verification