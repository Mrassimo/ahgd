"""Data downloader module for ABS geographic and census data."""

import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from tqdm import tqdm
import time
import hashlib

from ..config import get_settings
from ..utils import get_logger


class DataDownloader:
    """Downloads and extracts ABS geographic and census data files."""
    
    def __init__(self, force_download: bool = False):
        """Initialize the data downloader.
        
        Args:
            force_download: If True, re-download files even if they exist
        """
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self.force_download = force_download or self.settings.force_download
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AHGD-ETL/1.0 (Australian Healthcare Geographic Database ETL)'
        })
    
    def download_all(self) -> Dict[str, List[Path]]:
        """Download all configured geographic and census data files.
        
        Returns:
            Dictionary mapping data types to lists of extracted file paths
        """
        self.logger.info("Starting data download process")
        
        results = {
            "geographic": [],
            "census": []
        }
        
        # Download geographic data
        self.logger.info("Downloading geographic boundary files...")
        for geo_level in self.settings.geographic_sources.keys():
            try:
                extracted_files = self.download_geographic_data(geo_level)
                results["geographic"].extend(extracted_files)
            except Exception as e:
                self.logger.error(f"Failed to download {geo_level} data: {e}")
                if self.settings.stop_on_error:
                    raise
        
        # Download census data
        self.logger.info("Downloading census data pack...")
        try:
            extracted_files = self.download_census_data()
            results["census"].extend(extracted_files)
        except Exception as e:
            self.logger.error(f"Failed to download census data: {e}")
            if self.settings.stop_on_error:
                raise
        
        self.logger.info(f"Download complete. Geographic files: {len(results['geographic'])}, "
                        f"Census files: {len(results['census'])}")
        
        return results
    
    def download_geographic_data(self, level: str) -> List[Path]:
        """Download and extract geographic boundary files for a specific level.
        
        Args:
            level: Geographic level (e.g., 'sa1', 'sa2')
            
        Returns:
            List of paths to extracted files
            
        Raises:
            ValueError: If geographic level not configured
            requests.RequestException: If download fails
        """
        url = self.settings.get_geographic_url(level)
        filename = self.settings.get_geographic_filename(level)
        
        if not url or not filename:
            raise ValueError(f"Geographic level '{level}' not configured")
        
        # Check if manual download is required
        if url == "MANUAL_DOWNLOAD_REQUIRED":
            self.logger.warning(f"Manual download required for {level.upper()}")
            self.logger.info(f"Please download {filename} from ABS website and place in:")
            self.logger.info(f"  {self.settings.raw_data_dir / 'geographic' / filename}")
            
            # Check if file exists from manual download
            zip_path = self.settings.raw_data_dir / "geographic" / filename
            if not zip_path.exists():
                self.logger.warning(f"File not found: {zip_path}")
                return []
            else:
                self.logger.info(f"Found manually downloaded file: {filename}")
        else:
            # Download the file
            zip_path = self.settings.raw_data_dir / "geographic" / filename
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self._should_download(zip_path):
                self.logger.info(f"Downloading {level.upper()} boundary file: {filename}")
                self._download_file(url, zip_path)
            else:
                self.logger.info(f"Using existing {level.upper()} file: {filename}")
        
        # Extract files if zip exists
        if zip_path.exists() and zipfile.is_zipfile(zip_path):
            extract_dir = self.settings.raw_data_dir / "geographic" / level
            return self._extract_zip(zip_path, extract_dir)
        else:
            return []
    
    def download_census_data(self) -> List[Path]:
        """Download and extract census data pack.
        
        Returns:
            List of paths to extracted CSV files
            
        Raises:
            ValueError: If census data not configured
            requests.RequestException: If download fails
        """
        url = self.settings.get_census_url()
        filename = self.settings.get_census_filename()
        
        if not url or not filename:
            raise ValueError("Census data source not configured")
        
        # Download the file
        zip_path = self.settings.raw_data_dir / "census" / filename
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self._should_download(zip_path):
            self.logger.info(f"Downloading census data pack: {filename}")
            self.logger.info("Note: This is a large file (~1GB) and may take several minutes")
            self._download_file(url, zip_path)
        else:
            self.logger.info(f"Using existing census file: {filename}")
        
        # Extract only the tables we need
        extract_dir = self.settings.raw_data_dir / "census" / "extracted"
        return self._extract_census_tables(zip_path, extract_dir)
    
    def _should_download(self, file_path: Path) -> bool:
        """Check if a file should be downloaded.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file should be downloaded
        """
        if self.force_download:
            return True
        
        if not file_path.exists():
            return True
        
        # Check if file is complete (has reasonable size)
        if file_path.stat().st_size < 1000:  # Less than 1KB is suspicious
            self.logger.warning(f"File {file_path} appears incomplete, will re-download")
            return True
        
        return False
    
    def _download_file(self, url: str, dest_path: Path, chunk_size: int = 8192) -> None:
        """Download a file with progress bar and retry logic.
        
        Args:
            url: URL to download from
            dest_path: Destination file path
            chunk_size: Download chunk size
            
        Raises:
            requests.RequestException: If download fails after retries
        """
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                # Make request
                response = self.session.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                # Get total size
                total_size = int(response.headers.get('content-length', 0))
                
                # Download with progress bar
                with open(dest_path, 'wb') as f:
                    with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                # Verify download
                if total_size > 0 and dest_path.stat().st_size != total_size:
                    raise ValueError(f"Downloaded file size mismatch: "
                                   f"expected {total_size}, got {dest_path.stat().st_size}")
                
                self.logger.info(f"Successfully downloaded: {dest_path.name}")
                return
                
            except Exception as e:
                self.logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise
    
    def _extract_zip(self, zip_path: Path, extract_dir: Path) -> List[Path]:
        """Extract all files from a ZIP archive.
        
        Args:
            zip_path: Path to ZIP file
            extract_dir: Directory to extract to
            
        Returns:
            List of extracted file paths
        """
        self.logger.info(f"Extracting {zip_path.name} to {extract_dir}")
        
        extracted_files = []
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files
            file_list = zip_ref.namelist()
            
            # Extract all files
            for file_name in tqdm(file_list, desc="Extracting"):
                # Skip directories and system files
                if file_name.endswith('/') or file_name.startswith('__MACOSX'):
                    continue
                
                # Extract file
                zip_ref.extract(file_name, extract_dir)
                extracted_path = extract_dir / file_name
                
                # Only track files we care about
                if extracted_path.suffix.lower() in ['.shp', '.dbf', '.shx', '.prj', '.csv']:
                    extracted_files.append(extracted_path)
        
        self.logger.info(f"Extracted {len(extracted_files)} files")
        return extracted_files
    
    def _extract_census_tables(self, zip_path: Path, extract_dir: Path) -> List[Path]:
        """Extract only required census table CSV files.
        
        Args:
            zip_path: Path to census ZIP file
            extract_dir: Directory to extract to
            
        Returns:
            List of extracted CSV file paths
        """
        self.logger.info(f"Extracting census tables from {zip_path.name}")
        
        extracted_files = []
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        # Get list of tables we need
        required_tables = self.settings.census_tables
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of all files
            all_files = zip_ref.namelist()
            
            # Filter for CSV files matching our table patterns
            csv_files = []
            for table in required_tables:
                pattern = f"2021Census_{table}"
                matching_files = [f for f in all_files 
                                if pattern in f and f.endswith('.csv')]
                csv_files.extend(matching_files)
            
            # Extract matching files
            self.logger.info(f"Found {len(csv_files)} CSV files to extract")
            
            for file_name in tqdm(csv_files, desc="Extracting census tables"):
                try:
                    zip_ref.extract(file_name, extract_dir)
                    extracted_path = extract_dir / file_name
                    extracted_files.append(extracted_path)
                except Exception as e:
                    self.logger.warning(f"Failed to extract {file_name}: {e}")
        
        self.logger.info(f"Extracted {len(extracted_files)} census CSV files")
        return extracted_files
    
    def verify_downloads(self) -> Dict[str, bool]:
        """Verify that all required files have been downloaded.
        
        Returns:
            Dictionary mapping data types to verification status
        """
        self.logger.info("Verifying downloaded files...")
        
        verification = {}
        
        # Check geographic files
        for level in self.settings.geographic_sources.keys():
            geo_dir = self.settings.raw_data_dir / "geographic" / level
            shp_files = list(geo_dir.glob("*.shp")) if geo_dir.exists() else []
            verification[f"geo_{level}"] = len(shp_files) > 0
        
        # Check census files
        census_dir = self.settings.raw_data_dir / "census" / "extracted"
        for table in self.settings.census_tables:
            csv_files = list(census_dir.glob(f"*{table}*.csv")) if census_dir.exists() else []
            verification[f"census_{table}"] = len(csv_files) > 0
        
        # Log results
        all_verified = all(verification.values())
        if all_verified:
            self.logger.info("✅ All required files verified")
        else:
            missing = [k for k, v in verification.items() if not v]
            self.logger.warning(f"❌ Missing files: {missing}")
        
        return verification
    
    def cleanup(self):
        """Clean up resources."""
        self.session.close()