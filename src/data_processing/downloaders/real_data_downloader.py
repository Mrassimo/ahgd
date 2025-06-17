"""
Real Australian data downloader using verified working URLs.

This replaces the original ABSDownloader with actual working downloads
from ABS, data.gov.au, and other Australian government sources.
"""

import asyncio
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import httpx
from loguru import logger
from rich.console import Console
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn

console = Console()

# VERIFIED WORKING DATA SOURCES (tested in test_real_data_sources.py)
VERIFIED_DATA_SOURCES = {
    # ABS Digital Boundaries (validated working)
    "sa2_boundaries_gda2020": {
        "url": "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_SHP_GDA2020.zip",
        "filename": "SA2_2021_AUST_SHP_GDA2020.zip",
        "size_mb": 96,
        "format": "zip",
        "description": "SA2 Statistical Area boundaries 2021 (GDA2020)"
    },
    
    "sa2_boundaries_gda94": {
        "url": "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_SHP_GDA94.zip",
        "filename": "SA2_2021_AUST_SHP_GDA94.zip", 
        "size_mb": 47,
        "format": "zip",
        "description": "SA2 Statistical Area boundaries 2021 (GDA94)"
    },
    
    # SEIFA 2021 Socio-Economic Data (validated working)
    "seifa_2021_sa2": {
        "url": "https://www.abs.gov.au/statistics/people/people-and-communities/socio-economic-indexes-areas-seifa-australia/2021/Statistical%20Area%20Level%202%2C%20Indexes%2C%20SEIFA%202021.xlsx",
        "filename": "SEIFA_2021_SA2_Indexes.xlsx",
        "size_mb": 1.3,
        "format": "xlsx", 
        "description": "SEIFA 2021 socio-economic indexes by SA2"
    },
    
    # Medicare Benefits Schedule (validated working)
    "mbs_historical": {
        "url": "https://data.gov.au/data/dataset/8a19a28f-35b0-4035-8cd5-5b611b3cfa6f/resource/519b55ab-8f81-47d1-a483-8495668e38d8/download/mbs-demographics-historical-1993-2015.zip",
        "filename": "MBS_Demographics_Historical_1993-2015.zip",
        "size_mb": 50,
        "format": "zip",
        "description": "Medicare Benefits Schedule historical demographics data"
    },
    
    # Pharmaceutical Benefits Scheme (validated working)
    "pbs_current": {
        "url": "https://data.gov.au/data/dataset/14b536d4-eb6a-485d-bf87-2e6e77ddbac1/resource/08eda5ab-01c0-4c94-8b1a-157bcffe80d3/download/pbs-item-2016csvjuly.csv",
        "filename": "PBS_Item_Report_2016_Current.csv",
        "size_mb": 10,
        "format": "csv",
        "description": "Pharmaceutical Benefits Scheme current year data"
    },
    
    "pbs_historical": {
        "url": "https://data.gov.au/data/dataset/14b536d4-eb6a-485d-bf87-2e6e77ddbac1/resource/56f87bbb-a7cb-4cbf-a723-7aec22996eee/download/csv-pbs-item-historical-1992-2014.zip",
        "filename": "PBS_Item_Historical_1992-2014.zip",
        "size_mb": 25,
        "format": "zip",
        "description": "Pharmaceutical Benefits Scheme historical data"
    }
}


class RealDataDownloader:
    """
    Production-ready downloader for Australian government data sources.
    
    Uses verified working URLs and proper error handling.
    All data sources have been tested and validated.
    """
    
    def __init__(self, data_dir: Union[str, Path] = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # HTTP client settings for reliable downloads
        self.timeout = httpx.Timeout(300.0, connect=60.0)  # 5 min download, 1 min connect
        self.limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
        
        logger.info(f"Real Data Downloader initialized")
        logger.info(f"Data directory: {self.raw_dir}")
        logger.info(f"Available datasets: {len(VERIFIED_DATA_SOURCES)}")
    
    async def download_file(
        self, 
        session: httpx.AsyncClient,
        dataset_key: str,
        dataset_info: Dict,
        progress: Progress,
        task_id: TaskID,
    ) -> Optional[Path]:
        """
        Download a single dataset with progress tracking and validation.
        """
        file_path = self.raw_dir / dataset_info["filename"]
        
        # Skip if file already exists and is the right size
        if file_path.exists():
            existing_size_mb = file_path.stat().st_size / (1024 * 1024)
            expected_size = dataset_info["size_mb"]
            
            if existing_size_mb > expected_size * 0.8:  # Allow 20% variance
                logger.info(f"✓ File already exists: {dataset_info['filename']}")
                progress.update(task_id, completed=100)
                return file_path
        
        try:
            logger.info(f"Downloading {dataset_key}: {dataset_info['description']}")
            
            async with session.stream('GET', dataset_info["url"]) as response:
                response.raise_for_status()
                
                # Get content length for progress tracking
                total_size = int(response.headers.get('content-length', 0))
                if total_size > 0:
                    progress.update(task_id, total=total_size)
                
                downloaded_size = 0
                with open(file_path, 'wb') as file:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        file.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress.update(task_id, completed=downloaded_size)
            
            # Validate downloaded file
            actual_size_mb = file_path.stat().st_size / (1024 * 1024)
            expected_size = dataset_info["size_mb"]
            
            # Check size is reasonable (allow 3x variance for estimates)
            if actual_size_mb < expected_size * 0.3:
                logger.warning(f"Downloaded file seems too small: {actual_size_mb:.1f}MB (expected ~{expected_size}MB)")
            elif actual_size_mb > expected_size * 3.0:
                logger.warning(f"Downloaded file seems too large: {actual_size_mb:.1f}MB (expected ~{expected_size}MB)")
            
            # Validate file format
            self._validate_file_format(file_path, dataset_info["format"])
            
            logger.info(f"✓ Downloaded: {dataset_info['filename']} ({actual_size_mb:.1f}MB)")
            return file_path
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error downloading {dataset_key}: {e.response.status_code}")
            if file_path.exists():
                file_path.unlink()  # Remove partial download
            return None
            
        except Exception as e:
            logger.error(f"Error downloading {dataset_key}: {str(e)}")
            if file_path.exists():
                file_path.unlink()  # Remove partial download
            return None
    
    def _validate_file_format(self, file_path: Path, expected_format: str):
        """Validate downloaded file format."""
        
        if expected_format == "zip":
            # Check ZIP file magic bytes and can be opened
            with open(file_path, 'rb') as f:
                magic = f.read(4)
                if not magic.startswith(b'PK'):
                    raise ValueError(f"Not a valid ZIP file: {file_path}")
            
            # Test ZIP can be opened
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    if len(file_list) == 0:
                        raise ValueError(f"Empty ZIP file: {file_path}")
            except zipfile.BadZipFile:
                raise ValueError(f"Corrupted ZIP file: {file_path}")
        
        elif expected_format == "xlsx":
            # Check Excel file magic bytes
            with open(file_path, 'rb') as f:
                magic = f.read(4)
                if not magic.startswith(b'PK'):  # Excel files are ZIP-based
                    raise ValueError(f"Not a valid Excel file: {file_path}")
        
        elif expected_format == "csv":
            # Check CSV file is readable text
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    if len(first_line.strip()) == 0:
                        raise ValueError(f"Empty CSV file: {file_path}")
            except UnicodeDecodeError:
                # Try with different encoding
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        first_line = f.readline()
                        if len(first_line.strip()) == 0:
                            raise ValueError(f"Empty CSV file: {file_path}")
                except UnicodeDecodeError:
                    raise ValueError(f"Cannot read CSV file: {file_path}")
    
    async def download_essential_datasets(
        self, 
        datasets: Optional[List[str]] = None
    ) -> Dict[str, Optional[Path]]:
        """
        Download essential datasets for health analytics.
        
        Args:
            datasets: List of dataset keys to download. If None, downloads all.
            
        Returns:
            Dictionary mapping dataset keys to downloaded file paths.
        """
        if datasets is None:
            datasets = list(VERIFIED_DATA_SOURCES.keys())
        
        # Validate dataset keys
        invalid_datasets = set(datasets) - set(VERIFIED_DATA_SOURCES.keys())
        if invalid_datasets:
            raise ValueError(f"Unknown datasets: {invalid_datasets}")
        
        console.print(f"📡 [bold blue]Downloading {len(datasets)} Australian datasets...[/bold blue]")
        console.print()
        
        results = {}
        
        async with httpx.AsyncClient(timeout=self.timeout, limits=self.limits) as session:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                
                # Create download tasks
                download_tasks = []
                
                for dataset_key in datasets:
                    dataset_info = VERIFIED_DATA_SOURCES[dataset_key]
                    task_id = progress.add_task(
                        f"Downloading {dataset_info['filename']}", 
                        total=None
                    )
                    
                    task = self.download_file(
                        session, dataset_key, dataset_info, progress, task_id
                    )
                    download_tasks.append((dataset_key, task))
                
                # Execute all downloads concurrently
                for dataset_key, task in download_tasks:
                    try:
                        result = await task
                        results[dataset_key] = result
                    except Exception as e:
                        logger.error(f"Failed to download {dataset_key}: {e}")
                        results[dataset_key] = None
        
        # Summary
        successful = sum(1 for path in results.values() if path is not None)
        total_size = sum(
            (path.stat().st_size / (1024 * 1024)) 
            for path in results.values() 
            if path is not None
        )
        
        console.print()
        console.print(f"✅ [bold green]Download complete![/bold green]")
        console.print(f"   📁 Downloaded: {successful}/{len(datasets)} files")
        console.print(f"   💾 Total size: {total_size:.1f}MB")
        
        if failed := [key for key, path in results.items() if path is None]:
            console.print(f"   ❌ Failed: {', '.join(failed)}")
        
        return results
    
    async def download_specific_dataset(self, dataset_key: str) -> Optional[Path]:
        """Download a specific dataset by key."""
        if dataset_key not in VERIFIED_DATA_SOURCES:
            raise ValueError(f"Unknown dataset: {dataset_key}")
        
        return await self.download_essential_datasets([dataset_key])
    
    def extract_zip_files(self, extract_dir: Optional[Path] = None) -> Dict[str, List[Path]]:
        """
        Extract all downloaded ZIP files to specified directory.
        
        Args:
            extract_dir: Directory to extract to (default: data/processed)
            
        Returns:
            Dictionary mapping ZIP file names to list of extracted files.
        """
        if extract_dir is None:
            extract_dir = self.data_dir / "processed"
        
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        extracted_files = {}
        
        # Find all ZIP files in raw directory
        zip_files = list(self.raw_dir.glob("*.zip"))
        
        if not zip_files:
            logger.info("No ZIP files found to extract")
            return extracted_files
        
        console.print(f"📦 [bold blue]Extracting {len(zip_files)} ZIP files...[/bold blue]")
        
        for zip_path in zip_files:
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Create subdirectory for this ZIP file
                    zip_extract_dir = extract_dir / zip_path.stem
                    zip_extract_dir.mkdir(exist_ok=True)
                    
                    # Extract all files
                    zip_ref.extractall(zip_extract_dir)
                    
                    # List extracted files
                    file_list = [zip_extract_dir / name for name in zip_ref.namelist()]
                    extracted_files[zip_path.name] = file_list
                    
                    logger.info(f"✓ Extracted {zip_path.name} ({len(file_list)} files)")
                    
            except Exception as e:
                logger.error(f"Failed to extract {zip_path.name}: {e}")
                extracted_files[zip_path.name] = []
        
        console.print(f"✅ Extraction complete!")
        return extracted_files
    
    def list_available_datasets(self) -> None:
        """Display all available datasets with details."""
        console.print("📊 [bold blue]Available Australian datasets:[/bold blue]")
        console.print()
        
        # Group by data source
        abs_datasets = {k: v for k, v in VERIFIED_DATA_SOURCES.items() if "abs.gov.au" in v["url"]}
        datagov_datasets = {k: v for k, v in VERIFIED_DATA_SOURCES.items() if "data.gov.au" in v["url"]}
        
        if abs_datasets:
            console.print("[bold]Australian Bureau of Statistics (ABS):[/bold]")
            for key, info in abs_datasets.items():
                console.print(f"  • {key}: {info['description']} ({info['size_mb']}MB)")
            console.print()
        
        if datagov_datasets:
            console.print("[bold]data.gov.au Health Datasets:[/bold]")
            for key, info in datagov_datasets.items():
                console.print(f"  • {key}: {info['description']} ({info['size_mb']}MB)")
            console.print()
        
        total_size = sum(info["size_mb"] for info in VERIFIED_DATA_SOURCES.values())
        console.print(f"[bold]Total download size: {total_size}MB[/bold]")
    
    def get_downloaded_files(self) -> Dict[str, Path]:
        """Get list of already downloaded files."""
        downloaded = {}
        
        for dataset_key, dataset_info in VERIFIED_DATA_SOURCES.items():
            file_path = self.raw_dir / dataset_info["filename"]
            if file_path.exists():
                downloaded[dataset_key] = file_path
        
        return downloaded


# Convenience function for easy use
async def download_australian_health_data(
    datasets: Optional[List[str]] = None,
    data_dir: str = "data"
) -> Dict[str, Optional[Path]]:
    """
    Quick function to download Australian health analytics data.
    
    Usage:
        import asyncio
        results = asyncio.run(download_australian_health_data())
    """
    downloader = RealDataDownloader(data_dir)
    return await downloader.download_essential_datasets(datasets)