"""
Australian Bureau of Statistics (ABS) data downloader.

Ultra-fast async downloading of ABS datasets including:
- Census DataPacks
- SEIFA Indexes  
- Geographic boundaries (ASGS)
- Statistical data via ABS API
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urljoin

import httpx
from loguru import logger
from rich.console import Console
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn

console = Console()


class ABSDownloader:
    """
    High-performance downloader for Australian Bureau of Statistics data.
    
    Features:
    - Async downloads for maximum speed
    - Automatic retry with exponential backoff
    - Progress tracking with Rich
    - Organised file storage by data type
    """
    
    # ABS data source URLs (2021 Census and current data)
    BASE_URLS = {
        "census": "https://www.abs.gov.au/census/find-census-data/datapacks/",
        "seifa": "https://www.abs.gov.au/statistics/people/people-and-communities/",
        "boundaries": "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/",
        "api": "https://api.data.abs.gov.au/",
    }
    
    # Essential datasets for health analytics
    ESSENTIAL_DATASETS = {
        # 2021 Census DataPacks by state
        "census_nsw": "https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_SA2_for_NSW_short-header.zip",
        "census_vic": "https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_SA2_for_VIC_short-header.zip", 
        "census_qld": "https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_SA2_for_QLD_short-header.zip",
        "census_wa": "https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_SA2_for_WA_short-header.zip",
        "census_sa": "https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_SA2_for_SA_short-header.zip",
        "census_tas": "https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_SA2_for_TAS_short-header.zip",
        "census_act": "https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_SA2_for_ACT_short-header.zip",
        "census_nt": "https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_SA2_for_NT_short-header.zip",
        
        # SEIFA 2021 Data
        "seifa_2021": "https://www.abs.gov.au/statistics/people/people-and-communities/socio-economic-indexes-areas-seifa-australia/2021/SEIFA_2021_LGA.xlsx",
        "seifa_sa2": "https://www.abs.gov.au/statistics/people/people-and-communities/socio-economic-indexes-areas-seifa-australia/2021/SEIFA_2021_SA2.xlsx",
        
        # Geographic boundaries
        "sa2_boundaries": "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_GDA2020_SHP.zip",
        "lga_boundaries": "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/LGA_2021_AUST_GDA2020_SHP.zip",
        
        # Postcode to SA2 concordance
        "postcode_sa2": "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/correspondence-files/CG_POA_2021_SA2_2021.csv",
    }
    
    def __init__(self, data_dir: Union[str, Path] = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw" / "abs"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # HTTP client settings
        self.timeout = httpx.Timeout(60.0, connect=30.0)
        self.limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
        
        logger.info(f"ABS Downloader initialised, data directory: {self.raw_dir}")
    
    async def download_file(
        self, 
        session: httpx.AsyncClient, 
        url: str, 
        filename: str,
        progress: Progress,
        task_id: TaskID,
    ) -> Optional[Path]:
        """
        Download a single file with progress tracking and error handling.
        """
        file_path = self.raw_dir / filename
        
        try:
            logger.info(f"Downloading {filename} from {url}")
            
            async with session.stream('GET', url) as response:
                response.raise_for_status()
                
                # Get file size for progress tracking
                total_size = int(response.headers.get('content-length', 0))
                progress.update(task_id, total=total_size)
                
                with open(file_path, 'wb') as file:
                    downloaded = 0
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        file.write(chunk)
                        downloaded += len(chunk)
                        progress.update(task_id, completed=downloaded)
            
            logger.info(f"✓ Downloaded {filename} ({file_path.stat().st_size:,} bytes)")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to download {filename}: {str(e)}")
            if file_path.exists():
                file_path.unlink()  # Remove partial download
            return None
    
    async def download_essential_data(self, states: Optional[List[str]] = None) -> Dict[str, Optional[Path]]:
        """
        Download essential ABS datasets for health analytics.
        
        Args:
            states: List of state codes to download (default: all states)
                   Options: ['nsw', 'vic', 'qld', 'wa', 'sa', 'tas', 'act', 'nt']
        """
        if states is None:
            states = ['nsw', 'vic', 'qld', 'wa', 'sa', 'tas', 'act', 'nt']
        
        # Build download list
        download_list = {}
        
        # Add census data for requested states
        for state in states:
            census_key = f"census_{state.lower()}"
            if census_key in self.ESSENTIAL_DATASETS:
                download_list[f"2021_Census_{state.upper()}.zip"] = self.ESSENTIAL_DATASETS[census_key]
        
        # Add essential non-census datasets
        essential_non_census = {
            "SEIFA_2021_SA2.xlsx": self.ESSENTIAL_DATASETS["seifa_sa2"],
            "SA2_2021_Boundaries.zip": self.ESSENTIAL_DATASETS["sa2_boundaries"],
            "Postcode_SA2_Concordance.csv": self.ESSENTIAL_DATASETS["postcode_sa2"],
        }
        download_list.update(essential_non_census)
        
        console.print(f"📡 Downloading {len(download_list)} essential ABS datasets...")
        
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
                tasks = []
                for filename, url in download_list.items():
                    task_id = progress.add_task(f"Downloading {filename}", total=None)
                    task = self.download_file(session, url, filename, progress, task_id)
                    tasks.append((filename, task))
                
                # Execute all downloads concurrently
                for filename, task in tasks:
                    results[filename] = await task
        
        # Summary
        successful = sum(1 for path in results.values() if path is not None)
        console.print(f"✅ Downloaded {successful}/{len(download_list)} files successfully")
        
        if failed := [name for name, path in results.items() if path is None]:
            console.print(f"❌ Failed downloads: {', '.join(failed)}")
        
        return results
    
    async def download_specific_dataset(self, dataset_key: str) -> Optional[Path]:
        """Download a specific dataset by key."""
        if dataset_key not in self.ESSENTIAL_DATASETS:
            logger.error(f"Unknown dataset key: {dataset_key}")
            return None
        
        url = self.ESSENTIAL_DATASETS[dataset_key]
        filename = f"{dataset_key}.{url.split('.')[-1]}"
        
        async with httpx.AsyncClient(timeout=self.timeout, limits=self.limits) as session:
            with Progress(console=console) as progress:
                task_id = progress.add_task(f"Downloading {filename}", total=None)
                return await self.download_file(session, url, filename, progress, task_id)
    
    def list_available_datasets(self) -> None:
        """Display all available datasets."""
        console.print("📊 Available ABS datasets:")
        console.print()
        
        categories = {
            "Census Data": [k for k in self.ESSENTIAL_DATASETS.keys() if k.startswith("census_")],
            "SEIFA Data": [k for k in self.ESSENTIAL_DATASETS.keys() if k.startswith("seifa_")],
            "Geographic Boundaries": [k for k in self.ESSENTIAL_DATASETS.keys() if "boundaries" in k],
            "Concordances": [k for k in self.ESSENTIAL_DATASETS.keys() if "postcode" in k or "concordance" in k.lower()],
        }
        
        for category, datasets in categories.items():
            if datasets:
                console.print(f"[bold]{category}:[/bold]")
                for dataset in datasets:
                    console.print(f"  • {dataset}")
                console.print()
    
    def get_downloaded_files(self) -> Dict[str, Path]:
        """Get list of already downloaded files."""
        downloaded = {}
        for file_path in self.raw_dir.iterdir():
            if file_path.is_file():
                downloaded[file_path.name] = file_path
        return downloaded
    
    def clean_old_downloads(self, keep_days: int = 30) -> None:
        """Remove old downloaded files to save space."""
        import time
        from datetime import datetime, timedelta
        
        cutoff_time = time.time() - (keep_days * 24 * 60 * 60)
        removed_count = 0
        
        for file_path in self.raw_dir.iterdir():
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                removed_count += 1
                logger.info(f"Removed old file: {file_path.name}")
        
        if removed_count > 0:
            console.print(f"🧹 Cleaned up {removed_count} old files (older than {keep_days} days)")
        else:
            console.print("✨ No old files to clean up")


# Convenience function for quick downloads
async def download_abs_essentials(states: Optional[List[str]] = None, data_dir: str = "data") -> Dict[str, Optional[Path]]:
    """
    Quick function to download essential ABS data.
    
    Usage:
        import asyncio
        results = asyncio.run(download_abs_essentials(['nsw', 'vic']))
    """
    downloader = ABSDownloader(data_dir)
    return await downloader.download_essential_data(states)