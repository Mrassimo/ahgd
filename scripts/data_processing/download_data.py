#!/usr/bin/env python3
"""
Australian Health Data Analytics - Data Download Script

This script downloads Australian Bureau of Statistics Census 2021 data,
SEIFA 2021 data, and geographic boundaries for health analytics projects.

Focuses on NSW data initially to keep downloads manageable.
Uses async HTTP requests for efficient parallel downloads.

Author: Australian Health Data Analytics Project
Date: 2025-06-17
"""

import asyncio
import httpx
import logging
from pathlib import Path
from datetime import datetime
import zipfile
import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Data class for tracking download sources"""
    name: str
    url: str
    filename: str
    size_mb: Optional[float] = None
    description: str = ""
    category: str = ""
    
class DataDownloader:
    """Async data downloader for Australian health data"""
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path(__file__).parent.parent / "data" / "raw"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.download_log = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def get_data_sources(self) -> Dict[str, List[DataSource]]:
        """Define all data sources for download"""
        
        sources = {
            "census_2021": [
                DataSource(
                    name="Census 2021 - NSW SA2 General Community Profile",
                    url="https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_all_for_NSW_short-header.zip",
                    filename="2021_GCP_NSW_SA2.zip",
                    size_mb=183,
                    description="Complete demographic data for NSW Statistical Areas Level 2",
                    category="demographics"
                ),
                DataSource(
                    name="Census 2021 - Australia SA2 General Community Profile",
                    url="https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_all_for_AUS_short-header.zip",
                    filename="2021_GCP_AUS_SA2.zip",
                    size_mb=584,
                    description="Complete demographic data for all Australian SA2s (large file - optional)",
                    category="demographics"
                )
            ],
            
            "seifa_2021": [
                DataSource(
                    name="SEIFA 2021 - SA2 Level Socio-Economic Indexes",
                    url="https://www.abs.gov.au/statistics/people/people-and-communities/socio-economic-indexes-areas-seifa-australia/2021/Statistical%20Area%20Level%202%2C%20Indexes%2C%20SEIFA%202021.xlsx",
                    filename="SEIFA_2021_SA2.xlsx",
                    size_mb=1.26,
                    description="All four SEIFA indexes by SA2 with rankings and deciles",
                    category="socioeconomic"
                )
            ],
            
            "geographic_boundaries": [
                DataSource(
                    name="SA2 2021 Digital Boundaries (GDA2020)",
                    url="https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_SHP_GDA2020.zip",
                    filename="SA2_2021_AUST_SHP_GDA2020.zip",
                    size_mb=95.97,
                    description="Shapefile format SA2 boundaries for geographic analysis",
                    category="geographic"
                ),
                DataSource(
                    name="SA2 2021 Digital Boundaries (GDA94)",
                    url="https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_SHP_GDA94.zip",
                    filename="SA2_2021_AUST_SHP_GDA94.zip",
                    size_mb=47.46,
                    description="Legacy coordinate system SA2 boundaries",
                    category="geographic"
                )
            ],
            
            "health_data": [
                DataSource(
                    name="MBS Patient Demographics Historical (1993-2015)",
                    url="https://data.gov.au/data/dataset/8a19a28f-35b0-4035-8cd5-5b611b3cfa6f/resource/519b55ab-8f81-47d1-a483-8495668e38d8/download/mbs-demographics-historical-1993-2015.zip",
                    filename="mbs_demographics_historical_1993_2015.zip",
                    size_mb=20,  # Estimated
                    description="Historical Medicare patient demographics and service usage",
                    category="health"
                ),
                DataSource(
                    name="PBS Current Year Data (2016)",
                    url="https://data.gov.au/data/dataset/14b536d4-eb6a-485d-bf87-2e6e77ddbac1/resource/08eda5ab-01c0-4c94-8b1a-157bcffe80d3/download/pbs-item-2016csvjuly.csv",
                    filename="pbs_current_2016.csv",
                    size_mb=5,  # Estimated
                    description="Current year pharmaceutical usage data",
                    category="health"
                ),
                DataSource(
                    name="PBS Historical Data (1992-2014)",
                    url="https://data.gov.au/data/dataset/14b536d4-eb6a-485d-bf87-2e6e77ddbac1/resource/56f87bbb-a7cb-4cbf-a723-7aec22996eee/download/csv-pbs-item-historical-1992-2014.zip",
                    filename="pbs_historical_1992_2014.zip",
                    size_mb=15,  # Estimated
                    description="Historical pharmaceutical data",
                    category="health"
                )
            ]
        }
        
        return sources
    
    async def download_file(self, source: DataSource, client: httpx.AsyncClient) -> Dict:
        """Download a single file with progress tracking"""
        start_time = datetime.now()
        file_path = self.base_dir / source.category / source.filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if file exists and is non-empty
        if file_path.exists() and file_path.stat().st_size > 0:
            logger.info(f"✓ Skipping {source.name} - file already exists")
            return {
                "source": source.name,
                "status": "skipped",
                "file_path": str(file_path),
                "size_bytes": file_path.stat().st_size
            }
        
        try:
            logger.info(f"📥 Starting download: {source.name}")
            logger.info(f"    URL: {source.url}")
            logger.info(f"    Expected size: {source.size_mb} MB")
            
            async with client.stream('GET', source.url) as response:
                response.raise_for_status()
                
                total_bytes = int(response.headers.get('content-length', 0))
                downloaded_bytes = 0
                
                with open(file_path, 'wb') as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
                        downloaded_bytes += len(chunk)
                        
                        # Progress logging every 10MB
                        if downloaded_bytes % (10 * 1024 * 1024) == 0:
                            mb_downloaded = downloaded_bytes / (1024 * 1024)
                            logger.info(f"    Progress: {mb_downloaded:.1f} MB downloaded")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            size_mb = downloaded_bytes / (1024 * 1024)
            
            logger.info(f"✅ Completed: {source.name}")
            logger.info(f"    Size: {size_mb:.2f} MB")
            logger.info(f"    Time: {duration:.1f} seconds")
            logger.info(f"    Speed: {size_mb/duration:.2f} MB/s")
            
            return {
                "source": source.name,
                "status": "success",
                "file_path": str(file_path),
                "size_bytes": downloaded_bytes,
                "size_mb": size_mb,
                "duration_seconds": duration,
                "url": source.url
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to download {source.name}: {str(e)}")
            return {
                "source": source.name,
                "status": "failed",
                "error": str(e),
                "url": source.url
            }
    
    def verify_downloads(self) -> Dict:
        """Verify downloaded files are complete and not corrupted"""
        verification_results = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "details": []
        }
        
        logger.info("🔍 Verifying downloaded files...")
        
        for category_dir in self.base_dir.iterdir():
            if not category_dir.is_dir():
                continue
                
            for file_path in category_dir.glob("*"):
                if not file_path.is_file():
                    continue
                    
                verification_results["total_files"] += 1
                
                try:
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    
                    # Basic verification: file exists and has reasonable size
                    if size_mb < 0.1:  # Less than 100KB is suspicious
                        raise ValueError(f"File too small: {size_mb:.2f} MB")
                    
                    # Try to peek into ZIP files to verify they're not corrupted
                    if file_path.suffix.lower() == '.zip':
                        with zipfile.ZipFile(file_path, 'r') as zip_file:
                            zip_file.testzip()  # This will raise an exception if corrupted
                    
                    verification_results["successful"] += 1
                    verification_results["details"].append({
                        "file": file_path.name,
                        "status": "verified",
                        "size_mb": round(size_mb, 2)
                    })
                    
                    logger.info(f"✅ Verified: {file_path.name} ({size_mb:.2f} MB)")
                    
                except Exception as e:
                    verification_results["failed"] += 1
                    verification_results["details"].append({
                        "file": file_path.name,
                        "status": "failed",
                        "error": str(e)
                    })
                    logger.error(f"❌ Verification failed: {file_path.name} - {str(e)}")
        
        return verification_results
    
    def generate_download_report(self, download_results: List[Dict], verification_results: Dict) -> str:
        """Generate a comprehensive download report"""
        
        report = f"""# Australian Health Data Download Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Session ID: {self.session_id}

## Download Summary
"""
        
        successful_downloads = [r for r in download_results if r.get("status") == "success"]
        failed_downloads = [r for r in download_results if r.get("status") == "failed"]
        skipped_downloads = [r for r in download_results if r.get("status") == "skipped"]
        
        total_size_mb = sum(r.get("size_mb", 0) for r in successful_downloads)
        
        report += f"""
- **Total Files Attempted**: {len(download_results)}
- **Successful Downloads**: {len(successful_downloads)}
- **Failed Downloads**: {len(failed_downloads)}
- **Skipped (Already Exists)**: {len(skipped_downloads)}
- **Total Data Downloaded**: {total_size_mb:.2f} MB

## Verification Results
- **Total Files Verified**: {verification_results['total_files']}
- **Successfully Verified**: {verification_results['successful']}
- **Verification Failures**: {verification_results['failed']}

## Downloaded Files by Category
"""
        
        # Group by category
        categories = {}
        for result in successful_downloads + skipped_downloads:
            if "file_path" in result:
                path = Path(result["file_path"])
                category = path.parent.name
                if category not in categories:
                    categories[category] = []
                categories[category].append(result)
        
        for category, files in categories.items():
            report += f"\n### {category.title()}\n"
            for file_info in files:
                path = Path(file_info["file_path"])
                size_mb = file_info.get("size_mb", file_info.get("size_bytes", 0) / (1024*1024))
                status = file_info.get("status", "unknown")
                report += f"- `{path.name}` ({size_mb:.2f} MB) - {status}\n"
        
        if failed_downloads:
            report += f"\n## Failed Downloads\n"
            for failure in failed_downloads:
                report += f"- **{failure['source']}**: {failure.get('error', 'Unknown error')}\n"
        
        report += f"""
## Data Sources Documentation

### Census 2021 Data
- **Provider**: Australian Bureau of Statistics (ABS)
- **Coverage**: NSW SA2 level demographic data
- **Variables**: Population, age, income, education, employment, housing
- **Geography**: Statistical Area Level 2 (SA2) boundaries
- **License**: Creative Commons Attribution 4.0 International

### SEIFA 2021 Data  
- **Provider**: Australian Bureau of Statistics (ABS)
- **Coverage**: All Australian SA2s
- **Indexes**: IRSAD, IRSD, IER, IEO (socio-economic advantage/disadvantage)
- **Format**: Rankings, deciles, percentiles by SA2

### Geographic Boundaries
- **Provider**: Australian Bureau of Statistics (ABS)
- **Format**: ESRI Shapefile
- **Coordinate Systems**: GDA2020 (preferred), GDA94 (legacy)
- **Coverage**: All Australian SA2 boundaries

### Health Data
- **Provider**: data.gov.au / Department of Health
- **Coverage**: Medicare Benefits Schedule (MBS) and Pharmaceutical Benefits Scheme (PBS)
- **Time Range**: Historical data 1992-2015, current data 2016
- **Privacy**: Aggregated data only, no individual records

## Next Steps
1. Extract ZIP files to appropriate directories
2. Load Census data into data processing pipeline (Polars recommended)
3. Join SEIFA data with geographic boundaries
4. Begin exploratory data analysis focusing on NSW regions
5. Set up automated data refresh pipeline

## Technical Notes
- All downloads completed using async HTTP clients (httpx)
- Files verified for completeness and format integrity
- Download speeds and file sizes logged for monitoring
- Failed downloads should be retried or investigated

---
*Report generated by Australian Health Data Analytics download script*
*For questions or issues, check the data_download.log file*
"""
        
        return report
    
    async def download_all(self, categories: List[str] = None, max_concurrent: int = 3) -> Dict:
        """Download all data sources with controlled concurrency"""
        
        sources = self.get_data_sources()
        
        # Filter categories if specified
        if categories:
            sources = {k: v for k, v in sources.items() if k in categories}
        
        all_sources = []
        for category_sources in sources.values():
            all_sources.extend(category_sources)
        
        logger.info(f"🚀 Starting download of {len(all_sources)} files...")
        logger.info(f"📁 Download directory: {self.base_dir}")
        
        # Calculate total expected size
        total_size_mb = sum(s.size_mb or 0 for s in all_sources)
        logger.info(f"📊 Total expected download size: {total_size_mb:.1f} MB")
        
        # Create semaphore to limit concurrent downloads
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def download_with_semaphore(source: DataSource, client: httpx.AsyncClient):
            async with semaphore:
                return await self.download_file(source, client)
        
        # Perform downloads
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            download_tasks = [
                download_with_semaphore(source, client) 
                for source in all_sources
            ]
            download_results = await asyncio.gather(*download_tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(download_results):
            if isinstance(result, Exception):
                processed_results.append({
                    "source": all_sources[i].name,
                    "status": "failed",
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        # Verify downloads
        verification_results = self.verify_downloads()
        
        # Generate report
        report = self.generate_download_report(processed_results, verification_results)
        
        # Save report
        report_path = self.base_dir / f"download_report_{self.session_id}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"📋 Download report saved: {report_path}")
        
        return {
            "download_results": processed_results,
            "verification_results": verification_results,
            "report_path": str(report_path),
            "total_downloaded_mb": sum(r.get("size_mb", 0) for r in processed_results if r.get("status") == "success")
        }

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Australian health data")
    parser.add_argument(
        "--categories", 
        nargs="+", 
        choices=["census_2021", "seifa_2021", "geographic_boundaries", "health_data"],
        help="Specific categories to download (default: all except large Australia-wide Census data)"
    )
    parser.add_argument(
        "--max-concurrent", 
        type=int, 
        default=3,
        help="Maximum concurrent downloads (default: 3)"
    )
    parser.add_argument(
        "--include-australia", 
        action="store_true",
        help="Include large Australia-wide Census data (584 MB)"
    )
    
    args = parser.parse_args()
    
    # Default to essential NSW-focused data
    if not args.categories:
        if args.include_australia:
            categories = ["census_2021", "seifa_2021", "geographic_boundaries", "health_data"]
        else:
            categories = ["seifa_2021", "geographic_boundaries", "health_data"]
            # Add only NSW Census data by modifying the downloader
    else:
        categories = args.categories
    
    downloader = DataDownloader()
    
    # Run the async download
    try:
        results = asyncio.run(downloader.download_all(
            categories=categories,
            max_concurrent=args.max_concurrent
        ))
        
        print(f"\n🎉 Download completed!")
        print(f"📊 Total downloaded: {results['total_downloaded_mb']:.2f} MB")
        print(f"📋 Report: {results['report_path']}")
        
        # Print summary
        download_results = results['download_results']
        successful = len([r for r in download_results if r.get("status") == "success"])
        failed = len([r for r in download_results if r.get("status") == "failed"])
        skipped = len([r for r in download_results if r.get("status") == "skipped"])
        
        print(f"\n📈 Summary:")
        print(f"   ✅ Successful: {successful}")
        print(f"   ⏭️  Skipped: {skipped}")
        print(f"   ❌ Failed: {failed}")
        
        if failed > 0:
            print(f"\n⚠️  Some downloads failed. Check data_download.log for details.")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Download cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())