"""
Data Volume Monitor for Australian Health Data Pipeline.

Monitors data loss through pipeline stages and provides alerts
when significant data loss is detected.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import polars as pl
from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()


class DataVolumeMonitor:
    """
    Monitor data volumes through the processing pipeline.
    
    Tracks raw vs processed data sizes and calculates retention rates.
    Provides alerts when data loss exceeds acceptable thresholds.
    """
    
    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Alert thresholds
        self.WARNING_THRESHOLD = 0.7  # Warn if <70% retention
        self.CRITICAL_THRESHOLD = 0.3  # Critical if <30% retention
    
    def get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in megabytes."""
        if file_path.exists():
            return file_path.stat().st_size / (1024 * 1024)
        return 0.0
    
    def get_zip_sizes(self, zip_filenames: List[str]) -> float:
        """Get total size of ZIP files in MB."""
        total_size = 0.0
        for filename in zip_filenames:
            zip_path = self.raw_dir / filename
            total_size += self.get_file_size_mb(zip_path)
        return total_size
    
    def get_parquet_sizes(self, pattern: str) -> float:
        """Get total size of Parquet files matching pattern."""
        total_size = 0.0
        for file_path in self.processed_dir.glob(pattern):
            total_size += self.get_file_size_mb(file_path)
        return total_size
    
    def get_record_count(self, file_path: Path) -> int:
        """Get record count from Parquet file."""
        try:
            if file_path.exists() and file_path.suffix == '.parquet':
                df = pl.read_parquet(file_path)
                return len(df)
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
        return 0
    
    def calculate_retention_rate(self, raw_size_mb: float, processed_size_mb: float) -> float:
        """Calculate data retention rate as percentage."""
        if raw_size_mb == 0:
            return 0.0
        return (processed_size_mb / raw_size_mb) * 100
    
    def monitor_pipeline_data_loss(self) -> Dict[str, Dict]:
        """
        Monitor data loss through pipeline stages.
        
        Returns comprehensive data loss analysis.
        """
        logger.info("Starting data volume analysis")
        
        results = {}
        
        # Census Data Analysis - look in demographics directory
        census_raw_dir = self.raw_dir / "demographics"
        census_raw_size = 0.0
        for zip_name in ['2021_GCP_AUS_SA2.zip', '2021_GCP_NSW_SA2.zip']:
            census_raw_size += self.get_file_size_mb(census_raw_dir / zip_name)
        
        census_processed_size = self.get_parquet_sizes('census_*.parquet')
        census_retention = self.calculate_retention_rate(census_raw_size, census_processed_size)
        
        results['census'] = {
            'raw_size_mb': census_raw_size,
            'processed_size_mb': census_processed_size,
            'retention_rate': census_retention,
            'status': self._get_status(census_retention)
        }
        
        # Health Data Analysis - look in health directory
        health_raw_dir = self.raw_dir / "health"
        health_raw_size = 0.0
        for zip_name in ['mbs_demographics_historical_1993_2015.zip', 'pbs_historical_1992_2014.zip']:
            health_raw_size += self.get_file_size_mb(health_raw_dir / zip_name)
        
        health_processed_size = self.get_parquet_sizes('health_*.parquet')
        health_retention = self.calculate_retention_rate(health_raw_size, health_processed_size)
        
        results['health'] = {
            'raw_size_mb': health_raw_size,
            'processed_size_mb': health_processed_size,
            'retention_rate': health_retention,
            'status': self._get_status(health_retention)
        }
        
        # SEIFA Data Analysis - look in socioeconomic directory
        seifa_raw_dir = self.raw_dir / "socioeconomic"
        seifa_raw_size = self.get_file_size_mb(seifa_raw_dir / 'SEIFA_2021_SA2.xlsx')
        seifa_processed_size = self.get_parquet_sizes('seifa_*.parquet')
        seifa_retention = self.calculate_retention_rate(seifa_raw_size, seifa_processed_size)
        
        results['seifa'] = {
            'raw_size_mb': seifa_raw_size,
            'processed_size_mb': seifa_processed_size,
            'retention_rate': seifa_retention,
            'status': self._get_status(seifa_retention)
        }
        
        # Overall Analysis
        total_raw = census_raw_size + health_raw_size + seifa_raw_size
        total_processed = census_processed_size + health_processed_size + seifa_processed_size
        overall_retention = self.calculate_retention_rate(total_raw, total_processed)
        
        results['overall'] = {
            'raw_size_mb': total_raw,
            'processed_size_mb': total_processed,
            'retention_rate': overall_retention,
            'status': self._get_status(overall_retention)
        }
        
        # Generate alerts
        self._generate_alerts(results)
        
        return results
    
    def _get_status(self, retention_rate: float) -> str:
        """Get status based on retention rate."""
        if retention_rate >= self.WARNING_THRESHOLD * 100:
            return "✅ Good"
        elif retention_rate >= self.CRITICAL_THRESHOLD * 100:
            return "⚠️  Warning"
        else:
            return "❌ Critical"
    
    def _generate_alerts(self, results: Dict) -> None:
        """Generate alerts for significant data loss."""
        alerts = []
        
        for data_type, metrics in results.items():
            retention = metrics['retention_rate']
            
            if retention < self.CRITICAL_THRESHOLD * 100:
                alerts.append(f"❌ CRITICAL: {data_type} has {retention:.1f}% retention (expected >30%)")
            elif retention < self.WARNING_THRESHOLD * 100:
                alerts.append(f"⚠️  WARNING: {data_type} has {retention:.1f}% retention (expected >70%)")
        
        if alerts:
            console.print("\n🚨 Data Loss Alerts:")
            for alert in alerts:
                console.print(f"   {alert}")
        else:
            console.print("✅ No data loss alerts")
    
    def print_summary_table(self, results: Dict) -> None:
        """Print a formatted summary table."""
        table = Table(title="Data Pipeline Volume Analysis")
        
        table.add_column("Data Type", style="cyan")
        table.add_column("Raw Size (MB)", justify="right")
        table.add_column("Processed Size (MB)", justify="right")
        table.add_column("Retention Rate", justify="right")
        table.add_column("Status")
        
        for data_type, metrics in results.items():
            table.add_row(
                data_type.title(),
                f"{metrics['raw_size_mb']:.1f}",
                f"{metrics['processed_size_mb']:.1f}",
                f"{metrics['retention_rate']:.1f}%",
                metrics['status']
            )
        
        console.print("\n")
        console.print(table)
    
    def get_detailed_record_counts(self) -> Dict[str, int]:
        """Get detailed record counts from processed files."""
        counts = {}
        
        # Count records in each processed file
        for parquet_file in self.processed_dir.glob("*.parquet"):
            record_count = self.get_record_count(parquet_file)
            counts[parquet_file.name] = record_count
        
        return counts


def run_data_volume_analysis():
    """Run complete data volume analysis."""
    monitor = DataVolumeMonitor()
    
    console.print("🔍 Running Data Volume Analysis...")
    results = monitor.monitor_pipeline_data_loss()
    
    # Print summary table
    monitor.print_summary_table(results)
    
    # Print record counts
    record_counts = monitor.get_detailed_record_counts()
    if record_counts:
        console.print("\n📊 Record Counts by File:")
        for filename, count in record_counts.items():
            console.print(f"   {filename}: {count:,} records")
    
    return results


if __name__ == "__main__":
    run_data_volume_analysis()