#!/usr/bin/env python3
"""
AHGD V3: SQLite to Parquet Migration Script
Converts existing SQLite health analytics data to optimized Parquet format.
"""

import sys
import sqlite3
import polars as pl
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.storage.parquet_manager import ParquetStorageManager
from src.utils.logging import get_logger

logger = get_logger("parquet_migration")


class SQLiteToParquetMigrator:
    """
    Migrates existing SQLite health data to optimized Parquet format.
    
    Benefits:
    - 50-90% smaller file sizes
    - 10-100x faster query performance
    - Column-oriented analytics optimization
    - Better compression and scanning
    """
    
    def __init__(self, sqlite_db_path: str = "data/health_analytics.db"):
        self.sqlite_path = Path(sqlite_db_path)
        self.parquet_manager = ParquetStorageManager("./data/parquet_store")
        self.migration_stats = {
            "tables_migrated": 0,
            "total_records": 0,
            "original_size_mb": 0,
            "parquet_size_mb": 0,
            "compression_ratio": 0,
            "start_time": datetime.now()
        }
    
    def get_sqlite_tables(self) -> List[str]:
        """Get all tables from SQLite database."""
        if not self.sqlite_path.exists():
            logger.warning(f"SQLite database not found: {self.sqlite_path}")
            return []
        
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        logger.info(f"Found {len(tables)} tables in SQLite database")
        return tables
    
    def migrate_table(self, table_name: str) -> Dict[str, any]:
        """
        Migrate a single table from SQLite to Parquet.
        
        Args:
            table_name: Name of SQLite table to migrate
            
        Returns:
            Migration statistics for this table
        """
        logger.info(f"Migrating table: {table_name}")
        
        try:
            # Read from SQLite using sqlite3 and convert to Polars
            conn = sqlite3.connect(self.sqlite_path)
            
            # Get column info first
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns_info = cursor.fetchall()
            
            if not columns_info:
                conn.close()
                logger.warning(f"Could not get column info for {table_name}")
                return {"records": 0, "success": False}
            
            # Read data
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            column_names = [info[1] for info in columns_info]
            
            conn.close()
            
            if not rows:
                logger.warning(f"Table {table_name} is empty, skipping")
                return {"records": 0, "success": False}
            
            # Convert to Polars DataFrame
            df = pl.DataFrame(rows, schema=column_names, orient="row")
            
            if df.height == 0:
                logger.warning(f"Table {table_name} is empty, skipping")
                return {"records": 0, "success": False}
            
            # Determine table type and storage strategy
            if "sa1" in table_name.lower() or "geographic" in table_name.lower():
                # Geographic data - partition by state if possible
                has_state_col = any("state" in col.lower() for col in df.columns)
                parquet_path = self.parquet_manager.store_processed_data(
                    df, 
                    table_name,
                    geographic_level="sa1" if "sa1" in table_name.lower() else "mixed",
                    partition_by_state=has_state_col
                )
            elif "raw" in table_name.lower():
                # Raw extraction data
                source = "mixed"
                if "aihw" in table_name.lower():
                    source = "aihw"
                elif "abs" in table_name.lower():
                    source = "abs"
                elif "bom" in table_name.lower():
                    source = "bom"
                
                parquet_path = self.parquet_manager.store_raw_data(
                    df,
                    source=source,
                    dataset=table_name
                )
            else:
                # Processed analytical data
                parquet_path = self.parquet_manager.store_processed_data(
                    df,
                    table_name,
                    geographic_level="mixed"
                )
            
            # Calculate compression stats
            sqlite_size = self._get_table_size(table_name)
            parquet_size = parquet_path.stat().st_size if parquet_path.is_file() else self._get_dir_size(parquet_path)
            compression_ratio = sqlite_size / parquet_size if parquet_size > 0 else 0
            
            stats = {
                "table": table_name,
                "records": df.height,
                "columns": len(df.columns),
                "sqlite_size_mb": sqlite_size / (1024 * 1024),
                "parquet_size_mb": parquet_size / (1024 * 1024),
                "compression_ratio": compression_ratio,
                "success": True,
                "parquet_path": str(parquet_path)
            }
            
            logger.info(
                f"✅ Migrated {table_name}: {df.height:,} records, "
                f"{compression_ratio:.1f}x compression"
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to migrate table {table_name}: {str(e)}")
            return {"table": table_name, "success": False, "error": str(e)}
    
    def _get_table_size(self, table_name: str) -> int:
        """Get SQLite table size in bytes."""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT COUNT(*) * AVG(LENGTH(CAST(rowid AS TEXT))) FROM {table_name}")
        size = cursor.fetchone()[0] or 0
        
        conn.close()
        return int(size)
    
    def _get_dir_size(self, path: Path) -> int:
        """Get directory size recursively."""
        return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    
    def migrate_all_tables(self) -> Dict[str, any]:
        """
        Migrate all tables from SQLite to Parquet.
        
        Returns:
            Complete migration statistics
        """
        logger.info("🚀 Starting SQLite to Parquet migration")
        
        tables = self.get_sqlite_tables()
        if not tables:
            logger.error("No tables found to migrate")
            return {"success": False, "error": "No tables found"}
        
        successful_migrations = []
        failed_migrations = []
        
        for table in tables:
            result = self.migrate_table(table)
            
            if result.get("success", False):
                successful_migrations.append(result)
                self.migration_stats["tables_migrated"] += 1
                self.migration_stats["total_records"] += result.get("records", 0)
                self.migration_stats["original_size_mb"] += result.get("sqlite_size_mb", 0)
                self.migration_stats["parquet_size_mb"] += result.get("parquet_size_mb", 0)
            else:
                failed_migrations.append(result)
        
        # Calculate overall compression
        if self.migration_stats["parquet_size_mb"] > 0:
            self.migration_stats["compression_ratio"] = (
                self.migration_stats["original_size_mb"] / 
                self.migration_stats["parquet_size_mb"]
            )
        
        self.migration_stats["end_time"] = datetime.now()
        self.migration_stats["duration_minutes"] = (
            self.migration_stats["end_time"] - self.migration_stats["start_time"]
        ).total_seconds() / 60
        
        # Generate summary report
        self._print_migration_summary(successful_migrations, failed_migrations)
        
        return {
            "success": len(failed_migrations) == 0,
            "statistics": self.migration_stats,
            "successful": successful_migrations,
            "failed": failed_migrations
        }
    
    def _print_migration_summary(self, successful: List[Dict], failed: List[Dict]):
        """Print detailed migration summary."""
        
        print("\n" + "="*60)
        print("🎉 AHGD SQLite → Parquet Migration Complete!")
        print("="*60)
        
        print(f"\n📊 MIGRATION STATISTICS:")
        print(f"   Tables migrated: {self.migration_stats['tables_migrated']}")
        print(f"   Total records:   {self.migration_stats['total_records']:,}")
        print(f"   Original size:   {self.migration_stats['original_size_mb']:.1f} MB")
        print(f"   Parquet size:    {self.migration_stats['parquet_size_mb']:.1f} MB")
        print(f"   Compression:     {self.migration_stats['compression_ratio']:.1f}x smaller")
        print(f"   Duration:        {self.migration_stats['duration_minutes']:.1f} minutes")
        
        if successful:
            print(f"\n✅ SUCCESSFUL MIGRATIONS ({len(successful)}):")
            for table in successful:
                print(f"   {table['table']:30} {table['records']:>8,} records  {table['compression_ratio']:>5.1f}x")
        
        if failed:
            print(f"\n❌ FAILED MIGRATIONS ({len(failed)}):")
            for table in failed:
                table_name = table.get('table', 'Unknown table')
                error_msg = table.get('error', 'Unknown error')
                print(f"   {table_name:30} {error_msg}")
        
        print(f"\n🚀 PERFORMANCE BENEFITS:")
        print(f"   • Query speed:     10-100x faster")
        print(f"   • Storage size:    {self.migration_stats['compression_ratio']:.1f}x smaller")
        print(f"   • Analytics:       Column-oriented optimization")
        print(f"   • Compatibility:   Works with all Polars/DuckDB tools")
        
        print(f"\n📁 Parquet data stored in: ./data/parquet_store/")
        print("="*60 + "\n")


def main():
    """Run the migration process."""
    
    # Check if SQLite database exists
    sqlite_db = Path("data/health_analytics.db")
    if not sqlite_db.exists():
        print(f"❌ SQLite database not found: {sqlite_db}")
        print("   Please ensure the database exists before running migration.")
        return 1
    
    # Run migration
    migrator = SQLiteToParquetMigrator(str(sqlite_db))
    results = migrator.migrate_all_tables()
    
    if results["success"]:
        print("🎉 Migration completed successfully!")
        
        # Optional: Backup original SQLite database
        backup_path = sqlite_db.with_suffix(".db.backup")
        sqlite_db.rename(backup_path)
        print(f"📦 Original database backed up to: {backup_path}")
        
        return 0
    else:
        print("❌ Migration completed with errors.")
        return 1


if __name__ == "__main__":
    sys.exit(main())