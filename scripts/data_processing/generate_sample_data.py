#!/usr/bin/env python3
"""
Generate sample data for performance testing and development.

This script creates realistic sample datasets for testing the AHGD application
without requiring the full production datasets.
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Any
import json
import random
import time

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class SampleDataGenerator:
    """Generate sample data for testing and development."""
    
    def __init__(self, db_path: str = "health_analytics_test.db"):
        self.db_path = Path(db_path)
        self.conn = None
        
    def connect(self):
        """Connect to the database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        
    def disconnect(self):
        """Disconnect from the database."""
        if self.conn:
            self.conn.close()
            
    def create_tables(self):
        """Create sample tables for testing."""
        if not self.conn:
            self.connect()
            
        # Create sample health data table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS health_indicators (
                id INTEGER PRIMARY KEY,
                sa2_code TEXT NOT NULL,
                sa2_name TEXT NOT NULL,
                state TEXT NOT NULL,
                population INTEGER,
                median_age REAL,
                diabetes_rate REAL,
                heart_disease_rate REAL,
                mental_health_rate REAL,
                obesity_rate REAL,
                smoking_rate REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create sample geographic data table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS geographic_areas (
                id INTEGER PRIMARY KEY,
                sa2_code TEXT UNIQUE NOT NULL,
                sa2_name TEXT NOT NULL,
                state TEXT NOT NULL,
                area_sq_km REAL,
                latitude REAL,
                longitude REAL,
                postcode TEXT,
                seifa_score INTEGER,
                remoteness_category TEXT
            )
        """)
        
        # Create sample demographic data table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS demographics (
                id INTEGER PRIMARY KEY,
                sa2_code TEXT NOT NULL,
                age_group TEXT NOT NULL,
                gender TEXT NOT NULL,
                population_count INTEGER,
                income_median REAL,
                education_level TEXT,
                employment_rate REAL
            )
        """)
        
        self.conn.commit()
        
    def generate_health_indicators(self, num_records: int = 1000) -> None:
        """Generate sample health indicator data."""
        print(f"Generating {num_records} health indicator records...")
        
        states = ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT"]
        
        records = []
        for i in range(num_records):
            sa2_code = f"{random.randint(100000, 999999)}"
            state = random.choice(states)
            
            record = (
                sa2_code,
                f"Sample Area {i+1}",
                state,
                random.randint(500, 50000),  # population
                random.uniform(25, 55),      # median_age
                random.uniform(3, 15),       # diabetes_rate
                random.uniform(2, 12),       # heart_disease_rate  
                random.uniform(8, 25),       # mental_health_rate
                random.uniform(15, 35),      # obesity_rate
                random.uniform(5, 25),       # smoking_rate
            )
            records.append(record)
            
        self.conn.executemany("""
            INSERT INTO health_indicators 
            (sa2_code, sa2_name, state, population, median_age, diabetes_rate, 
             heart_disease_rate, mental_health_rate, obesity_rate, smoking_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, records)
        
        self.conn.commit()
        print(f"✅ Created {num_records} health indicator records")
        
    def generate_geographic_areas(self, num_records: int = 1000) -> None:
        """Generate sample geographic area data."""
        print(f"Generating {num_records} geographic area records...")
        
        states = ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT"]
        remoteness_categories = ["Major Cities", "Inner Regional", "Outer Regional", "Remote", "Very Remote"]
        
        records = []
        for i in range(num_records):
            sa2_code = f"{random.randint(100000, 999999)}"
            state = random.choice(states)
            
            # Generate realistic coordinates for Australia
            lat_range = {"NSW": (-37, -28), "VIC": (-39, -34), "QLD": (-29, -10)}
            lng_range = {"NSW": (141, 154), "VIC": (141, 150), "QLD": (138, 154)}
            
            lat_min, lat_max = lat_range.get(state, (-45, -10))
            lng_min, lng_max = lng_range.get(state, (113, 154))
            
            record = (
                sa2_code,
                f"Sample Geographic Area {i+1}",
                state,
                random.uniform(1, 1000),  # area_sq_km
                random.uniform(lat_min, lat_max),  # latitude
                random.uniform(lng_min, lng_max),  # longitude
                f"{random.randint(1000, 9999)}",   # postcode
                random.randint(800, 1200),         # seifa_score
                random.choice(remoteness_categories)
            )
            records.append(record)
            
        self.conn.executemany("""
            INSERT INTO geographic_areas 
            (sa2_code, sa2_name, state, area_sq_km, latitude, longitude, 
             postcode, seifa_score, remoteness_category)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, records)
        
        self.conn.commit()
        print(f"✅ Created {num_records} geographic area records")
        
    def generate_demographics(self, num_records: int = 5000) -> None:
        """Generate sample demographic data."""
        print(f"Generating {num_records} demographic records...")
        
        age_groups = ["0-14", "15-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        genders = ["Male", "Female", "Other"]
        education_levels = ["No schooling", "Primary", "Secondary", "Certificate", "Diploma", "Bachelor", "Postgraduate"]
        
        # Get existing SA2 codes
        cursor = self.conn.execute("SELECT DISTINCT sa2_code FROM health_indicators LIMIT 200")
        sa2_codes = [row[0] for row in cursor.fetchall()]
        
        if not sa2_codes:
            print("⚠️  No health indicators found, generating basic SA2 codes")
            sa2_codes = [f"{random.randint(100000, 999999)}" for _ in range(100)]
        
        records = []
        for _ in range(num_records):
            record = (
                random.choice(sa2_codes),
                random.choice(age_groups),
                random.choice(genders),
                random.randint(10, 5000),           # population_count
                random.uniform(30000, 120000),      # income_median
                random.choice(education_levels),
                random.uniform(60, 95),             # employment_rate
            )
            records.append(record)
            
        self.conn.executemany("""
            INSERT INTO demographics 
            (sa2_code, age_group, gender, population_count, income_median, 
             education_level, employment_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, records)
        
        self.conn.commit()
        print(f"✅ Created {num_records} demographic records")
        
    def create_indexes(self) -> None:
        """Create database indexes for better performance."""
        print("Creating database indexes...")
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_health_sa2 ON health_indicators(sa2_code)",
            "CREATE INDEX IF NOT EXISTS idx_health_state ON health_indicators(state)",
            "CREATE INDEX IF NOT EXISTS idx_geo_sa2 ON geographic_areas(sa2_code)",
            "CREATE INDEX IF NOT EXISTS idx_geo_state ON geographic_areas(state)",
            "CREATE INDEX IF NOT EXISTS idx_demo_sa2 ON demographics(sa2_code)",
            "CREATE INDEX IF NOT EXISTS idx_demo_age ON demographics(age_group)",
        ]
        
        for index_sql in indexes:
            self.conn.execute(index_sql)
            
        self.conn.commit()
        print("✅ Created database indexes")
        
    def generate_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics about the sample data."""
        stats = {}
        
        # Count records in each table
        tables = ["health_indicators", "geographic_areas", "demographics"]
        for table in tables:
            cursor = self.conn.execute(f"SELECT COUNT(*) FROM {table}")
            stats[f"{table}_count"] = cursor.fetchone()[0]
            
        # Get unique states
        cursor = self.conn.execute("SELECT DISTINCT state FROM health_indicators")
        stats["unique_states"] = [row[0] for row in cursor.fetchall()]
        
        # Calculate some basic statistics
        cursor = self.conn.execute("""
            SELECT 
                AVG(population) as avg_population,
                MIN(population) as min_population,
                MAX(population) as max_population,
                AVG(diabetes_rate) as avg_diabetes_rate
            FROM health_indicators
        """)
        row = cursor.fetchone()
        if row:
            stats["health_stats"] = {
                "avg_population": round(row[0], 2) if row[0] else 0,
                "min_population": row[1] if row[1] else 0,
                "max_population": row[2] if row[2] else 0,
                "avg_diabetes_rate": round(row[3], 2) if row[3] else 0,
            }
            
        return stats
        
    def create_sample_files(self) -> None:
        """Create sample data files for testing file operations."""
        data_dir = Path("data/test")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample CSV file
        csv_file = data_dir / "sample_health_data.csv"
        with open(csv_file, "w") as f:
            f.write("sa2_code,sa2_name,population,diabetes_rate\n")
            for i in range(100):
                f.write(f"{random.randint(100000, 999999)},Sample Area {i+1},{random.randint(500, 50000)},{random.uniform(3, 15):.2f}\n")
        
        print(f"✅ Created sample CSV file: {csv_file}")
        
        # Create sample JSON file
        json_file = data_dir / "sample_config.json"
        sample_config = {
            "version": "1.0.0",
            "data_sources": ["census_2021", "seifa_2021", "health_indicators"],
            "default_state": "NSW",
            "map_center": {"lat": -33.8688, "lng": 151.2093},
            "performance_thresholds": {
                "query_time_ms": 1000,
                "memory_usage_mb": 512
            }
        }
        
        with open(json_file, "w") as f:
            json.dump(sample_config, f, indent=2)
            
        print(f"✅ Created sample JSON file: {json_file}")


def main():
    """Main entry point for sample data generation."""
    parser = argparse.ArgumentParser(description="Generate sample data for AHGD testing")
    parser.add_argument(
        "--database", 
        default="health_analytics_test.db",
        help="Path to test database file"
    )
    parser.add_argument(
        "--health-records", 
        type=int, 
        default=1000,
        help="Number of health indicator records to generate"
    )
    parser.add_argument(
        "--geo-records", 
        type=int, 
        default=1000,
        help="Number of geographic area records to generate"
    )
    parser.add_argument(
        "--demo-records", 
        type=int, 
        default=5000,
        help="Number of demographic records to generate"
    )
    parser.add_argument(
        "--skip-files", 
        action="store_true",
        help="Skip generating sample files"
    )
    
    args = parser.parse_args()
    
    print("🧪 AHGD Sample Data Generator")
    print("=" * 40)
    
    generator = SampleDataGenerator(args.database)
    
    try:
        generator.connect()
        
        # Create database structure
        generator.create_tables()
        
        # Generate sample data
        start_time = time.time()
        
        generator.generate_health_indicators(args.health_records)
        generator.generate_geographic_areas(args.geo_records)
        generator.generate_demographics(args.demo_records)
        
        # Create indexes for performance
        generator.create_indexes()
        
        generation_time = time.time() - start_time
        
        # Generate summary statistics
        stats = generator.generate_summary_stats()
        
        # Create sample files
        if not args.skip_files:
            generator.create_sample_files()
        
        print("\n📊 Sample Data Summary")
        print("=" * 30)
        print(f"Database: {args.database}")
        print(f"Generation time: {generation_time:.2f} seconds")
        print(f"Health indicators: {stats.get('health_indicators_count', 0)}")
        print(f"Geographic areas: {stats.get('geographic_areas_count', 0)}")
        print(f"Demographics: {stats.get('demographics_count', 0)}")
        print(f"Unique states: {', '.join(stats.get('unique_states', []))}")
        
        if 'health_stats' in stats:
            health_stats = stats['health_stats']
            print(f"Avg population: {health_stats['avg_population']}")
            print(f"Avg diabetes rate: {health_stats['avg_diabetes_rate']}%")
        
        print("\n✅ Sample data generation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error generating sample data: {e}")
        sys.exit(1)
    finally:
        generator.disconnect()


if __name__ == "__main__":
    main()