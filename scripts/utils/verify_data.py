#!/usr/bin/env python3
"""
Data Verification Script
Quick tests to validate the processed data
"""

import duckdb
from pathlib import Path

# Connect to the database
db_path = Path("/Users/massimoraso/AHGD/data/health_analytics.db")
conn = duckdb.connect(str(db_path))

print("🔍 Verifying Australian Health Data Analytics Database")
print("=" * 55)

# Basic table information
print("\n📊 Database Tables:")
tables = conn.execute("SHOW TABLES").fetchall()
for table in tables:
    count = conn.execute(f"SELECT COUNT(*) FROM {table[0]}").fetchone()[0]
    print(f"  • {table[0]}: {count:,} records")

# SEIFA data analysis
print("\n📈 SEIFA Data Analysis:")
seifa_stats = conn.execute("""
    SELECT 
        COUNT(*) as total_areas,
        MIN(IRSD_Score) as min_disadvantage_score,
        MAX(IRSD_Score) as max_disadvantage_score,
        AVG(IRSD_Score) as avg_disadvantage_score,
        COUNT(DISTINCT State_Name_2021) as states_territories
    FROM seifa_2021
""").fetchone()

print(f"  • Total SA2 Areas: {seifa_stats[0]:,}")
print(f"  • Disadvantage Score Range: {seifa_stats[1]:.1f} - {seifa_stats[2]:.1f}")
print(f"  • Average Score: {seifa_stats[3]:.1f}")
print(f"  • States/Territories: {seifa_stats[4]}")

# Geographic coverage
print("\n🗺️ Geographic Coverage:")
coverage = conn.execute("""
    SELECT 
        STE_NAME21 as state,
        COUNT(*) as sa2_count,
        ROUND(AVG(AREASQKM21), 1) as avg_area_km2
    FROM sa2_boundaries 
    GROUP BY STE_NAME21 
    ORDER BY COUNT(*) DESC
""").fetchall()

for state_info in coverage:
    print(f"  • {state_info[0]}: {state_info[1]} SA2s (avg {state_info[2]} km²)")

# Top and bottom disadvantaged areas
print("\n🎯 Most Disadvantaged SA2 Areas (lowest IRSD scores):")
most_disadvantaged = conn.execute("""
    SELECT SA2_Name_2021, State_Name_2021, IRSD_Score, IRSD_Decile_Australia
    FROM seifa_2021 
    ORDER BY IRSD_Score ASC 
    LIMIT 5
""").fetchall()

for area in most_disadvantaged:
    print(f"  • {area[0]}, {area[1]} - Score: {area[2]:.1f} (Decile {int(area[3])})")

print("\n🌟 Least Disadvantaged SA2 Areas (highest IRSD scores):")
least_disadvantaged = conn.execute("""
    SELECT SA2_Name_2021, State_Name_2021, IRSD_Score, IRSD_Decile_Australia
    FROM seifa_2021 
    ORDER BY IRSD_Score DESC 
    LIMIT 5
""").fetchall()

for area in least_disadvantaged:
    print(f"  • {area[0]}, {area[1]} - Score: {area[2]:.1f} (Decile {int(area[3])})")

# Data quality check
print("\n✅ Data Quality Validation:")
quality_checks = conn.execute("""
    SELECT 
        'Boundaries with SEIFA data' as check_type,
        COUNT(*) as count
    FROM sa2_analysis 
    WHERE SA2_Code_2021 IS NOT NULL
    
    UNION ALL
    
    SELECT 
        'Boundaries without SEIFA data' as check_type,
        COUNT(*) as count
    FROM sa2_analysis 
    WHERE SA2_Code_2021 IS NULL
""").fetchall()

for check in quality_checks:
    print(f"  • {check[0]}: {check[1]:,}")

print("\n🎉 Database verification complete!")
print(f"📁 Database file: {db_path}")
print(f"🗺️ Map visualization: {db_path.parent / 'docs' / 'initial_map.html'}")

conn.close()