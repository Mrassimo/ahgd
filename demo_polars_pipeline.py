#!/usr/bin/env python3
"""
AHGD V3: Demo Polars Pipeline
Demonstrates the high-performance pipeline with mock Australian health data.
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import polars as pl
import pandas as pd
from src.storage.parquet_manager import ParquetStorageManager
from src.utils.logging import get_logger

logger = get_logger(__name__)

def create_mock_health_data() -> pl.DataFrame:
    """Create realistic mock Australian health data."""
    import random
    random.seed(42)  # Reproducible data
    
    print("🏗️ Generating mock Australian health data...")
    
    # SA1 codes for different states
    sa1_prefixes = {
        "NSW": ["101", "102", "103", "104", "105", "106", "107", "108", "109"],
        "VIC": ["201", "202", "203", "204", "205", "206", "207", "208", "209"],
        "QLD": ["301", "302", "303", "304", "305", "306", "307", "308", "309"],
        "WA": ["501", "502", "503", "504", "505", "506", "507", "508", "509"],
        "SA": ["401", "402", "403", "404", "405", "406"],
        "TAS": ["601", "602", "603", "604", "605"],
        "ACT": ["801", "802"],
        "NT": ["701", "702", "703"]
    }
    
    # Generate SA1 areas
    data = []
    record_id = 1
    
    for state, prefixes in sa1_prefixes.items():
        for prefix in prefixes:
            # Generate 20-50 SA1 areas per prefix
            num_areas = random.randint(20, 50)
            
            for i in range(num_areas):
                sa1_code = f"{prefix}{random.randint(10000, 99999):05d}"
                
                # Create realistic health indicators
                # Use state-based variations for realism
                base_diabetes = {
                    "NSW": 6.2, "VIC": 5.8, "QLD": 7.1, "WA": 6.0,
                    "SA": 6.5, "TAS": 7.2, "ACT": 5.1, "NT": 8.5
                }[state]
                
                base_life_exp = {
                    "NSW": 82.1, "VIC": 82.3, "QLD": 81.8, "WA": 82.0,
                    "SA": 81.5, "TAS": 81.2, "ACT": 83.2, "NT": 78.9
                }[state]
                
                # Add realistic variation
                diabetes_rate = max(2.0, base_diabetes + random.gauss(0, 1.5))
                life_expectancy = max(75.0, base_life_exp + random.gauss(0, 2.0))
                
                # Correlate socioeconomic disadvantage with health outcomes
                seifa_rank = random.randint(1, 1000)
                
                # Lower SEIFA = more disadvantaged = worse health outcomes
                if seifa_rank <= 200:  # Most disadvantaged
                    diabetes_rate += random.uniform(1.0, 3.0)
                    life_expectancy -= random.uniform(2.0, 5.0)
                elif seifa_rank >= 800:  # Least disadvantaged  
                    diabetes_rate -= random.uniform(0.5, 1.5)
                    life_expectancy += random.uniform(1.0, 3.0)
                
                record = {
                    "record_id": record_id,
                    "sa1_code": sa1_code,
                    "area_name": f"{state} Area {i+1:03d}",
                    "state": state,
                    "population": random.randint(300, 1200),
                    
                    # Health indicators
                    "diabetes_prevalence": round(max(2.0, diabetes_rate), 1),
                    "life_expectancy": round(min(95.0, life_expectancy), 1),
                    "obesity_rate": round(random.uniform(20.0, 40.0), 1),
                    "mental_health_score": round(random.uniform(60.0, 85.0), 1),
                    
                    # Healthcare utilization 
                    "gp_visits_per_capita": round(random.uniform(3.0, 12.0), 1),
                    "specialist_visits_per_capita": round(random.uniform(0.5, 4.0), 1),
                    "hospital_admissions_per_1000": round(random.uniform(80.0, 250.0), 1),
                    
                    # Socioeconomic indicators
                    "seifa_irsad_rank": seifa_rank,
                    "median_age": round(random.uniform(25.0, 50.0), 1),
                    "median_income": random.randint(35000, 120000),
                    
                    # Geographic data
                    "remoteness_category": random.choice([
                        "Major Cities", "Inner Regional", "Outer Regional", 
                        "Remote", "Very Remote"
                    ]),
                    
                    # Data quality
                    "extraction_date": datetime.now(),
                    "data_quality_score": round(random.uniform(0.8, 1.0), 2)
                }
                
                data.append(record)
                record_id += 1
    
    df = pl.DataFrame(data)
    print(f"✅ Generated {len(df):,} health records across {len(sa1_prefixes)} states/territories")
    return df

def run_polars_performance_demo():
    """Demonstrate Polars performance with Australian health data."""
    
    print("\n🚀 AHGD V3: High-Performance Polars Demo")
    print("=" * 60)
    
    # Generate mock data
    start_time = datetime.now()
    health_df = create_mock_health_data()
    generation_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n📊 Dataset Overview:")
    print(f"   Records: {len(health_df):,}")
    print(f"   Columns: {len(health_df.columns)}")
    print(f"   Generation time: {generation_time:.2f}s")
    print(f"   Memory usage: {health_df.estimated_size('mb'):.1f}MB")
    
    # Initialize Parquet storage
    parquet_manager = ParquetStorageManager("./data/demo_polars_cache")
    
    print(f"\n💾 Storing in Parquet format...")
    start_time = datetime.now()
    parquet_path = parquet_manager.store_processed_data(
        health_df, 
        "demo_health_data",
        partition_by_state=True
    )
    storage_time = (datetime.now() - start_time).total_seconds()
    
    print(f"✅ Stored to: {parquet_path}")
    print(f"   Storage time: {storage_time:.2f}s")
    print(f"   File size: {parquet_path.stat().st_size / (1024*1024):.1f}MB")
    
    # Demonstrate Polars performance
    print(f"\n⚡ Polars Performance Demonstrations:")
    
    # 1. State-level aggregations
    start_time = datetime.now()
    state_stats = health_df.group_by("state").agg([
        pl.col("diabetes_prevalence").mean().alias("avg_diabetes"),
        pl.col("life_expectancy").mean().alias("avg_life_expectancy"),
        pl.col("population").sum().alias("total_population"),
        pl.count().alias("sa1_areas")
    ]).sort("avg_diabetes", descending=True)
    agg_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n📈 State Health Rankings:")
    print(state_stats.to_pandas().to_string(index=False, float_format='%.1f'))
    print(f"   ⏱️ Aggregation time: {agg_time*1000:.1f}ms")
    
    # 2. High-risk area identification
    start_time = datetime.now()
    high_risk_areas = health_df.filter(
        (pl.col("diabetes_prevalence") > 8.0) &
        (pl.col("life_expectancy") < 80.0) &
        (pl.col("seifa_irsad_rank") < 300)
    ).select([
        "sa1_code", "area_name", "state", "diabetes_prevalence", 
        "life_expectancy", "seifa_irsad_rank"
    ]).sort("diabetes_prevalence", descending=True)
    filter_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n🚨 High-Risk Health Areas:")
    print(high_risk_areas.head(10).to_pandas().to_string(index=False, float_format='%.1f'))
    print(f"   Found {len(high_risk_areas)} high-risk areas")
    print(f"   ⏱️ Filter time: {filter_time*1000:.1f}ms")
    
    # 3. Healthcare utilization analysis
    start_time = datetime.now()
    healthcare_analysis = health_df.with_columns([
        (pl.col("gp_visits_per_capita") + pl.col("specialist_visits_per_capita"))
        .alias("total_visits_per_capita"),
        
        (pl.col("hospital_admissions_per_1000") > 200)
        .alias("high_hospital_use"),
        
        pl.when(pl.col("seifa_irsad_rank") <= 300)
        .then(pl.lit("Disadvantaged"))
        .when(pl.col("seifa_irsad_rank") >= 700)
        .then(pl.lit("Advantaged"))
        .otherwise(pl.lit("Middle"))
        .alias("socioeconomic_group")
    ])
    
    utilization_stats = healthcare_analysis.group_by("socioeconomic_group").agg([
        pl.col("total_visits_per_capita").mean().alias("avg_visits"),
        pl.col("high_hospital_use").sum().alias("high_hospital_areas"),
        pl.count().alias("total_areas")
    ])
    calc_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n🏥 Healthcare Utilization by Socioeconomic Group:")
    print(utilization_stats.to_pandas().to_string(index=False, float_format='%.1f'))
    print(f"   ⏱️ Calculation time: {calc_time*1000:.1f}ms")
    
    # 4. Performance comparison with pandas
    print(f"\n🏆 Polars vs Pandas Performance Comparison:")
    
    # Convert to pandas for comparison
    pandas_df = health_df.to_pandas()
    
    # Polars aggregation
    start_time = datetime.now()
    polars_result = health_df.group_by("state").agg([
        pl.col("diabetes_prevalence").mean(),
        pl.col("life_expectancy").mean(),
        pl.col("population").sum()
    ])
    polars_time = (datetime.now() - start_time).total_seconds()
    
    # Pandas aggregation (equivalent)
    start_time = datetime.now()
    pandas_result = pandas_df.groupby("state").agg({
        "diabetes_prevalence": "mean",
        "life_expectancy": "mean", 
        "population": "sum"
    })
    pandas_time = (datetime.now() - start_time).total_seconds()
    
    speedup = pandas_time / polars_time
    
    print(f"   Polars time: {polars_time*1000:.1f}ms")
    print(f"   Pandas time: {pandas_time*1000:.1f}ms")
    print(f"   🚀 Speedup: {speedup:.1f}x faster with Polars")
    
    print(f"\n🎯 Demo Summary:")
    print(f"   ✅ Generated realistic Australian health data")
    print(f"   ✅ Demonstrated Parquet storage optimization")
    print(f"   ✅ Showed complex health analytics queries")
    print(f"   ✅ Confirmed {speedup:.1f}x performance improvement")
    print(f"   ✅ Ready for real government data integration")
    
    return health_df, parquet_path

if __name__ == "__main__":
    try:
        demo_df, demo_path = run_polars_performance_demo()
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"   Demo data available at: {demo_path}")
        print(f"   Records processed: {len(demo_df):,}")
        print(f"\nNext steps:")
        print(f"   • Replace mock data with real ABS/AIHW sources")
        print(f"   • Integrate with SA1 geographic boundaries") 
        print(f"   • Connect to live government APIs")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)