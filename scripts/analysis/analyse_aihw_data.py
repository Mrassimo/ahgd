#!/usr/bin/env python3
"""
Quick analysis of extracted AIHW data to understand structure and content
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyse_mort_data():
    """Analyse AIHW MORT table data"""
    logger.info("Analysing AIHW MORT data...")
    
    processed_dir = Path("data/processed")
    mort_file = processed_dir / "aihw_mort_table1.parquet"
    
    if not mort_file.exists():
        logger.error("MORT data file not found")
        return
    
    df = pd.read_parquet(mort_file)
    
    logger.info(f"MORT Data Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Geographic coverage
    logger.info(f"Geography types: {df['geography'].value_counts()}")
    
    # Sample records
    logger.info("Sample records:")
    print(df.head())
    
    # Check for SA2/SA3/SA4 data
    geographic_areas = df['geography'].unique()
    sa2_areas = [g for g in geographic_areas if 'SA2' in str(g)]
    sa3_areas = [g for g in geographic_areas if 'SA3' in str(g)]
    sa4_areas = [g for g in geographic_areas if 'SA4' in str(g)]
    
    logger.info(f"SA2 areas found: {len(sa2_areas)}")
    logger.info(f"SA3 areas found: {len(sa3_areas)}")
    logger.info(f"SA4 areas found: {len(sa4_areas)}")
    
    if sa2_areas:
        logger.info(f"Sample SA2 areas: {sa2_areas[:5]}")

def analyse_grim_data():
    """Analyse AIHW GRIM data"""
    logger.info("Analysing AIHW GRIM data...")
    
    processed_dir = Path("data/processed")
    grim_file = processed_dir / "aihw_grim_data.parquet"
    
    if not grim_file.exists():
        logger.error("GRIM data file not found")
        return
    
    df = pd.read_parquet(grim_file)
    
    logger.info(f"GRIM Data Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Year coverage
    logger.info(f"Year range: {df['year'].min()} - {df['year'].max()}")
    
    # Causes of death
    logger.info(f"Causes of death: {df['cause_of_death'].unique()[:10]}")
    
    # Sample chronic disease records
    chronic_causes = df[df['cause_of_death'].str.contains('diabetes|heart|cancer|mental', case=False, na=False)]
    logger.info(f"Chronic disease records: {len(chronic_causes)}")
    
    if len(chronic_causes) > 0:
        logger.info("Sample chronic disease data:")
        print(chronic_causes.head())

def analyse_phidu_data():
    """Analyse PHIDU data"""
    logger.info("Analysing PHIDU data...")
    
    processed_dir = Path("data/processed")
    lga_file = processed_dir / "phidu_lga_data.parquet"
    
    if not lga_file.exists():
        logger.error("PHIDU LGA data file not found")
        return
    
    df = pd.read_parquet(lga_file)
    
    logger.info(f"PHIDU LGA Data Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # This might be just the header info - check content
    logger.info("Sample data:")
    print(df.head())

def main():
    """Run analysis of all extracted AIHW data"""
    logger.info("Starting analysis of extracted AIHW data...")
    
    analyse_mort_data()
    print("\n" + "="*50 + "\n")
    
    analyse_grim_data()
    print("\n" + "="*50 + "\n")
    
    analyse_phidu_data()
    
    logger.info("Analysis completed")

if __name__ == "__main__":
    main()