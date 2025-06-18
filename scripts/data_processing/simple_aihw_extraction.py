#!/usr/bin/env python3
"""
Simplified AIHW data extraction focusing on successful data sources
"""

import pandas as pd
import sqlite3
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyse_mort_geographic_coverage():
    """Analyse geographic coverage in MORT data"""
    logger.info("Analysing MORT data geographic coverage...")
    
    processed_dir = Path("data/processed")
    mort_file = processed_dir / "aihw_mort_table1.parquet"
    
    if not mort_file.exists():
        logger.error("MORT data file not found")
        return
    
    df = pd.read_parquet(mort_file)
    
    # Look at geography column in detail
    unique_geogs = df['geography'].unique()
    
    # Group by potential geographic types
    logger.info(f"Total unique geographic areas: {len(unique_geogs)}")
    
    # Find patterns
    states = [g for g in unique_geogs if g in ['New South Wales', 'Victoria', 'Queensland', 'South Australia', 'Western Australia', 'Tasmania', 'Northern Territory', 'Australian Capital Territory']]
    cities = [g for g in unique_geogs if any(city in g for city in ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide', 'Canberra', 'Darwin', 'Hobart'])]
    
    logger.info(f"States/Territories: {len(states)}")
    logger.info(f"Cities: {len(cities)}")
    
    # Sample some non-state areas
    other_areas = [g for g in unique_geogs if g not in states and g not in ['Australia (total)', 'Unknown/missing']]
    logger.info(f"Other areas (LGA/regional): {len(other_areas)}")
    logger.info(f"Sample other areas: {other_areas[:10]}")
    
    # Check for any SA2, SA3, SA4 codes
    sa_areas = [g for g in unique_geogs if 'SA' in str(g) and any(digit in str(g) for digit in '234')]
    logger.info(f"Statistical Areas found: {len(sa_areas)}")
    if sa_areas:
        logger.info(f"SA areas: {sa_areas}")
    
    # Look at years and categories
    logger.info(f"Years: {sorted(df['YEAR'].unique())}")
    logger.info(f"Categories: {df['category'].unique()}")
    
    # Examine a sample record
    sample = df[df['geography'] == 'Australia (total)'].head(1)
    logger.info("Sample record (Australia total):")
    for col in df.columns:
        logger.info(f"  {col}: {sample[col].iloc[0]}")

def analyse_grim_chronic_diseases():
    """Analyse chronic disease data in GRIM dataset"""
    logger.info("Analysing chronic diseases in GRIM data...")
    
    processed_dir = Path("data/processed") 
    grim_file = processed_dir / "aihw_grim_data.parquet"
    
    if not grim_file.exists():
        logger.error("GRIM data file not found")
        return
    
    df = pd.read_parquet(grim_file)
    
    # Focus on recent years and chronic diseases
    recent_df = df[df['year'] >= 2015]
    
    chronic_keywords = ['diabetes', 'heart', 'cancer', 'mental', 'chronic', 'cardiovascular']
    chronic_df = df[df['cause_of_death'].str.contains('|'.join(chronic_keywords), case=False, na=False)]
    
    logger.info(f"Total GRIM records: {len(df)}")
    logger.info(f"Recent years (2015+): {len(recent_df)}")
    logger.info(f"Chronic disease records: {len(chronic_df)}")
    
    # Show chronic disease causes
    chronic_causes = chronic_df['cause_of_death'].unique()
    logger.info(f"Chronic disease causes found: {len(chronic_causes)}")
    for cause in chronic_causes:
        logger.info(f"  - {cause}")
    
    # Show recent data sample
    diabetes_recent = chronic_df[(chronic_df['cause_of_death'].str.contains('diabetes', case=False)) & 
                                (chronic_df['year'] >= 2020)]
    if len(diabetes_recent) > 0:
        logger.info("Sample diabetes data (2020+):")
        print(diabetes_recent.head())

def create_summary_tables():
    """Create summary tables in the database"""
    logger.info("Creating summary tables in database...")
    
    db_path = Path("health_analytics.db")
    processed_dir = Path("data/processed")
    
    with sqlite3.connect(db_path) as conn:
        # Load MORT data
        mort_file = processed_dir / "aihw_mort_table1.parquet"
        if mort_file.exists():
            mort_df = pd.read_parquet(mort_file)
            
            # Clean and standardise the data
            mort_clean = mort_df.copy()
            mort_clean['extraction_date'] = '2025-06-17'
            mort_clean['data_source'] = 'AIHW_MORT_2025'
            
            # Store in database
            mort_clean.to_sql('aihw_mort_raw', conn, if_exists='replace', index=False)
            logger.info(f"Stored {len(mort_clean)} MORT records in database")
        
        # Load GRIM data
        grim_file = processed_dir / "aihw_grim_data.parquet"
        if grim_file.exists():
            grim_df = pd.read_parquet(grim_file)
            
            # Filter for chronic diseases and recent years
            chronic_keywords = ['diabetes', 'heart', 'cancer', 'mental', 'chronic', 'cardiovascular', 'respiratory', 'kidney']
            chronic_grim = grim_df[
                (grim_df['cause_of_death'].str.contains('|'.join(chronic_keywords), case=False, na=False)) &
                (grim_df['year'] >= 2000)
            ].copy()
            
            chronic_grim['extraction_date'] = '2025-06-17'
            chronic_grim['data_source'] = 'AIHW_GRIM_2025'
            
            # Store in database
            chronic_grim.to_sql('aihw_grim_chronic', conn, if_exists='replace', index=False)
            logger.info(f"Stored {len(chronic_grim)} chronic disease GRIM records in database")
        
        # Create summary view
        conn.execute('''
            CREATE VIEW IF NOT EXISTS health_indicators_summary AS
            SELECT 
                'MORT' as data_source,
                geography,
                YEAR as year,
                'All Causes' as indicator,
                deaths as value,
                'Count' as unit
            FROM aihw_mort_raw
            WHERE deaths IS NOT NULL
            
            UNION ALL
            
            SELECT 
                'GRIM' as data_source,
                'Australia' as geography,
                year,
                cause_of_death as indicator,
                deaths as value,
                'Count' as unit
            FROM aihw_grim_chronic
            WHERE deaths IS NOT NULL
        ''')
        
        conn.commit()
        logger.info("Created health_indicators_summary view")

def main():
    """Run simplified AIHW data analysis and database creation"""
    logger.info("Starting simplified AIHW data analysis...")
    
    analyse_mort_geographic_coverage()
    print("\n" + "="*50 + "\n")
    
    analyse_grim_chronic_diseases()
    print("\n" + "="*50 + "\n")
    
    create_summary_tables()
    
    logger.info("Analysis and database creation completed")

if __name__ == "__main__":
    main()