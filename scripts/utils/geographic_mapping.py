#!/usr/bin/env python3
"""
Geographic Mapping Functions for Health Data Integration

This module provides critical functions for mapping postcodes to SA2 areas,
which is essential for integrating health data with Australian Bureau of Statistics
geographic boundaries.

Functions:
    - postcode_to_sa2(postcode): Returns SA2 codes with weights for a given postcode
    - aggregate_postcode_data_to_sa2(df): Aggregates postcode data to SA2 level
    - Validation functions to check mapping quality
"""

import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import sys

# Add src to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import get_global_config

# Get configuration
config = get_global_config()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PostcodeToSA2Mapper:
    """
    A class to handle postcode to SA2 mapping using ABS correspondence data.
    
    This class provides robust mapping functionality with population-weighted 
    allocation for cases where postcodes span multiple SA2 areas.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the mapper with data directory path.
        
        Args:
            data_dir: Path to the data directory containing the database (optional, uses config if not provided)
        """
        if data_dir is None:
            # Use configuration to get data directory
            self.data_dir = config.data_source.raw_data_dir.parent  # Go up from raw to data
        else:
            self.data_dir = Path(data_dir)
        self.db_path = config.database.path
        self.raw_geographic_dir = config.data_source.raw_data_dir / "geographic"
        self._correspondence_loaded = False
        
    def load_correspondence_data(self) -> None:
        """
        Load postcode to SA2 correspondence data into DuckDB database.
        
        This method reads the ABS correspondence file and creates a table
        in the database with proper indexing for fast lookups.
        """
        correspondence_file = self.raw_geographic_dir / "CG_POA_2021_SA2_2021.xlsx"
        
        if not correspondence_file.exists():
            raise FileNotFoundError(
                f"Correspondence file not found: {correspondence_file}. "
                "Please ensure the ABS correspondence files have been downloaded."
            )
        
        logger.info("Loading postcode to SA2 correspondence data...")
        
        # Read the correspondence file
        df = pd.read_excel(correspondence_file)
        
        # Clean and prepare the data
        df = df.rename(columns={
            'POA_CODE_2021': 'postcode',
            'SA2_CODE_2021': 'sa2_code',
            'SA2_NAME_2021': 'sa2_name',
            'RATIO_FROM_TO': 'weight',
            'OVERALL_QUALITY_INDICATOR': 'quality'
        })
        
        # Select only the columns we need
        df = df[['postcode', 'sa2_code', 'sa2_name', 'weight', 'quality']].copy()
        
        # Convert postcode to string and pad with zeros to ensure 4 digits
        df['postcode'] = df['postcode'].astype(str).str.zfill(4)
        
        # Convert SA2 code to string
        df['sa2_code'] = df['sa2_code'].astype(str)
        
        # Handle missing weights (set to 1.0 for complete mappings)
        df['weight'] = df['weight'].fillna(1.0)
        
        # Store the data in DuckDB
        with duckdb.connect(str(self.db_path)) as conn:
            # Create the table
            conn.execute("DROP TABLE IF EXISTS postcode_sa2_mapping")
            conn.execute("""
                CREATE TABLE postcode_sa2_mapping (
                    postcode VARCHAR,
                    sa2_code VARCHAR,
                    sa2_name VARCHAR,
                    weight DOUBLE,
                    quality VARCHAR,
                    PRIMARY KEY (postcode, sa2_code)
                )
            """)
            
            # Insert the data
            conn.execute("INSERT INTO postcode_sa2_mapping SELECT * FROM df")
            
            # Create indexes for fast lookups
            conn.execute("CREATE INDEX IF NOT EXISTS idx_postcode ON postcode_sa2_mapping(postcode)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sa2_code ON postcode_sa2_mapping(sa2_code)")
            
        self._correspondence_loaded = True
        logger.info(f"Loaded {len(df)} postcode-SA2 mapping records into database")
        
        # Log summary statistics
        unique_postcodes = df['postcode'].nunique()
        unique_sa2s = df['sa2_code'].nunique()
        many_to_many = len(df) > max(unique_postcodes, unique_sa2s)
        
        logger.info(f"Summary: {unique_postcodes} unique postcodes, {unique_sa2s} unique SA2s")
        if many_to_many:
            logger.info("Note: Many-to-many relationships detected (expected for postcode boundaries)")
    
    def postcode_to_sa2(self, postcode: Union[str, int]) -> List[Dict[str, Union[str, float]]]:
        """
        Map a postcode to SA2 codes with weights.
        
        Args:
            postcode: The postcode to map (as string or integer)
            
        Returns:
            List of dictionaries containing SA2 codes, names, and weights
            
        Example:
            >>> mapper = PostcodeToSA2Mapper()
            >>> result = mapper.postcode_to_sa2("2000")
            >>> print(result)
            [{'sa2_code': '117011215', 'sa2_name': 'Sydney', 'weight': 1.0, 'quality': 'Good'}]
        """
        if not self._correspondence_loaded:
            self.load_correspondence_data()
            
        # Ensure postcode is a 4-digit string
        postcode_str = str(postcode).zfill(4)
        
        with duckdb.connect(str(self.db_path)) as conn:
            result = conn.execute("""
                SELECT sa2_code, sa2_name, weight, quality
                FROM postcode_sa2_mapping
                WHERE postcode = ?
                ORDER BY weight DESC
            """, [postcode_str]).fetchall()
            
        if not result:
            logger.warning(f"No SA2 mapping found for postcode {postcode_str}")
            return []
            
        return [
            {
                'sa2_code': row[0],
                'sa2_name': row[1],
                'weight': row[2],
                'quality': row[3]
            }
            for row in result
        ]
    
    def aggregate_postcode_data_to_sa2(self, 
                                      df: pd.DataFrame, 
                                      postcode_col: str = 'postcode',
                                      value_cols: List[str] = None,
                                      method: str = 'weighted_sum') -> pd.DataFrame:
        """
        Aggregate postcode-level data to SA2 level using correspondence weights.
        
        Args:
            df: DataFrame with postcode-level data
            postcode_col: Name of the postcode column
            value_cols: List of columns to aggregate (if None, aggregate all numeric columns)
            method: Aggregation method ('weighted_sum', 'weighted_mean', or 'sum')
            
        Returns:
            DataFrame aggregated to SA2 level
        """
        if not self._correspondence_loaded:
            self.load_correspondence_data()
            
        # Ensure postcode is string and padded
        df = df.copy()
        df[postcode_col] = df[postcode_col].astype(str).str.zfill(4)
        
        # If value_cols not specified, use all numeric columns except postcode
        if value_cols is None:
            value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if postcode_col in value_cols:
                value_cols.remove(postcode_col)
        
        # Load correspondence data
        with duckdb.connect(str(self.db_path)) as conn:
            correspondence = conn.execute("""
                SELECT postcode, sa2_code, sa2_name, weight
                FROM postcode_sa2_mapping
            """).df()
        
        # Merge with correspondence data
        merged = df.merge(correspondence, left_on=postcode_col, right_on='postcode', how='inner')
        
        if len(merged) == 0:
            logger.warning("No postcodes could be mapped to SA2s")
            return pd.DataFrame()
        
        # Apply weights to value columns
        for col in value_cols:
            if col in merged.columns:
                merged[f'{col}_weighted'] = merged[col] * merged['weight']
        
        # Perform aggregation using a simpler approach
        grouped = merged.groupby('sa2_code')
        
        # Start with SA2 name
        result = grouped['sa2_name'].first().reset_index()
        
        # Add aggregated value columns
        if method in ['weighted_sum', 'sum']:
            for col in value_cols:
                if col in merged.columns:
                    if method == 'weighted_sum' and f'{col}_weighted' in merged.columns:
                        result[col] = grouped[f'{col}_weighted'].sum().values
                    elif col in merged.columns:
                        result[col] = grouped[col].sum().values
        elif method == 'weighted_mean':
            for col in value_cols:
                if col in merged.columns and f'{col}_weighted' in merged.columns:
                    sum_weighted = grouped[f'{col}_weighted'].sum()
                    sum_weights = grouped['weight'].sum()
                    result[col] = (sum_weighted / sum_weights).values
        
        
        logger.info(f"Aggregated {len(df)} postcode records to {len(result)} SA2 areas")
        
        return result
    
    def validate_mapping_coverage(self, postcodes: List[Union[str, int]]) -> Dict[str, Union[int, float, List]]:
        """
        Validate mapping coverage for a list of postcodes.
        
        Args:
            postcodes: List of postcodes to validate
            
        Returns:
            Dictionary with coverage statistics and unmapped postcodes
        """
        if not self._correspondence_loaded:
            self.load_correspondence_data()
            
        # Ensure postcodes are 4-digit strings
        postcodes_clean = [str(pc).zfill(4) for pc in postcodes]
        
        with duckdb.connect(str(self.db_path)) as conn:
            # Get all postcodes that can be mapped
            mapped_postcodes = conn.execute("""
                SELECT DISTINCT postcode
                FROM postcode_sa2_mapping
                WHERE postcode IN ({})
            """.format(','.join(f"'{pc}'" for pc in postcodes_clean))).fetchall()
            
        mapped_postcodes = [row[0] for row in mapped_postcodes]
        unmapped_postcodes = [pc for pc in postcodes_clean if pc not in mapped_postcodes]
        
        coverage_stats = {
            'total_postcodes': len(postcodes_clean),
            'mapped_postcodes': len(mapped_postcodes),
            'unmapped_postcodes': len(unmapped_postcodes),
            'coverage_percentage': (len(mapped_postcodes) / len(postcodes_clean)) * 100 if postcodes_clean else 0,
            'unmapped_postcode_list': unmapped_postcodes
        }
        
        return coverage_stats
    
    def get_mapping_quality_summary(self) -> pd.DataFrame:
        """
        Get a summary of mapping quality indicators.
        
        Returns:
            DataFrame with quality summary statistics
        """
        if not self._correspondence_loaded:
            self.load_correspondence_data()
            
        with duckdb.connect(str(self.db_path)) as conn:
            quality_summary = conn.execute("""
                SELECT 
                    quality,
                    COUNT(*) as mapping_count,
                    COUNT(DISTINCT postcode) as unique_postcodes,
                    COUNT(DISTINCT sa2_code) as unique_sa2s,
                    AVG(weight) as avg_weight,
                    MIN(weight) as min_weight,
                    MAX(weight) as max_weight
                FROM postcode_sa2_mapping
                GROUP BY quality
                ORDER BY mapping_count DESC
            """).df()
            
        return quality_summary


# Convenience functions for direct use
def postcode_to_sa2(postcode: Union[str, int], data_dir: str = None) -> List[Dict[str, Union[str, float]]]:
    """
    Quick function to map a postcode to SA2 codes with weights.
    
    Args:
        postcode: The postcode to map
        data_dir: Path to data directory
        
    Returns:
        List of SA2 mappings with weights
    """
    mapper = PostcodeToSA2Mapper(data_dir)
    return mapper.postcode_to_sa2(postcode)


def aggregate_postcode_data_to_sa2(df: pd.DataFrame, 
                                  postcode_col: str = 'postcode',
                                  value_cols: List[str] = None,
                                  method: str = 'weighted_sum',
                                  data_dir: str = None) -> pd.DataFrame:
    """
    Quick function to aggregate postcode data to SA2 level.
    
    Args:
        df: DataFrame with postcode data
        postcode_col: Name of postcode column
        value_cols: Columns to aggregate
        method: Aggregation method
        data_dir: Path to data directory
        
    Returns:
        DataFrame aggregated to SA2 level
    """
    mapper = PostcodeToSA2Mapper(data_dir)
    return mapper.aggregate_postcode_data_to_sa2(df, postcode_col, value_cols, method)


if __name__ == "__main__":
    # Example usage and testing
    mapper = PostcodeToSA2Mapper()
    
    # Load the correspondence data
    mapper.load_correspondence_data()
    
    # Test with some common postcodes
    test_postcodes = ["2000", "3000", "4000", "5000", "6000"]
    
    print("Testing postcode to SA2 mapping:")
    for postcode in test_postcodes:
        result = mapper.postcode_to_sa2(postcode)
        print(f"Postcode {postcode}: {len(result)} SA2 mappings")
        for mapping in result[:2]:  # Show first 2 mappings
            print(f"  - {mapping['sa2_code']} ({mapping['sa2_name']}) - Weight: {mapping['weight']}")
    
    # Validation example
    print("\nMapping coverage validation:")
    coverage = mapper.validate_mapping_coverage(test_postcodes)
    print(f"Coverage: {coverage['coverage_percentage']:.1f}% ({coverage['mapped_postcodes']}/{coverage['total_postcodes']})")
    
    # Quality summary
    print("\nMapping quality summary:")
    quality_summary = mapper.get_mapping_quality_summary()
    print(quality_summary)