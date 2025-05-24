"""
Utility functions for Census transformers.

This module contains helper functions for Census data processing.
"""

import logging
from typing import List, Dict, Optional, Any, Union

import polars as pl

def clean_polars_geo_code(col: pl.Expr) -> pl.Expr:
    """
    Clean a geographic code column to ensure consistent format.
    
    Args:
        col: Polars column expression for the geographic code
        
    Returns:
        Cleaned geographic code column expression
    """
    return col.cast(pl.Utf8).str.strip()

def safe_polars_int(col: pl.Expr) -> pl.Expr:
    """
    Safely convert values to integers, handling NULL values and non-numeric strings.
    
    Args:
        col: Polars column expression to convert
        
    Returns:
        Integer column expression
    """
    return (
        pl.when(col.is_null())
        .then(None)
        .otherwise(
            pl.when(col.cast(pl.Utf8).str.contains(r'^-?\d+$'))
            .then(col.cast(pl.Int64))
            .otherwise(None)
        )
    )

def find_geo_column(df: pl.DataFrame, geo_column_options: List[str]) -> Optional[str]:
    """
    Find the geographic code column in a DataFrame.
    
    Args:
        df: DataFrame to search
        geo_column_options: List of possible geographic column names
        
    Returns:
        Name of the geographic column if found, None otherwise
    """
    for col_option in geo_column_options:
        if col_option in df.columns:
            return col_option
    
    return None