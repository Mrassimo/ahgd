"""Time dimension generation module for the AHGD ETL pipeline.

This module creates a time dimension table with various date attributes
needed for temporal analysis in the AHGD data warehouse.
"""

import logging
import polars as pl
from datetime import datetime, date, timedelta
from typing import List, Dict, Any

from . import config

logger = logging.getLogger('ahgd_etl')

def create_time_dimension(start_year: int, end_year: int) -> pl.DataFrame:
    """Creates a time dimension with a row for each day between start_year and end_year.
    
    Args:
        start_year (int): The starting year for the time dimension.
        end_year (int): The ending year for the time dimension (inclusive).
        
    Returns:
        pl.DataFrame: A DataFrame containing the time dimension with all attributes.
    """
    logger.info(f"Creating time dimension from {start_year} to {end_year}")
    
    # Generate a sequence of dates using Python's date range
    start_date = date(start_year, 1, 1)
    end_date = date(end_year, 12, 31)
    delta = end_date - start_date
    
    # Create a list of dates
    date_list = [start_date + timedelta(days=i) for i in range(delta.days + 1)]
    
    # Create the initial DataFrame with the date list
    dates_df = pl.DataFrame({
        'full_date': date_list
    })
    
    # First, add all date components
    time_df = dates_df.with_columns([
        # time_sk as YYYYMMDD integer format
        pl.col("full_date").dt.strftime("%Y%m%d").cast(pl.Int64).alias("time_sk"),
        
        # Extract year, month, day components
        pl.col("full_date").dt.year().cast(pl.Int32).alias("year"),
        pl.col("full_date").dt.quarter().cast(pl.Int32).alias("quarter"),
        pl.col("full_date").dt.month().cast(pl.Int32).alias("month"),
        pl.col("full_date").dt.day().cast(pl.Int32).alias("day_of_month"),
        pl.col("full_date").dt.weekday().cast(pl.Int32).alias("day_of_week"),
        
        # Use strftime to get correct day and month names
        pl.col("full_date").dt.strftime("%A").alias("day_name"),
        
        # Add timestamp for data lineage tracking
        pl.lit(datetime.now()).alias("etl_processed_at")
    ])
    
    # Month name mapping in a separate step using when-then expressions
    time_df = time_df.with_columns([
        pl.when(pl.col("month") == 1).then(pl.lit("January"))
        .when(pl.col("month") == 2).then(pl.lit("February"))
        .when(pl.col("month") == 3).then(pl.lit("March"))
        .when(pl.col("month") == 4).then(pl.lit("April"))
        .when(pl.col("month") == 5).then(pl.lit("May"))
        .when(pl.col("month") == 6).then(pl.lit("June"))
        .when(pl.col("month") == 7).then(pl.lit("July"))
        .when(pl.col("month") == 8).then(pl.lit("August"))
        .when(pl.col("month") == 9).then(pl.lit("September"))
        .when(pl.col("month") == 10).then(pl.lit("October"))
        .when(pl.col("month") == 11).then(pl.lit("November"))
        .otherwise(pl.lit("December"))
        .alias("month_name")
    ])
    
    # Add weekday flag in a separate step
    time_df = time_df.with_columns([
        pl.col("day_name").is_in(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
        .cast(pl.Boolean).alias("is_weekday")
    ])
    
    # Add fiscal year calculation (July-June format for Australia: 2021-07-01 is in FY 2021/22)
    time_df = time_df.with_columns([
        pl.when(pl.col("month") >= 7)
        .then(pl.format("{}/{}",
                       pl.col("year"),
                       (pl.col("year") + 1) % 100))
        .otherwise(pl.format("{}/{}",
                            pl.col("year") - 1,
                            pl.col("year") % 100))
        .alias("financial_year")
    ])
    
    # Add Census year flag (assuming Census years are 2011, 2016, 2021, etc. with 5-year intervals)
    census_years = [2011, 2016, 2021, 2026, 2031, 2036]
    time_df = time_df.with_columns([
        pl.col("year").is_in(census_years).cast(pl.Boolean).alias("is_census_year")
    ])
    
    # Use the schema from config for column ordering and types
    # This ensures consistency with the data dictionary
    schema_columns = list(config.TIME_DIMENSION_SCHEMA.keys())
    
    # Cast all columns to match the expected schema types
    # Extract schema as dictionary mapping column names to types
    schema_types = {name: dtype for name, dtype in config.TIME_DIMENSION_SCHEMA.items()}
    
    # Apply the schema by selecting and casting columns
    time_df = time_df.select([
        pl.col(col).cast(schema_types[col]) if col in time_df.columns else pl.lit(None).cast(schema_types[col]).alias(col)
        for col in schema_columns
    ])
    
    logger.info(f"Created time dimension with {len(time_df)} rows")
    return time_df

def generate_time_dimension(output_dir, start_year=2011, end_year=2031):
    """Generates and saves the time dimension to a Parquet file.
    
    Args:
        output_dir: Path where the time dimension will be saved
        start_year: Starting year (default: 2011)
        end_year: Ending year (default: 2031)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create the time dimension DataFrame
        time_df = create_time_dimension(start_year, end_year)
        
        # Write to Parquet file
        output_file = output_dir / "dim_time.parquet"
        time_df.write_parquet(output_file)
        logger.info(f"Successfully wrote time dimension to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error generating time dimension: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage when run directly
    import sys
    from pathlib import Path
    from etl_logic import utils, config
    
    # Set up logging
    logger = utils.setup_logging(config.PATHS.get('LOG_DIR'))
    
    # Default years
    start_year = 2011
    end_year = 2031
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        start_year = int(sys.argv[1])
    if len(sys.argv) > 2:
        end_year = int(sys.argv[2])
    
    # Generate time dimension
    output_dir = config.PATHS['OUTPUT_DIR']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success = generate_time_dimension(output_dir, start_year, end_year)
    
    if success:
        print(f"Time dimension successfully created for years {start_year}-{end_year}")
        sys.exit(0)
    else:
        print("Failed to create time dimension")
        sys.exit(1) 