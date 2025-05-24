"""
Time dimension generation module for the AHGD ETL pipeline.

This module creates a time dimension table with various date attributes
needed for temporal analysis in the AHGD data warehouse.
"""

import logging
import polars as pl
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..config import settings

logger = logging.getLogger('ahgd_etl.models.time_dimension')

class TimeDimensionModel:
    """
    Time dimension model class.
    
    This class provides methods for creating and managing the time dimension
    for the AHGD data warehouse.
    """
    
    def __init__(self):
        """Initialize the TimeDimensionModel."""
        self.logger = logger
        
        # Get the schema from settings
        self.schema = settings.get_schema('dim_time')
        if not self.schema:
            # Default schema if not in YAML
            self.schema = {
                'time_sk': pl.Int64,
                'full_date': pl.Date,
                'year': pl.Int32,
                'quarter': pl.Int32,
                'month': pl.Int32,
                'month_name': pl.Utf8,
                'day_of_month': pl.Int32,
                'day_of_week': pl.Int32,
                'day_name': pl.Utf8,
                'is_weekday': pl.Boolean,
                'financial_year': pl.Utf8,
                'is_census_year': pl.Boolean,
                'etl_processed_at': pl.Datetime
            }
    
    def create_dimension(self, start_year: int, end_year: int) -> pl.DataFrame:
        """
        Create time dimension with a row for each day between start_year and end_year.
        
        Args:
            start_year: The starting year for the time dimension
            end_year: The ending year for the time dimension (inclusive)
            
        Returns:
            DataFrame containing the time dimension with all attributes
        """
        self.logger.info(f"Creating time dimension from {start_year} to {end_year}")
        
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
            
            # Use strftime to get correct day names
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
        
        # Add Census year flag (assuming Census years are 2011, 2016, 2021, 2026, etc. with 5-year intervals)
        census_years = [2011, 2016, 2021, 2026, 2031, 2036]
        time_df = time_df.with_columns([
            pl.col("year").is_in(census_years).cast(pl.Boolean).alias("is_census_year")
        ])
        
        # Use the schema from settings for column ordering and types
        schema_columns = list(self.schema.keys())
        
        # Apply the schema by selecting and casting columns
        time_df = time_df.select([
            pl.col(col).cast(self.schema[col]) if col in time_df.columns 
            else pl.lit(None).cast(self.schema[col]).alias(col)
            for col in schema_columns if col in self.schema
        ])
        
        self.logger.info(f"Created time dimension with {len(time_df)} rows")
        return time_df
    
    def save_dimension(self, time_df: pl.DataFrame, output_path: Path) -> bool:
        """
        Save time dimension to a Parquet file.
        
        Args:
            time_df: Time dimension DataFrame
            output_path: Path to save the Parquet file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Write to Parquet file
            time_df.write_parquet(output_path)
            self.logger.info(f"Successfully wrote time dimension to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving time dimension: {e}")
            return False
    
    def get_time_sk_for_date(self, date_value: date, time_dim_path: Path) -> Optional[int]:
        """
        Get the time dimension surrogate key for a given date.
        
        Args:
            date_value: Date to find surrogate key for
            time_dim_path: Path to the time dimension Parquet file
            
        Returns:
            Surrogate key or None if not found
        """
        try:
            if not Path(time_dim_path).exists():
                self.logger.error(f"Time dimension not found at {time_dim_path}")
                return None
                
            # Read the time dimension Parquet file
            time_dim = pl.read_parquet(time_dim_path)
            
            if not isinstance(date_value, str):
                date_str = date_value.isoformat()
            else:
                date_str = date_value
                
            # Find the matching date and return the surrogate key
            matching_row = time_dim.filter(pl.col("full_date").cast(pl.Utf8).str.contains(date_str))
            
            if len(matching_row) > 0:
                return matching_row["time_sk"][0]
            else:
                # If no exact match, try to find the closest date
                # For Census 2021, use an approximate date if needed (August 10, 2021)
                if "2021" in date_str:
                    census_rows = time_dim.filter(pl.col("full_date").dt.year() == 2021)
                    if len(census_rows) > 0:
                        # Get August 2021 or the closest available date
                        august_rows = census_rows.filter(pl.col("full_date").dt.month() == 8)
                        if len(august_rows) > 0:
                            return august_rows["time_sk"][0]
                        else:
                            return census_rows["time_sk"][0]
                
                self.logger.warning(f"No matching date found in time dimension for {date_str}")
                return None
        except Exception as e:
            self.logger.error(f"Exception in get_time_sk_for_date: {e}")
            return None


def create_time_dimension(start_year: int, end_year: int) -> pl.DataFrame:
    """
    Create a time dimension with a row for each day between start_year and end_year.
    
    This function is a wrapper around the TimeDimensionModel.create_dimension method
    for backward compatibility.
    
    Args:
        start_year: The starting year for the time dimension
        end_year: The ending year for the time dimension (inclusive)
        
    Returns:
        DataFrame containing the time dimension with all attributes
    """
    # Create TimeDimensionModel instance
    model = TimeDimensionModel()
    
    # Create and return time dimension
    return model.create_dimension(start_year, end_year)


def generate_time_dimension(output_path: Path, start_year: int = 2011, end_year: int = 2031) -> bool:
    """
    Generate and save the time dimension to a Parquet file.
    
    Args:
        output_path: Path where the time dimension will be saved
        start_year: Starting year (default: 2011)
        end_year: Ending year (default: 2031)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create TimeDimensionModel instance
        model = TimeDimensionModel()
        
        # Create the time dimension DataFrame
        time_df = model.create_dimension(start_year, end_year)
        
        # Save the time dimension
        return model.save_dimension(time_df, output_path)
    except Exception as e:
        logger.error(f"Error generating time dimension: {e}")
        return False


def get_time_dimension_sk(date_value: date, time_dim_path: Optional[Path] = None) -> Optional[int]:
    """
    Get the time dimension surrogate key for a given date.
    
    This function is a wrapper around the TimeDimensionModel.get_time_sk_for_date method
    for backward compatibility.
    
    Args:
        date_value: Date to find surrogate key for
        time_dim_path: Path to the time dimension Parquet file
            
    Returns:
        Surrogate key or None if not found
    """
    # Create TimeDimensionModel instance
    model = TimeDimensionModel()
    
    # Use default path if not provided
    if time_dim_path is None:
        time_dim_path = settings.get_path('OUTPUT_DIR') / "dim_time.parquet"
    
    # Get and return time surrogate key
    return model.get_time_sk_for_date(date_value, time_dim_path)


if __name__ == "__main__":
    # Example usage when run directly
    import sys
    from pathlib import Path
    
    # Set up logging
    from ..utils import setup_logging
    logger = setup_logging(settings.get_path('LOG_DIR'))
    
    # Default years
    start_year = 2011
    end_year = 2031
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        start_year = int(sys.argv[1])
    if len(sys.argv) > 2:
        end_year = int(sys.argv[2])
    
    # Generate time dimension
    output_dir = settings.get_path('OUTPUT_DIR')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "dim_time.parquet"
    success = generate_time_dimension(output_path, start_year, end_year)
    
    if success:
        print(f"Time dimension successfully created for years {start_year}-{end_year}")
        sys.exit(0)
    else:
        print("Failed to create time dimension")
        sys.exit(1)


# Alias for backward compatibility
TimeDimensionBuilder = TimeDimensionModel