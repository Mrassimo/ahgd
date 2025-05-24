"""Time dimension builder for the data warehouse."""

from datetime import date, datetime, timedelta
from typing import List, Dict, Any
import polars as pl
from pathlib import Path

from ..config import get_settings
from ..utils import get_logger


class TimeDimensionBuilder:
    """Builds a complete time dimension table for the data warehouse."""
    
    def __init__(self, start_year: int = None, end_year: int = None):
        """Initialize the time dimension builder.
        
        Args:
            start_year: Starting year (defaults to settings)
            end_year: Ending year (defaults to settings)
        """
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        self.start_year = start_year or self.settings.time_dim_start_year
        self.end_year = end_year or self.settings.time_dim_end_year
        
        # Schema definition
        self.schema = self.settings.get_schema("dim_time")
        
        # Census years (could be configurable)
        self.census_years = [1996, 2001, 2006, 2011, 2016, 2021, 2026]
    
    def build(self) -> pl.DataFrame:
        """Build the complete time dimension table.
        
        Returns:
            Polars DataFrame containing the time dimension
        """
        self.logger.info(f"Building time dimension from {self.start_year} to {self.end_year}")
        
        # Generate date range
        start_date = date(self.start_year, 1, 1)
        end_date = date(self.end_year, 12, 31)
        
        # Create list of all dates
        date_list = []
        current_date = start_date
        
        while current_date <= end_date:
            date_list.append(self._create_date_record(current_date))
            current_date += timedelta(days=1)
        
        # Add unknown member
        unknown_record = self._create_unknown_member()
        date_list.insert(0, unknown_record)
        
        # Create DataFrame
        df = pl.DataFrame(date_list)
        
        # Enforce schema
        df = self._enforce_schema(df)
        
        self.logger.info(f"Time dimension complete: {len(df)} records")
        
        return df
    
    def _create_date_record(self, d: date) -> Dict[str, Any]:
        """Create a single date record for the time dimension.
        
        Args:
            d: Date to process
            
        Returns:
            Dictionary containing all time attributes
        """
        # Day names
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        # Month names
        month_names = ["January", "February", "March", "April", "May", "June",
                      "July", "August", "September", "October", "November", "December"]
        
        # Calculate financial year (July to June in Australia)
        if d.month >= 7:
            fy_start = d.year
            fy_end = d.year + 1
        else:
            fy_start = d.year - 1
            fy_end = d.year
        
        financial_year = f"{fy_start}/{str(fy_end)[2:]}"
        
        # Create record
        return {
            "time_sk": int(d.strftime("%Y%m%d")),  # YYYYMMDD as integer
            "full_date": d,
            "year": d.year,
            "quarter": (d.month - 1) // 3 + 1,
            "month": d.month,
            "month_name": month_names[d.month - 1],
            "day_of_month": d.day,
            "day_of_week": d.weekday(),  # 0 = Monday, 6 = Sunday
            "day_name": day_names[d.weekday()],
            "is_weekday": d.weekday() < 5,  # Monday-Friday
            "financial_year": financial_year,
            "is_census_year": d.year in self.census_years,
            "is_unknown": False,
            "etl_processed_at": datetime.now()
        }
    
    def _create_unknown_member(self) -> Dict[str, Any]:
        """Create the unknown member record for the time dimension.
        
        Returns:
            Dictionary containing the unknown member attributes
        """
        return {
            "time_sk": self.settings.unknown_time_sk,  # 19000101
            "full_date": date(1900, 1, 1),
            "year": 1900,
            "quarter": 1,
            "month": 1,
            "month_name": "Unknown",
            "day_of_month": 1,
            "day_of_week": 0,
            "day_name": "Unknown",
            "is_weekday": False,
            "financial_year": "Unknown",
            "is_census_year": False,
            "is_unknown": True,
            "etl_processed_at": datetime.now()
        }
    
    def _enforce_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Enforce the target schema on the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with enforced schema
        """
        # Get expected columns from schema
        expected_columns = [col["name"] for col in self.schema["columns"]]
        
        # Select and reorder columns
        df = df.select(expected_columns)
        
        # Cast to correct types
        type_mapping = {
            "Int64": pl.Int64,
            "Int32": pl.Int32,
            "Int8": pl.Int8,
            "Utf8": pl.Utf8,
            "Date": pl.Date,
            "Boolean": pl.Boolean,
            "Datetime": pl.Datetime,
            "Categorical": pl.Categorical
        }
        
        cast_expressions = []
        for col_def in self.schema["columns"]:
            col_name = col_def["name"]
            col_type = col_def["dtype"]
            
            if col_type in type_mapping:
                if col_type == "Categorical":
                    cast_expressions.append(
                        pl.col(col_name).cast(pl.Utf8).cast(pl.Categorical)
                    )
                else:
                    cast_expressions.append(
                        pl.col(col_name).cast(type_mapping[col_type])
                    )
            else:
                cast_expressions.append(pl.col(col_name))
        
        df = df.select(cast_expressions)
        
        return df
    
    def save_to_parquet(self, df: pl.DataFrame, output_path: Path = None) -> Path:
        """Save the time dimension to Parquet format.
        
        Args:
            df: DataFrame to save
            output_path: Optional custom output path
            
        Returns:
            Path to the saved file
        """
        if output_path is None:
            output_path = self.settings.output_dir / "dim_time.parquet"
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to Parquet
        df.write_parquet(output_path, compression="snappy")
        
        self.logger.info(f"Saved time dimension to {output_path}")
        
        return output_path