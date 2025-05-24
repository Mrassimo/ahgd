"""
Dimension model classes for AHGD ETL pipeline.

This module defines the dimension table models used in the AHGD data warehouse.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, ClassVar

import polars as pl

from .base import DimensionModel

@dataclass
class GeoDimension(DimensionModel):
    """Geographic dimension model."""
    
    table_name: ClassVar[str] = "geo_dimension"
    
    # Natural key fields
    geo_id: str = ""
    geo_level: str = ""
    geo_name: str = ""
    
    # Additional attributes
    state_code: str = ""
    state_name: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    geom: str = ""
    parent_geo_sk: str = ""  # Reference to parent geography
    
    @classmethod
    def from_dataframe(cls, df: pl.DataFrame) -> 'GeoDimension':
        """Create GeoDimension instance from DataFrame."""
        if len(df) == 0:
            return cls()
        
        row = df.row(0, named=True)
        return cls(
            surrogate_key=str(row.get("geo_sk", "")),
            geo_id=str(row.get("geo_id", "")),
            geo_level=str(row.get("geo_level", "")),
            geo_name=str(row.get("geo_name", "")),
            state_code=str(row.get("state_code", "")),
            state_name=str(row.get("state_name", "")),
            latitude=float(row.get("latitude", 0.0)),
            longitude=float(row.get("longitude", 0.0)),
            geom=str(row.get("geom", "")),
            parent_geo_sk=str(row.get("parent_geo_sk", "")),
            is_unknown=bool(row.get("is_unknown", False)),
            etl_processed_at=row.get("etl_processed_at", datetime.now())
        )
    
    def to_dataframe(self) -> pl.DataFrame:
        """Convert GeoDimension to DataFrame."""
        return pl.DataFrame({
            "geo_sk": [self.surrogate_key],
            "geo_id": [self.geo_id],
            "geo_level": [self.geo_level],
            "geo_name": [self.geo_name],
            "state_code": [self.state_code],
            "state_name": [self.state_name],
            "latitude": [self.latitude],
            "longitude": [self.longitude],
            "geom": [self.geom],
            "parent_geo_sk": [self.parent_geo_sk],
            "is_unknown": [self.is_unknown],
            "etl_processed_at": [self.etl_processed_at]
        })

@dataclass
class TimeDimension(DimensionModel):
    """Time dimension model."""
    
    table_name: ClassVar[str] = "dim_time"
    
    # Natural key fields
    full_date: datetime = field(default_factory=datetime.now)
    
    # Time attributes
    year: int = 0
    quarter: int = 0
    month: int = 0
    month_name: str = ""
    day_of_month: int = 0
    day_of_week: int = 0
    day_name: str = ""
    is_weekday: bool = True
    financial_year: str = ""
    is_census_year: bool = False
    
    @classmethod
    def from_dataframe(cls, df: pl.DataFrame) -> 'TimeDimension':
        """Create TimeDimension instance from DataFrame."""
        if len(df) == 0:
            return cls()
        
        row = df.row(0, named=True)
        return cls(
            surrogate_key=str(row.get("time_sk", "")),
            full_date=row.get("full_date", datetime.now()),
            year=int(row.get("year", 0)),
            quarter=int(row.get("quarter", 0)),
            month=int(row.get("month", 0)),
            month_name=str(row.get("month_name", "")),
            day_of_month=int(row.get("day_of_month", 0)),
            day_of_week=int(row.get("day_of_week", 0)),
            day_name=str(row.get("day_name", "")),
            is_weekday=bool(row.get("is_weekday", True)),
            financial_year=str(row.get("financial_year", "")),
            is_census_year=bool(row.get("is_census_year", False)),
            is_unknown=bool(row.get("is_unknown", False)),
            etl_processed_at=row.get("etl_processed_at", datetime.now())
        )
    
    def to_dataframe(self) -> pl.DataFrame:
        """Convert TimeDimension to DataFrame."""
        return pl.DataFrame({
            "time_sk": [self.surrogate_key],
            "full_date": [self.full_date],
            "year": [self.year],
            "quarter": [self.quarter],
            "month": [self.month],
            "month_name": [self.month_name],
            "day_of_month": [self.day_of_month],
            "day_of_week": [self.day_of_week],
            "day_name": [self.day_name],
            "is_weekday": [self.is_weekday],
            "financial_year": [self.financial_year],
            "is_census_year": [self.is_census_year],
            "is_unknown": [self.is_unknown],
            "etl_processed_at": [self.etl_processed_at]
        })

@dataclass
class HealthConditionDimension(DimensionModel):
    """Health condition dimension model."""
    
    table_name: ClassVar[str] = "dim_health_condition"
    
    # Natural key fields
    condition_code: str = ""
    
    # Additional attributes
    condition_name: str = ""
    condition_category: str = ""
    
    @classmethod
    def from_dataframe(cls, df: pl.DataFrame) -> 'HealthConditionDimension':
        """Create HealthConditionDimension instance from DataFrame."""
        if len(df) == 0:
            return cls()
        
        row = df.row(0, named=True)
        return cls(
            surrogate_key=str(row.get("condition_sk", "")),
            condition_code=str(row.get("condition_code", "")),
            condition_name=str(row.get("condition_name", "")),
            condition_category=str(row.get("condition_category", "")),
            is_unknown=bool(row.get("is_unknown", False)),
            etl_processed_at=row.get("etl_processed_at", datetime.now())
        )
    
    def to_dataframe(self) -> pl.DataFrame:
        """Convert HealthConditionDimension to DataFrame."""
        return pl.DataFrame({
            "condition_sk": [self.surrogate_key],
            "condition_code": [self.condition_code],
            "condition_name": [self.condition_name],
            "condition_category": [self.condition_category],
            "is_unknown": [self.is_unknown],
            "etl_processed_at": [self.etl_processed_at]
        })

@dataclass
class DemographicDimension(DimensionModel):
    """Demographic dimension model."""
    
    table_name: ClassVar[str] = "dim_demographic"
    
    # Natural key fields
    age_group: str = ""
    sex: str = ""
    
    @classmethod
    def from_dataframe(cls, df: pl.DataFrame) -> 'DemographicDimension':
        """Create DemographicDimension instance from DataFrame."""
        if len(df) == 0:
            return cls()
        
        row = df.row(0, named=True)
        return cls(
            surrogate_key=str(row.get("demographic_sk", "")),
            age_group=str(row.get("age_group", "")),
            sex=str(row.get("sex", "")),
            is_unknown=bool(row.get("is_unknown", False)),
            etl_processed_at=row.get("etl_processed_at", datetime.now())
        )
    
    def to_dataframe(self) -> pl.DataFrame:
        """Convert DemographicDimension to DataFrame."""
        return pl.DataFrame({
            "demographic_sk": [self.surrogate_key],
            "age_group": [self.age_group],
            "sex": [self.sex],
            "is_unknown": [self.is_unknown],
            "etl_processed_at": [self.etl_processed_at]
        })

@dataclass
class PersonCharacteristicDimension(DimensionModel):
    """Person characteristic dimension model."""
    
    table_name: ClassVar[str] = "dim_person_characteristic"
    
    # Natural key fields
    characteristic_type: str = ""
    characteristic_value: str = ""
    
    # Additional attributes
    characteristic_category: str = ""
    
    @classmethod
    def from_dataframe(cls, df: pl.DataFrame) -> 'PersonCharacteristicDimension':
        """Create PersonCharacteristicDimension instance from DataFrame."""
        if len(df) == 0:
            return cls()
        
        row = df.row(0, named=True)
        return cls(
            surrogate_key=str(row.get("characteristic_sk", "")),
            characteristic_type=str(row.get("characteristic_type", "")),
            characteristic_value=str(row.get("characteristic_value", "")),
            characteristic_category=str(row.get("characteristic_category", "")),
            is_unknown=bool(row.get("is_unknown", False)),
            etl_processed_at=row.get("etl_processed_at", datetime.now())
        )
    
    def to_dataframe(self) -> pl.DataFrame:
        """Convert PersonCharacteristicDimension to DataFrame."""
        return pl.DataFrame({
            "characteristic_sk": [self.surrogate_key],
            "characteristic_type": [self.characteristic_type],
            "characteristic_value": [self.characteristic_value],
            "characteristic_category": [self.characteristic_category],
            "is_unknown": [self.is_unknown],
            "etl_processed_at": [self.etl_processed_at]
        })