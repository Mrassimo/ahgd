"""
BOM Environmental data schemas for AHGD.

This module defines schemas for Bureau of Meteorology (BOM) environmental data
including weather observations, climate data, and environmental health indicators.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
from enum import Enum
from pydantic import Field, field_validator, model_validator

from .base_schema import (
    VersionedSchema,
    DataSource,
    SchemaVersion,
    DataQualityLevel,
    TemporalData
)


class WeatherStationType(str, Enum):
    """Types of weather monitoring stations."""
    AUTOMATIC = "automatic"  # Automatic Weather Station
    MANUAL = "manual"  # Manual observation
    COMPOSITE = "composite"  # Composite site with multiple sensors
    AIRPORT = "airport"  # Airport weather station
    MARINE = "marine"  # Marine observation
    RADAR = "radar"  # Weather radar
    SATELLITE = "satellite"  # Satellite observation


class ObservationQuality(str, Enum):
    """Quality codes for weather observations."""
    EXCELLENT = "Y"  # Excellent quality
    GOOD = "G"  # Good quality
    QUESTIONABLE = "Q"  # Questionable quality
    POOR = "P"  # Poor quality
    MISSING = "M"  # Missing data
    NOT_CHECKED = "N"  # Not quality checked


class ClimateVariable(str, Enum):
    """Types of climate variables."""
    TEMPERATURE = "temperature"
    RAINFALL = "rainfall"
    HUMIDITY = "humidity"
    WIND_SPEED = "wind_speed"
    WIND_DIRECTION = "wind_direction"
    PRESSURE = "pressure"
    SOLAR_RADIATION = "solar_radiation"
    EVAPORATION = "evaporation"
    SUNSHINE_HOURS = "sunshine_hours"


class WeatherObservation(VersionedSchema, TemporalData):
    """Schema for individual weather observations."""
    
    # Station identification
    station_id: str = Field(..., description="BOM station identifier")
    station_name: str = Field(..., description="Weather station name")
    station_type: WeatherStationType = Field(..., description="Type of weather station")
    
    # Geographic location
    latitude: float = Field(..., ge=-90, le=90, description="Station latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Station longitude")
    elevation: float = Field(..., description="Station elevation (metres above sea level)")
    
    # Temporal information
    observation_date: date = Field(..., description="Date of observation")
    observation_time: Optional[str] = Field(None, description="Time of observation (24hr format)")
    
    # Temperature (°C)
    max_temperature: Optional[float] = Field(None, ge=-50, le=60, description="Maximum temperature")
    min_temperature: Optional[float] = Field(None, ge=-50, le=60, description="Minimum temperature")
    temperature_9am: Optional[float] = Field(None, ge=-50, le=60, description="Temperature at 9am")
    temperature_3pm: Optional[float] = Field(None, ge=-50, le=60, description="Temperature at 3pm")
    
    # Rainfall (mm)
    rainfall_24hr: Optional[float] = Field(None, ge=0, le=1000, description="24-hour rainfall")
    rainfall_days: Optional[int] = Field(None, ge=0, le=31, description="Number of rain days in period")
    
    # Humidity (%)
    humidity_9am: Optional[float] = Field(None, ge=0, le=100, description="Relative humidity at 9am")
    humidity_3pm: Optional[float] = Field(None, ge=0, le=100, description="Relative humidity at 3pm")
    
    # Wind
    wind_speed_9am: Optional[float] = Field(None, ge=0, le=200, description="Wind speed at 9am (km/h)")
    wind_speed_3pm: Optional[float] = Field(None, ge=0, le=200, description="Wind speed at 3pm (km/h)")
    wind_direction_9am: Optional[int] = Field(None, ge=0, le=360, description="Wind direction at 9am (degrees)")
    wind_direction_3pm: Optional[int] = Field(None, ge=0, le=360, description="Wind direction at 3pm (degrees)")
    wind_gust_speed: Optional[float] = Field(None, ge=0, le=300, description="Maximum wind gust speed (km/h)")
    wind_gust_direction: Optional[int] = Field(None, ge=0, le=360, description="Wind gust direction (degrees)")
    
    # Pressure (hPa)
    pressure_msl_9am: Optional[float] = Field(None, ge=900, le=1100, description="MSL pressure at 9am")
    pressure_msl_3pm: Optional[float] = Field(None, ge=900, le=1100, description="MSL pressure at 3pm")
    
    # Cloud and visibility
    cloud_cover_9am: Optional[int] = Field(None, ge=0, le=8, description="Cloud cover at 9am (octas)")
    cloud_cover_3pm: Optional[int] = Field(None, ge=0, le=8, description="Cloud cover at 3pm (octas)")
    visibility_9am: Optional[float] = Field(None, ge=0, description="Visibility at 9am (km)")
    visibility_3pm: Optional[float] = Field(None, ge=0, description="Visibility at 3pm (km)")
    
    # Solar and evaporation
    sunshine_hours: Optional[float] = Field(None, ge=0, le=24, description="Daily sunshine hours")
    solar_radiation: Optional[float] = Field(None, ge=0, description="Solar radiation (MJ/m²)")
    evaporation: Optional[float] = Field(None, ge=0, le=50, description="Class A pan evaporation (mm)")
    
    # Quality indicators
    max_temp_quality: ObservationQuality = Field(default=ObservationQuality.NOT_CHECKED)
    min_temp_quality: ObservationQuality = Field(default=ObservationQuality.NOT_CHECKED)
    rainfall_quality: ObservationQuality = Field(default=ObservationQuality.NOT_CHECKED)
    
    # Data source
    data_source: DataSource = Field(..., description="Source of weather observation")
    
    @field_validator('observation_time')
    @classmethod
    def validate_time_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate time format (HHMM)."""
        if v is None:
            return v
        
        if not v.isdigit() or len(v) != 4:
            raise ValueError("Time must be in HHMM format")
        
        hour = int(v[:2])
        minute = int(v[2:])
        
        if hour > 23 or minute > 59:
            raise ValueError("Invalid time values")
        
        return v
    
    @model_validator(mode='after')
    def validate_weather_consistency(self) -> 'WeatherObservation':
        """Validate weather observation consistency."""
        # Check temperature consistency
        if (self.max_temperature is not None and self.min_temperature is not None):
            if self.min_temperature > self.max_temperature:
                raise ValueError("Minimum temperature cannot exceed maximum temperature")
        
        # Check 9am and 3pm temperatures against min/max
        if self.temperature_9am is not None and self.min_temperature is not None:
            if self.temperature_9am < self.min_temperature - 2:  # Allow small tolerance
                raise ValueError("9am temperature below daily minimum")
        
        if self.temperature_3pm is not None and self.max_temperature is not None:
            if self.temperature_3pm > self.max_temperature + 2:  # Allow small tolerance
                raise ValueError("3pm temperature above daily maximum")
        
        # Check humidity consistency (9am typically higher than 3pm)
        if (self.humidity_9am is not None and self.humidity_3pm is not None):
            if self.humidity_9am < self.humidity_3pm - 30:  # Allow for weather patterns
                # Warning rather than error - this can happen
                pass
        
        # Check sunshine hours against day length (approximate)
        if self.sunshine_hours is not None and self.sunshine_hours > 16:
            # Even in summer, >16 hours would be unusual in Australia
            raise ValueError("Sunshine hours exceed reasonable maximum")
        
        return self
    
    def get_temperature_range(self) -> Optional[float]:
        """Calculate daily temperature range."""
        if self.max_temperature is not None and self.min_temperature is not None:
            return self.max_temperature - self.min_temperature
        return None
    
    def get_heat_index(self) -> Optional[float]:
        """Calculate approximate heat index using temperature and humidity."""
        if (self.max_temperature is not None and self.humidity_3pm is not None and 
            self.max_temperature >= 27):  # Heat index only meaningful for warm weather
            
            # Simplified heat index calculation
            t = self.max_temperature
            h = self.humidity_3pm
            
            hi = (-42.379 + 2.04901523 * t + 10.14333127 * h - 0.22475541 * t * h -
                  6.83783e-3 * t * t - 5.481717e-2 * h * h + 1.22874e-3 * t * t * h +
                  8.5282e-4 * t * h * h - 1.99e-6 * t * t * h * h)
            
            return hi
        return None
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "WeatherObservation"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate weather observation data integrity."""
        errors = []
        
        # Check for extreme values
        if self.max_temperature is not None and self.max_temperature > 55:
            errors.append(f"Maximum temperature extremely high: {self.max_temperature}°C")
        
        if self.rainfall_24hr is not None and self.rainfall_24hr > 500:
            errors.append(f"Daily rainfall extremely high: {self.rainfall_24hr}mm")
        
        if self.wind_gust_speed is not None and self.wind_gust_speed > 200:
            errors.append(f"Wind gust speed extremely high: {self.wind_gust_speed}km/h")
        
        # Check for data quality flags
        poor_quality_vars = []
        if self.max_temp_quality in [ObservationQuality.POOR, ObservationQuality.QUESTIONABLE]:
            poor_quality_vars.append("maximum temperature")
        if self.min_temp_quality in [ObservationQuality.POOR, ObservationQuality.QUESTIONABLE]:
            poor_quality_vars.append("minimum temperature")
        if self.rainfall_quality in [ObservationQuality.POOR, ObservationQuality.QUESTIONABLE]:
            poor_quality_vars.append("rainfall")
        
        if poor_quality_vars:
            errors.append(f"Poor quality data for: {', '.join(poor_quality_vars)}")
        
        return errors


class ClimateStatistics(VersionedSchema, TemporalData):
    """Schema for aggregated climate statistics."""
    
    # Geographic identification
    geographic_id: str = Field(..., description="Geographic area identifier")
    geographic_level: str = Field(..., description="Geographic level")
    geographic_name: str = Field(..., description="Geographic area name")
    
    # Time period
    reference_period: str = Field(..., description="Reference period (e.g., '1991-2020')")
    statistic_period: str = Field(..., description="Period for statistics (monthly/seasonal/annual)")
    month: Optional[int] = Field(None, ge=1, le=12, description="Month (if monthly statistics)")
    season: Optional[str] = Field(None, description="Season (if seasonal statistics)")
    
    # Temperature statistics (°C)
    mean_max_temperature: Optional[float] = Field(None, description="Mean maximum temperature")
    mean_min_temperature: Optional[float] = Field(None, description="Mean minimum temperature")
    mean_temperature: Optional[float] = Field(None, description="Mean temperature")
    highest_temperature: Optional[float] = Field(None, description="Highest temperature on record")
    lowest_temperature: Optional[float] = Field(None, description="Lowest temperature on record")
    
    # Temperature extremes
    days_over_35c: Optional[float] = Field(None, ge=0, description="Average days over 35°C")
    days_over_40c: Optional[float] = Field(None, ge=0, description="Average days over 40°C")
    days_under_0c: Optional[float] = Field(None, ge=0, description="Average days under 0°C")
    days_under_minus5c: Optional[float] = Field(None, ge=0, description="Average days under -5°C")
    
    # Rainfall statistics (mm)
    mean_rainfall: Optional[float] = Field(None, ge=0, description="Mean rainfall")
    median_rainfall: Optional[float] = Field(None, ge=0, description="Median rainfall")
    rainfall_decile_1: Optional[float] = Field(None, ge=0, description="10th percentile rainfall")
    rainfall_decile_9: Optional[float] = Field(None, ge=0, description="90th percentile rainfall")
    highest_daily_rainfall: Optional[float] = Field(None, ge=0, description="Highest daily rainfall")
    
    # Rainfall patterns
    mean_rain_days: Optional[float] = Field(None, ge=0, description="Average number of rain days")
    days_over_1mm: Optional[float] = Field(None, ge=0, description="Days with >1mm rain")
    days_over_10mm: Optional[float] = Field(None, ge=0, description="Days with >10mm rain")
    days_over_25mm: Optional[float] = Field(None, ge=0, description="Days with >25mm rain")
    
    # Humidity and pressure
    mean_humidity_9am: Optional[float] = Field(None, ge=0, le=100, description="Mean 9am humidity")
    mean_humidity_3pm: Optional[float] = Field(None, ge=0, le=100, description="Mean 3pm humidity")
    mean_pressure: Optional[float] = Field(None, description="Mean sea level pressure")
    
    # Wind statistics
    mean_wind_speed: Optional[float] = Field(None, ge=0, description="Mean wind speed")
    prevailing_wind_direction: Optional[str] = Field(None, description="Prevailing wind direction")
    highest_gust: Optional[float] = Field(None, ge=0, description="Highest wind gust recorded")
    
    # Solar and evaporation
    mean_sunshine_hours: Optional[float] = Field(None, ge=0, description="Mean daily sunshine hours")
    mean_solar_radiation: Optional[float] = Field(None, ge=0, description="Mean solar radiation")
    mean_evaporation: Optional[float] = Field(None, ge=0, description="Mean daily evaporation")
    
    # Climate indices
    heat_wave_days: Optional[float] = Field(None, ge=0, description="Average heat wave days")
    frost_days: Optional[float] = Field(None, ge=0, description="Average frost days")
    growing_degree_days: Optional[float] = Field(None, ge=0, description="Growing degree days")
    
    # Data completeness
    temperature_completeness: float = Field(..., ge=0, le=100, description="Temperature data completeness %")
    rainfall_completeness: float = Field(..., ge=0, le=100, description="Rainfall data completeness %")
    overall_completeness: float = Field(..., ge=0, le=100, description="Overall data completeness %")
    
    # Data source
    data_source: DataSource = Field(..., description="Source of climate statistics")
    
    @field_validator('season')
    @classmethod
    def validate_season(cls, v: Optional[str]) -> Optional[str]:
        """Validate season names."""
        if v is None:
            return v
        
        valid_seasons = {'Summer', 'Autumn', 'Winter', 'Spring', 'DJF', 'MAM', 'JJA', 'SON'}
        if v not in valid_seasons:
            raise ValueError(f"Invalid season: {v}")
        return v
    
    @field_validator('prevailing_wind_direction')
    @classmethod
    def validate_wind_direction(cls, v: Optional[str]) -> Optional[str]:
        """Validate wind direction."""
        if v is None:
            return v
        
        valid_directions = {'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                           'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'CALM'}
        if v not in valid_directions:
            raise ValueError(f"Invalid wind direction: {v}")
        return v
    
    @model_validator(mode='after')
    def validate_climate_consistency(self) -> 'ClimateStatistics':
        """Validate climate statistics consistency."""
        # Check temperature relationships
        if (self.mean_max_temperature is not None and self.mean_min_temperature is not None):
            if self.mean_min_temperature >= self.mean_max_temperature:
                raise ValueError("Mean minimum temperature must be less than mean maximum")
        
        if self.mean_temperature is not None:
            if (self.mean_max_temperature is not None and self.mean_min_temperature is not None):
                expected_mean = (self.mean_max_temperature + self.mean_min_temperature) / 2
                if abs(self.mean_temperature - expected_mean) > 2:
                    raise ValueError("Mean temperature inconsistent with max/min temperatures")
        
        # Check rainfall percentiles
        if (self.rainfall_decile_1 is not None and self.rainfall_decile_9 is not None):
            if self.rainfall_decile_1 >= self.rainfall_decile_9:
                raise ValueError("90th percentile rainfall must exceed 10th percentile")
        
        if (self.median_rainfall is not None and self.mean_rainfall is not None):
            # For rainfall, mean is typically higher than median due to skewness
            if self.mean_rainfall < self.median_rainfall * 0.5:
                raise ValueError("Mean rainfall unusually low compared to median")
        
        return self
    
    def get_temperature_variability(self) -> Optional[float]:
        """Calculate temperature variability (diurnal range)."""
        if self.mean_max_temperature is not None and self.mean_min_temperature is not None:
            return self.mean_max_temperature - self.mean_min_temperature
        return None
    
    def get_aridity_index(self) -> Optional[float]:
        """Calculate approximate aridity index (rainfall/evaporation)."""
        if (self.mean_rainfall is not None and self.mean_evaporation is not None and 
            self.mean_evaporation > 0):
            # Convert to same units (assuming monthly data)
            monthly_evap = self.mean_evaporation * 30  # mm/month
            return self.mean_rainfall / monthly_evap
        return None
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "ClimateStatistics"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate climate statistics integrity."""
        errors = []
        
        # Check data completeness
        if self.overall_completeness < 70:
            errors.append(f"Low data completeness: {self.overall_completeness}%")
        
        # Check for extreme values
        if self.mean_max_temperature is not None and self.mean_max_temperature > 50:
            errors.append(f"Mean maximum temperature extremely high: {self.mean_max_temperature}°C")
        
        if self.mean_rainfall is not None and self.mean_rainfall > 1000:
            # This could be valid for tropical areas, but worth flagging
            errors.append(f"Mean rainfall very high: {self.mean_rainfall}mm")
        
        # Check heat wave consistency
        if (self.heat_wave_days is not None and self.days_over_35c is not None):
            if self.heat_wave_days > self.days_over_35c:
                errors.append("Heat wave days exceed days over 35°C")
        
        return errors


class EnvironmentalHealthIndex(VersionedSchema, TemporalData):
    """Schema for environmental health risk indices."""
    
    # Geographic identification
    geographic_id: str = Field(..., description="Geographic area identifier")
    geographic_level: str = Field(..., description="Geographic level")
    assessment_date: date = Field(..., description="Date of assessment")
    
    # Heat-related indices
    heat_index_mean: Optional[float] = Field(None, description="Mean heat index")
    heat_index_max: Optional[float] = Field(None, description="Maximum heat index")
    extreme_heat_days: Optional[int] = Field(None, ge=0, description="Days with extreme heat index")
    
    # Air quality proxies (from weather data)
    atmospheric_stability_index: Optional[float] = Field(None, description="Atmospheric stability index")
    ventilation_coefficient: Optional[float] = Field(None, description="Atmospheric ventilation coefficient")
    dust_storm_risk_days: Optional[int] = Field(None, ge=0, description="Days with dust storm risk")
    
    # UV and solar radiation
    uv_index_mean: Optional[float] = Field(None, ge=0, le=15, description="Mean UV index")
    uv_index_max: Optional[float] = Field(None, ge=0, le=15, description="Maximum UV index")
    extreme_uv_days: Optional[int] = Field(None, ge=0, description="Days with extreme UV")
    
    # Humidity and comfort
    discomfort_index_mean: Optional[float] = Field(None, description="Mean discomfort index")
    very_humid_days: Optional[int] = Field(None, ge=0, description="Days with very high humidity")
    
    # Extreme weather events
    heatwave_events: Optional[int] = Field(None, ge=0, description="Number of heatwave events")
    severe_storm_days: Optional[int] = Field(None, ge=0, description="Days with severe storms")
    drought_index: Optional[float] = Field(None, description="Drought severity index")
    flood_risk_days: Optional[int] = Field(None, ge=0, description="Days with flood risk")
    
    # Composite indices
    overall_health_risk_score: float = Field(..., ge=0, le=100, description="Overall environmental health risk score")
    heat_stress_risk: float = Field(..., ge=0, le=100, description="Heat stress risk score")
    respiratory_risk: float = Field(..., ge=0, le=100, description="Respiratory health risk score")
    
    # Vulnerable population impacts
    elderly_risk_multiplier: float = Field(1.0, ge=1.0, le=5.0, description="Risk multiplier for elderly")
    children_risk_multiplier: float = Field(1.0, ge=1.0, le=5.0, description="Risk multiplier for children")
    outdoor_worker_risk: float = Field(..., ge=0, le=100, description="Risk score for outdoor workers")
    
    # Data source
    data_source: DataSource = Field(..., description="Source of environmental health data")
    
    @model_validator(mode='after')
    def validate_health_indices(self) -> 'EnvironmentalHealthIndex':
        """Validate environmental health indices."""
        # Check that extreme days don't exceed total days in period
        # This would require knowing the assessment period length
        
        # Check index consistency
        if (self.heat_index_mean is not None and self.heat_index_max is not None):
            if self.heat_index_mean > self.heat_index_max:
                raise ValueError("Mean heat index cannot exceed maximum")
        
        if (self.uv_index_mean is not None and self.uv_index_max is not None):
            if self.uv_index_mean > self.uv_index_max:
                raise ValueError("Mean UV index cannot exceed maximum")
        
        return self
    
    def get_composite_risk_category(self) -> str:
        """Get overall risk category based on composite score."""
        if self.overall_health_risk_score >= 80:
            return "Very High"
        elif self.overall_health_risk_score >= 60:
            return "High"
        elif self.overall_health_risk_score >= 40:
            return "Moderate"
        elif self.overall_health_risk_score >= 20:
            return "Low"
        else:
            return "Very Low"
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "EnvironmentalHealthIndex"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate environmental health index integrity."""
        errors = []
        
        # Check for unrealistic UV values
        if self.uv_index_max is not None and self.uv_index_max > 12:
            # UV index >12 is possible but rare in Australia
            errors.append(f"UV index extremely high: {self.uv_index_max}")
        
        # Check risk score consistency
        if self.heat_stress_risk > 90 and self.overall_health_risk_score < 50:
            errors.append("High heat stress risk but low overall risk - check calculation")
        
        return errors


# Migration functions for environmental schemas

def migrate_environmental_v1_to_v2(old_data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate environmental data from v1.0.0 to v2.0.0."""
    new_data = old_data.copy()
    
    # Example migration: convert old temperature units
    if 'max_temp_f' in old_data:
        # Convert Fahrenheit to Celsius
        temp_f = old_data.pop('max_temp_f')
        new_data['max_temperature'] = (temp_f - 32) * 5/9
    
    # Standardise station identifiers
    if 'site_id' in old_data:
        new_data['station_id'] = old_data.pop('site_id')
    
    # Convert old quality codes
    quality_mapping = {'1': 'Y', '2': 'G', '3': 'Q', '4': 'P', '9': 'M'}
    if 'temp_quality' in old_data:
        old_quality = str(old_data.pop('temp_quality'))
        new_data['max_temp_quality'] = quality_mapping.get(old_quality, 'N')
    
    # Update schema version
    new_data['schema_version'] = SchemaVersion.V2_0_0.value
    
    return new_data