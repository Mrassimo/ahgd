"""
Climate and Environmental Data Models

Pydantic models for Bureau of Meteorology climate data, air quality indicators,
and environmental health risk factors for Australian health analytics.
"""

from typing import Optional, List
from decimal import Decimal
from datetime import date
from enum import Enum

from pydantic import Field, field_validator, model_validator
from pydantic.types import confloat, conint

from .base import GeographicModel, DataQualityMixin, TimestampedModel


class ClimateStation(str, Enum):
    """Major Australian climate monitoring stations."""
    SYDNEY_OBSERVATORY = "066062"
    MELBOURNE_REGIONAL = "086071"
    BRISBANE_AERO = "040913"
    PERTH_METRO = "009225"
    ADELAIDE_WEST = "023000"
    HOBART_ELLERSLIE = "094029"
    CANBERRA_AIRPORT = "070351"
    DARWIN_AIRPORT = "014015"


class ClimateVariable(str, Enum):
    """Climate variables tracked."""
    TEMPERATURE_MAX = "TEMP_MAX"
    TEMPERATURE_MIN = "TEMP_MIN"  
    TEMPERATURE_MEAN = "TEMP_MEAN"
    RAINFALL = "RAINFALL"
    HUMIDITY = "HUMIDITY"
    WIND_SPEED = "WIND_SPEED"
    SOLAR_RADIATION = "SOLAR_RADIATION"
    EVAPORATION = "EVAPORATION"
    PRESSURE = "PRESSURE"


class Season(str, Enum):
    """Australian seasons."""
    SUMMER = "SUMMER"  # Dec-Feb
    AUTUMN = "AUTUMN"  # Mar-May
    WINTER = "WINTER"  # Jun-Aug
    SPRING = "SPRING"  # Sep-Nov


class AirQualityPollutant(str, Enum):
    """Air quality pollutants monitored."""
    PM2_5 = "PM2.5"        # Fine particulate matter
    PM10 = "PM10"          # Coarse particulate matter  
    OZONE = "OZONE"        # Ground-level ozone
    NO2 = "NO2"            # Nitrogen dioxide
    SO2 = "SO2"            # Sulfur dioxide
    CO = "CO"              # Carbon monoxide
    LEAD = "LEAD"          # Lead particles


class AirQualityCategory(str, Enum):
    """Air quality index categories."""
    VERY_GOOD = "VERY_GOOD"      # 0-33
    GOOD = "GOOD"                # 34-66
    FAIR = "FAIR"                # 67-99
    POOR = "POOR"                # 100-149
    VERY_POOR = "VERY_POOR"      # 150+
    HAZARDOUS = "HAZARDOUS"      # 200+


class ClimateRecord(GeographicModel, DataQualityMixin, TimestampedModel):
    """
    Bureau of Meteorology climate data for health analytics.
    
    Links weather patterns to geographic health outcomes and provides
    environmental context for health analysis.
    """
    
    # Station identification
    station_code: str = Field(
        ...,
        pattern=r"^[0-9]{6}$",
        description="BOM weather station code",
        examples=["066062", "040913"]
    )
    
    station_name: str = Field(
        ...,
        description="Weather station name"
    )
    
    # Location details
    latitude: confloat(ge=-90, le=90) = Field(
        ...,
        description="Station latitude (decimal degrees)"
    )
    
    longitude: confloat(ge=-180, le=180) = Field(
        ...,
        description="Station longitude (decimal degrees)"
    )
    
    elevation_metres: Optional[confloat(ge=0)] = Field(
        None,
        description="Station elevation above sea level (metres)"
    )
    
    # Climate measurements
    temperature_max_celsius: Optional[confloat(ge=-50, le=60)] = Field(
        None,
        description="Maximum temperature (°C)"
    )
    
    temperature_min_celsius: Optional[confloat(ge=-50, le=60)] = Field(
        None,
        description="Minimum temperature (°C)"
    )
    
    temperature_mean_celsius: Optional[confloat(ge=-50, le=60)] = Field(
        None,
        description="Mean temperature (°C)"
    )
    
    rainfall_mm: Optional[confloat(ge=0)] = Field(
        None,
        description="Rainfall (millimetres)"
    )
    
    relative_humidity_percent: Optional[confloat(ge=0, le=100)] = Field(
        None,
        description="Relative humidity (%)"
    )
    
    wind_speed_kmh: Optional[confloat(ge=0)] = Field(
        None,
        description="Wind speed (km/h)"
    )
    
    solar_radiation_mj: Optional[confloat(ge=0)] = Field(
        None,
        description="Solar radiation (MJ/m²)"
    )
    
    evaporation_mm: Optional[confloat(ge=0)] = Field(
        None,
        description="Pan evaporation (millimetres)"
    )
    
    atmospheric_pressure_hpa: Optional[confloat(ge=800, le=1200)] = Field(
        None,
        description="Atmospheric pressure (hPa)"
    )
    
    # Temporal aggregation
    observation_date: date = Field(
        ...,
        description="Date of climate observation"
    )
    
    aggregation_period: str = Field(
        ...,
        pattern=r"^(DAILY|WEEKLY|MONTHLY|SEASONAL|ANNUAL)$",
        description="Temporal aggregation of the data"
    )
    
    season: Optional[Season] = Field(
        None,
        description="Australian season"
    )
    
    # Extreme weather indicators
    heat_wave_day: Optional[bool] = Field(
        None,
        description="Whether day qualifies as heat wave conditions"
    )
    
    frost_day: Optional[bool] = Field(
        None,
        description="Whether minimum temperature below 2°C"
    )
    
    heavy_rainfall_day: Optional[bool] = Field(
        None,
        description="Whether rainfall exceeded 25mm"
    )
    
    # Health-relevant derived indicators
    heat_index: Optional[confloat(ge=0)] = Field(
        None,
        description="Heat index combining temperature and humidity"
    )
    
    uv_index: Optional[conint(ge=0, le=15)] = Field(
        None,
        description="UV radiation index"
    )
    
    fire_weather_index: Optional[confloat(ge=0)] = Field(
        None,
        description="Fire weather risk index"
    )
    
    @field_validator('temperature_mean_celsius')
    @classmethod
    def validate_mean_temperature(cls, v, info):
        """Validate mean temperature is between min and max."""
        temp_min = info.data.get('temperature_min_celsius') if info.data else None
        temp_max = info.data.get('temperature_max_celsius') if info.data else None
        
        if v is not None and temp_min is not None and temp_max is not None:
            if v < temp_min or v > temp_max:
                raise ValueError(f"Mean temperature ({v}) must be between min ({temp_min}) and max ({temp_max})")
        
        return v
    
    @field_validator('season')
    @classmethod
    def infer_season_from_date(cls, v, info):
        """Infer season from observation date if not provided."""
        if v is None:
            obs_date = info.data.get('observation_date') if info.data else None
            if obs_date:
                month = obs_date.month
                if month in [12, 1, 2]:
                    return Season.SUMMER
                elif month in [3, 4, 5]:
                    return Season.AUTUMN
                elif month in [6, 7, 8]:
                    return Season.WINTER
                elif month in [9, 10, 11]:
                    return Season.SPRING
        return v


class AirQualityRecord(GeographicModel, DataQualityMixin, TimestampedModel):
    """
    Air quality monitoring data for health impact analysis.
    
    Tracks air pollution levels and health-relevant air quality indicators
    across Australian metropolitan and regional areas.
    """
    
    # Monitoring station
    monitoring_station_code: str = Field(
        ...,
        description="Air quality monitoring station identifier"
    )
    
    monitoring_station_name: str = Field(
        ...,
        description="Air quality monitoring station name"
    )
    
    station_type: str = Field(
        ...,
        pattern=r"^(URBAN|SUBURBAN|RURAL|INDUSTRIAL|ROADSIDE|BACKGROUND)$",
        description="Type of monitoring station environment"
    )
    
    # Location
    latitude: confloat(ge=-90, le=90) = Field(
        ...,
        description="Station latitude"
    )
    
    longitude: confloat(ge=-180, le=180) = Field(
        ...,
        description="Station longitude"
    )
    
    # Air quality measurements
    pm2_5_ugm3: Optional[confloat(ge=0)] = Field(
        None,
        description="PM2.5 concentration (μg/m³)"
    )
    
    pm10_ugm3: Optional[confloat(ge=0)] = Field(
        None,
        description="PM10 concentration (μg/m³)"
    )
    
    ozone_ugm3: Optional[confloat(ge=0)] = Field(
        None,
        description="Ozone concentration (μg/m³)"
    )
    
    no2_ugm3: Optional[confloat(ge=0)] = Field(
        None,
        description="Nitrogen dioxide concentration (μg/m³)"
    )
    
    so2_ugm3: Optional[confloat(ge=0)] = Field(
        None,
        description="Sulfur dioxide concentration (μg/m³)"
    )
    
    co_mgm3: Optional[confloat(ge=0)] = Field(
        None,
        description="Carbon monoxide concentration (mg/m³)"
    )
    
    # Air Quality Index
    air_quality_index: Optional[conint(ge=0)] = Field(
        None,
        description="Overall air quality index value"
    )
    
    air_quality_category: Optional[AirQualityCategory] = Field(
        None,
        description="Air quality category rating"
    )
    
    dominant_pollutant: Optional[AirQualityPollutant] = Field(
        None,
        description="Primary pollutant driving AQI"
    )
    
    # Temporal information
    measurement_date: date = Field(
        ...,
        description="Date of air quality measurement"
    )
    
    measurement_period: str = Field(
        ...,
        pattern=r"^(HOURLY|DAILY|WEEKLY|MONTHLY)$",
        description="Temporal resolution of measurement"
    )
    
    # Health advisories
    health_advisory_level: Optional[str] = Field(
        None,
        pattern=r"^(NONE|SENSITIVE|GENERAL|HAZARDOUS)$",
        description="Health advisory level for air quality"
    )
    
    sensitive_groups_warning: Optional[bool] = Field(
        None,
        description="Whether advisory issued for sensitive groups"
    )
    
    # Source attribution
    bushfire_influence: Optional[bool] = Field(
        None,
        description="Whether air quality affected by bushfire smoke"
    )
    
    dust_storm_influence: Optional[bool] = Field(
        None,
        description="Whether air quality affected by dust storms"
    )
    
    industrial_source: Optional[bool] = Field(
        None,
        description="Whether air quality affected by industrial emissions"
    )
    
    traffic_source: Optional[bool] = Field(
        None,
        description="Whether air quality affected by traffic emissions"
    )
    
    @model_validator(mode='after')
    @classmethod
    def validate_pm_relationship(cls, model):
        """Validate that PM2.5 concentration doesn't exceed PM10."""
        if hasattr(model, 'pm2_5_ugm3') and hasattr(model, 'pm10_ugm3'):
            if model.pm2_5_ugm3 is not None and model.pm10_ugm3 is not None:
                if model.pm2_5_ugm3 > model.pm10_ugm3:
                    raise ValueError("PM2.5 concentration cannot exceed PM10 concentration")
        return model
    
    @field_validator('air_quality_category')
    @classmethod
    def infer_category_from_index(cls, v, info):
        """Infer air quality category from AQI value if not provided."""
        if v is None:
            aqi = info.data.get('air_quality_index') if info.data else None
            if aqi is not None:
                if aqi <= 33:
                    return AirQualityCategory.VERY_GOOD
                elif aqi <= 66:
                    return AirQualityCategory.GOOD
                elif aqi <= 99:
                    return AirQualityCategory.FAIR
                elif aqi <= 149:
                    return AirQualityCategory.POOR
                elif aqi <= 199:
                    return AirQualityCategory.VERY_POOR
                else:
                    return AirQualityCategory.HAZARDOUS
        return v


class EnvironmentalRiskFactor(GeographicModel, DataQualityMixin, TimestampedModel):
    """
    Environmental health risk factors by geographic area.
    
    Aggregates climate and environmental data into health-relevant risk indicators
    for population health analysis.
    """
    
    # Risk categories
    heat_stress_risk: Optional[confloat(ge=0, le=1)] = Field(
        None,
        description="Heat stress risk score (0-1, higher = greater risk)"
    )
    
    air_pollution_risk: Optional[confloat(ge=0, le=1)] = Field(
        None,
        description="Air pollution health risk score (0-1)"
    )
    
    extreme_weather_risk: Optional[confloat(ge=0, le=1)] = Field(
        None,
        description="Extreme weather event risk score (0-1)"
    )
    
    uv_exposure_risk: Optional[confloat(ge=0, le=1)] = Field(
        None,
        description="UV radiation exposure risk score (0-1)"
    )
    
    # Composite indicators
    overall_environmental_risk: Optional[confloat(ge=0, le=1)] = Field(
        None,
        description="Composite environmental health risk score"
    )
    
    climate_health_vulnerability: Optional[confloat(ge=0, le=1)] = Field(
        None,
        description="Climate change health vulnerability index"
    )
    
    # Vulnerable populations
    elderly_risk_multiplier: Optional[confloat(ge=1)] = Field(
        None,
        description="Risk multiplier for elderly populations"
    )
    
    children_risk_multiplier: Optional[confloat(ge=1)] = Field(
        None,
        description="Risk multiplier for children under 5"
    )
    
    chronic_disease_risk_multiplier: Optional[confloat(ge=1)] = Field(
        None,
        description="Risk multiplier for populations with chronic disease"
    )
    
    # Time period
    assessment_year: conint(ge=2000, le=2030) = Field(
        ...,
        description="Year of environmental risk assessment"
    )
    
    projection_scenario: Optional[str] = Field(
        None,
        pattern=r"^(CURRENT|RCP26|RCP45|RCP85)$",
        description="Climate scenario for future projections"
    )
    
    def calculate_population_weighted_risk(
        self, 
        elderly_pop: Optional[int] = None,
        children_pop: Optional[int] = None, 
        chronic_disease_pop: Optional[int] = None,
        total_pop: Optional[int] = None
    ) -> Optional[float]:
        """
        Calculate population-weighted environmental risk score.
        
        Adjusts overall risk based on vulnerable population demographics.
        """
        if not self.overall_environmental_risk or not total_pop:
            return None
            
        base_risk = float(self.overall_environmental_risk)
        weighted_risk = base_risk
        
        # Apply risk multipliers for vulnerable populations
        if elderly_pop and self.elderly_risk_multiplier:
            elderly_weight = elderly_pop / total_pop
            weighted_risk += base_risk * elderly_weight * (float(self.elderly_risk_multiplier) - 1)
            
        if children_pop and self.children_risk_multiplier:
            children_weight = children_pop / total_pop  
            weighted_risk += base_risk * children_weight * (float(self.children_risk_multiplier) - 1)
            
        if chronic_disease_pop and self.chronic_disease_risk_multiplier:
            chronic_weight = chronic_disease_pop / total_pop
            weighted_risk += base_risk * chronic_weight * (float(self.chronic_disease_risk_multiplier) - 1)
        
        return min(weighted_risk, 1.0)  # Cap at 1.0