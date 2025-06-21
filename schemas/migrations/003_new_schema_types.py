"""
New schema types introduction for AHGD v2.0.0.

This migration introduces new schema types for census, mortality, and environmental data
along with enhanced base schema features.
"""

from datetime import datetime
from typing import Dict, Any, List

from schemas.base_schema import SchemaVersion


class Migration003:
    """New schema types migration."""
    
    version = "003"
    description = "Introduction of census, mortality, and environmental schemas"
    created_at = datetime(2024, 3, 1)
    
    # New schema types introduced in v2.0.0
    NEW_SCHEMAS = {
        "CensusDemographics": {
            "category": "census",
            "description": "Basic demographic data from ABS census",
            "key_fields": [
                "geographic_id", "census_year", "total_population",
                "males", "females", "age_groups", "indigenous"
            ]
        },
        "CensusEducation": {
            "category": "census",
            "description": "Education attainment data from ABS census",
            "key_fields": [
                "geographic_id", "census_year", "education_pop_base",
                "year_12_completion", "university_qualification"
            ]
        },
        "CensusEmployment": {
            "category": "census", 
            "description": "Employment and labour force data from ABS census",
            "key_fields": [
                "geographic_id", "census_year", "labour_force_pop",
                "employed_full_time", "employed_part_time", "unemployed"
            ]
        },
        "CensusHousing": {
            "category": "census",
            "description": "Housing and dwelling data from ABS census",
            "key_fields": [
                "geographic_id", "census_year", "dwelling_structure",
                "tenure_type", "mortgage_rent"
            ]
        },
        "MortalityRecord": {
            "category": "mortality",
            "description": "Individual mortality records from AIHW",
            "key_fields": [
                "record_id", "registration_year", "age_at_death",
                "underlying_cause_icd10", "place_of_death"
            ]
        },
        "MortalityStatistics": {
            "category": "mortality",
            "description": "Aggregated mortality statistics",
            "key_fields": [
                "geographic_id", "reference_year", "total_deaths",
                "crude_death_rate", "age_standardised_rate"
            ]
        },
        "MortalityTrend": {
            "category": "mortality",
            "description": "Mortality trend analysis over time",
            "key_fields": [
                "geographic_id", "cause_of_death", "start_year",
                "end_year", "trend_direction", "annual_change_rate"
            ]
        },
        "WeatherObservation": {
            "category": "environmental",
            "description": "Weather observations from BOM",
            "key_fields": [
                "station_id", "observation_date", "max_temperature",
                "min_temperature", "rainfall_24hr", "humidity"
            ]
        },
        "ClimateStatistics": {
            "category": "environmental",
            "description": "Climate statistics and long-term averages",
            "key_fields": [
                "geographic_id", "reference_period", "mean_temperature",
                "mean_rainfall", "extreme_events"
            ]
        },
        "EnvironmentalHealthIndex": {
            "category": "environmental",
            "description": "Environmental health risk indices",
            "key_fields": [
                "geographic_id", "assessment_date", "heat_index",
                "air_quality_proxy", "overall_health_risk_score"
            ]
        }
    }
    
    # Enhanced base schema features in v2.0.0
    BASE_SCHEMA_ENHANCEMENTS = {
        "thread_safety": "Added thread-safe validation metrics",
        "compatibility_checking": "Automated compatibility validation between versions",
        "performance_monitoring": "Built-in validation performance tracking",
        "enhanced_error_reporting": "Detailed error messages with field-level context",
        "validation_caching": "Optional caching for repeated validation operations"
    }
    
    @staticmethod
    def get_schema_category(schema_type: str) -> str:
        """Get the category for a new schema type."""
        for schema_name, details in Migration003.NEW_SCHEMAS.items():
            if schema_name == schema_type:
                return details["category"]
        return "unknown"
    
    @staticmethod
    def create_default_instance(schema_type: str, geographic_id: str = "default") -> Dict[str, Any]:
        """Create a default instance for a new schema type."""
        base_data = {
            "id": f"{schema_type.lower()}_{geographic_id}",
            "schema_version": SchemaVersion.V2_0_0.value,
            "created_at": datetime.utcnow().isoformat(),
            "data_quality": "medium"
        }
        
        category = Migration003.get_schema_category(schema_type)
        
        if category == "census":
            base_data.update({
                "geographic_id": geographic_id,
                "geographic_level": "SA2",
                "census_year": 2021,
                "data_source": {
                    "source_name": "Australian Bureau of Statistics",
                    "source_date": "2021-08-10",
                    "attribution": "© Australian Bureau of Statistics"
                }
            })
            
        elif category == "mortality":
            base_data.update({
                "geographic_id": geographic_id,
                "registration_year": 2021,
                "data_source": {
                    "source_name": "Australian Institute of Health and Welfare",
                    "source_date": "2022-06-30",
                    "attribution": "© Australian Institute of Health and Welfare"
                }
            })
            
        elif category == "environmental":
            base_data.update({
                "geographic_id": geographic_id,
                "observation_date": "2021-12-31",
                "data_source": {
                    "source_name": "Bureau of Meteorology",
                    "source_date": "2021-12-31",
                    "attribution": "© Bureau of Meteorology"
                }
            })
        
        return base_data
    
    @staticmethod
    def validate_new_schema_requirements(data: Dict[str, Any], schema_type: str) -> List[str]:
        """Validate requirements for new schema types."""
        errors = []
        
        category = Migration003.get_schema_category(schema_type)
        
        # Category-specific validation
        if category == "census":
            if "census_year" not in data:
                errors.append("Census schemas require census_year field")
            elif data["census_year"] < 1911 or data["census_year"] > 2030:
                errors.append("Invalid census year")
                
        elif category == "mortality":
            if schema_type == "MortalityRecord" and "record_id" not in data:
                errors.append("MortalityRecord requires unique record_id")
            elif "registration_year" not in data:
                errors.append("Mortality schemas require registration_year")
                
        elif category == "environmental":
            if schema_type == "WeatherObservation" and "station_id" not in data:
                errors.append("WeatherObservation requires station_id")
            elif "observation_date" not in data and "assessment_date" not in data:
                errors.append("Environmental schemas require date field")
        
        # Common validation for all new schemas
        if "data_source" not in data:
            errors.append("All v2.0.0 schemas require data_source information")
        
        return errors
    
    @staticmethod
    def migrate_legacy_environmental_data(old_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate legacy environmental data to new schema format."""
        new_data = old_data.copy()
        
        # Convert old temperature fields
        if "temp_max" in old_data:
            new_data["max_temperature"] = old_data.pop("temp_max")
        if "temp_min" in old_data:
            new_data["min_temperature"] = old_data.pop("temp_min")
        
        # Convert old rain field
        if "rain" in old_data:
            new_data["rainfall_24hr"] = old_data.pop("rain")
        
        # Convert old site ID
        if "site_id" in old_data:
            new_data["station_id"] = old_data.pop("site_id")
        
        # Update schema version
        new_data["schema_version"] = SchemaVersion.V2_0_0.value
        
        return new_data


def upgrade() -> None:
    """Apply migration - new schema types."""
    print("Migration 003: Adding new schema types for v2.0.0")
    print("- Census data schemas (demographics, education, employment, housing)")
    print("- Mortality data schemas (records, statistics, trends)")
    print("- Environmental data schemas (weather, climate, health indices)")
    print("- Enhanced base schema with thread safety and performance monitoring")


def downgrade() -> None:
    """Revert migration - remove new schema types."""
    print("Migration 003 downgrade: Removing v2.0.0 schema types")
    print("WARNING: This will remove all census, mortality, and environmental schemas")
    print("- Reverting to basic geographic and health schemas only")
    print("- Removing enhanced validation features")