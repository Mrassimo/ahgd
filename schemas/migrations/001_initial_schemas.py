"""
Initial schema definitions for AHGD data versioning.

This migration establishes the baseline v1.0.0 schemas for all data types.
"""

from datetime import datetime
from typing import Dict, Any, List

from schemas.base_schema import SchemaVersion


class Migration001:
    """Initial schema setup migration."""
    
    version = "001"
    description = "Initial schema definitions"
    created_at = datetime(2024, 1, 1)
    
    # Define v1.0.0 schema structures
    V1_SCHEMAS = {
        "SA2Coordinates": {
            "fields": [
                "area_id",  # Will become geographic_id in v2
                "area_name",
                "lat",  # Will move to boundary_data.centroid_lat in v2
                "lon",  # Will move to boundary_data.centroid_lon in v2
                "state",
                "population",
                "sa3_code",
                "sa4_code"
            ],
            "version": SchemaVersion.V1_0_0
        },
        "SEIFAScore": {
            "fields": [
                "area_id",  # Will become geographic_id in v2
                "area_name",
                "index_type",
                "score",
                "rank",  # Will become national_rank in v2
                "decile",
                "total_areas",  # Used to calculate percentile in v2
                "state"
            ],
            "version": SchemaVersion.V1_0_0
        },
        "HealthIndicator": {
            "fields": [
                "area_id",
                "indicator_name",
                "value",
                "unit",
                "year",
                "age_group",
                "sex"
            ],
            "version": SchemaVersion.V1_0_0
        }
    }
    
    @staticmethod
    def get_v1_schema_template(schema_type: str) -> Dict[str, Any]:
        """Get v1.0.0 schema template for a given type."""
        if schema_type not in Migration001.V1_SCHEMAS:
            raise ValueError(f"Unknown schema type: {schema_type}")
            
        schema_def = Migration001.V1_SCHEMAS[schema_type]
        template = {
            "schema_version": schema_def["version"].value,
            "_migration": Migration001.version
        }
        
        # Add placeholder fields
        for field in schema_def["fields"]:
            template[field] = None
            
        return template
    
    @staticmethod
    def validate_v1_schema(data: Dict[str, Any], schema_type: str) -> List[str]:
        """Validate data against v1.0.0 schema."""
        errors = []
        
        if schema_type not in Migration001.V1_SCHEMAS:
            errors.append(f"Unknown schema type: {schema_type}")
            return errors
            
        schema_def = Migration001.V1_SCHEMAS[schema_type]
        required_fields = schema_def["fields"]
        
        # Check for required fields
        for field in required_fields[:5]:  # First 5 fields are required
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")
                
        # Type-specific validation
        if schema_type == "SA2Coordinates":
            if "lat" in data and data["lat"] is not None:
                if not (-90 <= data["lat"] <= 90):
                    errors.append("Latitude out of range")
            if "lon" in data and data["lon"] is not None:
                if not (-180 <= data["lon"] <= 180):
                    errors.append("Longitude out of range")
                    
        elif schema_type == "SEIFAScore":
            if "score" in data and data["score"] is not None:
                if not (400 <= data["score"] <= 1400):
                    errors.append("SEIFA score out of expected range")
            if "decile" in data and data["decile"] is not None:
                if not (1 <= data["decile"] <= 10):
                    errors.append("Decile must be between 1 and 10")
                    
        return errors


def upgrade() -> None:
    """Apply migration - initial schema setup."""
    # This is the initial migration, no upgrade needed
    pass


def downgrade() -> None:
    """Revert migration - not applicable for initial setup."""
    # Cannot downgrade from initial setup
    raise NotImplementedError("Cannot downgrade from initial schema setup")